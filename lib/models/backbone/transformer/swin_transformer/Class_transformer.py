import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from ..utils.helpers import to_2tuple, trunc_normal_, _assert
from ..layers.drop_path import DropPath
import numpy as np
from torchvision.ops import RoIPool
from torchvision.ops import nms
# from torchsummary import summary

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.dim = embed_dim
        self.c_in = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    def flops(self):
        # calculate flops for 1 window with token length of N
        flops = 0
        # project
        flops += self.img_size[0]*self.img_size[1]*self.c_in*self.dim
        # norm
        flops += self.img_size[0]*self.img_size[1]/self.patch_size[0]/self.patch_size[1]*self.dim
        return flops

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        # print("patch mergeing", input_resolution)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.convert = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        assert (L-1)%(H*W) == 0, "input feature has wrong size"
        k = (L-1)//(H*W)

        cls_token = x[:,0:1,:] #B,1,C
        x = x[:,1:,:]
        x = x.view(B, k, H, W, C)

        x0 = x[:, :, 0::2, 0::2, :]  # B, k, H/2, W/2, C
        x1 = x[:, :, 1::2, 0::2, :]  # B, k, H/2, W/2, C
        x2 = x[:, :, 0::2, 1::2, :]  # B, k, H/2, W/2, C
        x3 = x[:, :, 1::2, 1::2, :]  # B, k, H/2, W/2, C
        x = torch.cat([x0, x1, x2, x3], -1)  # B, k, H/2, W/2, 4*C
        x = x.view(B, k, -1, 4 * C)  # B, k, H/2*W/2, 4*C

        x = self.norm(x)
        x = self.reduction(x).reshape(B,-1,2*C) # B, k, H/2*W/2, 4*C  -> B, k, H/2*W/2, 2*C  -> B, kHW/4, 2C
        cls_token = self.convert(cls_token) #B,1,C -> B,1,2C
        x = torch.cat([cls_token,x], 1)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        #class token convert
        flops += 1*self.dim*2*self.dim
        return flops

##########################
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # print("partition", B, H, W, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    #B = int(windows.shape[0] / (H * W / window_size / window_size))
    B = windows.shape[0] // (H * W // window_size // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)#nB, M, M, C -> B, H/M, W/M, M, M, C
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)# B, H/M, W/M, M, M, C -> B, H/M, M, W/M, M, C -> B, H, W, C
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, rel_pos=False, window_size=7, qk_scale=None, attn_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.window_size = window_size  # Wh, Ww
        self.rel_pos = rel_pos
        self.scale = qk_scale or head_dim ** -0.5
        
        if self.rel_pos:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size)
            coords_w = torch.arange(self.window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size - 1
            relative_coords[:, :, 0] *= 2 * self.window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, mask=None):
        """
        Args:
            qkv: input features with shape of (num_windows*B, N+1, C)
        """
        B_, M_, C = qkv.shape #B_=B*H*W/(M*M), M_=MM+1, C=3*dim
        qkv = qkv.reshape(B_, M_, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)#3*B_*num_heads*(MM+1)*C'
        q, k, v = qkv[0,:,:,1:,:], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale  #B_*num_heads*MM*C'
        attn = (q @ k.transpose(-2, -1))  #B_*num_heads*MM*(MM+1)
        
        if self.rel_pos:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # MM,num_heads
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, MM, MM

            patch_attn = attn[:,:,:,1:] + relative_position_bias.unsqueeze(0)
            cls_attn = attn[:,:,:,0:1]
            attn = torch.cat((cls_attn, patch_attn),dim=3)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, M_-1, M_) + mask.unsqueeze(1).unsqueeze(0)#nW,MM,MM+1 -> nW,1,MM,MM+1 -> 1,nW,1,MM,MM+1
            attn = attn.view(-1, self.num_heads, M_-1, M_)#B_*num_heads*MM*C'
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, M_-1, self.dim)##B_*num_heads*MM*C' -> B_*MM*num_heads*C' -> B_*MM*C
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * (N+1)
        #  x = (attn @ v)
        flops += self.num_heads * N * (N+1) * (self.dim // self.num_heads)
        return flops

class ClsAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qk_scale=None, attn_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        """
        Args:
            qkv: input features with shape of (num_windows*B, N+1, C)
        """
        B, N_, _ = qkv.shape #B*(N+1)*3C
        qkv = qkv.reshape(B, N_, 3, self.num_heads, self.dim  // self.num_heads).permute(2, 0, 3, 1, 4) #3*B*num_heads*(N+1)*C'
        q0, k, v = qkv[0,:,:,0:1,:], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)


        q0 = q0 * self.scale  #B*num_heads*1*C'
        attn = self.softmax(q0 @ k.transpose(-2, -1)) #B*num_heads*1*(N+1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, self.dim) #B*num_heads*1*C' -> B*1*num_heads*C' ->B*1*C
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # attn = (q0 @ k.transpose(-2, -1))
        flops += self.num_heads * 1 * (self.dim // self.num_heads) * (N+1)
        #  x = (attn @ v)
        flops += self.num_heads * 1 * (N+1) * (self.dim // self.num_heads)

        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,rel_pos=False,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size

            img_mask = torch.zeros((1, Hp, Wp, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, M, M, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # nW, MM
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) #nW,MM,MM
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            nW, M_, _ = attn_mask.shape
            cls_mask = torch.zeros((nW, M_, 1),dtype=torch.float)
            attn_mask = torch.cat((cls_mask,attn_mask),dim=2) #nW,MM,MM+1
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn = WindowAttention(
            dim, num_heads=num_heads,window_size=window_size,rel_pos=rel_pos,
            qk_scale=qk_scale, attn_drop=attn_drop)
        self.cls_atten = ClsAttention(
            dim, num_heads=num_heads,
            qk_scale=qk_scale, attn_drop=attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # print("begin", H, W, B, L, C)
        # assert L == H * W+1, "input feature has wrong size"

        shortcut = x
        #1: obtain qkv
        x = self.norm1(x)
        qkv = self.qkv(x)  #B*L*3C,  L = N+1

        #2: window partition on qkv and W/S-SMA
        #obtain updated class token
        out_cls_token = self.cls_atten(qkv) #B*1*C

        qkv_patch = qkv[:,1:,:] #B*N*3C
        qkv_cls = qkv[:,0:1,:] #B*1*3C
        qkv_patch = qkv_patch.view(B, H, W, 3*C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r>0 or pad_b>0:
            qkv_patch = F.pad(qkv_patch, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = qkv_patch.shape  #B*H'*W'*3C
        # print("begin1", Hp, Wp)

        # cyclic shift
        if self.shift_size > 0:
            qkv_patch = torch.roll(qkv_patch, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        qkv_patch_windows = window_partition(qkv_patch, self.window_size)  # nW*B, M, M, 3C
        qkv_patch_windows = qkv_patch_windows.view(-1, self.window_size * self.window_size, 3*C)  # nW*B, MM, 3C
        B_ = qkv_patch_windows.shape[0]  # B_ = nW*B
        nW = B_ // B
        qkv_cls = qkv_cls.expand(B, nW, 3*C).reshape(B_,1,3*C)
        qkv_windows = torch.cat((qkv_cls, qkv_patch_windows), dim=1) # nW*B, MM+1, 3C

        # W-MSA/SW-MSA
        #B_*MM*C
        attn_windows = self.attn(qkv_windows, mask=self.attn_mask)

        # merge windows  B_*M*M*C -> B*Hp*Wp*C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) #B_*M*M*C 
        attn_windows = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B Hp Wp C

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_windows = torch.roll(attn_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # B*Hp*Wp*C  -> B*H*W*C -> B*N*C
        if pad_r > 0 or pad_b > 0:
            attn_windows = attn_windows[:, :H, :W, :]
        attn_windows = attn_windows.reshape(B,-1,C)

        # project multiple heads to the output
        z = torch.cat((out_cls_token,attn_windows), dim=1).reshape(B, L, C) #B*(N+1)*C
        z = self.proj(z)  #B*(N+1)*C
        z = self.proj_drop(z)

        # Feed Forward Network
        z = shortcut + self.drop_path(z)
        z = z + self.drop_path(self.mlp(self.norm2(z)))

        return z, qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * (H * W+1)
        #qkv
        flops += 3*(H * W+1)*self.dim *self.dim 
        # MSA
        nW = H * W / self.window_size / self.window_size
        N = self.window_size * self.window_size
        flops += nW * self.attn.flops(N)+self.cls_atten.flops(N)
        #proj
        flops += (H * W+1)*self.dim *self.dim 
        # mlp
        flops += 2 * (H * W+1) * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * (H * W+1)
        return flops

class SwinBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,rel_pos=False,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.input_resolution = input_resolution

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            input_resolution = (input_resolution[0]//2, input_resolution[1]//2)
            dim = 2*dim
            # print("11", input_resolution)
        else:
            self.downsample = None
    
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=(input_resolution[0], input_resolution[1]),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,rel_pos=rel_pos,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, qkv = blk(x)

        return x, qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        if self.downsample is not None:
            flops += self.downsample.flops()

        for blk in self.blocks:
            flops += blk.flops()
        
        return flops

###########################
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class ViTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=None, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#3*B*num_heads*N*C'
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale#B*num_heads*N*N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#B*num_heads*N*C' -> B*N*num_heads*C' -> B*N*C
        x = self.proj(x) #B*N*C
        x = self.proj_drop(x)
        return x
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        #qkv
        flops += 3*N*self.dim *self.dim 
        # attn = (q0 @ k.transpose(-2, -1))
        flops += self.num_heads * (N+1) * (self.dim // self.num_heads) * (N+1)
        #  x = (attn @ v)
        flops += self.num_heads * (N+1) * (N+1) * (self.dim // self.num_heads)
        # project
        flops += N*self.dim *self.dim 

        return flops

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, qk_scale=None,mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ViTAttention(dim, num_heads=num_heads, qk_scale=qk_scale, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp_ratio = mlp_ratio
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
    
    def flops(self):
        H, W = self.input_resolution
        N = H*W
        # calculate flops for 1 window with token length of N
        flops = 0
        #norm1 
        flops += self.dim * N
        # MSA
        flops += self.attn.flops(N)
        # norm2
        flops += self.dim * N
        # mlp
        flops += 2 * N * self.dim * self.dim * self.mlp_ratio
       
        return flops

class ViTBasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # print("###", input_resolution)

        # patch merging layer
        if downsample is not None:
            # print("00", input_resolution)
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            input_resolution = (input_resolution[0]//2, input_resolution[1]//2)
            dim = 2*dim
            # print("11", input_resolution)
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            ViTBlock(dim=dim, 
                    num_heads=num_heads,
                    input_resolution = input_resolution,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        #x: B,MM,C
        if self.downsample is not None:
            x = self.downsample(x)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

##############################
class LocalizationNet(nn.Module):
    '''
    auxiliary ConvNet for predicting the part of the body
    '''
    def __init__(self,dim,input_resolution,cls_num):
        super().__init__()

        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(dim, 2*dim, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2*dim, 4*dim, 5, stride=4, padding=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(4*dim,cls_num)
        self.flatten = nn.Flatten()

    def forward(self,x):
        H, W = self.input_resolution
        B, L, C = x.shape

        x = x[:,1:,:]
        x = x.transpose(1,2) #B,C,L
        x = x.view(B,C,H,W)
        x = F.relu(self.conv1(x)) #B,H/2,W/2,2C
        x = F.relu(self.conv2(x)) #B,H/8,W/8,4C
        x = self.flatten(self.global_pool(x)) #B,4C
        x = self.fc(x)

        return F.softmax(x, dim=1)

############################
def GetAweight(qkv, dim,num_heads):
    '''
    '''
    B, N_, C_ = qkv.shape #B, N_=N+1, C_=3*dim
    qkv = qkv.reshape(B, N_, 3, num_heads, dim // num_heads).permute(2, 0, 3, 1, 4)#3,B,head_num,N_,C
    q,k = qkv[0], qkv[1] #B,head_num,N_,C

    query = q[:,:,0:1,:]
    key = k
    weight1 = (query @ key.transpose(-2, -1)).squeeze() ##B,head_num,1,N_ -> B,head_num,N_ 
    query = q
    key = k[:,:,0:1,:]
    weight2 = (query @ key.transpose(-2, -1)).squeeze() ##B,head_num,N_,1 -> B,head_num,N_ 
    Aweight = F.softmax(weight1,dim=2)*F.softmax(weight2,dim=2) #B,head_num,N_
    #obtain the attention matrix (A) between cls token and patch token
    Aweight = Aweight[:,:,1:]#B,head_num,N
    #compute the average of A along with head channel
    Aweight = torch.mean(Aweight, dim=1, keepdim=False) #B,N

    return Aweight

def GetWeightScale(Hp,Wp):
    loc = torch.tensor([Hp/2,Wp/2])
    mtn = torch.distributions.multivariate_normal.MultivariateNormal(loc, 0.1*torch.eye(2))

    yv, xv = torch.meshgrid([torch.arange(Hp),torch.arange(Wp)])
    grid = torch.stack((xv, yv), 2).view((1, Hp, Wp, 2)).float()
    
    scale = mtn.log_prob(grid)
    scale = (scale-scale.min())/(scale.max()-scale.min())
    
    return scale.exp().cuda()

def GetRoIIndices(weight,roi_size,Is_scale=False, k=100, roi_num=4):
    '''
    select a roi with the max weight
    return the coordinate of the selected roi: x1,y1,x2,y2
    '''
    B,H,W = weight.shape #B,H,W
    weight = F.avg_pool2d(weight, kernel_size=roi_size, stride=1)#weight: B,H-window_size+1,W-window_size+1
    _,Hp,Wp = weight.shape
    if Is_scale:
        # print("use the position prior")
        scale = GetWeightScale(Hp,Wp)
        scale = torch.clamp(scale,min=1.0,max=2.0)
        # print(scale)
        # print('before scale:', weight)
        weight = weight*scale
        # print('after scale:', weight)
        # import pdb;pdb.set_trace()
    weight = weight.reshape(B,-1) #B,Hp*Wp
    scores, idx = torch.sort(weight, dim=1, descending=True)#descending为alse，升序，为True，降序
    idx = idx[:,:k]
    scores = scores[:,:k]#B,k
    
    x1= idx // Wp #B,k
    y1 = idx % Wp #B,k
    x2 = x1+roi_size
    y2 = y1+roi_size
    bbox = torch.stack((x1, y1, x2, y2),dim=2)#B,k,4
    output = scores.new(B,roi_num,5).zero_()
    for i in range(B):
        proposals_single = bbox[i] #k,4
        scores_single = scores[i] #k
        keep_idx_i = nms(proposals_single.float(), scores_single, iou_threshold=0.3)
        keep_idx_i = keep_idx_i.long().view(-1)
        keep_idx_i = keep_idx_i[:roi_num]
        
        proposals_single = proposals_single[keep_idx_i]
        # scores_single = scores_single[keep_idx_i]
        num_proposal = proposals_single.size(0)
        output[i,:,0] = i
        output[i,:num_proposal,1:] = proposals_single
    # batch_indices = torch.arange(B).unsqueeze(1).cuda() #B,1
    # # print(x1.shape,x2.shape,y1.shape,y2.shape,batch_indices.shape)
    # roi = torch.cat((batch_indices, x1, y1, x2, y2),dim=1) #B,5
    # import pdb;pdb.set_trace()

    return output#roi

class RoICreator():
    '''
    return the coordinate of the selected window: x1,y1,x2,y2
    '''
    def __init__(self,dim,input_resolution,num_heads,roi_size,pre_nms_num=1000,roi_num=10):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.roi_size = roi_size
        self.input_resolution = input_resolution
        self.pre_nms_num = pre_nms_num
        self.roi_num = roi_num
    
    def __call__(self,qkv):
        H, W = self.input_resolution

        Aweight = GetAweight(qkv,dim=self.dim,num_heads=self.num_heads)#B,N
        Aweight = Aweight.view(-1,H,W)#B,H,W
        indices_and_rois = GetRoIIndices(Aweight,roi_size=self.roi_size,k=self.pre_nms_num,roi_num=self.roi_num)#B,5,  b,x1,y1,x2,y2

        return indices_and_rois.float()

class RoiSelect(nn.Module):
    def __init__(self,dim,input_resolution,num_heads,roi_size,pre_nms_num=100,roi_num=4):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        #compute window_size and ensure that the window_size is a multiple of 4
        self.roi_size = roi_size
        self.input_resolution = input_resolution
        self.roi_num = roi_num
        self.roi_indices =RoICreator(dim=dim,input_resolution=input_resolution,num_heads=num_heads,
                                     roi_size=roi_size,pre_nms_num=pre_nms_num,roi_num=roi_num)
        self.GetRoi = RoIPool((self.roi_size, self.roi_size), 1)#1 denote that the operator is conducted in the feature map scale
    
    def forward(self,x,qkv):
        H, W = self.input_resolution
        B, L, C = x.shape#L=N+1

        cls_token = x[:,0:1,:] #B,1,C
        img = x[:,1:,:].transpose(1,2).reshape(B,C,H,W) #B,N,C -> B,C,N -> B,C,H,W
    
        indices_of_rois = self.roi_indices(qkv)#B,k,5,  b,x1,y1,x2,y2
        indices_of_rois = indices_of_rois.reshape(-1,5) #B_,5
        roi = self.GetRoi(img, indices_of_rois) #B_,C,M,M
        # import pdb;pdb.set_trace()
        roi = roi.reshape(B,self.roi_num,C,-1)#B,k,C,MM
        roi = roi.transpose(3,2).reshape(B,-1,C)#B,k,MM,C -> B,kMM,C

        return torch.cat((cls_token, roi),dim=1), indices_of_rois  #B,kMM+1,C

############################
class Cls_Transformer(nn.Module):
    r""" Window-selected Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 4, 4, 2], num_heads=[3, 6, 12, 24],aux_flag=False,
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,aux_ncls=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,class_token=True,
                 norm_layer=nn.LayerNorm, ape=True,rel_pos=False, patch_norm=True,roi_ratio=0.25,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        # print("********", in_chans)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.aux_flag = aux_flag
        self.num_tokens = 1 if class_token else 0
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution
        # print("***",patches_resolution)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.num_tokens > 0 else None
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches+self.num_tokens, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        #shared layer
        self.sharedlayers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            factor = 1 if i_layer<2 else (2 ** (i_layer-1))
            layer = SwinBasicLayer(dim=int(embed_dim * factor),
                               input_resolution=(patches_resolution[0] // factor, patches_resolution[1] // factor),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,rel_pos=rel_pos,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer > 0 else None,#先patch merging后block
                               use_checkpoint=use_checkpoint)
            self.sharedlayers.append(layer)

        if self.aux_flag:
            self.aux_convNet = LocalizationNet(dim=embed_dim * 2,
                                    input_resolution=(patches_resolution[0] // 2,patches_resolution[1] // 2),
                                    cls_num=aux_ncls)


        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], 1, self.embed_dim), x), dim=1)#B，N+1,C
        # add absolute position embedding
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # shared layer deal with the whole-size image based on SwinTransformer block
        for layer in self.sharedlayers:
            x, qkv = layer(x)
    
        return x, qkv
        

    def forward_head(self, x, pre_logits: bool = False):
        '''
        recongnize the img using the class token
        '''
        x =  x[:, 0:1, :]
        x = torch.flatten(x, 1)
        x = self.norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        if self.aux_flag:
            last_layer,cls_pos_y = self.forward_features(x)
            out = self.forward_head(last_layer)
            # return out,cls_pos_y,last_layer
            return out
        else:
            last_layer, indices_of_rois = self.forward_features(x)
            out = self.forward_head(last_layer)
            # return out,last_layer
            return out, indices_of_rois

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for _, layer in enumerate(self.sharedlayers):
            flops += layer.flops()
        for _, layer in enumerate(self.patchlayers):
            flops += layer.flops()
        ##lack the flops of auxirary ConvNet
        #norm
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        #head
        flops += self.num_features * self.num_classes
        return flops

# def main():
    
#     # seed_everything(0)
#     # initial_seeds(0)
    
#     # set parameter
#     num_classes = 7
#     window_ratio = 0.25
#     img_size = 896
#     class_token = True

#     img = torch.rand((2,3,img_size,img_size))
#     model = WS_Transformer(num_classes=num_classes, window_ratio=window_ratio,
#                             img_size=img_size, class_token=class_token)
#     summary(model,(3,img_size,img_size))
#     print(model.flops())
#     z, auxi_z, feat_map = model(img)
#     import pdb; pdb.set_trace()

# main()