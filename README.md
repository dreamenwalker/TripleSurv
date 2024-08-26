# TripleSurv
- **2024-08-23:** Our work is accepted at IEEE Transactions on knowledge and data engineering. 🎉🎉🎉

# TripleSurv: Triplet Time-adaptive Coordinate Learning Approach for Survival Analysis

Author: Liwen Zhang, Lianzhen Zhong, Fan Yang, Linglong Tang, Di Dong, Hui Hui, Jie Tian, Fellow, IEEE



Environment
Our code is based on pytorch tool. The main packages can be found in the file requirements.txt.

- Python 3.11.9
- PyTorch 2.2.1

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

# Clone the github repo and go to the default directory

conda create -n tkde python=3.11.9
conda activate tkde
pip install -r requirements.txt

## Training

The main train file is train.py.


 python train.py 



Acknowledgement

We thanks a lot for publish studies[1-3].Our code is mainly inspired by the x-cal[1], deephit[2].


[1] M. Goldstein, X. Han, A. Puli, A. Perotte, and R. Ranganath, “Xcal: Explicit calibration for survival analysis,” Advances in neural
information processing systems, vol. 33, pp. 18 296–18 307, 2020
[2] Lee C, Zame WR, Yoon J, van der Schaar M. Deephit: A deep learning approach to survival analysis with competing risks.  Thirty-Second AAAI Conference on Artificial Intelligence; 2018; 2018.

[3] A. Avati, T. Duan, S. Zhou, K. Jung, N. H. Shah, and A. Y. Ng, “Countdown regression: Sharp and calibrated survival predictions,” pp. 145–155, 2020.
