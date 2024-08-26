#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:43:09 2020

@author: cimon
"""

import time
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
import collections
from ..losses.loss_func import XCAL, crps, likelihood_loss, partial_likelihood_loss_breslow, partial_likelihood_loss_efron, ranking_loss_hit, ranking_loss, self_ranking_loss_weight, calibration_loss
from ..utils.util import get_risk, get_surv_mask1, get_surv_mask2, get_cat_time
from lifelines.utils import concordance_index
#from torch.profiler import profile, record_function, ProfilerActivity
def get_loss(preds, Y_label_T, Y_label_E, key, mydict):
    if key == "likelihood":
        surv_mask1 = get_surv_mask1(Y_label_T, Y_label_E, num_Category = mydict['Kbins'])
        return mydict["gamma"] * likelihood_loss(preds, surv_mask1)
        
    elif key == "crps":
        loss_f = crps(mydict['Kbins'])
        cat_time = get_cat_time(Y_label_T, mydict['Kbins'])
        return mydict["gamma"] * loss_f(preds, cat_time, Y_label_E)
    
    elif key == "xcal":
        loss_f = XCAL(num_dcal_bins=mydict['num_dcal_bins'], train_gamma=mydict['train_gamma'])
        cat_time = get_cat_time(Y_label_T, mydict['Kbins'])
        return mydict["gamma"] * loss_f(preds, cat_time, Y_label_E)
        
    elif key == "partial_likelihood":
        if mydict["mode"] == "breslow":
            return mydict["gamma"] * partial_likelihood_loss_breslow(preds, Y_label_T, Y_label_E)
        else:
            return mydict["gamma"] * partial_likelihood_loss_efron(preds, Y_label_T, Y_label_E)

    elif key == "rank":
        if mydict['mode'] == "ours":
            return mydict["gamma"] * ranking_loss(preds, Y_label_T, Y_label_E, sigma=mydict["sigma"])
        elif mydict['mode'] == "hit":
            surv_mask2 = get_surv_mask2(Y_label_T, num_Category = mydict['Kbins'])
            return mydict["gamma"] * ranking_loss_hit(preds, Y_label_T, Y_label_E, surv_mask2, sigma=mydict["sigma"])
        else:
            raise Exception("No implementation of the rank loss: {}".format(mydict['mode']))

    elif key == "self_rank":
        return mydict["gamma"] * self_ranking_loss_weight(preds, Y_label_T, Y_label_E, sigma=mydict["sigma"], scale=mydict["scale"])

    elif key == "calibration":
        surv_mask1 = get_surv_mask1(Y_label_T, Y_label_E, num_Category = mydict['Kbins'])
        surv_mask2 = get_surv_mask2(Y_label_T, num_Category = mydict['Kbins'])
        return mydict["gamma"] *  calibration_loss(preds, Y_label_E, surv_mask1, surv_mask2, nbins=mydict["nbins"])

    else:
         raise Exception("No implementation of the loss: {}".format(key))

def train(model_name, loss_names, train_loader, model, epoch, optimizer, los_args, logger, grad_accum_step=1, writer_dict=None):
    if writer_dict:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_loss = AverageMeter()
        train_cindex = AverageMeter()

    model.train()
    logger.info("="*100)
    end = time.time()
    surv_time = []
    surv_status = []
    pred_risk = []
    msg_loss = []
    for step, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        images,Y_label_T, Y_label_E = data[0], data[1], data[2]
        Y_label_T = Y_label_T.unsqueeze(1)
        Y_label_E = Y_label_E.unsqueeze(1)
        num_images = images.size(0)
        surv_time.append(Y_label_T)
        surv_status.append(Y_label_E)
        # compute output
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            Y_label_T = Y_label_T.cuda(non_blocking=True)
            Y_label_E = Y_label_E.cuda(non_blocking=True)
        # import pdb;pdb.set_trace()
        preds = model(images)
        if model_name == "deep_cox":
            risks = preds
        else:
            risks = get_risk(preds)
        pred_risk.append(risks)
        loss_list = []
        # msg_loss = []
        for key in loss_names:
            if key == "rank":
                if los_args[key]["mode"] == "hit":
                    los_args[key]["risk_flag"] = False
            if not los_args[key]["risk_flag"]:
                loss_list.append(get_loss(preds, Y_label_T, Y_label_E, key, los_args[key]))
                # msg_loss.append("{}: {:.4f}".format(key, loss_list[-1].detach().cpu().numpy()))
            else:
                loss_list.append(get_loss(risks, Y_label_T, Y_label_E, key, los_args[key]))
                # msg_loss.append("{}: {:.4f}".format(key, loss_list[-1].detach().cpu().numpy()))
        
        loss_list = torch.stack(loss_list)
        msg_loss.append(loss_list.detach().cpu().numpy())
        loss = torch.sum(loss_list)
        if grad_accum_step > 1:
            loss = loss / grad_accum_step
        loss.backward()
        
        # import pdb;pdb.set_trace()
        # clip the grad
        clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

        # compute gradient and do update step
        if (step+1) % grad_accum_step == 0 and (Y_label_E.shape[0]>20):
            # import pdb;pdb.set_trace()
            optimizer.step()
            optimizer.zero_grad()
            if torch.isnan(risks).any():
                import pdb;pdb.set_trace()
            if Y_label_E.sum() > 0:
                c_index = concordance_index(Y_label_T.detach().cpu().numpy(), 1.0-risks.cpu().detach().numpy(), Y_label_E.cpu().detach().numpy())
            else:
                c_index = 0
            ##这里得到的train_correct是一个longtensor型，需要转换为float
            # measure accuracy and record loss
            train_loss.update(loss.item()*grad_accum_step, num_images)
            train_cindex.update(c_index)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # profiler.step()

            if (step+1) % 10 == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.avg:.5f}\t C-index {acc.avg:.3f}\t lr {lr:}'.format(
                          epoch, step, len(train_loader), batch_time = batch_time,
                          speed = images.size(0)/batch_time.val,
                          data_time = data_time, loss = train_loss, acc = train_cindex,
                          lr = optimizer.param_groups[0]['lr'])
                logger.info(msg)
    # import pdb;pdb.set_trace()
    msg_loss = np.stack(msg_loss)
    msg_loss = np.mean(msg_loss, axis=0)
    msg = ["{}: {:.4f}".format(loss_names[i], msg_loss[i]) for i in range(len(loss_names))]
    print("    ".join(msg))
    surv_time = torch.cat(surv_time, dim=0)
    surv_status = torch.cat(surv_status, dim=0)
    pred_risk = torch.cat(pred_risk, dim=0)
    c_index = concordance_index(surv_time.detach().cpu().numpy(), 1.0-pred_risk.detach().cpu().numpy(), surv_status.detach().cpu().numpy())
    logger.info("#########################Training C-index: {:.3f}".format(c_index))
    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train/loss.avg', train_loss.avg, global_steps)
        writer.add_scalar('train/c_index', c_index, global_steps)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_steps)
        writer_dict['train_global_steps'] = global_steps + 1
    return train_loss.avg, c_index
        

def validate(model_name, loss_names, val_loader, model, logger, los_args = None, writer_dict=None):
    if writer_dict:
        batch_time = AverageMeter()
        valid_loss = AverageMeter()
        valid_cindex = AverageMeter()

    # switch to evaluate mode
    model.eval()
    logger.info("-"*100)
    surv_time = []
    surv_status = []
    pred_risk = []
    with torch.no_grad():
        end = time.time()
        for step, data in enumerate(val_loader):
            images,Y_label_T, Y_label_E = data[0], data[1], data[2]
            Y_label_T = Y_label_T.unsqueeze(1)
            Y_label_E = Y_label_E.unsqueeze(1)
            num_images = images.size(0)
            surv_time.append(Y_label_T)
            surv_status.append(Y_label_E)
            # compute output
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                Y_label_T = Y_label_T.cuda(non_blocking=True)
                Y_label_E = Y_label_E.cuda(non_blocking=True)

            preds = model(images)
            if model_name == "deep_cox":
                risks = preds
            else:
                risks = get_risk(preds)
            pred_risk.append(risks)
            loss_list = []
            for key in loss_names:
                if key == "rank":
                    if los_args[key]["mode"] == "hit":
                        los_args[key]["risk_flag"] = False
                if not los_args[key]["risk_flag"]:
                    loss_list.append(get_loss(preds, Y_label_T, Y_label_E, key, los_args[key]))
                else:
                    loss_list.append(get_loss(risks, Y_label_T, Y_label_E, key, los_args[key]))
            loss_list = torch.stack(loss_list)
            loss = torch.sum(loss_list)
            if Y_label_E.sum() > 0:
                c_index = concordance_index(Y_label_T.detach().cpu().numpy(), 1.0-risks.detach().cpu().numpy(), Y_label_E.detach().cpu().numpy())
            else:
                c_index = 0
            # measure accuracy and record loss
            valid_loss.update(loss.item(), num_images)
            valid_cindex.update(c_index)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % 10 == 0:
                msg = 'Validation: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.avg:.5f}\t  C-index {acc.avg:.3f}'.format(
                          step, len(val_loader), batch_time = batch_time,
                          loss = valid_loss, acc=valid_cindex)
                      
                logger.info(msg)
    surv_time = torch.cat(surv_time, dim=0)
    surv_status = torch.cat(surv_status, dim=0)
    pred_risk = torch.cat(pred_risk, dim=0)
    c_index = concordance_index(surv_time.detach().cpu().numpy(), 1.0-pred_risk.detach().cpu().numpy(), surv_status.detach().cpu().numpy())
    logger.info("#########################Validation C-index: {:.3f}".format(c_index))
    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid/loss.avg', valid_loss.avg, global_steps)
        writer.add_scalar('valid/c_index', c_index, global_steps)
        # writer.add_scalar('valid/loss.val', valid_loss.val, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return c_index

def testing(model_name, loss_names, test_loader, model, logger, los_args = None, writer_dict=None):
    if writer_dict:
        batch_time = AverageMeter()
        test_loss = AverageMeter()
        test_cindex = AverageMeter()

    # switch to evaluate mode
    model.eval()
    logger.info("-"*100)
    surv_time = []
    surv_status = []
    pred_risk = []
    with torch.no_grad():
        end = time.time()
        for step, data in enumerate(test_loader):
            images,Y_label_T, Y_label_E = data[0], data[1], data[2]
            Y_label_T = Y_label_T.unsqueeze(1)
            Y_label_E = Y_label_E.unsqueeze(1)
            num_images = images.size(0)
            surv_time.append(Y_label_T)
            surv_status.append(Y_label_E)
            # compute output
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                Y_label_T = Y_label_T.cuda(non_blocking=True)
                Y_label_E = Y_label_E.cuda(non_blocking=True)

            preds = model(images)
            if model_name == "deep_cox":
                risks = preds
            else:
                risks = get_risk(preds)
            pred_risk.append(risks)
            loss_list = []
            for key in loss_names:
                if key == "rank":
                    if los_args[key]["mode"] == "hit":
                        los_args[key]["risk_flag"] = False
                if not los_args[key]["risk_flag"]:
                    loss_list.append(get_loss(preds, Y_label_T, Y_label_E, key, los_args[key]))
                else:
                    loss_list.append(get_loss(risks, Y_label_T, Y_label_E, key, los_args[key]))
            loss_list = torch.stack(loss_list)
            loss = torch.sum(loss_list)

            c_index = concordance_index(Y_label_T.detach().cpu().numpy(), 1.0-risks.detach().cpu().numpy(), Y_label_E.detach().cpu().numpy())
            # measure accuracy and record loss
            test_loss.update(loss.item(), num_images)
            test_cindex.update(c_index)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % 10 == 0:
                msg = 'testing: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.avg:.5f}\t  C-index {acc.avg:.3f}'.format(
                          step, len(test_loader), batch_time = batch_time,
                          loss = test_loss, acc=test_cindex)
                      
                logger.info(msg)
    surv_time = torch.cat(surv_time, dim=0)
    surv_status = torch.cat(surv_status, dim=0)
    pred_risk = torch.cat(pred_risk, dim=0)
    c_index = concordance_index(surv_time.detach().cpu().numpy(), 1.0-pred_risk.detach().cpu().numpy(), surv_status.detach().cpu().numpy())
    logger.info("#########################Test C-index: {:.3f}".format(c_index))
    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['test_global_steps']
        writer.add_scalar('test/loss.avg', test_loss.avg, global_steps)
        writer.add_scalar('test/c_index', c_index, global_steps)
        writer_dict['test_global_steps'] = global_steps + 1

    return c_index

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg =  self.sum/ self.count if self.count != 0 else 0
