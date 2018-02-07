import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from torchvision import transforms
from torch.autograd import Variable
from utils.metric import accuracy,update_class_acc
from utils.AverageMeter import AverageMeter
import os
import utils.loc as loc
from models.Resnet50_new import Resnet50_new
import utils.get_data as gd
import numpy as np
import random
import utils.loss as loss
from utils.AverageMeter import AverageMeter
from PIL import Image
from models.AAE_Semantic import AAE_Semantic_Wrapper
### Resnet 求正确率
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def test_resnet():
    model = Resnet50_new()

    top1 = AverageMeter()
    top5 = AverageMeter()
    accu_20 = []
    for i in range(80):
        accu_20.append(AverageMeter())

    for step, sample in enumerate(gd.val_loader):
        weight = sample[0].shape[0]
        s = Variable(sample[0].cuda())
        pre = model(s)
        prec1, prec5 = accuracy(pre.data, sample[1].cuda(), topk=(1, 5))
        top1.update(prec1[0], n=weight)
        top5.update(prec5[0], n=weight)
        update_class_acc(accu_20, pre.data, sample[1].cuda())


        print("Step: {step}, top1: {top1.avg:.3f}({top1.val:.3f}), "
              "top5: {top5.avg:.3f}({top5.val:.3f})".format(step=step, top1=top1, top5=top5))

    for k, j in enumerate(accu_20):
        print("{k}: {top1.avg:.3f}({top1.val:.3f}), ".format(k=k, top1=j))

def get_group_euclidean(group1, group2 = None):
    eu = AverageMeter()
    if group2 is None:
        for i, image1 in enumerate(group1):
            for j, image2 in enumerate(group1[i+1:]):
                eu.update(loss.euclidean_loss(image1, image2)[0])
    else:
        for i, image1 in enumerate(group1):
            for j, image2 in enumerate(group2):
                eu.update(loss.euclidean_loss(image1, image2)[0])
    return eu


def get_margin():
    images = gd.get_image_featrues()
    same = []
    diff = []
    for i, group1 in enumerate(images):
        same.append(get_group_euclidean(group1))
        print("{}.same".format(i), same[-1].avg, same[-1].count)
        for j, group2 in enumerate(images[i+1:]):
            diff.append(get_group_euclidean(group1, group2))
            print("{}-{}.diff".format(i, j), diff[-1].avg, diff[-1].count)


if __name__ == '__main__':
    # p = "/home/xmz/mini-imagenet-v2/pretrain_data/train/n01632777/n01632777_19393.JPEG"
    # a = gd.get_image(p)
    # a = a.view(1, *a.shape)
    # a = Variable(a.cuda())
    # model = Resnet50_new()
    # model.cuda()
    # output = model(a)
    # print(torch.sum(torch.pow(output, 2)))
    # print(output)
    # a.append(gd.get_image(p))
    # a.append(gd.get_image(p))
    # print(torch.stack(a))
    # for i, sample in enumerate(gd.val_loader):
    #     print(sample[0])
    loc.params["pretrain"] = True
    loc.params["version"] = 0
    loc.params["p"] = 20
    loc.aae_semantic_params["decoder"] = True

    aae_semantic_wrapper = AAE_Semantic_Wrapper(latent_dim=loc.aae_semantic_params["latent_dim"],
                                                dims=loc.aae_semantic_params["dims"])
    loc.params["pretrain"] = True
    loc.params["version"] = 0
    loc.params["p"] = 10
    loc.aae_semantic_params["decoder"] = True
    aae_semantic_wrapper.load_model()



