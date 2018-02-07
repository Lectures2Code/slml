from utils.get_data import get_image_featrues
import utils.loc as loc
import utils.get_data as gd
from torch.autograd import Variable
from models.Resnet50_new import Resnet50_new
import torch
import random
import numpy as np
import os
from utils.AverageMeter import AverageMeter
from sklearn import linear_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def get_features(sample_size = 30, way = 5):
    targets = []
    for i in range(way):
        targets += [i] * sample_size
    targets = torch.FloatTensor(targets)
    feats = (gd.get_image_featrues(loc.test_dir, sample_size=sample_size, way=way))
    return feats, targets

def sample_train(feats, shot=1, way = 5):
    X = []
    y = []
    n = feats.shape[0]
    for i in range(way):
        y += [i] * shot
        X.append(feats[random.sample(range(int(n / way * i), int(n/way * (i+1))), shot)])

    y = torch.FloatTensor(y)
    X = torch.cat(X)
    assert X.shape[0] == y.shape[0]
    return X.numpy(), y.numpy()

def svr(feats, targets, train_x, train_y, shot=1, way=5):
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(train_x, train_y)
    pred = torch.FloatTensor(clf.predict(feats))
    assert targets.shape[0] == pred.shape[0]
    # print(targets)
    # print(pred)
    # print(targets.shape[0])
    acc = torch.mean(torch.eq(pred, targets).type(torch.FloatTensor))
    print("svr, shot[{}], way[{}], acc[{}]".format(shot, way, acc))
    return acc

def knn(feats, targets, train_x, train_y, k = 1, shot=1, way=5):
    from sklearn.neighbors import KNeighborsClassifier
    assert feats.shape[0] == targets.shape[0]
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_x, train_y)

    pred = torch.FloatTensor(neigh.predict(feats))
    assert targets.shape[0] == pred.shape[0]
    acc = torch.mean(torch.eq(pred, targets).type(torch.FloatTensor))
    print("knn, k[{}], shot[{}], way[{}], acc[{}]".format(k, shot, way, acc))
    return acc

def logistic(feats, targets, train_x, train_y, shot = 1, way = 5):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(train_x, train_y)
    pred = torch.FloatTensor(logreg.predict(feats))
    assert targets.shape[0] == pred.shape[0]
    acc = torch.mean(torch.eq(pred, targets).type(torch.FloatTensor))
    print("logistic, shot[{}], way[{}], acc[{}]".format(shot, way, acc))
    return acc

if __name__ == '__main__':
    feats, targets = get_features(sample_size=600, way=5)

    acc_svr_f_5s = AverageMeter()
    acc_knn_f_5s = AverageMeter()
    acc_log_f_5s = AverageMeter()

    acc_svr_f_1s = AverageMeter()
    acc_knn_f_1s = AverageMeter()
    acc_log_f_1s = AverageMeter()

    X_f_1_size = 0
    X_f_5_size = 0
    X_f_5_size = 0

    feat_size = feats.shape[0]
    for i in range(100):
        # X, y = sample_train(feats=feats, shot=5, way=5)
        # print(X.shape)
        # print(y.shape)
        # acc_svr_f_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=5)
        # acc_knn_f_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=5)
        # acc_log_f_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=5)
        # acc_svr_f_5s.update(acc_svr_f_5)
        # acc_knn_f_5s.update(acc_knn_f_5)
        # acc_log_f_5s.update(acc_log_f_5)
        # X_f_5_size = X.shape[0]

        X, y = sample_train(feats=feats, shot=1, way=5)
        acc_svr_f_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=5)
        acc_knn_f_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=5)
        acc_log_f_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=5)
        acc_svr_f_1s.update(acc_svr_f_1)
        acc_knn_f_1s.update(acc_knn_f_1)
        acc_log_f_1s.update(acc_log_f_1)
        X_f_1_size = X.shape[0]

    print("feat_size: ", feat_size)
    print()

    print("X_f_1_size: ", X_f_1_size)
    print()

    print("acc_svr_f_1: ", acc_svr_f_1s.avg)
    print()

    print("acc_knn_f_1: ", acc_knn_f_1s.avg)
    print()

    print("acc_log_f_1: ", acc_log_f_1s.avg)
    print()

    print("X_f_5_size: ", X_f_5_size)
    print()

    print("acc_svr_f_5: ", acc_svr_f_5s.avg)
    print()

    print("acc_knn_f_5: ", acc_knn_f_5s.avg)
    print()

    print("acc_log_f_5: ", acc_log_f_5s.avg)
    print()
