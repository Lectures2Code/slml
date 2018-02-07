import utils.loc as loc
import utils.get_data as gd
from torch.autograd import Variable
from models.Resnet50_new import Resnet50_new
import torch
import random
import numpy as np
import os
from models.CVAEGAN import CVAEGANWrapper
from utils.AverageMeter import AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_cvae_image_features(dir, cvaegan_wrapper, sample_size = 30, way = 5):



    images = []
    recon = []

    classes = os.listdir(dir)
    classes = random.sample(classes, way)
    for clazz in classes:
        images.append([])
        recon.append([])
        p = os.path.join(dir, clazz)
        clazz_images = os.listdir(p)
        samples = random.sample(clazz_images, sample_size)
        for sample in samples:
            sample = os.path.join(p, sample)
            image = gd.get_image(sample)
            feat = cvaegan_wrapper.nn.encoder.resnet.forward(Variable((image.view(1, *image.shape)).cuda()))
            word_em = cvaegan_wrapper.wordemb.get_word_embedding_by_nindex(clazz)
            word_em = torch.from_numpy(word_em)
            word_em = word_em.view(1, *word_em.shape)
            word_em = word_em.float()
            if loc.params["cuda"]:
                word_em = word_em.cuda()
            word_em = Variable(word_em)
            z_mu, z_var = cvaegan_wrapper.nn.encoder.forward(feat, word_em)
            recon_x = cvaegan_wrapper.nn.generator.forward(z_mu=z_mu, z_var=z_var, c=word_em)
            images[-1].append(feat.cpu().data)
            recon[-1].append(recon_x.cpu().data)
        images[-1] = torch.cat(images[-1])
        recon[-1] = torch.cat(recon[-1])
    images = torch.cat(images)
    recon = torch.cat(recon)

    samples = []
    for clazz in classes:
        z_s = Variable(torch.randn(5, cvaegan_wrapper.nn.latent_dim).cuda())
        word_em = cvaegan_wrapper.wordemb.get_batch_word_embedding_by_nindex([clazz]*5)
        word_em = torch.from_numpy(word_em)
        word_em = word_em.float()
        if loc.params["cuda"]:
            word_em = word_em.cuda()
        word_em = Variable(word_em)
        recon_x_s = cvaegan_wrapper.nn.generator.forward(0, 0, word_em, z_s)
        samples.append(recon_x_s.cpu().data)
    samples = torch.cat(samples)
    return images, recon, samples


def get_features(cvaegan_wrapper, sample_size = 30, way = 5):
    targets = []
    for i in range(way):
        targets += [i] * sample_size
    targets = torch.FloatTensor(targets)
    feats, recon, samples = (get_cvae_image_features(loc.test_dir, cvaegan_wrapper=cvaegan_wrapper, sample_size=sample_size, way=way))
    return feats, recon, samples, targets

def sample_train(n, feats=None, recon=None, samples=None, shot=1, way = 5):
    X = []
    y = []
    for i in range(way):
        index = random.sample(range(int(n / way * i), int(n / way * (i + 1))), shot)
        if feats is not None:
            y += [i] * shot
            X.append(feats[index])
        if recon is not None:
            X.append(recon[index])
            y += [i] * shot
        if samples is not None:
            X.append(samples[[i * 5 + j for j in range(shot)]])
            y += [i] * shot
    y = torch.FloatTensor(y)
    X = torch.cat(X)

    assert X.shape[0] == y.shape[0]
    return X, y


def svr(feats, targets, train_x, train_y, shot=1, way=5):
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(train_x, train_y)
    pred = torch.FloatTensor(clf.predict(feats))
    # print(targets)
    # print(pred)
    assert targets.shape[0] == pred.shape[0]
    acc = torch.mean(torch.eq(pred, targets).type(torch.FloatTensor))
    # print("svr, shot[{}], way[{}], acc[{}]".format(shot, way, acc))
    return acc

def knn(feats, targets, train_x, train_y, k = 1, shot=1, way=5):
    from sklearn.neighbors import KNeighborsClassifier
    assert feats.shape[0] == targets.shape[0]
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_x, train_y)

    pred = torch.FloatTensor(neigh.predict(feats))
    assert targets.shape[0] == pred.shape[0]
    acc = torch.mean(torch.eq(pred, targets).type(torch.FloatTensor))
    # print("knn, k[{}], shot[{}], way[{}], acc[{}]".format(k, shot, way, acc))
    return acc

def logistic(feats, targets, train_x, train_y, shot = 1, way = 5):
    from sklearn import linear_model
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(train_x, train_y)
    pred = torch.FloatTensor(logreg.predict(feats))
    assert targets.shape[0] == pred.shape[0]
    acc = torch.mean(torch.eq(pred, targets).type(torch.FloatTensor))
    # print("logistic, shot[{}], way[{}], acc[{}]".format(shot, way, acc))
    return acc

if __name__ == '__main__':
    way = 20
    loc.params["pretrain"] = True
    loc.params["version"] = 5
    loc.cvae_params["wgan"] = True
    loc.params["p"] = 1

    cvaegan_wrapper = CVAEGANWrapper(con_dim=loc.cvae_params["con_dim"], latent_dim=loc.cvae_params["latent_dim"],
                                     dims=loc.cvae_params["dims"])
    cvaegan_wrapper.nn.cuda()
    cvaegan_wrapper.nn.eval()

    for i in [10]:
        print("p={}".format(i))
        loc.params["p"] = i
        cvaegan_wrapper.load_model()
        feats, recon, samples, targets = get_features(cvaegan_wrapper=cvaegan_wrapper, sample_size=600, way=way)
        print(feats.shape)
        print(recon.shape)
        print(samples.shape)

        acc_svr_frs_5s = AverageMeter()
        acc_knn_frs_5s = AverageMeter()
        acc_log_frs_5s = AverageMeter()

        acc_svr_fr_5s = AverageMeter()
        acc_knn_fr_5s = AverageMeter()
        acc_log_fr_5s = AverageMeter()

        acc_svr_fs_5s = AverageMeter()
        acc_knn_fs_5s = AverageMeter()
        acc_log_fs_5s = AverageMeter()

        acc_svr_f_5s = AverageMeter()
        acc_knn_f_5s = AverageMeter()
        acc_log_f_5s = AverageMeter()

        acc_svr_s_5s = AverageMeter()
        acc_knn_s_5s = AverageMeter()
        acc_log_s_5s = AverageMeter()

        acc_svr_frs_1s = AverageMeter()
        acc_knn_frs_1s = AverageMeter()
        acc_log_frs_1s = AverageMeter()

        acc_svr_fs_1s = AverageMeter()
        acc_knn_fs_1s = AverageMeter()
        acc_log_fs_1s = AverageMeter()

        acc_svr_fr_1s = AverageMeter()
        acc_knn_fr_1s = AverageMeter()
        acc_log_fr_1s = AverageMeter()

        acc_svr_f_1s = AverageMeter()
        acc_knn_f_1s = AverageMeter()
        acc_log_f_1s = AverageMeter()

        acc_svr_s_1s = AverageMeter()
        acc_knn_s_1s = AverageMeter()
        acc_log_s_1s = AverageMeter()

        X_frs_1_size = 0
        X_fs_1_size = 0
        X_fr_1_size = 0
        X_f_1_size = 0
        X_s_1_size = 0

        X_frs_5_size = 0
        X_fs_5_size = 0
        X_fr_5_size = 0
        X_f_5_size = 0
        X_s_5_size = 0


        feat_size = feats.shape[0]

        for i in range(100):
            X, y = sample_train(n=feat_size, feats=feats, recon=recon, samples=samples, shot=5, way=way)
            acc_svr_frs_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_frs_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_frs_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_frs_5s.update(acc_svr_frs_5)
            acc_knn_frs_5s.update(acc_knn_frs_5)
            acc_log_frs_5s.update(acc_log_frs_5)
            X_frs_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, samples=samples, shot=5, way=way)
            acc_svr_fs_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_fs_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_fs_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_fs_5s.update(acc_svr_fs_5)
            acc_knn_fs_5s.update(acc_knn_fs_5)
            acc_log_fs_5s.update(acc_log_fs_5)
            X_fs_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon=recon, shot=5, way=way)
            acc_svr_fr_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_fr_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_fr_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_fr_5s.update(acc_svr_fr_5)
            acc_knn_fr_5s.update(acc_knn_fr_5)
            acc_log_fr_5s.update(acc_log_fr_5)
            X_fr_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, shot=5, way=way)
            acc_svr_f_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_f_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_f_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_f_5s.update(acc_svr_f_5)
            acc_knn_f_5s.update(acc_knn_f_5)
            acc_log_f_5s.update(acc_log_f_5)
            X_f_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, samples=samples, shot=5, way=way)
            acc_svr_s_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_s_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_s_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_s_5s.update(acc_svr_s_5)
            acc_knn_s_5s.update(acc_knn_s_5)
            acc_log_s_5s.update(acc_log_s_5)
            X_s_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon=recon, samples=samples, shot=1, way=way)
            acc_svr_frs_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_frs_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_frs_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_frs_1s.update(acc_svr_frs_1)
            acc_knn_frs_1s.update(acc_knn_frs_1)
            acc_log_frs_1s.update(acc_log_frs_1)
            X_frs_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, samples=samples, shot=1, way=way)
            acc_svr_fs_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_fs_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_fs_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_fs_1s.update(acc_svr_fs_1)
            acc_knn_fs_1s.update(acc_knn_fs_1)
            acc_log_fs_1s.update(acc_log_fs_1)
            X_fs_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon=recon, shot=1, way=way)
            acc_svr_fr_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_fr_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_fr_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_fr_1s.update(acc_svr_fr_1)
            acc_knn_fr_1s.update(acc_knn_fr_1)
            acc_log_fr_1s.update(acc_log_fr_1)
            X_fr_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, shot=1, way=way)
            acc_svr_f_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_f_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_f_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_f_1s.update(acc_svr_f_1)
            acc_knn_f_1s.update(acc_knn_f_1)
            acc_log_f_1s.update(acc_log_f_1)
            X_f_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, samples=samples, shot=1, way=way)
            acc_svr_s_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_s_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_s_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_s_1s.update(acc_svr_s_1)
            acc_knn_s_1s.update(acc_knn_s_1)
            acc_log_s_1s.update(acc_log_s_1)
            X_s_1_size = X.shape[0]
            
        print("feat_size: ", feat_size)
        print()

        print("X_frs_1_size: ", X_frs_1_size)
        print("X_fs_1_size: ", X_fs_1_size)
        print("X_fr_1_size: ", X_fr_1_size)
        print("X_s_1_size: ", X_s_1_size)
        print("X_f_1_size: ", X_f_1_size)
        print()

        print("acc_svr_frs_1: ", acc_svr_frs_1s.avg)
        print("acc_svr_fs_1: ", acc_svr_fs_1s.avg)
        print("acc_svr_fr_1: ", acc_svr_fr_1s.avg)
        print("acc_svr_s_1: ", acc_svr_s_1s.avg)
        print("acc_svr_f_1: ", acc_svr_f_1s.avg)
        print()

        print("acc_knn_frs_1: ", acc_knn_frs_1s.avg)
        print("acc_knn_fs_1: ", acc_knn_fs_1s.avg)
        print("acc_knn_fr_1: ", acc_knn_fr_1s.avg)
        print("acc_knn_s_1: ", acc_knn_s_1s.avg)
        print("acc_knn_f_1: ", acc_knn_f_1s.avg)
        print()

        print("acc_log_frs_1: ", acc_log_frs_1s.avg)
        print("acc_log_fs_1: ", acc_log_fs_1s.avg)
        print("acc_log_fr_1: ", acc_log_fr_1s.avg)
        print("acc_log_s_1: ", acc_log_s_1s.avg)
        print("acc_log_f_1: ", acc_log_f_1s.avg)
        print()

        print("X_frs_5_size: ", X_frs_5_size)
        print("X_fs_5_size: ", X_fs_5_size)
        print("X_fr_5_size: ", X_fr_5_size)
        print("X_s_5_size: ", X_s_5_size)
        print("X_f_5_size: ", X_f_5_size)
        print()

        print("acc_svr_frs_5: ", acc_svr_frs_5s.avg)
        print("acc_svr_fs_5: ", acc_svr_fs_5s.avg)
        print("acc_svr_fr_5: ", acc_svr_fr_5s.avg)
        print("acc_svr_s_5: ", acc_svr_s_5s.avg)
        print("acc_svr_f_5: ", acc_svr_f_5s.avg)
        print()

        print("acc_knn_frs_5: ", acc_knn_frs_5s.avg)
        print("acc_knn_fs_5: ", acc_knn_fs_5s.avg)
        print("acc_knn_fr_5: ", acc_knn_fr_5s.avg)
        print("acc_knn_s_5: ", acc_knn_s_5s.avg)
        print("acc_knn_f_5: ", acc_knn_f_5s.avg)
        print()

        print("acc_log_frs_5: ", acc_log_frs_5s.avg)
        print("acc_log_fs_5: ", acc_log_fs_5s.avg)
        print("acc_log_fr_5: ", acc_log_fr_5s.avg)
        print("acc_log_s_5: ", acc_log_s_5s.avg)
        print("acc_log_f_5: ", acc_log_f_5s.avg)
        print()