import utils.loc as loc
import utils.get_data as gd
from torch.autograd import Variable
from models.Resnet50_new import Resnet50_new
import torch
import random
import numpy as np
import os
from models.AAE_Semantic import AAE_Semantic_Wrapper
from utils.AverageMeter import AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_aae_image_features(dir, aae_semantic_wrapper, sample_size=30, way=5):
    images = []
    recon = []
    affine_d = []
    recon_latent_gaussian = []
    recon_word_gaussian = []

    classes = os.listdir(dir)
    classes = random.sample(classes, way)
    for clazz in classes:
        images.append([])
        recon.append([])
        affine_d.append([])
        recon_latent_gaussian.append([])
        recon_word_gaussian.append([])
        p = os.path.join(dir, clazz)
        clazz_images = os.listdir(p)
        samples = random.sample(clazz_images, sample_size)
        for sample in samples:
            sample = os.path.join(p, sample)
            image = gd.get_image(sample)
            feat = aae_semantic_wrapper.nn.encoder.resnet.forward(Variable((image.view(1, *image.shape)).cuda()))
            affine = gd.get_affine_transformation(sample)
            affine = aae_semantic_wrapper.nn.encoder.resnet.forward(Variable(affine).cuda())
            # print(affine.shape)
            word_em = aae_semantic_wrapper.wordemb.get_word_embedding_by_nindex(clazz)
            word_em = torch.from_numpy(word_em)
            word_em = word_em.view(1, *word_em.shape)
            word_em = word_em.float()
            if loc.params["cuda"]:
                word_em = word_em.cuda()
            word_em = Variable(word_em)
            word_em_gaussian = aae_semantic_wrapper.gaussian_sample(word_em)
            latent_x = aae_semantic_wrapper.nn.encoder.forward(feat)
            latent_gaussian_x = aae_semantic_wrapper.gaussian_sample(latent_x)

            recon_x = aae_semantic_wrapper.nn.decoder.forward(x=latent_x)
            recon_gaussian_x = aae_semantic_wrapper.nn.decoder.forward(x=latent_gaussian_x)
            recon_word_em_x = aae_semantic_wrapper.nn.decoder.forward(x=word_em_gaussian)

            images[-1].append(feat.cpu().data)
            affine_d[-1].append(affine.cpu().data)
            recon[-1].append(recon_x.cpu().data)
            recon_latent_gaussian[-1].append(recon_gaussian_x.cpu().data)
            recon_word_gaussian[-1].append(recon_word_em_x.cpu().data)
        images[-1] = torch.cat(images[-1])
        affine_d[-1] = torch.cat(affine_d[-1])
        recon[-1] = torch.cat(recon[-1])
        recon_latent_gaussian[-1] = torch.cat(recon_latent_gaussian[-1])
        recon_word_gaussian[-1] = torch.cat(recon_word_gaussian[-1])
    images = torch.cat(images)
    affine_d = torch.cat(affine_d)
    recon = torch.cat(recon)
    recon_gaussian = torch.cat(recon_latent_gaussian)
    recon_word_gaussian = torch.cat(recon_word_gaussian)

    return images, affine_d, recon, recon_gaussian, recon_word_gaussian


def get_features(aae_semantic_wrapper, sample_size=30, way=5):
    targets = []
    for i in range(way):
        targets += [i] * sample_size
    targets = torch.FloatTensor(targets)
    feats, affine, recon, recon_gaussian, recon_word_gaussian = (
        get_aae_image_features(loc.test_dir, aae_semantic_wrapper=aae_semantic_wrapper, sample_size=sample_size, way=way))
    return feats, affine, recon, recon_gaussian, recon_word_gaussian, targets


def sample_train(n, feats=None, affine=None, recon=None, recon_gaussian=None, recon_word_gaussian=None,  shot=1, way=5):
    X = []
    y = []
    for i in range(way):
        index = random.sample(range(int(n / way * i), int(n / way * (i + 1))), shot)
        if feats is not None:
            y += [i] * shot
            X.append(feats[index])
        if affine is not None:
            idx = []
            for k in index:
                idx += [k * 5 + j for j in range(5)]
                y += [i] * 5
            X.append(affine[[idx]])
        if recon is not None:
            X.append(recon[index])
            y += [i] * shot
        if recon_gaussian is not None:
            X.append(recon_gaussian[[index]])
            y += [i] * shot
        if recon_word_gaussian is not None:
            X.append(recon_word_gaussian[[index]])
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
    assert targets.shape[0] == pred.shape[0]
    acc = torch.mean(torch.eq(pred, targets).type(torch.FloatTensor))
    # print("svr, shot[{}], way[{}], acc[{}]".format(shot, way, acc))
    return acc


def knn(feats, targets, train_x, train_y, k=1, shot=1, way=5):
    from sklearn.neighbors import KNeighborsClassifier
    assert feats.shape[0] == targets.shape[0]
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_x, train_y)

    pred = torch.FloatTensor(neigh.predict(feats))
    assert targets.shape[0] == pred.shape[0]
    acc = torch.mean(torch.eq(pred, targets).type(torch.FloatTensor))
    # print("knn, k[{}], shot[{}], way[{}], acc[{}]".format(k, shot, way, acc))
    return acc


def logistic(feats, targets, train_x, train_y, shot=1, way=5):
    from sklearn import linear_model
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(train_x, train_y)
    pred = torch.FloatTensor(logreg.predict(feats))
    assert targets.shape[0] == pred.shape[0]
    acc = torch.mean(torch.eq(pred, targets).type(torch.FloatTensor))
    # print("logistic, shot[{}], way[{}], acc[{}]".format(shot, way, acc))
    return acc


if __name__ == '__main__':
    way = 5
    print("way={}".format(way))

    loc.params["pretrain"] = True
    loc.params["version"] = 0
    loc.params["p"] = 97
    loc.aae_semantic_params["decoder"] = True

    aae_semantic_wrapper = AAE_Semantic_Wrapper(latent_dim=loc.aae_semantic_params["latent_dim"],
                                                dims=loc.aae_semantic_params["dims"])
    aae_semantic_wrapper.nn.cuda()
    aae_semantic_wrapper.nn.eval()

    for i in [97]:
        print("p={}".format(i))
        # loc.params["p"] = i
        # aae_semantic_wrapper.load_model()
        feats, affine, recon, recon_gaussian, recon_word_gaussian, targets \
            = get_features(aae_semantic_wrapper=aae_semantic_wrapper, sample_size=600, way=way)

        print(feats.shape)
        print(affine.shape)
        print(recon.shape)
        print(recon_gaussian.shape)
        print(recon_word_gaussian.shape)

        acc_svr_frs_5s = AverageMeter()
        acc_knn_frs_5s = AverageMeter()
        acc_log_frs_5s = AverageMeter()

        acc_svr_frw_5s = AverageMeter()
        acc_knn_frw_5s = AverageMeter()
        acc_log_frw_5s = AverageMeter()

        acc_svr_fs_5s = AverageMeter()
        acc_knn_fs_5s = AverageMeter()
        acc_log_fs_5s = AverageMeter()

        acc_svr_fw_5s = AverageMeter()
        acc_knn_fw_5s = AverageMeter()
        acc_log_fw_5s = AverageMeter()

        acc_svr_fd_5s = AverageMeter()
        acc_knn_fd_5s = AverageMeter()
        acc_log_fd_5s = AverageMeter()

        acc_svr_s_5s = AverageMeter()
        acc_knn_s_5s = AverageMeter()
        acc_log_s_5s = AverageMeter()

        acc_svr_w_5s = AverageMeter()
        acc_knn_w_5s = AverageMeter()
        acc_log_w_5s = AverageMeter()

        acc_svr_r_5s = AverageMeter()
        acc_knn_r_5s = AverageMeter()
        acc_log_r_5s = AverageMeter()
        
        acc_svr_f_5s = AverageMeter()
        acc_knn_f_5s = AverageMeter()
        acc_log_f_5s = AverageMeter()

        acc_svr_frs_1s = AverageMeter()
        acc_knn_frs_1s = AverageMeter()
        acc_log_frs_1s = AverageMeter()

        acc_svr_fs_1s = AverageMeter()
        acc_knn_fs_1s = AverageMeter()
        acc_log_fs_1s = AverageMeter()

        acc_svr_frw_1s = AverageMeter()
        acc_knn_frw_1s = AverageMeter()
        acc_log_frw_1s = AverageMeter()

        acc_svr_fw_1s = AverageMeter()
        acc_knn_fw_1s = AverageMeter()
        acc_log_fw_1s = AverageMeter()

        acc_svr_fd_1s = AverageMeter()
        acc_knn_fd_1s = AverageMeter()
        acc_log_fd_1s = AverageMeter()

        acc_svr_s_1s = AverageMeter()
        acc_knn_s_1s = AverageMeter()
        acc_log_s_1s = AverageMeter()

        acc_svr_w_1s = AverageMeter()
        acc_knn_w_1s = AverageMeter()
        acc_log_w_1s = AverageMeter()

        acc_svr_r_1s = AverageMeter()
        acc_knn_r_1s = AverageMeter()
        acc_log_r_1s = AverageMeter()

        acc_svr_f_1s = AverageMeter()
        acc_knn_f_1s = AverageMeter()
        acc_log_f_1s = AverageMeter()

        X_frs_1_size = 0
        X_fs_1_size = 0
        X_frw_1_size = 0
        X_fd_5_size = 0
        X_fw_1_size = 0
        X_s_1_size = 0
        X_w_1_size = 0
        X_r_1_size = 0
        X_f_1_size = 0
        X_frs_5_size = 0
        X_fs_5_size = 0
        X_frw_5_size = 0
        X_fd_5_size = 0
        X_fw_5_size = 0
        X_s_5_size = 0
        X_w_5_size = 0
        X_r_5_size = 0
        X_f_5_size = 0

        feat_size = feats.shape[0]

        for i in range(100):
            X, y = sample_train(n=feat_size, feats=feats, recon=recon, recon_gaussian=recon_gaussian, shot=5, way=way)
            acc_svr_frs_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_frs_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_frs_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_frs_5s.update(acc_svr_frs_5)
            acc_knn_frs_5s.update(acc_knn_frs_5)
            acc_log_frs_5s.update(acc_log_frs_5)
            X_frs_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon_gaussian=recon_gaussian, shot=5, way=way)
            acc_svr_fs_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_fs_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_fs_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_fs_5s.update(acc_svr_fs_5)
            acc_knn_fs_5s.update(acc_knn_fs_5)
            acc_log_fs_5s.update(acc_log_fs_5)
            X_fs_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon=recon, recon_word_gaussian=recon_word_gaussian, shot=5, way=way)
            acc_svr_frw_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_frw_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_frw_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_frw_5s.update(acc_svr_frw_5)
            acc_knn_frw_5s.update(acc_knn_frw_5)
            acc_log_frw_5s.update(acc_log_frw_5)
            X_frw_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon_word_gaussian=recon_word_gaussian, shot=5, way=way)
            acc_svr_fw_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_fw_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_fw_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_fw_5s.update(acc_svr_fw_5)
            acc_knn_fw_5s.update(acc_knn_fw_5)
            acc_log_fw_5s.update(acc_log_fw_5)
            X_fw_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, affine=affine, shot=5, way=way)
            acc_svr_fd_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_fd_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_fd_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_fd_5s.update(acc_svr_fd_5)
            acc_knn_fd_5s.update(acc_knn_fd_5)
            acc_log_fd_5s.update(acc_log_fd_5)
            X_fd_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, recon_gaussian=recon_gaussian, shot=5, way=way)
            acc_svr_s_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_s_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_s_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_s_5s.update(acc_svr_s_5)
            acc_knn_s_5s.update(acc_knn_s_5)
            acc_log_s_5s.update(acc_log_s_5)
            X_s_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, recon_word_gaussian=recon_word_gaussian, shot=5, way=way)
            acc_svr_w_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_w_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_w_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_w_5s.update(acc_svr_w_5)
            acc_knn_w_5s.update(acc_knn_w_5)
            acc_log_w_5s.update(acc_log_w_5)
            X_w_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, recon=recon, shot=5, way=way)
            acc_svr_r_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_r_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_r_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_r_5s.update(acc_svr_r_5)
            acc_knn_r_5s.update(acc_knn_r_5)
            acc_log_r_5s.update(acc_log_r_5)
            X_r_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, shot=5, way=way)
            acc_svr_f_5 = svr(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_knn_f_5 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=5, way=way)
            acc_log_f_5 = logistic(feats, targets, train_x=X, train_y=y, shot=5, way=way)
            acc_svr_f_5s.update(acc_svr_f_5)
            acc_knn_f_5s.update(acc_knn_f_5)
            acc_log_f_5s.update(acc_log_f_5)
            X_f_5_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon=recon, recon_gaussian=recon_gaussian, shot=1, way=way)
            acc_svr_frs_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_frs_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_frs_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_frs_1s.update(acc_svr_frs_1)
            acc_knn_frs_1s.update(acc_knn_frs_1)
            acc_log_frs_1s.update(acc_log_frs_1)
            X_frs_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon_gaussian=recon_gaussian, shot=1, way=way)
            acc_svr_fs_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_fs_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_fs_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_fs_1s.update(acc_svr_fs_1)
            acc_knn_fs_1s.update(acc_knn_fs_1)
            acc_log_fs_1s.update(acc_log_fs_1)
            X_fs_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon=recon, recon_word_gaussian=recon_word_gaussian, shot=1, way=way)
            acc_svr_frw_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_frw_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_frw_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_frw_1s.update(acc_svr_frw_1)
            acc_knn_frw_1s.update(acc_knn_frw_1)
            acc_log_frw_1s.update(acc_log_frw_1)
            X_frw_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, recon_word_gaussian=recon_word_gaussian, shot=1, way=way)
            acc_svr_fw_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_fw_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_fw_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_fw_1s.update(acc_svr_fw_1)
            acc_knn_fw_1s.update(acc_knn_fw_1)
            acc_log_fw_1s.update(acc_log_fw_1)
            X_fw_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, affine=affine, shot=1, way=way)
            acc_svr_fd_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_fd_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_fd_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_fd_1s.update(acc_svr_fd_1)
            acc_knn_fd_1s.update(acc_knn_fd_1)
            acc_log_fd_1s.update(acc_log_fd_1)
            X_fd_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, recon_gaussian=recon_gaussian, shot=1, way=way)
            acc_svr_s_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_s_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_s_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_s_1s.update(acc_svr_s_1)
            acc_knn_s_1s.update(acc_knn_s_1)
            acc_log_s_1s.update(acc_log_s_1)
            X_s_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, recon_word_gaussian=recon_word_gaussian, shot=1, way=way)
            acc_svr_w_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_w_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_w_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_w_1s.update(acc_svr_w_1)
            acc_knn_w_1s.update(acc_knn_w_1)
            acc_log_w_1s.update(acc_log_w_1)
            X_w_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, recon=recon, shot=1, way=way)
            acc_svr_r_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_r_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_r_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_r_1s.update(acc_svr_r_1)
            acc_knn_r_1s.update(acc_knn_r_1)
            acc_log_r_1s.update(acc_log_r_1)
            X_r_1_size = X.shape[0]

            X, y = sample_train(n=feat_size, feats=feats, shot=1, way=way)
            acc_svr_f_1 = svr(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_knn_f_1 = knn(feats, targets, train_x=X, train_y=y, k=3, shot=1, way=way)
            acc_log_f_1 = logistic(feats, targets, train_x=X, train_y=y, shot=1, way=way)
            acc_svr_f_1s.update(acc_svr_f_1)
            acc_knn_f_1s.update(acc_knn_f_1)
            acc_log_f_1s.update(acc_log_f_1)
            X_f_1_size = X.shape[0]

        print("feat_size: ", feat_size)
        print()

        print("X_frs_1_size: ", X_frs_1_size)
        print("X_fs_1_size: ", X_fs_1_size)
        print("X_frw_1_size: ", X_frw_1_size)
        print("X_fw_1_size: ", X_fw_1_size)
        print("X_fd_1_size: ", X_fd_1_size)
        print("X_s_1_size: ", X_s_1_size)
        print("X_w_1_size: ", X_w_1_size)
        print("X_r_1_size: ", X_r_1_size)
        print("X_f_1_size: ", X_f_1_size)
        print()

        print("acc_svr_frs_1: ", acc_svr_frs_1s.avg)
        print("acc_svr_fs_1: ", acc_svr_fs_1s.avg)
        print("acc_svr_frw_1: ", acc_svr_frw_1s.avg)
        print("acc_svr_fw_1: ", acc_svr_fw_1s.avg)
        print("acc_svr_fd_1: ", acc_svr_fd_1s.avg)
        print("acc_svr_w_1: ", acc_svr_w_1s.avg)
        print("acc_svr_s_1: ", acc_svr_s_1s.avg)
        print("acc_svr_r_1: ", acc_svr_r_1s.avg)
        print("acc_svr_f_1: ", acc_svr_f_1s.avg)
        print()

        print("acc_knn_frs_1: ", acc_knn_frs_1s.avg)
        print("acc_knn_fs_1: ", acc_knn_fs_1s.avg)
        print("acc_knn_frw_1: ", acc_knn_frw_1s.avg)
        print("acc_knn_fw_1: ", acc_knn_fw_1s.avg)
        print("acc_knn_fd_1: ", acc_knn_fd_1s.avg)
        print("acc_knn_w_1: ", acc_knn_w_1s.avg)
        print("acc_knn_s_1: ", acc_knn_s_1s.avg)
        print("acc_knn_r_1: ", acc_knn_r_1s.avg)
        print("acc_knn_f_1: ", acc_knn_f_1s.avg)
        print()

        print("acc_log_frs_1: ", acc_log_frs_1s.avg)
        print("acc_log_fs_1: ", acc_log_fs_1s.avg)
        print("acc_log_frw_1: ", acc_log_frw_1s.avg)
        print("acc_log_fw_1: ", acc_log_fw_1s.avg)
        print("acc_log_fd_1: ", acc_log_fd_1s.avg)
        print("acc_log_w_1: ", acc_log_w_1s.avg)
        print("acc_log_s_1: ", acc_log_s_1s.avg)
        print("acc_log_r_1: ", acc_log_r_1s.avg)
        print("acc_log_f_1: ", acc_log_f_1s.avg)
        print()

        print("X_frs_5_size: ", X_frs_5_size)
        print("X_fs_5_size: ", X_fs_5_size)
        print("X_frw_5_size: ", X_frw_5_size)
        print("X_fw_5_size: ", X_fw_5_size)
        print("X_fd_5_size: ", X_fd_5_size)
        print("X_s_5_size: ", X_s_5_size)
        print("X_w_5_size: ", X_w_5_size)
        print("X_r_5_size: ", X_r_5_size)
        print("X_f_5_size: ", X_f_5_size)
        print()

        print("acc_svr_frs_5: ", acc_svr_frs_5s.avg)
        print("acc_svr_fs_5: ", acc_svr_fs_5s.avg)
        print("acc_svr_frw_5: ", acc_svr_frw_5s.avg)
        print("acc_svr_fw_5: ", acc_svr_fw_5s.avg)
        print("acc_svr_fd_5: ", acc_svr_fd_5s.avg)
        print("acc_svr_w_5: ", acc_svr_w_5s.avg)
        print("acc_svr_s_5: ", acc_svr_s_5s.avg)
        print("acc_svr_r_5: ", acc_svr_r_5s.avg)
        print("acc_svr_f_5: ", acc_svr_f_5s.avg)
        print()

        print("acc_knn_frs_5: ", acc_knn_frs_5s.avg)
        print("acc_knn_fs_5: ", acc_knn_fs_5s.avg)
        print("acc_knn_frw_5: ", acc_knn_frw_5s.avg)
        print("acc_knn_fw_5: ", acc_knn_fw_5s.avg)
        print("acc_knn_fd_5: ", acc_knn_fd_5s.avg)
        print("acc_knn_w_5: ", acc_knn_w_5s.avg)
        print("acc_knn_s_5: ", acc_knn_s_5s.avg)
        print("acc_knn_r_5: ", acc_knn_r_5s.avg)
        print("acc_knn_f_5: ", acc_knn_f_5s.avg)
        print()

        print("acc_log_frs_5: ", acc_log_frs_5s.avg)
        print("acc_log_fs_5: ", acc_log_fs_5s.avg)
        print("acc_log_frw_5: ", acc_log_frw_5s.avg)
        print("acc_log_fw_5: ", acc_log_fw_5s.avg)
        print("acc_log_fd_5: ", acc_log_fd_5s.avg)
        print("acc_log_w_5: ", acc_log_w_5s.avg)
        print("acc_log_s_5: ", acc_log_s_5s.avg)
        print("acc_log_r_5: ", acc_log_r_5s.avg)
        print("acc_log_f_5: ", acc_log_f_5s.avg)
        print()
