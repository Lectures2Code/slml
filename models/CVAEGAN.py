import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import utils.loc as loc
from models.Resnet50_new import Resnet50_new
from utils.WordEmb import WordEmb
import numpy as np
import random
import torch.optim as optim
from utils.loss import euclidean_loss
from utils.AverageMeter import AverageMeter
from utils.save_model import save_model
import os
import time
import utils.get_data as gd
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class Encoder(nn.Module):
    def __init__(self, dims, latent_dim, con_dim):
        super(Encoder, self).__init__()
        self.dims = dims
        self.latent_dim = latent_dim
        self.con_dim = con_dim
        self.name = "Encoder"

        self.resnet = Resnet50_new()
        if not loc.params["fine_tune_resnet"]:
            for param in self.resnet.parameters():
                param.requires_grad = False

        encoder_layers = []
        # [2048, 1024, 512, 256]
        encoder_layers.append(nn.Linear(dims[0] + con_dim, dims[1]))
        encoder_layers.append(nn.BatchNorm1d(dims[1]))
        encoder_layers.append(nn.ReLU())

        dims = dims[1:]
        for i, dim in enumerate(dims[:-1]):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
            encoder_layers.append(nn.ReLU())

        self.encoder_layers = encoder_layers
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu_fc = nn.Linear(self.dims[-1], self.latent_dim)
        self.var_fc = nn.Linear(self.dims[-1], self.latent_dim)

    def forward(self, X, C):
        # X: (64, 4096)
        # c: (64, 100)
        inputs = torch.cat((X, C), 1)
        h = self.encoder(inputs)
        z_mu = self.mu_fc(h)
        z_var = self.var_fc(h)
        return z_mu, z_var

class Generator(nn.Module):
    def __init__(self, dims, latent_dim, con_dim):
        super(Generator, self).__init__()
        self.dims = dims
        self.latent_dim = latent_dim
        self.con_dim = con_dim
        self.name = "Generator"

        decoder_layers = []
        dims.reverse()
        decoder_layers.append(nn.Linear(self.latent_dim + self.con_dim, dims[0]))
        decoder_layers.append(nn.BatchNorm1d(dims[0]))
        decoder_layers.append(nn.ReLU())
        for i, dim in enumerate(dims[:-1]):
            decoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            decoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers = decoder_layers[:-2]
        self.decoder = nn.Sequential(*decoder_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # reparemerization to allow backprogagation
    def sample_z(self, mu, log_var):
        eps = Variable(torch.randn(mu.data.shape[0], self.latent_dim).cuda())
        z = mu + torch.exp(log_var / 2) * eps
        return z

    def forward(self, z_mu, z_var, c, z=None, num=1):
        if num > 1:
            a = []
            for i in range(num):
                a.append(self.sample_z(z_mu, z_var))
            z = torch.cat(a, 0)
            c = c.repeat(num, 1)
        else:
            if z is None:
                z = self.sample_z(z_mu, z_var)
        inputs = torch.cat((z, c), 1)
        X = self.decoder(inputs)
        return X

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc31 = nn.Linear(2048, 80)
        self.fc32 = nn.Linear(2048, 20)
        self.fc33 = nn.Linear(2048, 5)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)

        self.name = "classifier"

    def forward(self, x, output_size):
        x = self.dropout(x)
        y1 = self.fc1(x)
        x = self.relu(y1)
        x = self.dropout(x)
        y2 = self.fc2(x)
        x = self.relu(y2)
        if output_size == 20:
            x = self.fc32(x)
        elif output_size == 5:
            x = self.fc33(x)
        else:
            x = self.fc31(x)
        return y1, y2, x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.name = "discriminator"
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        y1 = self.fc1(x)
        x = self.relu(y1)
        x = self.dropout(x)
        y2 = self.fc2(x)
        x = self.relu(y2)
        x = self.fc3(x)
        if not loc.cvae_params["wgan"]:
            x = (self.sigmoid(x) + 1) / 2
        return y1, y2, x

class CVAEGAN(nn.Module):
    def __init__(self, dims, con_dim, latent_dim):
        super(CVAEGAN, self).__init__()
        self.name = "cvaegan"
        self.dims = dims
        self.con_dim = con_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(dims=dims, con_dim=con_dim, latent_dim=latent_dim)
        self.generator = Generator(dims=dims, con_dim=con_dim, latent_dim=latent_dim)
        self.classifier = Classifier()
        self.discriminator = Discriminator()

class CVAEGANWrapper():
    def __init__(self, dims, con_dim, latent_dim):
        self.nn = CVAEGAN(dims, con_dim, latent_dim)

        if loc.cvae_params["wgan"]:
            self.dir = loc.cvaewgan_dir
        else:
            self.dir = loc.cvaegan_dir

        if loc.params["cuda"]:
            self.nn.cuda()
        if not loc.params["pretrain"]:
            length = len(os.listdir(self.dir))
            self.version = length
            new_path = os.path.join(self.dir, "version_{}".format(self.version))
            if not os.path.exists(new_path):
                os.mkdir(new_path)
                self.f = open(os.path.join(new_path, "log.txt"), "a")
        else:
            assert loc.params["p"] is not None
            assert loc.params["version"] is not None
            self.version = loc.params["version"]
            self.p = loc.params["p"]
            new_path = os.path.join(self.dir, "version_{}".format(self.version))
            self.f = open(os.path.join(new_path, "log.txt"), "a")
            self.load_model()

        self.wordemb = WordEmb()




    def train(self, train_dataloader, val_dataloader, test_dataloader):
        self.nn.train()
        class_to_idx = train_dataloader.dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        d_optimizer = optim.RMSprop(self.nn.discriminator.parameters(), alpha=0.00005)
        g_optimizer = optim.Adam(self.nn.generator.parameters())
        e_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.nn.encoder.parameters()))

        d_losses = AverageMeter()
        g_losses = AverageMeter()
        e_losses = AverageMeter()

        one = torch.FloatTensor([1])
        mone = one * -1
        one = one.cuda()
        mone = mone.cuda()
        for i in range(loc.params["epoch"]):
            for step, (x, label) in enumerate(train_dataloader):
                for parm in self.nn.discriminator.parameters():
                    parm.data.clamp_(-loc.cvae_params["norm"], loc.cvae_params["norm"])

                label_value = label
                label_nindex = [idx_to_class[int(i)] for i in label_value]

                label = label.float()
                if loc.params["cuda"]:
                    x, label = x.cuda(), label.cuda()
                x = Variable(x)
                feat = self.nn.encoder.resnet.forward(x)

                # c_loss = self.c_loss(feat, label)
                word_em = self.wordemb.get_batch_word_embedding_by_nindex(label_nindex)
                word_em = torch.from_numpy(word_em)
                word_em = word_em.float()
                if loc.params["cuda"]:
                    word_em = word_em.cuda()
                word_em = Variable(word_em)
                z_mu, z_var = self.nn.encoder.forward(feat, word_em)
                kl_loss = self.kl_loss(z_mu, z_var)

                recon_x = self.nn.generator(z_mu, z_var, word_em).detach()

                ### Sample from z and c
                z_s = torch.randn(x.shape[0], self.nn.latent_dim)
                l_s = random.choices(range(len(idx_to_class)), k=x.shape[0])
                c_s = []
                for l in l_s:
                    c_s.append(idx_to_class[l])
                word_em_s = self.wordemb.get_batch_word_embedding_by_nindex(c_s)
                word_em_s = torch.from_numpy(word_em_s)
                word_em_s = word_em_s.float()
                if loc.params["cuda"]:
                    word_em_s = word_em_s.cuda()
                    z_s = z_s.cuda()
                word_em_s = Variable(word_em_s)
                z_s = Variable(z_s)
                recon_x_s = self.nn.generator.forward(0, 0, word_em_s, z_s).detach()

                x_d_y1, x_d_y2, x_d = self.nn.discriminator.forward(feat)
                recon_x_d_y1, recon_x_d_y2, recon_x_d = self.nn.discriminator.forward(recon_x)
                recon_x_s_y1, recon_x_s_y2, recon_x_s = self.nn.discriminator.forward(recon_x_s)


                #### Update discriminator
                self.nn.discriminator.zero_grad()
                if not loc.cvae_params["wgan"]:
                    d_loss, gd_loss = self.d_loss(x_d, recon_x_d, recon_x_s, x_d_y2, recon_x_s_y2)
                    d_loss.backward(retain_graph=True)
                    d_optimizer.step()
                else:
                    x_d = x_d.mean()
                    recon_x_d = recon_x_d.mean()
                    recon_x_s = recon_x_s.mean()
                    x_d.backward(mone, retain_graph=True)
                    recon_x_d.backward(one, retain_graph=True)
                    recon_x_s.backward(one, retain_graph=True)
                    d_loss = - x_d + recon_x_d + recon_x_s
                    d_optimizer.step()
                    print("x_d", x_d.data[0])
                    print("recon_x_d", recon_x_d.data[0])
                    print("recon_x_s", recon_x_s.data[0])
                print("d_loss", d_loss.data[0])
                self.f.write("Epoch {}, Iter {}\n".format(i, step))
                self.f.write("d_loss {}\n".format(d_loss.data[0]))
                # print(x_d.data[0], recon_x_d.data[0], recon_x_s.data[0])

                #### Update generator
                # z_s = torch.randn(x.shape[0], self.nn.latent_dim)
                # l_s = random.choices(range(len(idx_to_class)), k=x.shape[0])
                # c_s = []
                # for l in l_s:
                #     c_s.append(idx_to_class[l])
                # word_em_s = self.wordemb.get_batch_word_embedding_by_nindex(c_s)
                # word_em_s = torch.from_numpy(word_em_s)
                # word_em_s = word_em_s.float()
                # if loc.params["cuda"]:
                #     word_em_s = word_em_s.cuda()
                #     z_s = z_s.cuda()
                # word_em_s = Variable(word_em_s)
                # z_s = Variable(z_s)
                # recon_x_s = self.nn.generator.forward(0, 0, word_em_s, z_s).detach()
                # recon_x_s_y1, recon_x_s_y2, recon_x_s = self.nn.discriminator.forward(recon_x_s)
                # recon_x_s = recon_x_s.mean()

                self.nn.generator.zero_grad()
                if not loc.cvae_params["wgan"]:
                    g_loss = torch.mean((euclidean_loss(feat, recon_x))) + torch.mean((euclidean_loss(x_d_y2, recon_x_d_y2)))
                    g_combined_loss = g_loss + 0.001 * gd_loss
                    g_combined_loss.backward(retain_graph=True)
                    g_optimizer.step()
                else:
                    recon_x_s.backward(mone, retain_graph=True)
                    g_combined_loss = -recon_x_s
                    g_optimizer.step()
                print("g_loss", g_combined_loss.data[0])
                self.f.write("g_loss {}\n".format(g_combined_loss.data[0]))

                #### Update encoder
                self.nn.encoder.zero_grad()
                if not loc.cvae_params["wgan"]:
                    e_combine_loss = kl_loss + g_loss
                else:
                    e_combine_loss = kl_loss + torch.mean((euclidean_loss(feat, recon_x))) + g_combined_loss
                e_combine_loss.backward(retain_graph=True)
                e_optimizer.step()
                print("kl_loss", kl_loss.data[0])
                print("e_loss", e_combine_loss.data[0])
                self.f.write("kl_loss {}\n".format(kl_loss.data[0]))
                self.f.write("e_loss {}\n".format(e_combine_loss.data[0]))

                print(step)
                d_losses.update(d_loss.data[0])
                g_losses.update(g_combined_loss.data[0])
                e_losses.update(e_combine_loss.data[0])



            self.save_model()
            if loc.cvae_params["wgan"]:
                gan_type = "WGAN"
            else:
                gan_type = "GAN"
            print("CVAE{} train, Epoch[{}], D-Loss[{}]({}), G-Loss[{}]({}), E-Loss[{}]({})".format(
                gan_type, i, d_losses.val, d_losses.avg, g_losses.val, g_losses.avg, e_losses.val, e_losses.avg))
            self.f.write("CVAE{} train, Epoch[{}], D-Loss[{}]({}), G-Loss[{}]({}), E-Loss[{}]({})\n".format(
                gan_type, i, d_losses.val, d_losses.avg, g_losses.val, g_losses.avg, e_losses.val, e_losses.avg))
            self.evaluate(val_dataloader, test_dataloader)


    def get_dis_pred(self, x, label, idx_to_class):
        label_value = label
        label_nindex = [idx_to_class[int(i)] for i in label_value]

        label = label.float()
        if loc.params["cuda"]:
            x, label = x.cuda(), label.cuda()
        x = Variable(x)
        feat = self.nn.encoder.resnet.forward(x)

        word_em = self.wordemb.get_batch_word_embedding_by_nindex(label_nindex)
        word_em = torch.from_numpy(word_em)
        word_em = word_em.float()
        if loc.params["cuda"]:
            word_em = word_em.cuda()
        word_em = Variable(word_em)
        z_mu, z_var = self.nn.encoder.forward(feat, word_em)

        recon_x = self.nn.generator.forward(z_mu, z_var, word_em)
        _, _, x_d = self.nn.discriminator.forward(feat)
        _, _, recon_x_d = self.nn.discriminator.forward(recon_x)

        z_s = torch.randn(x.shape[0], self.nn.latent_dim)
        l_s = random.choices(range(len(idx_to_class)), k=x.shape[0])
        c_s = []
        for l in l_s:
            c_s.append(idx_to_class[l])
        word_em_s = self.wordemb.get_batch_word_embedding_by_nindex(c_s)
        word_em_s = torch.from_numpy(word_em_s)
        word_em_s = word_em_s.float()
        if loc.params["cuda"]:
            word_em_s = word_em_s.cuda()
            z_s = z_s.cuda()
        word_em_s = Variable(word_em_s)
        z_s = Variable(z_s)
        recon_x_s = self.nn.generator.forward(0, 0, word_em_s, z_s)
        _, _, recon_x_s_d = self.nn.discriminator.forward(recon_x_s)

        return x_d, recon_x_d, recon_x_s_d

    def evaluate(self, val_dataloader, test_dataloader):
        self.nn.eval()
        x_val_d = AverageMeter()
        recon_x_val_d = AverageMeter()
        recon_x_s_val_d = AverageMeter()

        x_test_d = AverageMeter()
        recon_x_test_d = AverageMeter()
        recon_x_s_test_d = AverageMeter()

        class_to_idx = val_dataloader.dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        for step, (x, label) in enumerate(val_dataloader):
            x_d, recon_x_d, recon_x_s_d = self.get_dis_pred(x, label, idx_to_class)
            x_val_d.update(torch.eq((x_d.data > 0.5).float().cpu(), torch.ones(*x_d.shape)).float().mean(0)[0], x.shape[0])
            recon_x_val_d.update(torch.eq((recon_x_d.data > 0.5).float().cpu(),torch.ones(*x_d.shape)).float().mean(0)[0], x.shape[0])
            recon_x_s_val_d.update(torch.eq((recon_x_s_d.data > 0.5).float().cpu(), torch.ones(*x_d.shape)).float().mean(0)[0], x.shape[0])

        class_to_idx = test_dataloader.dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        for step, (x, label) in enumerate(test_dataloader):
            x_d, recon_x_d, recon_x_s_d = self.get_dis_pred(x, label, idx_to_class)
            x_test_d.update(torch.eq((x_d.data > 0.5).float().cpu(), torch.ones(*x_d.shape)).float().mean(0)[0], x.shape[0])
            recon_x_test_d.update(torch.eq((recon_x_d.data > 0.5).float().cpu(),torch.ones(*x_d.shape)).float().mean(0)[0], x.shape[0])
            recon_x_s_test_d.update(torch.eq((recon_x_s_d.data > 0.5).float().cpu(), torch.ones(*x_d.shape)).float().mean(0)[0], x.shape[0])
        print("Siamese x_val_d, acc[{}]({})".format(x_val_d.val, x_val_d.avg))
        print("Siamese recon_x_val_d, acc[{}]({})".format(recon_x_val_d.val, recon_x_val_d.avg))
        print("Siamese recon_x_s_val_d, acc[{}]({})".format(recon_x_s_val_d.val, recon_x_s_val_d.avg))
        print("Siamese x_test_d, acc[{}]({})".format(x_test_d.val, x_test_d.avg))
        print("Siamese recon_x_test_d, acc[{}]({})".format(recon_x_test_d.val, recon_x_test_d.avg))
        print("Siamese recon_x_s_test_d, acc[{}]({})".format(recon_x_s_test_d.val, recon_x_s_test_d.avg))

    def generate_feats(self, train_data_loader, val_data_loader, test_data_loader):
        if loc.params["mode"] == "infer":
            if loc.cvae_params["wgan"]:
                self.dir = loc.cvaewgan_sample_dir
            else:
                self.dir = loc.cvaegan_sample_dir
        self.nn.eval()
        handle = open(os.path.join(self.dir, 'sample_version_{}_p_{}.pkl'.format(self.version, self.p)), 'wb')
        num_class = 5
        num_shot = 5
        num_sample = 5

        result = {}
        train_loader_block = self.generate(loader=train_data_loader,
                                           dir=loc.train_dir,
                                           status="train",
                                           num_class=num_class,
                                           num_shot_per_class=num_shot,
                                           num_sample_per_shot=num_sample)
        val_loader_block = self.generate(loader=val_data_loader,
                                         dir=loc.val_dir,
                                         status="val",
                                         num_class=num_class,
                                         num_shot_per_class=num_shot,
                                         num_sample_per_shot=num_sample)
        test_loader_block = self.generate(loader=test_data_loader,
                                         dir=loc.test_dir,
                                         status="val",
                                         num_class=num_class,
                                         num_shot_per_class=num_shot,
                                         num_sample_per_shot=num_sample)
        result["train"] = train_loader_block
        result["val"] = val_loader_block
        result["test"] = test_loader_block
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def generate(self, loader, dir, status, num_class, num_shot_per_class, num_sample_per_shot):
        self.nn.eval()
        class_to_idx = loader.dataset.class_to_idx
        classes = random.sample(list(class_to_idx.keys()), num_class)
        loader_block = {}
        for clazz in classes:
            pics = random.sample(os.listdir(os.path.join(dir, clazz)), num_shot_per_class)
            pics = [os.path.join(dir, clazz, pic) for pic in pics]
            class_index = class_to_idx[clazz]
            class_block = {}
            class_block["index"] = class_index
            class_block["nindex"] = clazz
            for i, pic in enumerate(pics):
                block = {}
                image = gd.get_image(pic)
                if loc.params["cuda"]:
                    image = image.cuda()
                image = Variable(image.view(1, *image.shape))
                feat = self.nn.encoder.resnet.forward(image)

                word_em = self.wordemb.get_word_embedding_by_nindex(clazz)
                word_em = torch.from_numpy(word_em)
                word_em = word_em.view(1, *word_em.shape)
                word_em = word_em.float()
                if loc.params["cuda"]:
                    word_em = word_em.cuda()
                word_em = Variable(word_em)
                z_mu, z_var = self.nn.encoder.forward(feat, word_em)

                recon_x = self.nn.generator.forward(z_mu=z_mu, z_var=z_var, c=word_em, num=5)

                ## Sample
                z_s = Variable(torch.randn(num_sample_per_shot, self.nn.latent_dim).cuda())

                word_em = word_em.repeat(num_sample_per_shot, 1)
                recon_x_s = self.nn.generator.forward(0, 0, word_em, z_s)

                feat_y1, feat_y2, feat_d = self.nn.discriminator.forward(feat)
                recon_x_y1, recon_x_y2, recon_x_d = self.nn.discriminator.forward(recon_x)
                recon_x_s_y1, recon_x_s_y2, recon_x_s_d = self.nn.discriminator.forward(recon_x_s)
                print(feat_d.data[0], recon_x_d.data[0], recon_x_s_d.data[0])
                block["feat"] = feat.data.cpu().numpy()
                block["recon"] = recon_x.data.cpu().numpy()
                block["sample"] = recon_x_s.data.cpu().numpy()
                block["pic_id"] = i
                class_block["pic{}".format(i)] = block
            loader_block[clazz] = class_block
        loader_block["loader"] = status
        return loader_block


    def load_model(self):
        path = os.path.join(self.dir, "version_{}".format(self.version))
        if not os.path.exists(path):
            print(path)
            raise ("No model in path {}".format(path))

        checkpoint = torch.load(os.path.join(path, "{}th.model".format(self.p)))
        for key in checkpoint:
            if key != "state_dict":
                print(key, checkpoint[key])
        self.nn.load_state_dict(checkpoint["state_dict"])

    def save_model(self):
        new_path = os.path.join(self.dir, "version_{}".format(self.version))
        length2 = len(os.listdir(new_path))
        dicta = {
            'state_dict': self.nn.state_dict(),
        }
        dicta.update(loc.params)
        torch.save(dicta, os.path.join(new_path, "{}th.model".format(length2)))
        print("{}th.model saved".format(length2))

    def c_loss(self, feat, label):
        _, _, pred = self.nn.classifier.forward(feat, 80)
        criterion = nn.CrossEntropyLoss().cuda()
        return criterion(pred, label)

    def wgan_d_loss(self, x_d, recon_x_d, recon_x_s):
        d_loss = torch.mean(- x_d + recon_x_d + recon_x_s)
        return d_loss

    def d_loss(self, x_d, recon_x_d, recon_x_s, x_d_y2, recon_x_s_y2):
        d_loss = - torch.sum((torch.log(x_d) + torch.log(1 - recon_x_d) + torch.log(1 - recon_x_s))) / x_d.shape[0]
        x_d_y2_mean = x_d_y2.mean(0)
        recon_x_s_y2_mean = recon_x_s_y2.mean(0)
        x_d_y2_mean = x_d_y2_mean.view(1, *x_d_y2_mean.shape)
        recon_x_s_y2_mean = recon_x_s_y2_mean.view(1, *recon_x_s_y2_mean.shape)
        gd_loss = torch.sum(euclidean_loss(x_d_y2_mean, recon_x_s_y2_mean)) / x_d.shape[0]
        return d_loss, gd_loss

    def kl_loss(self, z_mu, z_var):
        return torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))

def run():
    # Train WGAN
    loc.params["epoch"] = 100
    loc.params["pretrain"] = True
    loc.params["mode"] = "train"
    loc.params["batch_size"] = "32"
    loc.params["version"] = 4
    loc.params["p"] = 33
    loc.cvae_params["wgan"] = True
    loc.params["fine_tune_resnet"] = True

    cvaegan_wrapper = CVAEGANWrapper(con_dim=loc.cvae_params["con_dim"], latent_dim=loc.cvae_params["latent_dim"],
                                     dims = loc.cvae_params["dims"])
    print(cvaegan_wrapper.nn)
    end = time.time()
    if loc.params["mode"] == "train":
        print("Size of Train Batch: {}, Spent Time: {}".format(len(gd.train_loader), time.time()-end))
        cvaegan_wrapper.train(train_dataloader=gd.train_loader, val_dataloader = gd.val_loader, test_dataloader=gd.test_loader)
    elif loc.params["mode"] == "test":
        return cvaegan_wrapper.evaluate(gd.val_loader, gd.test_loader)
    else:
        return cvaegan_wrapper.generate_feats(gd.train_loader, gd.val_loader, gd.test_loader)

if __name__ == '__main__':
    run()
    # Train WGAN
    # loc.params["epoch"] = 100
    # loc.params["pretrain"] = False
    # loc.params["mode"] = "train"
    # loc.params["batch_size"] = "32"
    # loc.params["version"] = 0
    # loc.params["p"] = 1
    # loc.cvae_params["wgan"] = True

    # Infer  WGAN
    # loc.params["epoch"] = 100
    # loc.params["pretrain"] = True
    # loc.params["mode"] = "infer"
    # loc.params["batch_size"] = "32"
    # loc.params["version"] = 0
    # loc.params["p"] = 1
    # loc.cvae_params["wgan"] = False




