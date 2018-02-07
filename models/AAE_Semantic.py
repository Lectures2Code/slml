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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class Encoder(nn.Module):
    def __init__(self, dims, latent_dim, ):
        super(Encoder, self).__init__()
        self.dims = dims
        self.latent_dim = latent_dim
        self.name = "Encoder"

        self.resnet = Resnet50_new()
        if not loc.params["fine_tune_resnet"]:
            for param in self.resnet.parameters():
                param.requires_grad = False

        encoder_layers = []
        # [2048, 1024, 512, 256]
        for i, dim in enumerate(dims[:-1]):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(self.dims[-1], self.latent_dim))
        self.encoder_layers = encoder_layers
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, X):
        # X: (64, 4096)
        # c: (64, 100)
        return self.encoder(X)

class Decoder(nn.Module):
    def __init__(self, dims, latent_dim):
        super(Decoder, self).__init__()
        self.dims = dims
        self.latent_dim = latent_dim
        self.name = "Generator"

        decoder_layers = []
        dims.reverse()
        decoder_layers.append(nn.Linear(self.latent_dim, dims[0]))
        decoder_layers.append(nn.BatchNorm1d(dims[0]))
        decoder_layers.append(nn.ReLU())
        for i, dim in enumerate(dims[:-1]):
            decoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            decoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers = decoder_layers[:-2]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(x)

class AAE_Semantic(nn.Module):
    def __init__(self, dims, latent_dim):
        super(AAE_Semantic, self).__init__()
        self.name = "aae_semantic"
        self.dims = dims
        self.latent_dim = latent_dim

        self.encoder = Encoder(dims=dims, latent_dim=latent_dim)
        self.decoder = Decoder(dims=dims, latent_dim=latent_dim)

class AAE_Semantic_Wrapper():
    def __init__(self, dims, latent_dim):
        self.nn = AAE_Semantic(dims, latent_dim)

        if loc.aae_semantic_params["decoder"]:
            self.dir = loc.aae_semantic_dir
        else:
            self.dir = loc.semantic_dir

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

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.nn.parameters()))
        losses = AverageMeter()
        semantic_losses = AverageMeter()
        recon_losses = AverageMeter()

        mse_loss = nn.MSELoss(size_average=False)
        for i in range(loc.params["epoch"]):
            for step, (x, label) in enumerate(train_dataloader):
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
                word_em = Variable(word_em.cuda(), requires_grad=False)
                latent = self.nn.encoder.forward(feat)
                # print(word_em.shape)
                # print(latent.shape)
                semantic_loss = mse_loss(latent, word_em) / x.shape[0]
                if loc.aae_semantic_params["decoder"]:
                    recon_x = self.nn.decoder.forward(latent)
                    recon_loss = mse_loss(recon_x, feat) / x.shape[0]
                    loss = semantic_loss + recon_loss
                    recon_losses.update(recon_loss.data[0], n=x.shape[0])
                else:
                    loss = semantic_loss
                semantic_losses.update(semantic_loss.data[0], n=x.shape[0])
                losses.update(loss.data[0], n=x.shape[0])

                self.nn.zero_grad()
                loss.backward()
                optimizer.step()

            self.save_model()
            if loc.aae_semantic_params["decoder"]:
                model_type = "AAE_Semantic"
                print("{} train, Epoch[{}], Loss[{}]({}), Recon-Loss[{}]({}), Semantic-Loss[{}]({})".format(
                    model_type, i, losses.val, losses.avg, recon_losses.val, recon_losses.avg, semantic_losses.val, semantic_losses.avg))
                self.f.write("{} train, Epoch[{}], Loss[{}]({}), Recon-Loss[{}]({}), Semantic-Loss[{}]({})".format(
                    model_type, i, losses.val, losses.avg, recon_losses.val, recon_losses.avg, semantic_losses.val, semantic_losses.avg))
            else:
                model_type = "Semantic"
                print("{} train, Epoch[{}], Loss[{}]({}), Semantic-Loss[{}]({})".format(
                    model_type, i, losses.val, losses.avg, semantic_losses.val, semantic_losses.avg))
                self.f.write("{} train, Epoch[{}], Loss[{}]({}), Semantic-Loss[{}]({})".format(
                    model_type, i, losses.val, losses.avg, semantic_losses.val, semantic_losses.avg))

            # self.evaluate(train_dataloader, val_dataloader, test_dataloader)

    def gaussian_sample(self, latent_embedding):
        noise = torch.zeros(latent_embedding.size())
        noise = noise.normal_(0, std=0.5).cuda()
        latent_embedding = Variable(torch.add(latent_embedding.data, noise))
        return latent_embedding

    def evaluate_aae_semantic(self, train_dataloader, val_dataloader, test_dataloader):
        self.nn.eval()

        train_class_to_idx = train_dataloader.dataset.class_to_idx
        val_class_to_idx = val_dataloader.dataset.class_to_idx
        test_class_to_idx = test_dataloader.dataset.class_to_idx

        for i, (x, label) in enumerate(train_dataloader):
            label_value = label
            label_nindex = [train_class_to_idx[int(i)] for i in label_value]

            label = label.float()
            if loc.params["cuda"]:
                x, label = x.cuda(), label.cuda()
            x = Variable(x)
            feat = self.nn.encoder.resnet.forward(x)


    def save_model(self):
        new_path = os.path.join(self.dir, "version_{}".format(self.version))
        length2 = len(os.listdir(new_path))
        dicta = {
            'state_dict': self.nn.state_dict(),
        }
        dicta.update(loc.params)
        torch.save(dicta, os.path.join(new_path, "{}th.model".format(length2)))
        print("{}th.model saved".format(length2))

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

def run():
    loc.params["epoch"] = 100
    loc.params["pretrain"] = False
    loc.params["mode"] = "train"
    loc.params["batch_size"] = "32"
    loc.params["version"] = None
    loc.params["p"] = None
    loc.params["fine_tune_resnet"] = False
    loc.aae_semantic_params["decoder"] = False

    if loc.params["mode"]:
        aae_semantic_wrapper = AAE_Semantic_Wrapper(loc.aae_semantic_params["dims"],
                                                    loc.aae_semantic_params["latent_dim"])
        aae_semantic_wrapper.train(gd.train_loader, gd.val_loader, gd.test_loader)

if __name__ == '__main__':
    run()


