import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import utils.loc as loc
from models.Resnet50_new import Resnet50_new
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import io
import os
import pickle
import math
import random
import utils.get_data as gd
import numpy as np
from utils.loss import contrastive_loss, euclidean_loss
import torch.optim as optim
from utils.AverageMeter import AverageMeter
import time
import profile
from collections import deque
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init__()

        if loc.params["feature_extractor_name"] == "resnet50":
            self.feature_extractor_model = Resnet50_new()

    def forward_once(self, x):
        return self.feature_extractor_model.forward(x)

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

class SiameseNNWrapper():
    def __init__(self):
        self.nn = SiameseNN()

        if not loc.params["pretrain"]:
            length = len(os.listdir(loc.siamese_dir))
            self.version = length
        else:
            assert loc.params["p"] is not None
            assert loc.params["version"] is not None
            self.version = loc.params["version"]
            self.p = loc.params["p"]
            self.load_model()

        if loc.params["cuda"]:
            self.nn.cuda()

    def train(self, train_dataloader, eval_dataloader, fine_tune=False):
        self.nn.train()
        optimizer = optim.Adam(self.nn.parameters())


        batch_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        if fine_tune:
            status = "fine_tune"
        else:
            status = "train"
        for j in range(loc.params["epoch"]):
            for i, (x1, x2, label) in enumerate(train_dataloader):
                label = label.float()
                if loc.params["cuda"]:
                    x1, x2, label = x1.cuda(), x2.cuda(), label.cuda()
                x1 = Variable(x1)
                x2 = Variable(x2)
                label = Variable(label)

                output1, output2 = self.nn.forward(x1, x2)
                loss = contrastive_loss(output1, output2, label, margin=loc.params["margin"])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()
                losses.update(loss.data[0], x1.shape[0])

            self.save_model()
            print("Simese {}, Epoch[{}], Loss[{}]({})".format(status, j, losses.val, losses.avg))
            self.evaluate(eval_dataloader)

    def evaluate(self, dataloader):
        self.nn.eval()
        eval_acc = AverageMeter()
        output2 = None
        for i, (x1, x2, label) in enumerate(dataloader):
            # x1 = torch.squeeze(x1, 0)
            x2 = x2[0]
            if loc.params["cuda"]:
                x1, x2 = x1.cuda(), x2.cuda()
            x1 = Variable(x1)
            if output2 is None:
                x2 = Variable(x2)
                output2 = self.nn.forward_once(x2)
            output1 = self.nn.forward_once(x1)
            acc = self.get_acc(output1, output2, label)
            eval_acc.update(acc, x1.shape[0])
        # print("Siamese Test, acc[{}]({})".format(eval_acc.val, eval_acc.avg))
        return eval_acc.avg

    def get_acc(self, o1, o2, label):
        result = []

        length = o2.shape[0]
        if length == 5:
            way = 5
            shot = 1
        elif length == 20:
            way = 20
            shot = 1
        elif length == 25:
            way = 5
            shot = 5
        else:
            way = 20
            shot = 5

        for i, feat1 in enumerate(o1):
            for k in range(shot):
                if k == 0:
                    result.append([])
                for j in range(way):
                    feat2 = o2[j*shot + k]
                    if k == 0:
                        result[i].append(euclidean_loss(feat1.view(1, *feat1.shape), feat2.view(1, *feat2.shape)).data.cpu())
                    else:
                        result[i][j] += euclidean_loss(feat1.view(1, *feat1.shape), feat2.view(1, *feat2.shape)).data.cpu()

        result = [np.argmin(arr) for arr in result]
        acc = np.sum(np.array(result) == label.cpu().numpy()) / o1.shape[0]
        return acc

    def predict(self, x1, x2):
        self.nn.eval()
        return self.nn(x1, x2).data.cpu().numpy()[0]

    def save_model(self):
        new_path = os.path.join(loc.siamese_dir, "version_{}".format(self.version))
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        length2 = len(os.listdir(new_path))
        dicta = {
            'state_dict': self.nn.state_dict(),
        }
        dicta.update(loc.params)
        torch.save(dicta, os.path.join(new_path, "{}th.model".format(length2)))
        print("{}th.model saved".format(length2))

    def load_model(self):
        path = os.path.join(loc.siamese_dir, "version_{}".format(self.version))
        if not os.path.exists(path):
            print(path)
            raise ("No model in path {}".format(path))

        checkpoint = torch.load(os.path.join(path, "{}th.model".format(self.p)))
        for key in checkpoint:
            if key != "state_dict":
                print(key, checkpoint[key])
        self.nn.load_state_dict(checkpoint["state_dict"])

class SiameseDataset(Dataset):
    """Siamese Dataset"""
    def __init__(self, dir, number_of_class, number_per_class):
        x1, x2, label = self.create_pairs(dir, number_of_class, number_per_class)

        self.size = len(label)
        self.x0 = x1
        self.x1 = x2
        self.label = label


    def __getitem__(self, index):
        return (gd.get_image(self.x0[index]),
                gd.get_image(self.x1[index]),
                self.label[index])

    def __len__(self):
        return self.size

    def create_pairs(self, dir, number_of_class, number_per_class):
        print("Start Creating pairs")
        classes = os.listdir(dir)
        classes = random.sample(classes, number_of_class)

        x1 = []
        x2 = []
        label = []

        for j, clazz in enumerate(classes):
            p1 = os.path.join(dir, clazz)
            for i in range(number_per_class):
                clazz_images = os.listdir(p1)
                samples = random.sample(clazz_images, 2)
                sample1 = os.path.join(p1, samples[0])
                sample2 = os.path.join(p1, samples[1])
                x1.append(sample1)
                x2.append(sample2)
                label.append(1)

                ind = random.randint(1, len(classes)-1)
                index = (ind + j) % len(classes)
                p2 = os.path.join(dir, classes[index])
                clazz2_images = os.listdir(p2)
                sample2 = random.sample(clazz2_images, 1)
                sample2 = os.path.join(p2, sample2[0])
                x1.append(sample1)
                x2.append(sample2)
                label.append(0)

        print("Length of Dataset: {} pairs".format(len(label)))
        return x1, x2, label

class SiameseEvalDataset(Dataset):
    """Siamese Evaluate Dataset"""
    def __init__(self, dir, number_of_class, number_per_class, shot=1):
        x2, classes = self.get_train(dir, number_of_class, shot)
        self.classes = classes

        x1 = self.create_pairs(dir, number_of_class, number_per_class, shot=shot)

        self.size = len(x1)
        self.x1 = x1
        self.x2 = x2


    def get_train(self, dir, number_of_class, shot):
        classes = os.listdir(dir)
        classes = random.sample(classes, number_of_class)

        groups = {}
        for j, clazz in enumerate(classes):
            p1 = os.path.join(dir, clazz)
            samples = random.sample(os.listdir(p1), shot)
            samples = [os.path.join(p1, sample) for sample in samples]
            groups[j] = deque(samples)

        x2 = []
        for i, clazz in enumerate(groups):
            b = []
            for j in range(shot):
                url = groups[clazz].pop()
                b.append(gd.get_image(url))
            x2.append(b)
        x2 = [torch.stack(x) for x in x2]
        x2 = torch.cat(x2)
        print(x2.shape)
        return x2, classes


    def __getitem__(self, index):
        k = index // len(self.classes)
        m = index % len(self.classes)
        x1 = self.x1[k][m]
        x2 = self.x2
        a = gd.get_image(x1)
        return a, x2, m

    def __len__(self):
        return len(self.x1) * len(self.x1[0])

    def create_pairs(self, dir, number_of_class, number_per_class, shot):

        groups = {}
        for j, clazz in enumerate(self.classes):
            p1 = os.path.join(dir, clazz)
            samples = os.listdir(p1)
            samples = [os.path.join(p1, sample) for sample in samples]
            groups[j] = deque(samples)

        x1 = []
        for k in range(number_per_class):
            a = []
            for i, clazz in enumerate(groups):
                a.append(groups[clazz].pop())
            x1.append(a)


        # print("Length of trials: {}".format(number_per_class * number_of_class * shot))

        return x1


def run():
    siamese_wrapper = SiameseNNWrapper()
    end = time.time()
    if loc.params["mode"] == "train":
        train_ds = SiameseDataset(loc.train_dir, number_of_class=80, number_per_class=600)
        train_dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=loc.params["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        eval_ds = SiameseEvalDataset(loc.train_dir, number_of_class=20, number_per_class=200)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        print("Size of Batch: {}, Spent Time: {}".format(len(train_dataloader), time.time()-end))
        siamese_wrapper.train(train_dataloader=train_dataloader, eval_dataloader = eval_dataloader, fine_tune=False)
    else:
        num_of_class = 20
        shot = 5
        print("way = {}, shot={}".format(num_of_class, shot))
        result = AverageMeter()
        for i in range(10):
            ds = SiameseEvalDataset(loc.val_dir, number_of_class=num_of_class, number_per_class=50, shot=shot)
            dataloader = torch.utils.data.DataLoader(
                ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

            # print("Size of Eval Batch: {}, Spent Time: {}".format(len(dataloader), time.time() - end))
            result.update(siamese_wrapper.evaluate(dataloader))
        return result.avg





def test(time):
    """Get one shot performance"""
    loc.params["pretrain"] = True
    loc.params["mode"] = "test"
    loc.params["epoch"] = 100
    loc.params["version"] = 1
    for i in list(range(0,131))[::10]:
        print("i={}".format(i))
        loc.params["p"] = i
        result = run()
        print("Final Result p[{}] [{}]".format(i, result))

if __name__ == '__main__':
    test(100)
    # loc.params["pretrain"] = True
    # loc.params["mode"] = "test"
    # loc.params["epoch"] = 100
    #
    # loc.params["version"] = 1
    # loc.params["p"] = 30
    # run()
    # profile.run("run()", "prof.txt")
    # import pstats
    # p = pstats.Stats("prof.txt")
    # p.sort_stats("time").print_stats()


