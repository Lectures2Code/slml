import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import utils.loc as loc
from torchvision import models

class Resnet50_new(nn.Module):
    def __init__(self):
        super(Resnet50_new, self).__init__()
        self.resnet = models.resnet50()

        checkpoints = torch.load(loc.resnet_model_dir)
        state_dict = checkpoints["state_dict"]

        ### In resnet model there is no module but in checkpoint there is "module"
        new_state_dict = {}

        for k in state_dict:
            new_state_dict[k[7:]] = state_dict[k]

        self.resnet.load_state_dict(new_state_dict)
        self.resnet.name = checkpoints["arch"]

        print("##### Loaded the best resnet50 ckpt... #####")

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


