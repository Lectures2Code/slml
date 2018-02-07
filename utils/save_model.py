import torch
import os
import utils.loc as loc


def save_model(dir):
    def save_model(self):
        new_path = os.path.join(dir, "version_{}".format(self.version))
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        length2 = len(os.listdir(new_path))
        dicta = {
            'state_dict': self.nn.state_dict(),
        }
        dicta.update(loc.params)
        torch.save(dicta, os.path.join(new_path, "{}th.model".format(length2)))
        print("{}th.model saved".format(length2))