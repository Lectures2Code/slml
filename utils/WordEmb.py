import pickle
import utils.loc as loc
import os
import pandas as pd
import numpy as np

class WordEmb():
    def __init__(self):
        self.csv = pd.read_csv("/home/xmz/dict/imagenet_label.txt", sep=" ")
        self.label_embedding = pickle.load(open("/home/xmz/dict/label_embedding.pkl", "rb"))

    def get_class_name_by_nindex(self, nindex):
        return self.csv[self.csv["class_nindex"]==nindex]["class_name"].values[0]

    def get_word_embedding_by_nindex(self, nindex):
        return self.label_embedding[nindex]

    def get_batch_word_embedding_by_nindex(self, nindexes):
        result = []
        for nindex in nindexes:
            result.append(self.label_embedding[nindex])
        return np.stack(result)