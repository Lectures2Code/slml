import utils.loc as loc
import os
import pickle
def pkl_to_tsv(gantype, version, p):
    if gantype == "wgan":
        dir = loc.cvaewgan_sample_dir
    else:
        dir = loc.cvaegan_sample_dir

    file = os.path.join(dir, "sample_version_{}_p_{}.pkl".format(version, p))
    f = open(file, "rb")
    pic = pickle.load(f)

    feature_tsv = open(os.path.join(dir, "sample_version_{}_p_{}.tsv".format(version, p)), "w")
    label_tsv = open(os.path.join(dir, "sample_version_{}_p_{}_label.tsv".format(version, p)), "w")

    for loader_type in pic.keys():
        for clazz in pic[loader_type].keys():
            if clazz != "loader":
                for pic_num in pic[loader_type][clazz]:
                    if pic_num.startswith("pic"):
                        # "feat" "recon" "sample"
                        block = pic[loader_type][clazz][pic_num]
                        for type in ["feat", "recon", "sample"]:
                            for i, feat in enumerate(block[type]):
                                print(feat.shape)
                                feature_tsv.write('\t'.join(list(feat.astype("str"))) + "\n")
                                label_tsv.write("{}_{}_{}_{}_{}\n".format(loader_type, clazz, pic_num, type, i))

if __name__ == '__main__':
    pkl_to_tsv("gan", 0, 1)
