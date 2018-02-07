resnet_model_dir = "/home/xmz/pretrain_ckpt/resnet50/model_best.pth.tar"
train_dir = "/home/xmz/mini-imagenet-v2/pretrain_data/train"
val_dir = "/home/xmz/mini-imagenet-v2/pretrain_data/val"
test_dir = "/home/xmz/mini-imagenet-v2/test"

siamese_dir = "/home/xmz/slml/results/siamese"
cvaegan_dir = "/home/xmz/slml/results/cvaegan"
cvaewgan_dir = "/home/xmz/slml/results/cvaewgan"

cvaegan_sample_dir = "/home/xmz/slml/results/cvaegan_sample"
cvaewgan_sample_dir = "/home/xmz/slml/results/cvaewgan_sample"

semantic_dir = "/home/xmz/slml/results/semantic"
aae_semantic_dir = "/home/xmz/slml/results/aae_semantic"


params = {
    "cuda": True,
    "feature_extractor_name": "resnet50",
    "margin": 10,
    "epoch": 10,
    "pretrain": False,
    "mode": "train",
    "version": None,
    "p": None,
    "batch_size": 32,
    "fine_tune_resnet": True
}

cvae_params = {
    "con_dim": 100,
    "dims": [2048, 1024, 512, 256],
    "latent_dim": 256,
    "norm": 0.01,
    "wgan": False
}

aae_semantic_params = {
    "decoder": True,
    "dims": [2048, 1024, 512, 256],
    "latent_dim": 100,
}