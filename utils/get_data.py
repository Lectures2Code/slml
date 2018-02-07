from torchvision import transforms
import torchvision.datasets as datasets
import utils.loc as loc
import torch
from torch.utils.data import DataLoader
from models.Resnet50_new import Resnet50_new
import random
import os
from PIL import Image
from torch.autograd import Variable

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    loc.train_dir,
    transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    num_workers=4, pin_memory=True)

val_dataset = datasets.ImageFolder(
    loc.val_dir,
    transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
]))

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=True,
    num_workers=4, pin_memory=True)


test_dataset = datasets.ImageFolder(
    loc.test_dir,
    transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
]))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=True,
    num_workers=4, pin_memory=True)

def transform_image(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return transform(image)

def get_image(url):
    image = Image.open(url)
    image = image.convert('RGB')
    image = transform_image(image)
    return image

def get_affine_transformation(url):
    image = Image.open(url)

    image1 = image.transpose(Image.ROTATE_90)
    image1 = image1.convert('RGB')
    image1 = transform_image(image1)

    image2 = image.transpose(Image.ROTATE_180)
    image2 = image2.convert('RGB')
    image2 = transform_image(image2)

    image3 = image.transpose(Image.ROTATE_270)
    image3 = image3.convert('RGB')
    image3 = transform_image(image3)

    image4 = image.transpose(Image.FLIP_TOP_BOTTOM)
    image4 = image4.convert('RGB')
    image4 = transform_image(image4)

    image5 = image.transpose(Image.FLIP_LEFT_RIGHT)
    image5 = image5.convert('RGB')
    image5 = transform_image(image5)

    a = [image1, image2, image3, image4, image5]
    return torch.stack(a)



def get_image_featrues(dir = loc.val_dir, sample_size = 30, way = 5):
    model = Resnet50_new()
    model.cuda()
    model.eval()

    images = []
    classes = os.listdir(dir)
    classes = random.sample(classes, way)
    for clazz in classes:
        images.append([])
        p = os.path.join(dir, clazz)
        clazz_images = os.listdir(p)
        samples = random.sample(clazz_images, sample_size)
        for sample in samples:
            sample = os.path.join(p, sample)
            image = get_image(sample)
            feat = model.forward(Variable((image.view(1, *image.shape)).cuda()))
            images[-1].append(feat.cpu().data)
        images[-1] = torch.cat(images[-1])
    images = torch.cat(images)
    return images

