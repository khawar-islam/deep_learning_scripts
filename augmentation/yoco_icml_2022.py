import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

training_data = datasets.ImageFolder(root="/media/cvpr/CM_24/atu", transform=ToTensor())
train_loader = DataLoader(training_data, batch_size=64, shuffle=True)


def YOCO(images, aug, h, w):
    images = torch.cat((aug(images[:, :, :, 0:int(w / 2)]), aug(images[:, :, :, int(w / 2):w])), dim=3) if \
        torch.rand(1) > 0.5 else torch.cat((aug(images[:, :, 0:int(h / 2), :]), aug(images[:, :, int(h / 2):h, :])),
                                           dim=2)
    return images


for i, (images, target) in enumerate(train_loader):
    aug = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(), )
    _, _, h, w = images.shape
    # perform augmentations with YOCO
    images = YOCO(images, aug, h, w)
    save_image(images, 'yoco' + str(i) + '.png')