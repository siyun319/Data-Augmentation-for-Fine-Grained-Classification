# Visulize the results of generative models
from types import new_class
from sklearn.preprocessing import label_binarize
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
import torch.optim as optim


# from torchmetrics import F1Score

import torch

latent_vector_size = 150
n_classes = 50

import matplotlib.pyplot as plt

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, temp = 2):
        super(DecoderBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=temp)
        # 24 * 4
        self.conv = nn.Conv2d(channel_in, channel_out, 3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(channel_out, 0.8)
        # nn.Le(akyReLU(0.02, inplace=True),
        self.relu = nn.ReLU()


    def forward(self, ten):
        ten = self.up(ten)
        ten = self.conv(ten)
        ten = self.norm(ten)
        ten = self.relu(ten)
        return ten

class Decoder(nn.Module):
    def __init__(self, size = 512, z_size = latent_vector_size):
        super(Decoder, self).__init__()

        # start from B * z_size
        # concatenate one hot encoded class vector
        # size is the input channel
        # self.label_emb = nn.Embedding(n_classes, n_classes)
        self.fc = nn.Sequential(nn.Linear(in_features=(z_size + n_classes), out_features=(12 * 2 * size), bias=False),
                                nn.BatchNorm1d(num_features=12 * 2 * 512, momentum=0.9),
                                nn.ReLU(True))
        self.size = size

        layers = [
            # 512 -> 512
            # 12 * 2 -> 24 * 4
            DecoderBlock(channel_in=self.size, channel_out=self.size),
            # 512 -> 256
            # 24 * 4 -> 48 * 8
            DecoderBlock(channel_in=self.size, channel_out=self.size // 2)]

        self.size = self.size // 2
        # 256 -> 128
        # 48 * 8 -> 96 * 16
        layers.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 2))

        self.size = self.size // 2
        # 128 -> 128
        # 96 * 16 -> 240 * 40
        layers.append(DecoderBlock(channel_in=self.size, channel_out=self.size, temp = 2.5))

        # final conv to get 3 channels and tanh layer
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        ))
        self.conv = nn.Sequential(*layers)

    def forward(self, z, classes_info):
        # classes_info = self.label_emb(classes_info)
        # print("classes_info.shape: ",classes_info.shape)
        # print("z shape:", z.shape)
        ten_cat = torch.cat((z, classes_info), -1)
        # print("ten_cat:", ten_cat.shape)
        ten = self.fc(ten_cat)
        ten = ten.view(len(ten), -1, 12, 2)
        # print("ten:", ten.shape)
        ten = self.conv(ten)
        # print("ten_final:", ten.shape)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)

generator = Decoder()

# "/content/gdrive/MyDrive/ic/trained_models/top_50_encoder_SN/2_G_model_classification_loss_epoch_300_SA.pt"
path = "/content/gdrive/MyDrive/ic/trained_models/top_50/2_G_model_classification_loss_epoch_1400.pt"
generator.load_state_dict(torch.load(path))

labels = torch.tensor([ 0, 1, 1,  1, 1, 2, 2, 2, 3,  3, 3, 32,  32, 32,  5,  5,  5, 47,
        18, 25, 33,  0, 44, 31, 23, 31, 21, 18, 23, 19, 19,  1, 28, 35,  8, 11,
        13, 16, 12, 25, 14, 15, 35,  2, 20, 13, 44,  5,  5, 35, 19, 22, 23, 32,
         1,  1,  6, 39, 18, 49, 49, 49, 49,  49])
one_hot_class = F.one_hot(labels, num_classes=n_classes)
random_z = torch.randn(64, 150)
# print(one_hot_class)
fake_images = generator(random_z, one_hot_class)

samples = fake_images.cpu()
samples = make_grid(samples, nrow=16, padding=2, normalize=True,
                        range=None, scale_each=False, pad_value=0)
plt.figure(figsize = (12,12))
show(samples)


