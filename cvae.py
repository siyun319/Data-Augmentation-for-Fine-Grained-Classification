from re import M
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
import data_loader

import os

import baseline
import Constant
# *CODE FOR PART 1.1a IN THIS CELL*

googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)

latent_vector_size = 150

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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Decoder
        z_size = latent_vector_size
        n_classes = 50
        size = 512
        self.fc_decode = nn.Sequential(nn.Linear(in_features=(z_size + n_classes), out_features=(12 * 2 * size), bias=False),
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
        # Decoder

        self.model = googlenet
        
        # remove the last two layers (fc and dropout)
        self.model = nn.Sequential(*list(self.model.children())[:-6])
        # print(self.model)
        

        self.dropout = nn.Dropout(0.2, inplace=False)

        self.conv1 = nn.Conv2d(832, 256,3,1,1)
        self.norm1 = nn.BatchNorm2d(num_features=256, momentum=0.9)



        # self.fc = nn.Linear(1024, 200, bias=False)
        self.fc = nn.Sequential(nn.Linear(in_features=256*15*2, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                nn.ReLU(True))

        self.l_mu = nn.Linear(in_features=1024, out_features=latent_vector_size)
        self.l_var = nn.Linear(in_features=1024, out_features=latent_vector_size)

        
    def encode(self, x):
        x = self.model(x)
        x = self.conv1(x)
        x = self.norm1(x)

        # x = F.layer_norm(x,x.size[1:],elementwise_affine=False)

        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        mu = self.l_mu(x)
        logvar = self.l_var(x)

        # print("mu", mu.shape)
        return mu, logvar

    
    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar/2.0)
        noise_norm = torch.randn_like(std)
        return mu + noise_norm*std



    # TODO:Change this
    def decode(self, z, classes_info):
        # classes_info = self.label_emb(classes_info)
        # print("classes_info.shape: ",classes_info.shape)
        # print("z shape:", z.shape)
        ten_cat = torch.cat((z, classes_info), -1)
        # print("ten_cat:", ten_cat.shape)
        ten = self.fc_decode(ten_cat)
        ten = ten.view(len(ten), -1, 12, 2)
        # print("ten:", ten.shape)
        ten = self.conv(ten)
        # print("ten_final:", ten.shape)
        return ten

    def forward(self, x, classes_info):


        mu, logvar = self.encode(x)

        # print("x",x.shape)
        # print("mu", mu.shape)
        z = self.reparametrize(mu, logvar)
        # print("z",z.shape)
        result = self.decode(z, classes_info)
        # print(x.shape)
        return result, mu, logvar




device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
# print(model)
# optimizer
learning_rate =  2e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


from torch import optim
# *CODE FOR PART 1.1b IN THIS CELL*

def loss_function_VAE(recon_x, x, mu, logvar, beta = 5):
        # reconstruction_loss = F.mse_loss(recon_x, x)

        kld_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim = 0)
        

        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')

        loss = reconstruction_loss + beta * kld_loss

        print("recon:",reconstruction_loss )
        print("kld:", kld_loss)

        return reconstruction_loss, kld_loss, loss



def get_test_loss(loader, model):
        loss_accum = 0; 
        recon_loss_accum = 0;
        kld_accum = 0;

        for data in loader:
          img,_ = data
          img = img.to(device)
          recon_batch, mu, logvar = model(img)
          recon_loss_, KLD, loss = loss_function_VAE(recon_batch, img , mu, logvar)
          loss_accum += loss.item()
          recon_loss_accum += recon_loss_.item()
          kld_accum += KLD.item()
        return recon_loss_accum/len(loader.dataset), kld_accum/len(loader.dataset), loss_accum/len(loader.dataset), 


kld_loss_train = []
reconstruction_loss_train = []
loss_train = []

kld_loss_test = []
reconstruction_loss_test = []
loss_test = []


torch.manual_seed(0)

batch_size = 64
the_data_loader = data_loader.Data_Loader(batch_size)
loader_train, loader_val, loader_test = the_data_loader.loader()

num_epochs = 2001
path = "/content/gdrive/MyDrive/ic/trained_models/top_50_CVAE/VAE_epoch_180.pt"
model.load_state_dict(torch.load(path))
for epoch in range(181, num_epochs):  
        loss_accum = 0; 
        recon_loss_accum = 0;
        kld_accum = 0;
        current_len = 0
        print("epoch:", epoch)

        for i, data in enumerate(loader_train):   
            print("i:", i)

            model.train()

            inputs = data[0].to(device)
            labels = data[1].to(device)

            # Move to GPU
            # print(data.shape)
            one_hot_class = F.one_hot(labels, num_classes=50)
            recon_result, mu, logvar = model(inputs, one_hot_class)

            # print("shape:", recon_result.shape)

            # print(recon_result.shape)
            recon_loss, KLD_loss, loss = loss_function_VAE(recon_result, inputs, mu, logvar)


            # print("loss", loss)
            # loss_accum += loss.item()

            # recon_loss_accum += recon_loss.item()

            # kld_accum += KLD_loss.item()

            # current_len += len(inputs)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


        # 1 epoch finish

        # loss_train.append(loss_accum / current_len)
        # kld_loss_train.append(kld_accum / current_len)
        # reconstruction_loss_train.append(recon_loss_accum / current_len)

        # Get test loss
        # recon_loss_test, KLD_loss_test, loss_test_ = get_test_loss(loader_test, model)

        # loss_test.append(loss_test_)

        # kld_loss_test.append(KLD_loss_test)

        # reconstruction_loss_test.append(recon_loss_test)
        # print("test loss", loss_test_)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50_CVAE/VAE_epoch_{}.pt'.format(epoch))

        with torch.no_grad():
            save_image(recon_result.cpu().float(), '/content/gdrive/MyDrive/ic/CVAE/with_labels_top_50_result/fake_samples_epoch_{}d.png'.format(epoch), normalize = True)