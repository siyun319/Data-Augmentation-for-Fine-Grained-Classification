# CGAN conditional GAN
GPU = True 
colab = False

from pathlib import Path
from tkinter import N

import tqdm

import os
import Constant
import numpy as npmmi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import data_loader

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt

mean = torch.Tensor([0.4914, 0.4822, 0.4465])
std = torch.Tensor([0.247, 0.243, 0.261])
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

def denorm(x, channels=None, w=None ,h=None, resize = False):
    x = unnormalize(x)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std),                        
])


import torch
import torchvision.datasets as dsets
from torchvision import transforms
# import json
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image


from CVAEGAN import VAEGAN



"""We'll visualize a subset of the test set: """
def visualize_samples():
  samples, _ = next(iter(loader_train))
  samples = samples.cpu()



  samples = make_grid(samples, nrow=8, padding=2, normalize=False,
                          range=None, scale_each=False, pad_value=0)
  plt.figure(figsize = (100,100))
  plt.axis('off')
  show(samples)

# visualize_samples()
n_classes = 50

batch_size = 64
the_data_loader = data_loader.Data_Loader(batch_size)
loader_train, loader_val, loader_test = the_data_loader.loader()
dataloaders = {'train': loader_train,
                'val' :loader_val}

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        latent_vector_size = 150
        self.fc1 = nn.Linear(latent_vector_size + n_classes, 512 * 12 * 2)

        self.encoder = nn.Sequential(          
            nn.BatchNorm2d(512),

            nn.Upsample(scale_factor=2),
            # 24 * 4
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, 0.8),
            # nn.Le(akyReLU(0.02, inplace=True),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            # 48 * 8
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            # nn.LeakyReLU(0.02, inplace=True),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            # 96 * 16
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            # nn.LeakyReLU(0.02, inplace=True),
            nn.ReLU(),

            nn.Upsample(scale_factor=2.5),
            # 240 * 40
            nn.Conv2d(128,128 , 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            # nn.LeakyReLU(0.02, inplace=True),
            nn.ReLU(),

            nn.Conv2d(128, 3, 3, stride=1, padding=1),
            # (32 + 2 - 3) / 1 + 1 = 32
            # Size match 3 x 32 x 32
            nn.Tanh())



    def forward(self, z, classes_info):
        if len(z.shape) == 4:
          z = z.reshape(z.shape[0], -1)
        z = torch.cat((z, classes_info), -1)
        z = self.fc1(z)
        # L
        z = z.view(z.shape[0], 512, 12, 2) 
        out = self.encoder(z)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3,1, (6, 4), stride=2, padding=1)
        self.fc = nn.Linear(1*119*20 + n_classes, 119*20)
        self.discriminator = nn.Sequential(           
            nn.Conv2d(1, 256, (6, 4), stride=2, padding=1),
            # 119,20
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
            # nn.Dropout2d(p=0.25, inplace=False),
            # nn.Conv2d(128, 256, (6,4), stride=2, padding=1),
            # # 58,10
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.02),
            # nn.Dropout2d(p=0.25, inplace=False),
            nn.Conv2d(256, 512, (6, 4), stride=2, padding=1),
            # 28,5
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.02),
            # nn.Dropout2d(p=0.25, inplace=False),
            nn.Conv2d(512, 256, (6,4), stride=2, padding=1),
            # 13,2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),

            nn.Conv2d(256, 128, (6, 4), stride=2, padding=1),
            # 5,1
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),

            nn.Conv2d(128, 1, (6,1), stride=(2,1), padding=(1,0)))

            # 1,1
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.02),

            # nn.Dropout2d(p=0.25, inplace=False),
            # nn.Conv2d(256, 1, 4, 2, 1))
        # 2 + 2 - 4 + 1 = 1
        # size 1 * 1 for latent variable

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, classes_info):
        x = self.conv1(x)
        x = torch.flatten(x, 1)

        # print(x.shape)


        x = torch.cat((x, classes_info), -1)
        x = self.fc(x)
        x = x.view(64, 1, 119, 20)
        x = self.discriminator(x)
        # print(x.shape)
        out = self.sigmoid(x)
        out = torch.squeeze(out)
        return out


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def loss_function(out, label):
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    loss = adversarial_loss(out, label)
    return loss



if __name__ == "__main__":
    num_epochs = 80
    learning_rate =  5e-5
    latent_vector_size = 150

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    if GPU:
        device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f'Using {device}')

    net = VAEGAN(z_size=latent_vector_size).to(device)
    classifier = net.classifier.load_state_dict(torch.load("/content/gdrive/MyDrive/ic/trained_models/top_50_my_model/2_C_model_classification_loss_epoch_300.pt"))

    use_weights_init = True


    model_G = Generator().to(device)
    if use_weights_init:
        model_G.apply(weights_init)
    params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
    print("Total number of parameters in Generator is: {}".format(params_G))
    print(model_G)
    print('\n')

    model_D = Discriminator().to(device)
    if use_weights_init:
        model_D.apply(weights_init)
    params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
    print("Total number of parameters in Discriminator is: {}".format(params_D))
    print(model_D)
    print('\n')

    print("Total number of parameters is: {}".format(params_G + params_D))

    beta1 = 0.5
    optimizerD = torch.optim.Adam(model_D.parameters(), lr=learning_rate * 0.7, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(model_G.parameters(), lr=learning_rate * 1.5, betas=(beta1, 0.999))
    # optimizerD = torch.optim.SGD(model.parameters(), lr=learning_rate * 0.5, momentum=0.9)

    """<h3> Define fixed input vectors to monitor training and mode collapse. </h3>"""
    batch_size = 64
    latent_vector_size = 150
    fixed_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)


    train_losses_G = []
    train_losses_D = []

    import tqdm
    for epoch in range(num_epochs):
        epoch_accuracy = []
        with tqdm.tqdm(loader_train, unit="batch") as tepoch: 
            train_loss_D = 0
            train_loss_G = 0
            for i, data in enumerate(tepoch):
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

                # train with real
                model_D.zero_grad()
                real_data = data[0].to(device)
                labels_original = data[1].to(device)
                labels = F.one_hot(labels_original, num_classes=n_classes)
                batch_size = real_data.shape[0]
                label_real = torch.full((batch_size,), 1.0, device=device)

                output = model_D(real_data, labels)

                output = output.to(torch.float32)

                D_error_real = loss_function(output, label_real.float())

                D_error_real.backward()

                D_x = output.mean().item()


                # train with fake
                noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
                # print("noise", noise.shape)
                # print("data",real_cpu.shape)
                fake_data = model_G(noise, labels)

                # Fake label
                label_fake = torch.full((batch_size,), 0.0, device=device)

                # print(fake_data.shape)

                output = model_D(fake_data.detach(), labels)

                # print("", output.shape)
                D_error_fake = loss_function(output, label_fake.float())
                D_error_fake.backward()
                D_G_z1 = output.mean().item()
                errD = D_error_real + D_error_fake

                train_loss_D += errD.item()

                optimizerD.step()

                # (2) Update G network: maximize log(D(G(z)))
                model_G.zero_grad()
                label_real.fill_(1)
                output = model_D(fake_data, labels)
                errG = loss_function(output, label_real.float())
                errG.backward()
                D_G_z2 = output.mean().item()
                train_loss_G += errG.item()
                optimizerG.step()

                classification_result_fake, _ = net.classifier(fake_data)
                _, preds_fake = torch.max(classification_result_fake, 1)

                correct = (preds_fake == labels_original).sum().item()
                current_accuracy = (correct / float(labels_original.size(0)))
                epoch_accuracy.append(current_accuracy)
                
                print("current accuracy:", current_accuracy)
                ####################################################################### 
                #                       ** END OF YOUR CODE **
                ####################################################################### 
                # Logging 
                if i % 50 == 0:
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(D_G_z=f"{D_G_z1:.3f}/{D_G_z2:.3f}", D_x=D_x,
                                    Loss_D=errD.item(), Loss_G=errG.item())


        # if epoch == 0:
        #     save_image(denorm(real_data.cpu()).float(), content_path/'CW_GAN/real_samples.png')
        with torch.no_grad():
            fake = model_G(fixed_noise, labels)
                    # save_image(fake_images.data,
                    #            os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            if Constant.colab:
                save_image(fake.cpu().float(), '/content/gdrive/MyDrive/ic/CGAN/fake_samples_epoch_{}.png'.format(epoch), normalize = True)
            else:
                save_image(fake.cpu().float(), './samples/traditional_gan/fake_samples_epoch_{}.png'.format(epoch), normalize = True)
            # save_image(denorm(fake.cpu()).float(), content_path/'CW_DCGAN/fake_samples_epoch_%03d.png'  % epoch )\
        train_losses_D.append(train_loss_D / len(loader_train))
        train_losses_G.append(train_loss_G / len(loader_train))

        print("epoch accuracy:", sum(epoch_accuracy) / len(epoch_accuracy))
        

        # one_epoch_end
    # save  models 
        if epoch > 30:
            torch.save(model_G.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/CGAN/CGAN_G_model_{}.pt'.format(epoch))
            torch.save(model_D.state_dict(), './trained_models/CGAN_D_model_{}.pt'.format(epoch))

