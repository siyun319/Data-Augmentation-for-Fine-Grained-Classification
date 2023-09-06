# Not used
import math
import numpy as np
import matplotlib.pyplot as plt
import pdb

# TODO: Modify accuracy reporting function, make network smaller? Maybe it is too large, remember to fine tune!! learning for generator, discriminator and classifier!!
# TODO: Add early stopping!! class imbalance problem, how to generate for monority class? Currently GAN might better learn from the major class
# torch imports
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader,Dataset
from torch import nuclear_norm, optim,nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid

import Constant
import baseline
import data_loader

if Constant.GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


batch_size = 64
the_data_loader = data_loader.Data_Loader(batch_size)
loader_train, loader_val = the_data_loader.loader()
dataloaders = {'train': loader_train,
                'val' :loader_val}

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        latent_vector_size = 150
        self.fc1 = nn.Linear(latent_vector_size, 1024 * 12 * 2)

        # Modified original discriminator from the paper
        self.encoder = nn.Sequential(          
            nn.BatchNorm2d(1024),

            nn.Upsample(scale_factor=2),
            # 24 * 4
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
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



    def forward(self, z):
        if len(z.shape) == 4:
          z = z.reshape(z.shape[0], -1)
        z = self.fc1(z)
        # L
        z = z.view(z.shape[0], 1024, 12, 2) 
        out = self.encoder(z)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(           
            nn.Conv2d(3, 128, (6, 4), stride=2, padding=1),
            # 119,20
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            # nn.Dropout2d(p=0.25, inplace=False),
            nn.Conv2d(128, 256, (6,4), stride=2, padding=1),
            # 58,10
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
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

        

    def forward(self, x):
        x = self.discriminator(x)
        # print(x.shape)
        out = self.sigmoid(x)
        out = torch.squeeze(out)
        return out


def train(dataloaders):
  subTrainLoader = dataloaders['train']
  subValLoader = dataloaders['val']
  for epoch in range(epochs):
    netC.train()
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    running_corrects = 0
    total_train = 0
    correct_train = 0
    for i, data in enumerate(subTrainLoader):
  
      dataiter = iter(subTrainLoader)
      inputs, labels = dataiter.next()
      inputs, labels = inputs.to(device), labels.to(device)
      tmpBatchSize = len(labels)  

      # create label arrays 
      true_label = torch.full((tmpBatchSize,), 1.0, device=device)
      fake_label = torch.full((tmpBatchSize,), 0.0, device=device)

      r = torch.randn(tmpBatchSize, 150, 1, 1, device=device)
      fakeImageBatch = netG(r)

      real_cpu = data[0].to(device)
      batch_size = real_cpu.size(0)

      # train discriminator on real images
      predictionsReal = netD(inputs)
      lossDiscriminator = loss(predictionsReal, true_label) 
      lossDiscriminator.backward(retain_graph = True)

      # train discriminator on fake images
      predictionsFake = netD(fakeImageBatch)
      lossFake = loss(predictionsFake, fake_label)
      lossFake.backward(retain_graph= True)
      optD.step() # update discriminator parameters    

      # train generator 
      optG.zero_grad()
      predictionsFake = netD(fakeImageBatch)
      lossGenerator = loss(predictionsFake, true_label)
      lossGenerator.backward(retain_graph = True)
      optG.step()

      optC.zero_grad()
      torch.autograd.set_detect_anomaly(True)
      fakeImageBatch = fakeImageBatch.detach().clone()
      # train classifier on real data
      predictions = netC(inputs)
      realClassifierLoss = criterion(predictions, labels)
      realClassifierLoss.backward(retain_graph=True)
      
      optC.step()
      optC.zero_grad()

      # Pesudo-labeling -------------------------------------------
      # update the classifer on fake data
      predictionsFake = netC(fakeImageBatch)
      # get a tensor of the labels that are most likely according to model
      predictedLabels = torch.argmax(predictionsFake, 1) # -> [0 , 5, 9, 3, ...]
      confidenceThresh = .2

      # psuedo labeling threshold
      probs = F.softmax(predictionsFake, dim=1)
      mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for  i in range(len(probs))])
      toKeep = mostLikelyProbs > confidenceThresh
    #   half batch size compared to the baseline model, so quicker learning. 15 or smaller?
    # TODO:change this!
      if sum(toKeep) != 0 and epoch > 30:
          fakeClassifierLoss = criterion(predictionsFake[toKeep], predictedLabels[toKeep]) * advWeight
          fakeClassifierLoss.backward()
          print("using generated images to train classifier now!")

          gridOfFakeImages = torchvision.utils.make_grid(fakeImageBatch.cpu())
          # Save fake images used for training classifier
          if Constant.colab:
            save_image(gridOfFakeImages.cpu().float(), '/content/gdrive/MyDrive/ic/EC-GAN/used_fake_{}.png'.format(i), normalize = True)
          else:
            save_image(gridOfFakeImages.cpu().float(), './samples/used_fake_{}.png'.format(i), normalize = True)
          
      optC.step()

      # reset the gradients
      optD.zero_grad()
      optG.zero_grad()
      optC.zero_grad()

      # save losses for graphing
      generatorLosses.append(lossGenerator.item())
      discriminatorLosses.append(lossDiscriminator.item())
      classifierLosses.append(realClassifierLoss.item())

      _, predicted = torch.max(predictions, 1)
    #   correct predict counts for the whole batch
      running_corrects += predicted.eq(labels.data).sum().item()
      # report accuracy every 50 mini batches
      if(i % 50 == 0):
        netC.eval()
        # accuracy
        _, predicted = torch.max(predictions, 1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
        train_accuracy = 100 * correct_train / float(total_train)
        print("train accuracy :", train_accuracy)
        print("generator loss:", lossGenerator.item())
        print("discriminator loss:", lossDiscriminator.item())

        netC.train()

    print("Epoch " + str(epoch) + "Complete")
    
    epoch_acc = float(running_corrects) / len(subTrainLoader.dataset)
    print("epoch train accuracy:", epoch_acc)
    # save gan image
    gridOfFakeImages = torchvision.utils.make_grid(fakeImageBatch.cpu())
    if Constant.colab:
        save_image(gridOfFakeImages.cpu().float(), '/content/gdrive/MyDrive/ic/EC-GAN/fake_samples_epoch_{}d.png'.format(epoch), normalize = True)
    else:
        save_image(gridOfFakeImages.cpu().float(), './samples/EC-GAN/fake_samples_epoch_{}d.png'.format(epoch), normalize = True)

    validate(subValLoader)

def validate(testloader):
  print("validating----------------------")
  netC.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for i, data in enumerate(testloader):
          inputs, labels = data
          inputs, labels = data[0].to(device), data[1].to(device)
          outputs = netC(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = (correct / float(total)) * 100 

  print('Accuracy of the network on the val images: %d %%' % (
      100 * correct / float(total)))
  netC.train()

device = torch.device("cuda:0")

# data for plotting purposes
generatorLosses = []
discriminatorLosses = []
classifierLosses = []

#training starts

epochs = 100
# TODO: change this!!
learning_rate =  5e-5

# models
netG = Generator()
netD = Discriminator()
# netC = ResNet18()

if Constant.GPU:
    netG.to(device)
    netD.to(device)
# netC.to(device)

# optimizers 
optD = optim.Adam(netD.parameters(), lr=learning_rate * 0.7, betas=(0.5, 0.999), weight_decay = 1e-3)
optG = optim.Adam(netG.parameters(), lr=learning_rate * 1.5, betas=(0.5, 0.999))
# optC = optim.Adam(netC.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay = 1e-3)

advWeight = 0.1 # adversarial weight

loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()



# Classifier, pre-trained Densenet!--------------------------------------------------------------
model_name = "densenet"
num_classes = 594
netC, input_size = baseline.initialize_model(model_name, num_classes, use_pretrained=True)

# Print the model we just instantiateds
# print(model_ft)


if Constant.GPU:
    netC = netC.to(device)
print(device)

params_to_update = netC.parameters()
# print("Params to learn:")

# for name,param in model_ft.named_parameters():
#     if param.requires_grad == True:
#         print("\t",name)
optC = optim.SGD(params_to_update, lr=0.001 * 0.6, momentum=0.9)

criterion = nn.CrossEntropyLoss()          
# ------------------------------------------------------------------------------

params = sum(p.numel() for p in netG.parameters() if p.requires_grad) + sum(p.numel() for p in netC.parameters() if p.requires_grad) + sum(p.numel() for p in netD.parameters() if p.requires_grad)
print("total number of parameters:", params)
train(dataloaders)