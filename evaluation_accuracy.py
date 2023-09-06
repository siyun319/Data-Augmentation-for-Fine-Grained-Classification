
# This one print and store the accuracy of classifier trained by different data augmentation methods.
from lib2to3.pgen2.pgen import generate_grammar
import numpy as np
import matplotlib.pyplot as plt
import pdb
from sklearn.feature_selection import r_regression
import os
# torch imports
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader,Dataset
from torch import nuclear_norm, optim,nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid

import random
import Constant
import baseline
import data_loader
from CVAEGAN import VAEGAN

from traditional_GAN import Generator



def validate(testloader, netC):
  print("validating----------------------")
  netC.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for i, data in enumerate(testloader):
          inputs, labels = data
          inputs, labels = data[0].to(device), data[1].to(device)
          outputs, _ = netC(inputs)
          # print(outputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = (correct / float(total))

  print('Accuracy of the network on the val images: %f %%' % (
      100.0 * correct / float(total)))
  netC.train()
  return accuracy


if Constant.GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


criterion = nn.CrossEntropyLoss()

print(os.getcwd())
batch_size = 64
the_data_loader = data_loader.Data_Loader(batch_size)
loader_train, loader_val, loader_test = the_data_loader.loader()
dataloaders = {'train': loader_train,
                'val' :loader_val}



# safe traidtional DA
def traditional_da(images):
  for i, image in enumerate(images):
     sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2)
     hflipper = T.RandomHorizontalFlip(p=0.3)
     vflipper = T.RandomVerticalFlip(p=0.3)
     images[i] = sharpness_adjuster(image)
     images[i] = hflipper(image)
     images[i] = hflipper(image)
  return images


# not safe traditional DA
def traditional_da_2(images):
  jitter = T.ColorJitter(brightness=.5, hue=.3)
  for i, image in enumerate(images):
    #  sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2)
    #  hflipper = T.RandomHorizontalFlip(p=0.3)
    #  vflipper = T.RandomVerticalFlip(p=0.3)
     images[i] = jitter(image)
    #  images[i] = hflipper(image)
    #  images[i] = hflipper(image)
  return images


def augMix(images):
    randomInt = 0
    augmenter = T.AugMix(all_ops = False)
    for i, image in enumerate(images):
        randomInt = random.randint(0, 9)
        if randomInt < 3:
            image = (image * 255).to(torch.uint8)
            image = augmenter(image)
            image = (image).to(torch.float32) * 255
            images[i] = image
    return images

model_name = "densenet"
n_classes = 50
advWeight = 0.1
latent_vector_size = 150

# Pre-trained denseNet
net = VAEGAN(z_size=latent_vector_size).to(device)

generator = Generator().to(device)

# path = "/content/gdrive/MyDrive/ic/trained_models/top_50_my_model/2_G_model_classification_loss_epoch_300.pt"
path = "/content/gdrive/MyDrive/ic/trained_models/traditional_GAN/traditional_GAN_G_model_200.pt"
if Constant.Model ==  "CAVEGAN" or Constant.Model ==  "my_model" or Constant.Model == "EC-GAN":
  net.decoder.load_state_dict(torch.load(path))
else:
  generator.load_state_dict(torch.load(path))
  


params_to_update = net.classifier.parameters()
optC = optim.SGD(params_to_update, lr=0.001 * 0.6, momentum=0.9)

# num_epochs = 50


criteron = nn.CrossEntropyLoss()

validate(loader_val, net.classifier)
print('baseline accuracy:')
validate(loader_val, net.classifier)

accuracy_list_test = []
accuracy_list_val = []
for epoch in range(150):

    print("epoch:", epoch)
    print("-------------------------------------")
    for i, data in enumerate(loader_train):
      net.classifier.zero_grad()
      input_real, label_real = data[0], data[1]
      input_real, label_real = input_real.cuda(), label_real.cuda()
      net.classifier.train()

    # Uncomment if use traidtional data augmentation
    # input_real = traditional_da(input_real)
    # input_real = traditional_da_2(input_real)
    # input_real = augMix(input_real)

      # labels = torch.randint(0,num_classes,(batch_size,)).cuda()
      one_hot_class = F.one_hot(label_real, num_classes=n_classes)

      # decode tensor
      # reconstructed_images = net.decoder(z, one_hot_class)

      # sample z randomly
      random_z = torch.randn(len(label_real), net.z_size).cuda()
      # x_p
      if Constant.Model ==  "CAVEGAN" or Constant.Model ==  "my_model" or Constant.Model == "EC-GAN":
          fake_images = net.decoder(random_z, one_hot_class)
      else:
          fake_images = generator(random_z)

      predictionsReal,_ = net.classifier(input_real)
      realClassifierLoss = criteron(predictionsReal, label_real)
      total_loss = realClassifierLoss
      predictionsFake,_ = net.classifier(fake_images)
      # get a tensor of the labels that are most likely according to model
      predictedLabels = torch.argmax(predictionsFake, 1) # -> [0 , 5, 9, 3, ...]
      confidenceThresh = .80

      # psuedo labeling threshold
      probs = F.softmax(predictionsFake, dim=1)
      mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for  i in range(len(probs))])
      toKeep = mostLikelyProbs > confidenceThresh
      if Constant.Model ==  "CAVEGAN" or Constant.Model ==  "my_model" or Constant.Model == "EC-GAN":


          # if evaluate basic image manipulations or augmix solely, change 0.3 to 0.
          fakeClassifierLoss = criteron(predictionsFake, label_real)
          realClassifierLoss = criteron(predictionsReal, label_real)

          total_loss += fakeClassifierLoss * 0.3


      elif sum(toKeep) > 0:
          fakeClassifierLoss = criteron(predictionsFake[toKeep], predictedLabels[toKeep]) 

          total_loss += fakeClassifierLoss * 0.3
      total_loss.backward()

      optC.step()

              
    print("1 epoch end")
    print("val accuracy")


    accuracy_list_val.append(validate(loader_val, net.classifier))

    print("test accuracy")
    accuracy_list_test.append(validate(loader_test, net.classifier))
print("val accuracy:", accuracy_list_val)
print("test accuracy:", accuracy_list_test)
  