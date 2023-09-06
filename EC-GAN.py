
# CGAN with external classifier, pre-trained(EC-GAN)
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


from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

model_name = "densenet"
num_classes = 50
n_classes = 50

# save_C = True

# f1 = F1Score(num_classes=num_classes)
# >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
# >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
# >>> f1 = F1Score(num_classes=3)
# >>> f1(preds, target)

netC, input_size = baseline.initialize_model(model_name, num_classes, use_pretrained=True)
mse_loss = nn.MSELoss()

latent_vector_size = 150
num_epochs = 2000
if Constant.GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

torch.manual_seed(0)

batch_size = 64
the_data_loader = data_loader.Data_Loader(batch_size)
loader_train, loader_val, loader_test = the_data_loader.loader()
dataloaders = {'train': loader_train,
                'val' :loader_val}


import torch
googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)

vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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


# Generator
# 2 fully-connected layer, followed by 6 deconv layers with 2-by-2 upsampling
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

def check_decoder():
    input, label = next(iter(loader_test))
    F.one_hot(label, num_classes= n_classes)
    input, label = input.cpu(), label.cpu()
    decoder = Decoder(512)
    decoder.apply(weights_init)
    noise = torch.randn(batch_size, latent_vector_size, device=device)
    result = decoder(noise, label)
    print("decoder shape:", result.shape)
    print("decoder done")

# check_decoder()



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator_1 = nn.Sequential(           
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
            nn.LeakyReLU(0.02),)

        self.discriminator_2 = nn.Sequential(

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
        x = self.discriminator_1(x)
        f_d = x
        x = self.discriminator_2(x)
        # print(x.shape)
        out = self.sigmoid(x)
        out = torch.squeeze(out)
        return out, f_d



def check_discriminator():
    input, label = next(iter(loader_test))
    input, label = input.cpu(), label.cpu()
    discriminator = Discriminator()
    result = discriminator(input)
    print("result shape:", result[0].shape)
    print("intermediate layer shape:", result[1].shape)
    print("Discriminator Done")

# check_discriminator()

def initialize_model( num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = vgg
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224

    return model_ft, input_size

# TODO: access intermdediate layer output.
vgg, input_size = initialize_model(n_classes)

# print("vgg:", vgg)

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2,self).__init__()
        self.features = netC.features
        # self.fc = nn.Linear(7168, 1024)
        self.classifer = netC.classifier

    def forward(self, x):
        f_c = self.features(x)
        out = F.relu(f_c, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        out = self.classifer(out)
        return out, f_c

def check_denseNet():
    classifier2 = Classifier2()
    print(netC)
    input, label = next(iter(loader_test))
    input, label = input.cpu(), label.cpu()
    classifier2(input)
    print(netC)
    

# check_denseNet()


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()

        # load the original google net
        self.feature = nn.Sequential(*list(vgg.features.children())[0:15])
        self.max1 = nn.MaxPool2d(kernel_size=(2, 1), stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv1 = nn.Conv2d(512, 256, (4,3),stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.avg = nn.AdaptiveAvgPool2d()
        # self.classifier = vgg.classifier

        self.fc1 = nn.Linear(in_features=10752, out_features=2048, bias = True)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        # self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias = True)
        self.fc3 = nn.Linear(in_features=2048, out_features=n_classes, bias = True)


    def forward(self, x):
        x = self.feature(x)
        # intermediate layer of the classification
        f_c = x
        # print("shape:", x.shape)
        x = self.max1(x)
        # print("shape after:", x.shape)
        x = self.conv1(x)
        # print("shape after conv1:", x.shape)

        # Don't use the original classifier, since it requires x to have size 7 x 7
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x, f_c

def check_classifier():
    input, label = next(iter(loader_test))
    input, label = input.cpu(), label.cpu()
    classifier = Classifier()
    result = classifier(input)
    print("result shape:", result[0].shape)
    print("intermediate layer shape:", result[1].shape)
    print("Done")

# check_classifier()

class EC_GAN(nn.Module):
    def __init__(self, z_size=latent_vector_size):
        super(EC_GAN, self).__init__()

        # latent space size
        self.z_size = z_size
        self.decoder = Decoder()

        self.discriminator = Discriminator()
        self.classifier = Classifier2()


def check_CVAE_GAN():
    net = EC_GAN(z_size=latent_vector_size).to(device)
    input, label = next(iter(loader_test))
    print("max-----", torch.max(label))
    input, label = input.cuda(), label.cuda()
    ten_original, reconstructed_images, feature_c_real, feature_c_fake, feature_d_real, feature_d_fake,aux_result_real, aux_result_fake,aux_result_reconstructed, mu, variances, labels, predicted_labels = net(input, label)
    print("reconstructed_images:", reconstructed_images.shape)
    print("feature_c_real:", feature_c_real.shape)
    print("feature_c_fake:", feature_c_fake.shape)
    print("feature_d_real:", feature_d_real.shape)
    print("feature_d_fake:", feature_d_fake.shape)
    print("aux_result_real", aux_result_real.shape)
    print("aux_result_fake", aux_result_fake.shape)
    print("classification_result_real", predicted_labels.shape)
    print("CVAE_GAN")

# check_CVAE_GAN()


def print_parameters(net):
    print("total number of parameters:")
    params_G = sum(p.numel() for p in net.decoder.parameters() if p.requires_grad)
    params_D = sum(p.numel() for p in net.discriminator.parameters() if p.requires_grad)
    params_E = sum(p.numel() for p in net.encoder.parameters() if p.requires_grad)
    params_C = sum(p.numel() for p in net.classifier.parameters() if p.requires_grad)
    print(params_G + params_D + params_E + params_C)
    print("generator:", params_G)
    print("discriminator:", params_D)
    print("classifier:", params_C)
    print("encoder:", params_E)

def loss_function(out, label):
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    loss = adversarial_loss(out, label)
    return loss

def calculate_mean(target_ten, input_ten, labels):
  counts = np.zeros(n_classes)
  input_ten = input_ten.cpu()
  for i, label in enumerate(labels):
    counts[label] = counts[label] + 1
    target_ten[label] += input_ten[i]

  for label in labels:
    target_ten[label] = torch.div(target_ten[label], counts[label])
  return target_ten

def train():
    save_C = True
  
    for epoch in range(num_epochs):
        net.train()
        running_corrects = 0
        corrects = 0
        current_min_accuracy = 0.7
        average_accuracy = []


        # historical_c_feature = torch.zeros(n_classes, 1024, 7, 1)
        # historical_d_feature = torch.zeros(2, 128, 5, 1)
        print("epoch:", epoch, "/", num_epochs)
        for i, data in enumerate(loader_train):
                historical_c_feature = torch.zeros(n_classes, 1024, 7, 1)

                print("epoch", epoch, "i:", i)
                inputs = data[0].to(device)
                labels = data[1].to(device)
                # print("labels:", labels)
                net.decoder.zero_grad()
                net.encoder.zero_grad()
                net.classifier.zero_grad()
                net.discriminator.zero_grad()

                label_1 = torch.full((batch_size,), 1.0, device=device)
                label_0 = torch.full((batch_size,), 0.0, device=device)


                # save original images
                inputs_original = inputs

                classification_result_real, feature_c_real = net.classifier(inputs_original)
                _, preds = torch.max(classification_result_real, 1)
                running_corrects += torch.sum(preds == labels.data).cpu()
                corrects = torch.sum(preds == labels.data).cpu()
                # print(len(labels))
                accuracy = corrects.double() / len(labels)
                # print("accuracy:", corrects.double() / len(labels))
                # print(classification_result_real)
                

                # Update classifier ------------------------------------------------------------
                
                loss_C = criteron(classification_result_real, labels)
                loss_C.backward()
                optimizer_ft.step()

                net.classifier.zero_grad()



                if accuracy > 0.75:
                    if save_C:
                        if Constant.colab:
                            torch.save(net.classifier.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50_EC-GAN/C_model_accuracy_75.pt')
                        else:
                            torch.save(net.classifier.state_dict(), './trained_models/top_50/C_model_accuracy_75.pt')
                        save_C = False

                    one_hot_class = F.one_hot(labels, num_classes=n_classes)


                    random_z = torch.randn(len(inputs), net.z_size).cuda()
                    # x_p
                    fake_images = net.decoder(random_z, one_hot_class)
                    labels_original, _ = net.discriminator(inputs)
                    labels_fake, _ = net.discriminator(fake_images.detach())
                    l_g = loss_function(labels_original, label_1.float()) + loss_function(labels_fake, label_0.float()) 
                    loss_D = torch.sum(l_g)

                    loss_D.backward()
                    optimizerD.step()

                    net.discriminator.zero_grad()
                    net.encoder.zero_grad()


                    # Update generator/decoder ------------------------------------------------------------
                    labels_original, feature_d_real = net.discriminator(inputs)
                    labels_fake, feature_d_fake = net.discriminator(fake_images)


                    l_g = loss_function(labels_fake, label_1.float())

                    classification_result_real, feature_c_real = net.classifier(inputs_original)
                    classification_result_fake, feature_c_fake = net.classifier(fake_images)


                    _, preds_fake = torch.max(classification_result_fake, 1)
                    _, preds_real = torch.max(classification_result_real, 1)

                    correct = (preds_fake == labels).sum().item()
                    current_accuracy = (correct / float(labels.size(0)))

                    if current_min_accuracy > current_accuracy:
                      current_min_accuracy = current_accuracy

                    average_accuracy.append(current_accuracy)

                    print("accuracy for fake images:", current_accuracy)



                    correct_classifications = (preds_real == labels)
                    correct_pred_samples = inputs[correct_classifications, :, :, :]

                    # TODO: incorporate classfication loss to the generator loss?
                    loss_C = criteron(classification_result_fake[correct_classifications,:], labels[correct_classifications])

                    current_average_c = calculate_mean(historical_c_feature, feature_c_real, labels)
                    historical_c_feature = torch.mul(historical_c_feature, 0.2) + torch.mul(current_average_c, 0.8)


                    labels_index = labels[correct_classifications].cpu()

                    target = historical_c_feature[labels_index,:,:,:].to(device)

                    l_gc = mse_loss(target, feature_c_fake[correct_classifications, :, :, :])
                    del target
                    if torch.sum(l_g) - loss_C > 4.5:
                      loss_G = 0.55 * torch.sum(l_g) + loss_C
                    #   print("1:loss_c:", loss_C)
                    else:
                      loss_G = 0.8 * torch.sum(l_g) + loss_C
                    #   print("2:loss_c:", loss_C)

                    loss_G.backward()

                    optimizerG.step()
                    # optimizer_decoder.step()
                    # lr_decoder.step()
                    net.decoder.zero_grad()

                    with torch.no_grad():
                          j = 0
                          for each_label in labels:
                                if Constant.colab:
                                    if not os.path.exists("/content/gdrive/MyDrive/ic/EC-GAN/with_labels_top_50/" + str(each_label.item())):
                                        os.makedirs("/content/gdrive/MyDrive/ic/EC-GAN/with_labels_top_50/" + str(each_label.item()))
                                    save_image(fake_images[j].cpu().float(), '/content/gdrive/MyDrive/ic/EC-GAN/with_labels_top_50/' + str(each_label.item()) + "/epoch{}.png".format(epoch))
                                else:
                                    if not os.path.exists("./samples/EC_GAN_result/with_labels_top_50/" + str(each_label.item())):
                                        os.makedirs("./samples/EC_GAN_result/with_labels_top_50/" + str(each_label.item()))
                                    save_image(fake_images[j].cpu().float(), './samples/EC_GAN_result/with_labels_top_50/' + str(each_label.item()) + "/epoch{}.png".format(epoch))

                                j+=1

        #     # 1 epoch end
        with torch.no_grad():
                if accuracy > 0.75:
                      if Constant.colab:
                            save_image(fake_images.cpu().float(), '/content/gdrive/MyDrive/ic/EC-GAN/with_labels_top_50_result/fake_samples_epoch_{}d.png'.format(epoch), normalize = True)
                            print("epoch end------------")
                            print("loss G:", loss_G)
                            # print("loss E:", loss_E)
                            print("l_g", l_g)
                            print("l_c", loss_C)
                            print("loss D:", loss_D)
                            print("loss C:", loss_C)
                            print("epoch avergae", sum(average_accuracy)/ float(len(average_accuracy)))
                      else:
                          save_image(fake_images.cpu().float(), './samples/EC_GAN_result/with_labels_top_50_result/fake_samples_epoch_{}d.png'.format(epoch), normalize = True)

                    # print("epoch", epoch,"accuracy:", running_corrects.double() / len(loader_train.dataset))
              # accuracy_old = running_corrects.double() / len(loader_train.dataset)
        
        if accuracy > 0.75:
          if sum(average_accuracy) / float(len(average_accuracy)) > 80:
            break

        if epoch % 100 == 0:
            if Constant.colab:
                torch.save(net.decoder.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50_EC-GAN/2_G_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(net.discriminator.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50_EC-GAN/2_D_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(net.classifier.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50_EC-GAN/2_C_model_classification_loss_epoch_{}.pt'.format(epoch))
                
                torch.save(optimizer_ft.state_dict(),  '/content/gdrive/MyDrive/ic/trained_models/top_50_EC-GAN/2_optC_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(optimizerD.state_dict(),  '/content/gdrive/MyDrive/ic/trained_models/top_50_EC-GAN/2_opt_D_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(optimizerG.state_dict(),  '/content/gdrive/MyDrive/ic/trained_models/top_50_EC-GAN/2_opt_G_model_classification_loss_epoch_{}.pt'.format(epoch))
            else:
                torch.save(net.decoder.state_dict(), './trained_models/top_50/G_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(net.discriminator.state_dict(), './trained_models/top_50/D_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(net.classifier.state_dict(), './trained_models/top_50/C_model_classification_loss_epoch_{}.pt'.format(epoch))


        if Constant.colab is False and accuracy > 0.75:
            print("epoch finished")
            print("loss G:", loss_G)
            print("loss D:", loss_D)
            print("loss C:", loss_C)
            print("epoch avergae", sum(average_accuracy)/ float(len(average_accuracy)))
                
    # Training end-----------
    accuracy_old = validate(loader_val, net.classifier)
    for i in range(20):
      input_real, label_real = next(iter(loader_train))
      input_real, label_real = input_real.cuda(), label_real.cuda()
      net.classifier.train()
      # labels = torch.randint(0,num_classes,(batch_size,)).cuda()
      one_hot_class = F.one_hot(label_real, num_classes=n_classes)

      # decode tensor
      # reconstructed_images = net.decoder(z, one_hot_class)

      # sample z randomly
      random_z = torch.randn(len(label_real), net.z_size).cuda()
      # x_p
      fake_images = net.decoder(random_z, one_hot_class)

      predictionsFake,_ = net.classifier(fake_images)
      predictionsReal,_ = net.classifier(input_real)
      # get a tensor of the labels that are most likely according to model
      predictedLabels = torch.argmax(predictionsFake, 1) # -> [0 , 5, 9, 3, ...]
      confidenceThresh = .85

      # psuedo labeling threshold
      probs = F.softmax(predictionsFake, dim=1)
      mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for  i in range(len(probs))])
      toKeep = mostLikelyProbs > confidenceThresh
      if sum(toKeep) != 0:
          fakeClassifierLoss = criteron(predictionsFake[toKeep], label_real[toKeep]) * 0.05
          realClassifierLoss = criteron(predictionsReal[toKeep], label_real[toKeep])
          fakeClassifierLoss.backward()
          realClassifierLoss.backward()

          optimizer_ft.step()

          _, predicted = torch.max(predictionsFake, 1)
          correct_train = predicted.eq(label_real.data).sum().item()
          # train_accuracy = correct_train / float(batch_size)
          accuracy_new = validate(loader_val, net.classifier)

          print("loss_C", loss_C)
          print("new loss", fakeClassifierLoss)
          print("old accuracy val:", accuracy_old)
          print("new accuracy val:", accuracy_new)
      net.classifier.zero_grad()


def test():
    net = EC_GAN(z_size=latent_vector_size).to(device)
    labels = torch.randint(0,num_classes,(batch_size,)).cuda()
    one_hot_class = F.one_hot(labels, num_classes=n_classes)
    random_z = torch.randn(batch_size, net.z_size).cuda()
    fake_images = net.decoder(random_z, one_hot_class)
    predictionsFake, _ = net.classifier(fake_images)

    print("prediction shape:", predictionsFake.shape)
    print("labels shape:", labels.shape)
    print("success")


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

  print('Accuracy of the network on the val images: %d %%' % (
      100 * correct / float(total)))
  netC.train()
  return accuracy


lr = 1.5e-3
decay_lr = 0.75

net = EC_GAN(z_size=latent_vector_size).to(device)

net.decoder.apply(weights_init)
net.discriminator.apply(weights_init)

if __name__ == "__main__":
  learning_rate =  2e-4
  optimizer_ft = optim.SGD(params=net.classifier.parameters(), lr=0.001, momentum=0.9)

  beta1 = 0.5
  optimizerD = torch.optim.Adam(net.discriminator.parameters(), lr=learning_rate * 0.5, betas=(beta1, 0.999))
  optimizerG = torch.optim.Adam(net.decoder.parameters(), lr=learning_rate * 1.5, betas=(beta1, 0.999))
  criteron = nn.CrossEntropyLoss()


  train()




