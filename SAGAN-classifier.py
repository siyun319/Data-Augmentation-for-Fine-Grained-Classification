# Conditional Self-attention GAN + encoder
# no good result,

from types import new_class
import data_loader
from spectral_norm import SpectralNorm
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid

import Constant


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


n_classes = 50
num_classes = 50
num_epochs = 2000
model_name = "densenet"

latent_vector_size = 150

netC, input_size = baseline.initialize_model(model_name, num_classes, use_pretrained=True)
mse_loss = nn.MSELoss()

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


def loss_function(out, label):
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    loss = adversarial_loss(out, label)
    return loss

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):

        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

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
        self.fc = nn.Sequential(nn.Linear(in_features=(z_size + num_classes), out_features=(12 * 2 * size), bias=False),
                                nn.BatchNorm1d(num_features=12 * 2 * 512, momentum=0.9),
                                nn.ReLU(True))
        self.size = size

        layers1 = [
            # 512 -> 512
            # 12 * 2 -> 24 * 4
            DecoderBlock(channel_in=self.size, channel_out=self.size),
            # 512 -> 256
            # 24 * 4 -> 48 * 8
            DecoderBlock(channel_in=self.size, channel_out=self.size // 2)]

        self.size = self.size // 2
        # 256 -> 128
        # 48 * 8 -> 96 * 16
        layers2 = [DecoderBlock(channel_in=self.size, channel_out=self.size // 2)]

        self.size = self.size // 2
        # 128 -> 128
        # 96 * 16 -> 240 * 40
        layers2.append(DecoderBlock(channel_in=self.size, channel_out=self.size, temp = 2.5))

        # final conv to get 3 channels and tanh layer
        layers2.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        ))
        self.conv1 = nn.Sequential(*layers1)
        self.conv2 = nn.Sequential(*layers2)

        self.attn = Self_Attn( 256,  'relu')



    def forward(self, z, classes_info):

        ten_cat = torch.cat((z, classes_info), -1)

        ten = self.fc(ten_cat)
        ten = ten.view(len(ten), -1, 12, 2)
        ten = self.conv1(ten)
        ten,_ = self.attn(ten)
        ten = self.conv2(ten)
        # print("ten_final:", ten.shape)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)

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
    




# check_classifier()

class VAEGAN(nn.Module):
    def __init__(self, z_size=latent_vector_size):
        super(VAEGAN, self).__init__()

        # latent space size
        self.z_size = z_size
        # self.encoder = GoogleNet()
        self.decoder = Decoder()

        self.discriminator = Discriminator()
        self.classifier = Classifier2()


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


        # historical_c_feature = torch.zeros(n_classes, 1024, 7, 1)
        # historical_d_feature = torch.zeros(2, 128, 5, 1)
        print("epoch:", epoch, "/", num_epochs)
        for i, data in enumerate(loader_train):
                historical_c_feature = torch.zeros(n_classes, 1024, 7, 1)

                print("epoch", epoch, "i:", i)
                inputs = data[0].to(device)
                labels = data[1].to(device)
                net.decoder.zero_grad()
                # net.encoder.zero_grad()
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
                            torch.save(net.classifier.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50/C_model_accuracy_75.pt')
                        else:
                            torch.save(net.classifier.state_dict(), './trained_models/top_50_SA/C_model_accuracy_75.pt')
                        save_C = False
                    # Update discriminator ------------------------------------------------------------
                    # encode
                    # mu, log_variances = net.encoder(inputs)
                    # variances = torch.exp(log_variances * 0.5)
                    # ten_from_normal = torch.randn(len(inputs), net.z_size).cuda()
                    # shift and scale using mean and variances
                    # z = ten_from_normal * variances + mu

                    # kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mu, 2) + variances + 1, 1)
                    one_hot_class = F.one_hot(labels, num_classes=n_classes)

                    # decode tensor
                    # reconstructed_images = net.decoder(z, one_hot_class)

                    # sample z randomly
                    random_z = torch.randn(len(inputs), net.z_size).cuda()
                    # x_p
                    fake_images = net.decoder(random_z, one_hot_class)
                    labels_original, _ = net.discriminator(inputs)
                    labels_fake, _ = net.discriminator(fake_images.detach())
                    # labels_recon, _ = net.discriminator(reconstructed_images.detach())
                    l_g = loss_function(labels_original, label_1.float()) + loss_function(labels_fake, label_0.float()) 
                    loss_D = torch.sum(l_g)

                    # print("-torch.log(labels_original):", torch.sum(-torch.log(labels_original)))

                    loss_D.backward()
                    optimizerD.step()
                    # optimizer_discriminator.step()
                    # lr_discriminator.step()

                    net.discriminator.zero_grad()
                    # net.encoder.zero_grad()


                    # Update generator/decoder ------------------------------------------------------------
                    labels_original, feature_d_real = net.discriminator(inputs)
                    labels_fake, feature_d_fake = net.discriminator(fake_images)

                    # print("feature_d_real shape", feature_d_real.shape)
                    # labels_recon, feature_d_recon = net.discriminator(reconstructed_images)
                    l_g = loss_function(labels_fake, label_1.float())

                    classification_result_real, feature_c_real = net.classifier(inputs_original)
                    classification_result_fake, feature_c_fake = net.classifier(fake_images)

                    # print("feature_c_real shape", feature_c_real.shape)

                    # TODO:change to only learn from those which classifier correctlt predicts for real samples
                    # loss_C = criteron(classification_result_fake, labels)

                    _, preds_fake = torch.max(classification_result_fake, 1)
                    _, preds_real = torch.max(classification_result_real, 1)

                    correct = (preds_fake == labels).sum().item()
                    current_accuracy = (correct / float(labels.size(0)))

                    if current_min_accuracy > current_accuracy:
                      current_min_accuracy = current_accuracy

                    print("accuracy for fake images:", current_accuracy)



                    correct_classifications = (preds_real == labels)
                    correct_pred_samples = inputs[correct_classifications, :, :, :]

                    # TODO: incorporate classfication loss to the generator loss?
                    loss_C = criteron(classification_result_fake[correct_classifications,:], labels[correct_classifications])

                    current_average_c = calculate_mean(historical_c_feature, feature_c_real, labels)
                    historical_c_feature = torch.mul(historical_c_feature, 0.2) + torch.mul(current_average_c, 0.8)

                    # print("average dim:", calculate_mean(historical_c_feature, feature_c_real, labels).shape)
                    # print("non zeros:",torch.nonzero(calculate_mean(historical_c_feature, feature_c_real, labels), as_tuple = True))

                    labels_index = labels[correct_classifications].cpu()

                    # print("hahha", historical_c_feature[labels_index,:,:,:].shape)
                    # print("hehe", feature_c_fake[correct_classifications, :, :, :].shape)
                    target = historical_c_feature[labels_index,:,:,:].to(device)

                    l_gc = mse_loss(target, feature_c_fake[correct_classifications, :, :, :])
                    # l_gd = 0.5 * mse_loss(feature_d_real, feature_d_fake)
                    del target
                    if torch.sum(l_g) - loss_C > 3.5:
                      loss_G = 0.6 * torch.sum(l_g) + loss_C
                    #   print("1:loss_c:", loss_C)
                    else:
                      loss_G = 0.8 * torch.sum(l_g) + loss_C
                    #   print("2:loss_c:", loss_C)
                    # print("l_g:", l_g)

                    loss_G.backward()

                    optimizerG.step()
                    # optimizer_decoder.step()
                    # lr_decoder.step()
                    net.decoder.zero_grad()


                    # # Update encoder ------------------------------------------------------------
                    # mu, log_variances = net.encoder(inputs)
                    # variances = torch.exp(log_variances * 0.5)
                    # ten_from_normal = torch.randn(len(inputs), net.z_size).cuda()
                    # # shift and scale using mean and variances
                    # z = ten_from_normal * variances + mu
                    # kl =  -0.5 * torch.sum(1 + log_variances - mu**2 - log_variances.exp())

                    # fake_images = net.decoder(z, one_hot_class)
                    # reconstructed_images = net.decoder(z, one_hot_class)

                    # labels_original, feature_d_real = net.discriminator(inputs)
                    # labels_fake, feature_d_fake = net.discriminator(fake_images)
                    # labels_recon, feature_d_recon = net.discriminator(reconstructed_images)
                    # # l_g = -torch.log(labels_original) - torch.log(1 - labels_fake) - torch.log(1 - labels_recon)

                    # classification_result_real, feature_c_real = net.classifier(inputs_original)
                    # classification_result_fake, feature_c_fake = net.classifier(fake_images)

                    # reconstructed_loss = 0.5 * mse_loss(inputs, reconstructed_images)

                    # l_gc = 0.5 * mse_loss(feature_c_real, feature_c_fake)
                    # l_gd = 0.5 * mse_loss(feature_d_real, feature_d_fake)

                    # loss_E = 3 * torch.sum(kl) + 10e-2*l_gc + 10e-2*l_gd
                    # # loss_E = 3 * torch.sum(kl) + reconstructed_loss
                    # loss_E.backward()
                    # optimizer_encoder.step()
                    # lr_encoder.step()
                    # net.encoder.zero_grad()


                    # print("i:", i)
                    # print("loss G:", loss_G)
                    # print("loss E:", loss_E)
                    # print("loss D:", loss_D)
                    # print("loss C:", loss_C)

                    with torch.no_grad():
                          j = 0
                          for each_label in labels:
                                if Constant.colab:
                                    if not os.path.exists("/content/gdrive/MyDrive/ic/with_labels_top_50/" + str(each_label.item())):
                                        os.makedirs("/content/gdrive/MyDrive/ic/with_labels_top_50/" + str(each_label.item()))
                                    save_image(fake_images[j].cpu().float(), '/content/gdrive/MyDrive/ic/with_labels_top_50/' + str(each_label.item()) + "/epoch{}.png".format(epoch))
                                else:
                                    if not os.path.exists("./samples/SA_GAN/with_labels_top_50/" + str(each_label.item())):
                                        os.makedirs("./samples/SA_GAN/with_labels_top_50/" + str(each_label.item()))
                                    save_image(fake_images[j].cpu().float(), './samples/SA_GAN/with_labels_top_50/' + str(each_label.item()) + "/epoch{}.png".format(epoch))

                                j+=1

        #     # 1 epoch end
        with torch.no_grad():
                    if accuracy > 0.75:
                      if Constant.colab:
                            save_image(fake_images.cpu().float(), '/content/gdrive/MyDrive/ic/with_labels_top_50_result/fake_samples_epoch_{}d.png'.format(epoch), normalize = True)
                            print("epoch end------------")
                            print("loss G:", loss_G)
                            print("l_g", l_g)
                            print("l_c", loss_C)
                            # print("loss E:", loss_E)
                            print("loss D:", loss_D)
                            print("loss C:", loss_C)
                      else:
                          save_image(fake_images.cpu().float(), './samples/SA_GAN/with_labels_top_50_result/fake_samples_epoch_{}d.png'.format(epoch), normalize = True)

                    # print("epoch", epoch,"accuracy:", running_corrects.double() / len(loader_train.dataset))
              # accuracy_old = running_corrects.double() / len(loader_train.dataset)
        
        if current_min_accuracy > 0.85:
            if Constant.colab:
                    torch.save(net.decoder.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50/G_model_classification_loss.pt')
            else:
                    torch.save(net.decoder.state_dict(), './trained_models/top_50_SA/G_model_classification_loss.pt')
            break

        if epoch % 100 == 0:
            if Constant.colab:
                torch.save(net.decoder.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50/2_G_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(net.discriminator.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50/2_D_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(net.classifier.state_dict(), '/content/gdrive/MyDrive/ic/trained_models/top_50/2_C_model_classification_loss_epoch_{}.pt'.format(epoch))
            else:
                torch.save(net.decoder.state_dict(), './trained_models/top_50_SA/G_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(net.discriminator.state_dict(), './trained_models/top_50_SA/D_model_classification_loss_epoch_{}.pt'.format(epoch))
                torch.save(net.classifier.state_dict(), './trained_models/top_50_SA/C_model_classification_loss_epoch_{}.pt'.format(epoch))


        if Constant.colab is False and accuracy > 0.75:
            print("epoch finished")
            print("loss G:", loss_G)
            print("loss D:", loss_D)
            print("loss C:", loss_C)
    #             # ten_original, reconstructed_images, feature_c_real, feature_c_fake, feature_d_real, feature_d_fake,aux_result_real, aux_result_fake,aux_result_reconstructed, mu, variances, labels, predicted_labels = net(inputs, labels)
                
    #             # ten_original, ten_recon, feature_c_real, feature_c_fake,feature_d_real, feature_d_fake, 
    #         #  labels_original, labels_fake, labels_recon, mu, variances, labels, predicted_labels,
    #             # loss_G, loss_E, loss_D, loss_C = net.loss(ten_original, reconstructed_images, feature_c_real, feature_c_fake, feature_d_real, feature_d_fake, aux_result_real, aux_result_fake,aux_result_reconstructed,mu, variances, labels, predicted_labels)
                



    #             # loss_G.backward(retain_graph=True)
    #             # optimizer_decoder.step()

    #             # loss_C.backward(retain_graph=True)
    #             # optimizer_classifier.step()

    #             # loss_D.backward(retain_graph=True)
    #             # optimizer_discriminator.step()

    #             # loss_E.backward(retain_graph=True)
    #             # optimizer_encoder.step()

    #             # lr_classifier.step()
    #             # lr_discriminator.step()
    #             # lr_decoder.step()
    #             # lr_encoder.step()
                


      

    
    # # model_G.apply(weights_init)
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
      random_z = torch.randn(len(inputs), net.z_size).cuda()
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
          fakeClassifierLoss = criteron(predictionsFake[toKeep], labels[toKeep]) * 0.05
          realClassifierLoss = criteron(predictionsReal[toKeep], label_real[toKeep])
          fakeClassifierLoss.backward()
          realClassifierLoss.backward()

          optimizer_ft.step()

          _, predicted = torch.max(predictionsFake, 1)
          correct_train = predicted.eq(labels.data).sum().item()
          # train_accuracy = correct_train / float(batch_size)
          accuracy_new = validate(loader_val, net.classifier)

          print("loss_C", loss_C)
          print("new loss", fakeClassifierLoss)
          print("old accuracy val:", accuracy_old)
          print("new accuracy val:", accuracy_new)
      net.classifier.zero_grad()

def test():
    net = VAEGAN(z_size=latent_vector_size).to(device)
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

net = VAEGAN(z_size=latent_vector_size).to(device)


learning_rate =  2e-4
optimizer_ft = optim.SGD(params=net.classifier.parameters(), lr=0.001, momentum=0.9)

beta1 = 0.5
optimizerD = torch.optim.Adam(net.discriminator.parameters(), lr=learning_rate * 0.5, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(net.decoder.parameters(), lr=learning_rate * 1.5, betas=(beta1, 0.999))
criteron = nn.CrossEntropyLoss()


train()