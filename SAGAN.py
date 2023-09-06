# Self-attention GAN
from ast import Constant
from types import new_class
import data_loader
from spectral_norm import SpectralNorm
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid

import Constant



# this function is from https://github.com/heykeetae/Self-Attention-GAN
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


class Generator(nn.Module):
    def __init__(self, latent_vector_size = 150, n_classes = 50):
        super(Generator, self).__init__()
        latent_vector_size = 150
        self.fc1 = nn.Linear(latent_vector_size, 512 * 12 * 2)
        # self.fc2 = nn.Linear(latent_vector_size + n_classes, 512 * 12 * 2)

        layer1 = []
        layer1.append(nn.BatchNorm2d(512))

        layer1.append(nn.Upsample(scale_factor=2))
        layer1.append(nn.Conv2d(512, 512, 3, stride = 1,padding = 1))
        layer1.append(nn.BatchNorm2d(512,0.8))
        layer1.append(nn.ReLU())


        layer2 = []
        layer2.append(nn.Upsample(scale_factor=2))
        layer2.append(nn.Conv2d(512, 256, 3, stride=1, padding=1))
        layer2.append(nn.BatchNorm2d(256, 0.8))
        layer2.append(nn.ReLU())

        layer3 = []
        layer3.append(nn.Upsample(scale_factor=2))
        layer3.append(nn.Conv2d(256, 128, 3, stride=1, padding=1))
        layer3.append(nn.BatchNorm2d(128, 0.8))
        layer3.append(nn.ReLU())

        layer4 = []
        layer4.append(nn.Upsample(scale_factor=2.5))
        layer4.append(nn.Conv2d(128, 128, 3, stride=1, padding=1))
        layer4.append(nn.BatchNorm2d(128, 0.8))
        layer4.append(nn.ReLU()) 


        # last layer
        layer5 = []
        layer5.append(nn.Conv2d(128, 3, 3, stride=1, padding=1))
        layer5.append(nn.Tanh())


        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)

        # self.attn1 = Self_Attn( 128, 'relu')
        self.attn1 = Self_Attn( 256,  'relu')
        self.attn2 = Self_Attn( 128,  'relu')

    # You can modify the arguments of this function if needed
    def forward(self, z):
        if len(z.shape) == 4:
          z = z.reshape(z.shape[0], -1)
        z = self.fc1(z)

        z = z.view(z.shape[0], 512, 12, 2) 
        z = self.l1(z)
        z = self.l2(z)
        # z, _ = self.attn1(z)
        z = self.l3(z)
        z, p1 = self.attn2(z)
        z = self.l4(z)
        # z,p2 = self.attn2(z)

        out = self.l5(z)

        return out



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        layer1 = []
        layer1.append(SpectralNorm(nn.Conv2d(3, 32, (6, 4), stride=2, padding=1)))
        layer1.append(nn.BatchNorm2d(32))
        layer1.append(nn.LeakyReLU(0.02))


        layer2 = []
        layer2.append(SpectralNorm(nn.Conv2d(32, 64, (6, 4), stride=2, padding=1)))
        layer2.append(nn.BatchNorm2d(64))
        layer2.append(nn.LeakyReLU(0.02))


        layer3 = []
        layer3.append(SpectralNorm(nn.Conv2d(64, 128, (6, 4), stride=2, padding=1)))
        layer3.append(nn.BatchNorm2d(128))
        layer3.append(nn.LeakyReLU(0.02))

        layer4 = []
        layer4.append(SpectralNorm(nn.Conv2d(128, 256, (6, 4), stride=2, padding=1)))
        layer4.append(nn.BatchNorm2d(256))
        layer4.append(nn.LeakyReLU(0.02)) 

        layer5 = []
        layer5.append(SpectralNorm(nn.Conv2d(256, 512, (6, 4), stride=2, padding=1)))
        layer5.append(nn.BatchNorm2d(512))
        layer5.append(nn.LeakyReLU(0.02))

        last = []
        last.append(SpectralNorm(nn.Conv2d(512, 1, (6,1), stride=(2,1), padding=(1,0))))
        last.append(nn.Sigmoid())


        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 512,  'relu')

    def forward(self, x, condition = None):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x,_ = self.attn1(x)
        x = self.l4(x)
        # x, p1 = self.attn1(x)
        x = self.l5(x)
        # x, p2 = self.attn2(x)
        out = self.last(x)


        out = torch.squeeze(out)
        return out



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# When using self-attention GAN, do not use_weights_init
# see https://github.com/heykeetae/Self-Attention-GAN/issues/45

if __name__ == "__main__":
    GPU = Constant.GPU
    colab = Constant.colab

    # TODO: add condition information
    # Increase batch size

    batch_size = 64
    n_classes = 50
    the_data_loader = data_loader.Data_Loader(batch_size)
    loader_train,_,_ = the_data_loader.loader()

    condition = False
    if GPU:
        device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    use_weights_init = False 


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


    def loss_function(out, label):
        adversarial_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()
        loss = adversarial_loss(out, label)
        return loss

    beta1 = 0.5
    learning_rate =  4e-4
    optimizerD = torch.optim.Adam(model_D.parameters(), lr=learning_rate * 0.7, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(model_G.parameters(), lr=learning_rate * 1.5, betas=(beta1, 0.999))
    # optimizerD = torch.optim.SGD(model.parameters(), lr=learning_rate * 0.5, momentum=0.9)

    """<h3> Define fixed input vectors to monitor training and mode collapse. </h3>"""

    latent_vector_size = 150
    fixed_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
    # Additional input variables should be defined here


    train_losses_G = []
    train_losses_D = []

    import tqdm
    num_epochs = 200


    for epoch in range(num_epochs):
    # <- You may wish to add logging info here
        with tqdm.tqdm(loader_train, unit="batch") as tepoch: 
            train_loss_D = 0
            train_loss_G = 0
            for i, data in enumerate(tepoch):
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # train with real
                model_D.zero_grad()
                real_data = data[0].to(device)
                batch_size = real_data.shape[0]
                label = torch.full((batch_size,), 1.0, device=device)

                output = model_D(real_data)
                output = output.to(torch.float32)

                D_error_real = loss_function(output, label.float())

                D_error_real.backward()

                torch.cuda.empty_cache()

                D_x = output.mean().item()

                del output


                # train with fake
                noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
                # print("noise", noise.shape)
                # print("data",real_cpu.shape)
                fake_data = model_G(noise)

                # Fake label
                label = torch.full((batch_size,), 0.0, device=device)

                # print(fake_data.shape)

                output = model_D(fake_data.detach())

                # print("", output.shape)
                D_error_fake = loss_function(output, label.float())
                D_error_fake.backward()

                torch.cuda.empty_cache()

                D_G_z1 = output.mean().item()
                errD = D_error_real + D_error_fake

                del D_error_real, D_error_fake, output

                train_loss_D += errD.item()

                optimizerD.step()

                torch.cuda.empty_cache()

                # (2) Update G network: maximize log(D(G(z)))
                model_G.zero_grad()
                label.fill_(1)
                output = model_D(fake_data)
                errG = loss_function(output, label.float())
                errG.backward()
                D_G_z2 = output.mean().item()
                train_loss_G += errG.item()
                optimizerG.step()
                torch.cuda.empty_cache()

                # Logging 
                if i % 50 == 0:
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(D_G_z=f"{D_G_z1:.3f}/{D_G_z2:.3f}", D_x=D_x,
                                    Loss_D=errD.item(), Loss_G=errG.item())

                del errD, errG


        # if epoch == 0:
        #     save_image(denorm(real_data.cpu()).float(), content_path/'CW_GAN/real_samples.png')
        with torch.no_grad():
            fake = model_G(fixed_noise)
                    # save_image(fake_images.data,
                    #            os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            if Constant.colab:
                save_image(fake.cpu().float(), '/content/gdrive/MyDrive/ic/SAGAN/with_labels_top_50_result/fake_samples_epoch_{}.png'.format(epoch), normalize = True)
            else:
                save_image(fake.cpu().float(), './samples/SAGAN/fake_samples_epoch_{}d.png'.format(epoch), normalize = True)
            # save_image(denorm(fake.cpu()).float(), content_path/'CW_DCGAN/fake_samples_epoch_%03d.png'  % epoch )\
        train_losses_D.append(train_loss_D / len(loader_train))
        train_losses_G.append(train_loss_G / len(loader_train))

        if epoch % 50 == 0 or epoch > 50:
          torch.save(model_G.state_dict(),"/content/gdrive/MyDrive/ic/trained_models/top_50_SAGAN/G_model{}.pt".format(epoch))


    
