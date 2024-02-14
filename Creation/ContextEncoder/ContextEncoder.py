# -*- coding: utf-8 -*-
"""
@author: Valentin meo (for any question or bug : valentin.meo.1@ulaval.ca)
This code is MIT licence
"""

#import
import numpy as np
import pandas as pd
import os
import glob
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm_notebook as tqdm


random.seed(42)
import warnings
warnings.filterwarnings("ignore")



########################################
#parameter
########################################
# number of epochs of training
n_epochs =250
# size of the batches
batch_size = 15
# name of the dataset
dataset_name = "/media/seagate/vmeo/campagne/petitEntrainementEchantilonageAleatoir"
# adam: learning rate
lr = 0.00008
# adam: decay of first order momentum of gradient
b1 = 0.5
# adam: decay of first order momentum of gradient
b2 = 0.999
# dimensionality of the latent space
latent_dim = 100
# size of each image dimension
img_size = 256
# size of random mask
mask_size =128
# number of image channels
channels = 3
# interval between image sampling
sample_interval = 500

cuda = True if torch.cuda.is_available() else False
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)



########################################
#dataset
########################################
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=1000, mask_size=498, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.png" % root))
        self.files = self.files[:-1] if mode == "train" else self.files[-1:]

    def apply_random_mask(self, img):
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1
        return masked_img, masked_part

    def apply_center_mask(self, img):
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1
        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            masked_img, aux = self.apply_random_mask(img)
        else:
            masked_img, aux = self.apply_center_mask(img)
        return img, masked_img, aux

    def __len__(self):
        return len(self.files)
    


transforms_ = [
    transforms.Resize((img_size, img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(dataset_name, transforms_=transforms_,img_size=img_size,mask_size=mask_size),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

test_dataloader = DataLoader(
    ImageDataset(dataset_name, transforms_=transforms_,img_size=img_size,mask_size=mask_size, mode="val"),
    batch_size=1,
    shuffle=True,
    num_workers=0,
)    
   
  
########################################
#Utility
########################################
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, real_center):
        # Do your print / debug stuff here
        print(real_center.size())
        return real_center

########################################
#model
########################################
class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU())
            return layers

        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),
#            PrintLayer(),
            *downsample(64, 64),
 #           PrintLayer(),
            *downsample(64, 128),
            nn.Dropout(p=0.1, inplace=False),
  #          PrintLayer(),
            *downsample(128, 256),
   #         PrintLayer(),
            *downsample(256, 512),
    #        PrintLayer(),
            nn.Conv2d(512, 2000, 4, bias=False),
            nn.BatchNorm2d(2000),
            nn.LeakyReLU(0.2),
     #       PrintLayer(),
            nn.ConvTranspose2d(2000, 512, 4,1,0, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(),
            
       #     PrintLayer(),
            nn.ConvTranspose2d(512,256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(),
      #      PrintLayer(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(),
        #    PrintLayer(),
            nn.ConvTranspose2d(64, channels, 4, 2,1, bias=False),
         #  PrintLayer(),
            nn.Tanh()
        )

    def forward(self, x):
        #print(x.size())
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize):#, stride,R,M
            layers = [nn.Conv2d(in_filters, out_filters,4,2,1, bias=False)]# M, stride, R,
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, normalize in [(10, False), (20, True), (30, True),(50,True)]:#, 2,1,4
            layers.extend(discriminator_block(in_filters, out_filters, normalize)) #, stride,R,M
            in_filters = out_filters
        #layers.append(PrintLayer())
        layers.append(nn.Conv2d(out_filters, 1, 5, 1, 2, bias=False))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, img):
        #print(img.size())
        return self.model(img)
    
    

 
########################################
# intialised Training
########################################
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def save_sample(batches_done):
    samples, masked_samples, i = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()  # Upper-left coordinate of mask
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i : i + mask_size, i : i + mask_size] = gen_mask
    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)

    
# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=channels)
discriminator = Discriminator(channels=channels)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()
    print('CUDA bien utiliser')

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
   

########################################
#training
########################################


if __name__ == '__main__':
    gen_adv_losses, gen_pixel_losses, disc_losses, counter = [], [], [], []
    for epoch in range(n_epochs):
        gen_adv_loss, gen_pixel_loss, disc_loss = 0, 0, 0
        tqdm_bar = tqdm(dataloader, desc=f'Training Epoch {epoch} ', total=int(len(dataloader)))
        for i, (imgs, masked_imgs, masked_parts) in enumerate(tqdm_bar):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], *(1, 8, 8)).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], *(1, 8, 8)).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = Variable(masked_parts.type(Tensor))
            
            #Generator part
            optimizer_G.zero_grad()
            gen_parts = generator(masked_imgs)
            overlapPred=4
            overlapL2Weight = 10
            wtl2Matrix = masked_parts.clone()
            wtl2Matrix.data.fill_(0.9992*overlapL2Weight)
            wtl2Matrix.data[:,:,int(overlapPred):int(img_size/2 - overlapPred),int(overlapPred):int(img_size/2 - overlapPred)] = 0.9992
            errG_l2 = (gen_parts-masked_parts).pow(2)
            errG_l2 = errG_l2 * wtl2Matrix
            g_pixel = errG_l2.mean()
            g_adv = adversarial_loss(discriminator(gen_parts), valid)
            
            # Total loss
            g_loss = 0.001 * g_adv + 0.999 * g_pixel
            g_loss.backward()
            optimizer_G.step()

            #discriminator part
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()
            gen_adv_loss, gen_pixel_loss, disc_loss
            gen_adv_losses, gen_pixel_losses, disc_losses, counter
            gen_adv_loss += g_adv.item()
            gen_pixel_loss += g_pixel.item()
            gen_adv_losses.append(g_adv.item())
            gen_pixel_losses.append(g_pixel.item())
            disc_loss += d_loss.item()
            disc_losses.append(d_loss.item())
            counter.append(i*batch_size + imgs.size(0) + epoch*len(dataloader.dataset))
            tqdm_bar.set_postfix(gen_adv_loss=gen_adv_loss/(i+1), gen_pixel_loss=gen_pixel_loss/(i+1), disc_loss=disc_loss/(i+1))

            # Generate sample at sample interval
            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_sample(batches_done)

        torch.save(generator.state_dict(), "saved_models/generator.pth")
        torch.save(generator, "saved_models/g")
        torch.save(discriminator.state_dict(), "saved_models/discriminator.pth")
        
        
        df = pd.DataFrame(counter, columns=["colummn"])
        df.to_csv('counter.csv', index=False)    
        
        
        df = pd.DataFrame(disc_losses, columns=["colummn"])
        df.to_csv('disc_losses.csv', index=False)    
        
        
        df = pd.DataFrame(gen_pixel_losses, columns=["colummn"])
        df.to_csv('gen_pixel_losses.csv', index=False)    
        
        
        df = pd.DataFrame(gen_adv_losses, columns=["colummn"])
        df.to_csv('gen_adv_losses.csv', index=False)    
        
            
