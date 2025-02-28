import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, face_data_path):
        super(FaceDataset, self).__init__()
        self.data = np.load(face_data_path) / 255.
        self.data = torch.from_numpy(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].unsqueeze(0)
        
        return data
    
# Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 6 * 6),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            # Upsample from 6x6 to 12x12
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Upsample from 12x12 to 24x24
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 6, 6)
        img = self.deconv(x)
        return img

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            # Input: 1 x 24 x 24, output: 64 x 12 x 12
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 12 x 12 -> 128 x 6 x 6
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        validity = self.fc(x)
        return validity


class VAE_Generator(nn.Module):
    '''
    This architecture is taken directly from homework 2 and loaded from those weights
    '''
    
    def __init__(self, latent_dim=32):
        super(VAE_Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder - included so that weights will still load
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # Output: 12x12
            nn.ReLU(),
            nn.BatchNorm2d(32),  
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 6x6
            nn.ReLU(),
            nn.BatchNorm2d(64), 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.fc_mu = nn.Linear(128 * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(128 * 3 * 3, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128 * 3 * 3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1, padding=1),  # Output: 6x6
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1, padding=1),  # Output: 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, output_padding=1, padding=1),  # Output: 24x24
            nn.Sigmoid()
        )
        
    def forward(self, z):
        return self.decoder_forward(z)
        
    def decoder_forward(self, z):
        with torch.no_grad():
            x = self.fc_dec(z).view(-1, 128, 3, 3)
            x = self.decoder(x)
        return x
    

# GAN class that encapsulates the generator, discriminator, and training/validation routines
class GAN:
    def __init__(self, latent_dim=20, g_mode='normal', lr=0.0002, device='cpu'):
        self.device = device
        self.latent_dim = latent_dim
        
        if(g_mode == 'normal'):
            self.generator = Generator(latent_dim).to(device)
            
        elif(g_mode == 'vae'):
            self.generator = VAE_Generator().to(device)
            
        self.discriminator = Discriminator().to(device)
        self.optim_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    # def train_step(self, real_imgs):
    #     batch_size = real_imgs.size(0)
    #     # Create labels for real and fake images
    #     valid = torch.ones(batch_size, 1, device=self.device)
    #     fake = torch.zeros(batch_size, 1, device=self.device)

    #     # -----------------
    #     #  Train Generator
    #     # -----------------
    #     self.optim_G.zero_grad()
    #     # Sample noise and generate images
    #     z = torch.randn(batch_size, self.latent_dim, device=self.device)
    #     gen_imgs = self.generator(z)
    #     # Want discriminator(gen_imgs) to be close to valid (1)
    #     g_loss = self.criterion(self.discriminator(gen_imgs), valid)
    #     g_loss.backward()
    #     self.optim_G.step()

    #     # ---------------------
    #     #  Train Discriminator
    #     # ---------------------
    #     self.optim_D.zero_grad()
    #     # Loss on real images
    #     real_loss = self.criterion(self.discriminator(real_imgs), valid)
    #     # Loss on fake images (detach so generator isn’t updated)
    #     fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)
    #     d_loss = (real_loss + fake_loss) / 2
    #     # d_loss.backward()
    #     self.optim_D.step()

    #     return g_loss.item(), d_loss.item()
    
    # Train step with relaxed GAN parameters
    def train_step(self, real_imgs):
        batch_size = real_imgs.size(0)
        
        # Define label smoothing values and noise parameters
        real_label = 0.9   # Smoothed label for real images
        fake_label = 0.0
        gen_target = 1.0   # Generator still aims for a label of 1.0
        noise_std = 0.1    # Standard deviation for noise added to real images

        # Optionally add a small amount of noise to the real images
        real_imgs_noisy = real_imgs + torch.randn_like(real_imgs) * noise_std

        # -----------------
        #  Train Generator
        # -----------------
        self.optim_G.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        gen_imgs = self.generator(z)
        # Generator tries to fool the discriminator into classifying generated images as real
        g_loss = self.criterion(self.discriminator(gen_imgs), 
                                torch.full((batch_size, 1), gen_target, device=self.device))
        g_loss.backward()
        self.optim_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optim_D.zero_grad()
        # Use label smoothing for real images
        real_loss = self.criterion(self.discriminator(real_imgs_noisy), 
                                torch.full((batch_size, 1), real_label, device=self.device))
        fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), 
                                torch.full((batch_size, 1), fake_label, device=self.device))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optim_D.step()

        return g_loss.item(), d_loss.item()

    def validate(self, test_dataloader):
        self.generator.eval()
        self.discriminator.eval()
        total_g_loss = 0
        total_d_loss = 0

        with torch.no_grad():
            for real_imgs in test_dataloader:
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)

                # Generate fake images
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                gen_imgs = self.generator(z)

                # Compute generator loss
                g_loss = self.criterion(self.discriminator(gen_imgs), valid)
                # Compute discriminator loss
                real_loss = self.criterion(self.discriminator(real_imgs), valid)
                fake_loss = self.criterion(self.discriminator(gen_imgs), fake)
                d_loss = (real_loss + fake_loss) / 2

                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()

        self.generator.train()
        self.discriminator.train()
        num_batches = len(test_dataloader)
        return total_g_loss / num_batches, total_d_loss / num_batches
    