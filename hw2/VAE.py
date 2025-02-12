import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, face_data_path):
        super(FaceDataset, self).__init__()
        self.data = np.load(face_data_path) / 255.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AE(nn.Module):
    def __init__(self, latent_dim=32):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
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
        self.fc_latent = nn.Linear(128 * 3 * 3, latent_dim)
        
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

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode
        x = self.encoder(x) # Shape: (batch_size, 128, 3, 3)
        z = x.view(batch_size, -1) # Shape: (batch_size, 128 * 3 * 3)
        z = self.fc_latent(z) # Shape: (batch_size, latent_dim)
        
        # Decode
        x = self.fc_dec(z).view(batch_size, 128, 3, 3)
        x = self.decoder(x)
        return x

    def encoder_forward(self, x):   
        with torch.no_grad():
            x = self.encoder(x)
            z = x.view(x.size(0), -1)
            z = self.fc_latent(z)
        return z
    
    def decoder_forward(self, z):
        with torch.no_grad():
            x = self.fc_dec(z).view(-1, 128, 3, 3)
            x = self.decoder(x)
        return x
    
    
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
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
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode
        x = self.encoder(x) # Shape: (batch_size, 128, 3, 3)
        x = x.view(batch_size, -1) # Shape: (batch_size, 128 * 3 * 3)
        
        # Mu shape: (batch_size, latent_dim)
        # Logvar shape: (batch_size, latent_dim)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar) # Shape: (batch_size, latent_dim)
        
        # Decode
        x = self.fc_dec(z).view(batch_size, 128, 3, 3)
        x = self.decoder(x)
        return x, mu, logvar
    
    def encoder_forward(self, x):   
        with torch.no_grad():
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
            mu, logvar = self.fc_mu(x), self.fc_logvar(x)
            z = self.reparameterize(mu, logvar)
        return z

    def decoder_forward(self, z):
        with torch.no_grad():
            x = self.fc_dec(z).view(-1, 128, 3, 3)
            x = self.decoder(x)
        return x