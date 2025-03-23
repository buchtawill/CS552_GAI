import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, image_file, age_file):
        raw = np.load(image_file)  # Load data from .npy file
        self.ages = np.load(age_file)
        self.data = raw / 255.
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy array to PyTorch tensor
            transforms.RandomHorizontalFlip(),
            transforms.v2.RandomPhotometricDistort(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize data
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = self.transform(image)
        return image, self.ages[idx]

T = 1000
TIME_EMBEDDING_DIM = 64

def sinusoidal_embedding(times):
    embedding_min_frequency = 1.0
    frequencies = torch.exp(
        torch.linspace(
            np.log(1.0),
            np.log(1000.),
            TIME_EMBEDDING_DIM // 2
        )
    ).view(1, -1).to(times.device)
    angular_speeds = 2.0 * torch.pi * frequencies
    times = times.view(-1, 1).float()
    embeddings = torch.cat(
        [torch.sin(times.matmul(angular_speeds) / T), torch.cos(times.matmul(angular_speeds) / T)], dim=1
    )
    return embeddings

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dconv_down1 = double_conv(in_channels + TIME_EMBEDDING_DIM, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up3 = double_conv(832, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, time_index):
        time_embedding = sinusoidal_embedding(time_index)
        x = torch.cat([x, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))], dim=1)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = torch.cat([x, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))], dim=1)

        x = self.upsample(x)

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out
    
@torch.no_grad()
def sample_single_ddpm(model, T=1000, img_size=(1, 48, 48), device='cuda'):
    """
    Generate a sample from a trained diffusion model using DDPM.

    Args:
        model: Trained UNet model for noise estimation.
        T: Number of diffusion steps.
        img_size: Shape of the output image (channels, height, width).
        device: 'cuda' or 'cpu'.

    Returns:
        Generated image as a PyTorch tensor.
    """
    model.eval()  # Set model to evaluation mode

    # Initialize pure Gaussian noise
    x_t = torch.randn((1, *img_size), device=device)

    # Define betas and compute alphas
    betas = torch.linspace(1e-4, 0.01, T, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # Cumulative product of alphas
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    for t in reversed(range(T)):  # Iterate backwards in time
        t_tensor = torch.tensor([t], device=device)  # Current timestep tensor
        predicted_noise = model(x_t, t_tensor)  # Predict noise using UNet

        # Compute mean of the posterior distribution
        mean = sqrt_recip_alphas[t] * (x_t - betas[t] * predicted_noise / torch.sqrt(1 - alphas_cumprod[t]))

        if t > 0:
            # Sample noise for stochasticity
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(betas[t])  # Standard deviation of noise
            x_t = mean + sigma_t * noise  # Add noise
        else:
            x_t = mean  # Final step: No noise added
            

    return x_t

@torch.no_grad()
def sample_n_ddpm(model, num_samples=1, T=1000, img_size=(1, 48, 48), device='cuda'):
    """
    Generate multiple samples from a trained diffusion model using DDPM.

    Args:
        model: Trained UNet model for noise estimation.
        num_samples: Number of images to generate.
        T: Number of diffusion steps.
        img_size: Shape of each output image (channels, height, width).
        device: 'cuda' or 'cpu'.

    Returns:
        Generated images as a PyTorch tensor of shape (num_samples, C, H, W).
    """
    model.eval()  # Set model to evaluation mode

    # Initialize pure Gaussian noise for a batch
    x_t = torch.randn((num_samples, *img_size), device=device)

    # Define betas and compute alphas
    betas = torch.linspace(1e-4, 0.01, T, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # Cumulative product of alphas
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    for t in tqdm(reversed(range(T))):  # Iterate backwards in time
        t_tensor = torch.full((num_samples,1), t, device=device)  # Batch of timestep indices
        predicted_noise = model(x_t, t_tensor)  # Predict noise using UNet

        # Compute mean of the posterior distribution
        mean = sqrt_recip_alphas[t] * (x_t - betas[t] * predicted_noise / torch.sqrt(1 - alphas_cumprod[t]))

        if t > 0:
            # Sample noise for stochasticity
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(betas[t])  # Standard deviation of noise
            x_t = mean + sigma_t * noise  # Add noise
        else:
            x_t = mean  # Final step: No noise added
            
    return x_t

def create_numpy_collage(images, grid_size=(10, 10), image_size=48):
    """
    Creates a 480x480 NumPy array collage of generated faces.

    Args:
        images: A NumPy array of shape (N, C, H, W), where N = grid_size[0] * grid_size[1].
        grid_size: Tuple (rows, cols) for the collage layout.
        image_size: Size of each image (assuming square images, default 48x48).

    Returns:
        A (480, 480) or (480, 480, 3) NumPy array representing the collage.
    """
    rows, cols = grid_size
    H, W = image_size, image_size
    num_images, C, _, _ = images.shape

    assert num_images == rows * cols, f"Number of images ({num_images}) must match grid size {rows}x{cols}"

    # Initialize blank canvas
    if C == 1:
        collage = np.zeros((rows * H, cols * W), dtype=np.float32)
    else:
        collage = np.zeros((rows * H, cols * W, C), dtype=np.float32)

    # Fill in the images
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        if C == 3:
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC format
        collage[row * H:(row + 1) * H, col * W:(col + 1) * W] = img

    return collage

if __name__ == '__main__':
    
    face_dataset = CustomDataset('./faces23k_48x48.npy', './ages23k.npy')
    data_loader = DataLoader(face_dataset, batch_size=64)

    device = 'cuda'
    
    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load('./diff_unet_faces.cpt'))

    betas = torch.linspace(1e-4, .01, T, dtype=torch.float32).to(device)

    # 1. Unconditional Generation

    # images = []
    # for i in tqdm(range(100)):
    #     image = sample_single_ddpm(model, T=T, img_size=(1, 48, 48), device=device).cpu()
    #     images.append(image)
        
    # images = torch.cat(images, dim=0)  # Shape: (100, 1, 48, 48)
    images = sample_n_ddpm(model, num_samples=100, T=T, img_size=(1, 48, 48), device=device)  # Shape: (100, 1, 48, 48)
    
    collage = create_numpy_collage(images.cpu().numpy(), grid_size=(10, 10), image_size=48)
    plt.imshow(collage, cmap='gray')
    plt.axis('off')
    plt.show()
    
    
    # 2. Inpainting/Merging
    # TODO

    # 3. Classifier Guidance
    # TODO
