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
def sample_single_ddpm(model, betas, T=1000, img_size=(1, 48, 48), device='cuda', tstart=T, zt=None):
    """
    Generate a sample from a trained diffusion model using DDPM.

    Args:
        model: Trained UNet model for noise estimation.
        betas: Precomputed tensor of beta values.
        T: Number of diffusion steps.
        img_size: Shape of the output image (channels, height, width).
        device: 'cuda' or 'cpu'.
        tstart: Starting timestep for diffusion process.
        zt: Initial image tensor to start the diffusion. Shape: (1, C, H, W).

    Returns:
        Generated image as a PyTorch tensor.
    """
    model.eval()  # Set model to evaluation mode

    if zt is not None:
        x_t = zt
    else:
        # Initialize pure Gaussian noise
        x_t = torch.randn((1, *img_size), device=device)

    # Define betas and compute alphas
    # betas = torch.linspace(1e-4, 0.01, T, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # Cumulative product of alphas
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    for t in reversed(range(tstart)):  # Iterate backwards in time
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

def apply_diffusion_kernel(x0, t, betas, device='cuda'):
    """
    Applies the diffusion kernel to an image at timestep t.

    Args:
        x0 (torch.Tensor): The original image of shape (C, H, W).
        t (int): The diffusion timestep.
        betas (torch.Tensor): A precomputed tensor of beta values.
        device (str): The device to use ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The noised image x_t.
    """
    # Compute cumulative product of alphas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Compute alpha_bar_t
    alpha_bar_t = alphas_cumprod[t]

    # Sample Gaussian noise
    noise = torch.randn_like(x0, device=device)

    # Apply the diffusion process
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    return xt

def problem1(model, device):
    images = sample_n_ddpm(model, num_samples=100, T=T, img_size=(1, 48, 48), device=device)  # Shape: (100, 1, 48, 48)
    
    collage = create_numpy_collage(images.cpu().numpy(), grid_size=(10, 10), image_size=48)
    plt.imshow(collage, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.close()
    
    
def problem2(faces_np, model, device, betas):
    
    face1s = []
    face2s = []
    merged_faces = []
    zts = []
    
    # Get 10 random faces (merge accordingly)
    rand_indices = np.random.choice(faces_np.shape[0], 10, replace=False)
    time_steps = [200, 500, 750, 999]
    for t in tqdm(time_steps, disable=False):
        for sample in tqdm(range(5), disable=False, leave=False):
            face1 = torch.tensor(faces_np[rand_indices[sample*2]]).unsqueeze(0).to(device)
            face2 = torch.tensor(faces_np[rand_indices[sample*2+1]]).unsqueeze(0).to(device)
            
            # Add noise to the faces
            zt1 = apply_diffusion_kernel(face1, t, betas, device)
            zt2 = apply_diffusion_kernel(face2, t, betas, device)
            
            zt = torch.zeros_like(zt1)
            zt[:, :, :24] = zt1[:, :, :24]
            zt[:, :, 24:] = zt2[:, :, 24:]
            
            # Diffuse Zt t times to get the final image, add a batch dimension
            # merged = sample_single_ddpm(model, T=T, img_size=(1, 48, 48), device=device, tstart=t, zt=zt.unsqueeze(0))
            merged = sample_single_ddpm(model, betas=betas, T=T, img_size=(1, 48, 48), device=device, tstart=t, zt=zt.unsqueeze(0))
            merged_faces.append(merged)
            face1s.append(face1)
            face2s.append(face2)
            zts.append(zt)
            
    # Plot in this order:
    # face1s[0], face2s[0], merged_faces[0], face1s[1], face2s[1], merged_faces[1], ..., faces1s[4], face2s[4], merged_faces[4]
    # face1s[5], face2s[5], merged_faces[5], face1s[6], face2s[6], merged_faces[6], ..., faces1s[9], face2s[9], merged_faces[9]
    # face1s[10], face2s[10], merged_faces[10], face1s[11], face2s[11], merged_faces[11], ..., faces1s[14], face2s[14], merged_faces[14]
    
    fig, axes = plt.subplots(4, 15, figsize=(10, 4))
    for row, t in enumerate(time_steps):
        start_idx = row * 5

        # Add text to the left of each row, making sure it appears in the figure
        axes[row, 0].annotate(f"t = {t}", xy=(-0.5, 0.5), xycoords='axes fraction',
                            ha='right', va='center', fontsize=12, fontweight='bold')

        for col in range(5):
            idx = start_idx + col
            axes[row, col * 3].imshow(face1s[idx].squeeze().cpu().numpy(), cmap='gray')
            axes[row, col * 3].axis('off')

            axes[row, col * 3 + 1].imshow(face2s[idx].squeeze().cpu().numpy(), cmap='gray')
            # axes[row, col * 3 + 1].imshow(zts[idx].squeeze().cpu().numpy(), cmap='gray')
            axes[row, col * 3 + 1].axis('off')

            axes[row, col * 3 + 2].imshow(merged_faces[idx].squeeze().cpu().numpy(), cmap='gray')
            axes[row, col * 3 + 2].axis('off')

    plt.tight_layout()
    plt.show()
    

def plot_5_ddpm_samples(model, T=1000, img_size=(1, 48, 48), device='cuda'):
    """
    Plot 5 samples generated from a trained diffusion model using DDPM.

    Args:
        model: Trained UNet model for noise estimation.
        T: Number of diffusion steps.
        img_size: Shape of each output image (channels, height, width).
        device: 'cuda' or 'cpu'.
    """

    images = []
    for i in range(5):
        image = sample_single_ddpm(model, T=T, img_size=img_size, device=device)
        images.append(image)

    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for ax, img in zip(axes, images):
        ax.imshow(img.squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.show()
    plt.close()

if __name__ == '__main__':
    
    face_dataset = CustomDataset('./faces23k_48x48.npy', './ages23k.npy')
    data_loader = DataLoader(face_dataset, batch_size=64)

    device = 'cuda'
    
    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load('./diff_unet_faces.cpt'))


    # 1. Unconditional Generation
    # betas = torch.linspace(1e-4, .01, T, dtype=torch.float32).to(device)
    # problem1(model, device)
    # exit()
    # Debug: Plot 5 samples
    # plot_5_ddpm_samples(model, T=1000, img_size=(1, 48, 48), device='cuda')
    # exit()
    
    # 2. Inpainting/Merging
    betas = torch.linspace(1e-4, .01, T, dtype=torch.float32).to(device)
    problem2(np.load('./faces23k_48x48.npy'), model, device, betas)
    
    # 3. Classifier Guidance
    # TODO
