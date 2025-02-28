import time

start_time = time.time() # Get this time before importing torch

import torch
import pickle
from GAN import *
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import torchinfo

BATCH_SIZE = 32
NUM_EPOCHS = 100

def plot_faces(faces: np.ndarray, num_pics:int=25, title:str='Generated Faces', save_path:str=None):
    '''
    Plot 576 column numpy arrays as 24x24 squares. faces is in the shape n x 24 x 24, where n is the number of photos.
    num_pics is the number of faces to display. Default is 25 samples
    '''

    # ChatGPT vvvvv
    # Determine the number of rows and columns for the plot
    num_rows = int(np.sqrt(num_pics))
    num_cols = int(np.ceil(num_pics / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5, 5))
    axes = axes.flatten()

    for i in range(num_pics):
        if i < faces.shape[0]:
            face = faces[i].reshape(24, 24)
            axes[i].imshow(face, cmap='gray')
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    # if(title):
    #     plt.title(title)
    
    if(save_path):
        plt.savefig(save_path)
    else:
        plt.show()

def print_model_details(model, model_name="Model"):
    print(f"--- {model_name} Summary ---")
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:20s}: shape={param.shape}, count={param_count}")
    print(f"Total parameters in {model_name}: {total_params}\n")

def print_model_summary(model, batch_size, in_channels, height, width):
    torchinfo.summary(model, input_size=(batch_size, in_channels, height, width))

def train_loop(gan, device):
    
    # Plot initial inference (should just be noise)
    # Use the same z's throughout to get a 1:1 comparison
    z = torch.randn(25, gan.latent_dim, device=device)
    gan.generator.eval()
    with torch.no_grad():
        gen_faces = gan.generator(z).cpu().detach().numpy()  # shape: (25, 1, 24, 24)
    # Squeeze the channel dimension for plotting
    gen_faces = np.squeeze(gen_faces, axis=1)
    plot_faces(gen_faces, num_pics=25, title='Generated Faces', save_path=f'./epochs/before_training.png')
    gan.generator.train()
    
    # Lists to record losses
    train_g_loss_history = []
    train_d_loss_history = []
    val_g_loss_history = []
    val_d_loss_history = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_g_loss = 0
        epoch_d_loss = 0
        for real_imgs in train_dataloader:
            real_imgs = real_imgs.to(device)
            g_loss, d_loss = gan.train_step(real_imgs)
            epoch_g_loss += g_loss
            epoch_d_loss += d_loss

        epoch_g_loss /= len(train_dataloader)
        epoch_d_loss /= len(train_dataloader)

        # Run validation on test set
        val_g_loss, val_d_loss = gan.validate(test_dataloader)
        train_g_loss_history.append(epoch_g_loss)
        train_d_loss_history.append(epoch_d_loss)
        val_g_loss_history.append(val_g_loss)
        val_d_loss_history.append(val_d_loss)

        print(f"Epoch [{epoch+1:>3}/{NUM_EPOCHS}] Train G Loss: {epoch_g_loss:8.4f} D Loss: {epoch_d_loss:8.4f} | Val G Loss: {val_g_loss:8.4f} D Loss: {val_d_loss:8.4f}")

        # Every 10 epochs, generate and plot sample faces
        if (epoch+1) % 10 == 0:
            gan.generator.eval()
            with torch.no_grad():
                # Use same z's from before
                gen_faces = gan.generator(z).cpu().detach().numpy()  # shape: (25, 1, 24, 24)
            # Squeeze the channel dimension for plotting
            gen_faces = np.squeeze(gen_faces, axis=1)
            plot_faces(gen_faces, num_pics=25, title=f'Generated Faces After Epoch {epoch+1}', save_path=f'./epochs/vae_epoch_{epoch+1}.png')
            gan.generator.train()
            
    return train_g_loss_history, train_d_loss_history, val_g_loss_history, val_d_loss_history

def sec_to_human(seconds):
    """Return a number of seconds to hours, minutes, and seconds"""
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hours, minutes, seconds)


if __name__ == '__main__':
    # Load and normalize faces to [0, 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'INFO [train_gan.py] Device: {device}')
    
    full_dataset = FaceDataset('../hw2/faces_vae.npy')
    
    train_dataset, test_dataset = random_split(full_dataset, [0.9, 0.1])
    num_train_samples = len(train_dataset)
    print(f'INFO [train_gan.py] Total num data samples:    {len(full_dataset)}')
    print(f'INFO [train_gan.py] Num of training samples:   {num_train_samples}')
    print(f'INFO [train_gan.py] Num of test samples:       {len(test_dataset)}')
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True)
    
    # Plot example faces
    # recon_batch = next(iter(train_dataloader))
    # recon_batch = recon_batch.squeeze(1).cpu().detach().numpy() # shape: (batch_size, 24, 24)
    # plot_faces(recon_batch[:25], num_pics=25)
    
    # Instantiate GAN
    gan = GAN(latent_dim=32, device=device, g_mode='vae')
    gan.generator.load_state_dict(torch.load('../hw2/vae_model_weights.pth'))
    # gan = GAN(latent_dim=32, device=device, g_mode='normal')
    
    # print_model_details(gan.generator, "Generator")
    # print_model_details(gan.discriminator, "Discriminator")

    train_g_loss_history, train_d_loss_history, val_g_loss, val_d_loss = train_loop(gan, device)
    
    torch.save(gan.generator.state_dict(), "generator_weights_z32_gan.pth")
    # torch.save(gan.discriminator.state_dict(), "discriminator_weights.pth")

    # train_g_loss = np.array(train_g_loss_history)
    # train_d_loss = np.array(train_d_loss_history)
    # val_g_loss   = np.array(val_g_loss)
    # val_d_loss   = np.array(val_d_loss)
    
    # x = np.arange(NUM_EPOCHS+1)[1:]
    # plt.plot(x, train_g_loss, label='G Train Loss')
    # plt.plot(x, train_d_loss, label='D Train Loss')
    # plt.plot(x, val_g_loss, label='G Validation Loss')
    # plt.plot(x, val_d_loss, label='D Validation Loss')
    
    # plt.show()

    end_time = time.time()
    print(f"INFO [train_gan.py] Script took {sec_to_human(end_time-start_time)} to complete")
    