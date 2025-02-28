import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define a simple 1D GAN class
class GAN_1d:
    def __init__(self, latent_dim=1, hidden_dim=2048, lr=0.001, device='cpu'):
        self.device = device
        self.latent_dim = latent_dim

        # Generator: maps a latent noise vector to a single scalar output
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

        # Discriminator: takes a scalar input and outputs a probability (real vs. fake)
        self.discriminator = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(device)

        self.optim_G = optim.Adam(self.generator.parameters(), lr=lr)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    def train_step(self, real_data):
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        # Create labels for real and fake data
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optim_D.zero_grad()

        # Loss for real data
        real_loss = self.criterion(self.discriminator(real_data), valid)

        # Generate fake data
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        gen_data = self.generator(z)
        fake_loss = self.criterion(self.discriminator(gen_data.detach()), fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optim_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        self.optim_G.zero_grad()
        # Generator tries to fool discriminator: aim for discriminator(gen_data) ≈ valid
        g_loss = self.criterion(self.discriminator(gen_data), valid)
        g_loss.backward()
        self.optim_G.step()

        return g_loss.item(), d_loss.item()

    def sample(self, n):
        """Generate n samples from the generator."""
        z = torch.randn(n, self.latent_dim, device=self.device)
        with torch.no_grad():
            samples = self.generator(z)
        return samples.cpu().numpy().flatten()


# Function to sample "real" data: y = sin(x) with x ∼ Uniform(-π, π)
def sample_real_data(n):
    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.sin(x)
    return torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape: (n, 1)

# Function to visualize normalized histograms of real and generated data
def visualize_distribution(real_samples, fake_samples, discriminator, epoch=1, bins=50):
    """
    Visualizes the distributions of real and fake data using normalized histograms.
    
    Parameters:
        real_samples (np.array): 1D array of real data values.
        fake_samples (np.array): 1D array of generated data values.
        bins (int): Number of bins for the histograms.
    """
    
    # Create a range of values to evaluate the discriminator
    xlo, xhi = -1.5, 1.5
    x_vals = np.linspace(xlo, xhi, 500)
    x_tensor = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1)

    # Ensure the discriminator is in evaluation mode and on the correct device
    discriminator.eval()
    device = next(discriminator.parameters()).device
    x_tensor = x_tensor.to(device)

    with torch.no_grad():
        d_output = discriminator(x_tensor).cpu().numpy().flatten()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot normalized histograms for real and fake data
    ax1.hist(real_samples, bins=bins,
            #  weights=np.ones_like(real_samples) / len(real_samples),
             alpha=0.5, label='Pdata(x)', color='cyan')
    ax1.hist(fake_samples, bins=bins,
            #  weights=np.ones_like(fake_samples) / len(fake_samples),
             alpha=0.5, label='P(G(z))', color='red')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Histogram of Real vs. Generated Data (Epcoch {epoch})')
    ax1.legend(loc='upper left')
    ax1.set_xlim(xlo, xhi)

    # If d_output is provided, add a second y-axis and plot it
    if d_output is not None:
        ax2 = ax1.twinx()
        # Create an x-axis for d_output assuming its values span [-3, 3]
        x_vals = np.linspace(xlo, xhi, len(d_output))
        ax2.plot(x_vals, d_output, color='orange', label='d_output', linewidth=2)
        ax2.set_ylabel('d_output')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')

    plt.savefig(f'./gan_training/epoch_{epoch}.png')
    plt.close()

if __name__ == '__main__':
    # Set device and hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 1
    num_epochs = 5000
    batch_size = 128

    # Instantiate the 1D GAN
    gan = GAN_1d(latent_dim=latent_dim, hidden_dim=2048, lr=0.0005, device=device)

    # For tracking losses
    g_losses = []
    d_losses = []

    # Training loop
    for epoch in range(num_epochs):
        real_data = sample_real_data(batch_size)
        g_loss, d_loss = gan.train_step(real_data)
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:>5}: Generator Loss = {g_loss:6.4f}, Discriminator Loss = {d_loss:6.4f}")

            # After training, generate samples from the generator
            fake_samples = gan.sample(2000)
            real_samples = sample_real_data(2000).numpy().flatten()

            # Visualize the distributions using the new function
            visualize_distribution(real_samples, fake_samples, discriminator=gan.discriminator, epoch=epoch+1, bins=100)
    