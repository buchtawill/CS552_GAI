import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# Define the VAE model for 1D input data (each sample is a scalar)
class VAE1D(nn.Module):
    def __init__(self, input_shape=(1,), hidden_dim=8, latent_dim=1):
        super(VAE1D, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_shape[0]  # For 1D data, input_dim = 1
        
        # Encoder layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.input_dim)
        
    def encode(self, x):
        # x is of shape (batch_size, 1)
        # x = x.view(-1, 1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        x_recon = torch.sigmoid(self.fc3(h))
        # Reshape back to (batch_size, 1)
        x_recon = x_recon.view(-1, 1)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def compute_true_log_likelihood(model, x, z_min=-10, z_max=10, num_points=100):
    """
    Method written by chat gpt
    Numerically approximates the true log likelihood:
      log p(x) = log ∫ p(x|z) p(z) dz

    Args:
        model: The trained VAE model.
        x: A tensor of shape (batch_size, input_dim). For our 1D case, e.g., shape (1, 1).
        z_min: Lower bound for z integration.
        z_max: Upper bound for z integration.
        num_points: Number of grid points to use for integration.
    
    Returns:
        The approximated log likelihood (a float).
    """
    model.eval()
    # Create a grid for z (latent_dim = 1)
    z_grid = torch.linspace(z_min, z_max, num_points).unsqueeze(1)  # shape: (num_points, 1)
    dz = (z_max - z_min) / (num_points - 1)
    
    with torch.no_grad():
        # Compute the prior p(z) for each point (standard normal)
        p_z = torch.exp(-0.5 * z_grid**2) / torch.sqrt(torch.tensor(2 * np.pi))
        p_z = p_z.squeeze()  # shape: (num_points,)
        
        # For each z, get the decoder's output probability p(x|z)
        # model.decode expects input of shape (batch_size, latent_dim)
        x_recon = model.decode(z_grid)  # shape: (num_points, 1)
        
        # Compute the Bernoulli likelihood:
        # For each element: p(x|z) = p^x (1-p)^(1-x)
        # Here we work in the log domain for numerical stability.
        eps = 1e-12
        # Expand x so it can be compared with each decoded output
        # x should be of shape (1, input_dim); we expand to (num_points, input_dim)
        x_expanded = x.expand(num_points, -1)
        
        log_px_given_z = x_expanded * torch.log(x_recon + eps) + \
                         (1 - x_expanded) * torch.log(1 - x_recon + eps)
        # Sum over the output dimension (here it's 1, so this is optional but kept for generality)
        log_px_given_z = log_px_given_z.sum(dim=1)  # shape: (num_points,)
        
        # Convert back from log domain
        p_x_given_z = torch.exp(log_px_given_z)  # shape: (num_points,)
        
        # Compute the integrand: p(x|z) * p(z)
        integrand = p_x_given_z * p_z
        
        # Numerically integrate using the trapezoidal rule
        integral = torch.trapz(integrand, dx=dz)
        
        # Take the log to obtain log p(x)
        true_log_likelihood = torch.log(integral + eps)
    
    return true_log_likelihood.item()

def compute_ELBO(model, x):
    """
    Computes the ELBO for a given input x.
    
    Since our loss function returns BCE + KLD (i.e., -ELBO), we return:
         ELBO = - (BCE + KLD)
    
    Args:
        model: The VAE model.
        x: Input tensor of shape (batch_size, input_dim) (e.g., (1, 1) for 1D data).
    
    Returns:
        The ELBO (a float).
    """
    model.eval()
    with torch.no_grad():
        recon_x, mu, logvar = model(x)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        elbo = - (BCE + KLD)
    return elbo.item()

# Loss function: reconstruction loss + KL divergence loss
# AKA Elbo 
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(model, X, num_epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):    
        model.train()
        optimizer.zero_grad()
        recon_x, mu, logvar = model(X)
        loss = loss_function(recon_x, X, mu, logvar)
        loss.backward()
        optimizer.step()
        if(epoch + 1 % 10 == 0):
            print(f'Epoch {epoch+1:>3}/{num_epochs}, Loss: {loss.item():.4f}')

def make_ll_plot(model, sample_to_test):
    # Compute the true log likelihood once for several different tweaks of theta (decoder)
    
    n_thetas = 200
    theta_diffs = np.linspace(-10.0, 10.0, n_thetas)
    
    true_ll = np.zeros(n_thetas)
    original_theta = model.fc3.weight.data[0, 0].item()
    
    for i, theta_diff in enumerate(theta_diffs):
        model.fc3.weight.data[0, 0] = original_theta + theta_diff
        true_ll[i] = compute_true_log_likelihood(model, sample_to_test)
    
    # Now tweak different values in the encoder and see how the true log likelihood changes
    n_phis = 5
    phi_diffs = np.linspace(-0.5, 0.0, n_phis)
    original_phi = model.fc_logvar.weight.data[0, 0].item()
    for i, phi_diff in enumerate(phi_diffs):
        model.fc_logvar.weight.data[0, 0] = original_phi + phi_diff
        
        elbos_phi = np.zeros(n_thetas)
        for j, theta_diff in enumerate(theta_diffs):
            model.fc3.weight.data[0, 0] = original_theta + theta_diff
            n_elbo_avg = 200
            elbo = 0.0
            for _ in range(n_elbo_avg):
                elbo += compute_ELBO(model, sample_to_test)
            elbo /= n_elbo_avg
            elbos_phi[j] = elbo
        
        plt.plot(theta_diffs, elbos_phi, label=f'Phi diff: {phi_diff:.2f}')
        
    plt.plot(theta_diffs, true_ll, label='True Log Likelihood', color='black')
    plt.xlabel(r'$\theta$ diff')
    plt.ylabel('True Log Likelihood')
    plt.title('True Log Likelihood vs Theta diff')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    # Example: generate 1-D data biased towards 2.0 from a normal distribution.
    num_samples = 200
    data = np.random.normal(loc=0.8, scale=0.05, size=(num_samples, 1))  # 1-D samples
    data_tensor = torch.tensor(data, dtype=torch.float32)
    data_tensor = torch.clamp(data_tensor, 0, 1)
    
    model = VAE1D()
    
    train_vae(model, data_tensor)
    
    sample_to_test = data_tensor[0].unsqueeze(0)
    
    make_ll_plot(model, sample_to_test)