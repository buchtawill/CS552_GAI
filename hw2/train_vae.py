import torch
import pickle
from VAE import *
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BATCH_SIZE = 32
LEARN_RATE = 0.001
KL_BETA = 0.95

def plot_single_face(face:np.ndarray):
    # face is a 576 element np vector. reshape it to 24x24 and plot it
    face_reshaped = face.reshape(24, 24)
    plt.imshow(face_reshaped, cmap='gray')
    plt.axis('off')
    plt.show()

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
        
def plot_faces_collage(faces: np.ndarray, num_pics:int=25, title:str='Generated Faces', save_path:str=None):
    '''
    Plot a collage of 24x24 faces as a grid. faces is in the shape n x 24 x 24, where n is the number of photos.
    num_pics is the number of faces to display. Default is 25 samples.
    This function arranges faces in a collage (large image) instead of individual subplots.
    '''
    
    # Determine the number of rows and columns for the collage grid
    num_rows = int(np.sqrt(num_pics))
    num_cols = int(np.ceil(num_pics / num_rows))
    
    # Create a blank canvas to hold the collage
    collage = np.ones((num_rows * 24, num_cols * 24))  # A grid of 24x24 faces
    
    for i in range(num_pics):
        if i < faces.shape[0]:
            row = i // num_cols  # Row in the collage
            col = i % num_cols   # Column in the collage
            face = faces[i]      # Get the face (24x24)
            collage[row * 24:(row + 1) * 24, col * 24:(col + 1) * 24] = face  # Place the face in the collage
    
    # Display the collage
    plt.figure(figsize=(6, 6))
    plt.imshow(collage, cmap='gray')
    plt.axis('off')
    # if title:
    #     plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    
def sample_new_faces(model, num_samples=25, device='cuda', epoch_number=0, save_path=None):
    """Generate and display new face samples from the trained VAE.
    
    Args:
        model: Trained VAE model.
        num_samples: Number of new face images to generate.
        device: Device to run the sampling on.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Sample random latent vectors from a standard normal distribution
        z = torch.randn(num_samples, model.latent_dim).to(device)  # model.latent_dim is the size of your latent space
        
        # Pass the latent vectors through the decoder
        generated_images = model.decoder_forward(z)  # Shape: (num_samples, 1, 24, 24)
        generated_images = torch.sigmoid(generated_images)  # Ensure values are in [0, 1]
        
        if(save_path):
            plot_faces_collage(generated_images.cpu().numpy(), num_pics=num_samples, title=f'Newly Generated Faces {epoch_number}', save_path=f'./epochs/{save_path}{epoch_number}.png')
        else:
            plot_faces_collage(generated_images.cpu().numpy(), num_pics=num_samples, title=None)
    
def train_vae(model, train_loader, test_loader, optimizer, num_epochs, device):
    model.train()
    
    losses = []
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        
        # Training phase
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            data = data.unsqueeze(1)
            
            # Forward pass
            recon_batch, mu, logvar = model(data)
            
            # Compute the VAE loss
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')  # Reconstruction loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())        # KL divergence
            
            loss = recon_loss + (kl_loss * KL_BETA)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        losses.append(avg_train_loss)
        print(f"Epoch {epoch:>4} Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase 
        if epoch % 5 == 0 or epoch == num_epochs:
            validate_vae(model, test_loader, device, epoch)
            
    return losses

def train_ae(model, train_loader, test_loader, optimizer, num_epochs, device):
    model.train()
    
    losses = []
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        
        # Training phase
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            data = data.unsqueeze(1)
            
            # Forward pass
            recon_batch = model(data)
            
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')  # Reconstruction loss
            
            loss = recon_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        losses.append(avg_train_loss)
        print(f"Epoch {epoch:>4} Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if epoch % 5 == 0 or epoch == num_epochs:
            # Plot example reconstructions from the test set
            validate_ae(model, test_loader, device, epoch)
            
    return losses

def validate_ae(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            data = data.unsqueeze(1)
            
            recon_batch = model(data)
            
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            test_loss += recon_loss
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}\n")
    
    recon_batch = recon_batch.squeeze(1).cpu().detach().numpy() # shape: (batch_size, 24, 24)
    plot_faces(recon_batch[:25], num_pics=25, title='Reconstructed Faces', save_path=f'./epochs/reconstructed_faces_ae{epoch}.png')
    

def validate_vae(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            data = data.unsqueeze(1)
            
            recon_batch, mu, logvar = model(data)
            
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            test_loss += recon_loss + kl_loss
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}\n")
    
    recon_batch = recon_batch.squeeze(1).cpu().detach().numpy() # shape: (batch_size, 24, 24)

    plot_faces(recon_batch[:25], num_pics=25, title='Reconstructed Faces', save_path=f'./epochs/reconstructed_faces_vae{epoch}.png')
    sample_new_faces(model, epoch_number=epoch)
    
def problem1_c(vae_model, ae_model, data):
    '''
    Get 1000 faces from the dataset and pass them through the encoder to get the latent space representation.
    Use PCA on the latent representation to reduce the dimensionality to 2D.
    '''
    
    train_tensor = torch.tensor(data).unsqueeze(1)
    
    vae_model.eval()
    with torch.no_grad():
        latent_vae_z = vae_model.encoder_forward(train_tensor.to(device)).cpu().detach().numpy()
        latent_ae_z  = ae_model.encoder_forward(train_tensor.to(device)).cpu().detach().numpy()
    
    # Perform PCA to reduce the dimensionality to 2D
    pca_vae = PCA(n_components=2)
    latent_z_2d_vae = pca_vae.fit_transform(latent_vae_z)
    
    pca_ae = PCA(n_components=2)
    latent_z_2d_ae = pca_ae.fit_transform(latent_ae_z)

    num_to_plot = 5000
    
    std_randoms = np.random.randn(num_to_plot, 2)
    # Plot the 2D latent space
    plt.figure(figsize=(8, 6))
    # plt.scatter(latent_z_2d_ae[0:num_to_plot, 0], latent_z_2d_ae[0:num_to_plot, 1], s=5, c='r', alpha=0.5, label='AE Latent Space')
    plt.scatter(latent_z_2d_vae[0:num_to_plot, 0], latent_z_2d_vae[0:num_to_plot, 1], s=5, c='b', alpha=0.5, label='VAE Latent Space')
    plt.scatter(std_randoms[0:num_to_plot, 0], std_randoms[0:num_to_plot, 1], s=5, c='g', alpha=0.5, label='Standard Normal Samples')
    plt.title('Latent Space of VAE vs AE')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def problem1_d(model):
    
    rand_z = torch.randn(model.latent_dim)       # 32
    rand_zprime = torch.randn(model.latent_dim)  # 32
    
    permuted_zs = torch.zeros(20, model.latent_dim)  # 20 x 32

    for i in range(model.latent_dim):
        permuted_zs[:, i] = torch.linspace(rand_z[i], rand_zprime[i], 20)
    
    reconstructed = model.decoder_forward(permuted_zs)
    
    # Plot the faces in a 4 row by 5 column collage
    plot_faces_collage(reconstructed.cpu().detach().numpy(), num_pics=20, title='Interpolated Faces')
    
def fit_ppca(face_data:np.ndarray, d_latent:int):
    '''
    Return the weight matrix of shape m features x d latent dim, covariance term, and mu
    '''
    
    # face data shape: 2000, 576 --> row vector
    n_samples = len(face_data)
    m_input_dim = len(face_data[0])
    W_weights = np.zeros((m_input_dim, d_latent))
    mu = np.mean(face_data, axis=0) # Shape: m_input_dim,
    
    # x(i): 1, 576
    # x(i).T: 576, 1
    # The assignment says (x-mu)(x-mu).T but it really should be (x-mu).T(x-mu)
    for i in range(n_samples):
        dif = face_data[i] - mu
        # dif shape = 1, 576
        if(i == 0):
            sum = np.outer(dif, dif.T)
        
        else:
            sum += np.outer(dif, dif.T)
    
    cov = sum / n_samples

    eigval, eigvec = np.linalg.eig(cov)
    eigval = eigval.real
    eigvec = eigvec.real
    
    sigma2 = 0.0
    for j in range(d_latent+1, m_input_dim):
        sigma2 += eigval[j]
        
    sigma2 = sigma2 / (m_input_dim - d_latent)
    
    #Find the largest eigenvalues
    largest_indices = np.flip(np.argsort(np.abs(eigval)))
    
    # From CS541 HW1!
    eigval = eigval[largest_indices]
    eigvec = eigvec[:, largest_indices]
    
    U_q = eigvec[:,0:d_latent] # shape: m x d
    lambda_q = np.diag(eigval[0:d_latent])
    identity = np.identity(d_latent)
    
    W_weights = U_q.dot(np.sqrt(lambda_q - sigma2*identity))
    
    return W_weights, sigma2, mu

def x_to_z(x, W, sigma2, mu)->np.ndarray:
    '''
    Given x, project it into the latent space
    '''
    
    d = len(W[0])
    M = np.dot(W.T, W) + sigma2*np.identity(d) # shape: d x d
    
    # print("\nM:")
    # print(M)
    # print()
    
    # M_inverse = np.linalg.inv(M) # shape: d x d
    M_inverse = np.linalg.solve(M, np.identity(d))
    # print("M inverse:")
    # print(M_inverse)
    # print()
    
    # Reparamaterization: multiply by sigma2, add mu to standard normal
    z = np.random.normal(loc=0, scale=1, size=d)
    z = np.diag(z)
    z *= sigma2 * (M_inverse)
    
    shift_amt = np.dot(M_inverse, W.T).dot(x-mu)
    
    diag = np.zeros(d)
    for i in range(d):
        diag[i] = z[i,i]    
    
    return diag + shift_amt

def z_to_x(z:np.ndarray, W:np.ndarray, sigma2, mu)->np.ndarray:
    
    
    # W:  m x d
    # z: d x 1
    d = len(z)
    m = len(W)
    
    # Reparamaterization: multiply by sigma2, add mu to standard normal
    normie = np.random.normal(loc=0, scale=1, size=m)
    
    normie = np.diag(normie)
    
    normie *= sigma2*np.identity(len(W))
    
    x_hat = np.zeros(m)
    for i in range(m):
        x_hat[i] = normie[i, i]
    mu_shift = np.dot(W, z) + mu
    
    return x_hat + mu_shift

def problem_1e(vae_model, data, W, sigma2, mu):
    
    # Perform PCA on the entire dataset
    # data shape is (23042, 24, 24)
    # Flatten the data to shape (num_samples, 576)
    flattened_data = data.reshape(data.shape[0], -1)
    
    # Perform PCA to reduce the dimensionality to 2D
    pca = PCA(n_components=2)
    train_set_pca = pca.fit_transform(flattened_data)
    
    # Generate 20 random faces, project them down to 2D
    num_to_plot = 20
    with torch.no_grad():
        # Sample random latent vectors from a standard normal distribution
        z = torch.randn(num_to_plot, vae_model.latent_dim).to(device)  # vae_model.latent_dim is the size of your latent space
        
        # Pass the latent vectors through the decoder
        generated_images = vae_model.decoder_forward(z)  # Shape: (num_to_plot, 1, 24, 24)
        generated_images = torch.sigmoid(generated_images)  # Ensure values are in [0, 1]
        
        generated_2d = pca.transform(generated_images.cpu().numpy().reshape(num_to_plot, -1))
        
        # Pass the same latent vectors through PPCA
        z = z.cpu().numpy()
        ppca_x = np.zeros((num_to_plot, 576))        
        for i in range(num_to_plot):
            ppca_x[i] = z_to_x(z[i], W, sigma2, mu)
        
        ppca_2d = pca.transform(ppca_x)
    
    # Plot the 2D representation of faces
    plt.figure(figsize=(6, 6))
    for i in range(num_to_plot):
        plt.scatter(generated_2d[i, 0], generated_2d[i, 1], s=5, c='b', alpha=0.5, label='VAE P(x|z) 2D' if i == 0 else "")
        plt.text(generated_2d[i, 0], generated_2d[i, 1], str(i+1), fontsize=9, color='blue')
        plt.scatter(ppca_2d[i, 0], ppca_2d[i, 1], s=5, c='r', alpha=0.5, label='PPCA P(x|z) 2D' if i == 0 else "")
        plt.text(ppca_2d[i, 0], ppca_2d[i, 1], str(i+1), fontsize=9, color='red')
    plt.title('P(x|z) of VAE vs PPCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    # Load and normalize faces to [0, 1]
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    print(f'INFO [train_vae.py] Device: {device}')
    
    full_dataset = FaceDataset('./faces_vae.npy')
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1])
    num_train_samples = len(train_dataset)
    print(f'INFO [train_vae.py] Total num data samples:    {len(full_dataset)}')
    print(f'INFO [train_vae.py] Num of training samples:   {num_train_samples}')
    print(f'INFO [train_vae.py] Num of test samples:       {len(test_dataset)}')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True)
    
    vae_model = VAE(latent_dim=32).to(device)
    vae_model.load_state_dict(torch.load('./vae_model_weights.pth'))
    # ae_model = AE(latent_dim=32).to(device)
    # ae_model.load_state_dict(torch.load('./ae_model_weights.pth'))
    
    # optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    
    # losses=train_vae(model=model, train_loader=train_dataloader, test_loader=test_dataloader, optimizer=optimizer, num_epochs=50, device=device)
    # torch.save(model.state_dict(), './vae_model_weights.pth')
    # losses = train_ae(model=model, train_loader=train_dataloader, test_loader=test_dataloader, optimizer=optimizer, num_epochs=50, device=device)
    # torch.save(model.state_dict(), './ae_model_weights.pth')
    
    # problem1_c(vae_model, ae_model, full_dataset.data)
    # problem1_d(vae_model.to('cpu'))
    
    ############################ Problem 1e ############################
    # Train PPCA with the same latent dimension as VAE. Train once and save because it takes a while
    # flattened_data = full_dataset.data.reshape(full_dataset.data.shape[0], -1)
    # W, sigma2, mu = fit_ppca(flattened_data, vae_model.latent_dim)
    # with open('ppca_weights.pkl', 'wb') as f:
    #     pickle.dump((W, sigma2, mu), f)
    
    # Load the weights
    with open('ppca_weights.pkl', 'rb') as f:
        W, sigma2, mu = pickle.load(f)
    
    problem_1e(vae_model.to('cpu'), full_dataset.data, W, sigma2, mu)
