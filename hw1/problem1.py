import torch
import pickle
from LSTM import *
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

def get_cpt_sample(data: list, flat:bool=False)->np.ndarray:
    """
    Given a CPT table 'data', get a sample from the table 
    
    Args:
        data (list): a list of CPT tables
        flat (bool): if True, return a flat array of 1s and 0s. If False, return a 5x5 image
    
    Returns:
        np.ndarray: a sample from the table in the shape (5,5)
    """
    indices = []
    colors = []
    for i in range(25):

        # Weird numpy indexing syntax
        tmp = data[i][(tuple(indices))]
            
        r = np.random.rand()
        if (r < tmp[0]):
            indices.append(0)
            colors.append(1)
        else:
            indices.append(1)
            colors.append(0)
    
    sample = np.array(colors)
    
    # For getting images
    if not flat:
        return sample.reshape(5,5)

    else:  
        return sample

def plot_100_images(images):
    # Plot the images
    # Parameters
    image_size = 5  # Size of each image (5x5 pixels)
    grid_size = 10  # 10x10 grid
    spacing = 1     # Space between images (in pixels)

    # Calculate the size of the canvas with spacing included
    collage_size = grid_size * image_size + (grid_size - 1) * spacing
    collage = np.ones((collage_size, collage_size))  # Initialize to white

    # Populate the collage with images and spacing
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        start_row = row * (image_size + spacing)
        start_col = col * (image_size + spacing)
        collage[start_row:start_row+image_size, start_col:start_col+image_size] = img

    # Display the collage
    plt.figure(figsize=(8, 8))
    plt.imshow(collage, cmap='gray')  # Use cmap='gray' for grayscale images
    plt.axis('off')  # Turn off axes
    plt.show()
    
def prob_1a(data):
    # Get 100 samples
    images = []
    for i in range(100):
        images.append(get_cpt_sample(data))
    
    plot_100_images(images)
    
def create_tensor_dataset(data):
    
    # Get 200 samples to make sure we get every character
    X = np.zeros((200, 25))
    for i in range(200):
        X[i] = get_cpt_sample(data, flat=True)
    
    # Extract unique letters
    unique = np.unique(X, axis=0)
    
    print(unique.shape)
    return torch.tensor(unique)

def prob1b():
    tensors = torch.load('./cpt_tensors.pt')
    
    dataset = TensorDataset(tensors)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize the model
    model = PixelLSTM(input_dim=1, hidden_dim=32, num_layers=2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    sequence_length = 25
    batch_size = 1

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        for batch in dataloader:
            targets = batch
            
            optimizer.zero_grad()
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size=batch_size)
            loss = 0.0  # Track loss for the sequence
            
            # Loop through the sequence timestep by timestep
            for t in range(sequence_length):
                print(f"t = {t}")
                # Input at timestep t (shape: [batch_size, 1, input_dim])
                if t == 0:
                    input_t = torch.zeros((batch_size, 1, 1))  # First input is 0.0
                else:
                    input_t = targets[:, t-1].unsqueeze(-1).unsqueeze(-1)  # Input is xi-1
                
                # Target at timestep t
                target_t = targets[:, t].unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, 1]
                
                # Forward pass for one timestep
                output_t, hidden = model(input_t, hidden)  # Output and updated hidden state
                
                # Compute the loss for the timestep
                loss += criterion(output_t.squeeze(), target_t.squeeze())  # Sum loss over timesteps
            
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

if __name__ == '__main__':
    
    # data is a list
    data = pickle.load(open('./cpts.pkl', 'rb'))
    
    # prob_1a(data)
    
    # Problem 1b
    # tensors = create_tensor_dataset(data)
    # torch.save(tensors, './cpt_tensors.pt')
    prob1b()
    