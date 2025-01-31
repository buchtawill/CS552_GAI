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
    
# def plot_1_image()
    
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
    return torch.tensor(unique, dtype=torch.float32)

def train_lstm():
    tensors = torch.load('./cpt_tensors.pt')
    
    # Initialize the model
    model = PixelLSTM(input_dim=1, hidden_dim=32, num_layers=2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Ensure tensors have the correct shape before padding
    tensors = tensors.unsqueeze(-1)  # Shape becomes (26, 25, 1)

    # Pad with a column of zeros at the beginning
    padded_tensors = torch.cat([torch.zeros((26, 1, 1)), tensors], dim=1)  # Shape: (26, 26, 1)

    # Create X (first 25 values) and Y (shifted 25 values)
    x = padded_tensors[:, :-1, :]  # Shape: (26, 25, 1) YES
    y = padded_tensors[:, 1:, :]   # Shape: (26, 25, 1) YES

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    num_epochs = 400
    epochs = [(i + 1) for i in range(num_epochs)]
    losses = np.zeros(num_epochs)
    # Training loop
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            batch_size = batch_x.size(0)
            hidden = model.init_hidden(batch_size)

            output, _ = model(batch_x, hidden)  # Output shape: (batch, 25, 1)
            
            loss = criterion(output, batch_y.float())  # Ensure matching shape

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1:>4}/{num_epochs}, Loss: {loss.item():.6f}")
        losses[epoch] = loss.item()
    plt.plot(epochs, losses)
    plt.show()
    
    torch.save(model.state_dict(), './LSTM_weights.pth')

def inference_lstm(model, start_value=0.0, flat:bool=False):
    
    seq_length = 25
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        generated_seq = [start_value]  # Start with the initial value
        hidden = model.init_hidden(batch_size=1)  # Initialize hidden state

        input_t = torch.tensor([[start_value]], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1, 1)

        for _ in range(seq_length):
            output, hidden = model(input_t, hidden)  # Forward pass

            prob = torch.sigmoid(output).item()  # Convert logits to probability

            next_value = 1 if prob >= 0.5 else 0  # Threshold at 0.5
            generated_seq.append(next_value)

            input_t = torch.tensor([[next_value]], dtype=torch.float32).unsqueeze(0)  # Update input

    ret = np.array(generated_seq[1:])
    if(flat):
        return ret
    
    else:
        return ret.reshape(5,5)

if __name__ == '__main__':
    
    # data is a list
    data = pickle.load(open('./cpts.pkl', 'rb'))
    
    # prob_1a(data)
    
    # Problem 1b
    # tensors = create_tensor_dataset(data)
    # torch.save(tensors, './cpt_tensors.pt')
    
    # train_lstm()
    
    # Problem 1b
    # model = PixelLSTM(input_dim=1, hidden_dim=32, num_layers=2)
    # model.load_state_dict(torch.load('./LSTM_weights.pth'))
    
    # test = inference_lstm(model)
    
    # images = []
    # for _ in range(100):
    #     images.append(inference_lstm(model, start_value=np.random.rand()))
        
    # plot_100_images(images)
    
    # # Print the number of parameters in the model
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters in the model: {total_params}")