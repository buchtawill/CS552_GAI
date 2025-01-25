import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_sample(data: list)->np.ndarray:
    """
    Given a CPT table 'data', get a sample from the table 
    
    Args:
        data (list): a list of CPT tables
    
    Returns:
        np.ndarray: a sample from the table in the shape (5,5)
    """
    indices = []
    colors = []
    for i in range(25):
        
        # Recurse down the list / array
        tmp = data[i]
        for index in indices:
            tmp = tmp[index]  # <-- chat GPT's idea
            
        r = np.random.rand()
        if (r < tmp[0]):
            indices.append(0)
            colors.append(1)
        else:
            indices.append(1)
            colors.append(0)
    
    sample = np.array(colors)
    return sample.reshape(5,5)

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
        images.append(get_sample(data))
    
    plot_100_images(images)

if __name__ == '__main__':
    
    # data is a list
    data = pickle.load(open('./cpts.pkl', 'rb'))
    
    # prob_1a(data)
    
    