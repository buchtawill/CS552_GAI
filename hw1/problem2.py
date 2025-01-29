import copy
import numpy as np
import matplotlib.pyplot as plt


def plot_faces(faces: np.ndarray, num_pics:int=25):
    '''
    Plot 576 column numpy arrays as 24x24 squares. faces is in the shape n x 576, where n is the number of photos.
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
    plt.show()
    
def plot_single_face(face:np.ndarray):
    # face is a 576 element np vector. reshape it to 24x24 and plot it
    face_reshaped = face.reshape(24, 24)
    plt.imshow(face_reshaped, cmap='gray')
    plt.axis('off')
    plt.show()
    
def fit_ppca(face_data:np.ndarray, d_latent:int):
    '''
    Return the weight matrix of shape m features x d latent dim, covariance term, and mu
    '''
    
    # face data shape: 2000, 576 --> row vector
    n_samples = len(face_data)
    m_input_dim = len(face_data[0])
    W_weights = np.zeros((m_input_dim, d_latent))
    mu = np.mean(faces, axis=0) # Shape: m_input_dim,
    
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
    
    # To manually make sure covariance is calculated correctly
    # var = np.var(face_data, axis=0)
    # print("Variance: ")
    # print(var.shape)
    # print(var)
    
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
    

def make_2d_scatter(face_data):
    '''
    Problem 2b of HW 1
    '''
    
    latent_dim = 2
    
    W, sigma2, mu = fit_ppca(face_data, latent_dim)
    
    latent = x_to_z(face_data[0], W, sigma2, mu)
    
    latent_vals = []
    for i in range(len(face_data)):
        latent_vals.append(x_to_z(face_data[i], W, sigma2, mu))
        
    z_values = np.array([x_to_z(x, W, sigma2, mu) for x in face_data])
    
    
    ################ 2D scatter plot ################
    # Extract x and y coordinates
    if(latent_dim == 2):
        x_coords, y_coords = z_values[:, 0], z_values[:, 1]

        # Plot scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, s=5, alpha=0.7)  # Adjust size and transparency if needed
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Scatter of Latent Z's")
        # plt.grid(True)
        plt.show()
    
    if(latent_dim == 3):
        
        # Extract x, y, and z coordinates
        x_coords, y_coords, z_coords = z_values[:, 0], z_values[:, 1], z_values[:, 2]

        # Create 3D scatter plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords, y_coords, z_coords, s=5, alpha=0.7)

        # Labels and title
        ax.set_xlabel("Principal 1")
        ax.set_ylabel("Principal 2")
        ax.set_zlabel("Principal 3")
        ax.set_title("3D Scatter Plot of Latent Projections")

        plt.show()
  
def problem_2c(faces, latent_d):
    W, sigma2, mu = fit_ppca(faces, latent_d)
    random_faces = np.random.randint(0, 2000, 25)
    random_faces = faces[random_faces, :]
    
    reconstructed_faces = np.zeros((25, 576))
    for i in range(len(random_faces)):
        z = x_to_z(random_faces[i], W, sigma2, mu)
        x_hat = z_to_x(z, W, sigma2, mu)
        reconstructed_faces[i, :] = x_hat
        
    plot_faces(reconstructed_faces)
    
def problem_2d(faces):
    
    d_latent_dim = 64
    W, sigma2, mu = fit_ppca(faces, d_latent_dim)
    
    random_faces = np.zeros((100, 576))
    for i in range(100):
        random_z = np.random.normal(loc=0, scale=1, size=d_latent_dim)
        x_hat = z_to_x(random_z, W, sigma2, mu)
        random_faces[i:] = x_hat
        
    plot_faces(random_faces, 100)
    
def problem_2e(faces):
    '''
    For 5 images, randomly vary a latent dimension linearly 10 different times
    '''
    d_latent_dim = 16
    W, sigma2, mu = fit_ppca(faces, d_latent_dim)
    
    dif_dims = []
    idx = np.random.randint(0, 2000)
    
    for i in range(5):
        zi = x_to_z(faces[idx], W, sigma2, mu)
        
        row_faces = []
        rand_shift = 0.1
        perturbs = np.linspace(0, rand_shift, 10)
        for j in range(10):
            
            # Perturb dimension i of latent vector
            perturbed_z = zi
            perturbed_z[i] += perturbs[j]
            
            row_faces.append(z_to_x(perturbed_z, W, sigma2, mu))
        
        dif_dims.append(row_faces)
            
    # Each entry in dif_dims is a list of 10 elements. Each element in dif_dims[i] is a np array, 576 entries. 
    # Make a collage, 5 rows and 10 columns, where each row is the different element in dif_dims[i].
    fig, axes = plt.subplots(5, 10, figsize=(15, 7))
    for i in range(5):
        for j in range(10):
            face = dif_dims[i][j].reshape(24, 24)
            axes[i, j].imshow(face, cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    
    faces = np.load('./eigenfaces.npy')
    
    # X is is in R(m), which is 24x24, AKA 576
    # Latent dimension 'd' is less than m. For the assignment, ranges between {2, 16, 32, 64}
    # Parameters: W, m x d matrix, projects latent to observation
    #             mu, the average of the entire dataset
    #             sigma2, the variance of the eigenvectors of the dataset
    # plot_faces(faces[0:100], 25)
    # make_2d_scatter(faces) # problem 2b
    
    # problem_2c(faces, latent_d = 32)
    # problem_2d(faces)
    problem_2e(faces)
    