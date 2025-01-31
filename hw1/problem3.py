import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1./(1 + np.exp(-x))

# Google gemini method
def mirror_lower_to_upper(matrix):
    """Mirrors the lower triangle of a matrix to the upper triangle."""

    # Get the lower triangle (including diagonal)
    lower = np.tril(matrix)

    # Transpose the lower triangle to get the upper triangle
    upper = lower.T

    # Combine the lower and upper triangles, overwriting the original upper triangle
    return lower + upper - np.diag(np.diag(matrix))

def single_mle_gradient(xi, xj, a, b, edj):
    
    # dL_db = 0.0
    # dL_da = 0.0
    
    inner_prod = np.dot(xi.T, xj)
    sigmoid_axxpb = sigmoid(a*inner_prod+b)
    
    dL_da = edj - sigmoid_axxpb * inner_prod
    dL_db = edj - sigmoid_axxpb
        
    return dL_da, dL_db

def compute_grad(adj_mat, X_sample, a, b):
    
    row_idxs, col_idxs = np.tril_indices(n=15, k=-1)
    num_elements = len(row_idxs)
    dL_da = 0.0
    dL_db = 0.0
    for edj_loc in range(num_elements):
        i = row_idxs[edj_loc]
        j = col_idxs[edj_loc]
        edj = adj_mat[i, j]
        next = single_mle_gradient(X_sample[i], X_sample[j], a, b, edj)
        dL_da += next[0]
        dL_db += next[1]
    
    return dL_da, dL_db    

def likelihood_of_data(adj_mat, X, a, b):
    '''
    Sum of MLE function for given parameters a and b, and a single graph of X
    '''
    
    row_idxs, col_idxs = np.tril_indices(n=15, k=-1)

    num_elements = len(row_idxs)
    sum = 0.0
    for edj_loc in range(num_elements):
        i = row_idxs[edj_loc]
        j = col_idxs[edj_loc]
        edj = adj_mat[i, j]
        if(edj == 1):
            sum += np.log(sigmoid(a * np.dot(X[i].T, X[j]) + b))
        else:
            sum += np.log(1 - sigmoid(a * np.dot(X[i].T, X[j]) + b))
    
    return sum

def single_edge_likelihood(xi, xj, a, b):
    return sigmoid(a*np.dot(xi.T, xj) + b)

def train_model(graphs):
    
    a, b = np.random.rand(), np.random.rand()
    dL_da, dL_db = 0.0, 0.0
    
    a_samples = []
    b_samples = []
    
    num_samples = len(graphs)
    
    lr = 0.01
    for i in range(20):
        
        for sample_idx in range(num_samples):
            X = graphs[sample_idx][0]
            adj = graphs[sample_idx][1]
            
            likely = likelihood_of_data(adj, X, a, b)
            dL_da, dL_db = compute_grad(adj, X, a, b)
        
            a += dL_da * lr
            b += dL_db * lr
            
        print(f'Epoch {i+1:>3} MLE: {likely:.6f} a: {a:.6f} b: {b:.6f}')
        a_samples.append(a)
        b_samples.append(b)
        
    plt.plot(a_samples)
    plt.plot(b_samples)
    plt.show()
    
    return a, b

# X is an (n x 3) feature matrix of n examples, each having 3 features in [-1,+1].
# E is an (n x n) symmetric binary edge matrix.
def render_graph (X, E):
    g = nx.Graph()
    for node, x in enumerate(X):
        g.add_node(node, color=x/2+0.5)  # rescale the features to be in [0,1]

    nodes = list(g.nodes)
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            if E[i,j]:
                g.add_edge(nodes[i], nodes[j])
    pos = nx.spring_layout(g, k=0.5)  # Layout for node positioning
    colors = [g.nodes[node]['color'] for node in g.nodes]

    # Draw the graph
    nx.draw(
        g, pos, with_labels=True, node_color=colors, edge_color='gray',
        node_size=500, font_size=10
    )

    plt.show()

def create_random_students(n:int=15, m:int=3):
    
    students = np.zeros((n,m))
    
    for student in range(n):
        for feature in range(m):
            students[student, feature] = (np.random.rand() * 2) - 1
            
    return students
    
if __name__ == '__main__':
    
    graphs = pickle.load(open('classroom_graphs.pkl', 'rb'))
    
    # render_graph(graphs[3][0], graphs[3][1])
    
    # a, b = train_model(graphs)
    
    # After training, a and b are:
    a =  6.6989
    # b = -6.1617
    b = -4
    
    # Problem 3c
    
    for g in range(2):
        students = create_random_students()
        edj_prob = np.zeros((15,15))
        adj_mat = np.zeros((15, 15), dtype=int)
        
        row_idxs, col_idxs = np.tril_indices(n=15, k=-1)
        num_elements = len(row_idxs)
        for edj_loc in range(num_elements):
            i = row_idxs[edj_loc]
            j = col_idxs[edj_loc]
            edj_prob[i,j] = single_edge_likelihood(students[i], students[j], a, b)
            adj_mat[i,j] = np.random.binomial(1, edj_prob[i,j])
        
        adj_mat = mirror_lower_to_upper(adj_mat)
        # print(adj_mat)
        render_graph(students, adj_mat)
    