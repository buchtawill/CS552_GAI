import matplotlib.pyplot as plt
import networkx as nx
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
    pos = nx.spring_layout(g)  # Layout for node positioning
    colors = [g.nodes[node]['color'] for node in g.nodes]

    # Draw the graph
    nx.draw(
        g, pos, with_labels=True, node_color=colors, edge_color='gray',
        node_size=500, font_size=10
    )

    plt.show()
