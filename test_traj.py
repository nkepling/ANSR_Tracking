import networkx as nx
import matplotlib.pyplot as plt
import random
import scipy.sparse as sp
import scipy
import scipy.sparse
import numpy as np

def create_grid_graph(rows, cols):
    """
    Creates an m x n grid graph.
    Nodes are (row, col) tuples.
    """
    G = nx.grid_2d_graph(rows, cols)
    return G

def add_coordinate_attributes(G):
    """
    Adds 'grid_coord' (row,col) and 'xy_plot_coord' (x,y for plotting)
    attributes to each node in G.
    Assumes nodes are (row, col) tuples from grid_2d_graph.
    """
    for node in G.nodes():
        row, col = node # Node itself is the (row, col) grid coordinate
        G.nodes[node]['grid_coord'] = (row, col)
        G.nodes[node]['xy_plot_coord'] = (col, -row) # Plotting coordinate (x=col, y=-row)
    return G

def randomly_remove_edges_while_connected(G,max_removals=80):
    """
    Randomly removes edges from graph G while ensuring G remains connected.

    Args:
        G (networkx.Graph): The input graph. This graph will be modified in place.

    Returns:
        int: The number of edges removed.
    """
    if not nx.is_connected(G):
        print("Input graph is not connected. No edges will be removed.")
        return 0

    # Get a list of all edges and shuffle them randomly
    # We convert to list because we'll be modifying the graph
    edges_to_consider = list(G.edges())
    random.shuffle(edges_to_consider)

    removed_count = 0
    for u, v in edges_to_consider:
        # Temporarily remove the edge
        G.remove_edge(u, v)

        # Check if the graph is still connected
        if nx.is_connected(G):
            # If still connected, the removal is permanent
            removed_count += 1
        else:
            # If disconnected, add the edge back
            G.add_edge(u, v)
        if removed_count >= max_removals:
            break
            
    return removed_count

def main():
    rows = 10
    cols = 10

    # 1. Create the initial grid graph
    grid_graph = create_grid_graph(rows, cols)
    
    print(f"Initial {rows}x{cols} grid graph:")
    print(f"Number of nodes: {grid_graph.number_of_nodes()}")
    initial_edges = grid_graph.number_of_edges()
    print(f"Number of edges: {initial_edges}")

    # Node positions for plotting (consistent for both plots)
    pos = {(r, c): (c, -r) for r, c in grid_graph.nodes()}

    grid_graph = add_coordinate_attributes(grid_graph)

    removed_edges_count = randomly_remove_edges_while_connected(grid_graph)

    pos = nx.get_node_attributes(grid_graph, 'xy_plot_coord')

    edge_betweenness = nx.edge_betweenness_centrality(grid_graph)

    # Prepare labels for edges
    # We'll format the scores to 2 decimal places for readability
    edge_labels = {edge: f'{score:.2f}' for edge, score in edge_betweenness.items()}

    plt.figure(figsize=(10, 8)) # Increased figure size for better label visibility
    
    nx.draw_networkx_nodes(grid_graph, pos, node_size=300, node_color='skyblue', alpha=0.8) # Increased node size
    nx.draw_networkx_edges(grid_graph, pos, edge_color='gray', alpha=0.6)
    
    # Node labels are still the (row,col) tuples (the node names themselves)
    nx.draw_networkx_labels(grid_graph, pos, font_size=8, font_family="sans-serif") 



    # Draw edge labels
    # `bbox` parameters can help make labels more readable if they overlap
    nx.draw_networkx_edge_labels(grid_graph, pos, edge_labels=edge_labels, font_size=7, bbox={"alpha": 0.5, "pad": 1})

    plt.title(f"{rows}x{cols} Grid Graph with Edge Betweenness Centrality")
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


    scipy_sparse_matrix = nx.to_scipy_sparse_array(grid_graph, dtype=np.float64,format='csr')
    scipy_sparse_matrix.indices = scipy_sparse_matrix.indices.astype(np.int32)
    scipy_sparse_matrix.indptr = scipy_sparse_matrix.indptr.astype(np.int32)
    

    
    yen_paths = scipy.sparse.csgraph.yen(scipy_sparse_matrix, 0, 1, 3)

    print("Yen's algorithm paths (as sparse matrix):")
    print(yen_paths)



if __name__ == '__main__':
    main()