import numpy as np
from scipy import sparse
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import random
from sklearn.manifold import SpectralEmbedding

def spectral_reorder(matrix, n_components=1):
    """Use scikit-learn's SpectralEmbedding to reorder the authors based on the spectral decomposition."""
    # Compute the adjacency matrix
    adjacency_matrix = np.dot(matrix.T, matrix)

    # Use SpectralEmbedding to compute the embedding
    embedding = SpectralEmbedding(n_components=n_components, affinity='precomputed')
    embedding_result = embedding.fit_transform(adjacency_matrix)

    # Compute the order based on the first component
    order = np.argsort(embedding_result[:, 0])

    return order

def visualize_coauthor_matrix(use_spectral_reorder=False):
    # Load the co-authorship matrix
    matrix = sparse.load_npz('coauthor_matrix.npz')
    print(f"Loaded coauthor matrix with shape {matrix.shape} and {matrix.nnz} non-zero entries.")

    # Convert sparse matrix to dense for visualization
    dense_matrix = matrix.toarray()

    # Load paper titles and author names
    with open('paper_titles.txt', 'r', encoding='utf-8') as f:
        paper_titles = [line.strip() for line in f.readlines()[:201]]
    with open('all_authors.txt', 'r', encoding='utf-8') as f:
        author_names = [line.strip() for line in f.readlines()]

    # Find the index of 'H. Mannila' in the original author list
    if 'H. Mannila' in author_names:
        mannila_idx = author_names.index('H. Mannila')
        dense_matrix = np.delete(dense_matrix, mannila_idx, axis=1)

    # Filter out H. Mannila from authors
    author_names = [author for author in author_names if author != 'H. Mannila']

    # Reorder authors using spectral decomposition if requested
    if use_spectral_reorder:
        order = spectral_reorder(dense_matrix)
        dense_matrix = dense_matrix[:, order]
        author_names = [author_names[i] for i in order]

    # Randomize the order of papers
    # random.shuffle(paper_titles)

    # Create a DataFrame for the heatmap
    df = pd.DataFrame(dense_matrix, index=paper_titles, columns=author_names)

    # Create the Dash app
    app = dash.Dash(__name__)

    # Define the layout
    app.layout = html.Div([
        html.H1("Co-authorship Matrix"),
        dcc.Graph(
            id='heatmap',
            figure=px.imshow(
                df,
                labels=dict(x="Authors", y="Papers", color="Co-authorship"),
                title="Co-authorship Matrix",
                color_continuous_scale='viridis',
                width=800,
                height=1200
            ).update_layout(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                coloraxis_showscale=False
            )
        )
    ])

    # Run the app
    app.run_server(debug=True)

if __name__ == '__main__':
    visualize_coauthor_matrix(use_spectral_reorder=True) 