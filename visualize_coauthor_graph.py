import numpy as np
from scipy import sparse
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd

def load_coauthor_matrix():
    """Load the coauthor matrix and author names."""
    matrix = sparse.load_npz('coauthor_matrix.npz')
    with open('all_authors.txt', 'r', encoding='utf-8') as f:
        authors = [line.strip() for line in f]
    return matrix, authors

def create_coauthor_graph(matrix, authors):
    """Create a graph from the coauthor matrix."""
    # Convert paper-author matrix to author-author adjacency matrix
    # This counts how many papers each pair of authors has co-authored
    adjacency_matrix = matrix.T @ matrix
    
    # Create NetworkX graph
    G = nx.from_scipy_sparse_array(adjacency_matrix)
    
    # Add author names as node attributes
    nx.set_node_attributes(G, {i: name for i, name in enumerate(authors)}, 'name')
    
    # Add edge weights (number of co-authored papers)
    for u, v, d in G.edges(data=True):
        d['weight'] = d['weight']
    
    return G

def compute_graph_layout(G):
    """Compute the Fruchterman-Reingold (spring) layout of the graph."""
    # Use NetworkX's spring layout
    pos = nx.spring_layout(G, weight='weight', seed=42)
    
    # Convert positions to lists for Plotly
    x_coords = []
    y_coords = []
    for node in G.nodes():
        x_coords.append(pos[node][0])
        y_coords.append(pos[node][1])
    
    return x_coords, y_coords

def create_graph_visualization(G, x_coords, y_coords):
    """Create a Plotly figure for the graph visualization."""
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = x_coords[edge[0]], y_coords[edge[0]]
        x1, y1 = x_coords[edge[1]], y_coords[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[node]['name'] for node in G.nodes()],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left'
            )
        )
    )
    
    # Add node colors based on number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    node_trace.marker.color = node_adjacencies
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title={'text': 'Co-authorship Network', 'font': {'size': 16}},
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def create_dashboard():
    """Create a Dash app for the graph visualization."""
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Load data
    matrix, authors = load_coauthor_matrix()
    G = create_coauthor_graph(matrix, authors)
    x_coords, y_coords = compute_graph_layout(G)
    fig = create_graph_visualization(G, x_coords, y_coords)
    
    # Create app layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Co-authorship Network Visualization", className="text-center my-4"),
                dcc.Graph(id='graph', figure=fig)
            ])
        ])
    ], fluid=True)
    
    return app

if __name__ == '__main__':
    app = create_dashboard()
    app.run(debug=True) 