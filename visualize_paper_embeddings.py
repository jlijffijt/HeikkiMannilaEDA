import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import bibtexparser
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
from enum import Enum

class EmbeddingType(Enum):
    TITLE_ONLY = "title_only"
    TITLE_AND_ABSTRACT = "title_and_abstract"

def load_bibtex(file_path):
    with open(file_path, 'r', encoding='utf-8') as bibtex_file:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        return parser.parse_file(bibtex_file)

def clean_bibtex_text(text):
    """Remove all curly braces and backslashes from text."""
    return text.replace('{', '').replace('}', '').replace('\\', '')

def get_paper_info(entry):
    # Clean and split authors, also clean each author name
    authors = [clean_bibtex_text(author) for author in entry.get('author', '').split(' and ')]
    # Clean title
    title = clean_bibtex_text(entry.get('title', ''))
    abstract = entry.get('abstract', '')
    year = entry.get('year', '')
    return {
        'authors': authors,
        'title': title,
        'abstract': abstract,
        'year': year
    }

def get_text_for_embedding(paper_info, embedding_type):
    if embedding_type == EmbeddingType.TITLE_ONLY:
        return [info['title'] for info in paper_info]
    else:  # TITLE_AND_ABSTRACT
        return [f"{info['title']}. {info['abstract']}" for info in paper_info]

def get_scibert_embeddings(texts, batch_size=8):
    # Load SciBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    embeddings = []
    
    # Process texts in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize and prepare input
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                         max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding as document embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def create_tsne_visualization(embeddings, paper_info):
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'title': [clean_bibtex_text(info['title']) for info in paper_info],  # Ensure titles are clean
        'authors': [', '.join(clean_bibtex_text(author) for author in info['authors']) for info in paper_info],  # Ensure authors are clean
        'year': [info['year'] for info in paper_info]
    })
    
    return df

def create_dashboard(df):
    app = Dash(__name__)
    
    # Convert year to numeric for proper coloring
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    app.layout = html.Div([
        html.H1("Paper Embeddings Visualization"),
        html.Div([
            html.P("This visualization shows a t-SNE projection of paper embeddings generated using SciBERT."),
            html.P("Hover over points to see paper details. Points are colored by publication year."),
        ]),
        dcc.Graph(
            id='scatter-plot',
            figure=px.scatter(
                df,
                x='x',
                y='y',
                color='year',
                hover_data=['title', 'authors', 'year'],
                title='t-SNE Visualization of Paper Embeddings',
                width=800,
                height=800,
                color_continuous_scale='bluered'
            ).update_layout(
                hovermode='closest',
                showlegend=True
            )
        )
    ])
    
    return app

def main():
    # Load bibliography
    bibtex_file = 'HMannila.bib'
    bibliography = load_bibtex(bibtex_file)
    
    # Extract paper information
    paper_info = []
    for entry in bibliography.entries:
        if 'author' in entry:
            paper_info.append(get_paper_info(entry))
    
    # Choose embedding type
    embedding_type = EmbeddingType.TITLE_ONLY  # For title-only embeddings
    # or
    # embedding_type = EmbeddingType.TITLE_AND_ABSTRACT  # For title and abstract embeddings
    
    # Get embeddings for all papers
    print(f"Generating embeddings using {embedding_type.value}...")
    texts_for_embedding = get_text_for_embedding(paper_info, embedding_type)
    embeddings = get_scibert_embeddings(texts_for_embedding)
    
    # Create t-SNE visualization
    print("Creating t-SNE visualization...")
    df = create_tsne_visualization(embeddings, paper_info)
    
    # Create and run dashboard
    print("Starting dashboard...")
    app = create_dashboard(df)
    app.run_server(debug=True)

if __name__ == '__main__':
    main() 