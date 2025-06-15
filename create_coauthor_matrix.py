import bibtexparser
import numpy as np
from scipy import sparse
import pandas as pd
from extract_coauthors import name_to_initials, is_heikki_mannila

def get_all_authors():
    """Get all unique authors (including Heikki Mannila) from the bibliography."""
    with open('HMannila.bib', 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    
    authors = set()
    for entry in bib_database.entries:
        if 'author' in entry:
            paper_authors = [a.strip() for a in entry['author'].replace('\n', ' ').split(' and ')]
            for author in paper_authors:
                norm = name_to_initials(author)
                authors.add(norm)
    def last_name_key(name):
        return name.split()[-1].lower() if name.split() else name.lower()
    return sorted(authors, key=last_name_key)
    

def create_coauthor_matrix():
    """Create a sparse matrix representing the co-authorship network."""
    # Get all unique authors
    all_authors = get_all_authors()
    author_to_idx = {author: idx for idx, author in enumerate(all_authors)}
    
    # Read bibliography
    with open('HMannila.bib', 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    print(f"Total entries in bibliography: {len(bib_database.entries)}")
    
    # Initialize lists for sparse matrix construction
    rows = []
    cols = []
    data = []
    paper_titles = []
    
    # Process each paper
    for entry in bib_database.entries:
        if 'author' not in entry:
            continue
            
        # Get paper title
        title = entry.get('title', '').replace('{', '').replace('}', '').replace('\n', ' ')
        paper_titles.append(title)
        
        # Get authors for this paper
        paper_authors = [a.strip() for a in entry['author'].replace('\n', ' ').split(' and ')]
        paper_authors = [name_to_initials(author) for author in paper_authors]
        
        # Add entries to sparse matrix
        paper_idx = len(paper_titles) - 1
        for author in paper_authors:
            if author in author_to_idx:
                rows.append(paper_idx)
                cols.append(author_to_idx[author])
                data.append(1)
    
    print(f"Number of paper titles collected: {len(paper_titles)}")
    
    # Create sparse matrix
    matrix = sparse.csr_matrix((data, (rows, cols)), 
                             shape=(len(paper_titles), len(all_authors)))
    
    # Save matrix and metadata
    sparse.save_npz('coauthor_matrix.npz', matrix)
    print(f"Saved coauthor matrix with shape {matrix.shape} and {matrix.nnz} non-zero entries.")
    
    # Save paper titles
    with open('paper_titles.txt', 'w', encoding='utf-8') as f:
        for title in paper_titles:
            f.write(title + '\n')
    
    # Save author list
    with open('all_authors.txt', 'w', encoding='utf-8') as f:
        for author in all_authors:
            f.write(author + '\n')
    
    # Create summary statistics
    num_papers_per_author = pd.Series(matrix.sum(axis=0).A1, index=all_authors)
    num_authors_per_paper = pd.Series(matrix.sum(axis=1).A1, index=paper_titles)

    print(f"Created co-authorship matrix with:")
    print(f"- {matrix.shape[0]} papers")
    print(f"- {matrix.shape[1]} unique authors")
    print(f"- {matrix.nnz} total author-paper connections")
    print("\nTop 10 most frequent co-authors:")
    top_authors = num_papers_per_author.sort_values(ascending=False)
    for author, count in top_authors.head(10).items():
        print(f"{author}: {int(count)} papers")
    print("\nTop 10 papers with most authors:")
    top_papers = num_authors_per_paper.sort_values(ascending=False)
    for title, count in top_papers.head(10).items():
        print(f"{title}: {int(count)} authors")

if __name__ == '__main__':
    create_coauthor_matrix() 