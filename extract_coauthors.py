import bibtexparser
import re
import unicodedata

def name_to_initials(name):
    # Remove braces and LaTeX commands
    name = re.sub(r'\\[a-zA-Z]+', '', name)
    name = name.replace('{', '').replace('}', '')
    name = name.replace('"', '').replace("'", '')
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    
    # Split by comma if present (BibTeX format: Last, First)
    if ',' in name:
        last, firsts = [x.strip() for x in name.split(',', 1)]
    else:
        parts = name.split()
        if len(parts) == 1:
            return name.strip()
        last = parts[-1]
        firsts = ' '.join(parts[:-1])
    
    # Get initials
    initials = ''.join([f'{p[0]}.' for p in firsts.split() if p])
    return f"{initials} {last}".strip()

def is_heikki_mannila(name):
    # Accept both full and initial forms
    name = name.lower().replace('.', '').replace(' ', '')
    return name in ['heikkimannila', 'hmannila']

def extract_coauthors():
    with open('HMannila.bib', 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    
    coauthors = set()
    for entry in bib_database.entries:
        if 'author' in entry:
            authors = [a.strip() for a in entry['author'].replace('\n', ' ').split(' and ')]
            for author in authors:
                norm = name_to_initials(author)
                if not is_heikki_mannila(norm.replace('.', '').replace(' ', '').lower()):
                    coauthors.add(norm)
    
    # Sort by last name
    def last_name_key(name):
        return name.split()[-1].lower() if name.split() else name.lower()
    sorted_coauthors = sorted(coauthors, key=last_name_key)
    
    with open('co-authors.txt', 'w', encoding='utf-8') as f:
        for author in sorted_coauthors:
            f.write(author + '\n')

if __name__ == '__main__':
    extract_coauthors() 