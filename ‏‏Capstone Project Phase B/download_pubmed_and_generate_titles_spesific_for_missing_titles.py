import os
import requests
from xml.etree import ElementTree
from torch_geometric.datasets import Planetoid

# Function to download the PubMed dataset
def download_pubmed_dataset(dataset_name, download_dir='data/OurResearch'):
    os.makedirs(download_dir, exist_ok=True)
    dataset = Planetoid(root=download_dir, name='PubMed')
    print(f"Dataset: {dataset_name}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of nodes: {dataset.data.num_nodes}")
    print(f"Number of edges: {dataset.data.num_edges}")
    return dataset

# Function to fetch article title and publication date from PubMed
def get_article_details(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=xml"
    try:
        response = requests.get(url, timeout=10)  # Add a timeout for requests
        response.raise_for_status()  # Raise an exception for HTTP errors
        tree = ElementTree.fromstring(response.content)
        title_element = tree.find(".//Item[@Name='Title']")
        pubdate_element = tree.find(".//Item[@Name='PubDate']")
        title = title_element.text if title_element is not None else 'Title Not Found'
        pubdate = pubdate_element.text if pubdate_element is not None else 'Publication Date Not Found'
        return title, pubdate
    except (requests.exceptions.RequestException, ElementTree.ParseError) as e:
        return 'Error fetching title', 'Error fetching publication date'

# Function to generate the article titles file
def generate_article_titles(dataset, output_dir='data/raw', filename='article_titles_with_date.txt'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    data = dataset[0]

    # Open the file in append mode
    with open(output_path, 'a', encoding='utf-8') as f:
        for i in range(50, 51):   
            pmid = str(i)
            try:
                # Fetch article details
                title, pubdate = get_article_details(pmid)
                # Write to file
                f.write(f"{pmid} {title} {pubdate}\n")
                f.flush()  # Ensure data is written to disk
                # Print progress
                print(f"Processed article {i}: {title[:50]}...")  # Show first 50 chars of title
            except Exception as e:
                print(f"Error processing article {i}: {e}")

    print(f"Article titles and publication dates saved in {output_path}")

# Main Function
def main():
    dataset_name = 'PubMed'
    dataset = download_pubmed_dataset(dataset_name)
    generate_article_titles(dataset, output_dir='data/raw', filename='article_titles_with_date_spesifc.txt')

if __name__ == '__main__':
    main()
