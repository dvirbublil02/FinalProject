import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import requests
from xml.etree import ElementTree

# Parse anomalous_edges.txt
def parse_anomalous_edges(file_path):
    anomalous_edges = defaultdict(dict)
    with open(file_path, 'r') as f:
        current_snap = None
        for line in f:
            line = line.strip()
            if not line or line == '---':
                continue
            if line.startswith("Snapshot:"):
                current_snap = int(line.split(":")[1].strip())
            elif line.startswith("Edge:"):
                edge = eval(line.split(":")[1].strip())
            elif line.startswith("Anomaly Score:"):
                score = float(line.split(":")[1].strip())
                anomalous_edges[current_snap][edge] = score
    return anomalous_edges

# Parse years_file.txt
def parse_years(years_file):
    node_years = {}
    with open(years_file, 'r') as file:
        for line in file:
            line = line.strip()
            try:
                parts = line.split()
                node = parts[0]
                year = parts[1]
                node_years[node] = year
            except IndexError:
                print(f"Skipping line with incorrect format: {line}")
    return node_years

# Map snapshot edges to years
def map_snapshot_to_years_ordered(anomalous_edges, node_years):
    snapshot_to_year_edges = {}
    for snapshot, edges in anomalous_edges.items():
        year_edges = []
        for (node1, node2), score in edges.items():
            year1 = node_years.get(str(node1))
            year2 = node_years.get(str(node2))
            if year1 and year2:
                year_edges.append((year1, year2, score))
        snapshot_to_year_edges[snapshot] = year_edges
    return snapshot_to_year_edges

# Track anomaly evolution
def track_anomaly_evolution(anomalous_edges, node_years):
    anomaly_status = defaultdict(lambda: [0] * len(node_years))
    years_set = sorted(set(node_years.values()))
    year_to_index = {year: i for i, year in enumerate(years_set)}

    cited_counts = defaultdict(int)
    citing_counts = defaultdict(int)
    single_citations = defaultdict(set)

    # Step 1: Count citing and cited occurrences
    for snapshot, edges in anomalous_edges.items():
        for (node1, node2), _ in edges.items():
            cited_counts[node2] += 1
            citing_counts[node1] += 1
            single_citations[node1].add(node2)

    # Step 2: Process each anomalous edge
    for snapshot, edges in anomalous_edges.items():
        for (node1, node2), _ in edges.items():
            year1 = node_years.get(str(node1))
            year2 = node_years.get(str(node2))
            
            if not year1 or not year2:
                continue

            # Condition 1: New Article Citing an Older Article
            if year1 > year2:
                anomaly_status[node1][year_to_index[year2]] = 1
            elif year1 < year2:
                anomaly_status[node1][year_to_index[year1]] = 1

            # Condition 2: Single Citation with Multiple References
            if cited_counts[node2] == 1 and citing_counts[node1] > 1:
                if year1 < year2:
                    anomaly_status[node2][year_to_index[year2]] = 0
                    anomaly_status[node1][year_to_index[year2]] = 1

            # Condition 3: Single Citation Between Both
            if len(single_citations[node1]) == 1 and len(single_citations[node2]) == 1:
                if year1 < year2:
                    anomaly_status[node2][year_to_index[year2]] = 1
                    anomaly_status[node1][year_to_index[year1]] = 1
                elif year1 == year2:
                    anomaly_status[node1][year_to_index[year1]] = 1
                    anomaly_status[node2][year_to_index[year2]] = 1

            # Condition 4: Article Citing Multiple Others
            if citing_counts[node1] > 1:
                if year1 < year2:
                    anomaly_status[node1][year_to_index[year2]] = 1
                else:
                    anomaly_status[node1][year_to_index[year1]] = 1

    # New Condition: Ensure that once an article is anomalous, it remains anomalous in all subsequent years
    for node, years in anomaly_status.items():
        for i in range(1, len(years)):
            if years[i-1] == 1:
                years[i] = 1

    return anomaly_status, years_set

# Print anomaly matrix
def print_anomaly_matrix(anomaly_status, node_years, years_set, output_file):
    with open(output_file, 'w') as f:
        f.write("Anomaly Status Matrix:\n")
        f.write("Node/Year ")
        for year in years_set:
            f.write(f"{year} ")
        f.write("\n")

        for node, status in anomaly_status.items():
            if all(state == 0 for state in status):
                continue
            publication_year = node_years.get(str(node))
            if publication_year:
                f.write(f"{node} ({publication_year}) ")
                for year, is_anomalous in zip(years_set, status):
                    f.write(f"{'Anomalous' if is_anomalous == 1 else 'Normal'} ")
                f.write("\n")

# Plot anomalies by year pair
def plot_anomalies_by_year_pair(snapshot_to_year_edges, anomaly_status, output_file):
    year_pair_counts = defaultdict(int)
    for snapshot, edges in snapshot_to_year_edges.items():
        for year1, year2, _ in edges:
            year_pair_counts[(year1, year2)] += 1

    sorted_pairs = sorted(year_pair_counts.keys())
    sorted_counts = [year_pair_counts[pair] for pair in sorted_pairs]
    labels = [f"{pair[0]} -> {pair[1]}" for pair in sorted_pairs]

    plt.bar(labels, sorted_counts, color='skyblue')
    plt.xlabel('Year Pair (Citing -> Cited)')
    plt.ylabel('Anomalies Count')
    plt.title('Anomalies by Year Pair (Citing -> Cited)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Plot anomalies per snapshot
def plot_anomalies_per_snapshot(anomalous_edges, anomaly_status, output_file):
    filtered_anomalous_edges = {
        snapshot: {edge: score for edge, score in edges.items() if any(anomaly_status[edge[0]] + anomaly_status[edge[1]])}
        for snapshot, edges in anomalous_edges.items()
    }

    snapshot_anomalies_count = {snapshot: len(edges) for snapshot, edges in filtered_anomalous_edges.items()}
    snapshots = list(snapshot_anomalies_count.keys())
    anomaly_counts = list(snapshot_anomalies_count.values())

    plt.bar(snapshots, anomaly_counts, color='orange')
    plt.xlabel('Snapshot Number')
    plt.ylabel('Number of Anomalies')
    plt.title('Number of Anomalies per Snapshot')
    plt.xticks(snapshots)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Fetch MeSH terms and categories for PubMed articles
def get_mesh_terms_and_categories(pmid):
    """
    Fetches MeSH terms and their associated categories for a PubMed article using efetch API.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Parse the XML response
        tree = ElementTree.fromstring(response.content)
        
        mesh_terms = defaultdict(list)  # Dictionary to map DescriptorName to QualifierNames
        
        # Extract MeSH terms and categories
        for mesh_heading in tree.findall(".//MeshHeading"):
            descriptor_name = mesh_heading.find("DescriptorName").text
            qualifier_names = [qualifier.text for qualifier in mesh_heading.findall("QualifierName")]
            mesh_terms[descriptor_name] = qualifier_names if qualifier_names else ["Unknown"]
        
        return mesh_terms if mesh_terms else {"No MeSH Terms Found": ["Unknown"]}

    except Exception as e:
        print(f"Error fetching MeSH terms for PubMed ID {pmid}: {e}")
        return None

# Categorize articles based on MeSH terms
def categorize_articles_by_mesh(pubmed_ids):
    """
    Fetches and categorizes articles based on their MeSH terms and categories.
    """
    categorized_articles = defaultdict(lambda: defaultdict(list))
    
    for pmid in pubmed_ids:
        mesh_terms = get_mesh_terms_and_categories(pmid)
        if mesh_terms is None:
            categorized_articles["Unknown"][pmid].append("Unknown")
            continue
        for term, categories in mesh_terms.items():
            for category in categories:
                categorized_articles[category][pmid].append(term)
    
    return categorized_articles

# New function to plot anomalies per category
def plot_anomalies_per_category(anomalous_pubmed_ids, output_file):
    """
    Plots a histogram of anomalies per category using MeSH terms and categories.
    """
    try:
        categorized_articles = categorize_articles_by_mesh(anomalous_pubmed_ids)

        # Count anomalies per category
        category_counts = {category: len(articles) for category, articles in categorized_articles.items()}

        # Plot histogram
        categories = list(category_counts.keys())
        counts = list(category_counts.values())

        plt.figure(figsize=(10, 6))
        plt.bar(categories, counts, color='green')
        plt.xlabel('Category')
        plt.ylabel('Number of Anomalies')
        plt.title('Anomalies per Category (Based on MeSH Terms)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Anomalies per category plot saved to {output_file}")
    except Exception as e:
        print(f"Error creating category plot: {e}")

# Main script
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)

    anomalous_edges_file = 'data/OurResearch/anomalous_edges.txt'
    years_file = 'data/OurResearch/extracted_article_dates.txt'

    anomaly_matrix_file = 'results/anomaly_status_matrix.txt'
    year_pair_plot_file = 'results/anomalies_by_year_pair.png'
    snapshot_plot_file = 'results/anomalies_per_snapshot.png'
    category_plot_file = 'results/anomalies_per_category.png'

    anomalous_edges = parse_anomalous_edges(anomalous_edges_file)
    node_years = parse_years(years_file)

    snapshot_to_year_edges = map_snapshot_to_years_ordered(anomalous_edges, node_years)
    anomaly_status, years_set = track_anomaly_evolution(anomalous_edges, node_years)

    print_anomaly_matrix(anomaly_status, node_years, years_set, anomaly_matrix_file)
    plot_anomalies_by_year_pair(snapshot_to_year_edges, anomaly_status, year_pair_plot_file)
    plot_anomalies_per_snapshot(anomalous_edges, anomaly_status, snapshot_plot_file)

    # Extract anomalous PubMed IDs from anomaly_status
    anomalous_pubmed_ids = [node for node, status in anomaly_status.items() if any(status)]

    # Categorize articles
    categories = categorize_articles_by_mesh(anomalous_pubmed_ids)

    # Create the new plot
    plot_anomalies_per_category(anomalous_pubmed_ids, category_plot_file)

    print(f"Anomaly status matrix saved to {anomaly_matrix_file}")
    print(f"Anomalies by year pair plot saved to {year_pair_plot_file}")
    print(f"Anomalies per snapshot plot saved to {snapshot_plot_file}")
    print(f"Anomalies per category plot saved to {category_plot_file}")
