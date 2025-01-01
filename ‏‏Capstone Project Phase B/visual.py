import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Reads the anomalous_edges.txt file and organizes its data
def parse_anomalous_edges(file_path):
    anomalous_edges = defaultdict(dict)
    with open(file_path, 'r') as f:
        current_snap = None
        for line in f:
            line = line.strip()
            if not line or line == '---':  # Skip empty lines or separators
                continue
            if line.startswith("Snapshot:"):  # Identify snapshot sections
                current_snap = int(line.split(":")[1].strip())
            elif line.startswith("Edge:"):  # Parse the edge info
                edge = eval(line.split(":")[1].strip())
            elif line.startswith("Anomaly Score:"):  # Get anomaly score
                score = float(line.split(":")[1].strip())
                anomalous_edges[current_snap][edge] = score
    return anomalous_edges

# Reads years_file.txt and maps nodes to their years
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

# Matches snapshot edges with their respective years
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

# Tracks how anomalies evolve across snapshots
def track_anomaly_evolution(anomalous_edges, node_years):
    anomaly_status = defaultdict(lambda: [0] * len(node_years))
    years_set = sorted(set(node_years.values()))
    year_to_index = {year: i for i, year in enumerate(years_set)}

    cited_counts = defaultdict(int)
    citing_counts = defaultdict(int)
    single_citations = defaultdict(set)

    # Count citations and references
    for snapshot, edges in anomalous_edges.items():
        for (node1, node2), _ in edges.items():
            cited_counts[node2] += 1
            citing_counts[node1] += 1
            single_citations[node1].add(node2)

    # Check conditions and update anomaly status
    for snapshot, edges in anomalous_edges.items():
        for (node1, node2), _ in edges.items():
            year1 = node_years.get(str(node1))
            year2 = node_years.get(str(node2))
            if not year1 or not year2:
                continue

            # New article citing older article
            if year1 > year2:
                anomaly_status[node1][year_to_index[year2]] = 1
            elif year1 < year2:
                anomaly_status[node1][year_to_index[year1]] = 1

            # Single citation with multiple references
            if cited_counts[node2] == 1 and citing_counts[node1] > 1:
                if year1 < year2:
                    anomaly_status[node2][year_to_index[year2]] = 0
                    anomaly_status[node1][year_to_index[year2]] = 1

            # Single citation between both
            if len(single_citations[node1]) == 1 and len(single_citations[node2]) == 1:
                if year1 < year2:
                    anomaly_status[node2][year_to_index[year2]] = 1
                    anomaly_status[node1][year_to_index[year1]] = 1
                elif year1 == year2:
                    anomaly_status[node1][year_to_index[year1]] = 1
                    anomaly_status[node2][year_to_index[year2]] = 1

            # Article citing multiple others
            if citing_counts[node1] > 1:
                if year1 < year2:
                    anomaly_status[node1][year_to_index[year2]] = 1
                else:
                    anomaly_status[node1][year_to_index[year1]] = 1

    # Keep articles anomalous in all subsequent years once flagged
    for node, years in anomaly_status.items():
        for i in range(1, len(years)):
            if years[i - 1] == 1:
                years[i] = 1

    return anomaly_status, years_set

# Saves anomaly status matrix to a file
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

# Plots anomalies by year pair
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

# Plots anomalies per snapshot
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

# Main script execution
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)

    anomalous_edges_file = 'data/OurResearch/anomalous_edges.txt'
    years_file = 'data/OurResearch/extracted_article_dates.txt'

    anomaly_matrix_file = 'results/anomaly_status_matrix.txt'
    year_pair_plot_file = 'results/anomalies_by_year_pair.png'
    snapshot_plot_file = 'results/anomalies_per_snapshot.png'

    anomalous_edges = parse_anomalous_edges(anomalous_edges_file)
    node_years = parse_years(years_file)

    snapshot_to_year_edges = map_snapshot_to_years_ordered(anomalous_edges, node_years)
    anomaly_status, years_set = track_anomaly_evolution(anomalous_edges, node_years)

    print_anomaly_matrix(anomaly_status, node_years, years_set, anomaly_matrix_file)
    plot_anomalies_by_year_pair(snapshot_to_year_edges, anomaly_status, year_pair_plot_file)
    plot_anomalies_per_snapshot(anomalous_edges, anomaly_status, snapshot_plot_file)

    # Saving someof the results. and also save the anaomaly matrix to be able load it in the visualtion process.
    print(f"Anomaly status matrix saved to {anomaly_matrix_file}")
    print(f"Anomalies by year pair plot saved to {year_pair_plot_file}")
    print(f"Anomalies per snapshot plot saved to {snapshot_plot_file}")
