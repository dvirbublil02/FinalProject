import os
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt

# Function to parse edges from a file
def parse_edges(file_path):
    edges = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('%'):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    article_id, citing_article_id = map(int, parts)
                    edges.append((article_id, citing_article_id))
                except ValueError as e:
                    print(f"Error parsing line: {line}. Error: {e}")
    return edges

# Function to parse anomalous edges
def parse_anomalous_edges(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Anomalous edges file not found at: {file_path}")
    anomalous_edges = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("Edge:"):
                edge_str = line.replace("Edge:", "").strip()
                edge = tuple(map(int, edge_str.strip("()").split(",")))
                anomalous_edges.append(edge)
    return anomalous_edges

# Function to create snapshots
def create_snapshots(edges, snapshot_size, output_folder, base_name):
    snapshots = []
    num_snapshots = len(edges) // snapshot_size + (1 if len(edges) % snapshot_size != 0 else 0)

    for i in range(num_snapshots):
        start_idx = i * snapshot_size
        end_idx = min((i + 1) * snapshot_size, len(edges))
        edges_subset = edges[start_idx:end_idx]

        G = nx.DiGraph()
        G.add_edges_from(edges_subset)

        # Draw the graph
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.1, iterations=50)
        nx.draw(G, pos, with_labels=True, node_size=50, node_color="skyblue", edge_color="gray")

        snapshot_name = f"{base_name}_snapshot_{i+1}.png"
        snapshot_path = os.path.join(output_folder, snapshot_name)
        plt.title(f"Snapshot {i+1} - Edges: {len(edges_subset)}", fontsize=14)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(snapshot_path)
        plt.close()
        snapshots.append(snapshot_path)

    return snapshots

# Function to create a full graph image
def create_full_graph(edges, anomalous_edges, output_path, add_title=True):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    normal_edges = [(u, v) for u, v in edges if (u, v) not in anomalous_edges]
    anomalous_subset = [(u, v) for u, v in edges if (u, v) in anomalous_edges]

    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.1, iterations=50)
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color="gray", alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=anomalous_subset, edge_color="red", width=2)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="skyblue")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    if add_title:
        plt.title(f"Full Graph - Total Edges: {len(edges)} (Anomalous: {len(anomalous_subset)})", fontsize=14)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(output_path)
    plt.close()
    print(f"Full graph saved at {output_path}")

# Function to combine snapshots with full graphs
def combine_snapshots_with_multiple_graphs(snapshot_paths, full_graph_paths, output_path, columns=3):
    images = [Image.open(path) for path in snapshot_paths]
    full_graph_images = [Image.open(full_graph_path) for full_graph_path in full_graph_paths]

    image_width, image_height = images[0].size
    full_graph_images = [img.resize((image_width, image_height)) for img in full_graph_images]

    total_images = len(images)
    rows = (total_images + columns - 1) // columns
    combined_width = image_width * columns
    combined_height = image_height * (rows + 1)

    combined_image = Image.new("RGB", (combined_width, combined_height), color="white")

    x_offset = 0
    y_offset = 0
    for i, image in enumerate(images):
        combined_image.paste(image, (x_offset, y_offset))
        x_offset += image_width
        if (i + 1) % columns == 0:
            x_offset = 0
            y_offset += image_height

    combined_image.paste(full_graph_images[0], (0, y_offset))
    combined_image.paste(full_graph_images[1], (combined_width - image_width, y_offset))

    combined_image.save(output_path)
    print(f"Combined image with full graphs saved at {output_path}")

# Main function
def main():
    input_folder = 'results/snapshots/test/'
    output_folder = 'snapshots/combined/'
    anomalous_file_path = 'data/OurResearch/anomalous_edges.txt'

    os.makedirs(output_folder, exist_ok=True)

    try:
        anomalous_edges = parse_anomalous_edges(anomalous_file_path)
    except FileNotFoundError as e:
        print(e)
        return

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)
            edges = parse_edges(file_path)
            print(f"Processing file: {file_name} with {len(edges)} edges.")

            base_name = os.path.splitext(file_name)[0]

            snapshot_paths = create_snapshots(edges, snapshot_size=500, output_folder=output_folder, base_name=base_name)

            full_graph_path_no_anomalies = os.path.join(output_folder, f"{base_name}_full_graph_no_anomalies.png")
            create_full_graph(edges, [], full_graph_path_no_anomalies, add_title=False)

            full_graph_path_with_anomalies = os.path.join(output_folder, f"{base_name}_full_graph_with_anomalies.png")
            create_full_graph(edges, anomalous_edges, full_graph_path_with_anomalies, add_title=False)

            combined_image_path = os.path.join(output_folder, f"{base_name}_combined_with_full_graphs.png")
            combine_snapshots_with_multiple_graphs(
                snapshot_paths,
                [full_graph_path_no_anomalies, full_graph_path_with_anomalies],
                combined_image_path,
                columns=3
            )

            # Delete all temporary snapshot images
            for snapshot_path in snapshot_paths:
                if os.path.exists(snapshot_path):
                    os.remove(snapshot_path)
                    print(f"Deleted temporary snapshot: {snapshot_path}")

            # Delete the temporary full graph images
            for graph_path in [full_graph_path_no_anomalies, full_graph_path_with_anomalies]:
                if os.path.exists(graph_path):
                    os.remove(graph_path)
                    print(f"Deleted temporary full graph: {graph_path}")

if __name__ == "__main__":
    main()
