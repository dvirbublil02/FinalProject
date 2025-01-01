from PIL import Image, ImageDraw, ImageFont
import networkx as nx
import matplotlib.pyplot as plt
import os

# Function to add title to the image with number of edges
def add_title(image, snapshot_number, edge_count, font, y_offset, image_width):
    draw = ImageDraw.Draw(image)
    
    # Create the title text with snapshot number and number of edges
    title = f"Snapshot {snapshot_number} - Number of edges: {edge_count}"
    
    # Get the bounding box of the text
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Position the title at the center of the image
    x_position = (image_width - text_width) // 2
    draw.text((x_position, y_offset), title, font=font, fill="black")

    return image

# Function to create and save snapshots
def create_snapshots(edges, snapshot_size=500, output_folder='snapshots/'):
    num_snapshots = len(edges) // snapshot_size + (1 if len(edges) % snapshot_size != 0 else 0)

    for i in range(num_snapshots):
        # Slice edges for this snapshot
        start_idx = i * snapshot_size
        end_idx = min((i + 1) * snapshot_size, len(edges))
        edges_subset = edges[start_idx:end_idx]
        
        # Create graph from the subset of edges
        G = nx.DiGraph()
        for edge in edges_subset:
            G.add_edge(edge[0], edge[1])
        
        # Draw the graph and save it
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.1, iterations=50)
        nx.draw(G, pos, with_labels=True, node_size=50, node_color="skyblue", font_size=8, font_weight="bold", edge_color="gray")
        
        # Calculate the number of edges in the current snapshot
        edge_count = len(edges_subset)
        
        # Manually adjust layout to prevent overlapping
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Save the snapshot image
        snapshot_path = f'{output_folder}snapshot_{i+1}.png'
        plt.savefig(snapshot_path)
        plt.close()

    print(f"Saved {num_snapshots} snapshots.")

# Function to combine all saved snapshots into a single image with multiple columns
def combine_snapshots(output_folder='snapshots/', num_snapshots=None, combined_output_path='combined_snapshots.png', all_edges=None, columns=2):
    images = []
    for i in range(1, num_snapshots + 1):
        img = Image.open(f'{output_folder}snapshot_{i}.png')
        
        # Calculate the number of edges for this snapshot
        edge_count = len(all_edges[(i-1)*500 : min(i*500, len(all_edges))])
        
        # Create a title image above each snapshot
        title = f"Snapshot {i} - Number of edges: {edge_count}"
        font = ImageFont.load_default()  # Use default font
        title_image = Image.new('RGB', (img.width, 50), color='white')
        
        # Add title to the title image
        title_image = add_title(title_image, i, edge_count, font, 10, img.width)
        
        # Combine title and snapshot
        combined_image = Image.new('RGB', (img.width, img.height + 50))  # Add space for the title
        combined_image.paste(title_image, (0, 0))
        combined_image.paste(img, (0, 50))  # Paste the snapshot below the title
        images.append(combined_image)
    
    # Create the grid layout
    grid_width = img.width * columns
    rows = (len(images) // columns) + (1 if len(images) % columns != 0 else 0)
    total_height = sum([image.height for image in images[:columns]])  # Remove the 50px for titles
    
    # Calculate the total height of all images
    for i in range(rows):
        total_height += images[i*columns].height if i*columns < len(images) else 0
    
    final_image = Image.new('RGB', (grid_width, total_height))
    
    # Place images in the grid
    x_offset = 0
    y_offset = 0
    for i, image in enumerate(images):
        final_image.paste(image, (x_offset, y_offset))
        
        # Move to the next column
        x_offset += image.width
        if (i + 1) % columns == 0:  # Move to the next row after the last column
            x_offset = 0
            y_offset += image.height
    
    # Create and save final snapshot of all edges (this is the final snapshot with all 5800 edges)
    final_graph = nx.DiGraph()
    for edge in all_edges:
        final_graph.add_edge(edge[0], edge[1])

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(final_graph, k=0.1, iterations=50)
    nx.draw(final_graph, pos, with_labels=True, node_size=50, node_color="skyblue", font_size=8, font_weight="bold", edge_color="gray")
    
    # Add title for the final snapshot
    plt.title(f"Snapshot of All Edges - Number of edges: {len(all_edges)}")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Save final snapshot image
    final_snapshot_path = os.path.join(output_folder, 'final_snapshot_all_edges.png')
    plt.savefig(final_snapshot_path)
    plt.close()
    print(f"Final snapshot of all edges saved as '{final_snapshot_path}'.")

    # Now add the final snapshot to the combined image
    final_snapshot_img = Image.open(final_snapshot_path)
    
    # Create title for the final snapshot
    final_title = f"Snapshot of All Edges - Number of edges: {len(all_edges)}"
    final_title_image = Image.new('RGB', (final_snapshot_img.width, 50), color='white')
    final_title_image = add_title(final_title_image, 'All Edges', len(all_edges), ImageFont.load_default(), 10, final_snapshot_img.width)
    
    # Combine title and final snapshot
    combined_final_image = Image.new('RGB', (final_snapshot_img.width, final_snapshot_img.height + 50))
    combined_final_image.paste(final_title_image, (0, 0))
    combined_final_image.paste(final_snapshot_img, (0, 50))
    
    # Add to the final combined image
    final_image.paste(combined_final_image, (0, y_offset))
    
    # Save the final combined image
    final_image.save(combined_output_path)
    print(f"Combined snapshots with final snapshot saved as '{combined_output_path}'.")

# Function to parse the uci dataset with two columns
def parse_edges(file_path):
    edges = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Skip comments starting with '%'
            if line.startswith('%'):
                continue
            
            parts = line.strip().split()
            if len(parts) == 2:  # Ensure there are 2 elements in the line (article_id and citing_article_id)
                try:
                    article_id, citing_article_id = map(int, parts)
                    edges.append((article_id, citing_article_id))
                except ValueError as e:
                    print(f"Error parsing line: {line}. Error: {e}")
    return edges

# Main function to run the entire process
def main():
    input_folder = 'results/snapshots/train/'
    output_folder = 'snapshots/combined/'
    
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder and process them
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)
            
            # Parse the edges for the current file
            edges = parse_edges(file_path)
            print(f"Processing file: {file_name} with {len(edges)} edges.")  # Now edges is properly defined
            
            # Create snapshots every 500 edges
            create_snapshots(edges, snapshot_size=500, output_folder=output_folder)

            # After all snapshots are created, combine them
            num_snapshots = len(edges) // 500 + (1 if len(edges) % 500 != 0 else 0)
            combined_output_path = os.path.join(output_folder, f'combined_{file_name}.png')
            combine_snapshots(output_folder=output_folder, num_snapshots=num_snapshots, combined_output_path=combined_output_path, all_edges=edges, columns=3)
            print(f"Completed combining snapshots for {file_name}.")

if __name__ == "__main__":
    main()
