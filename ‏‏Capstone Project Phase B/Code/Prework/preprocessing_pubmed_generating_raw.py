import torch
from torch_geometric.datasets import Planetoid
import os
import pandas as pd
from datetime import datetime

# Function to load article dates (only year)
def load_article_dates(filename):
    articles = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(maxsplit=1)  # Split into ID and date
        if len(parts) < 2:
            continue  # Skip malformed lines
        article_id = int(parts[0])  # Article ID
        year_str = parts[1][:4]  # Extract only the year (first 4 digits)

        try:
            # Convert the year into a timestamp for January 1st of that year
            year_date = datetime.strptime(f"{year_str}-01-01", "%Y-%m-%d")
            # Convert date to a Unix timestamp (seconds since Unix epoch)
            timestamp = int((year_date - datetime(1970, 1, 1)).total_seconds())
            articles[article_id] = timestamp
        except ValueError:
            print(f"Error parsing date for article ID {article_id}: {year_str}")
            continue  # Skip lines with invalid date formats

    print(f"Loaded {len(articles)} articles with years.")
    return articles

# Function to infer timestamp for articles with missing years
def infer_missing_timestamp(article_id, pub_dates, sorted_ids, year_counter):
    # If the article has a known date, return it
    if article_id in pub_dates:
        return pub_dates[article_id]

    # Otherwise, look at neighboring articles to infer the date
    idx = sorted_ids.index(article_id) if article_id in sorted_ids else None
    prev_date = None
    next_date = None

    # Look at neighbors to infer date
    if idx is not None:
        if idx > 0 and sorted_ids[idx - 1] in pub_dates:
            prev_date = pub_dates[sorted_ids[idx - 1]]
        if idx < len(sorted_ids) - 1 and sorted_ids[idx + 1] in pub_dates:
            next_date = pub_dates[sorted_ids[idx + 1]]

    # Inference logic: If both previous and next dates are available, choose the closer one
    if prev_date and next_date:
        if abs(article_id - sorted_ids[idx - 1]) < abs(article_id - sorted_ids[idx + 1]):
            return prev_date
        else:
            return next_date
    elif prev_date:
        return prev_date
    elif next_date:
        return next_date
    else:
        # If no date available, assign a default year-based timestamp
        if year_counter:
            last_year = year_counter.get('last_year', 1975)  # Default to 1975 if no history
            last_timestamp = year_counter.get(last_year, int((datetime(1975, 1, 1) - datetime(1970, 1, 1)).total_seconds()))
            # Increment the timestamp for this missing article within the year
            new_timestamp = last_timestamp + 1
            year_counter[last_year] = new_timestamp
            year_counter['last_year'] = last_year
            return new_timestamp
        else:
            return int((datetime(1975, 1, 1) - datetime(1970, 1, 1)).total_seconds())  # Start with 1975 if no data

# Function to process the dataset and generate the output file
def process_dataset(dataset_dir='data/OurResearch', output_dir='data/raw', output_file='processed_pubmed.txt', publication_file='article_titles_with_date.txt'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the PubMed dataset
    dataset = Planetoid(root=dataset_dir, name='PubMed')
    data = dataset[0]

    # Load article publication years from the file
    pub_dates = load_article_dates(publication_file)
    sorted_ids = sorted(pub_dates.keys())  # Sort by article ID

    # Prepare data for the output file
    file_data = []
    year_counter = {}  # Dictionary to count articles in the same year

    print(f"Processing {data.edge_index.shape[1]} edges...")

    for i in range(data.edge_index.shape[1]):  # Iterate over edges
        citing_id = data.edge_index[0, i].item()  # Citing paper (source)
        cited_id = data.edge_index[1, i].item()  # Cited paper (target)

        # Get the timestamp of the citing article
        citing_timestamp = infer_missing_timestamp(citing_id, pub_dates, sorted_ids, year_counter)

        if citing_timestamp:
            citing_year = datetime.utcfromtimestamp(citing_timestamp).year
            if citing_year not in year_counter:
                year_counter[citing_year] = citing_timestamp
            # Increment timestamp for articles published in the same year
            unique_timestamp = citing_timestamp

            # Add edge with the unique timestamp of the citing article
            file_data.append([citing_id, cited_id, 1, unique_timestamp])
        else:
            print(f"Skipping edge {citing_id} -> {cited_id} because citing article {citing_id} has no valid date. Assigning default timestamp.")
            # Here, we can assign a default timestamp based on the cited article's year or the earliest date
            default_timestamp = 0  # Replace with a strategy for defaulting if necessary
            file_data.append([citing_id, cited_id, 1, default_timestamp])

    # Sort the edges by the timestamp of the citing article
    file_data.sort(key=lambda x: x[3])  # Sorting by timestamp (the 4th element in each entry)

    print(f"Processed {len(file_data)} edges.")

    # Save the data to the specified output file
    df = pd.DataFrame(file_data, columns=['citing_id', 'cited_id', 'citation_label', 'timestamp'])
    df.to_csv(os.path.join(output_dir, output_file), sep='\t', index=False)

    print(f"Output file saved in {output_dir}: {output_file}")

# Main Function to process the dataset and generate the file
def main():
    process_dataset(dataset_dir='data/OurResearch', output_dir='data/raw', output_file='processed_pubmed.txt', publication_file='data/OurResearch/article_titles_with_date.txt')

# Run the main function
if __name__ == '__main__':
    main()
