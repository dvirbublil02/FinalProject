# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:08:06 2024

@author: dvirb
"""

import re

# Function to extract article ID and date
def extract_article_dates(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # List to hold the extracted article ID and date
    extracted_data = []

    # Regular expression to capture the ID and the date (ignore the title)
    for line in lines:
        # Skip lines with errors
        if "Error fetching title" in line:
            continue
        
        # Use regular expression to capture article ID and date at the end
        match = re.match(r"(\d+)\s+(.*?)\s+(\d{4}(?:\s+[A-Za-z]{3,9}(?:\s+\d{1,2})?)?)$", line.strip())
        if match:
            article_id = match.group(1)  # Article ID
            date = match.group(3)  # Publication date
            extracted_data.append((article_id, date))

    # Write the extracted data to a new file
    with open(output_file, 'w') as outfile:
        for article_id, date in extracted_data:
            outfile.write(f"{article_id}\t{date}\n")

    print(f"Extracted data saved to {output_file}")

# Specify your input and output files
input_file = 'data/OurResearch/article_titles_with_date.txt'  # original
output_file = 'data/OurResearch/extracted_article_dates.txt'  # Output file with article ID and date

# Run the extraction function
extract_article_dates(input_file, output_file)
