Dataset Preparation for TADDY Framework

This document outlines the steps and logic behind the dataset preparation for the TADDY framework. The preparation was performed on 15.11, and the resulting preprocessed dataset is located in data/raw. The preprocessing steps were executed only once to create a ready-to-use dataset. This code is retained for reference and can be re-executed if the dataset is deleted or requires updates.

Overview of the Process

The preparation process involves the following stages:

Downloading the PubMed Dataset

Fetch the PubMed dataset and retrieve the article titles and publication dates using an external API.

Output: article_titles_with_date.txt in data/OurResearch.

Handling Missing Titles and Dates

Retry fetching titles and dates for articles where this information is missing.

Extracting Dates

Extract article IDs and their corresponding dates into a separate file for quick access.

Preprocessing and Formatting

Organize and preprocess the raw data into the format required by the TADDY framework.

Attach timestamps to edges based on the extracted publication dates.

Output: preprocessed_pubmed in data/raw.

The final preprocessed dataset includes data spanning from 1945 to 1977 and is fully compatible with the TADDY framework.

File Descriptions

1. download_pubmed_raw

Purpose:

Downloads the PubMed dataset.

Fetches article titles and publication dates using an external API.

Output:

A text file named article_titles_with_date located in data/OurResearch. This file contains the article IDs, titles, and publication dates.

2. download_pubmed_and_generate_titles_specific_for_missing_titles

Purpose:

Handles missing data by retrying the fetch process for specific articles.

This code is manually executed for a specified range of article IDs.

Use Case:

To fill in gaps for articles that failed to retrieve their titles or dates in the first step.

3. extract_dates_before_creating_raw

Purpose:

Extracts article IDs and their corresponding dates into a simplified and easily readable file.

Output:

A file in data/OurResearch containing article IDs and their dates for faster lookup.

4. preprocessing_pubmed_generating_raw

Purpose:

Prepares the dataset for the TADDY framework by:

Formatting raw data.

Utilizing previously extracted titles and dates.

Generating timestamps for all edges based on publication dates.

Output:

A file named preprocessed_pubmed located in data/raw.

Important Notes

Run Frequency: The codes in this folder are not intended for repeated execution unless the dataset changes or is deleted.

Current Dataset: The preprocessed file preprocessed_pubmed was created on 15.11 and contains data from 1947 to 1977.

Location: The preprocessed file is located in data/raw and is ready for use with the TADDY framework.

Manual Execution: Some scripts, such as download_pubmed_and_generate_titles_specific_for_missing_titles, require manual intervention to specify the range of article IDs.

Conclusion

The dataset preparation workflow ensures the PubMed dataset is properly formatted, enriched with timestamps, and compatible with the TADDY framework. These steps provide a solid foundation for conducting anomaly detection on dynamic graphs while retaining the flexibility to regenerate the dataset if needed.


****************************************

if you want to run this codes in the order i mention above . put them in the main folder because of the path's in the files.

****************************************