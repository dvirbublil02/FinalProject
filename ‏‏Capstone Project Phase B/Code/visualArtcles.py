import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ipywidgets as widgets
from IPython.display import display,clear_output

# Detect if running in Google Colab
try:
    from google.colab import output
    COLAB = True
except ImportError:
    COLAB = False

# Function to parse the anomaly status matrix file and extract relevant data
def parse_anomaly_matrix(file_path):
    articles = defaultdict(list)
    years = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Extract years from the header row
        header = lines[1].strip().split()
        years = header[1:]

        # Process each line to extract article data
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < len(years) + 2:
                continue  # Skip lines that are not properly formatted
            node_id = parts[0]
            publication_year = parts[1].strip("()")
            statuses = parts[2:]
            articles[node_id] = (publication_year, statuses)

    return years, articles

# Function to parse the article titles file and match titles with IDs
def parse_article_titles(file_path):
    articles = {}
    # Define the format for extracting IDs, titles, and years
    pattern = r"(\d+)\s+(.*?)\s+(\d{4})"

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.match(pattern, line)
            if match:
                article_id = match.group(1)
                title = match.group(2).strip()
                articles[article_id] = f"{title}"
    return articles

# Function to plot the anomaly status over time and embed the plot into Tkinter
def plot_time_series(article_id, years, statuses, publication_year, frame):
    fig, ax = plt.subplots(figsize=(8, 4))
    # Create a binary plot to show anomaly status over time
    ax.plot(years, [1 if status == "Anomalous" else 0 for status in statuses], marker='o', label=article_id)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly Status (0=Normal, 1=Anomalous)")
    ax.set_title(f"Anomaly Time Series for Article {article_id}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Embed the plot into the Tkinter application
    canvas = FigureCanvasTkAgg(fig, master=frame)  # Create canvas for the plot
    canvas.draw()  # Render the plot
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Add the canvas to the Tkinter frame

# Function to plot the anomaly status over time for Colab (directly in notebook)
def plot_time_series_colab(article_id, years, statuses, publication_year):
    # Clear the previous output (plot)
    clear_output(wait=True)

    # Create a new plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(years, [1 if status == "Anomalous" else 0 for status in statuses], marker='o', label=article_id)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly Status (0=Normal, 1=Anomalous)")
    ax.set_title(f"Anomaly Time Series for Article {article_id}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Display the new plot in the same output cell
    plt.show()

# Function to display the widget-based UI and plot in Colab
def display_colab_ui(years, articles, article_titles):
    # Dropdown for selecting an article
    article_options = [(f"{article_id} - {article_titles.get(article_id, 'Unknown Title')}", article_id) for article_id in articles]
    article_dropdown = widgets.Dropdown(
        options=article_options,
        description='Select Article:',
        disabled=False,
    )

    # Function to update plot when an article is selected
    def on_article_select(change):
        # Clear the previous plot (but not the dropdown)
        clear_output(wait=True)
        
        article_id = change.new
        if article_id in articles:
            publication_year, statuses = articles[article_id]
            plot_time_series_colab(article_id, years, statuses, publication_year)
        
        # Re-display the dropdown after updating the plot
        display(article_dropdown)

    # Link dropdown to the plot function
    article_dropdown.observe(on_article_select, names='value')

    # Display the dropdown widget
    display(article_dropdown)


# Main function to set up the Tkinter GUI for local use
def create_gui(years, articles, article_titles):
    def on_select():
        try:
            # Get the selected article from the listbox
            selected_index = article_listbox.curselection()[0]
            selected_article = article_listbox.get(selected_index)
            article_id = selected_article.split()[0]

            # Check if the article exists in the data and display its plot
            if article_id in articles:
                publication_year, statuses = articles[article_id]
                # Clear any existing plot in the frame
                for widget in plot_frame.winfo_children():
                    widget.destroy()
                # Generate the new plot for the selected article
                plot_time_series(article_id, years, statuses, publication_year, plot_frame)
            else:
                messagebox.showwarning("Data Missing", f"No anomaly data available for article: {selected_article}")
        except IndexError:
            messagebox.showwarning("Selection Error", "Please select an article from the list.")

    # Initialize the main Tkinter window
    root = tk.Tk()
    root.title("Anomaly Time Series Viewer")
    root.geometry("800x600")  # Set the window size

    # Add an instruction label
    instruction_label = tk.Label(root, text="Select an article to view its anomaly time series:")
    instruction_label.pack(pady=5)

    # Create a listbox to display articles
    article_listbox = tk.Listbox(root, height=15, width=80)
    article_listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Populate the listbox with article data
    for article_id, (publication_year, _) in articles.items():
        article_name = article_titles.get(article_id, "Unknown Title")
        article_listbox.insert(tk.END, f"{article_id} - {article_name} ")

    # Add a button to select an article
    select_button = tk.Button(root, text="Select", command=on_select)
    select_button.pack(pady=5)

    # Create a frame to hold the plot
    plot_frame = tk.Frame(root)
    plot_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Start the Tkinter main event loop
    root.mainloop()

# Main script entry point
if __name__ == "__main__":
    # File paths for anomaly data and article titles (adjust the paths for your environment)
    anomaly_file = "results/anomaly_status_matrix.txt"  # Adjust to the correct path
    titles_file = "data/OurResearch/article_titles_with_date.txt"  # Adjust to the correct path

    # Parse the input files
    years, articles = parse_anomaly_matrix(anomaly_file)
    article_titles = parse_article_titles(titles_file)

    # If running in Google Colab, display the interactive UI
    if COLAB:
        print("Running in Google Colab. Interactive UI using ipywidgets.")
        display_colab_ui(years, articles, article_titles)
    else:
        # Launch the Tkinter GUI application in a local environment like Spyder or PyCharm
        create_gui(years, articles, article_titles)
