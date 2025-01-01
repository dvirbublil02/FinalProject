import os

def save_snapshots(dataset, train_size, test_size, snap_size, train, synthetic_test, results_dir, anomaly_per):
    """
    Saves snapshots (edges) for training and testing data to separate folders.

    Args:
        dataset: Name of the dataset being processed.
        train_size: Number of training snapshots.
        test_size: Number of testing snapshots.
        snap_size: Number of edges in each snapshot.
        train: List of edges for training data.
        synthetic_test: List of edges for synthetic testing data.
        results_dir: Base directory to save results (results/).
        anomaly_per: Anomaly percentage.
    """

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Create subdirectories for snapshots (snapshots/)
    train_dir = os.path.join(results_dir, "snapshots", "train")
    test_dir = os.path.join(results_dir, "snapshots", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Extract individual snapshots and save them as source-target pairs
    for i in range(train_size):
        start_loc = i * snap_size
        end_loc = (i + 1) * snap_size

        snapshot_edges = train[start_loc:end_loc]

        snapshot_filename = os.path.join(train_dir, f"{dataset}_{i:04d}_train_{anomaly_per:.2f}.txt")

        print(f"Saving training snapshot: {snapshot_filename}")  # Add debug print

        with open(snapshot_filename, 'w') as f:
            for edge in snapshot_edges:
                try:
                    source, target = edge[:2]  # Unpack the first two elements
                    f.write(f"{source} {target}\n")
                except ValueError:
                    print(f"Warning: Edge {edge} has an incorrect format, skipping.")

    for i in range(test_size):
        start_loc = i * snap_size
        end_loc = (i + 1) * snap_size

        snapshot_edges = synthetic_test[start_loc:end_loc]

        snapshot_filename = os.path.join(test_dir, f"{dataset}_{i:04d}_test_{anomaly_per:.2f}.txt")

        print(f"Saving testing snapshot: {snapshot_filename}")  # Add debug print

        with open(snapshot_filename, 'w') as f:
            for edge in snapshot_edges:
                try:
                    source, target = edge[:2]  # Unpack the first two elements
                    f.write(f"{source} {target}\n")
                except ValueError:
                    print(f"Warning: Edge {edge} has an incorrect format, skipping.")