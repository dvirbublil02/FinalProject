# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:36:34 2024

@author: dvirb
"""

import csv
import torch
def evaluate_anomalous_articles(self, int_embeddings, hop_embeddings, time_embeddings, output_file, threshold=0.5):
    """
    Evaluate anomalies and save anomalous edges and articles to a file.
    """
    self.eval()
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Snapshot", "Citing Node", "Cited Node", "Anomaly Score", "Anomaly Type"])

        for snap in self.data['snap_test']:
            int_embedding = int_embeddings[snap]
            hop_embedding = hop_embeddings[snap]
            time_embedding = time_embeddings[snap]

            with torch.no_grad():
                output = self.forward(int_embedding, hop_embedding, time_embedding, None)
                probabilities = torch.sigmoid(output).squeeze()

            # Threshold to classify anomalies
            anomaly_mask = probabilities > threshold
            anomalous_ids = torch.nonzero(anomaly_mask).squeeze().tolist()

            # Map back to actual IDs and save to file
            edge_ids = self.data['edges'][snap]
            for idx in anomalous_ids:
                citing_node, cited_node = edge_ids[idx]
                score = probabilities[idx].item()

                # Determine anomaly type (basic heuristic based on degree)
                citing_degree = self.data['degree'][snap][citing_node]
                cited_degree = self.data['degree'][snap][cited_node]
                anomaly_type = "Citing Node" if citing_degree > cited_degree else "Cited Node"

                writer.writerow([snap, citing_node, cited_node, score, anomaly_type])
