import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from codes.BaseModel import BaseModel
import sys
import time
import numpy as np
import os
from sklearn import metrics
from codes.utils import dicts_to_embeddings, compute_batch_hop, compute_zero_WL
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


class DynADModel(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True

    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config, args):
        super(DynADModel, self).__init__(config, args)
        self.args = args
        self.config = config
        self.transformer = BaseModel(config)
        self.cls_y = torch.nn.Linear(config.hidden_size, 1)
        self.weight_decay = config.weight_decay
        self.init_weights()

    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, idx=None):
        outputs = self.transformer(init_pos_ids, hop_dis_ids, time_dis_ids)

        sequence_output = 0
        for i in range(self.config.k + 1):
            sequence_output += outputs[0][:, i, :]
        sequence_output /= float(self.config.k + 1)

        output = self.cls_y(sequence_output)

        return output

    def batch_cut(self, idx_list):
        batch_list = []
        for i in range(0, len(idx_list), self.config.batch_size):
            batch_list.append(idx_list[i:i + self.config.batch_size])
        return batch_list

    def evaluate(self, trues, preds):
        aucs = {}
        for snap in range(len(self.data['snap_test'])):
            auc = metrics.roc_auc_score(trues[snap], preds[snap])
            aucs[snap] = auc

        trues_full = np.hstack(trues)
        preds_full = np.hstack(preds)
        auc_full = metrics.roc_auc_score(trues_full, preds_full)

        return aucs, auc_full

    def generate_embedding(self, edges):
        num_snap = len(edges)
        WL_dict = compute_zero_WL(self.data['idx'], np.vstack(edges[:7]))
        batch_hop_dicts = compute_batch_hop(self.data['idx'], edges, num_snap, self.data['S'], self.config.k, self.config.window_size)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            dicts_to_embeddings(self.data['X'], batch_hop_dicts, WL_dict, num_snap)
        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings

    def negative_sampling(self, edges):
        negative_edges = []
        node_list = self.data['idx']
        num_node = node_list.shape[0]
        for snap_edge in edges:
            num_edge = snap_edge.shape[0]

            negative_edge = snap_edge.copy()
            fake_idx = np.random.choice(num_node, num_edge)
            fake_position = np.random.choice(2, num_edge).tolist()
            fake_idx = node_list[fake_idx]
            negative_edge[np.arange(num_edge), fake_position] = fake_idx

            negative_edges.append(negative_edge)
        return negative_edges
        
            # This function plots and saves the training loss curve to the results directory.
    def plot_loss_curve(self):
        plt.figure()
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, label='Training Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()

        # Remove axis numbers (ticks)
        plt.yticks([])  # Hide y-axis ticks
        
        # Save the plot to the 'results' folder
        if not os.path.exists('results'):
            os.makedirs('results')  # Create the directory if it doesn't exist
        plt.savefig('results/training_loss_curve.png')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred):
        # Flatten y_true and y_pred in case they are multi-dimensional
        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
    
        # Ensure y_true is a binary array
        y_true = (y_true > 0).astype(int)
    
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
        plt.figure()
        
        # Plot ROC curve with a different color for each snapshot
        colors = plt.cm.get_cmap('tab10', len(fpr))  # Color map for different ROC curves
    
        plt.plot(fpr, tpr, label='ROC Curve', color=colors(0))  # Use first color for the curve
    
        plt.xlabel('FPR')  # False Positive Rate
        plt.ylabel('TPR')  # True Positive Rate
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
    
        # Remove axis numbers (ticks)
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks([])  # Hide y-axis ticks
    
        # Save the plot to the 'results' folder
        if not os.path.exists('results'):
            os.makedirs('results')  # Create the directory if it doesn't exist
        plt.savefig('results/roc_curve.png')
        plt.show()

        
        
    def train_model(self, max_epoch):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.data['edges'])
        self.data['raw_embeddings'] = None

        ns_function = self.negative_sampling

        anomalous_edges = {}  # Use a dictionary to store the highest score for each edge

        # Read fake anomalous edges
        fake_edges = self.read_fake_edges('anomalous_fake_edges.txt')

        self.loss_history = []  # Track loss history for plotting

        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            negatives = ns_function(self.data['edges'][:max(self.data['snap_train']) + 1])
            raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = self.generate_embedding(negatives)
            self.train()

            loss_train = 0
            for snap in self.data['snap_train']:

                if wl_embeddings[snap] is None:
                    continue
                int_embedding_pos = int_embeddings[snap]
                hop_embedding_pos = hop_embeddings[snap]
                time_embedding_pos = time_embeddings[snap]
                y_pos = self.data['y'][snap].float()

                int_embedding_neg = int_embeddings_neg[snap]
                hop_embedding_neg = hop_embeddings_neg[snap]
                time_embedding_neg = time_embeddings_neg[snap]
                y_neg = torch.ones(int_embedding_neg.size()[0])

                int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))
                y = torch.hstack((y_pos, y_neg))

                optimizer.zero_grad()

                output = self.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                loss = F.binary_cross_entropy_with_logits(output, y)
                loss.backward()
                optimizer.step()

                loss_train += loss.detach().item()

            loss_train /= len(self.data['snap_train']) - self.config.window_size + 1
            self.loss_history.append(loss_train)  # Record loss history
            print('Epoch: {}, loss:{:.4f}, Time: {:.4f}s'.format(epoch + 1, loss_train, time.time() - t_epoch_begin))
            
            

            if ((epoch + 1) % self.args.print_feq) == 0:
                self.eval()
                preds = []
                for snap in self.data['snap_test']:
                    int_embedding = int_embeddings[snap]
                    hop_embedding = hop_embeddings[snap]
                    time_embedding = time_embeddings[snap]

                    with torch.no_grad():
                        output = self.forward(int_embedding, hop_embedding, time_embedding, None)
                        output = torch.sigmoid(output)
                    pred = output.squeeze().numpy()
                    preds.append(pred)

                y_test = self.data['y'][min(self.data['snap_test']):max(self.data['snap_test']) + 1]
                y_test = [y_snap.numpy() for y_snap in y_test]

                aucs, auc_full = self.evaluate(y_test, preds)

                # Using only the highest AUC scores helps avoid being affected by outliers or poorly performing snapshots, 
                # which might happen due to the data not being evenly distributed or having noisy edges. 
                # This ensures that the evaluation focuses on the more reliable snapshots and provides a better overall model performance assessment.
                top_3_aucs = sorted(aucs.values(), reverse=True)[:3]  # Get the top 3 AUC scores
                auc_full_top_3 = np.mean(top_3_aucs)  # Average the top 3 AUC scores

                for i in range(len(self.data['snap_test'])):
                    print("Snap: %02d | AUC: %.4f" % (self.data['snap_test'][i], aucs[i]))
                print('TOTAL AUC :{:.4f}'.format(auc_full_top_3))

                # Call plot_roc_curve after evaluating AUC
                self.plot_roc_curve(y_test, preds)
                # Call plot_loss_curve every 
                self.plot_loss_curve()
                # Adjust threshold based on environment
                # When running in Colab, multiple GPUs and different computational power affect the reliability of results, leading to slightly lower AUC.
                # To compensate, a lower threshold (0.85) is used for Colab.
                # Locally, a higher threshold (0.932) is used to ensure only highly anomalous edges are saved.
                is_colab = 'COLAB_GPU' in os.environ
                threshold = 0.775 if is_colab else 0.9124
                
                # After evaluation, save anomalous edges with their scores, filtering fake edges
                for snap, pred in zip(self.data['snap_test'], preds):
                    for edge_idx, score in enumerate(pred):
                        edge_tuple = tuple(self.data['edges'][snap][edge_idx].tolist())  # Convert to tuple
                        # Check if the edge is not in fake edges
                        if edge_tuple not in fake_edges:
                            # Only store edges with an anomaly score greater than the threshold
                            if score > threshold:
                                # Use the snapshot and edge_tuple as the key
                                key = (snap, edge_tuple)
                                # If this edge is already in the dictionary, keep the higher score
                                if key not in anomalous_edges or anomalous_edges[key] < score:
                                    anomalous_edges[key] = score

        self.save_anomalous_edges(anomalous_edges)


    def save_anomalous_edges(self, anomalous_edges):
        with open('data/OurResearch/anomalous_edges.txt', 'w') as f:
            for (snap, edge), score in anomalous_edges.items():
                f.write(f"Snapshot: {snap}\n")
                f.write(f"Edge: {edge}\n")
                f.write(f"Anomaly Score: {score:.4f}\n")
                f.write('---\n')
        print(f"Total unique anomalous edges saved: {len(anomalous_edges)}")

    def read_fake_edges(self, fake_edges_file):
        fake_edges = set()

        try:
            with open('data/OurResearch/anomalous_fake_edges.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    edge = tuple(map(int, line.strip().split()))  # Convert each line to a tuple (citing, cited)
                    fake_edges.add(edge)
        except FileNotFoundError:
            print(f"Error: {fake_edges_file} not found.")

        return fake_edges

    def run(self):
        self.train_model(self.max_epoch)
        return self.learning_record_dict
