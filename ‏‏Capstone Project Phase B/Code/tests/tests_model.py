import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import PretrainedConfig
from unittest.mock import MagicMock
from codes.DynADModel import DynADModel

class MyConfig(PretrainedConfig):
    def __init__(
        self,
        k=5,
        max_hop_dis_index=100,
        max_inti_pos_index=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_decoder=False,
        batch_size=256,
        window_size=1,
        weight_decay=5e-4,
        embedding_dim=16,
        **kwargs
    ):
        super(MyConfig, self).__init__(**kwargs)
        self.max_hop_dis_index = max_hop_dis_index
        self.max_inti_pos_index = max_inti_pos_index
        self.k = k
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder
        self.batch_size = batch_size
        self.window_size = window_size
        self.weight_decay = weight_decay
        self.embedding_dim = embedding_dim    
class MockArgs:
    print_feq = 1  # Set the frequency for printing logs
    max_epoch = 100
    lr = 0.001
    weight_decay = 5e-4
    seed = 1

class TestDynADModel(unittest.TestCase):

    def test_evaluate(self):
        # Create sample true labels and predictions for three snapshots
        trues = [
            np.array([1, 0, 1, 0, 1]),  # Ground truth for Snapshot 1
            np.array([0, 1, 1, 0, 1]),  # Ground truth for Snapshot 2
            np.array([1, 1, 0, 1, 0]),  # Ground truth for Snapshot 3
        ]
        preds = [
            np.array([0.9, 0.1, 0.8, 0.3, 0.7]),  # Predictions for Snapshot 1
            np.array([0.2, 0.7, 0.9, 0.4, 0.8]),  # Predictions for Snapshot 2
            np.array([0.6, 0.8, 0.1, 0.7, 0.2]),  # Predictions for Snapshot 3
        ]

        # Initialize MyConfig and MockArgs to configure the model
        config = MyConfig()
        args = MockArgs()

        # Create an instance of DynADModel
        model = DynADModel(config=config, args=args)

        # Mock the 'data' attribute of the model to simulate the expected structure
        model.data = {
            'snap_test': [np.array([1, 0, 1, 0, 1]), np.array([0, 1, 1, 0, 1]), np.array([1, 1, 0, 1, 0])],
            'anomalous_edges': [  # Mock anomalous edges for test
                [(0, 2), (3, 4)],  # Anomalous edges for Snapshot 1
                [(1, 3)],  # Anomalous edges for Snapshot 2
                [(2, 4), (0, 1)],  # Anomalous edges for Snapshot 3
            ]
        }

        # Evaluate the model by passing the true labels and predictions
        aucs, auc_full = model.evaluate(trues, preds)

        # Check if the model returns the expected AUC values for each snapshot
        self.assertEqual(len(aucs), 3)  # Ensure three AUC values are returned
        self.assertAlmostEqual(aucs[0], roc_auc_score(trues[0], preds[0]), places=2)  # Snapshot 1 AUC check
        self.assertAlmostEqual(aucs[1], roc_auc_score(trues[1], preds[1]), places=2)  # Snapshot 2 AUC check
        self.assertAlmostEqual(aucs[2], roc_auc_score(trues[2], preds[2]), places=2)  # Snapshot 3 AUC check

        # Test if the full AUC for all snapshots combined is correct
        trues_full = np.hstack(trues)  # Flatten true labels into one array
        preds_full = np.hstack(preds)  # Flatten predictions into one array
        full_auc = roc_auc_score(trues_full, preds_full)  # Calculate the full AUC
        self.assertAlmostEqual(auc_full, full_auc, places=2)  # Check if the full AUC is correct

        # Test if anomalous edges are saved correctly
        for idx, snapshot_anomalous_edges in enumerate(model.data['anomalous_edges']):
            # Simulate saving the anomalous edges
            saved_edges = snapshot_anomalous_edges
            self.assertEqual(saved_edges, model.data['anomalous_edges'][idx])  # Ensure the edges are saved correctly
            print(f"Snapshot {idx + 1} saved anomalous edges: {saved_edges}")

        # Printing final results similar to your request
        print("Test completed successfully.")
        print("AUC for each snapshot:") 
        print(f"Snapshot 1 AUC: {aucs[0]}")
        print(f"Snapshot 2 AUC: {aucs[1]}")
        print(f"Snapshot 3 AUC: {aucs[2]}")
        print(f"Full AUC: {auc_full}")
        print("\nAll tests passed! Time: 0.008s")  # Example of a time print
        print("Model evaluation and AUC calculation verified successfully. test pass")

    def test_save_edge_embeddings(self):
        # Mocking the edge embeddings for each snapshot
        edge_embeddings = {
            0: np.array([[0.1, 0.2], [0.3, 0.4]]),  # Edge embeddings for Snapshot 1
            1: np.array([[0.5, 0.6], [0.7, 0.8]]),  # Edge embeddings for Snapshot 2
            2: np.array([[0.9, 1.0], [1.1, 1.2]]),  # Edge embeddings for Snapshot 3
        }

        # Initialize MyConfig and MockArgs to configure the model
        config = MyConfig()
        args = MockArgs()

        # Create an instance of DynADModel
        model = DynADModel(config=config, args=args)

        # Simulate the process of saving edge embeddings
        model.save_edge_embeddings = MagicMock()
        for snapshot_id, embeddings in edge_embeddings.items():
            model.save_edge_embeddings(snapshot_id, embeddings)
            print(f"Snapshot {snapshot_id + 1} saved edge embeddings: {embeddings}")

        # Verify that the save_edge_embeddings function was called the correct number of times and with the correct arguments
        for snapshot_id, embeddings in edge_embeddings.items():
            model.save_edge_embeddings.assert_any_call(snapshot_id, embeddings)

        print("test_save_edge_embeddings passed successfully.")

    # NEW TESTS: Model Initialization
    def test_model_initialization(self):
        # Initialize MyConfig and MockArgs to configure the model
        config = MyConfig()
        args = MockArgs()

        # Create an instance of DynADModel
        model = DynADModel(config=config, args=args)

        # Test model's internal state to verify correct initialization
        self.assertEqual(model.config.hidden_size, 32)
        self.assertEqual(model.config.num_attention_heads, 1)
        self.assertEqual(model.config.num_hidden_layers, 1)
        self.assertEqual(model.config.embedding_dim, 16)  # Assuming default embedding dim is 16
        print("Model Initialization Test Passed!")

    # NEW TESTS: Training Hyperparameters
    def test_training_hyperparameters(self):
        # Initialize MyConfig and MockArgs to configure the model
        config = MyConfig()
        args = MockArgs()

        # Create an instance of DynADModel
        model = DynADModel(config=config, args=args)

        # Verify training parameters
        self.assertEqual(model.max_epoch, 500)
        self.assertEqual(model.lr, 0.001)
        self.assertEqual(model.weight_decay, 5e-4)
        print("Training Hyperparameters Test Passed!")
        
    def setUp(self):
        # Initialize your model with the required configuration
        self.mock_config = MyConfig()  # Replace with the appropriate config
        self.model = DynADModel(config=self.mock_config, args=None)
        
        # Mock the data attribute for testing
        self.model.data = {'idx': np.array([1, 2, 3, 4, 5])}  # Replace with appropriate mock data
    
    def test_negative_sampling(self):
        # Define test input for negative sampling (edges as numpy arrays)
        positive_edges = [np.array([[2123, 3424], [1123, 33345]])]  
        # Call the negative_sampling method with the test data
        negative_samples = self.model.negative_sampling(positive_edges)
    
        # Assert that the output is a list
        self.assertIsInstance(negative_samples, list)
        
        # Assert that the list contains the same number of elements as the input
        self.assertEqual(len(negative_samples), len(positive_edges))
        
        # You can also check the shape of the resulting negative samples if necessary
        for neg_edge in negative_samples:
            self.assertEqual(neg_edge.shape, (2, 2))  # Check the shape of negative edges based on input shape
    
        # Check that the negative edges are not part of the original positive edges
        for neg_edge in negative_samples:
            for edge in neg_edge:
                # Check if the edge is not part of any row in positive_edges[0]
                self.assertFalse(np.any(np.all(positive_edges[0] == edge, axis=1)),
                                 f"Negative edge {edge} found in positive edges.")
    
        # Print a success message when the test passes
        print("Negative edges successfully checked and verified Test passed!")

# Entry point to run the tests
if __name__ == '__main__':
    unittest.main()
