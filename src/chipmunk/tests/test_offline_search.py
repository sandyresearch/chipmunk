import unittest
import torch
import sys
import os
import tempfile
import json

# Add parent directory to path for imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chipmunk.search.offline_search import OfflineLayerSearch

class TestOfflineSearch(unittest.TestCase):
    def setUp(self):
        # Create temporary file for output
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        # Initialize offline search
        self.num_layers = 5
        self.offline_search = OfflineLayerSearch(
            num_layers=self.num_layers,
            output_path=self.temp_file.name
        )
        self.offline_search.enabled = True
        
        # Create simple dense and sparse attention functions
        self.batch_size = 2
        self.seq_len = 16
        self.hidden_dim = 32
        self.heads = 4
        
        # Dense attention with controlled error per layer
        def dense_attn(layer):
            return torch.ones(self.batch_size, self.seq_len, self.hidden_dim) * (layer + 1)
        
        # Sparse attention with controlled error per layer
        def sparse_attn(q, k, v, layer):
            # Add increasing error for each layer
            error_factor = 0.1 * (self.num_layers - layer)
            return torch.ones(self.batch_size, self.seq_len, self.hidden_dim) * (layer + 1) * (1 + error_factor)
        
        self.dense_attn = dense_attn
        self.sparse_attn = sparse_attn
        
    def tearDown(self):
        # Clean up temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_offline_search_functionality(self):
        # Create dummy tensors
        q = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        k = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        v = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        # Simulate multiple forward passes
        num_samples = 3
        
        for sample in range(num_samples):
            # Reset layer counter at the beginning of each forward pass
            self.offline_search.reset_layer_counter()
            
            # Process each layer
            for layer in range(self.num_layers):
                # Get dense output
                dense_output = self.dense_attn(layer)
                self.offline_search.dense(dense_output)
                
                # Compute sparse output and measure error
                sparse_fn = lambda q, k, v: self.sparse_attn(q, k, v, layer)
                self.offline_search.sparse(q, k, v, sparse_fn)
        
        # Finalize search
        self.offline_search.finish()
        
        # Verify results were saved
        self.assertTrue(os.path.exists(self.temp_file.name))
        
        # Load and check results
        with open(self.temp_file.name, 'r') as f:
            results = json.load(f)
        
        # Check that results contain expected fields
        self.assertIn('full_layers', results)
        self.assertIn('avg_errors', results)
        
        # Check that full_layers contains highest error layers
        full_layers = results['full_layers']
        self.assertEqual(len(full_layers), int(self.num_layers * 0.3))
        
        # Verify that layers with highest error were selected
        # In our setup, lower layer indices have higher error
        expected_layers = list(range(int(self.num_layers * 0.3)))
        self.assertEqual(sorted(full_layers), expected_layers)
        
        # Check error values
        avg_errors = results['avg_errors']
        self.assertEqual(len(avg_errors), self.num_layers)
        
        # Errors should decrease with layer index in our test setup
        for i in range(1, self.num_layers):
            self.assertGreaterEqual(avg_errors[i-1], avg_errors[i])

if __name__ == '__main__':
    unittest.main()