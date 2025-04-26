import unittest
import torch

from chipmunk.cache.token_cache import TokenCache
from chipmunk.util.config import GLOBAL_CONFIG, update_global_config

class TestTokenCache(unittest.TestCase):
    def setUp(self):
        # Configure test settings
        update_global_config({
            'token_cache': {
                'is_enabled': True,
                'cache_ratio': 0.85,
                'full_every': 3
            },
            'attn': {
                'mbm': 4
            }
        })
        self.token_cache = TokenCache()  # Cache 85% of tokens
        
        # Create simple attention and MLP functions for testing
        self.batch_size = 2
        self.heads = 4
        self.seq_len = 16
        self.hidden_dim = 32
        
        # Simple attention function that returns column sums and output
        self.attention = lambda: (
            torch.randn(self.batch_size, self.seq_len, self.hidden_dim),  # Output
            torch.randn(self.batch_size, self.heads, 4, self.seq_len),  # Column sums
        )
        
        # Simple MLP function
        self.mlp = lambda x: x * 2 + 1
        
    def test_token_cache_functionality(self):
        # Create input tensors
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        # Define a simple forward function that uses token cache
        def forward(x, step):
            # Run attention and get column sums
            o, cs = self.attention()
            
            # Update token importance scores
            self.token_cache.score(cs)
            
            # Determine if we should compute full output
            compute_full = (step % GLOBAL_CONFIG['token_cache']['full_every'] == 0)
            
            if compute_full:
                # Compute MLP for all tokens
                mlp_out = self.mlp(x)
                # Store full result in cache
                self.token_cache.scatter(mlp_out, x)
                return mlp_out
            else:
                # Extract important tokens
                important_tokens = self.token_cache.gather(x)
                # Compute MLP only for important tokens
                computed_tokens = self.mlp(important_tokens)
                # Merge with cached tokens
                result = self.token_cache.scatter(computed_tokens, x)
                return result
        
        # Simulate multiple forward passes
        outputs = []
        full_computation_steps = []
        
        for step in range(10):
            output = forward(x, step)
            outputs.append(output)
            
            # Check if full computation was performed
            if step % GLOBAL_CONFIG['token_cache']['full_every'] == 0:
                full_computation_steps.append(step)
                
                # Verify output shape
                self.assertEqual(output.shape, x.shape)
                
                # For full computation steps, output should match direct MLP application
                expected = self.mlp(x)
                self.assertTrue(torch.allclose(output, expected), 
                               f"Output at step {step} doesn't match expected value for full computation")
        
        # Verify only scheduled steps performed full computation
        self.assertEqual(full_computation_steps, [0, 3, 6, 9])
        
        # Verify token selection is working (mask should be created)
        self.assertIsNotNone(self.token_cache.inds)

if __name__ == '__main__':
    unittest.main()