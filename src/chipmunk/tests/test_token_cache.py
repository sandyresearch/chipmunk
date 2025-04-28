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
                'cache_ratio': 0.6,
                'full_every': 3
            },
            'attn': {
                'mbm': 2
            }
        })
        self.token_cache = TokenCache()  # Cache 85% of tokens
        
        # Create simple attention and MLP functions for testing
        self.batch_size = 1
        self.heads = 2
        self.seq_len = 5
        self.hidden_dim = 8

        self.qg = (self.seq_len + GLOBAL_CONFIG['attn']['mbm'] - 1) // GLOBAL_CONFIG['attn']['mbm']
        
        # Simple attention function that returns column sums and output
        self.attention = lambda: (
            torch.randn(self.batch_size, self.seq_len, self.hidden_dim),  # Output
            torch.randn(self.batch_size, self.heads, self.qg, self.seq_len),  # Column sums
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

    def test_token_cache_operations_manually(self):
        # Create a controlled input tensor with recognizable values
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]], dtype=torch.float32)  # [1, 1, 8]
        x = x.repeat(1, 5, 1)  # Expand to [1, 5, 8]
        
        # Add position-specific values to make each token unique
        for i in range(5):
            x[0, i] += i * 0.1
        
        # Create fake column sums that will make tokens 1 and 3 the most important
        # Shape: [batch_size, heads, 4, seq_len]
        fake_col_sums = torch.zeros(self.batch_size, self.heads, self.qg, self.seq_len)
        # Make tokens 0 and 1 have high importance scores
        # Token from second and third query group should be selected by uniform spatial
        fake_col_sums[:, :, :, 0] = 10.0  # Token 0 is uniform spatial max
        fake_col_sums[:, :, :, 1] = 1.0   # Token 1 is cached
        fake_col_sums[:, :, :, 2] = 3.0   # Token 2 is uniform spatial max
        fake_col_sums[:, :, :, 3] = 2.0   # Token 3 is important
        fake_col_sums[:, :, :, 4] = 8.0   # Token 4 is cached (final chunk)
        
        # Score the tokens
        self.token_cache.score(fake_col_sums)
        
        # Verify the token selection - with 60% cache ratio, we should select 2 out of 5 tokens + 2 uniform spatial
        expected_inds = torch.tensor([[0, 2, 4, 3]], dtype=torch.int64)
        self.assertTrue(torch.equal(self.token_cache.inds, expected_inds), 
                       f"Expected inds {expected_inds}, got {self.token_cache.inds}")
        
        # Gather important tokens (those in the mask)
        important_tokens = self.token_cache.gather(x)
        
        # Verify shape of important tokens - should have 4 tokens (tokens 0, 1, 2, and 4)
        self.assertEqual(important_tokens.shape, (1, 4, 8))
        
        # Verify the gathered tokens are the correct ones
        expected_tokens = torch.stack([x[0, 0], x[0, 2], x[0, 4], x[0, 3]], dim=0).unsqueeze(0)
        self.assertTrue(torch.allclose(important_tokens, expected_tokens),
                       f"Expected important tokens {expected_tokens}, got {important_tokens}")
        
        # Apply computation only to important tokens
        computed_tokens = self.mlp(important_tokens)  # mlp = x * 2 + 1
        
        # Scatter the computed tokens back
        result = self.token_cache.scatter(computed_tokens, x.clone())
        
        # Verify the result
        expected_result = x.clone()
        # Apply the transformation to tokens 1 and 3
        expected_result[0, 0] = x[0, 0] * 2 + 1
        expected_result[0, 2] = x[0, 2] * 2 + 1
        expected_result[0, 3] = x[0, 3] * 2 + 1
        expected_result[0, 4] = x[0, 4] * 2 + 1
        
        self.assertTrue(torch.allclose(result, expected_result),
                       f"Expected scattered result to match manual calculation")

if __name__ == '__main__':
    unittest.main()