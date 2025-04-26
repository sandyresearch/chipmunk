import unittest
import torch

from chipmunk.cache.tea_cache import TeaCache
from chipmunk.util.config import GLOBAL_CONFIG, update_global_config

class TestTeaCache(unittest.TestCase):
    def setUp(self):
        # Configure test settings
        update_global_config({
            'tea_cache': {
                'is_enabled': True,
                'threshold': 1e-5,
            }
        })
        self.tea_cache = TeaCache()
        
        # Create a simple model for testing
        self.model = lambda x: x * 2 + 1
        
    def test_tea_cache_functionality(self):
        # Create input tensor
        x = torch.randn(2, 3, 4)
        
        # Define a simple forward function that uses step cache
        def forward(x, step):
            should_skip = self.tea_cache.step(x)
            if should_skip:
                cached = self.tea_cache.load()
                if cached is not None:
                    return cached, should_skip
            
            # Compute output
            output = self.model(x)

            # Store in cache
            self.tea_cache.store(output)
            
            return output, should_skip
        
        # Simulate multiple forward passes
        outputs = []

        computed_steps = []
        skipped_steps = []
        
        for step in range(5):
            output, should_skip = forward(x, step)
            outputs.append(output)
            if should_skip:
                skipped_steps.append(step)
            else:
                computed_steps.append(step)

        x = x + 5
        output, should_skip = forward(x, 5)
        outputs.append(output)
        if should_skip:
            skipped_steps.append(5)
        else:
            computed_steps.append(5)

        # Verify only non-skipped steps were computed
        self.assertEqual(computed_steps, [0, 5])
        self.assertEqual(skipped_steps, [1, 2, 3, 4])
        
        # Verify cache was updated correctly
        self.assertTrue(torch.allclose(self.tea_cache.load(), outputs[-1]))

if __name__ == '__main__':
    unittest.main()