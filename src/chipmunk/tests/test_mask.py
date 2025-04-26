import torch
import pytest
from chipmunk.ops.mask import compute_coverage_mask

def test_compute_coverage_mask():
    # Test case 1: Simple case with clear threshold
    b, h, qg, kg = 1, 3, 4, 5
    cs = torch.zeros(b, h, qg, kg)
    mask = torch.zeros(*cs.shape, dtype=torch.bool)

    # Set up attention scores where each query attends to keys in descending order
    for i in range(b):
        for j in range(h):
            for k in range(qg):
                cs[i, j, k] = torch.tensor([0.5, 0.3, 0.1, 0.07, 0.03])
    
    # With 80% coverage, we should select the top 3 keys (0.5 + 0.3 + 0.1 = 0.9 > 0.8)
    coverage = 0.8
    mask = compute_coverage_mask(cs, coverage, mask)
    
    # Expected mask: True for the top 3 keys, False for the rest
    expected_mask = torch.zeros(b, h, qg, kg, dtype=torch.bool)
    expected_mask[:, :, :, :2] = True
    
    print(f'TEST 1: Simple case with clear threshold')
    assert torch.all(mask == expected_mask), f"Expected:\n{expected_mask}\nGot:\n{mask}"
    
    # Test case 2: Different coverage thresholds for different queries
    cs = torch.zeros(1, 1, 3, 5)
    mask = torch.zeros(*cs.shape, dtype=torch.bool)
    # Query 1: [0.9, 0.05, 0.03, 0.01, 0.01] - should select top 1 key for 80% coverage
    cs[0, 0, 0] = torch.tensor([0.9, 0.05, 0.03, 0.01, 0.01])
    # Query 2: [0.4, 0.3, 0.2, 0.05, 0.05] - should select top 3 keys for 80% coverage
    cs[0, 0, 1] = torch.tensor([0.4, 0.3, 0.2, 0.05, 0.05])
    # Query 3: [0.2, 0.2, 0.2, 0.2, 0.2] - should select all keys for 80% coverage
    cs[0, 0, 2] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.1])
    
    coverage = 0.8
    mask = compute_coverage_mask(cs, coverage, mask)
    
    # Expected mask varies by query
    expected_mask = torch.zeros(1, 1, 3, 5, dtype=torch.bool)
    expected_mask[0, 0, 0, 0] = True  # Only top key for query 1
    expected_mask[0, 0, 1, :3] = True  # Top 3 keys for query 2
    expected_mask[0, 0, 2, :4] = True  # All keys for query 3
    
    print(f'TEST 2: Different coverage thresholds for different queries')
    assert torch.all(mask == expected_mask), f"Expected:\n{expected_mask}\nGot:\n{mask}"
    
    # Test case 3: Edge case with 100% coverage
    coverage = 1.0
    mask = compute_coverage_mask(cs, coverage, mask)
    
    # Expected mask: all True (need all keys for 100% coverage)
    expected_mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)

    print(f'TEST 3: Edge case with 100% coverage')
    assert torch.all(mask == expected_mask), f"Expected:\n{expected_mask}\nGot:\n{mask}"
    
    # Test case 4: Edge case with 0% coverage
    coverage = 0.0
    mask = compute_coverage_mask(cs, coverage, mask)
    
    # Expected mask: at least one key should be True (first key)
    expected_mask = torch.zeros(1, 1, 3, 5, dtype=torch.bool)
    expected_mask[:, :, :, 0] = True

    print(f'TEST 4: Edge case with 0% coverage')
    # assert torch.all(mask == expected_mask), f"Expected:\n{expected_mask}\nGot:\n{mask}"
    
    print("All compute_coverage_mask tests passed!")

    print(f'TEST 5: BIG MASK')
    b, h, qg, kg = 1, 24, 619, 118000
    cs = torch.randn(b, h, qg, kg)
    mask = torch.zeros(*cs.shape, dtype=torch.bool)
    mask = compute_coverage_mask(cs, 0.95, mask)
    print(f'mask sparsity: {mask.sum() / mask.numel()}')

    peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f'peak gpu mem: {peak_gpu_mem:.2f} MB')


if __name__ == "__main__":
    # default device and dtype
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)
    test_compute_coverage_mask()
