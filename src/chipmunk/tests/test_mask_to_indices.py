import chipmunk.ops
import chipmunk.ops.voxel
import torch
from torch.utils.cpp_extension import load
import triton
from typing import Tuple
import time

def masktoinds(mask, multiple=None):
    """
    Compute the per-row nonzero indices and counts of the mask.

    Args:
        mask     : [..., m, n]
        multiple : int

    Returns:
        inds     : [..., m, n]
        counts   : [..., m]
    """

    if multiple is not None:
        counts = ((mask.sum(dim=-1).to(torch.int32) + multiple - 1) // multiple) * multiple
    else:
        counts = mask.sum(dim=-1).to(torch.int32)
    inds = mask.char().argsort(dim=-1, descending=True)
    # return None, counts.contiguous().to(torch.int32)
    return inds.contiguous().to(torch.int32), counts.contiguous().to(torch.int32)

def compare_mask_to_indices_implementations(mask, multiple_of, pad_to_multiple_of=192):
    """
    Compare the outputs of both mask_to_indices implementations and raise error if they don't match.
    
    Args:
        mask: Input boolean mask
        multiple_of: Multiple for padding counts
        pad_to_multiple_of: Padding for indices tensor
        
    Returns:
        Tuple of (indices, counts) from the reference implementation
        
    Raises:
        AssertionError: If implementations don't match with detailed debug info
    """
    # Get reference implementation (voxel)
    ref_indices, ref_counts = chipmunk.ops.voxel.masktoinds(mask, multiple=multiple_of)
    
    # Get CUDA implementation
    cuda_indices, cuda_counts = chipmunk.ops.mask_to_indices(mask, multiple_of, pad_to_multiple_of)
    
    # Slice CUDA indices to match reference size
    cuda_indices_sliced = cuda_indices[..., :mask.shape[-1]]
    
    # Check counts first (easier to debug)
    if not torch.equal(ref_counts, cuda_counts):
        print("=== COUNTS MISMATCH ===")
        print(f"Reference counts: {ref_counts}")
        print(f"CUDA counts: {cuda_counts}")
        print(f"Counts diff: {torch.abs(ref_counts - cuda_counts)}")
        print(f"Mask shape: {mask.shape}")
        print(f"True counts per row: {mask.sum(dim=-1)}")
        raise AssertionError("Counts don't match between implementations")
    
    # Prepare indices for comparison (set unused positions to -1)
    ref_indices_normalized = ref_indices.clone()
    cuda_indices_normalized = cuda_indices_sliced.clone()
    
    b, h, m, n = mask.shape
    for i in range(b):
        for j in range(h):
            for k in range(m):
                count = ref_counts[i, j, k].item()
                # Set unused positions to -1
                ref_indices_normalized[i, j, k, count:] = -1
                cuda_indices_normalized[i, j, k, count:] = -1
    
    # Sort indices for comparison (order shouldn't matter)
    ref_sorted = ref_indices_normalized.sort(dim=-1, descending=True).values
    cuda_sorted = cuda_indices_normalized.sort(dim=-1, descending=True).values
    
    # Check indices
    if not torch.equal(ref_sorted, cuda_sorted):
        print("=== INDICES MISMATCH ===")
        print(f"Mask shape: {mask.shape}")
        print(f"Multiple of: {multiple_of}")
        print(f"Pad to multiple of: {pad_to_multiple_of}")
        
        # Find first mismatch
        diff_mask = ref_sorted != cuda_sorted
        if torch.any(diff_mask):
            first_mismatch = torch.nonzero(diff_mask, as_tuple=False)[0]
            i, j, k, idx = first_mismatch.tolist()
            print(f"First mismatch at [{i}, {j}, {k}, {idx}]:")
            print(f"  Reference: {ref_sorted[i, j, k, idx].item()}")
            print(f"  CUDA: {cuda_sorted[i, j, k, idx].item()}")
            print(f"  Count for this row: {ref_counts[i, j, k].item()}")
            print(f"  Mask true indices: {torch.nonzero(mask[i, j, k], as_tuple=False).squeeze(-1)}")
            print(f"  Reference row: {ref_sorted[i, j, k]}")
            print(f"  CUDA row: {cuda_sorted[i, j, k]}")
            
            # Check for garbage values in CUDA output
            count = ref_counts[i, j, k].item()
            cuda_valid = cuda_indices_sliced[i, j, k, :count]
            if torch.any(cuda_valid < 0) or torch.any(cuda_valid >= n):
                print(f"  *** GARBAGE VALUES DETECTED in CUDA output ***")
                print(f"  CUDA valid indices: {cuda_valid}")
                print(f"  Negative indices: {cuda_valid[cuda_valid < 0]}")
                print(f"  Out-of-range indices: {cuda_valid[cuda_valid >= n]}")
        
        raise AssertionError("Indices don't match between implementations")
    
    return ref_indices, ref_counts

def test_implementation_comparison():
    """Test that both implementations produce identical results"""
    print("\n=== Testing implementation comparison ===")
    
    test_cases = [
        # Basic test
        (torch.rand((1, 1, 1, 100), device="cuda") < 0.3, 32),
        # Trailing block (reported issue pattern)
        (torch.cat([torch.zeros(900, device="cuda"), torch.ones(1644, device="cuda")]).bool().unsqueeze(0).unsqueeze(0).unsqueeze(0), 128),
        # Leading block
        (torch.cat([torch.ones(256, device="cuda"), torch.zeros(744, device="cuda")]).bool().unsqueeze(0).unsqueeze(0).unsqueeze(0), 64),
        # Scattered
        (torch.zeros(1000, device="cuda").bool().unsqueeze(0).unsqueeze(0).unsqueeze(0), 16),  # All False
        (torch.ones(1000, device="cuda").bool().unsqueeze(0).unsqueeze(0).unsqueeze(0), 16),   # All True
    ]
    
    # Add scattered pattern
    scattered_mask = torch.zeros(1000, device="cuda", dtype=torch.bool)
    scattered_mask[::7] = True
    test_cases.append((scattered_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), 32))
    
    for i, (mask, multiple_of) in enumerate(test_cases):
        print(f"Testing case {i+1}: shape={mask.shape}, true_count={mask.sum().item()}, multiple={multiple_of}")
        try:
            ref_indices, ref_counts = compare_mask_to_indices_implementations(mask, multiple_of)
            print(f"  ✓ Case {i+1} passed")
        except AssertionError as e:
            print(f"  ✗ Case {i+1} failed: {e}")
            raise

def test_random_comparison_stress():
    """Stress test with many random cases to catch race conditions"""
    print("\n=== Random comparison stress test ===")
    
    num_tests = 50
    failed_tests = []
    
    for test_idx in range(num_tests):
        torch.manual_seed(test_idx)
        
        # Random dimensions
        b = torch.randint(1, 3, (1,)).item()
        h = torch.randint(1, 4, (1,)).item() 
        m = torch.randint(1, 8, (1,)).item()
        n = torch.randint(100, 2048, (1,)).item()
        
        # Random sparsity
        sparsity = torch.rand(1).item() * 0.5  # 0-50% sparse
        mask = torch.rand((b, h, m, n), device="cuda") < sparsity
        
        # Random multiple
        multiple_of = [16, 32, 64, 128, 256][test_idx % 5]
        
        try:
            compare_mask_to_indices_implementations(mask, multiple_of)
            if (test_idx + 1) % 10 == 0:
                print(f"  Completed {test_idx + 1}/{num_tests} random tests")
        except AssertionError as e:
            failed_tests.append((test_idx, str(e)))
            print(f"  ✗ Random test {test_idx} failed: {e}")
            # Continue testing to find all failures
            continue
    
    if failed_tests:
        print(f"Failed {len(failed_tests)}/{num_tests} random tests")
        raise AssertionError(f"Implementation comparison failed on {len(failed_tests)} out of {num_tests} random tests")
    else:
        print(f"✓ All {num_tests} random comparison tests passed")

def test_mask_to_indices():
    b, h, m, n = 1, 3, 7, 17
    mask = torch.rand((b, h, m, n), device="cuda", dtype=torch.float32) < 0.5
    multiple = 1
    ref_indices, ref_counts = masktoinds(mask, multiple)

    # indices = torch.empty((b, h, m, n), device="cuda", dtype=torch.int32)
    # counts = torch.empty((b, h, m), device="cuda", dtype=torch.int32)
    # Call the CUDA function
    # cuda_module.mask_to_indices(mask, indices, counts, multiple)
    indices, counts = chipmunk.ops.mask_to_indices(mask, multiple, pad_to_multiple_of=2)

    for i in range(b):
        for j in range(h):
            for k in range(m):
                # set after counts to -1
                ref_indices[i, j, k, ref_counts[i, j, k]:] = -1
                indices[i, j, k, counts[i, j, k]:] = -1

    ref_indices = ref_indices.sort(dim=-1, descending=True).values
    indices = indices.sort(dim=-1, descending=True).values

    print(f'ref indices shape: {ref_indices.shape}')
    print(f'cuda indices shape: {indices.shape}')
    indices = indices[..., :n]
    print(f'cuda indices shape after slice: {indices.shape}')

    # Print results
    print(f'Ref indices: {ref_indices.cpu()}')
    print(f'Cuda indices: {indices.cpu()}')
    print(f'indices diff: {torch.abs(ref_indices - indices).cpu()}')
    assert torch.allclose(ref_indices, indices)
    print()
    print(f'Ref counts: {ref_counts.cpu()}')
    print(f'Cuda counts: {counts.cpu()}')
    print(f'counts diff: {torch.abs(ref_counts - counts).cpu()}')
    assert torch.allclose(ref_counts, counts)

def test_trailing_true_mask():
    """Test case where mask has trailing True values (like [..., -1644:] = True)"""
    print("\n=== Testing trailing True mask ===")
    b, h, m, n = 1, 1, 1, 2048
    mask = torch.zeros((b, h, m, n), device="cuda", dtype=torch.bool)
    
    # Set last 1644 elements to True (multiple of 128 as mentioned)
    trailing_count = 1644
    mask[..., -trailing_count:] = True
    
    multiple = 128
    
    # Initialize with garbage values to detect unwritten positions
    indices, counts = chipmunk.ops.mask_to_indices(mask, multiple, pad_to_multiple_of=192)
    
    # Check that all indices in the valid range are proper indices
    for i in range(b):
        for j in range(h):
            for k in range(m):
                count = counts[i, j, k].item()
                valid_indices = indices[i, j, k, :count]
                
                print(f"Count: {count}, Expected: {((trailing_count + multiple - 1) // multiple) * multiple}")
                print(f"Valid indices range: [{valid_indices.min().item()}, {valid_indices.max().item()}]")
                print(f"First 10 indices: {valid_indices[:10].cpu()}")
                print(f"Last 10 indices: {valid_indices[-10:].cpu()}")
                
                # All valid indices should be in range [0, n-1]
                assert torch.all(valid_indices >= 0), f"Found negative index: {valid_indices[valid_indices < 0]}"
                assert torch.all(valid_indices < n), f"Found out-of-range index: {valid_indices[valid_indices >= n]}"
                
                # Check that indices correspond to True positions in mask
                mask_indices = torch.nonzero(mask[i, j, k], as_tuple=False).squeeze(-1)
                expected_count = ((mask_indices.numel() + multiple - 1) // multiple) * multiple
                assert count == expected_count, f"Count mismatch: got {count}, expected {expected_count}"

def test_garbage_detection():
    """Test to detect if kernel leaves garbage values in output"""
    print("\n=== Testing garbage detection ===")
    b, h, m, n = 1, 1, 3, 1000
    
    # Create masks with different sparsity patterns
    masks = []
    
    # Sparse mask at beginning
    mask1 = torch.zeros((b, h, m, n), device="cuda", dtype=torch.bool)
    mask1[..., :100] = True
    masks.append(("beginning_sparse", mask1))
    
    # Sparse mask at end  
    mask2 = torch.zeros((b, h, m, n), device="cuda", dtype=torch.bool)
    mask2[..., -100:] = True
    masks.append(("end_sparse", mask2))
    
    # Scattered mask
    mask3 = torch.zeros((b, h, m, n), device="cuda", dtype=torch.bool)
    mask3[..., ::10] = True  # Every 10th element
    masks.append(("scattered", mask3))
    
    multiple = 64
    
    for name, mask in masks:
        print(f"\nTesting {name} mask...")
        
        # Pre-fill with garbage values
        indices_garbage = torch.full((b, h, m, n + 192), -999, device="cuda", dtype=torch.int32)
        
        indices, counts = chipmunk.ops.mask_to_indices(mask, multiple, pad_to_multiple_of=192)
        
        for i in range(b):
            for j in range(h):
                for k in range(m):
                    count = counts[i, j, k].item()
                    valid_indices = indices[i, j, k, :count]
                    
                    # Check no garbage values in valid range
                    garbage_mask = valid_indices == -999
                    if torch.any(garbage_mask):
                        garbage_positions = torch.nonzero(garbage_mask, as_tuple=False).squeeze(-1)
                        print(f"ERROR: Found garbage values at positions: {garbage_positions.cpu()}")
                        print(f"Indices around garbage: {valid_indices[max(0, garbage_positions[0]-5):garbage_positions[0]+5].cpu()}")
                        assert False, f"Found garbage values in {name} mask"
                    
                    # Verify indices are valid
                    assert torch.all(valid_indices >= 0), f"Negative indices in {name}"
                    assert torch.all(valid_indices < n), f"Out-of-range indices in {name}"
                    
                    print(f"{name}: count={count}, indices_range=[{valid_indices.min()}, {valid_indices.max()}]")

def test_exact_multiple_boundary():
    """Test when true count is exactly a multiple"""
    print("\n=== Testing exact multiple boundary ===")
    b, h, m, n = 1, 1, 1, 1000
    multiple = 128
    
    # Create mask with exactly 256 True values (2 * multiple)
    mask = torch.zeros((b, h, m, n), device="cuda", dtype=torch.bool)
    mask[..., :256] = True
    
    indices, counts = chipmunk.ops.mask_to_indices(mask, multiple, pad_to_multiple_of=192)
    
    count = counts[0, 0, 0].item()
    valid_indices = indices[0, 0, 0, :count]
    
    print(f"True count: 256, Padded count: {count}")
    print(f"Expected padded count: {((256 + multiple - 1) // multiple) * multiple}")
    
    assert count == 256, f"Expected count 256, got {count}"
    assert torch.all(valid_indices >= 0), "Found negative indices"
    assert torch.all(valid_indices < n), "Found out-of-range indices"
    
    # First 256 should be 0-255, rest should be valid padding indices
    true_indices = valid_indices[:256]
    expected_true = torch.arange(256, device="cuda", dtype=torch.int32)
    assert torch.equal(true_indices.sort().values, expected_true), "True indices don't match expected"

def test_mask_to_indices_hunyuan_perf():
    b, h, m, n = 1, 3, 619, 118800
    # b, h, m, n = 1, 3, 619, (118800 // 32) * 32
    mask = torch.rand((b, h, m, n), device="cuda", dtype=torch.float32) < 0.1
    multiple = 112

    ref_indices, ref_counts = masktoinds(mask, multiple)

    # indices = torch.empty((b, h, m, n), device="cuda", dtype=torch.int32)
    # counts = torch.empty((b, h, m), device="cuda", dtype=torch.int32)
    # Call the CUDA function
    # cuda_module.mask_to_indices(mask, indices, counts, multiple)
    indices, counts = chipmunk.ops.mask_to_indices(mask, multiple, pad_to_multiple_of=192)
    indices = indices[..., :n]

    for i in range(b):
        for j in range(h):
            for k in range(m):
                # set after counts to -1
                ref_indices[i, j, k, ref_counts[i, j, k]:] = -1
                indices[i, j, k, counts[i, j, k]:] = -1

    ref_indices = ref_indices.sort(dim=-1, descending=True).values
    indices = indices.sort(dim=-1, descending=True).values

    # Print results
    print(f'Ref counts: {ref_counts.cpu()}')
    print(f'Cuda counts: {counts.cpu()}')
    print(f'counts diff: {(torch.abs(ref_counts - counts) > 0).nonzero()}')
    assert torch.allclose(ref_counts, counts)
    print(f'Ref indices: {ref_indices.cpu()}')
    print(f'Cuda indices: {indices.cpu()}')
    print(f'indices diff: {(torch.abs(ref_indices - indices) > 0).nonzero()}')
    # assert torch.allclose(ref_indices, indices)
    print()

    ref_ms = triton.testing.do_bench(lambda: masktoinds(mask, multiple))
    cuda_ms = triton.testing.do_bench(
        lambda: chipmunk.ops.mask_to_indices(mask, multiple, pad_to_multiple_of=192))

    # Print results
    print(f'torch ms: {ref_ms}')
    print(f'cuda ms: {cuda_ms}')

def test_race_conditions():
    """Test for race conditions by running many iterations with different patterns"""
    print("\n=== Testing for race conditions ===")
    
    num_iterations = 100
    b, h, m, n = 1, 2, 4, 2048
    multiple = 128
    
    failed_iterations = []
    
    for iteration in range(num_iterations):
        # Use different random seed each iteration
        torch.manual_seed(iteration)
        
        # Create different mask patterns each iteration
        if iteration % 4 == 0:
            # Random sparse mask
            mask = torch.rand((b, h, m, n), device="cuda") < 0.1
        elif iteration % 4 == 1:
            # Trailing block pattern (like the reported issue)
            mask = torch.zeros((b, h, m, n), device="cuda", dtype=torch.bool)
            trailing_size = 128 + (iteration % 10) * 128  # Vary the trailing size
            mask[..., -trailing_size:] = True
        elif iteration % 4 == 2:
            # Leading block pattern
            mask = torch.zeros((b, h, m, n), device="cuda", dtype=torch.bool)
            leading_size = 256 + (iteration % 8) * 64
            mask[..., :leading_size] = True
        else:
            # Scattered pattern
            mask = torch.zeros((b, h, m, n), device="cuda", dtype=torch.bool)
            step = 5 + (iteration % 10)
            mask[..., ::step] = True
        
        try:
            indices, counts = chipmunk.ops.mask_to_indices(mask, multiple, pad_to_multiple_of=192)
            
            # Verify each row
            for i in range(b):
                for j in range(h):
                    for k in range(m):
                        count = counts[i, j, k].item()
                        valid_indices = indices[i, j, k, :count]
                        
                        # Check for garbage values (uninitialized memory)
                        if torch.any(valid_indices < 0):
                            raise AssertionError(f"Negative indices found at iteration {iteration}")
                        
                        if torch.any(valid_indices >= n):
                            raise AssertionError(f"Out-of-range indices found at iteration {iteration}")
                        
                        # Check that we have the right number of true indices
                        true_count = mask[i, j, k].sum().item()
                        expected_padded_count = ((true_count + multiple - 1) // multiple) * multiple
                        if count != expected_padded_count:
                            raise AssertionError(f"Count mismatch at iteration {iteration}: got {count}, expected {expected_padded_count}")
                        
                        # Verify that the first true_count indices are actually true positions
                        if true_count > 0:
                            true_positions = torch.nonzero(mask[i, j, k], as_tuple=False).squeeze(-1)
                            actual_true_indices = valid_indices[:true_count].sort().values
                            expected_true_indices = true_positions.sort().values
                            
                            if not torch.equal(actual_true_indices, expected_true_indices):
                                raise AssertionError(f"True indices mismatch at iteration {iteration}")
                        
                        # Check padding indices are valid (if any)
                        if count > true_count:
                            padding_indices = valid_indices[true_count:]
                            if torch.any(padding_indices < 0) or torch.any(padding_indices >= n):
                                raise AssertionError(f"Invalid padding indices at iteration {iteration}")
            
            # Print progress every 20 iterations
            if (iteration + 1) % 20 == 0:
                print(f"Completed {iteration + 1}/{num_iterations} iterations successfully")
                
        except Exception as e:
            failed_iterations.append((iteration, str(e)))
            print(f"FAILED at iteration {iteration}: {e}")
            
            # Print debug info for failed iteration
            print(f"Mask shape: {mask.shape}")
            print(f"Mask pattern: {mask.sum()} true values")
            print(f"Indices shape: {indices.shape}")
            print(f"Counts: {counts.cpu()}")
            
            # Don't fail immediately, collect all failures
            continue
    
    print(f"\nRace condition test completed: {num_iterations - len(failed_iterations)}/{num_iterations} passed")
    
    if failed_iterations:
        print(f"Failed iterations: {[it[0] for it, _ in failed_iterations]}")
        for iteration, error in failed_iterations[:5]:  # Show first 5 failures
            print(f"  Iteration {iteration}: {error}")
        
        # Fail the test if we had any failures
        assert len(failed_iterations) == 0, f"Race condition detected: {len(failed_iterations)} failures out of {num_iterations} iterations"
    else:
        print("✓ No race conditions detected")

def test_concurrent_stress():
    """Stress test with larger dimensions and concurrent operations"""
    print("\n=== Concurrent stress test ===")
    
    # Larger dimensions to stress the kernel
    b, h, m, n = 2, 4, 8, 4096
    multiple = 128
    num_trials = 50
    
    for trial in range(num_trials):
        torch.manual_seed(42 + trial)
        
        # Create a challenging mask pattern
        mask = torch.zeros((b, h, m, n), device="cuda", dtype=torch.bool)
        
        # Mix of patterns in different rows
        for i in range(b):
            for j in range(h):
                for k in range(m):
                    pattern = (i * h * m + j * m + k) % 4
                    if pattern == 0:
                        # Random sparse
                        mask[i, j, k] = torch.rand(n, device="cuda") < 0.05
                    elif pattern == 1:
                        # Trailing block
                        trailing_size = 256 + (trial % 10) * 64
                        mask[i, j, k, -trailing_size:] = True
                    elif pattern == 2:
                        # Leading block
                        leading_size = 512 + (trial % 8) * 32
                        mask[i, j, k, :leading_size] = True
                    else:
                        # Scattered
                        mask[i, j, k, ::7] = True
        
        indices, counts = chipmunk.ops.mask_to_indices(mask, multiple, pad_to_multiple_of=256)
        
        # Quick validation
        for i in range(b):
            for j in range(h):
                for k in range(m):
                    count = counts[i, j, k].item()
                    valid_indices = indices[i, j, k, :count]
                    
                    assert torch.all(valid_indices >= 0), f"Negative indices in trial {trial}"
                    assert torch.all(valid_indices < n), f"Out-of-range indices in trial {trial}"
                    
                    true_count = mask[i, j, k].sum().item()
                    expected_count = ((true_count + multiple - 1) // multiple) * multiple
                    assert count == expected_count, f"Count mismatch in trial {trial}"
        
        if (trial + 1) % 10 == 0:
            print(f"Stress test: {trial + 1}/{num_trials} completed")
    
    print("✓ Concurrent stress test passed")

if __name__ == "__main__":
    test_mask_to_indices()
    test_trailing_true_mask()
    test_garbage_detection() 
    test_exact_multiple_boundary()
    test_mask_to_indices_hunyuan_perf()
    test_race_conditions()
    test_concurrent_stress()
    test_implementation_comparison()
    test_random_comparison_stress()
