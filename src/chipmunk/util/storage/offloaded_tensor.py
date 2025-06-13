import torch
from chipmunk.util import GLOBAL_CONFIG

# Determines how many layer "slots" we keep in GPU memory simultaneously
PIPELINE_DEPTH = 2
assert PIPELINE_DEPTH > 1, "Pipeline depth must be greater than 1 - if pipeline depth is 1, this means we are using naive offloading per-layer which doesn't make sense!"

# We create two dedicated streams for all offloading (rather than using per-layer or per-
# object streams) in order to avoid using PyTorch’s internal stream pool (default ~32 streams). 
# That pool can inadvertently cause collisions with the main compute stream every 32nd stream
# instantiation, making memory transfers slow.
global_offload_stream = torch.cuda.Stream()
global_load_stream = torch.cuda.Stream()

# A global dictionary to hold GPU tensors in a round-robin pipeline:
#  gpu_tensors[name] -> [gpu_tensor_for_slot_0, gpu_tensor_for_slot_1, ...]
gpu_tensors = {}


class MaybeOffloadedTensor:
    """
    MaybeOffloadedTensor implements a mechanism for maintaining a sliding pipeline of GPU tensors,
    that are dynamically loaded in and out of pinned CPU memory as the model progresses through layers.
    The pipeline depth is determined by a constant (`PIPELINE_DEPTH`), and the GPU storage is shared
    between layers.
    
    Design Goals:
      1) Maintain a small pipeline of loaded layers (determined by PIPELINE_DEPTH).
      2) Use pinned CPU memory for faster async transfers.
      3) Use a single dedicated CUDA stream to handle CPU↔GPU copies without
         competing with the default compute stream.
      4) Defer GPU allocations until load-time for memory efficiency.
    
    Usage Overview:
      - Instantiate this class with a unique `name` and a `layer_num`.
      - Offload any current GPU tensor using `offload()`.
      - Load back into GPU memory on demand via `load()`.
      - If you need to block until copies are done, call `wait_for_completion()`.
    """

    # Default buffer sizes for pinned CPU memory, tuned for typical shape sizes
    LARGE_BUF_SIZE  = int(1 * 32 * 150000 * 128 * torch.finfo(torch.bfloat16).bits // 8)
    MEDIUM_BUF_SIZE = int(1 * 32 * 90000 * 128 * torch.finfo(torch.bfloat16).bits // 8)
    SMALL_BUF_SIZE  = 1 * 32 * 15000 * 128 * torch.finfo(torch.bfloat16).bits // 8

    @torch.compiler.disable # torch.compile fails to allocate pinned CPU memory :(
    def __init__(self, name: str, layer_num: int, dtype: torch.dtype,
                 device: torch.device, cpu_buf_size: int = LARGE_BUF_SIZE):
        """
        :param name: Unique identifier for this tensor group, shared between layers.
        :param layer_num: Numeric index for the layer; used with PIPELINE_DEPTH
                          to map to a particular slot.
        :param dtype: PyTorch data type, e.g. torch.bfloat16.
        :param device: Device on which GPU allocation will occur (e.g. 'cuda').
        :param cpu_buf_size: Size of the pinned CPU buffer for offloading.
        """
        is_offload_enabled = GLOBAL_CONFIG['offloading']
        if name not in is_offload_enabled:
            raise ValueError(f"Invalid tensor name: {name}. Expected one of: {is_offload_enabled.keys()}")
        self.name = name
        self.layer_num = layer_num
        self.is_offload_enabled = not GLOBAL_CONFIG['offloading']['global_disable_offloading'] and is_offload_enabled[name]
        assert not (self.is_offload_enabled == True and name == 'attn.lse_constants'), "LSE constants cannot be offloaded (i) in Triton because they are passed in as a tuple. You will need to implement this manually yourself; and (ii) in CUDA because they are padded to 16-byte TMA-aligned tensors, and offloading with non-contiguous tensors is not yet tested."
        assert not (self.is_offload_enabled == True and name == 'attn.indices' and GLOBAL_CONFIG['attn']['provider'] == 'cuda' and GLOBAL_CONFIG['attn']['should_compress_indices'] == False), "Non-compressed indices cannot be offloaded in CUDA because they are padded to 16-byte TMA-aligned tensors, and offloading with non-contiguous tensors is not yet tested."
        # Choose a pipeline slot for this layer using modulo:
        self.layer_key = layer_num % PIPELINE_DEPTH
        self.device = device
        self.offload_stream = global_offload_stream
        self.load_stream = global_load_stream
        # Pre-allocate pinned CPU buffer to hold the tensor data
        if self.is_offload_enabled:
            print(f"Offloaded tensor {name} allocated {cpu_buf_size} bytes of pinned CPU memory for {name} layer {layer_num}")
            self.cpu_buf = [torch.empty(cpu_buf_size, dtype=dtype, device="cpu", pin_memory=True) for _ in range(GLOBAL_CONFIG['num_model_invocations_per_inference_step'])]
        else:
            self.gpu_tensor = [None for _ in range(GLOBAL_CONFIG['num_model_invocations_per_inference_step'])]
        # Will store the original shape of the tensor so we can reload properly
        self.real_shape = [None for _ in range(GLOBAL_CONFIG['num_model_invocations_per_inference_step'])]
        self.real_stride = [None for _ in range(GLOBAL_CONFIG['num_model_invocations_per_inference_step'])]
        self.load_completed_event = None

        self.model_invocation_count = 0
        
        # Allocate PIPELINE_DEPTH slots for this tensor name
        if name not in gpu_tensors:
            gpu_tensors[name] = [None] * PIPELINE_DEPTH

    def complete_cur_layer(self):
        self.model_invocation_count += 1

    def get_cur_model_invocation_key(self):
        return self.model_invocation_count % GLOBAL_CONFIG['num_model_invocations_per_inference_step']

    @torch.compiler.disable # disable torch.compile so that we can use pinned memory and .record_stream()
    def offload(self, gpu_tensor: torch.Tensor):
        """
        Asynchronously copy a GPU tensor into this object's pinned CPU buffer on self.offload_stream.
        We remember the shape to reconstruct the tensor when loading back to GPU.

        :param gpu_tensor: Tensor on GPU that will be copied out to CPU memory.
        """
        if not self.is_offload_enabled:
            self.gpu_tensor[self.get_cur_model_invocation_key()] = gpu_tensor
            return
        # Validate that our pinned buffer is large enough
        assert gpu_tensor.numel() <= self.cpu_buf[self.get_cur_model_invocation_key()].numel(), (
            f"Tensor {self.name} is too large to offload - try adjusting MaybeOffloadedTensor.LARGE_BUF_SIZE (requested {gpu_tensor.numel()} elements, available {self.cpu_buf[self.get_cur_model_invocation_key()].numel()} elements)"
        )
        # Record the original shape so we can create a matching GPU tensor on load
        self.real_shape[self.get_cur_model_invocation_key()] = gpu_tensor.size()
        self.real_stride[self.get_cur_model_invocation_key()] = gpu_tensor.stride()          # <── store the stride
        # Perform copy on our dedicated self.offload_stream
        self.offload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.offload_stream):
            self.cpu_buf[self.get_cur_model_invocation_key()][:gpu_tensor.numel()].view(gpu_tensor.shape).copy_(gpu_tensor, non_blocking=True)
            gpu_tensor.record_stream(self.offload_stream)

    def offload_cur_value(self):
        """
        Utility function to offload the currently loaded GPU value for this layer.
        If the GPU slot for this layer has data (i.e. was loaded), offload it to CPU.
        """
        self.offload(self.get_loaded_value())

    def get_loaded_value(self):
        """
        Retrieve the GPU tensor that corresponds to this layer_num's slot.
        :return: The GPU tensor if loaded, otherwise raises AssertionError.
        """
        if not self.is_offload_enabled:
            return self.gpu_tensor[self.get_cur_model_invocation_key()]
        
        gpu_tensor = gpu_tensors[self.name][self.layer_key]
        assert gpu_tensor is not None, (
            f"Tensor {self.name} is not loaded yet for layer {self.layer_num}. Please call load_async() first (followed by load_async_wait())"
        )
        return gpu_tensor

    @torch.compiler.disable # disable torch.compile so that we can use pinned memory and tensor.record_stream()
    def load_async(self):
        """
        Load the tensor from this object's pinned CPU buffer back into GPU memory.
        If the GPU slot is not allocated yet, allocate it now. Then copy asynchronously.

        :return: The GPU tensor now loaded into the correct slot.
        """
        key  = self.get_cur_model_invocation_key()
        size = self.real_shape[key]
        if size is None:           # nothing has been off-loaded yet
            return None

        if not self.is_offload_enabled:
            return self.gpu_tensor[key]

        stride = self.real_stride[key]

        # (re)allocate the GPU slot **with identical strides**
        slot = gpu_tensors[self.name]
        need_new = (
            slot[self.layer_key] is None
            or slot[self.layer_key].shape  != size
            or slot[self.layer_key].stride() != stride
        )
        if need_new:
            slot[self.layer_key] = torch.empty_strided(   # <── preserves layout
                size, stride,
                dtype=self.cpu_buf[key].dtype,
                device=self.device,
            )

        gpu_view = slot[self.layer_key]      # this is already non-contiguous if stride says so

        # copy row-major CPU view → strided GPU view
        flat_src = self.cpu_buf[key][:gpu_view.numel()]
        self.load_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.load_stream):
            gpu_view.copy_(flat_src.view(size), non_blocking=True)
            gpu_view.record_stream(self.offload_stream)

        return gpu_view
    
    def load_async_wait(self):
        """
        Instruct the current (default) stream to wait for the self.offload_stream.
        This ensures all CPU↔GPU transfers on self.offload_stream are finished before
        proceeding with compute on the default stream.
        """
        if not self.is_offload_enabled:
            return

        torch.cuda.current_stream().wait_stream(self.load_stream)
        torch.cuda.current_stream().wait_stream(self.offload_stream)

        if self.load_completed_event is not None:
            self.load_completed_event.wait()
            self.load_completed_event = None
