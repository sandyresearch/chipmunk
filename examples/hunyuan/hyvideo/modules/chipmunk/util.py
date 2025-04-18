import types
import torch

def offload(model):
    """
    (1) Create CPU state dict with pinned weights
    (2) Copy to GPU before forward pass
    (3) Delete from GPU after forward pass
    """
    original_device = model.device

    # Step (1): Create CPU pinned copies of every parameter
    cpu_weights = {}
    for name, param in model.named_parameters():
        cpu_weights[name] = param.detach().cpu().clone().pin_memory()
        # Replace the model's parameter with a zero tensor on CPU (so GPU is not used)
        param.data = torch.tensor(0.0, device='cpu', dtype=param.dtype)
    # model.to('cpu')

    original_forward = model.forward

    # Step (2) and (3): Wrap forward pass to copy weights in/out
    def forward(self, *args, **kwargs):
        # Load parameters onto GPU
        for name, param in self.named_parameters():
            param.data = torch.empty_like(cpu_weights[name], device='cuda')
            param.data.copy_(cpu_weights[name], non_blocking=False)
        # model.to(original_device)

        # Run the original forward
        out = original_forward(*args, **kwargs)

        # Unload after forward (replace with zeros on CPU again)
        for name, param in self.named_parameters():
            param.data = torch.tensor(0.0, device='cpu', dtype=param.dtype)
        # model.to('cpu')
        torch.cuda.empty_cache()

        return out

    model.forward = types.MethodType(forward, model)

    return model
