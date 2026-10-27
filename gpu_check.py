import json

import modal

app = modal.App("inference-gpu-check")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.6.0",
    index_url="https://download.pytorch.org/whl/cu124",
)


@app.function(
    image=image,
    gpu="T4",
    max_containers=1,
    timeout=60,
    retries=0,
)
def check_gpu():
    import torch

    assert torch.cuda.is_available(), "CUDA is unavailable"

    values = torch.arange(8, device="cuda")
    squared = values.square()
    result = squared.cpu().tolist()

    assert squared.is_cuda
    assert result == [0, 1, 4, 9, 16, 25, 36, 49]

    return {
        "gpu": torch.cuda.get_device_name(0),
        "device": str(squared.device),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "gpu_memory_gib": (torch.cuda.get_device_properties(0).total_memory / 1024**3),
        "squares": result,
    }


@app.local_entrypoint()
def main():
    print(json.dumps(check_gpu.remote(), indent=2))
