"""Report whether the active Python environment can run Windows CUDA training."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-no-cuda", action="store_true",
                        help="Validate installed packages without requiring a CUDA GPU")
    args = parser.parse_args()
    problems = []
    if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
        problems.append(f"Python 3.11 or 3.12 is required; found {sys.version.split()[0]}")
    try:
        import torch
        import torchvision
        import timm
        import safetensors
        import sklearn
        import yaml
        from PIL import Image
    except ImportError as error:
        print(f"Missing dependency: {error}")
        return 2

    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"torchvision:  {torchvision.__version__}")
    print(f"timm:         {timm.__version__}")
    print(f"safetensors:  {safetensors.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"PyYAML:       {yaml.__version__}")
    print(f"Pillow:       {Image.__version__}")
    if timm.__version__ != "1.0.30":
        problems.append(f"timm 1.0.30 is required; found {timm.__version__}")
    if "convnext_base.fb_in22k" not in timm.list_models(pretrained=True):
        problems.append("timm does not register convnext_base.fb_in22k")
    print(f"CUDA build:   {torch.version.cuda}")
    print(f"CUDA visible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / (1024 ** 3)
            print(f"GPU {index}:        {properties.name} ({memory_gib:.1f} GiB)")
        probe = torch.randn(64, 64, device="cuda")
        _ = probe @ probe.T
        torch.cuda.synchronize()
        print("CUDA allocation and matrix multiply: OK")
    elif not args.allow_no_cuda:
        problems.append("PyTorch cannot access a CUDA GPU; install a CUDA-enabled wheel and check the NVIDIA driver")

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    print("Environment check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
