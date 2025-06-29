import torch

def check_gpu_availability():
    """
    Checks for CUDA-enabled GPU availability and prints detailed information.
    """
    print("--- GPU Availability Check ---")

    # 1. Check if CUDA is available
    is_available = torch.cuda.is_available()

    if is_available:
        print("\n✅ Success! PyTorch can access your CUDA-enabled GPU.")

        # 2. Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"   Number of GPUs available: {gpu_count}")

        # 3. Get details for the primary GPU (device 0)
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"   Primary GPU (device {current_device}): {gpu_name}")

        # 4. Check CUDA version PyTorch was built with
        cuda_version = torch.version.cuda
        print(f"   PyTorch was built with CUDA version: {cuda_version}")

        print("\nYour system is ready for GPU-accelerated training.")

    else:
        print("\n❌ Error: PyTorch cannot find a CUDA-enabled GPU.")
        print("   The training script will run on the CPU, which will be extremely slow.")
        print("\n   Possible reasons:")
        print("   1. NVIDIA drivers are not installed or are outdated.")
        print("   2. You installed a version of PyTorch without CUDA support.")
        print("   3. Your GPU is not from NVIDIA or does not support CUDA.")

    print("\n----------------------------")


if __name__ == "__main__":
    check_gpu_availability()