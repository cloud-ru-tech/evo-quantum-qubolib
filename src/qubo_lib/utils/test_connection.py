"""
Test connection utility for qubo_lib.
Use this to verify CUDA/GPU availability before running GPU-accelerated solvers.
"""


def test_cuda_connection() -> dict:
    """
    Test CUDA/GPU connection and return device information.

    Returns:
        dict: Device information including:
            - 'cuda_available': bool - Whether CUDA is available
            - 'cuda_version': str - CUDA version (if available)
            - 'device_count': int - Number of CUDA devices (if available)
            - 'device_name': str - Name of the first CUDA device (if available)
            - 'device_memory': int - Total memory of the first device in MB (if available)
            - 'error': str - Error message if CUDA is not available (None if successful)
    """
    result = {
        'cuda_available': False,
        'cuda_version': None,
        'device_count': 0,
        'device_name': None,
        'device_memory': None,
        'error': None
    }

    try:
        import cupy as cp
        result['cuda_available'] = True

        # Get CUDA version
        result['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()

        # Get device count
        result['device_count'] = cp.cuda.runtime.getDeviceCount()

        if result['device_count'] > 0:
            # Get first device info
            device = cp.cuda.Device(0)
            result['device_name'] = device.name
            result['device_memory'] = device.mem_info[1] // (1024 * 1024)  # Convert bytes to MB

        # Run a small test to verify device is operational
        test_array = cp.array([1, 2, 3, 4, 5])
        result_array = test_array * 2
        assert cp.all(result_array == cp.array([2, 4, 6, 8, 10]))

    except ImportError:
        result['error'] = "CuPy is not installed. Install it with: pip install cupy-cuda12x (for CUDA 12.x)"
    except Exception as e:
        result['cuda_available'] = False
        result['error'] = f"CUDA error: {str(e)}"

    return result


def print_connection_info():
    """
    Test connection and print formatted device information.
    """
    info = test_cuda_connection()

    print("=" * 60)
    print("QUBO_lib - CUDA/GPU Connection Test")
    print("=" * 60)

    if info['cuda_available']:
        print(f"✓ CUDA is available!")
        print(f"  CUDA Version: {info['cuda_version']}")
        print(f"  Device Count: {info['device_count']}")

        if info['device_count'] > 0:
            print(f"\n  Primary Device:")
            print(f"    Name: {info['device_name']}")
            print(f"    Total Memory: {info['device_memory']} MB")
            print(f"\n✓ GPU is ready to use!")
        else:
            print("\n⚠ No CUDA devices found")
    else:
        print("✗ CUDA is not available")
        print(f"  Error: {info['error']}")

    print("=" * 60)

    return info['cuda_available']


if __name__ == "__main__":
    print_connection_info()
