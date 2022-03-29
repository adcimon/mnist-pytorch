import torch

def print_backend_support():

    cuda_built = torch.backends.cuda.is_built()
    cuda_available = torch.cuda.is_available()

    cudnn_version = torch.backends.cudnn.version()
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_enabled = torch.backends.cudnn.enabled

    mkl_available = torch.backends.mkl.is_available()
    mkldnn_available = torch.backends.mkldnn.is_available()
    openmp_available = torch.backends.openmp.is_available()

    print('CUDA: {} | {}'.format(
        'Built' if cuda_built else 'Not Built',
        'Available' if cuda_available else 'Not Available'))

    print('cuDNN: {} | {} ({})'.format(
        'Available' if cudnn_available else 'Not Available',
        'Enabled' if cudnn_enabled else 'Not Enabled',
        cudnn_version))

    print('MKL: {}'.format('Available' if mkl_available else 'Not Available'))
    print('MKL-DNN: {}'.format('Available' if mkldnn_available else 'Not Available'))
    print('OpenMP: {}'.format('Available' if openmp_available else 'Not Available'))

    print('Processing unit: {}'.format('GPU' if cuda_available else 'CPU'))
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            if i == torch.cuda.current_device():
                print('[x] {}'.format(torch.cuda.get_device_name(i)))
            else:
                print('[ ] {}'.format(torch.cuda.get_device_name(i)))

def print_device_usage(device):

    if torch.device.type == 'cuda':
        print(torch.cuda.get_device_name(device))
        print('Memory allocated: {} GB'.format(round(torch.cuda.memory_allocated(device) / 1024**3, 1)))
        print('Memory cached: {} GB'.format(round(torch.cuda.memory_reserved(device) / 1024**3, 1)))
