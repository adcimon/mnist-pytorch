import torch

def print_backend_support():

    print('Processing unit: {}'.format('GPU' if torch.cuda.is_available() else 'CPU'))

    if torch.cuda.is_available():

        for i in range(torch.cuda.device_count()):
            if i == torch.cuda.current_device():
                print('\t[x] {}'.format(torch.cuda.get_device_name(i)))
            else:
                print('\t[ ] {}'.format(torch.cuda.get_device_name(i)))

def print_device_usage(device):

    if torch.device.type == 'cuda':
        print(torch.cuda.get_device_name(device))
        print('Memory allocated: {} GB'.format(round(torch.cuda.memory_allocated(device) / 1024**3, 1)))
        print('Memory cached: {} GB'.format(round(torch.cuda.memory_reserved(device) / 1024**3, 1)))