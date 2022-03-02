import argparse
import torch
import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

def training_set():
    dataset = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
    return dataset

def test_set():
    dataset = torchvision.datasets.MNIST('data', train=False, transform=transform)
    return dataset

def main():

    parser = argparse.ArgumentParser(description='MNIST PyTorch: Download Dataset')
    args = parser.parse_args()

    print('Downloading MNIST dataset...')

    # Train set.
    print('Train Set')
    trainset = training_set()
    print(trainset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    train_data = enumerate(train_loader)
    batch, (data, target) = next(train_data)
    print('Shape: ' + str(data.shape))

    # Test set.
    print('Test Set')
    testset = test_set()
    print(testset)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    test_data = enumerate(test_loader)
    batch, (data, target) = next(test_data)
    print('Shape: ' + str(data.shape))

if __name__ == '__main__':
    main()