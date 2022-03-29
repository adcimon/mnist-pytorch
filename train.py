import argparse
import torch

import utils
import dataset
import model
from test import test

def train(device, net, criterion, optimizer, loader):

    net.train()

    for batch, (data, target) in enumerate(loader):

        data = data.to(device)
        target = target.to(device)

        # Reset gradients.
        optimizer.zero_grad()

        # Forward.
        output = net(data)

        # Loss function.
        loss = criterion(output, target)

        # Back-propagation, calculates gradients automatically.
        loss.backward()

        # Perform a step, updating the weights.
        optimizer.step()

        if batch % 10 == 0:
            print('[{:.0f}%]\t{:05d}/{}\tLoss: {:.6f}'.format(
                100.0 * batch / len(loader),
                batch * len(data),
                len(loader.dataset),
                loss.item()))

def main():

    parser = argparse.ArgumentParser(description='MNIST PyTorch: Training')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--train-batch-size', type=int, default=64, help='Training batch size (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='Test batch size (default: 1000)')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs to train (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=1, help='Learning rate (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--model-name', type=str, default="mnist", help='Model name in disk (default: mnist.pt)')
    args = parser.parse_args()

    print('+------------------------------')
    print('| Arguments')
    print('+------------------------------')
    for arg in vars(args):
        print('{}: {}'.format(arg.replace('_', ' ').capitalize(), getattr(args, arg)))

    # Set the seed for generating random numbers.
    torch.manual_seed(args.seed)

    # Get the backend device.
    utils.print_backend_support()
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')

    # Data loaders.
    train_args = { 'batch_size': args.train_batch_size }
    test_args = { 'batch_size': args.test_batch_size }
    if cuda_available:
        cuda_args = { 'num_workers': 1, 'pin_memory': True, 'shuffle': True }
        train_args.update(cuda_args)
        test_args.update(cuda_args)

    train_loader = torch.utils.data.DataLoader(dataset.training_set(), **train_args)
    test_loader = torch.utils.data.DataLoader(dataset.test_set(), **test_args)

    # Initialize the neural network.
    net = model.Model()
    net.to(device)

    # Loss function.
    criterion = torch.nn.CrossEntropyLoss()
    print('Criterion: {}'.format(criterion))

    # Optimizer.
    optimizer = torch.optim.Adadelta(net.parameters(), lr=args.learning_rate)
    print('Optimizer: {}'.format(optimizer))

    # Training and evaluation.
    print('+------------------------------')
    print('| Training')
    print('+------------------------------')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs):
        print('[Epoch {}/{}]'.format(epoch, args.epochs))
        train(device, net, criterion, optimizer, train_loader)
        test(device, net, criterion, test_loader)
        scheduler.step()

    # Save the model to disk.
    print('Saving the model to disk: {}.pt'.format(args.model_name))
    torch.save(net.state_dict(), args.model_name + '.pt')
    torch.save(optimizer.state_dict(), args.model_name + '_optimizer.pt')

if __name__ == '__main__':
    main()
