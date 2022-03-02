import argparse
import torch

import dataset
import model

def test(device, net, criterion, loader):

    net.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():

        for data, target in loader:

            data = data.to(device)
            target = target.to(device)

            # Forward.
            output = net(data)

            # Sum up batch loss.
            test_loss += criterion(output, target).item()

            # Get the index of the max log-probability.
            prediction = output.argmax(dim=1, keepdim=True)

            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(loader.dataset)

    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(loader.dataset),
        100.0 * correct / len(loader.dataset)))

def main():

    parser = argparse.ArgumentParser(description='MNIST PyTorch: Testing')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='Test batch size (default: 1000)')
    parser.add_argument('--model-name', type=str, default="mnist", help='Model name in disk (default: mnist.pt)')
    args = parser.parse_args()

    print('+------------------------------')
    print('| Settings')
    print('+------------------------------')
    for arg in vars(args):
        print('{}: {}'.format(arg.replace('_', ' ').capitalize(), getattr(args, arg)))

    # Set the seed for generating random numbers.
    torch.manual_seed(args.seed)

    # Load the model from disk.
    net = model.Model()
    net.load_state_dict(torch.load(args.model_name + '.pt'))

    # Loss function.
    criterion = torch.nn.CrossEntropyLoss()
    print('Criterion: {}'.format(criterion))

    # Data loader.
    test_loader = torch.utils.data.DataLoader(dataset.test_set(), batch_size=args.test_batch_size)

    # Testing
    print('+------------------------------')
    print('| Testing')
    print('+------------------------------')
    test('cpu', net, criterion, test_loader)

if __name__ == '__main__':
    main()