from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import torch.nn.functional as F
from data import FaceScrub
from model import *
from train_inversion import get_dataset, get_classifier
import shutil

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument("dataset", type=str, help="dataset name")
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
parser.add_argument('--epoch-list', help="list of epochs to be stored", type=int, nargs="+", action="store")
parser.add_argument('--lr', type=float, default=0.0002, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
# parser.add_argument('--nc', type=int, default=1)
# parser.add_argument('--ndf', type=int, default=128)
# parser.add_argument('--nz', type=int, default=530)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument(
    "--ckpt-list", help="list of checkpoints to be tested, e.g.best, latest, epoch50", 
    type=str, nargs="+", action="store")

def train(classifier, log_interval, device, data_loader, optimizer, epoch):
    classifier.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        #print(data.shape, target.shape)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = classifier(data)
        #print(output.shape)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.max(1, keepdim=True)[1] # so max(...)[1] is the indices of maximum elements
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.4f}%)'.format(
                    epoch, batch_idx * len(data),
                    len(data_loader.dataset), loss.item(), correct, 
                    min( len(data_loader.dataset), (batch_idx+1) * data_loader.batch_size ),
                    100. * correct / min( len(data_loader.dataset), (batch_idx+1) * data_loader.batch_size ) ))

    return correct / len(data_loader.dataset)

def test(classifier, device, data_loader):
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return correct / len(data_loader.dataset)

def main():
    args = parser.parse_args()
    print("================================")
    print(args)
    print("================================")
    os.makedirs("classifier", exist_ok=True)

    # initialize torch parameters
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    torch.manual_seed(args.seed)

    # load training and testing datasets
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = get_dataset(args.dataset, train=True)
    test_set = get_dataset(args.dataset, train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # create and restore the classifier as well as the optimizer
    classifier = get_classifier(args.dataset, device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    # an additional function: evaluate trained checkpoints
    # activate by using the --ckpt-list argument in the command line
    # once activated, the training process will not carry on
    if args.ckpt_list is not None:
        test_ckpt_list = list(args.ckpt_list)
        print(f"Doing checkpoint evaluation! {test_ckpt_list}")
        for ckpt in test_ckpt_list:
            path = f"./classifier/classifier_{args.dataset}_{ckpt}.pth"

            print(f"\nLoading {ckpt} from {path} ...  ", end="")
            data_dict = torch.load(path)
            classifier.load_state_dict(data_dict["model"])
            optimizer.load_state_dict(data_dict["optimizer"])
            print("done!")

            print("Evaluating ...  ")
            train_acc = test(classifier, device, train_loader)
            test_acc = test(classifier, device, test_loader)
            print("done!\n")

            print(
                f"Report:\nTrain accuracy: {100.*train_acc:.4f}%\n"
                f"Test accuracy: {100.*test_acc:.4f}%")
        return

    best_cl_acc = 0
    best_cl_epoch = 0

    # Load classifier checkpoint
    latest_ckpt_path = f"./classifier/classifier_{args.dataset}_latest.pth"
    start_epoch = 1

    if os.path.exists(latest_ckpt_path):
        data_dict = torch.load(latest_ckpt_path)
        classifier.load_state_dict(data_dict["model"])
        start_epoch = data_dict["epoch"] + 1
        optimizer.load_state_dict(data_dict["optimizer"])
        best_cl_acc = data_dict["best_cl_acc"]

    # Train classifier
    epoch_list = list(args.epoch_list)
    epoch_list.sort()

    next_epoch_ind = 0 # starts from epoch_list[0]
    for next_epoch_ind, epoch in enumerate(epoch_list):
        if start_epoch <= epoch:
            break 
    print(f"next epoch ind starts from {next_epoch_ind} ({epoch_list[next_epoch_ind]})")

    acc_log_path = f"./classifier/classifier_{args.dataset}_log.txt"
    with open(acc_log_path, "a") as f:
        f.write("\n" + repr(classifier) + "\n")

    best_ckpt_path = f"./classifier/classifier_{args.dataset}_best.pth"

    for epoch in range(start_epoch, epoch_list[-1] + 1):
        print("[epoch {0}] epoch {0} starts ...".format(epoch))
        train_acc = train(classifier, args.log_interval, device, train_loader, optimizer, epoch)
        print("[epoch {0}] train function returned".format(epoch))
        cl_acc = test(classifier, device, test_loader)
        print("[epoch {0}] test function returned".format(epoch))
        
        state = {
            'epoch': epoch,
            'model': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_acc': train_acc,
            'best_cl_acc': best_cl_acc,
        }
        torch.save(state, latest_ckpt_path)

        if epoch == epoch_list[next_epoch_ind]:
            epoch_ckpt_path = f"./classifier/classifier_{args.dataset}_epoch{epoch_list[next_epoch_ind]}.pth"
            shutil.copy2(latest_ckpt_path, epoch_ckpt_path)
            with open(acc_log_path, "a") as acc_log_fd:
                acc_log_fd.write(
                    f"epoch {epoch}\ttrain {train_acc}\ttest "
                    f"{cl_acc}\tbest {best_cl_acc}\n")
            next_epoch_ind += 1
        
        if cl_acc > best_cl_acc:
            best_cl_acc = cl_acc
            best_cl_epoch = epoch
            shutil.copy2(latest_ckpt_path, best_ckpt_path)

        print(f"Best accuracy {100.*best_cl_acc:.4f}% epoch {best_cl_epoch}")
            

if __name__ == '__main__':
    main()
