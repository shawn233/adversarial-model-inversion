from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from data import *
from model import *
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import datasets
import pandas as pd
import time


def get_dataset(dataset, train=True, inv=False):
    dataset = dataset.lower()
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset == "facescrub":
        return FaceScrub("./data/facescrub/", train=train, transform=transform)

    elif "celeba" in dataset:
        if train == False:
            raise ValueError("celebA does not provide test dataset")
        if dataset == "celeba":
            return CelebA("./data/celebA", transform=transform)
        else:
            # celeba_100 celeba_1000 celeba_10000
            n_lim = int(dataset.split("_")[1])
            return CelebA_lim("./data/celebA_{}".format(n_lim), n_lim=n_lim, transform=transform)

    elif "insta" in dataset:
        city = dataset.split("-")[1].lower()
        return Insta("./data/insta/", city, train=train, transform=None)

    elif dataset == "purchase":
        return Purchase("./data/purchase", train=train, transform=None, inv=inv)

    elif "location" in dataset:
        if dataset == "location":
            return Location("./data/location", train=train, transform=None, inv=inv)
        else:
            n_lim = dataset.split("_")[1]
            if n_lim in ["random", "dist"]:
                # location_random, location_dist
                return Location("./data/location", random=n_lim)
            else:
                # location_10, location_100, location_1000
                n_lim = int(n_lim)
                return Location_lim("./data/location", transform=None, n_lim=n_lim)

    elif dataset == "cifar100":
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        data = datasets.CIFAR100("./data/cifar100/", train=train, transform=transform, download=False)
        print("CIFAR100 size", len(data))
        return data

    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        data = datasets.MNIST("./data/mnist/", train=train, transform=transform, download=False)
        print("MNIST size", len(data))
        return data

    elif dataset == "fashion-mnist" or dataset == "fashionmnist":
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        data = datasets.FashionMNIST("./data/fashionmnist/", train=train, transform=transform, download=False)
        print("FASHION MNIST size", len(data))
        return data

    elif dataset == "pubfig":
        raise ValueError("pubfig has not been implemented")

    elif dataset == "sentiment140":

        raise ValueError("sentiment140 has not been implemented")

    else:
        raise ValueError("unknown dataset name")


def get_classifier(dataset, device):
    classifier = None
    if dataset == "facescrub":
        classifier = nn.DataParallel( FaceClassifier() ).to(device)
    elif "insta" in dataset:
        classifier = nn.DataParallel( InstaClassifier() ).to(device)
    elif dataset == "purchase":
        classifier = nn.DataParallel( PurchaseClassifier() ).to(device)
    elif dataset == "location":
        classifier = nn.DataParallel( LocationClassifier() ).to(device)
    else:
        raise Exception("Classifier unknown! Allowed: facescrub, insta*, purchase.")
    
    return classifier


def get_inversion(dataset, device):
    inversion = None
    if dataset == "facescrub":
        inversion = nn.DataParallel( FaceInversion() ).to(device)
    elif "insta" in dataset:
        inversion = nn.DataParallel( InstaInversion() ).to(device)
    elif dataset == "purchase":
        inversion = nn.DataParallel( PurchaseInversion() ).to(device) 
    elif dataset == "location":
        inversion = nn.DataParallel( LocationInversion() ).to(device)
    else:
        raise Exception("Inversion unknown! Allowed: facescrub, insta*.")
    
    return inversion


def train(classifier, inversion, log_interval, device, data_loader, optimizer, epoch):
    classifier.eval()
    inversion.train()

    n_feat = data_loader.dataset[0][0].view(-1).shape[0]

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, release=True)
        reconstruction = inversion(prediction)
        loss = F.mse_loss(reconstruction, data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t Average Loss: {:.6f}'.format(
                    epoch, batch_idx * data_loader.batch_size,
                    len(data_loader.dataset), loss.item()))

def test(classifier, inversion, device, data_loader, epoch, data_name, msg, plot):
    classifier.eval()
    inversion.eval()
    mse_loss = 0

    record = not plot
    eval_diff = not plot

    n_feat = data_loader.dataset[0][0].view(-1).shape[0]
    diff = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            if eval_diff:
                reconstruction_binary = torch.round(reconstruction)
                diff += torch.sum(torch.abs(reconstruction_binary - data)).item()

            if plot:
                truth = data[0:32]
                inverse = reconstruction[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                vutils.save_image(out, 'out_{}/recon_{}_{}.png'.format(data_name, msg.replace(" ", ""), epoch), normalize=False)
                plot = False
            
            if record:
                truth = data[0:16].long()
                inverse = torch.round(reconstruction[0:16]).long()
                out = torch.cat((inverse, truth))
                for i in range(16):
                    out[i*2] = inverse[i]
                    out[i*2+1] = truth[i]
                df = pd.DataFrame(out.cpu().numpy())
                df.to_csv(f"out_{data_name}/recon_{msg.replace(' ', '')}_{epoch}.csv")
                record = False

    mse_loss /= len(data_loader.dataset) * n_feat
    diff /= len(data_loader.dataset) * n_feat
    print('\nTest inversion model on {} set: Average MSE loss: {:.6f} Average diff: {:.4f}\n'.format(msg, mse_loss, diff))
    return mse_loss, diff


def main():
    help_t = "train preliminary inversion model from auxiliary datasets"
    parser = argparse.ArgumentParser(description=help_t)
    parser.add_argument("dataset", help="name of the training dataset", type=str)
    parser.add_argument("test_dataset", help="name of the target dataset", type=str)
    parser.add_argument("--ckpt-dataset", help="name of the checkpoint dataset", type=str, default=None)
    parser.add_argument("--use-best-ckpt", help="whether to use the best inversion checkpoint", action="store_true")
    parser.add_argument("--no-name-expansion", help="set to cancel the inversion name expansion", action="store_true")
    parser.add_argument('--batch-size', type=int, default=128, metavar='')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
    parser.add_argument('--epochs', type=int, default=100, metavar='')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='')
    parser.add_argument('--log-interval', type=int, default=5, metavar='')
    # parser.add_argument('--nc', type=int, default=1)
    # parser.add_argument('--ndf', type=int, default=128)
    # parser.add_argument('--ngf', type=int, default=128)
    # parser.add_argument('--nz', type=int, default=530)
    # parser.add_argument('--truncation', type=int, default=530)
    # parser.add_argument('--c', type=float, default=50.)
    parser.add_argument('--num_workers', type=int, default=1, metavar='')
    #parser.add_argument('--lim', type=int, default=10000, help="number of limited samples")
    parser.add_argument("--test-suffix", help="suffix of the target checkpoint", type=str, default="best")
    args = parser.parse_args()

    print("================================")
    print(args)
    print("================================")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = get_dataset(args.dataset, train=True, inv=True)

    # Inversion attack on TRAIN data of target classifier
    test1_set = get_dataset(args.test_dataset, train=True)
    # Inversion attack on TEST data of target classifier
    test2_set = get_dataset(args.test_dataset, train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test1_loader = torch.utils.data.DataLoader(test1_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    classifier = get_classifier(args.test_dataset, device)
    inversion = get_inversion(args.test_dataset, device)
    optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    # Load classifier
    path = f'./classifier/classifier_{args.test_dataset}_{args.test_suffix}.pth'
    try:
        checkpoint = torch.load(path)
        classifier.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        best_cl_acc = checkpoint['best_cl_acc']
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
    except Exception:
        print("=> load classifier checkpoint '{}' failed".format(path))
        return

    # Load inversion model checkpoint

    # if ckpt_dataset is not set, then use the training dataset as the ckpt dataset
    ckpt_dataset = args.dataset if args.ckpt_dataset is None else args.ckpt_dataset
    best_recon_loss = 999
    best_recon_diff = 999
    best_recon_epoch = 0
    start_epoch = 1
    
    # suffix determines on whether we use the best checkpoint
    # if the best ckpt is used, then suffix should be empty, otherwise it's _latest
    inv_path_suffix = "_best" if args.use_best_ckpt else "_latest"
    inv_path_load = "./inversion/inversion_{}{}.pth".format(ckpt_dataset, inv_path_suffix)
    if os.path.exists(inv_path_load):
        data_dict = torch.load(inv_path_load)
        inversion.load_state_dict(data_dict['model'])
        optimizer.load_state_dict(data_dict['optimizer'])
        start_epoch = data_dict['epoch']+1
        best_recon_loss = data_dict['best_recon_loss']
        best_recon_diff = data_dict.get('best_recon_diff', 999)
        print("=> load inversion checkpoint '{}' (epoch {}, recon {:.4f})".format(inv_path_load, start_epoch, best_recon_loss))
    
    inv_path = "./inversion/inversion_{}_latest.pth".format(ckpt_dataset) # path to save the latest trained inversion
    if not args.no_name_expansion and args.ckpt_dataset is not None:
        inv_path = "./inversion/inversion_{}_{}_latest.pth".format(ckpt_dataset, args.dataset)
    print("inversion path", inv_path)

    # Train inversion model
    os.makedirs("inversion", exist_ok=True)
    os.makedirs("out_{}".format(args.dataset), exist_ok=True)
    inv_best_path = "./inversion/inversion_{}_best.pth".format(ckpt_dataset) # path to save the best inversion
    if not args.no_name_expansion and args.ckpt_dataset is not None:
        inv_best_path = "./inversion/inversion_{}_{}_best.pth".format(ckpt_dataset, args.dataset)
    print("inversion best path", inv_best_path)

    time.sleep(3) # pause to check paths are all correct

    plot = True if args.test_dataset == "facescrub" else False # only plot for face classifiers

    for epoch in range(start_epoch, args.epochs + 1):
        train(classifier, inversion, args.log_interval, device, train_loader, optimizer, epoch)
        recon_loss, recon_diff = test(classifier, inversion, device, test1_loader, epoch, args.dataset, 'test1', plot)
        test(classifier, inversion, device, test2_loader, epoch, args.dataset, 'test2', plot)
        state = {
            'epoch': epoch,
            'model': inversion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_recon_loss': best_recon_loss, # L2 loss
            'best_recon_diff': best_recon_diff, # L1 loss
        }
        torch.save(state, inv_path)

        if recon_diff < best_recon_diff:
            best_recon_diff = recon_diff

        if recon_loss < best_recon_loss:
            best_recon_loss = recon_loss
            best_recon_epoch = epoch
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recon_loss': best_recon_loss,
                'best_recon_diff': best_recon_diff,
            }
            torch.save(state, inv_best_path)
            if plot:
                shutil.copyfile('out_{}/recon_test1_{}.png'.format(args.dataset, epoch), 'out_{}/best_test1.png'.format(args.dataset))
                shutil.copyfile('out_{}/recon_test2_{}.png'.format(args.dataset, epoch), 'out_{}/best_test2.png'.format(args.dataset))
            else:
                shutil.copyfile(f"out_{args.dataset}/recon_test1_{epoch}.csv", f"out_{args.dataset}/best_test1.csv")
                shutil.copyfile(f"out_{args.dataset}/recon_test2_{epoch}.csv", f"out_{args.dataset}/best_test2.csv")

        print(f"Best reconstruction loss {best_recon_loss:.4f} epoch {best_recon_epoch} | Best diff {best_recon_diff:.4f}")


if __name__ == "__main__":
    main()