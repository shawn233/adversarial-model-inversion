from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from data import *
from model import Classifier, Inversion
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import datasets
import logging
from train_inversion import get_dataset
import time


def test(classifier, inversion, device, data_loader, msg, plot):
    classifier.eval()
    inversion.eval()
    mse_loss = 0
    plot_range = 50
    display_range = 20
    logging.info(f"({msg}) Evaluation in progress ...")
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            if idx % display_range == 0:
                logging.info(f"[{(idx+1)*len(target)} / {len(data_loader.dataset)}]")

            if plot and idx % plot_range == 0:
                truth = data[0:32]
                inverse = reconstruction[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                vutils.save_image(out, f"out/recon_{idx}_{msg}.png", normalize=False)

    mse_loss /= len(data_loader.dataset) * 64 * 64
    logging.info(f'({msg} Done! Average MSE loss: {mse_loss:.6f}\n')
    return mse_loss


def main():
    parser = argparse.ArgumentParser(description="evaluate trained models on certain datasets")
    parser.add_argument(
        "classifier", help="path of the classifier checkpoint", 
        type=str, action="store")
    parser.add_argument(
        "inversion", help="path of the inversion checkpoint",
        type=str, action="store")
    parser.add_argument(
        "dataset", help="name of the testing dataset",
        type=str, action="store")
        
    parser.add_argument('--batch-size', type=int, default=256, metavar='')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='')
    parser.add_argument('--log-interval', type=int, default=5, metavar='')
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--nz', type=int, default=530)
    parser.add_argument('--truncation', type=int, default=530)
    parser.add_argument('--c', type=float, default=50.)
    parser.add_argument('--num_workers', type=int, default=1, metavar='')
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()
    
    logging.basicConfig(
        format="[%(levelname)s] %(message)s", level=logging.DEBUG,
        filename=args.log_file, filemode="a")
    os.makedirs("./out", exist_ok=True)

    if (not os.path.exists(args.classifier)
            or not os.path.exists(args.inversion)):
        logging.error("classifier / inversion does not exist")
        raise Exception("Essential models not found")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    # Loading classifier ...
    classifier = None
    if args.dataset == "facescrub":
        classifier = nn.DataParallel( FaceClassifier(nc=args.nc, ndf=args.ndf, nz=args.nz) ).to(device)
    elif "insta" in args.dataset:
        classifier = nn.DataParallel( InstaClassifier() ).to(device)
    else:
        raise Exception("Classifier unknown! Allowed: facescrub, insta*.")

    data_dict = torch.load(args.classifier)
    classifier.load_state_dict(data_dict["model"])
    logging.info(
        f"==> load classifier checkpoint: epoch {data_dict['epoch']}"
        f" accuracy {data_dict['best_cl_acc']}")

    # Loading inversion ...
    inversion = None
    if args.dataset == "facescrub":
        inversion = nn.DataParallel( FaceInversion(
                        nc=args.nc, ndf=args.ndf, nz=args.nz, 
                        truncation=args.truncation, c=args.c) ).to(device)
    elif "insta" in args.dataset:
        inversion = nn.DataParallel( InstaInversion() ).to(device)
    else:
        raise Exception("Inversion unknown! Allowed: facescrub, insta*.")

    data_dict = torch.load(args.inversion)
    inversion.load_state_dict(data_dict["model"])
    logging.info(
        f"==> load inversion checkpoint: epoch {data_dict['epoch']}"
        f" recon_loss {data_dict['best_recon_loss']}")

    # Loading test dataset
    test1_set = get_dataset(args.dataset, train=True) # train data
    test2_set = get_dataset(args.dataset, train=False) # test data

    test1_loader = torch.utils.data.DataLoader(test1_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    msg = f"{os.path.basename(args.inversion).split('.')[0]}-{time.strftime('%m%d%H%M%S')}"
    plot = True if args.test_dataset == "facescrub" else False # only plot for face classifiers

    test(classifier, inversion, device, test1_loader, "test1-"+msg, plot)
    test(classifier, inversion, device, test2_loader, "test2-"+msg, plot) 


if __name__ == "__main__":
    main()