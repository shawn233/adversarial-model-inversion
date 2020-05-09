from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch


def onehot_to_label(one_hot):
    return np.argmax(one_hot, axis=1)

class FaceScrub(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'facescrub.npz'))
        actor_images = input['actor_images']
        actor_labels = input['actor_labels']
        actress_images = input['actress_images']
        actress_labels = input['actress_labels']

        data = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0)
        print("facescrub data shape", data.shape)
        print("facescrub label shape", labels.shape)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)

        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
        else:
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        data = []
        for i in range(10):
            data.append(np.load(os.path.join(self.root, 'celebA_64_{}.npy').format(i + 1)))
        data = np.concatenate(data, axis=0)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)
        labels = np.array([0] * len(data))

        print(f"celeba data shape {data.shape}")
        print(f"celeba label shape {labels.shape}")

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CelebA_lim(Dataset):

    def __init__(self, root, n_lim=None, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.n_lim = n_lim

        if n_lim is None:
            # then we assume that root is in the form celebA_<n_lim>
            if root[-1] == "/":
                root = root[:-1]
            n_lim = int(os.path.basename(root).split("_")[1])
            self.n_lim = n_lim

        path = os.path.join(self.root, "celebA_{}.npy".format(n_lim))
        data = np.load(path)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)
        labels = np.array([0] * len(data))
        print("celebA-lim-{} data shape".format(n_lim), data.shape)

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target



class Insta(Dataset):

    def __init__(self, root, city, transform=None, target_transform=None, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        path = os.path.join(self.root, f"insta-{city.lower()}.npz")
        obj = np.load(path)
        data, labels = obj["features"], obj["labels"]

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)
        
        print(f"Insta-{city.upper()} data shape {data.shape}")
        print(f"Insta-{city.upper()} label shape {labels.shape}")

        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = torch.from_numpy(data[perm]).float()
        labels = torch.from_numpy(labels[perm]).long()

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
        else:
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat, target = self.data[index], self.labels[index]

        if self.transform is not None:
            feat = self.transform(feat)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feat, target 



class Purchase(Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=True, inv=False):
        # if inv = True, then the inversion dataset will be used
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if inv:
            path = os.path.join(self.root, "purchase-inv.npz")
            obj = np.load(path)
            data, labels = obj['feat'], obj['label']-1

            print(f"Purchase-inv data shape: {data.shape}")
            print(f"Purchase-inv label shape: {labels.shape}")

            self.data = torch.from_numpy(data).float()
            self.labels = torch.from_numpy(labels).long()

        else:
            path = os.path.join(self.root, "purchase-train.npz")
            obj = np.load(path)
            data, labels = obj['feat'], obj['label']-1
            
            print(f"Purchase-train data shape: {data.shape}")
            print(f"Purchase-train label shape: {labels.shape}")

            np.random.seed(666)
            perm = np.arange(len(data))
            np.random.shuffle(perm)
            data = torch.from_numpy(data[perm]).float()
            labels = torch.from_numpy(labels[perm]).long()

            if train:
                self.data = data[0:int(0.8 * len(data))]
                self.labels = labels[0:int(0.8 * len(data))]
            else:
                self.data = data[int(0.8 * len(data)):]
                self.labels = labels[int(0.8 * len(data)):]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat, target = self.data[index], self.labels[index]

        if self.transform is not None:
            feat = self.transform(feat)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feat, target



class Location(Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=True, inv=False, random=None):
        # if inv = True, then the inversion dataset will be used
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if random is not None:
            path = os.path.join(self.root, f"location-inv_{random}.npy")
            data = np.load(path)
            labels = np.zeros(len(data))

            print(f"Location-inv {random} data shape: {data.shape}")
            print(f"Location-inv {random} label shape: {labels.shape}")

            self.data = torch.from_numpy(data).float()
            self.labels = torch.from_numpy(labels).long()

        elif inv:
            path = os.path.join(self.root, "location-inv.npz")
            obj = np.load(path)
            data, labels = obj['feat'], obj['label']-1

            print(f"Location-inv data shape: {data.shape}")
            print(f"Location-inv label shape: {labels.shape}")

            self.data = torch.from_numpy(data).float()
            self.labels = torch.from_numpy(labels).long()

        else:
            path = os.path.join(self.root, "location-train.npz")
            obj = np.load(path)
            data, labels = obj['feat'], obj['label']-1
            
            print(f"Location-train data shape: {data.shape}")
            print(f"Location-train label shape: {labels.shape}")

            np.random.seed(666)
            perm = np.arange(len(data))
            np.random.shuffle(perm)
            data = torch.from_numpy(data[perm]).float()
            labels = torch.from_numpy(labels[perm]).long()

            if train:
                self.data = data[0:int(0.8 * len(data))]
                self.labels = labels[0:int(0.8 * len(data))]
            else:
                self.data = data[int(0.8 * len(data)):]
                self.labels = labels[int(0.8 * len(data)):]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat, target = self.data[index], self.labels[index]

        if self.transform is not None:
            feat = self.transform(feat)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feat, target  



class Location_lim(Dataset):

    def __init__(self, root, n_lim=None, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.n_lim = n_lim

        path = os.path.join(self.root, "location-inv_{}.npz".format(n_lim))
        obj = np.load(path)
        data, labels = obj['feat'], obj['label'] - 1
        

        print("Location-lim-{} data shape".format(n_lim), data.shape)

        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat, target = self.data[index], self.labels[index]

        if self.transform is not None:
            feat = self.transform(feat)

        return feat, target  