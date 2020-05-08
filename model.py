from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


# Facescrub dataset
class FaceClassifier(nn.Module):
    def __init__(self, nc=1, ndf=128, nz=530):
        super(FaceClassifier, self).__init__()

        self.nc = nc # input channel
        self.ndf = ndf 
        self.nz = nz # output dimension

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release=False):

        x = x.view(-1, 1, 64, 64)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = self.fc(x)

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)

class FaceInversion(nn.Module):
    def __init__(self, nc=1, ngf=128, nz=530, truncation=530, c=50.):
        super(FaceInversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = c

        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        topk, indices = torch.topk(x, self.truncation) # return the top-k values and their indices
        topk = torch.clamp(torch.log(topk), min=-1000) + self.c # crop values that are too large or too small
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, topk)

        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, 1, 64, 64)
        return x



# for all fully connected models
class FcClassifier(nn.Module):
    # Fully connected classifier

    def __init__(self, model, batch_norm=False, tanh=False):
        # model is a list, translated into consecutive linear layers
        super(FcClassifier, self).__init__()

        assert(len(model) > 2) # at least one hidden layer

        self.nz = model[0]
        self.nc = model[-1]

        self.model = [nn.Linear(model[0], model[1])] # 0->1
        
        for i in range(2, len(model)): # start from 1->2
            if batch_norm:
                self.model.append(nn.BatchNorm1d(model[i-1]))
            self.model.extend([
                nn.Tanh() if tanh else nn.ReLU(True), 
                nn.Dropout(0.5), 
                nn.Linear(model[i-1], model[i])])

        self.fc = nn.Sequential(*self.model)

        print("\n"+repr(self)+"\n")


    def forward(self, x, release=False):
        x = x.view(-1, self.nz)
        x = self.fc(x)
        x = x.view(-1, self.nc)

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)



class FcInversion(nn.Module):
    # Fully connected inversion model

    def __init__(self, model, batch_norm=True, tanh=True):
        # model is a list, translated into consecutive linear layers
        super(FcInversion, self).__init__()

        assert(len(model) > 2) # at least one hidden layer

        self.nz = model[-1]
        self.nc = model[0]

        self.model = [nn.Linear(model[0], model[1])] # 0->1
        
        for i in range(2, len(model)): # start from 1->2
            if batch_norm:
                self.model.append(nn.BatchNorm1d(model[i-1]))
            self.model.extend([
                nn.Tanh() if tanh else nn.ReLU(True), 
                nn.Dropout(0.5), 
                nn.Linear(model[i-1], model[i])])

        self.model.append(nn.Sigmoid()) # output values between 0 and 1

        self.fc = nn.Sequential(*self.model)

        print("\n"+repr(self)+"\n")


    def forward(self, x, release=False):
        x = x.view(-1, self.nc)
        x = self.fc(x)
        x = x.view(-1, self.nz)

        return x



# Insta dataset
class InstaClassifier(FcClassifier):
    def __init__(self, **kwargs):
        super(InstaClassifier, self).__init__([168, 32, 16, 9])


class InstaInversion(FcInversion):
    def __init__(self, **kwargs):
        super(InstaInversion, self).__init__([9, 16, 32, 64, 168])


# Purchase dataset
class PurchaseClassifier(FcClassifier):
    def __init__(self, **kwargs):
        super(PurchaseClassifier, self).__init__([600, 256, 128, 100], batch_norm=False, tanh=True, **kwargs)


class PurchaseInversion(FcInversion):
    def __init__(self, **kwargs):
        super(PurchaseInversion, self).__init__([100, 128, 256, 600], **kwargs)


# Location dataset
class LocationClassifier(FcClassifier):
    def __init__(self, **kwargs):
        super(LocationClassifier, self).__init__([446, 128, 30], batch_norm=False, tanh=True, **kwargs)


class LocationInversion(FcInversion):
    def __init__(self, **kwargs):
        super(LocationInversion, self).__init__([30, 128, 256, 446], **kwargs)