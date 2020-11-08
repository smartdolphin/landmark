# coding: utf-8
from loss import *

import pandas as pd
import numpy as np
import os
import time
import math

import argparse
import torch

from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import Conv2d, AdaptiveAvgPool2d, Linear
from torch.nn.parameter import Parameter
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch_optimizer

from efficientnet_pytorch import EfficientNet

# arguments
# train_csv_exist, test_csv_exist는 glob.glob이 생각보다 시간을 많이 잡아먹어서 iteration 시간을 줄이기 위해 생성되는 파일입니다.
# 이미 생성되어 있을 경우 train_csv_exist.csv 파일로 Dataset을 생성합니다.
parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', dest='train_dir', default="./public/train/")
parser.add_argument('--train_csv_dir', dest='train_csv_dir', default="./public/train.csv")
parser.add_argument('--train_csv_exist_dir', dest='train_csv_exist_dir', default="./public/train_exist.csv")

parser.add_argument('--test_dir', dest='test_dir', default="./public/test/")
parser.add_argument('--test_csv_dir', dest='test_csv_dir', default="./public/sample_submission.csv")
parser.add_argument('--test_csv_exist_dir', dest='test_csv_exist_dir', default="./public/sample_submission_exist.csv")

parser.add_argument('--test_csv_submission_dir', dest='test_csv_submission_dir', default="./public/my_submission.csv")
parser.add_argument('--model_dir', dest='model_dir', default="./ckpt/")
parser.add_argument('--resume', dest='resume', default=None)

parser.add_argument('--n_classes', dest='n_classes', type=int, default=1049)
parser.add_argument('--max_size', dest='max_size', type=int, default=256)
parser.add_argument('--image_size', dest='image_size', type=int, default=224)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--num_workers', dest='num_workers', type=int, default=16)
parser.add_argument('--log_freq', dest='log_freq', type=int, default=10)

parser.add_argument('--depth', dest='depth', type=int, default=3)
parser.add_argument('--arcface_s', dest='arcface_s', type=float, default=35)
parser.add_argument('--arcface_m', dest='arcface_m', type=float, default=0.4)
parser.add_argument('--crit', dest='crit', type=str, default='bce')
args = parser.parse_args()


# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'

# 경로 생성
if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)

# 파이토치 Dataset 생성 for Train / Test
class TrainDataset(Dataset):
    def __init__(self, args, transform=None):
        self.train_dir = args.train_dir
        self.train_csv_dir = args.train_csv_dir
        self.train_csv_exist_dir = args.train_csv_exist_dir
        self.args = args
        self.transform = transform
        self.train_image = list()
        self.train_label = list()
        if not os.path.isfile(self.train_csv_exist_dir) :
            self.train_csv = pd.read_csv(self.train_csv_dir, encoding='utf-8')
            self.train_csv_exist = self.train_csv.copy()
            self.load_full_data()
            self.train_csv_exist.to_csv(self.train_csv_exist_dir, index=False)
        else:
            self.load_exist_data()

    def load_full_data(self):
        for i in tqdm(range(len(self.train_csv))):
            filename = self.train_csv['id'][i]
            fullpath = glob(self.train_dir + "*/*/" + filename.replace('[', '[[]') + ".JPG")[0]
            label = self.train_csv['landmark_id'][i]
            self.train_csv_exist.loc[i,'id'] = fullpath
            self.train_image.append(fullpath)
            self.train_label.append(label)


    def load_exist_data(self):
        self.train_csv_exist = pd.read_csv(self.train_csv_exist_dir, encoding='utf-8')
        for i in tqdm(range(len(self.train_csv_exist))):
            fullpath = self.train_csv_exist['id'][i]
            label = self.train_csv_exist['landmark_id'][i]
            self.train_image.append(fullpath)
            self.train_label.append(label)


    def __len__(self):
        return len(self.train_image)

    def __getitem__(self, idx):
        image = Image.open(self.train_image[idx])
        if self.transform is not None:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = image.resize((self.args.image_size, self.args.image_size))
            image = np.array(image) / 255.
            image = np.transpose(image, axes=(2, 0, 1))
        label = self.train_label[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, args, transform=None):
        self.test_dir = args.test_dir
        self.test_csv_dir = args.test_csv_dir
        self.test_csv_exist_dir = args.test_csv_exist_dir
        self.args = args
        self.transform = transform
        self.test_image = list()
        self.test_label = list()
        if not os.path.isfile(self.test_csv_exist_dir):
            self.test_csv = pd.read_csv(self.test_csv_dir, encoding='utf-8')
            self.test_csv_exist = self.test_csv.copy()
            self.load_full_data()
            self.test_csv_exist.to_csv(self.test_csv_exist_dir, index=False)
        else :
            self.load_exist_data()

    def load_full_data(self):
        for i in tqdm(range(len(self.test_csv))):
            filename = self.test_csv['id'][i]
            fullpath = glob(self.test_dir + "*/" + filename.replace('[', '[[]') + ".JPG")[0]
            label = self.test_csv['id'][i]

            self.test_csv_exist.loc[i,'id'] = fullpath
            self.test_image.append(fullpath)
            self.test_label.append(label)


    def load_exist_data(self):
        self.test_csv_exist = pd.read_csv(self.test_csv_exist_dir)
        for i in tqdm(range(len(self.test_csv_exist))):
            fullpath = self.test_csv_exist['id'][i]
            label = self.test_csv_exist['id'][i]

            self.test_image.append(fullpath)
            self.test_label.append(label)


    def __len__(self):
        return len(self.test_image)

    def __getitem__(self, idx):
        image = Image.open(self.test_image[idx])
        if self.transform is not None:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = image.resize((self.args.image_size, self.args.image_size))
            image = np.array(image) / 255.
            image = np.transpose(image, axes=(2, 0, 1))
        label = self.test_label[idx]
        return image, label


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.reset()
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Augmentation
train_transform = A.Compose([
    A.SmallestMaxSize(args.max_size),
    #A.RandomCrop(args.image_size, args.image_size, p=1.),
    #A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.HueSaturationValue(),
        A.ShiftScaleRotate()
    ], p=1),
    A.Normalize(mean=[0.4452, 0.4457, 0.4464],
                 std=[0.2592, 0.2596, 0.2600]),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.SmallestMaxSize(args.max_size),
    #A.CenterCrop(args.image_size, args.image_size, p=1.),
    A.Normalize(mean=[0.4452, 0.4457, 0.4464],
                 std=[0.2592, 0.2596, 0.2600]),
    ToTensorV2(),
])

# Dataset, Dataloader 정의
dataset = TrainDataset(args, transform=train_transform)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.transform = test_transform
test_dataset = TestDataset(args, transform=test_transform)

train_sampler = RandomSampler(train_dataset)
val_sampler = SequentialSampler(val_dataset)
test_sampler = SequentialSampler(test_dataset)

train_loader = DataLoader(train_dataset,
                          sampler=train_sampler,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          pin_memory=False,
                          drop_last=True)
val_loader = DataLoader(val_dataset,
                        sampler=val_sampler,
                        batch_size=args.batch_size//2,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=False)
test_loader = DataLoader(test_dataset,
                         sampler=test_sampler,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=False,
                         drop_last=False)

# Model
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EfficientNetEncoderHead(nn.Module):
    def __init__(self, depth, num_classes, feat_dim=512):
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        self.base = EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        self.gem = GeM()
        self.output_filter = self.base._fc.in_features
        self.neck = nn.Sequential(
            nn.Linear(self.output_filter, feat_dim),
            nn.BatchNorm1d(feat_dim),
            torch.nn.PReLU(),
        )
        self.head = ArcMarginProduct(feat_dim, num_classes)

    def forward(self, x, label=None):
        x = self.base.extract_features(x)
        x = self.gem(x).squeeze()
        x = self.neck(x)
        logits = self.head(x)
        return logits

model = EfficientNetEncoderHead(depth=args.depth, num_classes=args.n_classes)
model.cuda()

def radam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-3, weight_decay=0):
    if isinstance(betas, str):
        betas = eval(betas)
    return torch_optimizer.RAdam(parameters,
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 weight_decay=weight_decay)

def accuracy(pred, label):
    num_correct = torch.sum(pred.max(1)[1] == label.data)
    num_cnt = len(label)
    acc = (num_correct.double()/num_cnt).cpu() * 100
    return acc

#criterion = nn.CrossEntropyLoss()
criterion = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit)
#optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.wd, nesterov=True)
#optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
optimizer = radam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*args.epochs, eta_min=1e-6)

# Training
# 매 epoch마다 ./ckpt 파일에 모델이 저장됩니다.
# validation dataset 없이 모든 train data를 train하는 방식입니다.
if not args.test:
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    train_loss, train_acc = [], []
    best_acc, best_epoch = 0, 0

    end = time.time()
    start_epoch = 0
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
        start_epoch = int(args.resume[-7:-4])
        print(f'Loaded {start_epoch} epoch..')
        start_epoch += 1

    for epoch in range(start_epoch, args.epochs):
        for iter, (image, label) in enumerate(train_loader):
            image = image.cuda()
            label = label.cuda()
            pred = model(image, label)
            loss = loss_fn(criterion, label, pred, args.n_classes)
            acc = accuracy(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.update(loss.data.item(), image.size(0))
            batch_time.update(time.time() - end)
            avg_score.update(acc)

            end = time.time()
            if iter % args.log_freq == 0:
                print(f'epoch : {epoch} step : [{iter}/{len(train_loader)}]\t'
                      f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'acc {avg_score.val:.4f} ({avg_score.avg:.4f})')
        # validation
        model.eval()
        val_start = time.time()
        val_time = 0
        num_correct, num_cnt = 0, 0
        for i, (image, label) in enumerate(tqdm(val_loader)):
            image = image.cuda()
            label = label.cuda()
            pred = model(image)
            num_correct += torch.sum(pred.max(1)[1] == label.data)
            num_cnt += len(label)
        val_acc = (num_correct.double()/num_cnt).cpu() * 100
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
        print(f'epoch : {epoch} [{len(val_loader)}]\t'
              f'time {time.time()-val_start:.3f}\t'
              f'val acc {val_acc:.4f}\t'
              f'best acc {best_acc:.4f} ({best_epoch})\t')
        torch.save(model.state_dict(), os.path.join(args.model_dir, "epoch_{0:03}.pth".format(epoch)))
        model.train()
    # 모든 epoch이 끝난 뒤 test 진행
    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pth')))
    submission = pd.read_csv(args.test_csv_dir)
    print(f'Loaded {args.load_epoch} epoch ckpt..')
    for iter, (image, label) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        pred = model(image)
        pred = nn.Softmax(dim=1)(pred)
        pred = pred.detach().cpu().numpy()
        landmark_ids = np.argmax(pred, axis=1)
        for offset, landmark_id in enumerate(landmark_ids):
            confidence = pred[offset, landmark_id]
            cur_idx = (iter*args.batch_size) + offset
            submission.loc[cur_idx, 'landmark_id'] = landmark_id
            submission.loc[cur_idx, 'conf'] = confidence
    submission.to_csv(args.test_csv_submission_dir, index=False)
    print(f'Save submission: {len(submission)}')

# Test
# argument의 --train을 False로 두면 Test만 진행합니다.
# Softmax로 confidence score를 계산하고, argmax로 class를 추정하여 csv 파일로 저장합니다.
# 현재 batch=1로 불러와서 조금 느릴 수 있습니다.
else :
    if args.load_epoch is not None:
        ckpt = f'epoch_{args.load_epoch:03}.pth'
        print(f'Loaded {args.load_epoch} epoch ckpt..')
    else:
        ckpt = 'best_model.pth'
        print(f'Loaded best epoch ..')
    model.load_state_dict(torch.load(os.path.join(args.model_dir, ckpt)))
    model.eval()
    submission = pd.read_csv(args.test_csv_dir)
    for iter, (image, label) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        pred = model(image)
        pred = nn.Softmax(dim=1)(pred)
        pred = pred.detach().cpu().numpy()
        landmark_ids = np.argmax(pred, axis=1)
        for offset, landmark_id in enumerate(landmark_ids):
            confidence = pred[offset, landmark_id]
            cur_idx = (iter*args.batch_size) + offset
            submission.loc[cur_idx, 'landmark_id'] = landmark_id
            submission.loc[cur_idx, 'conf'] = confidence
    submission.to_csv(args.test_csv_submission_dir, index=False)
    print(f'Save submission: {len(submission)}')
