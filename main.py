# coding: utf-8

import pandas as pd
import numpy as np
import os
import time

import argparse
import torch

from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import Conv2d, AdaptiveAvgPool2d, Linear
from torch.utils.data.sampler import BatchSampler, RandomSampler
import torch.nn.functional as F
import albumentations
from albumentations.pytorch import ToTensorV2
import torch_optimizer

from efficientnet_pytorch import EfficientNet
from wrn import WRN


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

parser.add_argument('--image_size', dest='image_size', type=int, default=224)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=29)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--num_workers', dest='num_workers', type=int, default=4)
parser.add_argument('--log_freq', dest='log_freq', type=int, default=10)
args = parser.parse_args()


# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'

# 경로 생성
if not os.path.isdir(args.model_dir) :
    os.makedirs(args.model_dir)

# 파이토치 Dataset 생성 for Train / Test
class TrainDataset(Dataset) :
    def __init__(self, args, transform=None) :
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
        else :
            self.load_exist_data()

    def load_full_data(self) :
        for i in tqdm(range(len(self.train_csv))) :
            filename = self.train_csv['id'][i]
            fullpath = glob(self.train_dir + "*/*/" + filename.replace('[', '[[]') + ".JPG")[0]
            label = self.train_csv['landmark_id'][i]
            self.train_csv_exist.loc[i,'id'] = fullpath
            self.train_image.append(fullpath)
            self.train_label.append(label)


    def load_exist_data(self) :
        self.train_csv_exist = pd.read_csv(self.train_csv_exist_dir, encoding='utf-8')
        for i in tqdm(range(len(self.train_csv_exist))) :
            fullpath = self.train_csv_exist['id'][i]
            label = self.train_csv_exist['landmark_id'][i]
            self.train_image.append(fullpath)
            self.train_label.append(label)


    def __len__(self) :
        return len(self.train_image)

    def __getitem__(self, idx) :
        image = Image.open(self.train_image[idx])
        #image = image.resize((self.args.image_size, self.args.image_size))
        if self.transform is not None:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = np.array(image) / 255.
            image = np.transpose(image, axes=(2, 0, 1))
        label = self.train_label[idx]
        return image, label

class TestDataset(Dataset) :
    def __init__(self, args, transform=None) :
        self.test_dir = args.test_dir
        self.test_csv_dir = args.test_csv_dir
        self.test_csv_exist_dir = args.test_csv_exist_dir
        self.args = args
        self.transform = transform
        self.test_image = list()
        self.test_label = list()
        if not os.path.isfile(self.test_csv_exist_dir) :
            self.test_csv = pd.read_csv(self.test_csv_dir, encoding='utf-8')
            self.test_csv_exist = self.test_csv.copy()
            self.load_full_data()
            self.test_csv_exist.to_csv(self.test_csv_exist_dir, index=False)
        else :
            self.load_exist_data()

    def load_full_data(self) :
        for i in tqdm(range(len(self.test_csv))) :
            filename = self.test_csv['id'][i]
            fullpath = glob(self.test_dir + "*/" + filename.replace('[', '[[]') + ".JPG")[0]
            label = self.test_csv['id'][i]

            self.test_csv_exist.loc[i,'id'] = fullpath
            self.test_image.append(fullpath)
            self.test_label.append(label)


    def load_exist_data(self) :
        self.test_csv_exist = pd.read_csv(self.test_csv_exist_dir)
        for i in tqdm(range(len(self.test_csv_exist))) :
            fullpath = self.test_csv_exist['id'][i]
            label = self.test_csv_exist['id'][i]

            self.test_image.append(fullpath)
            self.test_label.append(label)


    def __len__(self) :
        return len(self.test_image)

    def __getitem__(self, idx) :
        image = Image.open(self.test_image[idx])
        #image = image.resize((self.args.image_size, self.args.image_size))
        if self.transform is not None:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
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


# DataLoader 생성을 위한 collate_fn
def collate_fn(batch) :
    image = [x['image'] for x in batch]
    label = [x['label'] for x in batch]

    return torch.tensor(image).float().cuda(), torch.tensor(label).long().cuda()

def collate_fn_test(batch) :
    image = [x['image'] for x in batch]
    label = [x['label'] for x in batch]

    return torch.tensor(image).float().cuda(), label

# Augmentation
train_transform = albumentations.Compose([
    albumentations.Resize(args.image_size, args.image_size),
    albumentations.RandomCrop(args.image_size, args.image_size),
    albumentations.HorizontalFlip(),
    albumentations.OneOf([
        albumentations.HueSaturationValue(),
        albumentations.ShiftScaleRotate()
    ], p=1),
    albumentations.Normalize(mean=[0.4452, 0.4457, 0.4464],
                            std=[0.2592, 0.2596, 0.2600]),
    ToTensorV2(),
])

test_transform = albumentations.Compose([
    albumentations.Resize(args.image_size, args.image_size),
    albumentations.RandomCrop(args.image_size, args.image_size),
    albumentations.Normalize(mean=[0.4452, 0.4457, 0.4464],
                            std=[0.2592, 0.2596, 0.2600]),
    ToTensorV2(),
])

# Dataset, Dataloader 정의
train_dataset = TrainDataset(args, transform=train_transform)
test_dataset = TestDataset(args, transform=test_transform)

train_sampler = RandomSampler(train_dataset)
test_sampler = RandomSampler(test_dataset)

train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=True)
test_batch_sampler = BatchSampler(test_sampler, args.batch_size, drop_last=False)

train_loader = DataLoader(train_dataset,
                          batch_sampler=train_batch_sampler,
                          num_workers=args.num_workers,
                          pin_memory=True)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=True,
                         drop_last=False)

# Model
# 여기서는 간단한 CNN 3개짜리 모델을 생성하였습니다.
class Network(nn.Module) :
    def __init__(self) :
        super(Network, self).__init__()
        self.conv1 = Conv2d(3, 64, (3,3), (1,1), (1,1))
        self.conv2 = Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.conv3 = Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.fc = Linear(64, 1049)

    def forward(self, x) :
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = AdaptiveAvgPool2d(1)(x).squeeze()
        x = self.fc(x)
        return x

#model = Network()
model = WRN(width=2, num_classes=1049, dropout=0.5)

class EfficientNetEncoderHead(nn.Module):
    def __init__(self, depth, num_classes, dropout=0.5):
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        self.base = EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output_filter = self.base._fc.in_features
        self.classifier = nn.Linear(self.output_filter, num_classes)

    def forward(self, x):
        x = self.base.extract_features(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x


#model = EfficientNetEncoderHead(depth=0, num_classes=1049)
model.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)

def radam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-3, weight_decay=0):
    if isinstance(betas, str):
        betas = eval(betas)
    return torch_optimizer.RAdam(parameters,
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 weight_decay=weight_decay)
optimizer = radam(model.parameters(), weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*args.epochs, eta_min=1e-6)

# Training
# 매 epoch마다 ./ckpt 파일에 모델이 저장됩니다.
# validation dataset 없이 모든 train data를 train하는 방식입니다.
if not args.test:
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    train_loss, train_acc = [], []

    model.train()
    end = time.time()
    start_epoch = 0
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
        start_epoch = int(args.resume[-7:-4])
        print(f'Loaded {start_epoch} epoch..')
        start_epoch += 1

    for epoch in range(start_epoch, args.epochs) :
        num_correct, num_cnt = 0, 0
        for iter, (image, label) in enumerate(train_loader) :
            image = image.cuda()
            label = label.cuda()
            pred = model(image)
            loss = criterion(input=pred, target=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            num_correct += torch.sum(pred.max(1)[1] == label.data)
            num_cnt += len(label)
            score = (num_correct.double()/num_cnt).cpu() * 100
            losses.update(loss.data.item(), image.size(0))
            batch_time.update(time.time() - end)
            avg_score.update(score)
            end = time.time()
            if iter % args.log_freq == 0:
                print(f'epoch : {epoch} step : [{iter}/{len(train_loader)}]\t'
                      f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'acc {avg_score.val:.4f} ({avg_score.avg:.4f})')
        torch.save(model.state_dict(), os.path.join(args.model_dir, "epoch_{0:03}.pth".format(epoch)))
    # 모든 epoch이 끝난 뒤 test 진행
    model.eval()
    submission = pd.read_csv(args.test_csv_dir)
    for iter, (image, label) in tqdm(enumerate(test_loader)):
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

# Test
# argument의 --train을 False로 두면 Test만 진행합니다.
# Softmax로 confidence score를 계산하고, argmax로 class를 추정하여 csv 파일로 저장합니다.
# 현재 batch=1로 불러와서 조금 느릴 수 있습니다.
else :
    model.load_state_dict(os.path.join(torch.load(args.model_dir, "epoch_{0:03}.pth".format(args.load_epoch))))
    print(f'Loaded {args.load_epoch} epoch ckpt..')
    model.eval()
    submission = pd.read_csv(args.test_csv_dir)
    for iter, (image, label) in enumerate(tqdm(test_loader)):
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
