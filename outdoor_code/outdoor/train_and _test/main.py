from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from process_data import Dataset
import pickle
#torch.cuda.set_device(3)
# Training settings
parser = argparse.ArgumentParser(description='Illumination')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--beta', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log', action='store_true', default=False,
                    help='log or not')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--eval_interval', type=int, default=1, metavar='N',
                    help='evaluation interval')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

PI = 3.14159
eps = 1e-8
sunpos_template = torch.zeros(3, 160)
for i in range(5):
    for j in range(32):
        theta = (0.5 * np.pi) * 0.1 * (i * 2 + 1)
        phi = -np.pi + (2.0 * np.pi) * 0.03125 * j
        x = -np.cos(theta) * np.sin(phi)
        y = np.sin(theta)
        z = np.cos(theta) * np.cos(phi)
        sunpos_template[:, i*32+j] = torch.from_numpy(np.array([x,y,z]))
if args.cuda:
    sunpos_template = sunpos_template.cuda()
sunpos_template = Variable(sunpos_template)

train_loader = torch.utils.data.DataLoader(
        Dataset(Training=True), batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        Dataset(Training=False), batch_size=args.batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)

        # self.fc1 = nn.Linear(20480, 2048)
        self.fc1 = nn.Linear(12*8*256, 2048)
        self.fc2 = nn.Linear(2048, 160)
        self.fc3 = nn.Linear(2048, 2)
        self.lsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        # x = x.view(-1, 10*8*256)
        x = x.view(-1, 12*8*256)
        x = F.dropout(x, p = 0.4, training = self.training)
        x = F.relu(self.fc1(x))

        x = F.dropout(x, p = 0.4, training = self.training)
        sun_pos = self.fc2(x)
        sun_pos = self.lsoftmax(sun_pos)
        sky_cam = F.hardtanh(self.fc3(x))

        return sun_pos, sky_cam

def cal_loss(sun_pos, sky_cam, gt_params):
    gt_sun = gt_params[:, :3]
    gt_sky_cam = gt_params[:, 3:]
    gt_sun = torch.matmul(gt_sun, sunpos_template)
    gt_sun = torch.exp(80*gt_sun)
#    gt_sun = torch.nn.functional.softmax(gt_sun, dim=1)
    gt_sun=gt_sun / torch.sum(gt_sun,dim=1).view(-1,1)

    sunpos_loss = gt_sun*(torch.log(gt_sun+eps)-sun_pos)
    sunpos_loss = torch.sum(sunpos_loss, 1)
    sunpos_loss = torch.mean(sunpos_loss)

    #params_loss = torch.pow(sky_cam-gt_sky_cam, 2)
    params_loss = torch.pow(sky_cam[:, :2]-gt_sky_cam, 2)
    params_loss = torch.mean(params_loss)

    return  sunpos_loss+160*params_loss

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta, 0.999))

def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (image, gt_params) in enumerate(train_loader):
        if args.cuda:
            image, gt_params = image.cuda(), gt_params.cuda()
        image, gt_params = Variable(image), Variable(gt_params)
        optimizer.zero_grad()
        sun_pos, sky_cam = model(image)
        loss = cal_loss(sun_pos, sky_cam, gt_params)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if args.log and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    train_loss /= len(train_loader)
    print('\nTrain Epoch: {} \nTrain set: Average loss: {:.4f}'.format(epoch,train_loss))

def test():
    model.eval()
    test_loss = 0
    for image, gt_params in test_loader:
        if args.cuda:
            image, gt_params = image.cuda(), gt_params.cuda()
        image, gt_params = Variable(image, volatile=True), Variable(gt_params)
        sun_pos, sky_cam = model(image)
        test_loss += cal_loss(sun_pos, sky_cam, gt_params).item()
 # sum up batch loss

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))


for epoch in range(0, args.epochs):
    train(epoch)
    if epoch%args.eval_interval==0:
        test()
        torch.save(model.state_dict(), 'model_3_11_%d.pkl'%epoch)

