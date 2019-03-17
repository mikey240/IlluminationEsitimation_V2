from __future__ import print_function
import argparse
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pickle
import os.path
import linecache
from PIL import Image
from torchvision import datasets, transforms
import torch.nn.functional as F

filenameToPILImage = lambda x: Image.open(x).resize((360, 240), Image.BICUBIC)

transform = transforms.Compose([filenameToPILImage,
                                transforms.ToTensor(),
                                transforms.Normalize([0.463, 0.479, 0.469], [1, 1, 1]),
                                ])
model_recover = None

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

        self.fc1 = nn.Linear(12*8*256, 2048)
        self.fc2 = nn.Linear(2048, 160)
        self.fc3 = nn.Linear(2048, 2)
        self.lsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        x = x.view(-1, 12*8*256)
        x = F.relu(self.fc1(x))

        sun_pos = self.fc2(x)
        sun_pos = self.lsoftmax(sun_pos)
        sky_cam = F.hardtanh(self.fc3(x))

        return sun_pos, sky_cam

if __name__ == '__main__':
    # load model
    model_recover = Net() 
    model_recover.load_state_dict(torch.load('./model/model_3_6_121_0.7312.pkl', map_location='cpu'))
    model_recover.eval()

    error_list = []

    root_dir = 'F:/三星室外代码/github最新代码/3_7/out6_7/'
    param_dir = 'F:/三星室外代码/github最新代码/3_7/out6_7param/'

    files = os.listdir(root_dir)

    fi = open('./confidence6_7.txt', 'w')
    fi.write('confidence' + ' elevation' + ' azimuth' + '\n')

    for fn in files:
        img_path = root_dir + fn

        image = transform(img_path)
        image = image.view(1, 3, 240, 360)
        image = Variable(image)
        sun_pos, sky_cam = model_recover(image)

        camera_param = linecache.getline(param_dir+'/'+fn[:19]+'_camera_param.txt',int(fn[26])+1).split(':')[1][1:-1].split(',')
        cam_azi = math.radians(float(camera_param[1])) # azimuth of camera

        sunsky_param = linecache.getline(param_dir+'/'+fn[:19]+'_sunsky_param.txt',1)[:-1].split(',')
        sun_reapos = sunsky_param[:2]
        sun_reapos = [float(i) for i in sun_reapos]

        sun_pos = sun_pos.cpu().detach()
        sun_pos = sun_pos.data.numpy()

        fi.write(str(math.exp(max(sun_pos[0]))) + ' ')

        index = np.argmax(sun_pos)  
        i = index // 32
        j = index % 32

        theta = (0.5 * np.pi) * 0.1 * (i * 2 + 1) # elevation
        phi = -np.pi + (2.0 * np.pi) * 0.03125 * j # azimuth
        phi = phi + cam_azi  
        
        if phi < -np.pi:
            phi = phi + 2 * np.pi
        if phi > np.pi:
            phi = phi - 2 * np.pi

        fi.write(str(theta) + ' ' + str(phi) + '\n')   

