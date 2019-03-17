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

data_dir = '/home/illu/old/Illumination/Illumination/outdoor_testcase_maker/outdoor_testset'

all_prefix = []
all_files = os.listdir(data_dir)
for one in all_files:
    if 'camera_param.txt' in one:
        all_prefix.append(one[:-17])

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
    model_recover.load_state_dict(torch.load('/home/illu/old/Illumination/Illumination/model_2_23_relu.pkl'))
    model_recover.cuda()
    model_recover.eval()

    error_list = []

    fi = open('./confidence.txt', 'w')
    fi.write('confidence' + ' elevation' + ' azimuth' + '\n')

    for one in all_prefix:
        sunsky_param = linecache.getline(data_dir + '/'+ one + '_sunsky_param.txt', 1)[:-1].split(',')
        sun_reapos = sunsky_param[:2]
        sun_reapos = [float(i) for i in sun_reapos]
        for m in range(7):
            img_path = data_dir + '/' + one + '_sample%d.png' % m

            name = one + '_sample%d.png' % m

            if not os.path.exists(img_path):
                continue
            image = transform(img_path)
            image = image.view(1, 3, 240, 360)
            image = image.cuda()
            image = Variable(image)
            sun_pos, sky_cam = model_recover(image)

            camera_param = linecache.getline(data_dir + '/' + one + '_camera_param.txt', m+1).split(':')[1][1:-1].split(',')
            cam_azi = math.radians(float(camera_param[1])) # azimuth of camera

            sun_pos = sun_pos.cpu().detach()
            sun_pos = sun_pos.data.numpy()

            fi.write(str(math.exp(max(sun_pos[0]))) + ' ')

            index = np.argmax(sun_pos)  
            i = index // 32
            j = index % 32

            theta = (0.5 * np.pi) * 0.1 * (i * 2 + 1) # elevation
            phi = -np.pi + (2.0 * np.pi) * 0.03125 * j # azimuth
            phi = phi + cam_azi  

            fi.write(str(theta) + ' ' + str(phi) + '\n')   

            #gt_vec = np.array((math.sin(sun_reapos[1]), math.cos(sun_reapos[1])))
            #pred_vec = np.array((math.sin(phi), math.cos(phi))) 
            gt_vec = np.array((math.cos(sun_reapos[0]) * math.sin(sun_reapos[1]), math.sin(sun_reapos[0]), math.cos(sun_reapos[0]) * math.cos(sun_reapos[1])))
            pred_vec = np.array((math.cos(theta) * math.sin(phi), math.sin(theta), math.cos(theta) * math.cos(phi)))     

            diff = math.acos(np.sum(gt_vec * pred_vec))
            error_list.append(diff)

    error_list = np.array(error_list, dtype=np.float32)
    error_list.sort()
    np.save('./error_list.npy', error_list)
    print('accuracy(22.5):', float(np.sum(error_list <= np.pi / 8.0)) / error_list.shape[0])
    print('accuracy(45.0):', float(np.sum(error_list <= np.pi / 4.0)) / error_list.shape[0])

    fi.close()
