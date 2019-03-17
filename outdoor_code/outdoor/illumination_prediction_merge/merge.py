import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from skimage import draw, io
from PIL import Image
from skimage import io
import numpy as np
#import SkyRenderPy
import linecache
import os
import math

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
        self.lsoftmax = nn.LogSoftmax()

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

def estimate_illumination(img_path):
    image = transform(img_path)
    image = image.view(1, 3, 240, 360)
    #image = image.cuda()
    sun_pos, sky_cam = model_recover(image)

    sun_pos = sun_pos.cpu().detach().numpy()[0]
    sky_cam = sky_cam.cpu().detach().numpy()[0]

    # get sun position
    index = np.argmax(sun_pos)    
    i = index // 32
    j = index % 32

    theta = (0.5 * np.pi) * 0.1 * (i * 2 + 1)
    phi = -np.pi + (2.0 * np.pi) * 0.03125 * j
    sunposition = np.zeros(2, dtype=np.float32)
    sunposition[0] = theta   # elevation
    sunposition[1] = phi   # azimuth

    # get sky parameters
    sky = np.zeros(2, dtype=np.float32)
    sky[0] = 9 * (sky_cam[0] + 1) / 2 + 1
    sky[1] = 2 * (sky_cam[1] + 1) / 2 + 0.5

    # get camera parameters
    camera = np.zeros(2, dtype=np.float32)
    camera[0] = 40 * (sky_cam[2] + 1) / 2 - 20
    camera[1] = 20 * (sky_cam[3] + 1) / 2 + 60

    return sunposition, sky, camera

if __name__ == '__main__':
    # load model
    model_recover=Net() 
    model_recover.load_state_dict(torch.load('F:/三星室外代码\github最新代码/3_7/model/model_3_6_121_0.7312.pkl', map_location='cpu'))
    #model_recover.cuda()
    model_recover.eval()

    root_dir = 'F:/三星室外代码/github最新代码/3_7/out6_7/'
    param_dir = 'F:/三星室外代码/github最新代码/3_7/out6_7param/'
    img = io.imread('F:\三星室外代码\github最新代码/3_8/back.jpg')

    fil = open('./confidence6_7.txt','r+')
    param_confi = np.zeros([7,3])
    i = 0
    for line in fil:
        line=line.strip('\n').split()
        param_confi[i] = line
        i = i + 1
    sum = 0
    param_merge = np.zeros([2])   # merged position
    for i in range(7):
        sum += param_confi[i][0]
    big = 0
    small = 0
    middle = 0
    for i in range(7):
        if param_confi[i][2] > np.pi / 2:
            big += 1
        elif param_confi[i][2] < -np.pi / 2:
            small += 1
    if big > 1 and small > 1:
         for i in range(7):
             if param_confi[i][2] > np.pi / 2:
                 param_confi[i][2] -= np.pi * 2
    
    for i in range(7):
        param_confi[i][1] *= param_confi[i][0] / sum  # elevation
        param_confi[i][2] *= param_confi[i][0] / sum  # azimuth

    for i in range(7):
        param_merge[0] += param_confi[i][1]
        param_merge[1] += param_confi[i][2]

    print ('merge',param_merge[0], param_merge[1])
    if param_merge[1] < -np.pi:
            param_merge[1] = param_merge[1] + 2 * np.pi
    if param_merge[1] > np.pi:
        param_merge[1] = param_merge[1] - 2 * np.pi
    param1 = (param_merge[0] + np.pi / 2) /  np.pi *  256
    param2 = ((param_merge[1] +  np.pi) / (2 *  np.pi)) *  512
    print ('merge',param_merge[0], param_merge[1])
    print ('merge',param1, param2)


    rr, cc = draw.circle(param1, param2, 5)
    img[rr, cc] = 100  
    #io.imsave('./merged_sun.jpg', img)

    files = os.listdir(root_dir)
    sun_reapos = np.zeros([2])
    fn = ' '
    for fn in files:
        img_path = root_dir + fn
        param = estimate_illumination(img_path)
        camera_param = linecache.getline(param_dir+'/'+fn[:19]+'_camera_param.txt',int(fn[26])+1).split(':')[1][1:-1].split(',')
        cam_azi = math.radians(float(camera_param[1])) # azimuth of camera
        param[0][1] += cam_azi # absolute position of sun

        if param[0][1] < -np.pi:
                param[0][1] = param[0][1] + 2 * np.pi
        if param[0][1] > np.pi:
            param[0][1] = param[0][1] - 2 * np.pi

        param1 = (param[0][0] + np.pi / 2) /  np.pi *  256
        param2 = ((param[0][1] +  np.pi) / (2 *  np.pi)) *  512
        print ('param', param[0][0], param[0][1])
        print ('param', param1, param2)
        rr, cc = draw.circle(param1, param2, 3)
        img[rr, cc] = 200  
        
    sunsky_param = linecache.getline(param_dir+'/'+fn[:19]+'_sunsky_param.txt',1)[:-1].split(',')
    sun_reapos = sunsky_param[:2]
    sun_reapos = [float(i) for i in sun_reapos]

    if sun_reapos[0] < -np.pi:
            sun_reapos[0] = sun_reapos[0] + 2 * np.pi
    if sun_reapos[1] > np.pi:
        sun_reapos[1] = sun_reapos[1] - 2 * np.pi
    #print ('sun_reapos', sun_reapos)
    param1 = (sun_reapos[0] + np.pi / 2) /  np.pi *  256
    param2 = ((sun_reapos[1] +  np.pi) / (2 *  np.pi)) *  512
    print ('real', sun_reapos[0], sun_reapos[1])
    print ('real', param1, param2)
    rr, cc = draw.circle(param1, param2, 7)
    img[rr, cc] = 1  
    io.imsave('./sun6.jpg', img)

    
