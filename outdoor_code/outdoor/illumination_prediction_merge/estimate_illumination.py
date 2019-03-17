import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from skimage import io
import numpy as np
import SkyRenderPy
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
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))
        x = F.elu(self.bn5(self.conv5(x)))
        x = F.elu(self.bn6(self.conv6(x)))
        x = F.elu(self.bn7(self.conv7(x)))

        x = x.view(-1, 12*8*256)
        x = F.elu(self.fc1(x))

        sun_pos = self.fc2(x)
        sun_pos = self.lsoftmax(sun_pos)
        sky_cam = F.hardtanh(self.fc3(x))

        return sun_pos, sky_cam

def estimate_illumination(img_path):
    image = transform(img_path)
    image = image.view(1, 3, 240, 360)
    image = image.cuda()
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
    sunposition[0] = theta
    sunposition[1] = phi

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
    model_recover.load_state_dict(torch.load('./model.pkl'))
    model_recover.cuda()
    model_recover.eval()

    root_dir = 'E:/npic/'

    files = os.listdir(root_dir)

    # only evaluate 10 images
    count = 10

    for fn in files:
        # if count > 0 and fn.endswith('_camera_param.txt'):
        #     with open(root_dir + fn, 'r') as f:
        #         fid = fn[:-17]
        #         for i in range(7):
        #             line = f.readline()
        #             line = line[9:]
        #             azimuth = math.radians(float(line.split(',')[1]))
        #             img_path = root_dir + fid + '_sample%d.png' % i
        #             param = estimate_illumination(img_path)
        #             img = SkyRenderPy.render(param[1][0], 512, param[0][0], azimuth + param[0][1])
        #             img *= param[1][1]
        #             img[img > 1] = 1
        #             io.imsave(root_dir + fid + '_sample%d_estimate.jpg' % i, img)
        #     count -= 1
        # if fn[1:] != '.jpg':
        #     continue
        img_path = root_dir + fn
        param = estimate_illumination(img_path)
        img = SkyRenderPy.render(param[1][0], 512, param[0][0], param[0][1])
        img *= param[1][1]
        img[img > 1] = 1
        # just a trick
        _img = img[:128, :, :]
        img[128:, :, :] = _img[::-1, :, :]
        io.imsave(root_dir + fn[:-4] + '_estimate.jpg', img)

        with open(root_dir + fn[:-4] + '_config.txt', 'wt') as f:
            # model
            f.write('model:\n\n')
            # background
            f.write('background:\n')
            f.write(root_dir + fn)
            f.write('\n')
            # envmap
            f.write('pano:\n')
            f.write(root_dir + fn[:-4] + '_estimate.jpg')
            f.write('\n')
            # shadow mask
            f.write('mask:\n')
            f.write(root_dir + fn[:-4] + '_mask.jpg')
            f.write('\n')
            # elevation
            f.write('elevation:\n')
            f.write(str(param[2][0]))
            f.write('\n')
            # fov
            f.write('fov:\n')
            f.write(str(param[2][1]))
            f.write('\n')
            # sun elevation
            f.write('theta:\n')
            f.write(str(param[0][0]))
            f.write('\n')
            # sun azimuth
            f.write('phi:\n')
            f.write(str(param[0][1]))
            f.write('\n')
            # objRotMat:
            f.write('objRotMat:\n1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n')
            # objMat:
            f.write('objMat:\n1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n')
            # sfMat:
            f.write('sfMat:\n1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n')
            # sfY:
            f.write('sfY:\n0\n')
