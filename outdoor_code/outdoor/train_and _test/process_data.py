import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os.path
import linecache
import pickle
import math

#illegal_file=open('/home/illu/data/outdoor/illegal_set.pkl','rb')
#illegaldata_set=pickle.load(illegal_file)
#illegal_file.close()
data_dir = '/home/ypp/sam_data/outdoor_10/outdoor_10'
training_percent = 0.8
PI = 3.14159

all_prefix = []
all_files = os.listdir(data_dir)
for one in all_files:
    if 'camera_param.txt' in one:
        #if one in illegaldata_set:
            #continue
        all_prefix.append(one[:-17])
all_prefix.sort()

ntraining = int(training_percent * len(all_prefix))
ntesting = len(all_prefix) - ntraining
training_prefix = all_prefix[:ntraining]
testing_prefix = all_prefix[ntraining:]

filenameToPILImage = lambda x: Image.open(x)

class Dataset(data.Dataset):
    def __init__(self, Training=True):
        self.Training = Training
        self.transform = transforms.Compose([filenameToPILImage,
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.463, 0.479, 0.469], [1, 1, 1]),
                                            ])

    def __len__(self):
        if self.Training:
            return ntraining*7
        else:
            return ntesting * 7

    def __getitem__(self, index):
        index_prefix = index/7
        index_inside = index%7
        if self.Training:
            prefix = training_prefix
        else:
            prefix = testing_prefix

        file_name = prefix[index_prefix]+'_sample'+str(index_inside)+'.png'
        image = self.transform(data_dir+'/'+file_name)
        camera_param = linecache.getline(data_dir+'/'+prefix[index_prefix]+'_camera_param.txt',
                                        index_inside+1).split(':')[1][1:-1].split(',')
        sunsky_param = linecache.getline(data_dir+'/'+prefix[index_prefix]+'_sunsky_param.txt',
                                        1)[:-1].split(',')
        sun_pos = sunsky_param[:2]
        sun_pos = [float(i) for i in sun_pos]
        sun_pos = np.array(sun_pos)  
        sky_cam = sunsky_param[2:]+camera_param
        sky_cam = [float(i) for i in sky_cam]
        sun_pos[1] -= math.radians(sky_cam[3])
        if sun_pos[1] < 0:
            sun_pos[1] += 2.0 * np.pi

        sky_cam[0] = 2*(sky_cam[0]-1)/9-1
        sky_cam[1] = 2*(sky_cam[1]-0.5)/2-1
        sky_cam[2] = 2*(sky_cam[2]+20)/40-1
        # sky_cam[3] = (sky_cam[3]+180)/360
        # sky_cam[4] = (sky_cam[4]-35)/33
        # sky_cam = np.array(sky_cam, dtype=np.float32)
        # sky_cam[3] = 2*(sky_cam[4] - 35) / 33-1
        sky_cam[3] = 2*(sky_cam[4] - 60) / 20-1
        sky_cam = np.array(sky_cam, dtype=np.float32)[:4]

        # if sun_pos[1]<0:
        #     sun_pos[1] = sun_pos[1]+2*PI
        x = -np.cos(sun_pos[0]) * np.sin(sun_pos[1])
        y = np.sin(sun_pos[0])
        z = np.cos(sun_pos[0]) * np.cos(sun_pos[1])
        sun_pos = torch.from_numpy(np.array([x, y, z], dtype=np.float32))
        sky_cam = torch.from_numpy(sky_cam)
        #gt_params = torch.cat((sun_pos, sky_cam))
        gt_params = torch.cat((sun_pos, sky_cam[:2]))
        return image, gt_params

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        Dataset(Training=True), batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        Dataset(Training=False), batch_size=1, shuffle=False)

    mean_train = np.zeros((ntraining*7, 3))
    stat_params = np.zeros((ntraining*7, 4))
    for i, (image, gt_params) in enumerate(train_loader):
        mean_train[i,:] = np.mean(image.numpy(), axis=(0,2,3))
        stat_params[i,:] = gt_params[0,3:]
    mean_train = np.mean(mean_train,axis=0)
    print mean_train
    print np.min(stat_params, axis=0)
    print np.max(stat_params, axis=0)

    mean_test = np.zeros((ntesting * 7, 3))
    stat_params = np.zeros((ntesting * 7, 4))
    for i, (image, gt_params) in enumerate(test_loader):
        mean_test[i, :] = np.mean(image.numpy(), axis=(0, 2, 3))
        stat_params[i, :] = gt_params[0, 3:]
    mean_test = np.mean(mean_test, axis=0)
    print mean_test
    print np.min(stat_params, axis=0)
    print np.max(stat_params, axis=0)

