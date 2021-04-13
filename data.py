import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.dataset import random_split
from sklearn.preprocessing import StandardScaler,Normalizer

torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self,path,fname,transform,window,valid=False):
        self.sigma=[0.1, 0.15, 0.2]
        self.clip=0.1
        self.path = path
        self.window = window
        self.transform = transform
        self.valid = valid
        self.fname = fname
        self.window = window
        self.files = []
        if self.valid == True:
            new_dir = self.path + r'validation/'
        else:
            new_dir = self.path + r'train/'

        mat =  scipy.io.loadmat(new_dir+self.fname)
        self.activities = mat['data'][:,70]
        self.label = mat['data'][:,72]
        self.data = mat['data'][:,:70][::self.window]
        for i in range(len(self.data)):
            sample = {}
            sample2 = {}
            if (i*self.window) < 90: #For starting frame
                #Zero pad
                lst = mat['data'][:,:70][:i*self.window+90]
                lst2 = list(np.tile(np.zeros(70),(90-i*self.window,1)))#(90-i*self.window)*list(np.zeros(70))
                lst_activity = list(self.activities[:i*self.window+90])
                lst_label = list(self.label[:i*self.window+90])
                lst_label = [lst_label[i] for i in range(len(lst_activity)) if lst_activity[i]<=5]
                if lst_label==[]:
                    continue
                labels = np.round(np.array(lst_label).sum()/len(lst_label))#(i*self.window+90))
                lst_data = []
                for i in range(len(lst_activity)):
                    if lst_activity[i]>5:
                        lst_data.append(np.zeros(70))
                    else:
                        lst_data.append(lst[i])
                lst_data = lst2+lst_data
                
            elif (len(mat['data'])-i*self.window) < 90: #For end frame
                #Zero pad
                lst = mat['data'][:,:70][i*self.window-90:]
                lst2 = list(np.tile(np.zeros(70),(90-(len(mat['data'])-(i*self.window)),1)))#(90-(len(mat['data'])-(i*self.window)))*list(np.zeros(70))
                lst_activity = list(self.activities[i*self.window-90:])
                lst_label = list(self.label[i*self.window-90:])
                lst_label = [lst_label[i] for i in range(len(lst_activity)) if lst_activity[i]<=5]
                if lst_label== []:
                    continue
                labels = np.round(np.array(lst_label).sum()/len(lst_label))#(len(mat['data'])-(i*self.window)+90))
                lst_data = []
                for i in range(len(lst_activity)):
                    if lst_activity[i]>5:
                        lst_data.append(np.zeros(70))
                    else:
                        lst_data.append(lst[i])
                lst_data = lst_data+lst2
                
            else: #For rest of frames
                lst = mat['data'][:,:70][i*self.window-90:i*self.window+90]
                lst_activity = list(self.activities[i*self.window-90:i*self.window+90])
                lst_label = list(self.label[i*self.window-90:i*self.window+90])
                lst_label = [lst_label[i] for i in range(len(lst_activity)) if lst_activity[i]<=5] 
                if lst_label== []:
                    continue
                labels = np.round(np.array(lst_label).sum()/len(lst_label))
                lst_data = []
                for i in range(len(lst_activity)):
                    if lst_activity[i]>5:
                        lst_data.append(np.zeros(70))
                    else:
                        lst_data.append(lst[i])
            
            sample['data'] = lst_data
            sample['label'] = labels
            self.files.append(sample)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        x = self.files[idx]['data']
        y = self.files[idx]['label']
            
        if self.transform != None:
            x = self.transform(x)

        data = np.array(x)[:][:,:66]
        emg = np.array(x)[:][:,66:]
        #Standard scaler
        scaler = StandardScaler()
        scaler2 = StandardScaler()
        scaler.fit(data)
        scaler2.fit(emg)
        emg = scaler2.transform(emg)
        data = scaler.transform(data)
        #Augmentation
        if self.valid == False:
            sigma = np.random.choice([0,0.1,0.15,0.2])
            if sigma==0:
                pass
            else:
                noise = np.random.normal(0, sigma, (180,66))
                noise2 = np.random.normal(0, sigma, (180,4))
                data += noise
                emg += noise2

        return {"data": data, "emg": emg, "labels": y}

    
train_transforms = transforms.Compose([
                    transforms.ToTensor()
                    ])

class CustomDataset2(Dataset):
    def __init__(self,path,fname,transform,window,valid=False):
        self.sigma=[0.1, 0.15, 0.2]
        self.clip=0.1
        self.path = path
        self.window = window
        self.transform = transform
        self.valid = valid
        self.fname = fname
        self.window = window
        self.files = []
        if self.valid == True:
            new_dir = self.path + r'validation/'
        else:
            new_dir = self.path + r'train/'

        mat =  scipy.io.loadmat(new_dir+self.fname)
        self.activities = mat['data'][:,70]
        self.label = mat['data'][:,72]
        self.data = mat['data'][:,:70][::self.window]
        for i in range(len(self.data)):
            sample = {}
            sample2 = {}
            if (i*self.window) < 90: #For starting frame
                #Zero pad
                lst = mat['data'][:,:70][:i*self.window+90]
                lst2 = list(np.tile(np.zeros(70),(90-i*self.window,1)))#(90-i*self.window)*list(np.zeros(70))
                lst_activity = list(self.activities[:i*self.window+90])
                lst_label = list(self.label[:i*self.window+90])
                lst_label = [lst_label[i] for i in range(len(lst_activity)) if lst_activity[i]<=5]
                if lst_label==[]:
                    continue
                labels = np.round(np.array(lst_label).sum()/len(lst_label))#(i*self.window+90))
                lst_data = []
                for i in range(len(lst_activity)):
                    if lst_activity[i]>5:
                        lst_data.append(np.zeros(70))
                    else:
                        lst_data.append(lst[i])
                lst_data = lst2+lst_data
                
            elif (len(mat['data'])-i*self.window) < 90: #For end frame
                #Zero pad
                lst = mat['data'][:,:70][i*self.window-90:]
                lst2 = list(np.tile(np.zeros(70),(90-(len(mat['data'])-(i*self.window)),1)))#(90-(len(mat['data'])-(i*self.window)))*list(np.zeros(70))
                lst_activity = list(self.activities[i*self.window-90:])
                lst_label = list(self.label[i*self.window-90:])
                lst_label = [lst_label[i] for i in range(len(lst_activity)) if lst_activity[i]<=5]
                if lst_label== []:
                    continue
                labels = np.round(np.array(lst_label).sum()/len(lst_label))#(len(mat['data'])-(i*self.window)+90))
                lst_data = []
                for i in range(len(lst_activity)):
                    if lst_activity[i]>5:
                        lst_data.append(np.zeros(70))
                    else:
                        lst_data.append(lst[i])
                lst_data = lst_data+lst2
                
            else: #For rest of frames
                lst = mat['data'][:,:70][i*self.window-90:i*self.window+90]
                lst_activity = list(self.activities[i*self.window-90:i*self.window+90])
                lst_label = list(self.label[i*self.window-90:i*self.window+90])
                lst_label = [lst_label[i] for i in range(len(lst_activity)) if lst_activity[i]<=5] 
                if lst_label== []:
                    continue
                labels = np.round(np.array(lst_label).sum()/len(lst_label))
                lst_data = []
                for i in range(len(lst_activity)):
                    if lst_activity[i]>5:
                        lst_data.append(np.zeros(70))
                    else:
                        lst_data.append(lst[i])
            
            sample['data'] = lst_data
            sample['label'] = labels
            self.files.append(sample)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        x = self.files[idx]['data']
        y = self.files[idx]['label']
            
        if self.transform != None:
            x = self.transform(x)
        #Standard scaler
        data = np.array(x)[:][:,:70]
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        #Augmentation
        if self.valid == False:
            sigma = np.random.choice([0,0.1,0.15,0.2])
            if sigma==0:
                pass
            else:
                noise = np.random.normal(0, sigma, (180,70))
                data += noise        

        return {"data": data, "labels": y}
