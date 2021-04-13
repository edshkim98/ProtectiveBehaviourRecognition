import os
import torch
import numpy as np
from path import Path
import sys
import scipy.spatial.distance
import math
import random
import utils
import random
import scipy.io
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
import itertools
from sklearn.preprocessing import StandardScaler,Normalizer
from loss import *
from model import *
from data import *

path = '/home/edshkim98/affective/CoordinateData/'
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
np.random.seed(42)

torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model,train_loader,valid_loader,weights,mname,loss_func='multi',epochs=300, save=True):
    patience = 40
    cnt = 0
    class_weights = torch.FloatTensor(weights).cuda()
    
    lr = 0.0001
    loss_plt = []
    f1_plt = []
    acc_plt = []
    
    params = ([p for p in model.parameters()])#([p for p in model.parameters()] + [log_var_a] + [log_var_b])   
    optimizer = torch.optim.Adam(params, lr=lr)
    step = 2
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step*len(train_loader), eta_min=1e-4)
    weights = [0.7, 1.0]
    class_weights = torch.FloatTensor(weights).cuda()    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    best_acc = 0
    best_f1 = 0
    for epoch in range(epochs):
        e = epoch
        model.train()
        running_loss = 0
        for j, data in enumerate(train_loader):
            total=0
            correct = 0
            body_coord = torch.tensor([data[i]['data'] for i in range(len(data))])#[:,:,:66]#torch.stack(data[0]['data']).permute(1,0,2)[:,:,:67]
            try:
                emg = torch.tensor([data[i]['emg'] for i in range(len(data))])#[:,:,66:]
                labels = torch.tensor([data[i]['labels'] for i in range(len(data))])
                input1, input2, labels = body_coord.to(device), emg.to(device), labels.to(device)
                input1, input2 = input1.to(dtype=torch.float32), input2.to(dtype=torch.float32) 
                labels = labels.to(dtype=torch.long)
                optimizer.zero_grad()
                out = model(input1,input2)
            except:
                labels = torch.tensor([data[i]['labels'] for i in range(len(data))])
                input1, labels = body_coord.to(device), labels.to(device)
                input1 = input1.to(dtype=torch.float32)
                labels = labels.to(dtype=torch.long)
                optimizer.zero_grad()
                out = model(input1)
            
            if loss_func == 'weighted':
                #criterion = torch.nn.CrossEntropyLoss()
                #loss = criterion(out, labels)
                l = F1_Loss()
                loss = l(out,labels)
            if loss_func == 'multi1':
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                ls = criterion(out, labels)
                f1_ls = F1_Loss()
                ls2 = f1_ls(out,labels)
                loss = 0.5*ls+ls2
            if loss_func == 'multi2':
                l = F1_Loss()
                ls = l(out,labels)
                l2 = FocalLoss()
                ls2 = l2(out, labels)
                loss = ls+0.5*ls2
            if loss_func == 'single':
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(out, labels)
            if loss_func == 'focal':
                f1_ls = FocalLoss()#F1_Loss()
                loss = f1_ls(out, labels)
            if j%100==0:
                _,pred = torch.max(out.data,1)
                total += batch_size 
                correct += (pred == labels).sum().item()
                acc = 100. * correct / total
                #std_1 = torch.exp(lambdas[0])**0.5
                #std_2 = torch.exp(lambdas[1])**0.5
                print("Training Epoch {} Acc: {} Loss: {}".format(e,acc,loss.cpu().detach()))
                #print("Lambda1 {} Lambda2 {}".format(std_1.cpu().detach().numpy(),std_2.cpu().detach().numpy()))
            loss.backward()
            optimizer.step()
            #scheduler.step()
        
        loss = []
        y_true = []
        y_pred = []
        total = 0
        correct = 0
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                body_coord = torch.tensor([data[i]['data'] for i in range(len(data))])#[:,:,:66]#torch.stack(data[0]['data']).permute(1,0,2)[:,:,:67]
                try:
                    emg = torch.tensor([data[i]['emg'] for i in range(len(data))])#[:,:,66:]
                    labels = torch.tensor([data[i]['labels'] for i in range(len(data))])
                    input1, input2, labels = body_coord.to(device), emg.to(device), labels.to(device)
                    input1, input2 = input1.to(dtype=torch.float32), input2.to(dtype=torch.float32) 
                    labels = labels.to(dtype=torch.long)
                    out = model(input1,input2)
                except:
                    labels = torch.tensor([data[i]['labels'] for i in range(len(data))])
                    input1, labels = body_coord.to(device), labels.to(device)
                    input1 = input1.to(dtype=torch.float32)
                    labels = labels.to(dtype=torch.long)
                    out = model(input1)
                criterion = torch.nn.CrossEntropyLoss()
                ls = criterion(out, labels)
                _,pred = torch.max(out.data,1)
                total += batch_size 
                correct += (pred == labels).sum().item()
                loss.append(ls.item())
                y_true.append(labels.cpu().detach())
                y_pred.append(pred.cpu().detach())
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred,average='macro')
            print("Correct: {} Total: {}".format(correct,total))
            acc = 100. * correct / total
            loss = np.mean(loss)
            loss_plt.append(loss)
            f1_plt.append(f1)
            acc_plt.append(acc/100)
            print("Validation Epoch {} Acc: {} F1: {} Precision: {} Recall: {} Loss: {}".format(e,acc,f1,precision,recall,loss))
            
            if f1 > best_f1:
                cnt = 0
                best_f1 = f1
                best_acc = acc
                num = f1
                if save:
                    torch.save(model.state_dict(), str(mname)+'_'+f'{num:.3}'+".pth")
            else:
                cnt+=1
                print("Patience: ",cnt)
                if cnt == patience:
                    return loss_plt,f1_plt,acc_plt
            print("Best ACC: {} Best F1: {}".format(best_acc, best_f1))
            print("#####################################")
    return loss_plt,f1_plt, acc_plt

def plot_model(loss,f1,acc):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(20,4))
    ax1.plot(np.arange(50), loss)
    ax1.title.set_text("Loss")
    ax2.plot(np.arange(50), f1)
    ax2.title.set_text("F1 score")
    ax3.plot(np.arange(50), acc)
    ax3.title.set_text("Accuracy")


if __name__ == "__main__":
    window = 45
    batch_size = 32
    trainxs = os.listdir(path+'train/')
    valxs = os.listdir(path+'validation/')
    train = []
    val = []
    for i in range(len(trainxs)):
        train.append(CustomDataset(path,trainxs[i],transform =None,window= window,valid = False))
    train_dataset = ConcatDataset(train)
    for i in range(len(valxs)):
        val.append(CustomDataset(path,valxs[i],transform =None,window= window,valid = True))
    val_dataset = ConcatDataset(val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True, collate_fn=lambda x: x,num_workers=2)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last = True, collate_fn=lambda x: x, num_workers=2)
    
    model = Model()
    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: ", params)

    lr = 0.001
    step = 2
    loss, f1, acc= train_model(model, train_loader,valid_loader,loss_func='multi2',weights=[0.5,1.0],mname='1dcnn')
    plot_model(loss,f1,acc)
    
