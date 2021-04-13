import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch

torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)
    
#Multi-modal CNN+Attention
class Model(nn.Module):
    def __init__(self):#, args):
        super(Model, self).__init__()
        #input shape: bs*seq_len*num_features
        self.conv1 = nn.Conv1d(66,32,31,padding=15)#nn.Conv1d(66,32,1)
        self.conv2 = nn.Conv1d(32,32,31,padding=15)
        #self.se1 = SE_Block(32)
        
        self.conv3 = nn.Conv1d(32,32,31,padding=15)#nn.Conv1d(66,32,1)
        self.conv4 = nn.Conv1d(32,32,31,padding=15)
        #self.se2 = SE_Block(32)
        
        self.conv1_2 = nn.Conv1d(4,32,31,padding=15)#nn.Conv1d(66,32,1)
        self.conv2_2 = nn.Conv1d(32,32,31,padding=15)
        
        self.conv3_2 = nn.Conv1d(32,32,31,padding=15)#nn.Conv1d(66,32,1)
        self.conv4_2 = nn.Conv1d(32,32,31,padding=15)
        self.se = SE_Block(64)

        self.conv5 = nn.Conv1d(64,64,1)
        self.linear1 = nn.Linear(64, 2)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.bn3_2 = nn.BatchNorm1d(32)
        self.bn4_2 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(64)
        
    def forward(self,input1, input2):
        #First model
        input1 = input1.permute(0,2,1)
        input2 = input2.permute(0,2,1)
        x = F.relu(self.bn1(self.conv1(input1)))
        x = F.relu(self.bn2(self.conv2(x)))
        #x = self.se1(x)
        x = nn.MaxPool1d(2)(x) #180->90
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = nn.MaxPool1d(2)(x) #90-> 45
        #Second model
        x1 = F.relu(self.bn1_2(self.conv1_2(input2)))
        x1 = F.relu(self.bn2_2(self.conv2_2(x1)))
        #x1 = self.se2(x1)
        x1 = nn.MaxPool1d(2)(x1)
        
        x1 = F.relu(self.bn3_2(self.conv3_2(x1)))
        x1 = F.relu(self.bn4_2(self.conv4_2(x1)))
        x1 = nn.MaxPool1d(2)(x1)
        #Fusion
        x2 = torch.cat((x,x1),axis=1)#x1+x
        x2 = self.se(x2)
        x2 = F.relu(self.bn5(self.conv5(x2)))
        
        x2, _ = torch.max(x2,2)
        #x2 = torch.flatten(x2,start_dim=1)
        out = self.linear1(x2)
        return out


#CNN-LSTM
class Model2(nn.Module):
    def __init__(self):#, args):
        super(Model2, self).__init__()
        #input shape: bs*seq_len*num_features
        self.conv1 = nn.Conv1d(70,64,31,padding=15)
        self.conv2 = nn.Conv1d(64,64,31,padding=15)
        self.lstm1 = nn.LSTM(64, 64, 3, batch_first=True)
        self.conv3 = nn.Conv1d(64,64,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.linear1 = nn.Linear(64*90, 2)
        self.se = SE_Block(64)
        
    def forward(self,input1):
        input1 = input1.permute(0,2,1)
        x = F.relu(self.bn1(self.conv1(input1)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = nn.MaxPool1d(2)(x)
        x = x.permute(0,2,1)
        #LSTM
        x2,_ = self.lstm1(x)
        x2 = x2.permute(0,2,1)
        x2 = self.se(x2)
        x2 = F.relu(self.bn3(self.conv3(x2)))
        
        x2 = torch.flatten(x2,start_dim=1)
        #x2,_ = torch.max(x2,2)
        out = self.linear1(x2)
        return out