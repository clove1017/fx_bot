from torch.utils.data import Dataset
from torch import nn
import numpy as np
import torch
#create custom dataset for pytorch 
class fx_data_set(Dataset):
    def __init__(self,in_dat,out_dat,num_prec):
        self.data=[]
        self.num_prec=num_prec
        for i in range(0,len(in_dat[0,0])):
            open_dat=in_dat[0:num_prec,0,i]
            close_dat=in_dat[num_prec:int(2*num_prec),0,i]
            high_dat=in_dat[int(2*num_prec):int(3*num_prec),0,i]
            low_dat=in_dat[int(3*num_prec):int(4*num_prec),0,i]
            
            input_set=f"{open_dat},{close_dat},{high_dat},{low_dat}"
            out=out_dat[0,0,i]
            self.data.append([input_set,out])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        num_prec=self.num_prec
        data_=self.data[idx][0].split(',')
        o1=(data_[0].split('[')[1].split(']')[0]).split(' ')
        o=np.float32(list(filter(('').__ne__, o1)))
        c1=(data_[1].split('[')[1].split(']')[0]).split(' ')
        c=np.float32(list(filter(('').__ne__, c1)))
        h1=(data_[2].split('[')[1].split(']')[0]).split(' ')
        h=np.float32(list(filter(('').__ne__, h1)))
        l1=(data_[3].split('[')[1].split(']')[0]).split(' ')
        l=np.float32(list(filter(('').__ne__, l1)))
        np_dat=np.empty((4*num_prec,1,1),dtype='float32')
        for i in range(0,num_prec):
            ind=i*4
            np_dat[ind,0,0]=o[i]
            np_dat[ind+1,0,0]=c[i]
            np_dat[ind+2,0,0]=h[i]
            np_dat[ind+3,0,0]=l[i]
        torch_dat=torch.from_numpy(np_dat)
        torch_dat=torch_dat.permute(2,0,1)
        out=torch.from_numpy((np.array(np.float32(self.data[idx][1]))))
        return torch_dat,out
    
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self,num_prec,device):
        self.device=device
        self.num_prec=num_prec
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.batch_norm=nn.BatchNorm1d(20)
        self.first_layer = nn.Sequential(
            nn.Linear(4,1,bias=True),
            nn.ELU()
            
        )
        self.second_layer=nn.Sequential(
            nn.Linear(num_prec,1,bias=False),
            nn.ELU()
        )
    
    def migrate_device(self,new_dev):
        self.device=new_dev

    def forward(self, x):
        device=self.device
        num_prec=self.num_prec
        if x.size()[0]==1:
            x=self.flatten(x)
            log_arr=torch.zeros(num_prec)
            log_arr=log_arr.to(device)
            for m in range(0,num_prec):
                ind=m*4
                #split=x[0][ind:ind+num_prec-1]
                #split=split.to(device)
                log_arr[m]=self.first_layer(x[0][ind:ind+num_prec-1])
            logits=self.second_layer(log_arr)
        else:
            x = self.flatten(x)
            logits=torch.zeros(len(x))
            logits=logits.to(device)
            for i in range(0,len(x)):
                log_arr=torch.zeros(num_prec)
                log_arr=log_arr.to(device)
                for m in range(0,num_prec):
                    ind=m*4
                    #split=x[i][ind:ind+num_prec-1]
                    #split=split.to(device)
                    log_arr[m]=self.first_layer(x[i][ind:ind+num_prec-1])
                logits[i]=self.second_layer(log_arr)
                
        
        logits=logits[:,None]
        return logits
