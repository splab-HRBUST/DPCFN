import data_utils
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import torch
from torch import nn
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock
from gru_am import VGGVox2
from scg_em_model import GatedRes2Net,SEGatedLinearConcatBottle2neck
from scipy.signal import medfilt
from scipy import signal
from dct_self import dct2,idct2
import data_utils
from torch.autograd import Variable
import torch.optim as optim
#from pick_data import get_data1,get_data2
from torch.nn.parameter import Parameter
import os
from dca2 import *
#from torchsummary import summary
#from sklearn.cross_decomposition import CCA
#from PCA import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




class cqt_mgd(nn.Module):
    def __init__(self, block, layers, num_classes, emb_dim1,emb_dim2,T,Q,E1,E11,E2,E22,E3,E33,Ax,Ay,
                 zero_init_residual=False):
        super(cqt_mgd, self).__init__()
        self.embedding_size1 = emb_dim1
        self.embedding_size2=emb_dim2
        self.num_classes = num_classes
        self.gru=VGGVox2(BasicBlock, [2, 2, 2, 2],emb_dim=self.embedding_size1)
        self.scg= GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, pretrained=False,loss='softmax')
        self.classifier_layer = nn.Linear(self.embedding_size2, self.num_classes)

        
        
        
        self.Ax=torch.nn.Parameter(Ax)
        self.Ay=torch.nn.Parameter(Ay)
        self.T=torch.nn.Parameter(torch.transpose(T,dim0=0,dim1=1))
        self.Q=torch.nn.Parameter(torch.transpose(Q,dim0=0,dim1=1))#（512，256）
        self.fai_x=torch.nn.Parameter(torch.ones(512))
        self.fai_y=torch.nn.Parameter(torch.ones(512))
        self.fai1=torch.nn.Parameter(torch.ones(256))
        self.fai2=torch.nn.Parameter(torch.ones(256))

        self.E1=E1
        self.E11=E11
        self.E2=E2
        self.E22=E22
        self.E3=E3
        self.E33=E33


    def forward(self,x,y,device):
   
        
        y=self.gru(y) #(32,512)
        x=self.scg(x) #(32,512)
        x=torch.transpose(x,dim0=0,dim1=1) #(512,32)
        y=torch.transpose(y,dim0=0,dim1=1)#(512,32)
        
        
        x=torch.mm(self.Ax,x)
        y=torch.mm(self.Ay,y)
        fai_x,fai_y,T,Q,m,mu_y,fai,E3=self.get_em_param(x,y,device)
      
        #out = torch.cat((x,y),dim=0)
        #print('out:{}'.format(out.shape))
        emb=torch.transpose(E3,dim0=0,dim1=1)#(32,256)
        out=self.classifier_layer(emb)
        #print('out:{}'.format(out.shape))
        
        return  x,y,fai_x,fai_y,T,Q,m,mu_y,fai,out,emb
    



    def get_em_param(self,x,y,device):#x:cqt(512,32) y:mgd(512,32) T:(512,256) Q:(512,256)

        
        #N1=x.shape[0]#N=512
        N2 = x.shape[1]#batch:32
        #N3=T.shape[1]#256
    
        #lamda=torch.cat((T,Q),0)
        T=self.T
        Q=self.Q
        fai_x=self.fai_x
        fai_y=self.fai_y
        #fai_z=torch.cat((fai_x,fai_y),0)#(1024)

        m=torch.sum(x,dim=1)/N2 # (512)
        mu_y=torch.sum(y,dim=1)/N2#(512)
        #mu_z=torch.cat((m,mu_y),0)

        centeredM=x-m.unsqueeze(1)#(512,32)
        variancesM = torch.mean(centeredM ** 2, dim=1)#(512)
        centeredY=y-mu_y.unsqueeze(1)#(512,32)
        variancesY =torch.mean(centeredY**2,dim=1) #(512)

        #centeredZ=torch.cat((centeredM,centeredY),0)#(1024,32)


   
        I=torch.eye(256,256)
        I=I.to(device)
        # E step 更新E1,E2
        B1=torch.transpose(T,dim0=0,dim1=1)/fai_x#(256,512)
        B2=torch.transpose(Q,dim0=0,dim1=1)/fai_y#(256,512)
   
        L1=I+B1@T#(256,256)   (256,512)@(512,256)
        L2=I+B2@Q#(256,256)   (256,512)@(512,256)
   
        cov1=torch.linalg.pinv(L1)#(256,256)
        E1=cov1@(B1@centeredM)#(256,32) E1为隐变量w1
        E11=E1@torch.transpose(E1,dim0=0,dim1=1)+cov1*N2#(256,256)
        E1_T=torch.transpose(E1,dim0=0,dim1=1)#(32,256)
    
        cov2=torch.linalg.pinv(L2)#(256,256)
        E2=cov2@(B2@centeredY)#(256,32) E2为隐变量w2
        E22=E2@torch.transpose(E2,dim0=0,dim1=1)+cov2*N2#(256,256)
        E2_T=torch.transpose(E2,dim0=0,dim1=1)#(32,256)

    
    
        # M step 更新T、Q
    
        T=(centeredM@E1_T)@torch.linalg.pinv(E11)#(512,256)
    
        Q=(centeredY@E2_T)@torch.linalg.pinv(E22)#(512,256)
    
        fai_x=variancesM -torch.mean(T *(T@E11),dim=1) #(512) 
     
        fai_y=variancesY -torch.mean(Q *(Q@E22),dim=1) #(512)


    
        #第二部分
        Z=torch.cat((E1,E2),0)
        fai1=self.fai1
        fai2=self.fai2
        fai=torch.cat((fai1,fai2),0)#(512)

        variancesZ = torch.mean(Z ** 2, dim=1)#(512)
    
        #第二个E步
        I2=torch.eye(512,512)
        I2=I2.to(device)
        
        B3=1/fai#(512)
    
        L3=I2+B3#(512,512)

        cov3=torch.linalg.pinv(L3)#(512,512)
        
        c=cov3/fai
        E3=c@Z#(512,512)
        E33=cov3*N2+E3@torch.transpose(E3,dim0=0,dim1=1)#(512,512)
        E3_T=torch.transpose(E3,dim0=0,dim1=1)

        #第二个M步
        fai=variancesZ-torch.mean(2*(Z@E3_T),dim=1)+torch.mean(E33,dim=1)
       
        return fai_x,fai_y,T,Q,m,mu_y,fai,E3
    
    
    def loss_em(self,x,y,fai_x,fai_y,T,Q,m,mu_y,fai,device):

        #z=torch.cat((x,y),0)
        #mu_z=torch.cat((m,mu_y),0)
        #fai_z=torch.cat((fai_x,fai_y),0)
        #lamda=torch.cat((T,Q),0)
        
        #loss1
        N1=x.shape[1]#32

        
        L_sum1= torch.tensor(0.0)
        L_sum2=torch.tensor(0.0)
       
        x_T=torch.transpose(x,dim0=0,dim1=1)#(32,512)
        E1_T=torch.transpose(self.E1,dim0=0,dim1=1)#(32,256)

        L_sum1= torch.tensor(0.0)
        L_sum2=torch.tensor(0.0)
        for i in range(x_T.shape[0]):
            xi = x_T[i].unsqueeze(1)#(512,1)
            Ei1 = E1_T[i].unsqueeze(1)#(256,1)

            mux=xi-m.unsqueeze(1)#(512,1)
            mux_T=torch.transpose(mux,dim0=0,dim1=1)#(1,512)

            cov_eps=torch.ones(fai_x.shape[0])* (1e-12)
            cov_eps=cov_eps.to(device)
            faix_=1/(fai_x+cov_eps)#(1024) fai逆

            L11=-0.5*torch.log(torch.norm(fai_x))-0.5*((mux_T*faix_)@mux)
            
            L21=(mux_T*faix_)@T@Ei1

            L_sum1=L_sum1+L11
            L_sum2=L_sum2+L21

        l=(torch.transpose(T,dim0=0,dim1=1)*faix_)@T
            
        L31=torch.sum(self.E11*l)
        
        loss1=L_sum1+L_sum2-0.5*L31
        loss1=loss1/N1

        #loss2
        N2=y.shape[1]#32

        
        L_sum3= torch.tensor(0.0)
        L_sum4=torch.tensor(0.0)
       
        y_T=torch.transpose(y,dim0=0,dim1=1)#(32,512)
        E2_T=torch.transpose(self.E2,dim0=0,dim1=1)#(32,256)

        L_sum3= torch.tensor(0.0)
        L_sum4=torch.tensor(0.0)
        for i in range(y_T.shape[0]):
            yi = y_T[i].unsqueeze(1)#(512,1)
            Ei2 = E2_T[i].unsqueeze(1)#(256,1)

            muy=yi-mu_y.unsqueeze(1)#(512,1)
            muy_T=torch.transpose(muy,dim0=0,dim1=1)#(1,512)

            cov_eps=torch.ones(fai_y.shape[0])* (1e-12)
            cov_eps=cov_eps.to(device)
            faiy_=1/(fai_y+cov_eps)#(1024) fai逆

            L12=-0.5*torch.log(torch.norm(fai_y))-0.5*((muy_T*faiy_)@muy)
            
            L22=(muy_T*faiy_)@Q@Ei2

            L_sum3=L_sum3+L12
            L_sum4=L_sum4+L22

        l=(torch.transpose(Q,dim0=0,dim1=1)*faiy_)@Q
            
        L32=torch.sum(self.E22*l)
        
        loss2=L_sum1+L_sum2-0.5*L32
        loss2=loss2/N2

        #loss3
        Z=torch.cat((self.E1,self.E2),0)
        N3=Z.shape[1]#32

        
        L_sum5= torch.tensor(0.0)
        L_sum6=torch.tensor(0.0)
       
        Z_T=torch.transpose(Z,dim0=0,dim1=1)#(32,512)
        E3_T=torch.transpose(self.E3,dim0=0,dim1=1)#(32,256)

        L_sum5= torch.tensor(0.0)
        L_sum6=torch.tensor(0.0)
        for i in range(Z_T.shape[0]):
            Zi = Z_T[i].unsqueeze(1)#(512,1)
            Ei3 = E3_T[i].unsqueeze(1)#(512,1)

            #muy=yi-mu_y.unsqueeze(1)#(512,1)
            Zi_T=torch.transpose(Zi,dim0=0,dim1=1)#(1,512)

            cov_eps=torch.ones(fai.shape[0])* (1e-12)
            cov_eps=cov_eps.to(device)
            fai_=1/(fai+cov_eps)#(1024) fai逆

            L13=-0.5*torch.log(torch.norm(fai))-0.5*((Zi_T*fai_)@Zi)
            
            L23=(Zi_T*fai_)@Ei3

            L_sum3=L_sum3+L13
            L_sum4=L_sum4+L23

        #l=(torch.transpose(Q,dim0=0,dim1=1)*faiy_)@Q
            
        L33=torch.sum(self.E33*fai_)
        
        loss3=L_sum5+L_sum6-0.5*L33
        loss3=loss3/N3

       
        return -(loss1+loss2+loss3)
        
    
    
   
    
def em_param(x,y,T,Q):#x:cqt(512,32) y:mgd(512,32) T:(512,256) Q:(512,256)

    x=torch.transpose(x,dim0=0,dim1=1) #(512,32)
    y=torch.transpose(y,dim0=0,dim1=1)#(512,32)
    T=torch.transpose(T,dim0=0,dim1=1)
    Q=torch.transpose(Q,dim0=0,dim1=1)

    N1=x.shape[0]#N=512
    N2 = x.shape[1]#batch:32
    N3=T.shape[1]#256
    
    #lamda=torch.cat((T,Q),0)

    fai_x=torch.ones(512)
    fai_y=torch.ones(512)
    #fai_z=torch.cat((fai_x,fai_y),0)#(1024)

    m=torch.mean(x,dim=1) # (512)
    mu_y=torch.mean(y,dim=1)#(512)
    #mu_z=torch.cat((m,mu_y),0)

    centeredM=x-m.unsqueeze(1)#(512,32)
    variancesM = torch.mean(centeredM ** 2, dim=1)#(512)
    centeredY=y-mu_y.unsqueeze(1)#(512,32)
    variancesY =torch.mean(centeredY**2,dim=1) #(512)

    #centeredZ=torch.cat((centeredM,centeredY),0)#(1024,32)


   
    I=torch.eye(256,256)
    #I=I.to(device)
    # E step 更新E1,E2
    B1=torch.transpose(T,dim0=0,dim1=1)/fai_x#(256,512)
    B2=torch.transpose(Q,dim0=0,dim1=1)/fai_y#(256,512)
   
    L1=I+B1@T#(256,256)   (256,512)@(512,256)
    L2=I+B2@Q#(256,256)   (256,512)@(512,256)
   
    cov1=torch.linalg.pinv(L1)#(256,256)
    E1=cov1@(B1@centeredM)#(256,32) E1为隐变量w1
    E11=E1@torch.transpose(E1,dim0=0,dim1=1)+cov1*N2#(256,256)
    E1_T=torch.transpose(E1,dim0=0,dim1=1)#(32,256)
    
    cov2=torch.linalg.pinv(L2)#(256,256)
    E2=cov2@(B2@centeredY)#(256,32) E2为隐变量w2
    E22=E2@torch.transpose(E2,dim0=0,dim1=1)+cov2*N2#(256,256)
    E2_T=torch.transpose(E2,dim0=0,dim1=1)#(32,256)

    
    
    # M step 更新T、Q
    
    T=(centeredM@E1_T)@torch.linalg.pinv(E11)#(512,256)
    
    Q=(centeredY@E2_T)@torch.linalg.pinv(E22)#(512,256)
    
    fai_x=variancesM -torch.mean(T *(T@E11),dim=1) #(512) 
     
    fai_y=variancesY -torch.mean(Q *(Q@E22),dim=1) #(512)


    
    #第二部分
    Z=torch.cat((E1,E2),0)
    fai1=torch.ones(256)
    fai2=torch.ones(256)
    fai=torch.cat((fai1,fai2),0)#(512)

    variancesZ = torch.mean(Z ** 2, dim=1)#(512)
    
    #第二个E步
    I2=torch.eye(512,512)
    
    B3=1/fai#(512)
    
    L3=I2+B3#(512,512)

    cov3=torch.linalg.pinv(L3)#(512,512)
    
    c=cov3/fai
    E3=c@Z#(512,32)
    
    E33=cov3*N2+E3@torch.transpose(E3,dim0=0,dim1=1)#(512,512)
    E3_T=torch.transpose(E3,dim0=0,dim1=1)

    #第二个M步
    fai=variancesZ-torch.mean(2*(Z@E3_T),dim=1)+torch.mean(E33,dim=1)
    

    return E1,E11,E2,E22,E3,E33
   
   
   
    
    



