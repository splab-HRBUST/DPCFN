import torch
from sklearn.decomposition import PCA
from torchvision import transforms
import data_utils
from torch.utils.data import DataLoader
from gru_am import VGGVox2
from scg_em_model import GatedRes2Net,SEGatedLinearConcatBottle2neck
from torchvision.models.resnet import Bottleneck, BasicBlock
from sklearn.datasets import make_blobs
from sklearn import svm
import numpy as np
import torch
import numpy
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def transform_matrix(PhibX):

    artSbx = torch.mm(PhibX,PhibX.t())
    eigVals, eigVecs = torch.linalg.eig(artSbx)
    eigVals = torch.abs(eigVals)
    # TODO
    eigVecs = torch.abs(eigVecs)

    # Ignore zero eigenvalues
    #maxEigVal = torch.max(eigVals)
    #non_zeroEigIndx = torch.nonzero(eigVals / maxEigVal > 1e-6).squeeze()
    #eigVals = torch.index_select(eigVals, 0, non_zeroEigIndx)
    #eigVecs = torch.index_select(eigVecs, 1, non_zeroEigIndx)

    # Sort in descending order
    #_, index = torch.sort(eigVals, descending=True)
    #eigVals = torch.index_select(eigVals, 0, index)
    #eigVecs = torch.index_select(eigVecs, 1, index)

    # Calculate the actual eigenvectors for the between-class scatter matrix (Sbx)
    #SbxEigVecs = torch.mm(PhibX, eigVecs)
    #SbxEigVecs.backward()
    SbxEigVecs = eigVecs

    # Normalize to unit length to create orthonormal eigenvectors for Sbx
    cx = eigVals.size(0)
    for i in range(cx):
        SbxEigVecs[:, i] =  SbxEigVecs[:, i] /torch.norm(SbxEigVecs[:, i])

    # Unitize the between-class scatter matrix (Sbx) for X
    SbxEigVals = torch.diag(eigVals)
    Wbx = torch.mm(SbxEigVecs, torch.inverse(torch.sqrt(SbxEigVals)))
    return cx, Wbx

def dca_fuse(X, Y, label,device):
    """
    :param X:   pxn matrix containing the first set of training feature vectors
%                   p:  dimensionality of the first feature set
%                   n:  number of training samples


    :param Y:   qxn matrix containing the second set of training feature vectors
%                   q:  dimensionality of the second feature set
%
    :param label: 1xn row vector of length n containing the class labels
    :return:
%       Ax  :   Transformation matrix for the first data set (rxp)
%               r:  maximum dimensionality in the new subspace
%       Ay  :   Transformation matrix for the second data set (rxq)
%       Xs  :   First set of transformed feature vectors (rxn)
%       Xy  :   Second set of transformed feature vectors (rxn)
    """

    #X = X.clone()
    #Y = Y.clone() 

    p, n = X.shape
    if Y.shape[1] != n:
        raise ValueError('X and Y must have the same number of columns (samples).')
    elif len(label) != n:
        raise ValueError('The length of the label must be equal to the number of samples.')
    elif n == 1:
        raise ValueError('X and Y must have more than one column (samples)')

    q = Y.shape[0]

    # Normalize features (this has to be done for both train and test data)
    # X = (X - torch.mean(X, dim=1, keepdim=True)) / torch.std(X, dim=1, keepdim=True)
    # Y = (Y - torch.mean(Y, dim=1, keepdim=True)) / torch.std(Y, dim=1, keepdim=True)

    # Compute mean vectors for each class and for all training data
    classes = torch.unique(label)
    c = len(classes)
    cellX = [None] * c
    cellY = [None] * c
    nSample = torch.zeros(c)
    #nSample = nSample.to(device)

    for i, class_label in enumerate(classes):
        index = torch.nonzero(label == class_label).squeeze()
        nSample[i] = len(index)
        cellX[i] = X[:, index]
        cellY[i] = Y[:, index]

    meanX = torch.mean(X, dim=1, keepdim=True)
    meanY = torch.mean(Y, dim=1, keepdim=True)

    classMeanX = torch.zeros(p, c)
    classMeanY = torch.zeros(q, c)
    classMeanX = classMeanX.to(device)
    classMeanY = classMeanY.to(device)

    for i in range(c):
        classMeanX[:, i] = torch.mean(cellX[i], dim=1)
        classMeanY[:, i] = torch.mean(cellY[i], dim=1)

    PhibX = torch.zeros(p, c)
    PhibY = torch.zeros(q, c)
    PhibX = PhibX.to(device)
    PhibY = PhibY.to(device)

    for i in range(c):
        PhibX[:, i] = torch.sqrt(nSample[i]) * (classMeanX[:, i].clone() - meanX.squeeze())
        PhibY[:, i] = torch.sqrt(nSample[i]) * (classMeanY[:, i].clone() - meanY.squeeze())

    #PhibX.backward()

    cx, Wbx = transform_matrix(PhibX)
    cy, Wby = transform_matrix(PhibY)
    Wbx = Wbx.to(device)
    Wby = Wby.to(device)
    print('Wbx:{}'.format(Wbx.shape))


    

    # Project data in a space, where the between-class scatter matrices are identity and the classes are separated
    r = min(cx, cy)  # Maximum length of the desired feature vector
    Wbx = Wbx[:, :r]
    Wby = Wby[:, :r]
    

    Xp = torch.mm(Wbx.t(), X)  # Transform X (pxn) to Xprime (rxn)
    Yp = torch.mm(Wby.t(), Y)  # Transform Y (qxn) to Yprime (rxn)
    

    # Unitize the between-set covariance matrix (Sxy)
    Sxy = torch.mm(Xp, Yp.t())  # Between-set covariance matrix
    #Sxy.backward()

    U, S, V = torch.svd(Sxy)  # Singular Value Decomposition (SVD)

    #if S.dim() < 2:
    #    S = S.unsqueeze(dim=1)
    S = torch.diag(S)
    Wcx = torch.mm(U, torch.linalg.pinv(torch.sqrt(S)))  # Transformation matrix for Xp
    Wcy = torch.mm(V, torch.linalg.pinv(torch.sqrt(S)))  # Transformation matrix for Yp


    Xs = torch.mm(Wcx.t(), Xp)  # Transform Xprime to XStar
    Ys = torch.mm(Wcy.t(), Yp)  # Transform Yprime to YStar

    Ax = torch.mm(Wcx.t(), Wbx.t())  # Final transformation Matrix of size (rxp) for X
    Ay = torch.mm(Wcy.t(), Wby.t())  # Final transformation Matrix of size (rxq) for Y

    return Ax, Ay, Xs, Ys

# Example usage:
# Ax, Ay, train
'''
n = 100
p = 256
q = 256
X = torch.randn(p, n)
Y = torch.randn(q, n)
labels = torch.randint(0, 2, (1, n)).squeeze()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Ax, Ay, Xs, Ys = dca_fuse(X, Y, labels)
print('Xs:{}'.format(Xs.shape))
'''

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    track='logical'
    is_logical = (track == 'logical')

    

    train_set = data_utils.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms,feature_name='spect')
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,pin_memory=True,num_workers=64)

    save_path1 = '/g813_u1/mkj/twice_attention_networks-main/ta-network-main/epoch_10.pth'
    save_path2 = '/g813_u1/mkj/twice_attention_networks-main/ta-network-main/epoch_95.pth'
    
    #arr = np.zeros((1, 1024))  
    #emb = torch.tensor(arr).to(device)
    
    target = []

    #for batch_x, batch_y, batch_meta in tqdm(train_loader):
    batch_x, batch_y, batch_meta = next(iter(train_loader))

    #model1 =torch.load(save_path1)
    batch_x = batch_x.to(device)
    net1 = VGGVox2(BasicBlock, [2, 2, 2, 2],emb_dim=512)
    model1 = net1.to(device)
    model1.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_path1,map_location="cpu").items()},strict=False)
    x = model1(batch_x)
    print('x:{}'.format(x.shape))

      
    net2 = GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, pretrained=False,loss='softmax')
    model2 = net2.to(device)
    model2.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_path2,map_location="cpu").items()},strict=False)
    y = model2(batch_x)
    print('y:{}'.format(y.shape))

    label = batch_meta[4]


    x=torch.transpose(x,dim0=0,dim1=1)
    y=torch.transpose(y,dim0=0,dim1=1)
    Ax,Ay,x,y  = dca_fuse(x,y,label,device)
    print('Ax:{}'.format(Ax.shape))

    torch.save(Ax, "./Ax.pt")
    torch.save(Ay, "./Ay.pt")

    out = torch.cat((x,y),dim=0) #特征融合
    out=torch.transpose(out,dim0=0,dim1=1)

    print('out:{}'.format(out.shape))
      #emb = torch.cat((emb,out),dim=0)#保存tensor
      #print('emb:{}'.format(emb.shape))

    label = batch_meta[4]
    print('label:{}'.format(label.shape))
    target.extend(label)
      #print('target:{}'.format(target))

    dev_set = data_utils.ASVDataset(is_train=False, is_logical=is_logical,
                                    transform=transforms,
                                    feature_name='spect', is_eval=True, eval_part=0)
    data_loader = DataLoader(dev_set, batch_size=30, shuffle=True)
    b_x, b_y, b_meta = next(iter(data_loader))
    b_x = b_x.to(device)
    a = model1(b_x)
    b = model2(b_x)
    la = b_meta[4]

    a=torch.transpose(a,dim0=0,dim1=1)
    b=torch.transpose(b,dim0=0,dim1=1)
    Ax,Ay,a,b  = dca_fuse(a,b,la,device)

    dev_c = torch.cat((a,b),dim=0) #特征融合
    dev_c=torch.transpose(dev_c,dim0=0,dim1=1)


    
    
    #out_train, out_test,target_train, target_test = train_test_split(out.detach().cpu().numpy(),target,test_size=0.3)
    clf = svm.SVC(C=5, gamma=0.05,max_iter=300)
    clf.fit(out.detach().cpu().numpy(), target)
    print(clf.score(dev_c.detach().cpu().numpy(),la))

    




    


    
