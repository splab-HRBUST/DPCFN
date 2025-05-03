import numpy as np
import torch

def dca_fuse(X, Y, label,device):
    p, n = X.shape
    if Y.shape[1] != n:
        raise ValueError('X and Y must have the same number of columns (samples).')
    elif len(label) != n:
        raise ValueError('The length of the label must be equal to the number of samples.')
    elif n == 1:
        raise ValueError('X and Y must have more than one column (samples)')

    q = Y.shape[0]
    
    classes = torch.unique(label)
    c = len(classes)
    cellX = [None] * c
    cellY = [None] * c
    nSample = torch.zeros(c)
    nSample = nSample.to(device)

    for i in range(c):
        index = torch.where(label==classes[i])[0]
        nSample[i] = len(index)
        cellX[i] = X[:,index]
        cellY[i] = Y[:,index]

    #meanX = np.mean(X, axis=1)  # Mean of all training data in X
    #meanY = np.mean(Y, axis=1)  # Mean of all training data in Y
    meanX = torch.mean(X, dim=1)  # Mean of all training data in X
    meanY = torch.mean(Y, dim=1)  # Mean of all training data in Y


    classMeanX = torch.zeros((p, c))
    classMeanY = torch.zeros((q, c))
    classMeanX = classMeanX.to(device)
    classMeanY = classMeanY.to(device)

    for i in range(c):
        classMeanX[:,i] = torch.mean(cellX[i], axis=1)   # Mean of each class in X
        classMeanY[:,i] = torch.mean(cellY[i], axis=1)   # Mean of each class in Y

    PhibX = torch.zeros((p, c))
    PhibY = torch.zeros((q, c))

    for i in range(c):
        PhibX[:,i] = torch.sqrt(nSample[i]) * (classMeanX[:,i]-meanX)
        PhibY[:,i] = torch.sqrt(nSample[i]) * (classMeanY[:,i]-meanY)
   
   #Diagolalize the between-class scatter matrix (Sb) for X
    artSbx = torch.matmul(PhibX.T ,PhibX)   # Artificial Sbx (artSbx) is a (c x c) matrix
    eigVals, eigVecs = torch.linalg.eig(artSbx) #eigVals是一个包含特征值的一维张量，而eigVecs则是一个包含特征向量的二维张量
    eigVals = torch.abs(eigVals)
    eigVecs = torch.real(eigVecs)
    #eigVals = torch.abs(torch.diag(eigVals))
    #print('eigVals:{}'.format(eigVals.shape))

    maxEigVal = torch.max(eigVals)
    zeroEigIndx = torch.nonzero(eigVals / maxEigVal < 1e-6).squeeze()
    eigVals = eigVals[~zeroEigIndx]
    eigVecs = eigVecs[:, ~zeroEigIndx]
    ##print('eigVals:{}'.format(eigVals.shape))
    ##print('eigVecs:{}'.format(eigVecs.shape))

    #eigVals = eigVals.long()
    index = torch.argsort(eigVals,descending=True)
    eigVals = eigVals[index]
    eigVecs = eigVecs[:,index]

    SbxEigVecs = torch.matmul(PhibX, eigVecs)
    #SbxEigVecs = torch.matmul(artSbx, eigVecs)
    #SbxEigVecs = eigVecs

    cx = len(eigVals)
    for i in range(cx):
        SbxEigVecs[:, i] = SbxEigVecs[:, i] / torch.linalg.norm(SbxEigVecs[:, i])

    SbxEigVals = torch.diag(eigVals)
    Wbx = torch.matmul(SbxEigVecs, torch.linalg.pinv(torch.sqrt(SbxEigVals)))
    Wbx = Wbx.to(device)

    #Diagolalize the between-class scatter matrix (Sb) for Y
    artSby = torch.matmul(PhibY.T,PhibY)
    eigVals, eigVecs = torch.linalg.eig(artSby) #eigVals将会存储artSby的特征值，而eigVecs将会存储artSby的特征向量
    eigVals = torch.abs(eigVals)
    eigVecs = torch.real(eigVecs)

    maxEigVal = max(eigVals)
    zeroEigIndx = torch.nonzero(eigVals / maxEigVal < 1e-6).squeeze()
    eigVals = eigVals[~zeroEigIndx]
    eigVecs = eigVecs[:, ~zeroEigIndx]

    index = torch.argsort(eigVals,descending=True)
    eigVals = eigVals[index]
    eigVecs = eigVecs[:,index]
   
    SbyEigVecs = torch.matmul(PhibY, eigVecs)
    #SbyEigVecs = eigVecs

    cy = len(eigVals)
    for i in range(cy):
        SbyEigVecs[:, i] = SbyEigVecs[:, i] / torch.linalg.norm(SbyEigVecs[:, i])

    SbyEigVals = torch.diag(eigVals)
    Wby = torch.matmul(SbyEigVecs, torch.linalg.pinv(torch.sqrt(SbyEigVals)))
    Wby = Wby.to(device)

   #Project data in a space, where the between-class scatter matrices are
    r = min(cx, cy)
    Wbx = Wbx[:,:r]
    Wby = Wby[:,:r]

    Xp = torch.matmul(Wbx.T, X)
    Yp = torch.matmul(Wby.T, Y)

    Sxy = torch.matmul(Xp, Yp.T)
    U, S, Vt = torch.linalg.svd(Sxy)
    S1 = torch.diag(S)

    Wcx = torch.matmul(U, torch.linalg.pinv(torch.sqrt(S1)))
    Wcy = torch.matmul(Vt, torch.linalg.pinv(torch.sqrt(S1)))

    Xs = torch.matmul(Wcx.T, Xp)
    Ys = torch.matmul(Wcy.T, Yp)

    Ax = torch.matmul(Wcx.T, Wbx.T)
    Ay = torch.matmul(Wcy.T, Wby.T)

    return Ax, Ay, Xs, Ys
