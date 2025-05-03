import torch
from sklearn.decomposition import PCA

def transform_matrix(PhibX):

    artSbx = torch.mm(PhibX,PhibX.t())
    eigVals, eigVecs = torch.linalg.eig(artSbx)
    eigVals = torch.abs(eigVals)
    # TODO
    eigVecs = torch.abs(eigVecs)

    # Ignore zero eigenvalues
    maxEigVal = torch.max(eigVals)
    non_zeroEigIndx = torch.nonzero(eigVals / maxEigVal < 1e-6).squeeze()
    eigVals = torch.index_select(eigVals, 0, non_zeroEigIndx)
    eigVecs = torch.index_select(eigVecs, 1, non_zeroEigIndx)

    # Sort in descending order
    _, index = torch.sort(eigVals, descending=True)
    eigVals = torch.index_select(eigVals, 0, index)
    eigVecs = torch.index_select(eigVecs, 1, index)

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
    nSample = nSample.to(device)

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
    #PhibX = PhibX.to(device)
    #PhibY = PhibY.to(device)

    for i in range(c):
        PhibX[:, i] = torch.sqrt(nSample[i]) * (classMeanX[:, i].clone() - meanX.squeeze())
        PhibY[:, i] = torch.sqrt(nSample[i]) * (classMeanY[:, i].clone() - meanY.squeeze())

    #PhibX.backward()

    cx, Wbx = transform_matrix(PhibX)
    cy, Wby = transform_matrix(PhibY)
    Wbx = Wbx.to(device)
    Wby = Wby.to(device)
    

    # # Diagonalize the between-class scatter matrix (Sb) for X
    # artSbx = torch.mm(PhibX.t(), PhibX)
    # eigVals, eigVecs = torch.linalg.eig(artSbx)
    # eigVals = torch.abs(eigVals)
    # # TODO
    # eigVecs = torch.abs(eigVecs)
    #
    # # Ignore zero eigenvalues
    # maxEigVal = torch.max(eigVals)
    # non_zeroEigIndx = torch.nonzero(eigVals / maxEigVal > 1e-6).squeeze()
    # eigVals = torch.index_select(eigVals, 0, non_zeroEigIndx)
    # eigVecs = torch.index_select(eigVecs, 1, non_zeroEigIndx)
    #
    # # Sort in descending order
    # _, index = torch.sort(eigVals, descending=True)
    # eigVals = torch.index_select(eigVals, 0, index)
    # eigVecs = torch.index_select(eigVecs, 1, index)
    #
    # # Calculate the actual eigenvectors for the between-class scatter matrix (Sbx)
    # SbxEigVecs = torch.mm(PhibX, eigVecs)
    #
    # # Normalize to unit length to create orthonormal eigenvectors for Sbx
    # cx = eigVals.size(0)
    # for i in range(cx):
    #     SbxEigVecs[:, i] /= torch.norm(SbxEigVecs[:, i])
    #
    # # Unitize the between-class scatter matrix (Sbx) for X
    # SbxEigVals = torch.diag(eigVals)
    # Wbx = torch.mm(SbxEigVecs, torch.inverse(torch.sqrt(SbxEigVals)))

    # Diagonalize the between-class scatter matrix (Sb) for Y
    # artSby = torch.mm(PhibY.t(), PhibY)
    # eigVals, eigVecs = torch.linalg.eig(artSby)
    # eigVals = torch.abs(eigVals)
    #
    # # Ignore zero eigenvalues
    # maxEigVal = torch.max(eigVals)
    # zeroEigIndx = torch.nonzero(eigVals / maxEigVal < 1e-6).squeeze()
    # eigVals = torch.index_select(eigVals, 0, zeroEigIndx)
    # eigVecs = torch.index_select(eigVecs, 1, zeroEigIndx)
    #
    # # Sort in descending order
    # _, index = torch.sort(eigVals, descending=True)
    # eigVals = torch.index_select(eigVals, 0, index)
    # eigVecs = torch.index_select(eigVecs, 1, index)
    #
    # # Calculate the actual eigenvectors for the between-class scatter matrix (Sby)
    # SbyEigVecs = torch.mm(PhibY, eigVecs)
    #
    # # Normalize to unit length to create orthonormal eigenvectors for Sby
    # cy = eigVals.size(0)
    # for i in range(cy):
    #     SbyEigVecs[:, i] /= torch.norm(SbyEigVecs[:, i])
    #
    # # Unitize the between-class scatter matrix (Sby) for Y
    # SbyEigVals = torch.diag(eigVals)
    # Wby = torch.mm(SbyEigVecs, torch.inverse(torch.sqrt(SbyEigVals)))

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
Ax, Ay, Xs, Ys = dca_fuse(X, Y, labels)
'''