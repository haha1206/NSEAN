import numpy as np
from W_Construct import norm_W,KNN
from data_loader import load_mat
import math
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
def NMD(X,n_knn,n_cluster,n_iter,lambda_1):
    n_view = X.size
    N = X[0].shape[0]
    W = []
    for i in range(n_view):
        W.append(norm_W(KNN(X[i],n_knn)))
    alpha = np.ones(n_view)/n_view
    W_new = np.zeros((N,N))
    for i in range(n_view):
        W_new += W[i]*alpha[i]
    val, vec = np.linalg.eigh(W[0])
    F = vec[:, -n_cluster:]
    F = WHH(W_new,F,n_cluster)
    F,alpha = opt(W,F,alpha,n_iter,n_cluster,lambda_1)
    y_pred = np.argmax(F,axis=1)+1
    return y_pred,alpha,F

def opt(W,  F, alpha, NITER,n_cluster,lambda_1):

    n_view = alpha.size
    N, _ = W[0].shape
    FF = F @ F.T
    for Iter in range(NITER):
        W_new = np.zeros((N, N))
        P_v = np.zeros(n_view)
        #alpha
        for i in range(n_view):
            P_v[i] = (1/(2*lambda_1))*np.square(np.linalg.norm(W[i]-FF))
        alpha = EProjSimplex_new(-P_v)
        for i in range(n_view):
            W_new += W[i] * alpha[i]
        # update F
        F = WHH(W_new ,F,n_cluster )

    return F,alpha

def EProjSimplex_new(v,k=1):
    ft = 1
    n = v.size
    v_0 = v - np.mean(v) + k/n
    v_min = np.min(v_0)
    if v_min<0:
        f = 1
        lambda_m = 0
        while abs(f)> 1e-10:
            v_1 = v_0 - lambda_m
            posidx = v_1>0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v_1[posidx])-k
            lambda_m = lambda_m-f/g
            ft = ft+1
            if ft>100:
                v_1[v_1<0]=0
                x = v_1
                break
            v_1[v_1<0]=0
            x = v_1
    else:
        x = v_0
    return x


def WHH(W, F, c,mu=3,rou=1.5):
    N_inter =10
    threshold = 1e-10
    val = 0
    cnt = 0
    n,k = F.shape
    G = F
    lambda_n = np.ones(F.shape)
    for a in range(N_inter):
        #update F
        M = -mu*G-W@G+lambda_n
        U, lambda_s, VT = np.linalg.svd(M)
        A = np.concatenate((-1*np.identity(c),np.zeros((n-c,c))),axis=0)
        F = U@A@VT
        #update G
        H = F+(1/mu)*lambda_n+(1/mu)*W@F
        G[H > 0] = H[H > 0]
        G[H <= 0] = 0
        #update lambda_n
        lambda_n = lambda_n + mu*(F-G)
        #update mu
        mu = rou*mu
        val_old = val
        val = np.trace(F.T@(np.identity(n)-W)@F)
        if abs(val - val_old) < threshold:
            if cnt >=5:
                break
            else:
                cnt +=1
        else:
            cnt = 0

    return F


def acc(y_true, y_pred):
    # Calculate clustering accuracy
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    r_ind,c_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(r_ind,c_ind)]) * 1.0 / y_pred.size


def NMI(A,B):
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)   # Find the intersection of two arrays.
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
        Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)


if __name__=="__main__":
    X,GT = load_mat('data/HW2.mat')
    GT = GT.reshape(np.max(GT.shape), )
    y_pred,alpha,F = NMD(X,15,10,20,100)
    ACC = acc(GT, y_pred)
    NMI = metrics.normalized_mutual_info_score(GT, y_pred)
    Purity = purity_score(GT, y_pred)
    ARI = metrics.adjusted_rand_score(GT, y_pred)
    print('clustering accuracy: {}, NMI: {}, Purity: {},ARI: {}'.format(ACC, NMI, Purity, ARI))
