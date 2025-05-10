import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as EuDist2
from W_Construct import KNN
from data_loader import load_mat
import math
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
def NMD(X,k,n_cluster,n_iter):
    n_view = X.size
    num = X[0].shape[0]

    # normalization
    for i in range(n_view):
        for j in range(num):
            X[i][j, :] = (X[i][j, :] - np.mean(X[i][j, :])) / np.std(X[i][j, :])

    # 初始化
    disX_In = np.zeros((num, num, n_view))
    index = np.zeros((num, num, n_view))
    for i in range(n_view):
        disX_In[:, :, i] = EuDist2(X[i], X[i])
        index[:, :, i] = np.argsort(disX_In[:, :, i], axis=1)
    ## init w
    w = np.ones((n_view, 1)) / n_view
    #初始化Sv
    S_mvc = np.zeros((n_view,num, num))
    S_SUM = np.zeros((num, num))
    for i in range(n_view):
        S_mvc[i] = KNN(X[i],k)
        S_SUM = S_SUM+S_mvc[i]
    # 初始化F
    S_int = S_SUM
    sum_s = np.sum(S_int, axis=1)
    D = np.zeros_like(S_int)
    np.fill_diagonal(D, sum_s)
    L = D - S_int
    val, vec = np.linalg.eigh(L)
    F = vec[:, :n_cluster]

    y_pred,G= opt(S_mvc,F,disX_In,index,w,k,n_iter,n_cluster)
    return y_pred,G
def opt(S_mvc,  F,disX_In,index, w, k,NITER,n_cluster):

    n_view = w.size
    N, _ = S_mvc[0].shape
    FF = F @ F.T
    eps = 2.2204e-16
    for Iter in range(NITER):
        W_new = np.zeros((N, N))
        #update Sv
        for v in range(n_view):
            for i in range(N):
                id = index[:,:,v][i, 1:(k + 2)]
                id = list(map(int, id))
                di = disX_In[:,:,v][i, id]
                numerator = di[k] - di + 2 * w[v]*  FF[i, id]-2 * w[v]*  FF[i, id[k]]
                denominator1 = k * di[k] - sum(di[:k-1])
                denominator2 = 2 * w[v]* sum(FF[i, id[:k]])-2 * k * w[v] * FF[i, id[k]]
                S_mvc[v][i, id] = numerator / (denominator1 + denominator2 + eps)
                S_mvc[v][i, id][S_mvc[v][i, id]<0]=0
        #update w
        for i in range(n_view):
            w[i] = 0.5/np.linalg.norm(S_mvc[i]-FF)
        for i in range(n_view):
            S_mvc[i] = (S_mvc[i] + S_mvc[i].T) / 2
            W_new += S_mvc[i] * w[i]
        F = WHH(W_new ,F,n_cluster)
        y_pred = np.argmax(F, axis=1) + 1
    return y_pred,F


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
def WHH(W, F, c,mu = 0.1,rou= 1.005):
    N_inter = 10
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
        G = F+(1/mu)*lambda_n+(1/mu)*W@F
        G[G < 0] = 0
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

def F_score(labels_true, labels_pred):
    n = len(labels_true)
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(n):
        for j in range(i + 1, n):
            if labels_true[i] == labels_true[j] and labels_pred[i] == labels_pred[j]:
                tp += 1
            elif labels_true[i] != labels_true[j] and labels_pred[i] == labels_pred[j]:
                fp += 1
            elif labels_true[i] == labels_true[j] and labels_pred[i] != labels_pred[j]:
                fn += 1
            else:
                tn += 1

    Pre = tp / (tp + fp)
    Rec = tp / (tp + fn)

    F_score = 2*Pre*Rec / (Pre+Rec)

    return F_score

def NE(y_pred):
    n = len(y_pred)
    ar, num = np.unique(y_pred, return_counts=True)
    c = len(ar)
    ne = 0
    for i in range(c):
        ne = ne + (num[i]/n)*np.log(num[i]/n)
    return (-1/np.log(c))*ne

if __name__=="__main__":

    import time
    X,GT = load_mat('data/100leaves.mat')
    n_cluster = len(np.unique(GT))
    N = X[0].shape[0]
    GT = GT.reshape(np.max(GT.shape), )
    c = n_cluster
    GT = GT.reshape(np.max(GT.shape), )
    t1 = time.time()
    n_cluster = len(np.unique(GT))
    y_pred,F= NMD(X,20,n_cluster,20)
    t2 = time.time()
    print(t2-t1)
    ACC = acc(GT, y_pred)
    NMI = metrics.normalized_mutual_info_score(GT, y_pred)
    Purity = purity_score(GT, y_pred)
    ARI = metrics.adjusted_rand_score(GT, y_pred)
    print('clustering accuracy: {}, NMI: {}, Purity: {},ARI: {}'.format(ACC, NMI, Purity, ARI))
