import numpy as np
from scipy.stats import pearsonr,kendalltau
import matplotlib
import matplotlib.pyplot as plt

def select_x(x_label,y_label):
    n = x_label.shape[0]
    idx = np.zeros(n)
    for i in range(n):
        arr_index = np.where(y_label == x_label[i][0] )
        if arr_index[0].size>0:
            idx[i] = 1
    print(np.sum(idx))
    return idx.astype('bool')

def pred_corr(pred,true):
    n = pred.shape[1]
    corr = np.zeros(n)
    for i in range(n):
        corr[i] = pearsonr(pred[:,i],true[:,i])[0]
    return np.mean(corr)

def rdm_both(x_pt,x_im,labels_pt,labels_im):
    labels_pt = labels_pt.astype('int')
    labels_im = labels_im.astype('int')
    print('#seen,#im: ',labels_pt.shape,labels_im.shape)
    label_set_pt = np.unique(labels_pt)
    label_set_im = np.unique(labels_im)
    n = label_set_pt.shape[0]+label_set_im.shape[0]
    print("size of rdm: ",n)

    metric = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i < label_set_pt.shape[0]:
                x_idx = np.where(labels_pt==label_set_pt[i])[0]
                # x_cat = np.mean(x_pt[x_idx, :],axis=0)
                x_cat = x_pt[x_idx[0],:]
            else:
                x_idx = np.where(labels_im == label_set_im[i-50])[0][0]
                x_cat = x_im[x_idx, :]
            if j < label_set_pt.shape[0]:
                y_idx = np.where(labels_pt==label_set_pt[j])[0]
                # y_cat = np.mean(x_pt[y_idx, :],axis=0)
                y_cat = x_pt[y_idx[0], :]
            else:
                y_idx = np.where(labels_im == label_set_im[j - 50])[0][0]
                y_cat = x_im[y_idx, :]
            corr,_ = pearsonr(x_cat, y_cat)
            metric[i,j] = 1-abs(corr)

    return metric

def rdm(x,labels):
    labels_cat = labels.astype('int')
    label_set = np.unique(labels_cat)
    n = label_set.shape[0]
    print("size of rdm: ",n)

    metric = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            x_idx = np.where(labels_cat==label_set[i])[0]
            x_cat = np.mean(x[x_idx, :],axis=0)
            # x_cat = x[x_idx[0],:]
            y_idx = np.where(labels_cat==label_set[j])[0]
            y_cat = np.mean(x[y_idx, :],axis=0)
            # y_cat = x[y_idx[0], :]
            corr,_ = pearsonr(x_cat, y_cat)
            metric[i,j] = 1-abs(corr)

    return metric

def kendall_tau(feature,brain):
    return kendalltau(feature,brain)

def plot_rdm_both(rdm,pt_len,comb):
    fig, ax = plt.subplots()
    im = ax.imshow(rdm)

    # We want to show all ticks...
    ax.set_xticks([0,pt_len,rdm.shape[0]])
    ax.set_yticks([0,pt_len,rdm.shape[0]])
    ax.tick_params(length=8 ,width=2)
    # ... and label them with the respective list entries
    ax.set_xticklabels(['seen','imaginary'])
    ax.set_yticklabels(['seen','imaginary'])
    ax.figure.colorbar(im)

    # Rotate the tick labels and set their alignment.
    import types
    SHIFT = 25.  # Data coordinates
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x + self.customShiftValue),
                                       label)
    for label in ax.yaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_y = types.MethodType(lambda self, x: matplotlib.text.Text.set_y(self, x + self.customShiftValue),
                                       label)

    ax.set_title("Representation dissimilarity metric "+comb)
    fig.tight_layout()
    plt.pause(0.3)
    plt.savefig('rdm_plots/'+comb)

def plot_rdm(rdm,comb):
    fig, ax = plt.subplots()
    im = ax.imshow(rdm)
    ax.figure.colorbar(im)
    ax.set_title("Representation dissimilarity metric "+comb)
    fig.tight_layout()
    # plt.pause(0.3)
    plt.savefig('rdm_plots/'+comb)