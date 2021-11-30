import nibabel as nib
import numpy as np
import bdpy, os, pickle
import matplotlib.pyplot as plt
from matplotlib import  colors
import numpy.linalg as npl
import god_config as config

data = bdpy.BData('data/Subject1.h5')

img_obj = nib.load('data/derivatives_preproc-spm_output_sub-01_ses-perceptionTraining01_func_sub-01_ses-perceptionTraining01_task-perception_run-01_bold_preproc (1).nii.gz')
M = img_obj.affine[:3,:3]
abc = img_obj.affine[:3,3]
img = img_obj.get_fdata()[:,:,:,:4]
img[:,:,:,1:]=254
print(img.shape)

# corresponding color
colors = [np.array([254,0,0]),np.array([0,254,0]),np.array([0,0,254]),np.array([51,102,0]),
          np.array([102,255,255]),np.array([102,0,0]),np.array([0,102,204]),np.array([255,102,255])]

# data.show_metadata()
v1 = data.get_metadata('ROI_V1')

# 1= res, 2=channel, 3= spatial, 4= mixed, 0=unknown
max_idx = np.random.randint(1, high=4, size=v1.shape[0])

v2 = data.get_metadata('ROI_V2')
v3 = data.get_metadata('ROI_V3')
v4 = data.get_metadata('ROI_V4')
loc = data.get_metadata('ROI_LOC')
ffa = data.get_metadata('ROI_FFA')
ppa = data.get_metadata('ROI_PPA')
def get_region(v1,v2,v3,v4,loc,ffa,ppa,max_idx):
    for i in range(v1.shape[0]):
        if ~np.isnan(v1[i]):
            max_idx[i]=0
        elif ~np.isnan(v2[i]):
            max_idx[i]=1
        elif ~np.isnan(v3[i]):
            max_idx[i]=2
        elif ~np.isnan(v4[i]):
            max_idx[i]=3
            # cannot see v4 on the brain
        elif ~np.isnan(loc[i]):
            max_idx[i]=4
        elif ~np.isnan(ffa[i]):
            max_idx[i]=5
        elif ~np.isnan(ppa[i]):
            max_idx[i]=6
        else:
            max_idx[i]=7
    print(np.sum(max_idx==7))
    return max_idx


def get_best_fit_model(l, region_data):
    subjects = config.subjects
    rois = config.rois

    results_dir = config.results_dir

    print('----------------------------------------')
    print('Load results')

    for roi in rois:
        print('--------------------')
        print('Roi:    %s' % roi)
        # Distributed computation
        analysis_id = 'encoding.py' + '-' + 'Subject1' + '-' + roi + '-' + 's2'
        results_file = os.path.join(results_dir, analysis_id + '.pkl')

        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        best_model = results['feature_portion'][0]
        l[~np.isnan(region_data[roi])] = best_model

    return l
region_data={'V1' : v1,
        'V2' : v2,
        'V3' : v3,
        'V4' : v4,
        'LOC' : loc,
        'FFA' : ffa,
        'PPA' : ppa}
# max_idx = get_best_fit_model(max_idx,region_data)
max_idx= get_region(v1,v2,v3,v4,loc,ffa,ppa,max_idx)
voxel_x = data.get_metadata('voxel_x')
voxel_y = data.get_metadata('voxel_y')
voxel_z = data.get_metadata('voxel_z')

def get_voxel_idx(M,v):
    idx = npl.inv(M).dot(v-abc)
    idx = idx[:3].astype('uint8')
    idx = np.minimum(idx,np.array([63,63,49]))
    idx = np.maximum(idx, np.array([0,0,0]))
    return idx[:3].astype('uint8')

for i in range(voxel_x.shape[0]):
    if np.isnan(voxel_x[i]):
        continue
    else:
        idx = get_voxel_idx(M,np.array([voxel_x[i],voxel_y[i],voxel_z[i]]))
        c = max_idx[i]
        c = colors[c]
        img[idx[0],idx[1],idx[2],1:] = c

# rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
# ras_pos = img.copy().view(dtype=rgb_dtype)

# show images
def show_slices(slices):
    fig,axes = plt.subplots(1,len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice[:,:,0].T,cmap="gray",origin='lower')
        axes[i].imshow(np.transpose(slice[:,:,1:],axes=[1,0,2]).astype(np.uint8),origin='lower', alpha=0.5)
    fig.savefig('brain_slice/region')

slice_0 = img[32,:,:,:]
slice_1 = img[:,32,:,:]
slice_2 = img[:,:,25,:]
show_slices([slice_0,slice_2])

