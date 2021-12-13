import torch
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
import os
import numpy as np

import pickle
# load both models
# To get the classifier output, turn features_only to False

'skresnext50_32x4d'
att_model = timm.create_model('skresnext50_32x4d', pretrained=True, features_only=True)
att_model.eval()
#print(f'features info: {att_model.feature_info.info}')

# this is same for both models
config = resolve_data_config({}, model=att_model)
transform = create_transform(**config)

# feature extraction example
#url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#urllib.request.urlretrieve(url, filename)

test_features = {}
test_features['skresnext50_32x4d'] = {}

for img_name in tqdm(os.listdir("dataset/test/")):
    try:
        print("opening: ", img_name)
        img = Image.open("dataset/test/"+img_name).convert('RGB')
    except:
        print("fail: ", img_name)
        from IPython import embed; embed()
        continue
    tensor = transform(img).unsqueeze(0)

    # get features only
    features_att = att_model(tensor)
    test_features['skresnext50_32x4d'][img_name] = {}

    # we got features from 5 different stages in the network
    # dim: [batch, channel, H, W]

    for stage_idx, stage_features in enumerate(features_att):
        test_features['skresnext50_32x4d'][img_name][stage_idx] = stage_features.detach().numpy()


with open('test_features_skresnext.pkl', 'wb') as pkl_f:
    pickle.dump(test_features, pkl_f, protocol=pickle.HIGHEST_PROTOCOL)


train_features = {}
train_features['skresnext50_32x4d'] = {}

for img_name in tqdm(os.listdir("dataset/training/")):
    try:
        img = Image.open("dataset/training/"+img_name).convert('RGB')
    except:
        print("fail: ", img_name)
        from IPython import embed; embed()
        continue
    tensor = transform(img).unsqueeze(0)

    # get features only
    features_att = att_model(tensor)
    train_features['skresnext50_32x4d'][img_name] = {}

    # we got features from 5 different stages in the network
    # dim: [batch, channel, H, W]

    for stage_idx, stage_features in enumerate(features_att):
        train_features['skresnext50_32x4d'][img_name][stage_idx] = stage_features.detach().numpy()


with open('train_features_skresnext.pkl', 'wb') as pkl_f:
    pickle.dump(train_features, pkl_f, protocol=pickle.HIGHEST_PROTOCOL)


## Load features by using:
#with open('train_features.pkl', 'rb') as pkl_f:
#    loaded_features = pickle.load(pkl_f)
