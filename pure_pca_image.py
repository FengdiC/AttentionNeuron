import numpy as np
import gc

gc.collect()

id = dict()
f = open("train_id.csv")
for line in f:
    line = line.strip('\n')
    (key, val) = line.split(",")
    id[val] = key

f = open("test_id.csv")
for line in f:
    line = line.strip('\n')
    (key, val) = line.split(",")
    id[val] = key

import pickle5 as pickle

f = open('data/train_features_resnest50d.pkl', 'rb')
train = pickle.load(f)

f = open('data/test_features_resnest50d.pkl', 'rb')
test = pickle.load(f)

# image = {'res0': [], 'res1': [], 'res2': [], 'res3': [], 'res4': [], 'seres0': [],
#          'seres1': [], 'seres2': [], 'seres3': [], 'seres4': [], 'label': []}
image = {'stres0':[],'stres1':[],'stres2':[],'stres3':[],'stres4':[],'label':[]}

pca_models_seres = {}
pca_models_res = {}

from sklearn.decomposition import PCA

for i in range(5):
    for key in list(train['resnest50d'].keys())[:150]:
        feat = train['resnest50d'][key][i].flatten()
        image['stres' + str(i)].append(feat)
    pca_models_seres['pca' + str(i)] = PCA(svd_solver='randomized')
    pca_models_seres['pca' + str(i)].fit(np.array(image['stres' + str(i)], dtype='float32'))
    print("seres: ", pca_models_seres['pca' + str(i)].transform(np.expand_dims(feat, axis=0)).shape)
    image['stres' + str(i)] = []
    gc.collect()

# # resnet
# for i in range(5):
#     for key in list(train['resnest50d'].keys())[:150]:
#         feat = train['resnest50d'][key][i].flatten()
#         image['res' + str(i)].append(feat)
#     pca_models_res['pca' + str(i)] = PCA(svd_solver='randomized')
#     pca_models_res['pca' + str(i)].fit(np.array(image['res' + str(i)], dtype='float32'))
#     image['res' + str(i)] = []
#     gc.collect()

print("START LOADING")
wrong_data = 0
for key in train['resnest50d'].keys():
    try:
        image['label'].append(id['"' + key + '"'])
        for i in range(5):
            feat = train['resnest50d'][key][i].flatten()
            feat = pca_models_seres['pca' + str(i)].transform(np.expand_dims(feat, axis=0))
            feat = np.squeeze(feat)
            image['stres' + str(i)].append(feat)

            # feat = train['resnet50'][key][i].flatten()
            # feat = pca_models_res['pca' + str(i)].transform(np.expand_dims(feat, axis=0))
            # feat = np.squeeze(feat)
            # image['res' + str(i)].append(feat)
    except KeyError:
        print(key)
        wrong_data += 1
        print(wrong_data)

for key in test['resnest50d'].keys():
    try:
        image['label'].append(id['"' + key + '"'])
        for i in range(5):
            feat = test['resnest50d'][key][i].flatten()
            feat = pca_models_seres['pca' + str(i)].transform(np.expand_dims(feat, axis=0))
            feat = np.squeeze(feat)
            image['stres' + str(i)].append(feat)

            # feat = test['resnet50'][key][i].flatten()
            # feat = pca_models_res['pca' + str(i)].transform(np.expand_dims(feat, axis=0))
            # feat = np.squeeze(feat)
            # image['res' + str(i)].append(feat)
    except KeyError:
        print(key)
        wrong_data += 1
        print(wrong_data)

for key in image.keys():
    image[key] = np.array(image[key])
with open('data/image_stres_pca.pkl', 'wb') as f:
    pickle.dump(image, f)