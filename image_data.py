import numpy as np
import gc, random
import pickle5 as pickle
from sklearn.decomposition import PCA
gc.collect()

def main():
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

    f = open('data/train_features_resnest50d.pkl','rb')
    train = pickle.load(f)

    f = open('data/test_features_resnest50d.pkl','rb')
    test = pickle.load(f)

    # image = {'res0':[],'res1':[],'res2':[],'res3':[],'res4':[],'seres0':[],
    #          'seres1':[],'seres2':[],'seres3':[],'seres4':[],'label':[]}
    image = {'stres0':[],'stres1':[],'stres2':[],'stres3':[],'stres4':[],'label':[]}

    # pca_models_stres={}
    #
    # from sklearn.decomposition import PCA
    # for i in range(5):
    #     for key in list(train['resnest50d'].keys())[:150]:
    #         feat = train['resnest50d'][key][i].flatten()
    #         image['stres' + str(i)].append(feat)
    #     pca_models_stres['pca' + str(i)] = PCA(svd_solver='randomized')
    #     pca_models_stres['pca' + str(i)].fit(np.array(image['stres' + str(i)],dtype='float32'))
    #     print("stres: ", pca_models_stres['pca' + str(i)].transform(np.expand_dims(feat,axis=0)).shape)
    #     image['stres' + str(i)] = []
    #     gc.collect()
    #
    # # # resnet
    # # for i in range(5):
    # #     for key in list(train['resnet50'].keys())[:150]:
    # #         feat = train['resnet50'][key][i].flatten()
    # #         image['res' + str(i)].append(feat)
    # #     pca_models_res['pca' + str(i)] = PCA( svd_solver='randomized')
    # #     pca_models_res['pca' + str(i)].fit(np.array(image['res' + str(i)],dtype='float32'))
    # #     image['res' + str(i)] = []
    # #     gc.collect()

    pca1_list=[]
    pca2_list=[]

    wrong_data=0
    idx = np.arange(70000)
    idx = np.random.choice(idx, size=20000)
    for i in range(5):
        for key in train['resnest50d'].keys():
            feat = train['resnest50d'][key][i].flatten()
            # feat = pca_models_stres['pca' + str(i)].transform(np.expand_dims(feat,axis=0))
            # feat = np.squeeze(feat)
            image['stres'+str(i)].append(feat[idx])
            #
            # feat = train['resnet50'][key][i].flatten()
            # # feat = pca_models_res['pca' + str(i)].transform(np.expand_dims(feat, axis=0))
            # # feat = np.squeeze(feat)
            # # image['res' + str(i)].append(feat)
            # image['res' + str(i)].append(feat[idx])
        for key in test['resnest50d'].keys():
            feat = test['resnest50d'][key][i].flatten()
            image['stres' + str(i)].append(feat[idx])

            # feat = test['resnet50'][key][i].flatten()
            # image['res' + str(i)].append(feat[idx])
        pca1_list.append(PCA(svd_solver='randomized'))
        image['stres'+str(i)]=list(pca1_list[-1].fit_transform(np.array(image['stres' + str(i)], dtype='float32')))
        # pca2_list.append(PCA(svd_solver='randomized'))
        # image['res' + str(i)] = list(pca2_list[-1].fit_transform(np.array(image['res' + str(i)], dtype='float32')))
        print("done")
    for key in train['resnest50d'].keys():
        try:
            image['label'].append(id['"' + key + '"'])
        except KeyError:
            print(key)
            wrong_data +=1
            print(wrong_data)
    for key in test['resnest50d'].keys():
        try:
            image['label'].append(id['"' + key + '"'])
        except KeyError:
            print(key)
            wrong_data +=1
            print(wrong_data)

    for key in image.keys():
        image[key] = np.array(image[key])
    with open('data/image_stres.pkl','wb') as f:
        pickle.dump(image,f)

def combine_date():
    image = pickle.load(open('data/image.pkl', 'rb'))
    image_stres = pickle.load(open('data/image_stres.pkl', 'rb'))
    image_comb = {'res':[],"seres":[],"stres":[],"s2":[]} #,"s1":[],"s2":[],"s3":[],"s4":[]}
    for i in range(image['res0'].shape[0]):
        res = np.concatenate((image['res0'][i],image['res1'][i],image['res2'][i],image['res3'][i],image['res4'][i]))
        image_comb['res'].append(res)

        seres = np.concatenate((image['seres0'][i], image['seres1'][i], image['seres2'][i],
                               image['seres3'][i], image['seres4'][i]))
        image_comb['seres'].append(seres)

        stres = np.concatenate((image_stres['stres0'][i], image_stres['stres1'][i], image_stres['stres2'][i],
                               image_stres['stres3'][i], image_stres['stres4'][i]))
        image_comb['stres'].append(stres)

        for j in range(2,3,1):
            s = np.concatenate((image['res'+str(j)][i], image['seres'+str(j)][i], image_stres['stres'+str(j)][i]))
            image_comb['s'+str(j)].append(s)
    for key in image_comb.keys():
        image_comb[key] = np.array(image_comb[key])
    image_comb['label']=image['label']
    with open('data/image_comb.pkl', 'wb') as f:
        pickle.dump(image_comb, f)

combine_date()