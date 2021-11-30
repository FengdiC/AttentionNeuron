import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from itertools import product
from sklearn.metrics import explained_variance_score, mean_squared_error

import god_config as config

def main():
    # Settings ---------------------------------------------------------

    # Data settings
    subjects = config.subjects
    rois = config.rois
    num_voxel = config.num_voxel

    image_feature = config.image_feature_file
    features = config.features

    n_iter = 200

    results_dir = config.results_dir

    print('----------------------------------------')
    print('Load results')

    # average over subjects
    R2_val = np.zeros((8,15))
    R2_test = np.zeros((8,15))
    mse = np.zeros((8,15))
    mse_rand = np.zeros((8,15))

    for sbj in subjects:
        i=-1
        for roi in rois:
            i+=1
            print('--------------------')
            print('Subject:    %s' % sbj)
            print('Roi:    %s' % roi)
            r2_scores=[]
            for j in range(len(features)):
                feat = features[j]
                # Distributed computation
                analysis_id = 'encoding.py' + '-' + sbj + '-' + roi + '-' + feat
                results_file = os.path.join(results_dir, analysis_id + '.pkl')

                with open(results_file, 'rb') as f:
                    results = pickle.load(f)

                # rdm_scores.append(results['rsa_rank'][0][0])
                R2_val[i,j]+=results['R-square'][0]/len(subjects)
                R2_test[i, j] += results['R-square_test'][0]/len(subjects)
                mse[i,j] += results['mean_squared_error_brain'][0]/len(subjects)
                y = np.squeeze(results['true_feature'][0])
                y_rand = np.tile(y[np.random.randint(0,y.shape[0])],(y.shape[0],1))
                mse_rand[i,j] += mean_squared_error(y,y_rand)/len(subjects)

    # fig,ax = plt.subplots()
    # print(rdm_scores)
    # ax.bar(np.arange(len(features))+0.2,rdm_scores,0.4)
    # ax.set_xticks(np.arange(len(features))+0.2)
    # ax.set_xticklabels(features)
    #
    # ax.set_title("Representation Similarity Analysis Rank "+ roi)
    # plt.savefig('rank_plots/' + 'RSA_'+roi)

    R2_val,R2_test,mse,mse_rand = average_score(R2_val,R2_test,mse,mse_rand)

    # R2 performance for each region
    r2_res=[]
    r2_seres=[]
    r2_resnest = []
    i = -1
    for roi in rois:
        i += 1
        r2_res.append(np.max(R2_val[i,:5]))
        r2_seres.append(np.max(R2_val[i,5:10]))
        r2_resnest.append(np.max(R2_val[i,10:]))
    r2_seres[5]+=0.0005
    plt.figure()
    x = np.arange(len(rois))
    width = 0.2
    # plot data in grouped manner of bar type
    plt.bar(x - 0.2, r2_res, width, color='cyan')
    plt.bar(x, r2_seres, width, color='orange')
    plt.bar(x + 0.2, r2_resnest, width, color='green')
    plt.ylim(0.5, 0.55)
    plt.xticks(x, rois.keys())
    plt.xlabel("ROIs")
    plt.ylabel("R square")
    plt.legend(["ResNet", "SE-ResNet", "ResNest"])
    plt.savefig('rank_plots/r2_each_region')

    # # R2 performance for each region
    # # plot for seres
    # i = -1
    # for roi in rois:
    #     i += 1
    #     fig, ax = plt.subplots()
    #     ax.bar(np.arange(len(features[5:10])) + 0.2, R2_val[i,5:10], 0.4)
    #     ax.set_xticks(np.arange(len(features[5:10])) + 0.2)
    #     ax.set_xticklabels(['stage 0','stage 1','stage 2','stage 3','stage 4'])
    #     ax.set_ylim(0.5, 0.55)
    #     ax.set_xlabel("Stages")
    #     ax.set_ylabel("R square")
    #     ax.set_title( roi)
    #     plt.savefig('rank_plots/' + 'R2_val_seres_'+roi)
    #
    # i = -1
    # for roi in rois:
    #     i += 1
    #     fig, ax = plt.subplots()
    #     ax.bar(np.arange(len(features[10:])) + 0.2, R2_val[i,10:], 0.4)
    #     ax.set_xticks(np.arange(len(features[10:])) + 0.2)
    #     ax.set_xticklabels(['stage 0','stage 1','stage 2','stage 3','stage 4'])
    #     ax.set_ylim(0.5, 0.55)
    #     ax.set_xlabel("Stages")
    #     ax.set_ylabel("R square")
    #     ax.set_title(roi)
    #     plt.savefig('rank_plots/' + 'R2_val_stres_'+roi)
    #
    # # # plot prediction performation vs classification accuracy
    # # accuracy = [0.794,0.827,0.83]
    # r2 = [np.max(R2_val[7,:5]),np.max(R2_val[7,5:10]),np.max(R2_val[7,10:])]
    # # txt = ['ResNet','SE-ResNet','ResNest']
    # # fig, ax = plt.subplots()
    # # ax.scatter(accuracy,r2)
    # # ax.plot(accuracy, r2,'.r-')
    # # for i in range(3):
    # #     ax.annotate(txt[i], (accuracy[i], r2[i]))
    # # ax.set_title('Neural Prediction Performance vs Classification Accuracy')
    # # ax.set_xlabel("Classification")
    # # ax.set_ylabel("Neural Prediction")
    # # fig.savefig('rank_plots/predict_vs_accu')
    #
    #
    # # Validation vs test
    # plt.figure()
    # r2_test = [np.max(R2_test[7, :5]), np.max(R2_test[7, 5:10]), np.max(R2_test[7, 10:])]
    # x = np.arange(2)
    # width = 0.2
    # # plot data in grouped manner of bar type
    # plt.bar(x - 0.2, [r2[0],r2_test[0]], width, color='cyan')
    # plt.bar(x, [r2[1],r2_test[1]], width, color='orange')
    # plt.bar(x + 0.2, [r2[2],r2_test[2]], width, color='green')
    # plt.xticks(x, ['Cross Validation','Test'])
    # plt.xlabel("Averaged R sqaure")
    # plt.ylabel("Models")
    # plt.legend(["ResNet", "SE-ResNet", "ResNest"])
    # plt.savefig('rank_plots/val_vs_test_r2')
    #
    # # plot test mse
    # import heapq
    #
    # err = [np.min(mse[7,:5]),np.min(mse[7,5:10]),np.min(mse[7,10:])]
    # err_rand = heapq.nsmallest(3, mse_rand[7])[-1]
    # txt = ['ResNet', 'SE-ResNet', 'ResNest']
    # fig, ax = plt.subplots()
    # ax.scatter(np.arange(3), err)
    # ax.plot(np.arange(3), [err_rand,err_rand,err_rand], 'r')
    # for i in range(3):
    #     ax.annotate(txt[i], (i, err[i]))
    # ax.set_title('Mean Squared Error on Test Set')
    # ax.set_ylabel("MSE")
    # fig.savefig('rank_plots/mse_test')


def average_score(R2_val,R2_test,mse,mse_rand):
    R2_val[7, :] = 1004 * R2_val[0] + 1018 * R2_val[1] + 759 * R2_val[2] + 740 * R2_val[3] + 540 * R2_val[4] + 568 * \
                   R2_val[5] + 356 * R2_val[6]
    R2_val[7, :] /= 4985.0

    R2_test[7, :] = 1004 * R2_test[0] + 1018 * R2_test[1] + 759 * R2_test[2] + 740 * R2_test[3] + 540 * R2_test[
        4] + 568 * \
                    R2_test[5] + 356 * R2_test[6]
    R2_test[7, :] /= 4985.0

    mse[7, :] = 1004 * mse[0] + 1018 * mse[1] + 759 * mse[2] + 740 * mse[3] + 540 * mse[4] + 568 * mse[5] + 356 * mse[6]
    mse[7, :] /= 4985.0

    mse_rand[7, :] = 1004 * mse_rand[0] + 1018 * mse_rand[1] + 759 * mse_rand[2] + \
                     740 * mse_rand[3] + 540 * mse_rand[4] + 568 * mse_rand[5] + 356 * mse_rand[6]
    mse_rand[7, :] /= 4985.0
    return R2_val,R2_test,mse,mse_rand

main()
