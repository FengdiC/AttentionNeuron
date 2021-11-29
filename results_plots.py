import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from itertools import product

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

    for sbj, feat in product(subjects, features):
        print('--------------------')
        print('Subject:    %s' % sbj)
        # print('Roi:    %s' % roi)
        rdm_scores=[]
        r2_scores=[]
        pred_corr=[]
        for roi in rois:

            # Distributed computation
            analysis_id = 'encoding.py' + '-' + sbj + '-' + roi + '-' + feat
            results_file = os.path.join(results_dir, analysis_id + '.pkl')

            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            rdm_scores.append(results['rsa_rank'][0][0])
            r2_scores.append(results['R-square'][0])
            pred_corr.append(results['Corr_brain'][0])
        fig,ax = plt.subplots()
        print(rdm_scores)
        ax.bar(np.arange(len(rois))+0.3,rdm_scores,0.6)
        ax.set_xticks(np.arange(len(rois))+0.3)
        ax.set_xticklabels(rois.keys())

        ax.set_title("Representation Similarity Analysis Rank "+ feat)
        plt.savefig('rank_plots/' + 'RSA_'+feat)

        # fig, ax = plt.subplots()
        # ax.bar(np.arange(len(features)) + 0.3, r2_scores, 0.6)
        # ax.set_xticks(np.arange(len(features)) + 0.3)
        # ax.set_xticklabels(features)
        #
        # ax.set_title("R Square analysis " + roi)
        # plt.savefig('rank_plots/' + 'R2_' + roi)
        #
        # fig, ax = plt.subplots()
        # ax.bar(np.arange(len(features)) + 0.3, pred_corr, 0.6)
        # ax.set_xticks(np.arange(len(features)) + 0.3)
        # ax.set_xticklabels(features)
        #
        # ax.set_title("Correlation Between True and Predicted Brain Signals " + roi)
        # plt.savefig('rank_plots/' + 'Corr_' + roi)

main()