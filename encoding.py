import os
import sys
import pickle,json
from itertools import product
from time import time

import numpy as np
import pandas as pd
from scipy import stats

from slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression  # For quick demo
import matplotlib.pyplot as plt

import bdpy
from bdpy.bdata import concat_dataset
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.dataform import append_dataframe
from bdpy.distcomp import DistComp

import god_config as config

from util import rdm,kendall_tau, plot_rdm, pred_corr, select_x

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

    # Misc settings
    analysis_basename = os.path.basename(__file__)

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_all = {}
    for sbj in subjects:
        if len(subjects[sbj]) == 1:
            data_all[sbj] = bdpy.BData(subjects[sbj][0])
        else:
            # Concatenate data
            suc_cols = ['Run', 'Block']
            data_all[sbj] = concat_dataset([bdpy.BData(f) for f in subjects[sbj]],
                                           successive=suc_cols)

    # load image
    image = pickle.load(open('data/image.pkl','rb'))
    # data_feature = bdpy.BData(image_feature)
    # Add any additional processing to data here

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for sbj, roi, feat in product(subjects, rois, features):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])
        print('Feature:    %s' % feat)

        # Distributed computation
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_file = os.path.join(results_dir, analysis_id + '.pkl')

        # if os.path.exists(results_file):
        #     print('%s is already done. Skipped.' % analysis_id)
        #     continue

        # Prepare data
        print('Preparing data')
        dat = data_all[sbj]
        # print(dat.metadata.key)
        # print(dat.select('category_index'))

        x = dat.select(rois[roi])           # Brain data
        datatype = dat.select('DataType')   # Data type
        labels = dat.select('stimulus_id')  # Image labels in brain data

        y = image[feat]
        y_label = np.expand_dims(image['label'],1).astype('float')
        # y does not contain all data; select x datasets
        idx = select_x(labels, y_label)

        x = x[idx]
        datatype = datatype[idx]
        labels = labels[idx]
        print(x.shape, y.shape)

        # Get training and test dataset
        i_train = (datatype == 1).flatten()    # Index for training5
        i_test_pt = (datatype == 2).flatten()  # Index for perception test
        i_test_im = (datatype == 3).flatten()  # Index for imagery test
        i_test = i_test_pt + i_test_im

        x_train = x[i_train, :]
        print(x_train.shape)
        x_test = x[i_test_pt, :]

        # normalize data
        norm_mean_x = np.mean(x_train, axis=0)
        norm_scale_x = np.std(x_train, axis=0, ddof=1)

        x_train = (x_train - norm_mean_x) / norm_scale_x
        x_test = (x_test - norm_mean_x) / norm_scale_x

        # y = data_feature.select(feat)  # Image features
        # print(x.shape,y.shape)
        # y_label = data_feature.select('ImageID')  # Image labels

        # For quick demo, reduce the number of units from 1000 to 100
        # y = y[:, :100]

        y_sorted = get_refdata(y, y_label, labels)  # Image features corresponding to brain data

        y_train = y_sorted[i_train, :]
        y_test = y_sorted[i_test_pt, :]
        norm_mean_y = np.mean(y_train, axis=0)
        norm_scale_y = np.std(y_train, axis=0, ddof=1)
        norm_scale_y[norm_scale_y==0] = np.ones(sum(norm_scale_y==0))

        y_train = (y_train - norm_mean_y) / norm_scale_y
        y_test = (y_test - norm_mean_y) / norm_scale_y

        # compute the rdm metric
        brain_rdm = rdm(x[i_train,:],labels[i_train, :].flatten())  # brain for the current roi
        plot_rdm(brain_rdm,'brain roi '+roi)
        feature_rdm = rdm(y_sorted[i_train,:],labels[i_train, :].flatten())
        plot_rdm(feature_rdm, 'feature layer ' + feat)
        rsa = kendall_tau(feature_rdm.flatten(),brain_rdm.flatten())
        print(rsa)

        # pred_y = true_y = y_test

        # Feature prediction
        # pred_y, true_y = feature_prediction(x_train, y_train,
        #                                     x_test, y_test,
        #                                     n_voxel=num_voxel[roi],
        #                                     n_iter=n_iter)
        # pred_y = pred_y * norm_scale_y + norm_mean_y

        # Brain prediction
        pred_y, true_y = feature_prediction(y_train, x_train,
                                            y_test, x_test,
                                            n_voxel=500,
                                            n_iter=n_iter)
        with open('tune_result.txt', 'a') as convert_file:
            convert_file.write(sbj+'\t'+roi+'\t'+feat+'\n')

        pred_y = pred_y * norm_scale_x + norm_mean_x
        true_y = true_y * norm_scale_x + norm_mean_x

        # Get averaged predicted feature
        test_label = labels[i_test_pt, :].flatten()

        # combine predictions for same images
        pred_y_av, true_y_av, test_label_set \
            = get_averaged_feature(pred_y, true_y, test_label)

        plt.figure()
        plt.plot(np.arange(pred_y_av.shape[0]),pred_y_av[:,1],label='prediction')
        plt.plot(np.arange(pred_y_av.shape[0]), true_y_av[:,1], label='true')
        plt.legend()
        plt.savefig('brain_pred/'+roi+'_'+feat)

        from sklearn.metrics import r2_score
        from sklearn.metrics import explained_variance_score
        r2 = explained_variance_score(true_y, pred_y)
        corr = pred_corr(pred_y_av,true_y_av)

        # Prepare result dataframe
        results = pd.DataFrame({'subject' : [sbj],
                                'roi' : [roi],
                                'feature' : [feat],
                                'test_type' : ['perception'],
                                'true_feature': [true_y],
                                'predicted_feature': [pred_y],
                                'test_label' : [test_label],
                                'test_label_set' : [test_label_set],
                                'true_feature_averaged' : [true_y_av],
                                'predicted_feature_averaged' : [pred_y_av],
                                'rsa_rank': [rsa],
                                'R-square':[r2],
                                'Corr_brain':[corr]})

        # Save results
        makedir_ifnot(os.path.dirname(results_file))
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        # # Prepare result dataframe
        # with open(results_file, 'rb') as f:
        #     results = pickle.load(f)
        #
        # results['rsa_rank'] = [rsa]
        #
        # # Save results
        # makedir_ifnot(os.path.dirname(results_file))
        # with open(results_file, 'wb') as f:
        #     pickle.dump(results, f)

        print('Saved %s' % results_file)


# Functions ############################################################

def feature_prediction(x_train, y_train, x_test, y_test, n_voxel=500, n_iter=200):
    '''Run feature prediction
    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test
    n_voxel : int
        The number of voxels
    n_iter : int
        The number of iterations
    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    ture_label : array_like [shape = (n_sample, n_unit)]
        True features in test data
    '''
    n_unit = y_train.shape[1]

    # Feature prediction for each unit
    print('Running feature prediction')

    y_true_list = []
    y_pred_list = []

    cv_score={'s25':0,'s50':0,'s80':0,'s100':0}
    sparse_score=0

    for i in range(n_unit):

        print('Unit %03d' % (i + 1))
        start_time = time()

        # Get unit features
        y_train_unit = y_train[:, i]
        y_test_unit =  y_test[:, i]

        # # Voxel selection
        # corr = corrcoef(y_train_unit, x_train, var='col')
        #
        # x_train_unit, voxel_index = select_top(x_train, np.abs(corr), n_voxel, axis=1, verbose=False)
        # x_test_unit = x_test[:, voxel_index]

        # # cross validation: check the importance of predictors
        # numbers = [25,50,80,100]
        # for k in numbers:
        #     from sklearn.feature_selection import SelectKBest, f_classif
        #     select = SelectKBest(f_classif, k=k)
        #     x_train_unit = select.fit_transform(x_train, y_train_unit)
        #     print(k)
        #     idx = select.get_support()
        #
        #     # Add bias terms
        #     x_train_unit = add_bias(x_train_unit, axis=1)
        #
        #     from sklearn.model_selection import cross_val_score
        #     from sklearn.linear_model import Ridge, Lasso,ElasticNet
        #     model = SparseLinearRegression(n_iter=n_iter, prune_mode=1)
        #     cv_score['s'+str(k)]+=sum(cross_val_score(model, x_train_unit, y_train_unit, cv=5,
        #                                                 scoring='neg_mean_squared_error'))
            # cv_score_1['l'+str(k)] += sum(cross_val_score(lasso, x_train_unit, y_train_unit, cv=5,
            #                                               scoring='neg_mean_squared_error'))
            #
            # ridge = Ridge(alpha=alpha[1])
            # lasso = Lasso(alpha=alpha[1])
            # cv_score_2['r' + str(k)] += sum(cross_val_score(ridge, x_train_unit, y_train_unit, cv=5,
            #                                                 scoring='neg_mean_squared_error'))
            # cv_score_2['l' + str(k)] += sum(cross_val_score(lasso, x_train_unit, y_train_unit, cv=5,
            #                                                 scoring='neg_mean_squared_error'))
            #
            # ridge = Ridge(alpha=alpha[2])
            # lasso = Lasso(alpha=alpha[2])
            # cv_score_3['r' + str(k)] += sum(cross_val_score(ridge, x_train_unit, y_train_unit, cv=5,
            #                                                 scoring='neg_mean_squared_error'))
            # cv_score_3['l' + str(k)] += sum(cross_val_score(lasso, x_train_unit, y_train_unit, cv=5,
            #                                                 scoring ='neg_mean_squared_error'))

        # feature selections
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        select = SelectKBest(f_classif, k=25)
        x_train_unit = select.fit_transform(x_train, y_train_unit)
        idx = select.get_support()
        x_test_unit = x_test[:, idx]

        # Add bias terms
        x_train_unit = add_bias(x_train_unit, axis=1)
        x_test_unit = add_bias(x_test_unit, axis=1)
        # Setup regression
        # For quick demo, use linaer regression
        from sklearn.linear_model import Ridge
        # model = Ridge(alpha=1.0)
        model = SparseLinearRegression(n_iter=n_iter, prune_mode=1)

        # Training and test
        try:
            model.fit(x_train_unit, y_train_unit)  # Training
            y_pred = model.predict(x_test_unit)# Test
        except:
            # When SLiR failed, returns zero-filled array as predicted features
            y_pred = np.zeros(y_test_unit.shape)
        from sklearn.metrics import mean_squared_error
        sparse_score+=mean_squared_error(y_test_unit, y_pred)

        y_true_list.append(y_test_unit)
        y_pred_list.append(y_pred)

        print('Time: %.3f sec' % (time() - start_time))

    # Create numpy arrays for return values
    y_predicted = np.vstack(y_pred_list).T
    y_true = np.vstack(y_true_list).T

    for key in cv_score.keys():
        cv_score[key]/=n_unit
    sparse_score/=n_unit

    with open('tune_result.txt', 'a') as convert_file:
        convert_file.write(json.dumps(cv_score))
        convert_file.write('\n')
        # convert_file.write(json.dumps(cv_score_2))
        # convert_file.write('\n')
        # convert_file.write(json.dumps(cv_score_3))
        # convert_file.write('\n')
        convert_file.write('sparse_model:'+str(sparse_score))
        convert_file.write('\n')

    return y_predicted, y_true


def get_averaged_feature(pred_y, true_y, labels):
    '''Return category-averaged features'''

    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()