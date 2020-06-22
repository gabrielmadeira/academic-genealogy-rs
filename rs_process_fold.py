import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse as sp
# from sklearn.neighbors.nearest_centroid import NearestCentroid
import time

import sys

import warnings

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.sparsefuncs import csc_median_axis_0
from sklearn.utils.multiclass import check_classification_targets

n_fold = int(sys.argv[1])
check_point_predicts_len = 100

class NearestCentroid(BaseEstimator, ClassifierMixin):

    def __init__(self, metric='euclidean', shrink_threshold=None):
        self.metric = metric
        self.shrink_threshold = shrink_threshold

    def fit(self, X, y):

        if self.metric == 'precomputed':
            raise ValueError("Precomputed is not supported.")
        # If X is sparse and the metric is "manhattan", store it in a csc
        # format is easier to calculate the median.
        if self.metric == 'manhattan':
            X, y = check_X_y(X, y, ['csc'])
        else:
            X, y = check_X_y(X, y, ['csr', 'csc'])
        is_X_sparse = sp.issparse(X)
        if is_X_sparse and self.shrink_threshold:
            raise ValueError("threshold shrinking not supported"
                             " for sparse input")
        check_classification_targets(y)

        n_samples, n_features = X.shape
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError('The number of classes has to be greater than'
                             ' one; got %d class' % (n_classes))

        # Mask mapping each class to its members.
        self.centroids_ = sp.lil_matrix((n_classes, n_features), dtype=np.float64)
        # Number of clusters in each class.
        nk = np.zeros(n_classes)

        for cur_class in range(n_classes):
            center_mask = y_ind == cur_class
            nk[cur_class] = np.sum(center_mask)
            if is_X_sparse:
                center_mask = np.where(center_mask)[0]

            # XXX: Update other averaging methods according to the metrics.
            if self.metric == "manhattan":
                # NumPy does not calculate median of sparse matrices.
                if not is_X_sparse:
                    self.centroids_[cur_class] = np.median(X[center_mask], axis=0)
                else:
                    self.centroids_[cur_class] = csc_median_axis_0(X[center_mask])
            else:
                if self.metric != 'euclidean':
                    warnings.warn("Averaging for metrics other than "
                                  "euclidean and manhattan not supported. "
                                  "The average is set to be the mean."
                                  )
                self.centroids_[cur_class] = X[center_mask].mean(axis=0)

        if self.shrink_threshold:
            dataset_centroid_ = np.mean(X, axis=0)

            # m parameter for determining deviation
            m = np.sqrt((1. / nk) - (1. / n_samples))
            # Calculate deviation using the standard deviation of centroids.
            variance = (X - self.centroids_[y_ind]) ** 2
            variance = variance.sum(axis=0)
            s = np.sqrt(variance / (n_samples - n_classes))
            s += np.median(s)  # To deter outliers from affecting the results.
            mm = m.reshape(len(m), 1)  # Reshape to allow broadcasting.
            ms = mm * s
            deviation = ((self.centroids_ - dataset_centroid_) / ms)
            # Soft thresholding: if the deviation crosses 0 during shrinking,
            # it becomes zero.
            signs = np.sign(deviation)
            deviation = (np.abs(deviation) - self.shrink_threshold)
            np.clip(deviation, 0, None, out=deviation)
            deviation *= signs
            # Now adjust the centroids using the deviation
            msd = ms * deviation
            self.centroids_ = dataset_centroid_[np.newaxis, :] + msd
        return self

    def predict(self, X):

        check_is_fitted(self, 'centroids_')

        X = check_array(X, accept_sparse='csr')
        
        return np.argsort(pairwise_distances(X, self.centroids_, metric=self.metric))[:10]


def score(clf, X, y):
    X_len = X.shape[0]
    aux_count = 0

    try:
        checkpoint = pickle.load(open('fold'+str(n_fold)+'_checkpoint', 'rb'))

    except (OSError, IOError) as e:
        checkpoint = [0, [None] * X_len, 0, 0, 0, 0, int(((X_len)/(check_point_predicts_len))+1), '', [0]*101]

	# 0: checkpoint number, 1: y predict, 2: sum_score, 3: final score, 4: checkpoint time, 5: total time, 6: total checkpoints, 7: checkpoints logs, 8: count_pos]
	
    current_start = (checkpoint[0]*check_point_predicts_len)

    aux_time = time.time()

    for i in range(current_start, X_len):
        out = clf.predict(X[i])
        count1 = 0
        count2 = 0
        sumAvg = 0
        #checkpoint[1][i] = out[0][:100]
        count_pos = False
        
        for i2 in out[0]:
            count1 += 1
            if clf.classes_[i2] in y[i]:
                count2 += 1
                sumAvg += count2/count1
                
                if not count_pos:
                    if count1 <= 100:
                        checkpoint[8][count1] += 1
                    count_pos = True
                    
                if count2 >= len(y[i]):
                    break
            if count1 > 100:
                break
        if sumAvg == 0:
            meanAvg = 0
        else:
            meanAvg = sumAvg/count2
        
        checkpoint[2] += meanAvg
            
            
        if aux_count >= (check_point_predicts_len-1):

            checkpoint[3] = checkpoint[2]/(i+1) 
            
            checkpoint[4] = time.time() - aux_time

            checkpoint[5] += checkpoint[4]

            #checkpoint[7] += str(checkpoint[0])+','+str(i-check_point_predicts_len+1)+'-'+str(i)+','+str(checkpoint[3])+','+str(checkpoint[4])+','+str(checkpoint[5])+';'

            score_out = open('console_out','a+')
            score_out.write('\nCheckpoint:'+str(checkpoint[0]))
            score_out.write('\nRange:'+str(i-check_point_predicts_len+1)+'-'+str(i))
            score_out.write('\n'+str(i)+' of '+ str(X_len-1))
            score_out.write('\nTime:'+str(checkpoint[4]))
            score_out.write('\nTotal time:'+str(checkpoint[5]))
            score_out.write('\nCurrent score:'+str(checkpoint[3]))
            score_out.write('\n -')
            score_out.close()

            with open('fold'+str(n_fold)+'_checkpoint', 'wb') as f:
                pickle.dump(checkpoint, f)

            aux_count = 0
            checkpoint[0] += 1
            aux_time = time.time()
        
        elif i == (X_len-1):
            checkpoint[3] = checkpoint[2]/(i+1) 
            
            checkpoint[4] = time.time() - aux_time

            checkpoint[5] += checkpoint[4]

            #checkpoint[7] += str(checkpoint[0])+',?-'+str(i)+','+str(checkpoint[3])+','+str(checkpoint[4])+','+str(checkpoint[5])+';'

            score_out = open('console_out','a+')
            score_out.write('\nCheckpoint:'+str(checkpoint[0]))
            score_out.write('\nRange: ?-'+str(i))
            score_out.write('\n'+str(i)+' of '+ str(X_len-1))
            score_out.write('\nTime:'+str(checkpoint[4]))
            score_out.write('\nTotal time:'+str(checkpoint[5]))
            score_out.write('\nCurrent score:'+str(checkpoint[3]))
            score_out.close()

            with open('fold'+str(n_fold)+'_checkpoint', 'wb') as f:
                pickle.dump(checkpoint, f)

        else:
            aux_count += 1
    
    return checkpoint


train_y_doc_vect_representation = pickle.load(open('fold'+str(n_fold)+'_train_y_doc_vect_representation', 'rb'))
train_X_doc_vect_representation = pickle.load(open('fold'+str(n_fold)+'_train_X_doc_vect_representation', 'rb'))

test_y_doc_vect_representation = pickle.load(open('fold'+str(n_fold)+'_test_y_doc_vect_representation', 'rb'))
test_X_doc_vect_representation = pickle.load(open('fold'+str(n_fold)+'_test_X_doc_vect_representation', 'rb'))

try:
	clf = pickle.load(open('fold'+str(n_fold)+'_clf', 'rb'))

	clf_out = open('console_out','a+')
	clf_out.write('\nLoading classifier...\n')
	clf_out.close()
except (OSError, IOError) as e:
	clf_out = open('console_out','a+')
	clf_out.write('\nFit classifier...\n')
	clf_out.close()

	clf = NearestCentroid()
	clf.fit(train_X_doc_vect_representation, train_y_doc_vect_representation)

	with open('fold'+str(n_fold)+'_clf', 'wb') as f:
	    pickle.dump(clf, f)

result = score(clf, test_X_doc_vect_representation, test_y_doc_vect_representation[1])

with open('fold'+str(n_fold)+'_result', 'wb') as f:
    pickle.dump(result, f)



