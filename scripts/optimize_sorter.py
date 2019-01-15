import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../app')
from global_constants import REPO_HOME
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import stats

df = pd.read_csv(REPO_HOME+'data/portmanteau_ranking_labeled_data.csv')

# Use the RankSVM code here: http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
# Don't bother with weighting

def is_comparable(i, j, df=df):
    if df.input_id[i] == df.input_id[j] and df.label[i] != df.label[j]:
        return True
    else:
        return False

def get_diff(i, j, df=df):
    d = {'g': 0, 'm': 1, 'b': 2}
    return d[df.label[i]] - d[df.label[j]]

def get_sign(i, j, df=df):
    d = {'g': 0, 'm': 1, 'b': 2}
    return np.sign(d[df.label[i]] - d[df.label[j]])

def get_weight(i, j, df=df):
    d = {'g': 0, 'm': 1, 'b': 2}
    return np.abs(d[df.label[i]] - d[df.label[j]])

# form all pairwise combinations
comb = combinations(range(df.shape[0]), 2)
k = 0
Xp, yp, diff, weight = [], [], [], []
for (i, j) in comb:
    if not is_comparable(i, j):
        continue
    Xp.append(np.array([df.phonetic_dist[i] - df.phonetic_dist[j], np.log(df.phonetic_prob[i]) - np.log(df.phonetic_prob[j])]))
    diff.append(get_diff(i, j))
    yp.append(get_sign(i, j))
    weight.append(get_weight(i, j))
    # enforce balanced classes
    if yp[-1] != (-1) ** k:
        yp[-1] *= -1
        Xp[-1] *= -1
        diff[-1] *= -1
    k += 1
Xp, yp, diff, weight = map(np.asanyarray, (Xp, yp, diff, weight))

# Visualize reframed problem:
# plt.scatter(Xp[:, 0], Xp[:, 1], c=diff, s=60, marker='o', cmap=plt.cm.Blues)
# plt.show()

# Fit SVM, and
clf = svm.SVC(kernel='linear', C=.1) # no grid search for now
clf.fit(Xp, yp)
# Compute L2-normalized coef's, and output them
coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)
print coef

# Compare rank-order correlation from before (~0.2) vs after (~0.7)
df['phonetic_prob_log']= np.log(df.phonetic_prob)
d = {'g': 0, 'm': 1, 'b': 2}
df['label_num'] = df.label.map(lambda x: d[x])
tau, _ = stats.kendalltau(np.dot(df[['phonetic_dist','phonetic_prob_log']], coef), df['label_num'])
print tau
tau, _ = stats.kendalltau(df.phonetic_dist + df.phonetic_prob, df['label_num']) # equivalent to lexical ordering b/c of differing order-of-magnitude
print tau

# Looking at the Portmanteau data, impose cutoff of -7, anything greater than that is junk

# now do the same for rhymes...
