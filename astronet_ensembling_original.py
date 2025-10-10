""" Quick code for ensembling individual model results from Astronet """

###################
### IMPORT PACKAGES 

import glob as glob
import pandas as pd
import numpy as np
import argparse
import pdb
import os

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

### plotting packages
import matplotlib as mpl
import matplotlib.pyplot as plt


###################
### PARSE ARGUMENTS

### an example input to command line: python astronet_ensembling.py '../astronet_ensembling' 250 'plots'
parser = argparse.ArgumentParser()
parser.add_argument("d_path", help="path to data")
parser.add_argument("i_val", help="number of iterations/epochs of trained model to use", type=int)
parser.add_argument("p_path", help="path for output plot")
args = parser.parse_args()


##########################
### LOAD INDIVIDUAL MODELS

### grab all prediction-groundtruth files for ensembling [assumes output format from astronet.py]
pt_files = np.sort(glob.glob(os.path.join(args.d_path, '*i' + str(args.i_val) + '-pt.csv')))

### put predictions from all models in a single dataframe
for i, val in enumerate(pt_files):
    r = pd.read_csv(val)
    if i==0:
        df = pd.DataFrame(np.round(r.pred.values, 3), columns=[val.split('-')[0][-3:]])
    else:
        df.insert(loc=len(df.columns), column=val.split('-')[0][-3:], value=np.round(r.pred.values, 3))
df['pred'] = df.mean(axis=1)

### insert groud truth values into data frame
df.insert(loc=0, column='groundtruth', value=r['gt'].values)


######################################################
### CALCULATE AVERAGE PRECISION (AP) AND ACCURACY (AC)

ap, ac = [], []
for i, val in enumerate(df.columns[1:]):
        
    ### calculate average precision (ap)
    ap.append(np.round(average_precision_score(df.groundtruth.values, df[val].values, average=None), 3))

    ### calculate accuracy (ac) using threshold = 0.5
    arr = np.copy(df[val].values)
    arr[arr >= 0.5] = 1.0
    arr[arr < 0.5] = 0.0
    ac.append(np.round(np.sum(arr == df.groundtruth.values) / len(df), 3))

### print out results of ensembling
print("\nMedian AP of runs: " + str(round(100 * np.median(ap[0:-1]), 3)) + '%')
print("AP of ensemble: " + str(round(100 * ap[-1], 3)) + '%')
print("Difference: " + str(round(100 * (ap[-1] - np.median(ap[0:-1])), 3)) + '%')
print("\nMedian ACC of runs: " + str(round(100 * np.median(ac[0:-1]), 3)) + '%')
print("ACC of ensemble: " + str(round(100 * ac[-1], 3)) + '%')
print("Difference: " + str(round(100 * (ac[-1] - np.median(ac[0:-1])), 3)) + '%')


##############################################################
### CALCULATE & PLOT PRECISION-RECALL CURVE OF ENSEMBLED MODEL

### calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(df.groundtruth.values, df.pred.values)

### setup plot
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)
mpl.rc('xtick.major', size=7, pad=7, width=2)
mpl.rc('ytick.major', size=7, pad=7, width=2)
mpl.rc('axes', linewidth=2)
mpl.rc('lines', markersize=5)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
xmin, xmax, ymin, ymax = 0.4, 1.01, 0.4, 1.01
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_xlabel('Recall', fontsize=17)
ax.set_ylabel('Precision', fontsize=17)

### precision-recall curve
ax.step(recall, precision, where="post", linewidth=2, color='black', zorder=1)

### plot different threshold values
print('\nThreshold   Precision    Recall')
thresh = [0.5, 0.6, 0.7, 0.8, 0.9]
for i, val in enumerate(thresh):
    ind = np.argmin(np.abs(thresholds - val))
    ax.scatter(recall[ind], precision[ind], facecolor='orange', edgecolor='orange', marker='o', alpha=0.7, s=120, lw=1, zorder=5)
    ax.text(recall[ind] + 0.007, precision[ind] + 0.007, str(val), fontsize=12)
    print('    ', val, '    ', round(recall[ind], 3), '    ', round(precision[ind], 3))

### save plot to output directory
plt.savefig(os.path.join(args.p_path, 'astronet_ensembling_output.pdf'), bbox_inches='tight', dpi=200, rastersized=True, alpha=True)
