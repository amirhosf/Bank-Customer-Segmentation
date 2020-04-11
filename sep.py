from sklearn import tree
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import call
from presep import *
from sklearn import mixture
import random
from collections import Counter
import sklearn.metrics as metrics
import copy
from scipy import stats
from scipy.stats import skew, boxcox
import seaborn as sns
from statsmodels.tools import categorical
import sklearn as skl
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,average_precision_score
from sklearn.linear_model import LinearRegression
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn import linear_model
from IPython.core.interactiveshell import InteractiveShell
import sklearn.preprocessing as preprocessing
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc,recall_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
import warnings
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture as GMM
from itertools import cycle, islice
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from sklearn import cluster
import time
import hdbscan
#------------------------------------------------------------
df = pd.read_csv('se.csv')
#printing the head of the data
print(df.head())
#to check for any missing data points in case imputation is needed
df.isnull().values.any()
#obtain describtion and information of data

print(df.describe())

print(df.info())
# sns.heatmap(df.corr(),square=True)
# #plt.show()
def distributions(df):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
    sns.distplot(df["MonthlyNetIncomeAmount"], ax=ax1)
    ax1.text(0.4,2,"Skewness: {0:.2f}".format(stats.skew(df['MonthlyNetIncomeAmount'])))
    sns.distplot(df["totalassets"], ax=ax2)
    ax2.text(0.4,200000,"Skewness: {0:.2f}".format(stats.skew(df['totalassets'])))
    sns.distplot(df["totaldebt"], ax=ax3)
    ax3.text(0.4,200000,"Skewness: {0:.2f}".format(stats.skew(df['totaldebt'])))
    plt.tight_layout()
    plt.show()

def distributions_boxcox(df):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
    sns.distplot(df["MonthlyNetIncomeAmount_boxcox"], ax=ax1)
    ax1.text(4,0.5,"Skewness: {0:.2f}".format(stats.skew(df['MonthlyNetIncomeAmount_boxcox'])))
    sns.distplot(df["totalassets_boxcox"], ax=ax2)
    ax2.text(2,0.5,"Skewness: {0:.2f}".format(stats.skew(df['totalassets_boxcox'])))
    sns.distplot(df["totaldebt_boxcox"], ax=ax3)
    ax3.text(1,0.5,"Skewness: {0:.2f}".format(stats.skew(df['totaldebt_boxcox'])))
    plt.tight_layout()
    plt.show()

def scatters(data, h=None, pal=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
    sns.scatterplot(x="MonthlyNetIncomeAmount_boxcox",y="totalassets_boxcox", hue=h, palette=pal, data=data, ax=ax1)
    sns.scatterplot(x="MonthlyNetIncomeAmount_boxcox",y="totaldebt_boxcox", hue=h, palette=pal, data=data, ax=ax2)
    sns.scatterplot(x="totalassets_boxcox",y="totaldebt_boxcox", hue=h, palette=pal, data=data, ax=ax3)
    plt.tight_layout()
    plt.show()
def scatters_gmm(data, h=None, pal=None):
    sns.scatterplot(x="totalassets_boxcox",y="totaldebt_boxcox", hue=h, palette=pal, data=data)
    plt.show()
# fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
# sns.scatterplot(x="MonthlyNetIncomeAmount",y="InternalDebts_UsedCredit", hue='ChildBirthYears', palette=None, data=df, ax=ax1)
# sns.scatterplot(x="MonthlyNetIncomeAmount",y="InternalDebts_Overdrawn", hue='ChildBirthYears', palette=None, data=df, ax=ax2)
# sns.scatterplot(x="MonthlyNetIncomeAmount",y="InternalDebts_Blanco", hue='ChildBirthYears', palette=None, data=df, ax=ax3)
# plt.tight_layout()
# plt.show()


# fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
# sns.distplot(df["MonthlyNetIncomeAmount"], ax=ax1)
# sns.distplot(df["InternalDebts_Overdrawn"], ax=ax2)
# sns.distplot(df["ChildBirthYears"], ax=ax3)
# plt.tight_layout()
# plt.show()

df['totaldebt'] = df.InternalDebts_Overdrawn+df.InternalDebts_UsedCredit+df.InternalDebts_Mortgage\
+df.InternalDebts_UnpaidEquities+df.ExternalDebts_Other+df.ExternalDebts_CreditCard+df.ExternalDebts_Mortgage

df['totalassets'] = df.InternalAssets_PensionIns+df.InternalAssets_Funds+df.InternalAssets_Accounts+df.InternalAssets_UnpaidEquities+df.ExternalAssets_PensionIns\
+df.ExternalAssets_Funds+df.ExternalAssets_Accounts+df.ExternalAssets_Other

df = df.drop (['InternalDebts_Overdrawn', 'InternalDebts_UsedCredit', 'InternalDebts_Blanco', 'InternalDebts_Mortgage','InternalDebts_UnpaidEquities','InternalDebts_CreditCard','ExternalDebts_Other', 'ExternalDebts_CreditCard',\
'ExternalDebts_Blanco', 'ExternalDebts_Mortgage','InternalAssets_PensionIns','InternalAssets_Funds', 'InternalAssets_Accounts','InternalAssets_UnpaidEquities','ExternalAssets_PensionIns','ExternalAssets_Funds',\
'ExternalAssets_Accounts','ExternalAssets_Other','CustomerAge','Advices_ordered' ,'MonthlyGrossIncomeAmount'],axis = 1)


X_noncatagorial = df.drop(['HousingTypeId', 'ChildBirthYears'],axis = 1)
X_gmm = df.drop(['HousingTypeId', 'ChildBirthYears','MonthlyNetIncomeAmount'],axis =1)
#print (X_noncatagorial)
# scaler = StandardScaler() 
# scalar2 = StandardScaler()
# X_scaled = scaler.fit_transform(X_noncatagorial)
# print ('scale this:',X_scaled)
# X_gmm_scaled = scalar2.fit_transform(X_gmm)
print(X_noncatagorial.describe())
X_noncatagorial['MonthlyNetIncomeAmount_boxcox'] = preprocessing.scale(stats.boxcox(X_noncatagorial['MonthlyNetIncomeAmount']+1)[0])
X_noncatagorial['totalassets_boxcox'] = preprocessing.scale(stats.boxcox(X_noncatagorial['totalassets']+1)[0])
X_noncatagorial['totaldebt_boxcox'] = preprocessing.scale(stats.boxcox(X_noncatagorial['totaldebt']+1)[0])
#finding the distribution before and after normalization
distributions(X_noncatagorial)
distributions_boxcox(X_noncatagorial)

print('SKewness MonthlyNetIncomeAmount:',(stats.skew(X_noncatagorial['MonthlyNetIncomeAmount'])))
print('SKewness totalassets:',(stats.skew(X_noncatagorial['totalassets'])))
print('SKewness totaldebt:',(stats.skew(X_noncatagorial['totaldebt'])))



X_noncatagorial = X_noncatagorial.drop (['MonthlyNetIncomeAmount' , 'totalassets', 'totaldebt'],axis = 1)


X_gmm['totalassets_boxcox'] = preprocessing.scale(stats.boxcox(X_gmm['totalassets']+1)[0])
X_gmm['totaldebt_boxcox'] = preprocessing.scale(stats.boxcox(X_gmm['totaldebt']+1)[0])

X_gmm = X_gmm.drop(['totalassets','totaldebt'],axis = 1)
#finding the best cluster amount for KMeans
clusters_range = [2,3,4,5,6,7,8,9,10,11,12,13,14]
inertias =[]

for c in clusters_range:
    kmeans = KMeans(n_clusters=c, random_state=0).fit(X_noncatagorial)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(clusters_range,inertias, marker='o')

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X['MonthlyNetIncomeAmount_boxcox'], X['totaldebt_boxcox'], c=labels, s=10, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, lw=3, alpha=0.25, zorder=0.5))

kmeans = KMeans(4, random_state=10)
labels = kmeans.fit(X_noncatagorial).predict(X_noncatagorial)
label2 = pd.DataFrame(kmeans.labels_)
cluster_data = X_noncatagorial.copy()
clustered_data = cluster_data.assign(Cluster=label2)
scatters(clustered_data, 'Cluster')
plot_kmeans(kmeans, X_noncatagorial)
plt.show()
grouped_km = clustered_data.groupby(['Cluster']).mean().round(1)
print('Kmean:',grouped_km)
#plt.scatter(X_noncatagorial['MonthlyNetIncomeAmount'], X_noncatagorial['totaldebt'], c=labels, s=10, cmap='viridis')
#plt.show()




#the gmm model
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        print ('cov', covariance)
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X['totalassets_boxcox'], X['totaldebt_boxcox'], c=labels, s=40, cmap='viridis', zorder=2)
        scatter = ax.scatter(X['totalassets_boxcox'], X['totaldebt_boxcox'], c=labels, s=40, cmap='viridis', zorder=2)
        ax.legend(*scatter.legend_elements(),
                    loc="top right", title="Clusters")
    else:
        ax.scatter(X['totalassets_boxcox'], X['totaldebt_boxcox'], s=40, zorder=2)
        scatter = ax.scatter(X['totalassets_boxcox'], X['totaldebt_boxcox'], c=labels, s=40, cmap='viridis', zorder=2)
        ax.legend(*scatter.legend_elements(),
                    loc="top right", title="Clusters")
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    print('gmm:',gmm.covariances_)
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    return labels       		

gmm = GMM(n_components=4, random_state=42)
labels = plot_gmm(gmm, X_gmm)
plt.show()
plt.scatter(X_gmm['totalassets_boxcox'], X_gmm['totaldebt_boxcox'], c=labels, s=10, cmap='viridis')
scatter=plt.scatter(X_gmm['totalassets_boxcox'], X_gmm['totaldebt_boxcox'], c=labels, s=10, cmap='viridis')
plt.legend(*scatter.legend_elements(),
                    loc="top right", title="Clusters")
plt.show()

#gmm = GMM(n_components=4, covariance_type='full', random_state=42)
#plot_gmm(gmm, X_gmm)
#plt.show()


x = np.linspace(-8., 9.)
y = np.linspace(-8., 9.)
Xs, Y = np.meshgrid(x, y)
XX = np.array([Xs.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(Xs.shape)

CS = plt.contour(Xs, Y, Z, norm=LogNorm(vmin=1.0, vmax=10.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_gmm['totalassets_boxcox'], X_gmm['totaldebt_boxcox'], .8)

plt.title('Gaussian Mixture prediction')
plt.axis('tight')
plt.show()

label2 = pd.DataFrame(labels)
cluster_data = X_noncatagorial.copy()
clustered_data = cluster_data.assign(Cluster=label2)
grouped_km = clustered_data.groupby(['Cluster']).mean().round(1)
print('GMM:',grouped_km)

now the agglormative clustering
ward = cluster.AgglomerativeClustering(
        n_clusters=4, linkage='ward')
complete = cluster.AgglomerativeClustering(
        n_clusters=4, linkage='complete')
average = cluster.AgglomerativeClustering(
        n_clusters=4, linkage='average')
single = cluster.AgglomerativeClustering(
        n_clusters=4, linkage='single')


clustering_algorithms = (
        ('Single Linkage', single),
        ('Average Linkage', average),
        ('Complete Linkage', complete),
        ('Ward Linkage', ward),
    )
plot_num = 1
for name, algorithm in clustering_algorithms:
	t0 = time.time()

	# catch warnings related to kneighbors_graph
	with warnings.catch_warnings():
		warnings.filterwarnings(
	"ignore",
	            message="the number of connected components of the " +
	            "connectivity matrix is [0-9]{1,2}" +
	            " > 1. Completing it to avoid stopping the tree early.",
	            category=UserWarning)
		algorithm.fit(X_gmm)
		t1 = time.time()
		if hasattr(algorithm, 'labels_'):
			y_pred = algorithm.labels_.astype(np.int)
			print('LABELED ALGOS')
		else:
			y_pred = algorithm.predict(X_gmm)
			print('unlabeled ALGOS')
		colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
	                                         '#f781bf', '#a65628', '#984ea3',
	                                         '#999999', '#e41a1c', '#dede00']),
	                                  int(max(y_pred) + 1))))
		plt.scatter(X_gmm['totalassets_boxcox'], X_gmm['totaldebt_boxcox'], s=10, color=colors[y_pred])
		plt.xlim(-9.5, 9.5)
		plt.ylim(-9.5, 9.5)
		plt.xticks(())
		plt.yticks(())
		plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
	             transform=plt.gca().transAxes, size=15,
	             horizontalalignment='right')
		plot_num += 1

		plt.show()


'''
New Mehtod of clustering is called HDBSCAN which is useful for high dimension data:
https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
1-Transform the space according to the density/sparsity.
2-Build the minimum spanning tree of the distance weighted graph.
3-Construct a cluster hierarchy of connected components.
4-Condense the cluster hierarchy based on minimum cluster size.
5-Extract the stable clusters from the condensed tree.


'''
clusterer = hdbscan.HDBSCAN(min_cluster_size=60, gen_min_span_tree=True,metric = 'euclidean')
clusterer.fit(X_noncatagorial)
# HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#     gen_min_span_tree=True, leaf_size=40, memory=Memory(cachedir=None),
#     metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
print (clusterer.labels_.max())
scores = clusterer.probabilities_
scorelist = []
# for i in range (5,70):
# 	clusterer = hdbscan.HDBSCAN(min_cluster_size=i, gen_min_span_tree=True,metric = 'euclidean')
# 	clusterer.fit(X_noncatagorial)
# 	tempscore = clusterer.probabilities_
# 	newtempscore = [tempscore < 0.05]
# 	sumscore = np.mean(newtempscore)
# 	scorelist.append(sumscore)


# i = np.arange(5,70)
# plt.plot (i, scorelist, '-*')
# plt.ylabel ('sumscore')
# plt.xlabel('min cluster size')
# plt.title('Kullback-Leibler divergence plot')
# plt.show()

#####
label2 = pd.DataFrame(clusterer.labels_)
cluster_data = X_noncatagorial.copy()
clustered_data = cluster_data.assign(Cluster=label2)
scatters(clustered_data, 'Cluster')
plt.show()
grouped_km = clustered_data.groupby(['Cluster']).mean().round(1)
print('HDBSCAN:',grouped_km)
######
labels = clusterer.labels_
plt.scatter(X_noncatagorial['totalassets_boxcox'], X_noncatagorial['totaldebt_boxcox'], c=labels, s=10, cmap='viridis')
scatter=plt.scatter(X_noncatagorial['totalassets_boxcox'], X_noncatagorial['totaldebt_boxcox'], c=labels, s=10, cmap='viridis')
plt.legend(*scatter.legend_elements(),
                    loc="upper right", title="Clusters")
plt.show()
#After plotting the Kullback-Leibler divergence plot to find the best probability of the score
#in clustering min cluster size of 60 was chosen






