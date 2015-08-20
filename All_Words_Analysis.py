__author__ = 'Nnamdi'

from __future__ import division
import sys
import pandas as pd
import numpy as np
import operator
import string
import os
import math
from scipy import stats
import subprocess

tmpfilepath='/analytic_store/nnamdi/'
outputfilepath='/analytic_store/nnamdi/'
filepath='/analytic_store/nnamdi/'

from sklearn.feature_selection import SelectFwe, SelectPercentile, SelectFpr,SelectKBest,VarianceThreshold, GenericUnivariateSelect, chi2, f_classif
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import roc_auc_score, r2_score, average_precision_score
import random
import zipfile
from scipy.sparse import hstack, vstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import ParameterGrid
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.cross_validation import StratifiedKFold


def load_object(filepath,filename):
    with open(filepath+filename, 'rb') as input:
        dictByUsers=pickle.load(input)
    return dictByUsers


def sigle_fit(clf, X_train, Y_train, X_test, Y_test, accuMeasure):
    clf.fit(X_train, Y_train)
    try:
        pred = clf.predict_proba(X_test)[:,1]
    except:
        pred =clf.predict(X_test)
    try:
        aucScore=accuMeasure(Y_test, pred)
    except:
        pred = clf.predict_proba(X_test)
        y_test = label_binarize(Y_test, classes=[0, 1, 2, 3])
        aucScore=accuMeasure(y_test, pred)
    return aucScore


def predP(clf, X_test):
    try:
        pred = clf.predict_proba(X_test)[:,1]
    except:
        pred =clf.predict(X_test)
    return pred


def stackGetTrainData(X_dev, X_test, Y_dev, clf, n_folds):
    skf = list(StratifiedKFold(Y_dev, n_folds))
    blend_train = np.zeros(X_dev.shape[0])
    blend_test_j = np.zeros((X_test.shape[0], len(skf)))
    for i, (train_index, cv_index) in enumerate(skf):
        Xdev_train = X_dev[train_index]
        Ydev_train = Y_dev[train_index]
        Xdev_cv = X_dev[cv_index]
        Ydev_cv = Y_dev[cv_index]
        clf.fit(Xdev_train, Ydev_train)
        blend_train[cv_index] = predP(clf,Xdev_cv)
        blend_test_j[:, i] = predP(clf, X_test)
    #Take the mean of the predictions of the cross validation set
    blend_test = blend_test_j.mean(1)
    return blend_train, blend_test


def stack_fit(clf, X_train_all, Y_train, X_test_all, Y_test, accuMeasure, n_folds):
    stackTrain=np.zeros((X_train_all[0].shape[0], len(X_train_all)))
    stackTest=np.zeros((X_test_all[0].shape[0], len(X_test_all)))
    for ii in xrange(len(X_train_all)):
        blend_train, blend_test = stackGetTrainData(X_train_all[ii], X_test_all[ii], Y_train, clf, n_folds)
        stackTrain[:,ii]=blend_train
        stackTest[:,ii]=blend_test
    bclf = LogisticRegression()
    bclf.fit(stackTrain, Y_train)
    Y_test_predict = predP(bclf,stackTest)
    aucScore = accuMeasure(Y_test, Y_test_predict)
    return aucScore


def crossvalidate(nrep, nfold, sparseArrayRowNorm, y_all, clf, accuMeasure, selection):
    nsample=sparseArrayRowNorm[0].shape[0]
    scaler = StandardScaler(with_mean=False)
    #scaler = MinMaxScaler()
    testsize=int(nsample/nfold)
    cvIdx=[1]*(nsample-testsize)+[2]*testsize
    random.seed(100)
    aucRes=[]
    for nn in range(nrep):
        #print nn
        random.shuffle(cvIdx)
        Y_train=y_all[np.where(np.array(cvIdx)==1)[0]]
        Y_test=y_all[np.where(np.array(cvIdx)==2)[0]]
        X_train_all=[]
        X_test_all=[]
        for ii in xrange(len(sparseArrayRowNorm)):
            varSelector = SelectKBest(f_classif, k=min(int(nsample*0.7), sparseArrayRowNorm[ii].shape[1]))
            X_train=sparseArrayRowNorm[ii][np.where(np.array(cvIdx)==1)[0],:]
            X_train =varSelector.fit_transform(X_train, Y_train)
            X_train_all=X_train_all+[X_train]
            X_test=sparseArrayRowNorm[ii][np.where(np.array(cvIdx)==2)[0],:]
            X_test= varSelector.transform(X_test)
            X_test_all=X_test_all+[X_test]
        X_train=hstack(X_train_all,format='csr')
        X_test=hstack(X_test_all,format='csr')
        del X_train_all
        del X_test_all
        aucRes.append(sigle_fit(clf, X_train, Y_train, X_test, Y_test, accuMeasure))
    print np.array(aucRes).mean()
    return np.array(aucRes).mean()


def processYX(y_all, sparseArrayRowNorm):
    if set(y_all)==set(range(2)):
      newSparseArray=sparseArrayRowNorm
      y_new=y_all
    if set(y_all)==set(range(3)):
      newSparseArray=[]
      for ii in xrange(len(sparseArrayRowNorm)):
         newSparseArray=newSparseArray+[sparseArrayRowNorm[ii][np.where(y_all!=2)]]
      y_new=y_all[y_all!=2]
    if set(y_all)==set(range(4)):
        newSparseArray=sparseArrayRowNorm
        y_new=np.zeros(y_all.shape, dtype='int8')
        y_new[y_all>=2]=1
    return y_new, newSparseArray


def matchDict(mat1,mat2, users1, users2):
   nrow=mat1.shape[0]
   ncol=mat2.shape[1]
   users1=list(users1)
   users2=list(users2)
   mat2=vstack([mat2, csr_matrix((1,ncol), dtype=np.int32)])
   nrowmat2=mat2.shape[0]
   matchIdx=[users2.index(x) if x in users2 else nrowmat2-1 for x in users1]
   tmp=mat2[matchIdx]
   return(tmp)


def combMats(mat1, mat2, keywords1, keywords2):
   mat1=mat1.tocsc()
   mat2=mat2.tocsc()
   n1=len(keywords1)
   n2=len(keywords2)
   k1=list(keywords1)
   k2=list(keywords2)
   indx1=[ii for ii in xrange(n1) if k1[ii] not in k2]
   indx2=[ii for ii in xrange(n2) if k2[ii] not in k1]
   indx12=[ii for ii in xrange(n1) if k1[ii] in k2]
   indx21=[k2.index(kk1[ii]) for ii in xrange(n1) if k1[ii] in kk2]
   matAll=hstack([mat1[:,indx1], mat2[:indx2], mat1[:,indx12]+mat2[:indx21]], format='csc')
   return matAll.tocsr()


def rmv2gram(dictByUsers2):
   p=re.compile(".*\s+.*")
   keepIdx=[ii for ii in xrange(len(dictByUsers2[2])) if p.match(dictByUsers2[2][ii]) is None]
   mattmp=dictByUsers2[0].tocsc()
   mattmp=mattmp[:,keepIdx]
   mat0=mattmp.tocsr()
   mattmp=dictByUsers2[1].tocsc()
   mattmp=mattmp[:,keepIdx]
   mat1=mattmp.tocsr()
   return mat0, mat1, np.array(dictByUsers2[2])[keepIdx]


def getDataRisk(filepath, description, remove2gram):
   dictByUsers = load_object(filepath, "smoking_1_analytic_data_mapreduce.pkl")
   transformer = TfidfTransformer(use_idf=False)
   if dictByUsers[0] is not None:
      varSelector1 = VarianceThreshold(threshold=0.001)
      if remove2gram:
         mattmp0, mattmp1, tmp = rmv2gram(dictByUsers)
         sparseArrayVarianceFilter1 = varSelector1.fit_transform(mattmp0)
         sparseArrayVarianceFilter4 = varSelector1.fit_transform(mattmp1)
      else:
         sparseArrayVarianceFilter1 = varSelector1.fit_transform(dictByUsers[0])
         sparseArrayVarianceFilter4 = varSelector1.fit_transform(dictByUsers[1])
      transformer = TfidfTransformer(use_idf=False)
      sparseArrayRowNorm1 = transformer.fit_transform(sparseArrayVarianceFilter1)
      sparseArrayRowNorm4 = transformer.fit_transform(sparseArrayVarianceFilter4)
      y_all=np.array(dictByUsers[4])
      sparseArrayRowNorm = [hstack([sparseArrayRowNorm1,sparseArrayRowNorm4],format='csr')]
   if description:
      dictByUsers2 = load_object(filepath+"description_data/", "smoking_1_analytic_data_mapreduce.pkl")
      if dictByUsers2[0] is not None:
         varSelector1 = VarianceThreshold(threshold=0.001)
         if remove2gram:
            mattmp0, mattmp1, tmp = rmv2gram(dictByUsers2)
            mat1=matchDict(dictByUsers[0], mattmp0, dictByUsers[3], dictByUsers2[3])
            mat4=matchDict(dictByUsers[1], mattmp1, dictByUsers[3], dictByUsers2[3])
            del mattmp0
            del mattmp1
         else:
            mat1=matchDict(dictByUsers[0], dictByUsers2[0], dictByUsers[3], dictByUsers2[3])
            mat4=matchDict(dictByUsers[1], dictByUsers2[1], dictByUsers[3], dictByUsers2[3])
         sparseArrayVarianceFilter1 = varSelector1.fit_transform(mat1)
         sparseArrayVarianceFilter4 = varSelector1.fit_transform(mat4)
         sparseArrayRowNorm1_2=transformer.fit_transform(sparseArrayVarianceFilter1)
         sparseArrayRowNorm4_2=transformer.fit_transform(sparseArrayVarianceFilter4)
         sparseArrayRowNorm= sparseArrayRowNorm + [sparseArrayRowNorm1_2,sparseArrayRowNorm4_2]
   return processYX(y_all, sparseArrayRowNorm)


def cvGridSearch(paramDict, clf, sparseArrayRowNorm, y_all, accuMeasure):
   paramIter=list(ParameterGrid(paramDict))
   auclist=[]
   print clf
   for ii in xrange(len(paramIter)):
      print paramIter[ii]
      selection=paramIter[ii]['selection']
      paramComb0=paramIter[ii].copy()
      paramComb0.pop('selection',None)
      clf.set_params(**paramComb0)
      tmp=crossvalidate(10, 5, sparseArrayRowNorm, y_all, clf, accuMeasure, selection)
      auclist.append(tmp)
   maxIdx=np.argmax(auclist)
   maxAUC=np.amax(auclist)
   return clf, paramIter[maxIdx], maxAUC


def cvGridSearchMutiple(paramClfPairs, sparseArrayRowNorm, y_all, accuMeasure):
   auclist=[]
   clflist=[]
   paramlist=[]
   for xx in paramClfPairs:
      clf, params, maxAUC= cvGridSearch(xx[0], xx[1], sparseArrayRowNorm, y_all, accuMeasure)
      auclist.append(maxAUC)
      clflist.append(clf)
      paramlist.append(params)
   maxIdx=np.argmax(auclist)
   maxAUCFinal=np.amax(auclist)
   clfFinal=clflist[maxIdx]
   paramFinal=paramlist[maxIdx]
   return clfFinal, paramFinal, maxAUCFinal


y_all, sparseArrayRowNorm = getDataRisk(filepath, False, False)
paramClfPairs = [[{'selection':[100],'n_estimators': [1000], 'criterion': ['entropy']}, RandomForestClassifier(n_jobs=16, random_state=0)]]
accuMeasure = roc_auc_score
clf, params, maxAUC = cvGridSearchMutiple(paramClfPairs, sparseArrayRowNorm, y_all, accuMeasure)