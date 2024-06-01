# General Functions for binary, ternary and Bethesda classifier
# Classification, cross validation, data augmentation

import numpy as np
import pandas as pd
from math import sqrt
import os
import sys
import csv
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import feature_selection as fs
from sklearn import preprocessing
from datetime import datetime
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from itertools import cycle
from random import randint
from random import sample
import xgboost as xgb


## FUNCTIONS used in hierarquical classifiers: 

# Select indexes from (idx_test) k-fold indexes, of samples that received a cls prediction by the model
def index_pred_from_class(idx_test, pred_y, cls=0):
    idx = []
    for i, pred in zip(idx_test, pred_y):
        if (pred == cls):
            idx.append(i)
    return idx

## Filter idx of a dataframe based on cls value
def filter_lines(y, cls):
   lines = []
   for idx, value in zip(y.index, y.values):
        if (value in cls):
            lines.append(idx)
   return lines

def filter_dataXY(X, y, cls_type):
    # X e y devem ser do tipo dataframe 
    if cls_type == 1: # (normal/anormal)
        return (X, y['binary'])
    elif cls_type == 2: # (baixo/alto grau) 
          lines = filter_lines(y['ternary'], [1,2])
          return (X.loc[lines], y['ternary'].loc[lines])  
    elif cls_type == 3: # (ASCUS/LSIL)
          lines = filter_lines(y['bethesda'], [1,2])
          return (X.loc[lines], y['bethesda'].loc[lines])  
    else: #(ASCH/HSIL/Car)    
          lines = filter_lines(y['bethesda'], [3,4,5])
          return (X.loc[lines], y['bethesda'].loc[lines])   

## Filter samples to enter in the level 1 of the hierarquical classifier
def filter_Xy_from_cls1_to_cls2(data, target, predics_bin, idx_test):
    lines = []
    for i in idx_test:
        if predics_bin[i] == 1: #Anormal  
             lines.append(i)
            
    X = data.loc[lines]
    y = target['ternary'].loc[lines]
    return (lines, X, y)

## Filtra dados para teste do classificador 3 com base nas predições do classificador 2:
def filter_Xy_from_cls1_to_cls3(data, target, predics_ter, idx_test):
    lines = []
    for i in idx_test:
        if predics_ter[i] == 1:  #lesão de baixo grau
             lines.append(i)
            
    X = data.loc[lines]
    y = target['bethesda'].loc[lines]
    return (lines, X, y)

## Filtra dados para teste do classificador 4 com base nas predições do classificador 2:
def filter_Xy_from_cls2_to_cls4(data, target, predics_ter, idx_test):
    lines = []
    for i in idx_test:
        if predics_ter[i] == 2:  #lesão de alto grau
             lines.append(i)
            
    X = data.loc[lines]
    y = target['bethesda'].loc[lines]
    return (lines, X, y)


## FUNCTIONS used in plain/hierarquical classifiers: 

def split_per_classes(df):
  # separate data for classification showing the number of samples
  # per Bethesda diagnostic's class
  data_nilm = df[df['bethesda'] == 0].copy()
  data_nilm.set_index((i for i in range(data_nilm.shape[0])), inplace=True)

  data_ascus = df[df['bethesda'] == 1].copy()
  data_ascus.set_index((i for i in range(data_ascus.shape[0])), inplace=True)

  data_lsil = df[df['bethesda'] == 2].copy()
  data_lsil.set_index((i for i in range(data_lsil.shape[0])), inplace=True)

  data_asch = df[df['bethesda'] == 3].copy()
  data_asch.set_index((i for i in range(data_asch.shape[0])), inplace=True)

  data_hsil = df[df['bethesda'] == 4].copy()
  data_hsil.set_index((i for i in range(data_hsil.shape[0])), inplace=True)

  data_scc = df[df['bethesda'] == 5].copy()
  data_scc.set_index((i for i in range(data_scc.shape[0])), inplace=True)

  print("--- Samples per class  --- ")
  print("NILM.....: ", data_nilm.values.shape[0])
  print("ASC-Us...: ", data_ascus.values.shape[0])
  print("ASC-H....: ", data_asch.values.shape[0])
  print("LSIL.....: ", data_lsil.values.shape[0])
  print("HSIL.....: ", data_hsil.values.shape[0])
  print("SCC......: ", data_scc.values.shape[0])
  print("TOTAL....: ", data_nilm.values.shape[0]+ data_ascus.values.shape[0] + \
        data_lsil.values.shape[0] + data_asch.values.shape[0] + \
        data_hsil.values.shape[0] + data_scc.values.shape[0])

  return data_nilm, data_ascus, data_lsil, data_asch, data_hsil, data_scc



def split_data_targe_ids(nilm, ascus, lsil, asch, hsil, scc):
  # Split CRIC dataset to binary, ternary and bethesda classification
  # return data(all columns), targets(y) and cells_ids (cell_id, image_id, image_filename)
  data =  pd.DataFrame(data=np.vstack([
                          nilm.values,
                          ascus.values,
                          lsil.values,
                          asch.values,
                          hsil.values,
                          scc.values]),
                         columns = scc.columns)

  cells_ids= data[['bethesda','image_id', 'cell_id']].copy()

  # Ajusta y(target) para classificação binária, ternária além de bethesda
  y = data['bethesda'].values
  y_bin = y.copy()
  y_ter = y.copy()

  for i in np.arange(data.shape[0]):
        y_bin[i] = 0 if y_bin[i]==0 else 1

  for i in np.arange(data.shape[0]):
        if ((y_ter[i] > 0) and (y_ter[i] < 3)):  ## ASCUS(1) or Lsil(2)
            y_ter[i] = 1
        elif (y_ter[i] >= 3):  ##ASCH(3), HSIl(4) e SCC or carcinoma(5)
              y_ter[i] = 2

  target = pd.DataFrame(data = np.stack([y_bin,
                                          y_ter,
                                          y], axis=-1),
                        columns = ['binary', 'ternary', 'bethesda'])

  return (data, target, cells_ids)

 
#Calc matriz de p-values para dif de médias de uma features entre as classes 0 a 6
def getMatrix_ttest(stats, variable, varEqual = False, alpha = 0.05):
     ## stats: dataframe de features normalizadas
     ## variable: list of features to analysis
     ## varEqual: True to same variance T-test or False to not equal variance T-test
     ## alpha: significant level of analysis
     lista=[stats[stats['bethesda'] == 0][variable].values, \
            stats[stats['bethesda'] == 1][variable].values, \
            stats[stats['bethesda'] == 2][variable].values, \
            stats[stats['bethesda'] == 3][variable].values, \
            stats[stats['bethesda'] == 4][variable].values, \
            stats[stats['bethesda'] == 5][variable].values]

     mat=np.zeros((6,6), dtype=float)
     for i in np.arange(6):
        for j in np.arange((i+1), 6):
             res = ttest_ind(lista[i], lista[j],  equal_var=varEqual)
             tupla_vne = (res.statistic,  res.pvalue, (res.pvalue > alpha))
             mat[j,i] = res.pvalue
             mat[i,j] = res.pvalue > alpha
     return mat

# TODO: separate issues so that this fuction shows only the matrix with respective values
def plot_pvalues_variables(stats, vars, c, l):
## Plot T-test values in a matrix similar to a confuzion matrix (in Matplotlib) 
# acording to the values of var (columns) in dataframe informing width and lenght of plot
    fig, ax= plt.subplots(l, c, figsize=(3*c, 3.6*l))
    ind = 0
    for il in np.arange(l):
        for ic in np.arange(c):
            if ind < len(vars):
                var = vars[ind]
                ind = ind+1
                mat = getMatrix_ttest(stats, var, varEqual = False)
                ax[il, ic].matshow(mat, cmap='cool')
                for (i, j), z in np.ndenumerate(mat):  # ndenumerate extracts a lin,col tuble and the matrix value
                    ax[il, ic].text(i, j, '{:0.1g}'.format(z), fontsize=7, ha='center', va='center')
                ax[il, ic].set(title = var+' - T-Test p_value')
    plt.show()

# Build Models 
def getModel(params, classifier = 'SVM', class_type = 'binary'):
    if classifier == 'SVM':
          model = SVC(probability = True, random_state=27).set_params(**params)
    elif classifier == 'RF':
          model = RandomForestClassifier(oob_score=True, random_state=27).set_params(**params)
    elif classifier == 'XGBoost':
        if class_type == 'binary':
            model = xgb.XGBClassifier(objective= 'binary:logistic',seed=27).set_params(**params)
        else:    # multiclass  
            model = xgb.XGBClassifier(objective= 'multi:softprob', seed=27).set_params(**params) 
    else:
        model = None # 'MLP toDo'    
    return model   
 
  
## Fit models for each cross validation fold using the given data augmentation method (SMOTE if smote = 0, otherwise BorderlineSMOTE) )
## at each fold (return train metrics)
def fit_model(X, y, model, cls_type= 1, smote=0):
    le = preprocessing.LabelEncoder()
    if cls_type == 1:  ## binary clas. normal/anormal (or non hierarquical)
            cls = None
    elif cls_type == 2:   ## ternary clas. - low/high degree
            cls = [1,2]
            le.fit(cls)
    elif cls_type == 3:  ## bethesda labels for asc-us/lsil classes
            cls = [1,2]
            le.fit(cls)
    elif cls_type == 4:   ## bethesda labels for asch/hsil/car classes
            cls = [3,4,5]
            le.fit(cls)
       
    if smote == 0:
        smoter = SMOTE(random_state=42)
    else:    
        smoter = BorderlineSMOTE(random_state=42)
            
    # Make Upsample for training data
    X_train, y_train = X,y
    #X_train_upsample, y_train_upsample = smoter.fit_resample(X_train, y_train)
    X_train_upsample, y_train_upsample = X_train, y_train
    
    ## Codify y's labels for running a classifiers type param is 2, 3 or 4
    if (cls_type != 1):  
        y_train_upsample = le.transform(y_train_upsample.astype(np.int32))
    else:  ## if clas_type = 1 it will be made a binary or
           ## a hierarquical bethesda Low/Hight classification
        y_train_upsample = y_train_upsample.astype(np.int32)
            
    fitted_model = model.fit(X_train_upsample, y_train_upsample)            

    return None, fitted_model

 
# Calc metrics: (vide metrics_type and classifiers_type)
def calc_metric(target_test, target_predict, metric_type='acc', class_type ='binary', pos_label=1, classes=[0,1],zero_division=0):   
    if (metric_type == 'acc'):
        return accuracy_score(target_test, target_predict,zero_division=0)
    elif (metric_type == 'prec'):
         if (class_type == 'binary'):  ## caso classificadores binário
            return  precision_score(target_test, target_predict, pos_label= pos_label, zero_division=0) 
            return f1 
         else:  ## multiclasses
            return precision_score(target_test, target_predict, average='weighted',zero_division=0)
    elif (metric_type == 'rec'):
        if (class_type == 'binary'):  ## classificadores binários
            return recall_score(target_test, target_predict, pos_label= pos_label,zero_division=0)
        else:  ## multiclasses
            return  recall_score(target_test, target_predict, average ='weighted',zero_division=0)
    elif (metric_type == 'spec'):   
         if (class_type == 'binary'):  ## classificadores binários
            tn, fp, fn, tp = confusion_matrix(target_test, target_predict).ravel()
            if (tn + fp) == 0:
                print('tn + fp is zero!')
                return 0
            else:
                return (tn/(tn + fp))
         else:  ##  multiclasses - média aritmética  
            spec = 0
            for l in classes:
                tn, fp, fn, tp = confusion_matrix((np.array(target_test)==l), (np.array(target_predict)==l)).ravel()
                if int((tn + fp)) == 0:
                    spec+=0
                else:    
                    spec += float(tn)/float((tn + fp))
            return spec/len(classes)  ##specificity as 'average' equals micro
    elif (metric_type == 'f1_score'):      
         if (class_type == 'binary'):  ## classificadores binários
            f1 = f1_score(target_test, target_predict, pos_label= pos_label, zero_division=0) 
         else:  ## multiclasses
            f1 = f1_score(target_test, target_predict, average= 'weighted', zero_division=0)
         return f1 
    else:
        return None
    

def fill_line_metrics_CV(model_name, featur, line_results, metrics, results, class_type='binary'):
    line_series = pd.Series(data = np.array([class_type, model_name, featur,
             '{:.4f}'.format(metrics['acc']), '{:.4f}'.format(metrics['prec']),
             '{:.4f}'.format(metrics['rec']),'{:.4f}'.format((1.0- metrics['spec'])),
             '{:.4f}'.format((1.0 - metrics['rec'])), '{:.4f}'.format(metrics['spec']), 
             '{:.4f}'.format(metrics['f1_score'])], dtype = object), 
              index=['Clas.Type', 'Model', 'Features', 'Accuracy', 'Precision', 'Recall' , 
                     'False Pos', 'False Neg.', 'Specif', 'F1_measure']) 
    return line_series.values  #onlin data

## Time spent to write:
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

        
# Show ROC curve for binary classifications: 
def plot_roc_curve_CV(roc_curve_list, labels_list, title = "ROC Curve"):
    fig, ax = plt.subplots(figsize=(9,5))
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "deeppink", "navy", "darkorange"])
    plt.style.use("bmh")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    for i,color in zip(range(len(roc_curve_list)), colors):
        ax.plot(
            roc_curve_list[i][0],
            roc_curve_list[i][1],
            color=color,
            label=labels_list[i],
            lw=2,
            alpha=0.8,
        )
 
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title= title
    )
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    plt.show()


  # Show 3 confusion matrix for all classifiers model:
def plot_conf_matrix(preds_to_conf_matrix, lbls=[0,1], disp_lbls=['normal', 'anormal']):
    fig, axes = plt.subplots(nrows=1, ncols= 3, figsize=(15,9))
    # (TODO: alter ncols from 3 to n) 
    for i,ax in enumerate(axes.flatten()):
          ConfusionMatrixDisplay.from_predictions(preds_to_conf_matrix[i][0], preds_to_conf_matrix[i][1], 
                                    labels= lbls, cmap='Blues', colorbar=False, ax=ax, display_labels=disp_lbls)
          ax.title.set_text(preds_to_conf_matrix[i][2])
    plt.tight_layout()  
    plt.show()  