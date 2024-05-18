import sys
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from common.data import GetData
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from colorama import Fore
from scipy.optimize import differential_evolution





ZERO_NUM = 0
ERROR = 0.0
DIM = 0



class DataManager:
    data = GetData(ZERO_NUM, DIM)
    test = GetData(ZERO_NUM, DIM)
    
    @classmethod
    def get_statistic_df(cls):
        res = pd.DataFrame([dict(i.get_stats()) for i in cls.data]).T
        res.columns = [i.get_name() for i in cls.data]
        return res

    @classmethod
    def get_metrics_df(cls):
        return pd.DataFrame([i.get_metrics() for i in cls.data])

    @classmethod
    def get_united_metrics_df(cls):
        return pd.DataFrame([i.get_united_metrics() for i in cls.data])

    @classmethod
    def get_united_metrics_val(cls):
        return np.array([i.get_united_metrics().values() for i in cls.data])

    @classmethod
    def calculate_J_val(cls, c_array, mean, std):
        res = []
        for gpu_ in cls.data:
            metrics = list(gpu_.get_united_metrics().values())
            sum = 0
            for metr_, _ in enumerate(metrics):
                sum+= c_array[metr_]*((metrics[metr_] - mean[metr_])/std[metr_])
            res.append((sum, gpu_.get_Jval()))
        return res

    @classmethod
    def get_united_metric_names(cls):
        return pd.DataFrame(cls.data[0].get_united_metrics().keys())

    @classmethod
    def get_classification_df(cls):
        classification_table = []
        for i in [(gpu.get_name(), gpu.get_Jval()) for gpu in cls.data]:
            row = [i[0]]
            for j in [(gpu.get_name(), gpu.get_Jval()) for gpu in cls.data]:
                if j[0] == i[0]:
                    row.append(0)
                    continue
                if i[1] > j[1]:
                    row.append(-1)
                else:
                    row.append(1)
            classification_table.append(row)
        classificationDF = pd.DataFrame(classification_table, columns=['name']+[gpu.get_name() for gpu in cls.data])
        return classification_table
    
    @classmethod
    def get_training_Y(cls):
        res = []
        gpu_vals = [gpu.get_Jval() for gpu in cls.data]
        gpu_n = len(gpu_vals)
        for i in range(gpu_n):
            for j in range(gpu_n):
                if abs(gpu_vals[i] - gpu_vals[j]) > ERROR:
                    if j == i:
                        continue
                    if gpu_vals[i] > gpu_vals[j]:
                        res.append(1)
                    else:
                        res.append(-1)
        return res
    
    @classmethod
    def get_training_X(cls, pca_data):
        pca_data = np.array(pca_data)
        res = []
        gpu_n = len(pca_data)
        metric_n = len(pca_data[0])
        for i in range(gpu_n):
            for j in range(gpu_n):
                if abs(cls.data[i].get_Jval() - cls.data[j].get_Jval()) > ERROR:
                    if j == i:
                        continue
                    row = []
                    for k in range(metric_n):
                            row.append(pca_data[i][k] - pca_data[j][k])
                    res.append(row)
        return res
    
    @classmethod
    def testing(cls, c_array, mean, std):
        res1 = []
        for gpu_ in cls.test:
            metrics = list(gpu_.get_united_metrics().values())
            sum = 0
            for metr_, _ in enumerate(metrics):
                sum+= c_array[metr_]*((metrics[metr_] - mean[metr_])/std[metr_])
            res1.append((sum, gpu_.get_name()))
        res1.sort(key= lambda x: x[0])

        res=[]
        for gpu_ in cls.test:
            res.append((gpu_.get_Jval(), gpu_.get_name()))
        res.sort(key= lambda x: x[0])
        s = 0
        for i in range(len(res1)-1):
            k=-1
            for j in range(len(res)):
                if res1[i][1] == res[j][1]:
                    k = j
                if res1[i+1][1] == res[j][1] and k!=-1:
                    s+=1
        print(f"TESTING ACCURACY: {s/len(cls.test)} ZERO_NUM: {ZERO_NUM}, ERROR: {ERROR}, DIM: {DIM}")
        return s/len(cls.test)

class myPSA:
    def __init__(self, df) -> None:
        data_scaled = np.array(df)
        self.data_scaled = data_scaled
        pca = PCA().fit(data_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        # n_components = np.argmax(cumulative_explained_variance >= 0.9999) + 1
        n_components = min(len(data_scaled[0]), len(data_scaled))
        print(f"Optimal number of components: {n_components}")
        pca = PCA(n_components=n_components)
        self.data_pca = pca.fit_transform(data_scaled)
        self.components = pca.components_

    def get_new_data(self):
        # return pd.DataFrame(self.data_scaled @ self.components.T)
        return pd.DataFrame(self.data_pca)

    def get_components(self):
        return self.components


class Regression:
    def __init__(self, data) -> None:
        classification_X = DataManager.get_training_X(data)
        classification_Y = DataManager.get_training_Y()
        
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(classification_X, classification_Y, test_size=0.01, random_state=0)
        clf = svm.SVC(kernel='linear')
        clf.fit(classification_X, classification_Y)
        y_pred = clf.predict(self.X_test)
        self.coefficients = clf.coef_
        self.accuracy = accuracy_score(self.y_test, y_pred)

    def get_coefficients(self):
        return self.coefficients


class PicturesGenerator:
    @staticmethod
    def get_df_plot(df):
        df.plot(logy=False, legend=False)
        plt.show()
        
    @staticmethod
    def get_df_corr_matrix(df):
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='vlag',
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
        plt.show()



class UIController:
    @staticmethod
    def show_graphs():
        statsDF = DataManager.get_statistic_df()
        PicturesGenerator.get_df_plot(statsDF)

        metricsDF= DataManager.get_metrics_df()
        PicturesGenerator.get_df_corr_matrix(metricsDF)

        metricsDF= DataManager.get_united_metrics_df()
        PicturesGenerator.get_df_corr_matrix(metricsDF) 


def execute(h):
    global ZERO_NUM
    global ERROR
    global DIM
    ZERO_NUM = h[0]
    ERROR = h[1]
    DIM = h[2]
    DataManager.data = GetData(ZERO_NUM, DIM)
    DataManager.test = GetData(ZERO_NUM, DIM)
    
    metricsDF= DataManager.get_united_metrics_df()
    scaledMetricsDF = (metricsDF - metricsDF.mean()) / metricsDF.std()
    # print(scaledMetricsDF)
    psa = myPSA(scaledMetricsDF)
    NewData = psa.get_new_data()
    svm = Regression(NewData)
    # print(f"Accuracy: {svm.accuracy}")
    CompCoeff = psa.get_components()
    SvmCoeff = svm.get_coefficients()
    coefArray = (SvmCoeff @ CompCoeff)
    coefArray =  coefArray[0]
    mean = list(metricsDF.mean())
    std = list(metricsDF.std())
    s = DataManager.calculate_J_val(coefArray, mean, std)
    s.sort(key=lambda x: x[1])
    # print(DataManager.testing(coefArray, mean, std))
    return 1/DataManager.testing(coefArray, mean, std)
    
    
if __name__ == '__main__':
    print(differential_evolution(execute, bounds=[(30,80), (0, 0.06), (0, 4)]))