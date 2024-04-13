import sys
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy 

from common.data import GetData

def main():
    data = GetData()
    statsDF = pd.DataFrame([dict(i.get_stats()) for i in data]).T
    statsDF.columns = [i.get_name() for i in data]
    statsDF.plot(logy=False)
    plt.show()
    statsDF.plot(logy=False, legend=False)
    plt.show()
    statsDF.plot(logy=True, legend=False)
    plt.show()

    metricsDF= pd.DataFrame([i.get_metrics() for i in data])

    corr = metricsDF.corr()
    sns.heatmap(corr, annot=True, cmap='vlag',
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.show()

    unitedDF = pd.DataFrame([i.get_united_metrics() for i in data])

    corr = unitedDF.corr()
    sns.heatmap(corr, annot=True, cmap='vlag',
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.show()

    classification_table = []
    for i in [(gpu.get_name(), gpu.get_Jval()) for gpu in data]:
        row = [i[0]]
        for j in [(gpu.get_name(), gpu.get_Jval()) for gpu in data]:
            if j[0] == i[0]:
                row.append(0)
                continue
            if i[1] > j[1]:
                row.append(1)
            else:
                row.append(-1)
        classification_table.append(row)

    classificationDF = pd.DataFrame(classification_table, columns=['name']+[gpu.get_name() for gpu in data])
    classificationDF.to_csv('ClassificationData.csv')

    
    
    j_arr = [(gpu.get_name(), gpu.get_Jval()) for gpu in data]
    j_arr.sort(key=lambda x: x[1])

    for i in j_arr:
        print(i)
if __name__ == '__main__':
    sys.exit(main())