import sys
from matplotlib import pyplot as plt
import pandas as pd

from common.data import get_data

def main():
    data = get_data('./BenchDataset.csv')

    statsDF = pd.DataFrame([i.get_stats() for i in data]).T
    print(statsDF[1])
    statsDF.plot(logy=False)
    plt.show()

    metricsDF= pd.DataFrame([i.get_metrics() for i in data])
    print(metricsDF)




if __name__ == '__main__':
    sys.exit(main())