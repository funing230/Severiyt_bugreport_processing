import pandas as pd
import numpy as np
from sklearn import preprocessing




metrics = pd.read_csv('dataset/' + 'commit_metrics' + '.csv') #
metrics_drop=metrics.drop(labels=['class','type', 'lcom*', 'tcc', 'lcc'],axis=1)


# # complexity.rename(columns = {'Unnamed: 10' : 'result'}, inplace = True)
# # print(metrics.isna().sum());
# # print("-----------------------")
# # metrics.fillna(0);
# # print(metrics.columns);
# # print("***********************")
# # metrics.drop([metrics.columns[[2,12,13,14]]],axis=1,inplace=True);
#
# print(metrics_drop.columns);
# print("++++++++++++++++++++++++")
# print(metrics_drop.isna().sum());
# print("++++++++++++++++++++++++")
#
# metrics_drop=pd.DataFrame(metrics_drop)
# print(metrics_drop.tail(10))

mix_max_scaler=preprocessing.MinMaxScaler()
metrics_scaler=mix_max_scaler.fit_transform(metrics_drop)
metrics_scaler=pd.DataFrame(metrics_scaler)
print(metrics_scaler.tail(10))
