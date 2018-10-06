import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from spherecluster import SphericalKMeans

types=['cid','genres','language','starCast']

#df.drop(['_id'], axis = 1, inplace = True)
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df


def preprocessdata(df,type):
    df = handle_non_numerical_data(df)
    #df.drop(['_id',type],axis=1)
    df.drop(['_id'],axis=1)


    X=np.array(df.astype(float))
    X= preprocessing.scale(X)

    return X


for type in types:
    df=pd.read_csv(type+'.csv', index_col=0)
    X=preprocessdata(df, type)

    skm = SphericalKMeans(n_clusters=2)
    skm.fit(X)

    df=pd.read_csv(type+'.csv', index_col=0)
    df['cluster_id']=skm.labels_
    print(df)
