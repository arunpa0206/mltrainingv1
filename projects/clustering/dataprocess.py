
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize #package for flattening json in pandas df

#Input Gile Name
#raw_filename = 'dump.json'
raw_filename = '20k_records.json'

#Columen and JSON Tag Names
__id='_id'
__content='cid#name'
__starcast='starCast'
__language='language'
__genres ='genres'

#Generate a dataframe for the given data dictionary with type
def getdf(data,type):
    list_id_type_dict=[]

    for i in (0,len(data)-1,1):
        if(len(data[i][type])==1):
            temp={}
            temp[__id] = str(i)
            temp[type] = data[i][type][0]['fieldName']
            list_id_type_dict.append(temp)
            continue
        else:
            try:
                for j in (0, len(data[i][type])-1,1):
                    #print(data[i][type])

                    temp={}
                    if(__id in data[i].keys()):
                        #temp[__id] = data[i][__id]
                        temp[__id] = str(i)+str(j)
                    #print(data[i][type][i]['fieldName'])
                    if(type in data[i].keys()):
                        temp[type] = data[i][type][j]['fieldName']
                    list_id_type_dict.append(temp)
            except IndexError:
                #Indicates we got past the IndexError
                continue
    #print('No of ', type, len(list_id_type_dict), 'from', len( data), 'rows')
    df = pd.DataFrame(list_id_type_dict)
    return df


#Calculate the frequency of content watch
#Extend this for different types of engagement
def addengagementmetrics(df,type):
    df['freq'] = df.groupby(type)[type].transform('count')
    df=df.drop_duplicates()
    return df

#Build different perspectives from the giben data
def buildperspective(data,type):
    df=getdf(data,type)
    #print('Data Points for building perspective',len(df))
    df=addengagementmetrics(df,type)
    print('=== '+type+' Perspective====')
    print(df.head())
    df.to_csv(type+'.csv', sep=',', encoding='utf-8')




#Read the JSON fule and load in data dictionary
data={}
with open(raw_filename, 'r') as f:
    data=json.load(f)
    #print('No of data items:',len(data))

#Build different perspectives
buildperspective(data, __content)
buildperspective(data, __starcast)
buildperspective(data, __language)
buildperspective(data, __genres)

'''
from spherecluster import SphericalKMeans
skm = SphericalKMeans(n_clusters=K)
skm.fit(X)
'''
