from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv('daily-temp.csv', header=0)
temps = DataFrame(series.values)

print(temps.head())



#Sliding window of size 1
print('Sliding window of size 1')
dataframe = concat([temps.shift(1), temps], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))



#Sliding window of size 3
print('Sliding window of size 3')
dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
dataframe.columns = ['t-3', 't-2', 't-1', 't']
print(dataframe.head(5))


#Creating a rolling window with means
#print('Creating a rolling window with mean')
shifted = temps.shift(1)
window = shifted.rolling(window=2)
print('++++++++++++++++')
print(window)
means = window.mean()
dataframe = concat([means, temps], axis=1)
dataframe.columns = ['mean(t-2,t-1)', 't+1']
print(dataframe.head(5))


#More statistics on the rolling window
print('Adding additional statistucs')
width = 3
shifted = temps.shift(width - 1)
window = shifted.rolling(window=width)
dataframe = concat([window.min(), window.mean(), window.max(), temps], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(5))
'''
#Expanding window and statistics
print('Expanding window and statistics')
window = temps.expanding()
dataframe = concat([window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(5))
'''
