import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

data = pd.read_csv('AirPassengers.csv')
print(data.head())
print ('\n Data Types:')
print(data.dtypes)

#Create dates for analysis, we only have the month and year
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
#################################################
#parse_dates: This specifies the column which contains the date-time information. As we say above, the column name is ‘Month’.
#index_col: A key idea behind using Pandas for TS data is that the index has to be the variable depicting date-time information. So this argument tells pandas to use the ‘Month’ column as index.
#date_parser: This specifies a function which converts an input string into datetime variable. Be default Pandas reads data in format ‘YYYY-MM-DD HH:MM:SS’. If the data is not in this format, the format has to be manually defined. Something similar to the dataparse function defined here can be used for this purpose.
###############################################
print(data.head())
#print(data.index)


#1. Specific the index as a string constant:

#print(data['Passengers'])
ts = data['Passengers']
print(ts.head(10))

print(ts['1949-01-01'])

#1. Specify the entire range:
print(ts['1949-01-01':'1949-05-01'])
#Unlike numeric indexing, the end index is included here. For instance, if we index a list as a[:5] then it would return the values at indices – [0,1,2,3,4]. But here the index ‘1949-05-01’ was included in the output.
#The indices have to be sorted for ranges to work. If you randomly shuffle the index, this won’t work

#Consider another instance where you need all the values of the year 1949. This can be done as:
print(ts['1949'])
#Visualize the dataset
plt.plot(ts)
plt.show()

#############################################
#Identify if data is stationary
#Stationarity is defined using very strict criterion. However, for practical purposes we can assume the series to be stationary if it has constant statistical properties over time, ie. the following:

#constant mean
#constant variance
#an autocovariance that does not depend on time.
#Determing rolling statistics
#rolmean = pd.rolling_mean(ts, window=12)

rolmean = ts.rolling(window=12).mean()

rolstd = ts.rolling(window=12).std()

#Plot rolling statistics:
orig = plt.plot(ts, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()
#One of the first tricks to reduce trend can be transformation. For example, in this case we can clearly see that the there is a significant positive trend. So we can apply transformation which penalize higher values more than smaller values. These can be taking a log, square root, cube root, etc. Lets take a log transform here for simplicity:
ts_log = np.log(ts)
plt.plot(ts_log)
plt.show()

#Smoothing - Remove the rolling mean at any point
moving_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()
#Since we use moving avg for 12 monthsm it is not applicable for first 11 minths
ts_log_moving_avg_diff = ts_log - moving_avg
print(ts_log_moving_avg_diff.head(12))
#Dropping NAN predicted_values

ts_log_moving_avg_diff.dropna(inplace=True)
plt.plot(ts_log)
plt.plot(ts_log_moving_avg_diff, color='red')
plt.show()
'''
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# prepare data
data = ts_log
# create class
model = SimpleExpSmoothing(data)
'''

#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf

ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)


lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()




from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2, 1, 0))

results_AR = model.fit(disp=-1)
#plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

plt.show()
