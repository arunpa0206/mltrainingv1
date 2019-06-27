#Input height and weight data
height=[[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]]
weight=[  42 ,  44 , 49, 55  , 53  , 58   , 60  , 64  ,  66 ,  69]

print("height -> weight")
for row in zip(height, weight):
    print(row[0][0],"->",row[1])

#Plot the input data
import matplotlib.pyplot as plt
plt.scatter(height,weight,color='black')
plt.xlabel("height")
plt.ylabel("weight")
plt.show()

#Rub linear regression on input mlodel
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(height,weight)

#Get linear reegression coefficients
m=reg.coef_[0]
b=reg.intercept_
print("slope=",m, "intercept=",b)

#Plot the data points and the line
plt.scatter(height,weight,color='black')
predicted_values = [reg.coef_ * i + reg.intercept_ for i in height]
plt.plot(height, predicted_values, 'b')
plt.xlabel("height")
plt.ylabel("weight")
plt.show()

#Make predictions
print('Weight for heoght = 5.2',reg.predict(X=[[6.2]]))
