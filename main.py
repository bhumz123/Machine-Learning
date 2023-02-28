import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housedata=pd.read_csv("house_prices.csv")
price=housedata['price']
size = housedata['sqft_living']

# MACHINE LEARNING HANDLES ARRAYS
x=np.array(size).reshape(-1,1)
y=np.array(price).reshape(-1,1)

# TO FIT THE MODEL
model= LinearRegression()
model.fit(x,y)

regression_model_mse=mean_squared_error(x,y)
print("MSE : ", math.sqrt(regression_model_mse))
print("RSE : ", model.score(x,y))

print(model.coef_[0])
print(model.intercept_[0])


# now analyse it visually
plt.scatter(x,y,color='green')
plt.plot(x,model.predict(x))

plt.show()


