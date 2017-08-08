'''
Deep Learning Nanodegree Udacity
Juan M. Gandarias
Linear Regression example
07/08/17
-----------------------------------
Encironment:
conda create -n siraj-regression python=2
conda install pandas matplotlib scikit-learn
'''

# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv") 

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model =  LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])
print(laos_life_exp)

# Print the linear regression model
plt.scatter(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])
plt.plot(bmi_life_data[['BMI']], bmi_life_model.predict(bmi_life_data[['BMI']]))
plt.show()