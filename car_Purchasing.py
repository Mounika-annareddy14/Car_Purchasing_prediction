'''
In this file we are going to Develop Multiple Linear Regression Model for Car Purchasing
'''

import sys
import math
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score , mean_squared_error


import warnings
warnings.filterwarnings('ignore')

class CarPurchasing:

  def __init__(self , data):
    try:
      self.df = pd.read_csv(data , encoding='latin-1')
      self.df = self.df.drop(['Customer Name', 'Customer e-mail','Country'] , axis = 1)
      self.X = self.df.iloc[: , :-1] # independent columns
      self.y = self.df.iloc[: , 1]  # dependent columns
      self.X_train , self.X_test , self.y_train , self.y_test = train_test_split(self.X ,self.y,test_size=0.3 ,random_state=42 )
      #print(self.df)
      #print(self.df.columns)
    except Exception as e:
      error_type , error_msg , error_line = sys.exc_info()
      print(f'error from line : {error_line.tb_lineno}--> {error_msg}--> {error_type}')

  def preparing_data(self):
    try:
      self.model = LinearRegression()
      self.model.fit(self.X_train , self.y_train)
    except Exception as e:
      error_type , error_msg , error_line = sys.exc_info()
      print(f'error line from {error_line.tb_lineno}--> {error_msg}-->{error_type}')

  def Train_performance(self):
    ''' Checking the Performance of Trained Data'''
    try:
      self.y_train_predict = self.model.predict(self.X_train)
      print(f'Trained Accuracy : {r2_score(self.y_train , self.y_train_predict)}')
      #print(f'Trained MSE : {mean_squared_error(self.y_train , self.y_train_predict)}')
#Mean Squared Residual
      msr = np.mean((self.y_train - self.y_train_predict)**2)
      print(f' Trained Mean Squared Error : {msr}')
      print()
# Root Mean Squared Residual
      msr = np.mean((self.y_train - self.y_train_predict)**2)
      rsmr = math.sqrt(msr)
      print(f'Trained Root Mean Squared Residual (RMSR):{rsmr}')
      print()

# Absolute Mean Square Error
      total_absolute_error = 0
      for i in range(len(self.y_train)):
        if self.y_train[i] - self.y_train_predict[i] < 0:
          error = -(self.y_train[i] - self.y_train_predict[i])  # Manual absolute calculation
        else:
          error = self.y_train[i] - self.y_train_predict[i]
        total_absolute_error += error
      mean_absolute_error = total_absolute_error / len(self.y_train)
      print(f"Trained Mean Absolute Error (MAE): {mean_absolute_error}")
      print()
    except Exception as e:
      error_type , error_msg , error_line = sys.exc_info()
      print(f'error from line no : {error_line.tb_lineno}-->{error_msg}-->{error_type}')


  def Test_performance(self):
    ''' Checking the Performance of Tested Data'''
    try:
      self.y_test_predict = self.model.predict(self.X_test)
      print(f'Tested Accuracy : {r2_score(self.y_test , self.y_test_predict)}')
      #print(f'Tested MSE : {mean_squared_error(self.y_test , self.y_test_predict)}')
# Mean squared Residuals
      msr = np.mean((self.y_test - self.y_test_predict)**2)
      print(f'Tested Mean Squared Error : {msr}')
      print()
# Root Mean Squared Error
      msr = np.mean((self.y_test - self.y_test_predict)**2)
      rmsr = math.sqrt(msr)
      print(f'Tested Root Mean Squared Residual (RMSR): {rmsr}')
      print()
# Absolute Mean Error
      total_absolute_error = 0
      for i in range(len(self.y_test)):
        if self.y_test[i] - self.y_test_predict[i] < 0:
          error = -(self.y_test[i] - self.y_test_predict[i])
        else:
          error = self.y_test[i] - self.y_test_predict[i]
        total_absolute_error += error
      mean_absolute_error = total_absolute_error / len(self.y_test)
      print(f"Tested Mean Absolute Error (MAE): {mean_absolute_error}")
      print()
    except Exception as e:
      error_msg , error_type , error_line = sys.exc_info()
      print(f'error from line : {error_line.tb_lineno}-->{error_msg}-->{error_type}')
  def Ridge(self):
    try:
      self.ridge = Ridge()
      self.ridge.fit(self.X_train , self.y_train)
      print(f'Ridge Model Train Accuracy : {self.ridge.score(self.X_train , self.y_train)}')
      print(f'Ridge Model Test Accuracy : {self.ridge.score(self.X_test , self.y_test)}')
      print('-------')
    except Exception as e:
      error_msg , error_line , error_type  = sys.exc_info()
      print(f'error from line no : {error_line.tb_lineno}-->{error_msg}-->{error_type}')

  def Lasso(self):
    try:
      self.lasso = Lasso()
      self.lasso.fit(self.X_train , self.y_train)
      print(f'lasso Model Train Accuracy : {self.ridge.score(self.X_train , self.y_train)}')
      print(f'lasso Model Test Accuracy : {self.ridge.score(self.X_test , self.y_test)}')
      print('--------')
    except Exception as e:
      error_msg , error_line , error_type  = sys.exc_info()
      print(f'error from line no : {error_line.tb_lineno}-->{error_msg}-->{error_type}')

  def Table(self):
     f = pd.DataFrame()
     f['Independent_columns'] = self.X_train.columns
     f['Model_m_values'] = self.model.coef_
     f ['Ridge_m_values'] = self.ridge.coef_
     f['lasso_m_values'] = self.lasso.coef_
     print(f)



if __name__ == '__main__':
  data = CarPurchasing('G:\\My Drive\\Car_purchasing\\Car_Purchasing\\Car_Purchasing_Data.csv')
  data.preparing_data()
  data.Train_performance()
  data.Test_performance()
  data.Ridge()
  data.Lasso()
  data.Table()