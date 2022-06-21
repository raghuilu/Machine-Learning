import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score
fish_data=pd.read_csv('/Users/ilu/Downloads/Fish.csv')
#print(dfcro_outliers)
without_outliers=fish_data.drop([142,143,144])
#print(without_outliers.describe().T)
#dependent
y=without_outliers['Weight']
x=without_outliers.iloc[:,2:7]#independent
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=1)
def models_score(train_data, y_train, val_data, y_val):
        regression = Lasso(alpha=1.0,tol=0.01)
        regression.fit(train_data, y_train)
        pred = regression.predict(val_data)
        score_MSE = mean_squared_error(pred, y_val)
        print("R-Squared:" +str(r2_score(y_val,pred)))
        print("95% confidence interval: ",st.t.interval(alpha=0.95,df=len(pred)-1, loc=np.mean(pred),scale=st.sem(pred)))
        print("Standard error: "+str(np.std(pred,ddof=1)/np.sqrt(np.size(val_data))))
        print("Coefficients for the model:",regression.coef_)
        return round(score_MSE, 2)
print(models_score(xTrain,yTrain,xTest,yTest))
