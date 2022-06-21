from sklearn.datasets import load_iris
iris=load_iris()
import pandas as pd
data=pd.DataFrame(iris.data)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(data.head())
target = pd.DataFrame(iris.target)
target = target.rename(columns = {0: 'target'})
print(target.head())
df = pd.concat([data, target], axis = 1)
X = df.copy()
y = X.pop('target')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify = y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
df.target.value_counts(normalize= True)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
df_coef = pd.DataFrame(model.coef_, columns=X_train.columns)
print(df_coef)
