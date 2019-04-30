from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()

X = iris.data[:,[1,3]]
y = iris.target

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0 )

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

lr = LogisticRegression(C=500, random_state = 0)
lr.fit(X_train,y_train)
acc =lr.score(X_test, y_test)

print(acc*100,"%")
