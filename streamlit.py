import pandas as pd
import numpy as np

df= pd.read_csv("Covid Dataset.csv")

df_conv = np.where(df.values == 'Yes',1,0)
df_conv = pd.DataFrame(df_conv, columns=df.columns)

X = df_conv.iloc[:,:-1]
y = df_conv.iloc[:,-1]


from collections import Counter
from imblearn.under_sampling import RandomUnderSampler 

tl = RandomUnderSampler(random_state=10)
X_res, y_res = tl.fit_resample(X, y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size = 0.7, random_state=0)

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train) 

import pickle
pickle.dump(decision_tree, open('covid_DT.pkl', 'wb'))
pickle.dump(X_test, open('covid_Xtest.pkl', 'wb'))
pickle.dump(y_test, open('covid_Ytest.pkl', 'wb'))