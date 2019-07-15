import pandas as pd

df = pd.read_csv('dataset.csv', index_col = 0)

x = df.drop(['id1', 'id2', 'winner'], axis = 1)
y = df['winner']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size = 0.1
)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear',multi_class='auto')
model.fit(xtrain, ytrain)

import joblib
joblib.dump(model,'modeljoblib')