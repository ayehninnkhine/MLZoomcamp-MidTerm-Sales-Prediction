import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("advertising.csv")
df["Sales"] = np.log1p(df["Sales"].values)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_train = df_train.Sales.values
y_test = df_test.Sales.values

del df_train['Sales']
del df_test['Sales']

dv = DictVectorizer(sparse=True)
train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)


rf = RandomForestRegressor(n_estimators=125,
                           max_depth=25,
                           random_state=1,
                           n_jobs=-1)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

mse = mean_squared_error(y_pred_rf, y_test)
print('The Mean Square Error(MSE) RF:', mse)

output_file = 'model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)