import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def ohe_train_encode(X):
  ohe = OneHotEncoder(sparse_output=False)
  X_ohe = ohe.fit_transform(X)
  std = StandardScaler()
  X_ohe_scl = std.fit_transform(X_ohe)
  return X_ohe_scl, ohe, std

def ohe_not_train_encode(X, ohe=None, std=None):
  X_ohe = ohe.transform(X)
  X_ohe_scl = std.transform(X_ohe)
  return X_ohe_scl

def cyclical_encode(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    df.drop(col, axis=1, inplace=True)
    return df

def train_label_encoder(y):
  lbl_enc = LabelEncoder()
  encoded_labels = lbl_enc.fit_transform(y)
  return encoded_labels, lbl_enc

def not_train_label_encoder(y, lbl_enc=None):
  encoded_labels = lbl_enc.transform(y)
  return encoded_labels

def prepare_data(data_variables):

    X_train, y_train, X_val, y_val, X_test, y_test = data_variables
    
    y_train, lbl_enc = train_label_encoder(y_train)
    y_val = not_train_label_encoder(y_val, lbl_enc=lbl_enc)
    y_test = not_train_label_encoder(y_test, lbl_enc=lbl_enc)

    X_train_to_ohe = X_train[['category','main_category', 'currency' ]]

    X_val_to_ohe = X_val[['category','main_category', 'currency' ]]

    X_test_to_ohe = X_test[['category','main_category', 'currency' ]]

    X_train['deadline'] = pd.to_datetime(X_train['deadline'])
    X_train['year_deadline'] = X_train['deadline'].dt.year
    X_train['month_deadline'] = X_train['deadline'].dt.month
    X_train['day_deadline'] = X_train['deadline'].dt.day
    X_train.drop('deadline', axis=1, inplace=True)

    X_train['launched'] = pd.to_datetime(X_train['launched'])
    X_train['year_launched'] = X_train['launched'].dt.year
    X_train['month_launched'] = X_train['launched'].dt.month
    X_train['day_launched'] = X_train['launched'].dt.day
    X_train['hour_launched'] = X_train['launched'].dt.hour
    X_train.drop('launched', axis=1, inplace=True)

    X_val['deadline'] = pd.to_datetime(X_val['deadline'])
    X_val['year_deadline'] = X_val['deadline'].dt.year
    X_val['month_deadline'] = X_val['deadline'].dt.month
    X_val['day_deadline'] = X_val['deadline'].dt.day
    X_val.drop('deadline', axis=1, inplace=True)

    X_val['launched'] = pd.to_datetime(X_val['launched'])
    X_val['year_launched'] = X_val['launched'].dt.year
    X_val['month_launched'] = X_val['launched'].dt.month
    X_val['day_launched'] = X_val['launched'].dt.day
    X_val['hour_launched'] = X_val['launched'].dt.hour
    X_val.drop('launched', axis=1, inplace=True)

    X_test['deadline'] = pd.to_datetime(X_test['deadline'])
    X_test['year_deadline'] = X_test['deadline'].dt.year
    X_test['month_deadline'] = X_test['deadline'].dt.month
    X_test['day_deadline'] = X_test['deadline'].dt.day
    X_test.drop('deadline', axis=1, inplace=True)

    X_test['launched'] = pd.to_datetime(X_test['launched'])
    X_test['year_launched'] = X_test['launched'].dt.year
    X_test['month_launched'] = X_test['launched'].dt.month
    X_test['day_launched'] = X_test['launched'].dt.day
    X_test['hour_launched'] = X_test['launched'].dt.hour
    X_test.drop('launched', axis=1, inplace=True)

    X_train_to_cyclical = X_train[['month_deadline', 'day_deadline', 'month_launched', 'day_launched', 'hour_launched']]
    X_train_to_cyclical.head()

    X_val_to_cyclical = X_val[['month_deadline', 'day_deadline', 'month_launched', 'day_launched', 'hour_launched']]
    X_val_to_cyclical.head()

    X_test_to_cyclical = X_test[['month_deadline', 'day_deadline', 'month_launched', 'day_launched', 'hour_launched']]
    X_test_to_cyclical.head()

    X_train_ohe, ohe, std = ohe_train_encode(X_train_to_ohe)
    X_val_ohe = ohe_not_train_encode(X_val_to_ohe, ohe=ohe, std=std)
    X_test_ohe = ohe_not_train_encode(X_test_to_ohe, ohe=ohe, std=std)

    X_train_month_deadline_encoded = cyclical_encode(X_train_to_cyclical[['month_deadline']], 'month_deadline', 12)
    X_val_month_deadline_encoded = cyclical_encode(X_val_to_cyclical[['month_deadline']], 'month_deadline', 12)
    X_test_month_deadline_encoded = cyclical_encode(X_test_to_cyclical[['month_deadline']], 'month_deadline', 12)

    X_train_month_launched_encoded = cyclical_encode(X_train_to_cyclical[['month_launched']], 'month_launched', 12)
    X_val_month_launched_encoded = cyclical_encode(X_val_to_cyclical[['month_launched']], 'month_launched', 12)
    X_test_month_launched_encoded = cyclical_encode(X_test_to_cyclical[['month_launched']], 'month_launched', 12)

    X_train_day_deadline_encoded = cyclical_encode(X_train_to_cyclical[['day_deadline']], 'day_deadline', 30)
    X_val_day_deadline_encoded = cyclical_encode(X_val_to_cyclical[['day_deadline']], 'day_deadline', 30)
    X_test_day_deadline_encoded = cyclical_encode(X_test_to_cyclical[['day_deadline']], 'day_deadline', 30)

    X_train_day_launched_encoded = cyclical_encode(X_train_to_cyclical[['day_launched']], 'day_launched', 30)
    X_val_day_launched_encoded = cyclical_encode(X_val_to_cyclical[['day_launched']], 'day_launched', 30)
    X_test_day_launched_encoded = cyclical_encode(X_test_to_cyclical[['day_launched']], 'day_launched', 30)

    X_train_hour_launched_encoded = cyclical_encode(X_train_to_cyclical[['hour_launched']], 'hour_launched', 24)
    X_val_hour_launched_encoded = cyclical_encode(X_val_to_cyclical[['hour_launched']], 'hour_launched', 24)
    X_test_hour_launched_encoded = cyclical_encode(X_test_to_cyclical[['hour_launched']], 'hour_launched', 24)
    
    X_train = d = np.concatenate((X_train_ohe, X_train_month_deadline_encoded,
                              X_train_month_launched_encoded,
                              X_train_day_deadline_encoded,
                              X_train_day_launched_encoded,
                              X_train_hour_launched_encoded
                              ), axis=1)
    
    X_val = d = np.concatenate((X_val_ohe, X_val_month_deadline_encoded,
                              X_val_month_launched_encoded,
                              X_val_day_deadline_encoded,
                              X_val_day_launched_encoded,
                              X_val_hour_launched_encoded
                              ), axis=1)
    
    X_test = d = np.concatenate((X_test_ohe, X_test_month_deadline_encoded,
                              X_test_month_launched_encoded,
                              X_test_day_deadline_encoded,
                              X_test_day_launched_encoded,
                              X_test_hour_launched_encoded
                              ), axis=1)
    
    return X_train, X_val, X_test
