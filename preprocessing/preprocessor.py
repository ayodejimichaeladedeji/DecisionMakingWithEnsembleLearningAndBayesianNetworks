from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encoder(X):
    le = LabelEncoder()

    X['type'] = le.fit_transform(X['type'])
    X['nameOrig'] = le.fit_transform(X['nameOrig'])
    X['nameDest'] = le.fit_transform(X['nameDest'])

    return X

def data_balance(X, y):
    smote = SMOTE(random_state=42)

    X_res, y_res = smote.fit_resample(X, y)

    print(Counter(y_res))

    return X_res, y_res

def split_train_test(X_res, y_res):
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test