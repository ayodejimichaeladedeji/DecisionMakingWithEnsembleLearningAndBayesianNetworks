import pandas as pd
from collections import Counter
from models.naive_bayes import naive_bayes
from models.ensemble_learning import gradient_boost
from preprocessing.preprocessor import encoder, split_train_test, data_balance

transactions_df = pd.read_csv('transactions.csv')

print(transactions_df.shape)
print(Counter(transactions_df['isFraud']))

new_transactions_df = transactions_df.head(500000)

print(new_transactions_df.isna().sum())
print(Counter(new_transactions_df['isFraud']))

X = new_transactions_df.drop(columns=['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud', 'isFraud'])
y = new_transactions_df['isFraud']

X_encoded = encoder(X)

X_res, y_res = data_balance(X_encoded, y)

X_train, X_test, y_train, y_test = split_train_test(X_res, y_res)

naive_bayes(X_train, X_test, y_train, y_test)

gradient_boost(X_train, X_test, y_train, y_test)
