from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def naive_bayes(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report for Naive Bayes Classifier:")
    print(report)