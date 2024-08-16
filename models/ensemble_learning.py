from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def gradient_boost(X_train, X_test, y_train, y_test):
    # model = GradientBoostingClassifier()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    model1 = GradientBoostingClassifier()
    model2 = RandomForestClassifier()

    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    ensemble_model = VotingClassifier(estimators=[('gb', model1), ('rf', model2)], voting='hard')
    ensemble_model.fit(X_train, y_train)
    y_pred_ensemble = ensemble_model.predict(X_test)

    report = classification_report(y_test, y_pred_ensemble)
    print("Classification Report for Ensemble Learning:")
    print(report)