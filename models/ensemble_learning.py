import psutil
from datetime import datetime
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def gradient_boost(X_train, X_test, y_train, y_test):

    model1 = GradientBoostingClassifier()
    model2 = RandomForestClassifier()
    ensemble_model = VotingClassifier(estimators=[('gb', model1), ('rf', model2)], voting='hard')

    start_time = datetime.now()
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    start_cpu = process.cpu_percent(interval=None)

    ensemble_model.fit(X_train, y_train)
    y_pred_ensemble = ensemble_model.predict(X_test)

    end_time = datetime.now()
    end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    end_cpu = process.cpu_percent(interval=None)
    peak_memory = max(start_memory, end_memory)

    execution_time = end_time - start_time
    memory_usage = end_memory - start_memory
    cpu_usage = end_cpu - start_cpu

    print(f"Execution time for Ensemble Learning: {execution_time} seconds")
    print(f"Memory usage for Ensemble Learning: {memory_usage:.2f} MB")
    print(f"CPU usage for Ensemble Learning: {cpu_usage}%")
    print(f"Peak memory usage during training of Ensemble Learning model: {peak_memory:.2f} MB")

    report = classification_report(y_test, y_pred_ensemble)
    print("Classification Report for Ensemble Learning:")
    print(report)