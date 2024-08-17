import psutil
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def naive_bayes(X_train, X_test, y_train, y_test):
    model = GaussianNB()

    start_time = datetime.now()
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    start_cpu = process.cpu_percent(interval=None)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end_time = datetime.now()
    end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    end_cpu = process.cpu_percent(interval=None)
    peak_memory = max(start_memory, end_memory)

    execution_time = end_time - start_time
    memory_usage = end_memory - start_memory
    cpu_usage = end_cpu - start_cpu

    print(f"Execution time for Naive Bayes Classifier: {execution_time} seconds")
    print(f"Memory usage for Naive Bayes Classifier: {memory_usage:.2f} MB")
    print(f"CPU usage for Naive Bayes Classifier: {cpu_usage}%")
    print(f"Peak memory usage during training of Naive Bayes Classifier: {peak_memory:.2f} MB")

    report = classification_report(y_test, y_pred)
    print("Classification Report for Naive Bayes Classifier:")
    print(report)