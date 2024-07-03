from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def generate_results(y_true, y_pred, labels):
    generate_classification_report(y_true, y_pred)
    generate_confusion_matrix(y_true, y_pred, labels)


def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)


def generate_confusion_matrix(y_true, y_pred, labels):
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
