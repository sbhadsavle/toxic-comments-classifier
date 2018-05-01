from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def classification_report(y_predicted, y_test, output_filename):
    with open(output_filename, "w+") as f:
        f.write(metrics.classification_report(y_predicted, y_test))

def confusion_matrix(y_predicted, y_test, output_filename):
    mat = confusion_matrix(y_predicted, y_test)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig(output_filename)
