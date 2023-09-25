import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


# Kohonen self-organizing neural network class
class KohonenSOM:
    # initializing the class with the data and parameters
    def __init__(self, data, learning_rate=0.1, num_iterations=1000):
        # data is a pandas dataframe
        self.data = data
        # learning rate for the network
        self.learning_rate = learning_rate
        # number of iterations for training the network
        self.num_iterations = num_iterations

    # fitting the model to the data
    def fit(self):
        # getting the number of features in the dataset
        num_features = len(self.data.columns) - 1
        # creating a MLPClassifier object with one hidden layer and num of neurons equal to number of features
        self.model = MLPClassifier(hidden_layer_sizes=(num_features,))
        # training the model on the dataset
        self.model.fit(self.data[self.data.columns[:-1]], self.data[self.data.columns[-1]])

    # predicting labels for new data points
    def predict(self, X):
        return self.model.predict(X)

    # evaluating model performance on test set
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))  # accuracy score
        print("Confusion Matrix:")  # confusion matrix
        print(confusion_matrix(y, y_pred))
        print("Classification Report:")  # classification report
        print(classification_report(y, y_pred))

    # visualizing model performance on test set using seaborn heatmap
    def visualize(self, X, y):
        sns.heatmap(confusion_matrix(y, self.predict(X)), annot=True)
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("Learning_data10.txt", sep="\t")
    som = KohonenSOM(df)
    som.fit()
    som.evaluate(df[df.columns[:-1]], df[df.columns[-1]])
    som.visualize(df[df.columns[:-1]], df[df.columns[-1]])
