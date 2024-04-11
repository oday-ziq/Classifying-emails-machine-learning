import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial import distance

TEST_SIZE = 0.3
K = 3

class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        predictions = []
        for test_point in features:
            # Calculate distances from the test point to all other points
            distances = np.array([distance.euclidean(test_point, train_point) for train_point in self.training_features])
            k_nearest_indices = distances.argsort()[:k]
            k_nearest_labels = [self.training_labels[i] for i in k_nearest_indices]
            # Choose the most common label among the k nearest neighbors
            majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(majority_vote)
        return predictions


def load_data(filename):
    if not isinstance(filename, str):
        raise ValueError('Filename must be a string.')
    
    # Check the file exists and is a .csv file
    if not (os.path.isfile(filename) and filename.endswith('.csv')):
        raise FileNotFoundError('The specified file does not exist or is not a .csv file.')
    
    try:
        df = pd.read_csv(filename, header=None)
    except pd.errors.ParserError:
        raise ValueError('The specified file could not be parsed as csv.')
    
    labels = df.iloc[:, -1]
    features = df.iloc[:, :-1]
    
    return features, labels


def preprocess(features):
    if not isinstance(features, (np.ndarray, pd.DataFrame)):
        raise TypeError("Input features must be either a numpy array or pandas DataFrame.")

    scaler = StandardScaler()

    # Fit and transform the data to the standard scale
    try:
        processed_features = scaler.fit_transform(features)
    except ValueError as ve:
        raise ValueError("StandardScaler failed to fit and transform the input features.") from ve

    return processed_features


def train_multilayer_perceptron(features: Union[np.ndarray, pd.DataFrame], 
                                labels: Union[np.ndarray, pd.Series]) -> MLPClassifier:
    
    if not isinstance(features, (np.ndarray, pd.DataFrame)):
        raise TypeError("Input features must be either a numpy array or pandas DataFrame.")

    if not isinstance(labels, (np.ndarray, pd.Series)):
        raise TypeError("Labels must be either a numpy array or pandas Series.")

    # Instantiate the MLPClassifier model
    model = MLPClassifier(hidden_layer_sizes=(10,5), activation='logistic', max_iter=10000)

    # Fit the model to the data
    try:
        model.fit(features, labels)
    except ValueError as ve:
        raise ValueError("Failed to fit the MLP model to the input data.") from ve

    return model


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Union

def evaluate(labels: Union[list, np.ndarray], 
             predictions: Union[list, np.ndarray]) -> tuple:
    
    if not isinstance(labels, (list, np.ndarray)):
        raise TypeError("The 'labels' parameter must be a list or numpy array.")

    if not isinstance(predictions, (list, np.ndarray)):
        raise TypeError("The 'predictions' parameter must be a list or numpy array.")

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return accuracy, precision, recall, f1



def main():
    # Set the path to your CSV file
    csv_path = 'E:\\project 2\\spambase.csv'

    features, labels = load_data(csv_path)
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    print("**** 1-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

if __name__ == "__main__":
    main()
