import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import skimage

from PIL import Image
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn import datasets, metrics, svm
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


class ImageQualityClassifier:

    def __init__(self):
        self.model = SVC(kernel='rbf', class_weight = 'balanced', probability = True)

    def loadModel(self, filename):
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)

    def saveModel(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def get_image(self, row, root = "datasets/"):
        filename = "{}".format(row)
        file_path = os.path.join(root, filename)
        img = Image.open(file_path)
        return np.array(img)

    def create_features(self, img):
        color_features = img.flatten()
        gray_image = rgb2gray(img)
        hog_features = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(16,16))
        flat_features = np.hstack(color_features)
        return flat_features

    def create_feature_matrix(self, label_dataframe, root):
        features_list = []
        
        for img_id in label_dataframe.index:
            # load image
            img = self.get_image(img_id, root=root)
            # get features for image
            image_features = self.create_features(img)
            features_list.append(image_features)
            
        # convert list of arrays into a matrix
        feature_matrix = np.array(features_list)
        return feature_matrix


    def run_PCA(self, feature_matrix, component_num):
        ss = StandardScaler()
        quality_stand = ss.fit_transform(feature_matrix)
        pca = PCA(n_components = component_num)
        quality_pca = pca.fit_transform(quality_stand)
        var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
        print(pca.components_)
        print(pca.explained_variance_ratio_)
        plt.plot(var1)

        return quality_pca

    def test_train(self, pca_data, data_labels):
        X = pd.DataFrame(pca_data)
        y = pd.Series(data_labels.values)

        skf = StratifiedKFold(n_splits = 5)

        all_feature_importances = np.zeros((len(set(y)), len(X.columns)))
        all_labels = []
        all_predictions = []
        accuracy = 0

        for train_index, test_index in skf.split(X.values, y):
            X_train, y_train = X.values[train_index], y[train_index]
            X_test, y_test = X.values[test_index], y[test_index]
            self.model.fit(X_train, y_train)

            # all_feature_importances += model.coef_
            predictions = self.model.predict(X_test)
            all_predictions.extend(predictions)
            all_labels.extend(y_test)
            accuracy += accuracy_score(predictions, y_test)

        probabilities = self.model.predict_proba(X_test)
        return accuracy/5, all_labels, all_predictions, probabilities, y_test

    def user_run_data(self, pca_data):
        X = pd.DataFrame(pca_data)

        # all_feature_importances = np.zeros((len(set(y)), len(X.columns)))
        all_predictions = []

        predictions = self.model.predict(X.values)
        all_predictions.extend(predictions)

        return all_predictions


    def get_confusion_matrix(self, all_labels, all_predictions):
        cm = confusion_matrix(all_labels, all_predictions, normalize='true')
        df_cm = pd.DataFrame(cm, index=[0,1], columns=[0,1])
        return sns.heatmap(df_cm, annot=True)

    def plot_AUC(self, y_test, probabilities):
        y_proba = probabilities[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=1)
        roc_auc = auc(fpr, tpr)

        plt.title("Receiver Operating Characteristic")
        roc_plot = plt.plot(fpr, tpr, label="AUC = {:0.2f}".format(roc_auc))
        plt.legend(loc=0)
        plt.plot([0,1], [0,1], ls="--")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")

    def run_and_train(self, root, csv_path, component_num):
        labels = pd.read_csv(root+csv_path, index_col=0)
        # run create_feature_matrix on our dataframe of images
        feature_matrix = self.create_feature_matrix(labels, root)
        quality_pca = self.run_PCA(feature_matrix, component_num)
        accuracy, all_labels, all_predictions, probabilities, y_test = self.test_train(quality_pca, labels.num_quality)
        plt.subplot(1,2,1)
        cm = self.get_confusion_matrix(all_labels, all_predictions)
        plt.subplot(1,2,2)
        self.plot_AUC(y_test, probabilities)
        print("Accuracy: {}".format(accuracy))
        print("Annotated values: {}".format(all_labels)) 
        print("Predicted values: {}".format(all_predictions))
        plt.show()

    def run(self, root, csv_path, component_num):
        labels = pd.read_csv(root+csv_path, index_col=0)
        # run create_feature_matrix on our dataframe of images
        feature_matrix = self.create_feature_matrix(labels, root)
        quality_pca = self.run_PCA(feature_matrix, component_num)
        all_predictions = self.user_run_data(quality_pca)
        labeled_predictions = list(zip(labels.index, all_predictions))
        print("Predicted values:", *labeled_predictions, sep='\n  ')

classifier = ImageQualityClassifier()
#to make predictions for new data, load the model then run
# classifier.loadModel('iqcModel.pickle')
# # run expects 3 arguments,
# #   the root folder (with trailing slash),
# #   a csv listing all the image file names,
# #   and the number of components
# classifier.run("datasets/new/", "filenames.csv", 125)

classifier.run_and_train("datasets/", "labels.csv", 125)
classifier.saveModel('iqcModel.pickle')

