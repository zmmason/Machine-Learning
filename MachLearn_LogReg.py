# !/usr/bin/env python3
# BME 205: Classification of Cancer Cells (malignant/benign)
# Name: Zachary Mason (zmmason@ucsc.edu)
# Univ: University of California, Santa Cruz
# Dept: Biomolecular Engineering & Bioinformatics

from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt


class DataLoad:
    """load data"""
    def __init__(self, csv):
        """ CommandLine constructor: Implements a parser to interpret the command line argv string using argparse. """
        self.csv = csv

    def load(self):
        """ Load the data into a Pandas data frame and read input. """
        df = pd.read_csv(self.csv)  # assuming dataset loaded into directory
        # Preprocess the data so we can use it for regression with SKLearn.
        encoder = preprocessing.LabelEncoder()
        for col in df.columns:  # For each column in the data frame
            df[col] = encoder.fit_transform(df[col])  # Transform the series so is zero based
        df.head()  # prints data
        return df


class LogRegression:
    """ Train and test a logistic Regression model with CSV data to predict benign or
    malignant cancer in patient trials. """
    def __init__(self, df):
        """ CommandLine constructor: Implements a parser to interpret the command line argv string using argparse. """
        self.df = df
        self.features = []

    def var_select(self):
        # variable selection while looping through features and fitting model for score (omitting 'id' and 'class')
        # features sorted based on score
        self.features = list(self.df.keys())
        self.features.remove('id')
        self.features.remove('class')
        # features.remove('uniformity-of-cell-size')  # explained in the conclusion section for features
        feature_scores = []  # list to hold features and their associated fit scores
        for feature in self.features:
            x = self.df[[feature]]
            y = self.df['class']
            logistic_model = LogisticRegression(fit_intercept=True, solver='liblinear')
            logistic_model.fit(x, y)
            feature_scores.append([feature, logistic_model.score(x, y)])
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        for feature_score in feature_scores:
            # print var and corresponding score
            print(feature_score)
        print()

    def logReg_score(self):
        """ Print logistic model score. """
        x = self.df[self.features]
        y = self.df['class']
        logistic_model = LogisticRegression(fit_intercept=True, solver='lbfgs')
        logistic_model.fit(x, y)
        print("Score: ", logistic_model.score(x, y))
        self.classification(logistic_model)

    def classification(self, logistic_model):
        """ Visualize distribution to obtain threshold for classification. """
        input_0 = self.df.loc[self.df['class'] == 0]
        y_0 = input_0['class']
        x_0 = input_0[self.features]

        input_1 = self.df.loc[self.df['class'] == 1]
        y_1 = input_1['class']
        x_1 = input_1[self.features]
        preds_0 = logistic_model.predict(x_0)
        preds_1 = logistic_model.predict(x_1)
        n, bins, patches = plt.hist(preds_0, bins=10, density=1, cumulative=0)
        plt.title('Predictive distribution for class y=0')
        plt.show()
        n, bins, patches = plt.hist(preds_1, bins=10, density=1, cumulative=0)
        plt.title('Predictive distribution for class y=1')
        plt.show()
        self.split(logistic_model)

    def split(self, logistic_model):
        """ split into train and test data to prevent overfitting. """
        (train, test) = train_test_split(self.df, test_size=0.35, random_state=0)  # data split in 65-35 for training-testing
        train_output = train['class']
        train_input = train[self.features]
        test_output = test['class']
        test_input = test[self.features]
        # Choosing a cutoff to look at accuracy of the logistic regression model classification
        logistic_model.fit(train_input, train_output)
        print("Score: ", logistic_model.score(train_input, train_output))
        self.accuracy(logistic_model, test_input, test_output)

    def accuracy(self, logistic_model, test_input, test_output):
        """"""
        y_pred = logistic_model.predict(test_input)
        y_pred = [1 if p > 0.45 else 0 for p in y_pred]  # cuttof chosen based on distribution
        print('Accuracy', accuracy_score(test_output, y_pred), '\n')

        # Sensitivity, specificity
        print(confusion_matrix(test_output, y_pred))
        tn, fp, fn, tp = confusion_matrix(test_output, y_pred).ravel()
        fpr = fp * 1.0 / (fp + tn)  # Fraction of true negative cases with a positive prediction
        fnr = fn * 1.0 / (tp + fn)  # Fraction of true positive cases with a negative prediction
        print("False positive rate: (predicting malignant while benign)", fpr)
        print("False negative rate: (predicting benign while malignant)", fnr, '\n')

        # Plot ROC curve - shows relationship between training and obtaining accurate binary outcomes
        y_pred_prob = logistic_model.predict_log_proba(test_input)[:, 1]  # probability predicted label is 1
        fpr, tpr, thresholds = roc_curve(test_output, y_pred_prob)
        plt.plot([0, 1], [0, 1])
        plt.plot(fpr, tpr, label='Logistic Regression')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Logistic Regression ROC Curve')
        plt.show()
        self.contributions(logistic_model)

    def contributions(self, logistic_model):
        """ List of coefficients to determine feature contribution. """
        coef_dict = {}
        for coef, feat in zip(logistic_model.coef_[0, :], self.features):
            coef_dict[feat] = coef

        top_feature = max(coef_dict.items(), key=lambda item: item[1])
        print('Largest contribution feature:\n{}, {}'.format(top_feature[0], top_feature[1]))

        plt.bar(range(len(coef_dict)), list(coef_dict.values()), align='center')
        plt.xticks(range(len(coef_dict)), list(coef_dict.keys()))
        plt.title('Regression Coefficients')
        ax = plt.gca()
        plt.xticks(rotation=90)
        for label in ax.get_xaxis().get_ticklabels():
            label.set_visible(True)
        plt.show()


def main():

    """
    This program is designed to take an input .csv data file containing features and outcomes of
    cancer patients and train and test a logstic model to produce binary predictions (classifications) as to the
    probability of a patient having benign or malignant cancer cells. Accuracy and false positive
    / false negative rates will be determined. A plot showing the logistic regression ROC curve
    for accuracy will also be presented. Feature contribution will be analyzed and plotted to
    show the impact of each feature on predictions.
    """
    csv = 'breast-cancer-wisconsin.data.csv'
    print(csv)
    data = DataLoad(csv)
    df = data.load()
    logReg = LogRegression(df)
    logReg.var_select()
    logReg.logReg_score()


if __name__ == "__main__":
    main()
