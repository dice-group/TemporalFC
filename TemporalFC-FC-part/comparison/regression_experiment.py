from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import classification_report
# from java.lang import sys
# from data import Data
# from data import Data


def load_train_data(path_dataset_folder="", path=None):
    if path == None:
        train_folder = ""
        test_folder = ""
    else:
        train_folder = "train/"+path
        test_folder = "test/" + path

    df_train_set = pd.read_csv(path_dataset_folder +train_folder+ 'trainCombinedEmbeddings.csv')
    y_train = df_train_set.iloc[:,-1:]

    leng = len(df_train_set.columns)
    x_train = df_train_set.iloc[:, :leng-1]
    df_test_set = pd.read_csv(path_dataset_folder +test_folder+ 'testCombinedEmbeddings.csv')
    y_test = df_test_set.iloc[:,-1:]
    leng = len(df_test_set.columns)
    x_test = df_test_set.iloc[:, :leng-1]


    return x_train,y_train, x_test, y_test


def appplyLogisticRegression ( dataset_path, path=None):
    X_train, y_train, X_test, y_test = load_train_data(dataset_path,path)
    print("starting regression experiments")
    # Create Logistic Regression classifer object
    clf = LogisticRegression(random_state=0).fit(X_train, y_train.values[:, 0])

    print('on train data')
    # Train Decision Tree Classifer
    print(accuracy_score(y_test, clf.predict(X_test)))
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print(classification_report(y_test.values[:, 0], y_pred))

    print('on test data')
    # Train Decision Tree Classifer
    print(accuracy_score(y_train, clf.predict(X_train)))
    # Predict the response for test dataset
    y_pred = clf.predict(X_train)

    print(classification_report(y_train.values[:, 0], y_pred))


def applyJ48 (dataset_path,path=None):
    X_train, y_train, X_test, y_test = load_train_data(dataset_path,path)
    print("starting J48 experiments")
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))





# delchars = "".join(c for c in map(chr, range(sys.maxunicode + 1)) if not c.isdecimal() or not c.isspace())

# X_train, y_train =
# X_test, y_test  = "date/",
# path_dataset_folder = 'dataset/'
path_dataset_folder = './dataset/data/'
# datasets_class = ["domain/","domainrange/","mix/","property/","random/","range/"]
datasets_class = ["domain/"]

# appplyLogisticRegression(path_dataset_folder)
for line2 in datasets_class:
    print(line2+" experiments")
    # dataset = Data(data_dir=path_dataset_folder, subpath=line2)
    appplyLogisticRegression(path_dataset_folder, line2)
