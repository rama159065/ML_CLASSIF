import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def split_data(input_df, input_data_columns):
    np.random.seed(0)
    Y = pd.factorize(input_df['label'])[0]
    print(Y)
    input_df['label'] = pd.Categorical.from_codes(Y, input_data_columns)

    input_df['is_train'] = np.random.uniform(0, 1, len(input_df)) <= .60
    train, test = input_df[input_df['is_train'] == True], input_df[input_df['is_train'] == False]

    y = pd.factorize(train['label'])[0]
    test_y = pd.factorize(test['label'])[0]
    train.drop(['is_train', 'label'], axis=1, inplace=True)
    test.drop(['is_train', 'label'], axis=1, inplace=True)

    return (train, test, y, test_y)

def trainModel(trainX, target):

    clf = RandomForestClassifier(n_jobs=3, criterion='entropy', random_state=0)
    clf.fit(trainX, target)
    return clf

def dump(filePath, fileName, model):
    rf_model_pkl = open(filePath + fileName, 'wb')
    pickle.dump(model, rf_model_pkl)
    rf_model_pkl.close()
    return

def load(filePath, fileName):
    model_pkl = open(filePath + fileName, 'rb')
    model = pickle.load(model_pkl)
    print("Loaded model :: ", model)
    return model

def test_accuracy(predictions, test_y):
        print("Model Accuracy is : ", accuracy_score(test_y, predictions))

def test_confusion_matrix(predictions, test_y):
    print("confusion matrix is : \n", confusion_matrix(test_y, predictions))

def predict(model, input):
    predictions = model.predict(input)
    return predictions

def predict_proba(model, input):
    predict_proba = model.predict_proba(input)
    return predict_proba