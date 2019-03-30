import pandas as pd
import otherdata.piiModel as piiModel
from otherdata import emailFeaturesPrepUtil
from otherdata import ipFeaturesPrepUtil
from otherdata import phoneNumPrepUtil
from otherdata import guidFeaturesPrepUtil

inputDir = 'D:\\python_examples\\MII_CLASSIF\\evaluate\\input\\'
picklePath = "D:\\python_examples\\MII_CLASSIF\\model\\"
feature_engg_data_path = "D:\\python_examples\\MII_CLASSIF\\evaluate\\train\\"
models = ['email', 'phone', 'ip', 'guid']

def buildModel(df, feature_engg_data_path, modelName):
    cols = None
    features_df = None
    cols = df.columns.values
    for col in cols:

        if('email' == modelName):
            cols, features_df = emailFeaturesPrepUtil.prepareData(df[col], feature_engg_data_path + 'Train_feature_'+modelName+'.csv', True)
        elif('phone' == modelName):
            cols, features_df = phoneNumPrepUtil.prepareData(df[col],feature_engg_data_path + 'Train_feature_' + modelName + '.csv', True)
        elif ('ip' == modelName):
            cols, features_df = ipFeaturesPrepUtil.prepareData(df[col], feature_engg_data_path + 'Train_feature_' + modelName + '.csv', True)
        elif ('guid' == modelName):
            cols, features_df = guidFeaturesPrepUtil.prepareData(df[col], feature_engg_data_path + 'Train_feature_' + modelName + '.csv', True)

        print(len(features_df))
        train, test, y, test_y = piiModel.split_data(features_df, cols)
        model = piiModel.trainModel(train, y)
        predictions = piiModel.predict(model, test)
        print(predictions)
        piiModel.test_accuracy(test_y, predictions)
        piiModel.test_confusion_matrix(predictions, test_y)
        piiModel.dump(picklePath, modelName+'_model.pkl', model)


for modelName in models:
    df = pd.read_csv(inputDir+modelName+'_data.csv')
    print(df.head())
    buildModel(df, feature_engg_data_path, modelName)

