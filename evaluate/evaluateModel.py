import otherdata.piiModel as piiModel
from data import dataAnalyzer
import pandas as pd
from otherdata import emailFeaturesPrepUtil
from otherdata import ipFeaturesPrepUtil
from otherdata import phoneNumPrepUtil
from otherdata import guidFeaturesPrepUtil
from collections import Counter
from ner.NERClassifier import NERClassifier

filePath = "D:\\python_examples\\MII_CLASSIF\\model\\"
#testFile = 'D:\\python_examples\\MII_CLASSIF\\evaluate\\test_test.csv'
#testFile = 'D:\\python_examples\\MII_CLASSIF\\evaluate\\all_data.csv'
testFile = 'D:\\python_examples\\MII_CLASSIF\\evaluate\\test_new.csv'
feature_engg_data_path = "D:\\python_examples\\MII_CLASSIF\\evaluate\\test\\"

modelDict = {}

def occurrence(ouput):
    for key, word_count in Counter(ouput).most_common(1):
        return key


df = pd.read_csv(testFile, delimiter='~', nrows=5)
vals = None
if(len(df) == 5):
    vals = 5
else:
    vals = len(df)


charCols, otherCols, charDF, otherDF = dataAnalyzer.splitData(df)

nerClassif = NERClassifier()
nerDict = nerClassif.predict(charDF)
print("predictions are", nerDict)


def checkPred(data, modelName):
    new = []
    for i in range(0, len(data), vals):

        new.append(data[i : i+vals])

    for i, e in enumerate(new):
        if occurrence(e) == 0:
            modelDict[otherCols[i]] = modelName


email_features_df = emailFeaturesPrepUtil.prepareTestData(otherDF,
                                                      feature_engg_data_path + 'feature_test_email.csv')
phone_features_df = phoneNumPrepUtil.prepareTestData(otherDF,
                                                 feature_engg_data_path + 'feature_test_phone.csv')
ip_features_df = ipFeaturesPrepUtil.prepareTestData(otherDF,
                                                   feature_engg_data_path + 'feature_test_ip.csv')
guid_features_df = guidFeaturesPrepUtil.prepareTestData(otherDF,
                                                     feature_engg_data_path + 'feature_test_guid.csv')

emailModel = piiModel.load(filePath, 'email_model.pkl')
guidModel = piiModel.load(filePath, 'guid_model.pkl')
ipModel = piiModel.load(filePath, 'ip_model.pkl')
phoneModel = piiModel.load(filePath, 'phone_model.pkl')

predictions_email = piiModel.predict(emailModel, email_features_df)
print("email predictions")
print(predictions_email)

predictions_guid = piiModel.predict(guidModel, guid_features_df)
print("guid predictions")
print(predictions_guid)

predictions_ip = piiModel.predict(ipModel, ip_features_df)
print("ip predictions")
print(predictions_ip)

predictions_phone = piiModel.predict(phoneModel, phone_features_df)
print("email predictions")
print(predictions_phone)

checkPred(predictions_guid, "guid")
checkPred(predictions_email, "email")
checkPred(predictions_ip, "ip")
checkPred(predictions_phone, "phone_num")

dict = dataAnalyzer.getColsDict(df)
print(dict)

modelDict.update(nerDict)

finalDict = {}

for key, value in dict.items():
    if (modelDict.get(key) == None):
        finalDict[value] = 'UNKNOWN'
    else:
        finalDict[value] = modelDict.get(key)

print("finalDict Dictionary is", finalDict)


(pd.DataFrame.from_dict(data=finalDict, orient='index')
     .to_csv('D:\\python_examples\\MII_CLASSIF\\evaluate\\evaluation.csv', header=False))

