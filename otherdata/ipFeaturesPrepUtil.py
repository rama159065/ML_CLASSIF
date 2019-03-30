import re
import pandas as pd

input_data_columns=['ip','other']
output_field = ['field_len','how_many_dot','is_number','is_ip_in_range']
#feature_engg_data_path = "D:\\python_examples\\PII_DATA\\files\\features_ip.csv"

def getTokens(data):
    return data.split('.')

def removeDot(data):
    return data.replace(".","")

def get_field_length(data):
    try :
        return len(removeDot(data))
    except:
        return 0

def how_many_dot(data):
    return data.count('.')

def is_numeric(data):
    tokens = getTokens(data)
    fields = [not token.isdigit() for token in tokens]
    if(any(fields)):
        return 0
    return 1

def is_ip_in_range(data):
    tokens = getTokens(data)
    fields = []
    for token in tokens:
        try:
            if(int(token) > 255 and int(token < 0)):
                fields.append(token)
        except:
            return 0
    if (any(fields)):
        return 0
    return 1

def is_ip_len_in_range(val):
    if( val >= 4 and val <= 12):
        return 1
    return 0


def prepareData(feature_data,feature_engg_data_path, is_train):
    total_row_list = list()
    if is_train:
        output_field.append("label")
    feature_data = feature_data[feature_data.notnull()]
    for colData in feature_data:
        each_row = generate_feature_list(colData)
        if(is_train):
            if(is_ip_len_in_range(each_row[0]) and each_row[1] == 3  and each_row[2] == 1 and each_row[3] == 1):
                each_row.append('ip')
            else:
                each_row.append("other")
        total_row_list.append(each_row)
    print(total_row_list)
    writeToFile(total_row_list, feature_engg_data_path)
    features_df = pd.DataFrame(total_row_list, columns=output_field)
    return input_data_columns, features_df

def generate_feature_list(colData):
    data = str(colData)
    return [get_field_length(data),
            how_many_dot(data),
            is_numeric(data),
            is_ip_in_range(data)]

def prepareTestData(df, feature_engg_data_path):
    cols = df.columns.values
    total_row_list = list()
    for eachColName in cols:
        feature_data = df[eachColName]
        for colData in feature_data:
            each_row = generate_feature_list(colData)
            total_row_list.append(each_row)
    writeToFile(total_row_list, feature_engg_data_path)
    features_df = pd.DataFrame(total_row_list, columns=output_field)
    return features_df


def writeToFile(total_row_list, feature_engg_data_path):
    file_object = open(feature_engg_data_path, "w")

    file_object.write(','.join(str(colVal) for colVal in output_field) + '\n')
    for item in total_row_list:
        file_object.write(','.join(str(colVal) for colVal in item) + '\n')
    file_object.close()

def main():
    df = pd.read_csv("D:\\python_examples\\PII_DATA\\files\\ip_address_data.csv")
    print(df)
    prepareData(df)

if __name__ == '__main__':
        main()