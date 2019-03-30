import pandas as pd

input_data_columns=['phone_num','other']
output_field = ['field_len','is_only_number','count_after_first_hyphen']


def removeHyphen(data):
    return data.replace("-","")

def getTokens(data):
    return data.split('-')

def get_field_len(data):
    return len(removeHyphen(data))

def is_only_number(data):
    if (removeHyphen(data)).isdigit():
        return 1
    else:
        return 0

def count_after_first_hyphen(data):
    tokens = getTokens(data)
    if(len(tokens)) == 3 :
        return len(tokens[1])
    else:
        return 0

def is_num_in_range(val):
    if( val >= 8 and val <= 12):
        return 1
    return 0



def prepareData(feature_data, feature_engg_data_path, is_train):
    total_row_list = list()
    if is_train:
        output_field.append("label")
    feature_data = feature_data[feature_data.notnull()]
    for colData in feature_data:
        each_row = generate_feature_list(colData)
        if is_train:
            if(is_num_in_range(each_row[0]) and each_row[1] == 1 and (each_row[2] == 3 or each_row[2] == 0)):
                each_row.append('phone_num')
            else:
                each_row.append("other")
        total_row_list.append(each_row)
    print(total_row_list)
    writeToFile(total_row_list, feature_engg_data_path)
    features_df = pd.DataFrame(total_row_list, columns=output_field)
    return input_data_columns, features_df

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

def generate_feature_list(colData):
    data = str(colData)
    return [get_field_len(data),
            is_only_number(data),
            count_after_first_hyphen(data)]


def writeToFile(total_row_list, feature_engg_data_path):
    file_object = open(feature_engg_data_path, "w")

    file_object.write(','.join(str(colVal) for colVal in output_field) + '\n')
    for item in total_row_list:
        file_object.write(','.join(str(colVal) for colVal in item) + '\n')
    file_object.close()

def main():
    df = pd.read_csv("D:\\python_examples\\PII_DATA\\files\\phone_num_data.csv")
    print(df)
    prepareData(df)

if __name__ == '__main__':
        main()