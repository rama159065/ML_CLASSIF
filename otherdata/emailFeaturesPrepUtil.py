import pandas as pd

input_data_columns=['email_id','other']
output_field = ['has_at','how_many_at','before_at_any_char','aft_at_any_char','has_dot_after_at','end_with_alpha_char']


def has_at(data):
    if('@' in data):
        return 1
    return 0


def how_many_at(data):
    try:
        return data.count('@')
    except:
        return 0

def before_at_any_char(data):
    try:
        if(data.index('@') > 0):
            if(data[data.index('@') + 1].isalnum()):
                return 1
            else:
                return 0
    except:
        return 0


def after_at_any_char(data):
    try:
        if(data.index('@') > 0):
            if(data[data.index('@') + 1].isalpha()):
                return 1
            else:
                return 0
    except:
        return 0


def has_dot_after_at(data):
    try:
        if (data.index('@') > 0):
            subset_aft_at = data[data.index('@'): len(data)]
            if dot_count(subset_aft_at) > 0:
                return 1
            else:
                return 0
    except:
        return 0

def dot_count(data):
    try:
        return data.count('.')
    except:
        return 0

def end_with_alpha_char(data):
    if (len(data) > 0):
        if(data[-1].isalpha()):
            return 1
    return 0

def prepareData(feature_data, feature_engg_data_path, is_train):
    total_row_list = list()
    if (is_train):
        output_field.append("label")
    feature_data = feature_data[feature_data.notnull()]
    for colData in feature_data:
        each_row = generate_feature_list(colData)
        if is_train :
            if(each_row == [1,1,1,1,1,1]):
                each_row.append('email_id')
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
    return [has_at(data),
            how_many_at(data),
            before_at_any_char(data),
            after_at_any_char(data),
            has_dot_after_at(data),
            end_with_alpha_char(data)]


def writeToFile(total_row_list, feature_engg_data_path):
    file_object = open(feature_engg_data_path, "w")

    file_object.write(','.join(str(colVal) for colVal in output_field) + '\n')
    for item in total_row_list:
        file_object.write(','.join(str(colVal) for colVal in item) + '\n')
    file_object.close()

