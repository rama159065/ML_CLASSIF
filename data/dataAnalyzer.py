import pandas as pd
from collections import Counter

def occurrence(ouput):
    for word, word_count in Counter(ouput).most_common(1):
        return word

def getColsDict(df):
    dict = {}
    for col in df:
        index = df.columns.get_loc(col)
        dict[col] = index
    return dict

def getColsIndexDict(df):
    dict = {}
    for col in df:
        index = df.columns.get_loc(col)
        dict[index] = col
    return dict

def splitData(df):
    charCols = []
    otherCols = []
    subDF = df.iloc[0:5]

    for key, value in subDF.iteritems():
        for val in value:
            boolList = []
            if (str(val).replace(" ", "").isalpha()):
                boolList.append(True)
            else:
                boolList.append(False)
        if(occurrence(boolList)):
            charCols.append(key)
        else:
            otherCols.append(key)

    print(otherCols)

    otherDF = df[otherCols]

    charDF = df[charCols]
    return (charCols, otherCols, charDF, otherDF)

