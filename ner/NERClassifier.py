from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import pandas as pd
from collections import Counter

class NERClassifier :
    model = 'D:\\python_examples\\nlp_library\\english.all.3class.caseless.distsim.crf.ser.gz'
    st = StanfordNERTagger(model,
                           'D:\\python_examples\\nlp_library\\stanford-ner-3.9.2.jar',
                           encoding='utf-8')
    ner_classes = ['LOCATION', 'PERSON', 'ORGANIZATION']
    output_field = ['TEXT','NER_CLASS']

    def predict(self, inputDF):
        cols = inputDF.columns.values
        finalData = {}
        for eachColName in cols:
            feature_data = inputDF[eachColName]
            feature_data = feature_data[feature_data.notnull()]
            length = len(feature_data)
            output = []
            for text in feature_data:
                text_tokenize = word_tokenize(text)
                classified_text = self.st.tag(text_tokenize)
                print(classified_text)
                row = self.evaluates(classified_text)
                output.append(row)
            finalData[eachColName] = self.occurrence(output)
        print(finalData)
        return finalData

    def evaluates(self, classified_text):
        for tuple in classified_text:
            if tuple[1] in self.ner_classes:
                return tuple[1]
            else:
                return 'NA'

    def occurrence(self, ouput):
        for word, word_count in Counter(ouput).most_common(1):
            return word



