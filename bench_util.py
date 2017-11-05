from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import time
import numpy as np
import os

def calcFscore(df):
    Fscore = {}
    cs = df.shape[1]
    for c in df.columns:
        if c == 'label':
            continue
        df1, df0 = df[df['label']==1], df[df['label']==0]
        m1, m0 = df1[c].mean(), df0[c].mean()
        v1, v0 = df1[c].var(), df0[c].var()
        Fscore[c]=(m1-m0)**2/(df1.shape[0]*v1+df0.shape[0]*v0)
    return Fscore
######### preprocessor ############
import re
from bs4 import BeautifulSoup

def preprocessor(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text.replace('&nbsp;', ' ')
    emoticons = re.findall('(?::|;|=|X)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
    return text
############ tokenizer ############
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
########### stop-words ############
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
eng_stop = stopwords.words('english')

def tokenizer_stem_nostop(text):
    porter = PorterStemmer()
    return [w for w in text.split(' ') if w not in eng_stop]
########### SGD ##############
def get_stream(path, size):
    for chunk in pd.read_csv(path, chunksize=size):
        yield chunk