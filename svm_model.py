import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler
import xgboost as xgb
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os
import nltk
import gensim
from nltk.stem.snowball import SnowballStemmer
from nltk import tokenize
from nltk.corpus import stopwords
import nltk


class StemmedCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        stemmer = SnowballStemmer("english")
        stops = set(stopwords.words("english"))
        return lambda doc: ([stemmer.stem(w) for w in analyzer(re.sub("[^\w\s]", "", doc)) if w not in stops])


def load_n_prep(name):
    df = pd.read_json(os.path.join('{}.json'.format(name)))
    df.set_index('id', inplace=True)
    df['ingredients_count'] = df['ingredients'].apply(lambda x: len(x))
    df['ingredients_word_count'] = df['ingredients'].apply(lambda ingredients: [len(i.split()) for i in ingredients])
    df['ingredients'] = df['ingredients'].apply(lambda ingredients: ' '.join(ingredients)) #ingredients of two words can get lost
    return df


train = load_n_prep('train')



# out_index = train[train.ingredients_count > 40].index
X = train.ingredients  # .drop(out_index)
y = train.cuisine  # .drop(out_index)

model = Pipeline([
    ('bag_of_words', StemmedCountVectorizer(ngram_range=(1, 2), stop_words='english')),

    # ('logreg', LogisticRegression(penalty='l1', C=10, max_iter=10000))
    #     ('svm', OneVsRestClassifier(SVC(C=100, coef0=1)))
    # ('svm', SVC(C=100, gamma=1))
    ('lin_svm', LinearSVC(dual=False, multi_class='ovr', class_weight=None, penalty='l1', C=1.0, max_iter=1000))
])
print('sjdjsj')
np.random.seed(123)
print(datetime.datetime.now())
# test_on_train(model, X, y)
# x_train, x_test, y_train, y_test = train_test_split(X, pd.get_dummies(y), stratify= pd.get_dummies(y))
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify= y)
model.fit(x_train, y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

print(metrics.accuracy_score(y_train, train_pred))
print(metrics.accuracy_score(y_test, test_pred))


print(datetime.datetime.now())