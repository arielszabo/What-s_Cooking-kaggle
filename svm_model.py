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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os
import nltk
import gensim
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
from nltk import tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import LabelEncoder

class StemmedCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        stemmer = SnowballStemmer("english")
        stops = set(stopwords.words("english"))
        sub_pattern = r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to|\d|[^\w])\s*[^a-z]'
        return lambda doc: ([stemmer.stem(w) for w in analyzer(re.sub(sub_pattern, "", doc)) if w not in stops])


def load_n_prep(name):
    df = pd.read_json(os.path.join('{}.json'.format(name)))
    df.set_index('id', inplace=True)
    df['ingredients_count'] = df['ingredients'].apply(lambda x: len(x))
    df['ingredients_word_count'] = df['ingredients'].apply(lambda ingredients: [len(i.split()) for i in ingredients])
    df['ingredients'] = df['ingredients'].astype(str).apply(lambda ingredients: re.sub(r'\[|\]', '', ingredients)) #ingredients of two words can get lost
    # df['ingredients'] = df['ingredients'].apply(lambda ingredients: ' '.join(ingredients)) #ingredients of two words can get lost
    return df


train = load_n_prep('train')


model = Pipeline([
    ('vectorizer', StemmedCountVectorizer(lowercase=True, binary=True, stop_words='english')),
    # ('BOW', CountVectorizer(lowercase=True, stop_words='english')),
    # ('tfidf', CountVectorizer(token_pattern=r"'[^']+'", lowercase=True)),
    # ('logreg', LogisticRegression(penalty='l2', C=10, dual=False, max_iter=10000))
    # ('SGD', SGDClassifier(n_jobs=-1))
    #     ('svm', OneVsRestClassifier(SVC(C=100, coef0=1, gamma=1)))
    # ('svm', SVC(C=100, coef0=1, gamma=1))
    # ('svm', SVC(kernel='sigmoid', gamma=1, C=100))
    ('ovr_svm',OneVsRestClassifier(SVC(C=100, kernel='rbf', gamma=1, probability=True,
                                       coef0=1, decision_function_shape=None)))
    # ('lin_svm', LinearSVC(dual=False, multi_class='ovr', class_weight=None, penalty='l1', C=1.0, max_iter=1000))
])
np.random.seed(123)
print(datetime.datetime.now())

#
X = train.ingredients
y = train.cuisine

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)

model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
print('-----------')
x_train_proba = model.predict_proba(x_train)
x_test_proba = model.predict_proba(x_test)

knn = KNeighborsClassifier(n_neighbors=100)

knn.fit(x_train_proba, y_train)
y_train_pred = knn.predict(x_train_proba)
y_test_pred = knn.predict(x_test_proba)

print(metrics.accuracy_score(y_train, y_train_pred))

print(metrics.accuracy_score(y_test, y_test_pred))
exit()

X = train.ingredients
lb = LabelEncoder()
y = lb.fit_transform(train.cuisine)

param_grid = {
    'vectorizer__binary':[True, False],
    'vectorizer__ngram_range':[(1, 1), (1, 2)],
    'vectorizer__min_df':np.arange(0.1, 0.6, 0.2),
    'vectorizer__max_df':np.arange(0.6, 1.2, 0.2),
    'ovr_svm__estimator__kernel':['rbf', 'sigmoid'],
    'ovr_svm__estimator__C':[0.1, 1, 10, 100],
    'ovr_svm__estimator__gamma':[0.5, 1, 5, 10]
}

GCV = GridSearchCV(model, param_grid=param_grid, n_jobs=4, verbose=2, cv=4)
GCV.fit(X, y)
print(GCV.best_params_)
print(GCV.best_score_)

print(datetime.datetime.now())
