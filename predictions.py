#==================================#
#         Prediction model         #
#            2023-01-16            #
#==================================#


## Libraries ---------------
#%%
import numpy as np 
import pandas as pd 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle


## Import Model --------
tfidf = TfidfVectorizer(stop_words='english')
filename = 'model/finalized_model.sav'
SVM = pickle.load(open(filename, 'rb'))

tfidf = pickle.load(open("model/tfidf.pickle", "rb"))

## Function --------
def text_to_predict():
    try:
        text_1 = input('Insert your comment: ')
        text_2 = tfidf.transform([text_1])
        prediction = SVM.predict(text_2)
        return print(prediction[0])
    except:
        pass
if __name__ == '__main__':
    text_to_predict()
# %%
