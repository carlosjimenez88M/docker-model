#==================================#
#        Sentiment model           #
#          SVM in Docker           #
#            2023-01-16            #
#==================================#

#%%
## Libraries ----------
import numpy as np 
import pandas as pd 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle


## Import dataset ----------

data = pd.read_csv('data/train.csv')


## Data Design -------

df_positive = data[data['sentiment']=='positive'][:9000]
df_negative = data[data['sentiment']=='negative'][:1000]
df_review_imb = pd.concat([df_positive,df_negative ]).reset_index(drop=True)

## model Design -----------

# Undersampling Strategy

rus = RandomUnderSampler(random_state= 0)
df_review_bal,df_review_bal['sentiment']=rus.fit_resample(df_review_imb[['review']],df_review_imb['sentiment'])

# Scenarios 
train,test = train_test_split(df_review_bal,test_size =0.33,random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
# also fit the test_x_vector
test_x_vector = tfidf.transform(test_x)

# Model 
svc = SVC(kernel='linear')
params = {'C': [1,4,8,16,32], 
          'gamma': [1,2,0.1,0.01],
          'kernel' : ['linear','rbf','sigmoid']}
svc = SVC()
svc_grid = GridSearchCV(svc,params, cv = 5)
svc_grid.fit(train_x_vector, train_y)
print(svc_grid.best_params_)
print(svc_grid.best_estimator_)

pickle.dump(tfidf, open("model/tfidf.pickle", "wb"))
## Report 


print(classification_report(test_y,
                            svc_grid.predict(test_x_vector),
                            labels = ['positive','negative']))






filename = 'model/finalized_model.sav'
pickle.dump(svc_grid, open(filename, 'wb'))

# loaded_model = pickle.load(open(filename, 'rb'))

# print(classification_report(test_y,
#                             loaded_model.predict(test_x_vector),
#                             labels = ['positive','negative']))

