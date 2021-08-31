# Analysing different ML models and pick the best for Training 

############################### preamble ##########################

# data analysis:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import mean
from numpy import std


# Model-Training:
import time
import datetime

# machine learning algorithms:
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier

# preprocessing and evaluating:
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


# saving model:
import pickle


############################# load dataframe ######################

data = pd.read_csv('data.csv').dropna()

############################# define help functions ######################

# evaluate a model
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
 
# define models to test
def get_models():
    models, names = list(), list()
    # GaussianNB
    models.append(GaussianNB())
    names.append('GNB')
    # LogisticRegression
    models.append(LogisticRegression())
    names.append('Log')
    # RidgeClassifier
    models.append(RidgeClassifier())
    names.append('RID')
    # MLPClassifier
    models.append(MLPClassifier(hidden_layer_sizes = 10, learning_rate='adaptive',tol=0.001,max_iter=2000))
    names.append('MLP')
    # LinearDiscriminantAnalysis
    models.append(LinearDiscriminantAnalysis())
    names.append('LDA')
    # SVM
    models.append(SVC(gamma='scale'))
    names.append('SVM')
    # KNeighbors
    models.append(KNeighborsClassifier())
    names.append('KN')
    # CART
    models.append(DecisionTreeClassifier())
    names.append('DT')
    # Bagging
    models.append(BaggingClassifier(n_estimators=100))
    names.append('BAG')
    # RF
    models.append(RandomForestClassifier(n_estimators=100))
    names.append('RF')
    # GBM
    models.append(GradientBoostingClassifier(n_estimators=100))
    names.append('GBM')
    return models, names

# define features and target
X, y = data.drop(['Datum','Lockdown-Strength'], axis=1), data['Lockdown-Strength']



# define models
models, names = get_models()
results_accuracy = list()
summary = list()

# calculate the base score
base_score = y.value_counts().max()/len(y)
print('Base score: %.2f' % (base_score))

y = LabelEncoder().fit_transform(y)

# main part of the code: evaluate the models by RepeatedStratifiedKFol defined above in get_models() and evaluate()
for i in range(len(models)):
    # start time measurement
    start_time = time.time()
    # wrap the model in a pipeline
    pipeline = Pipeline(steps=[('n',MinMaxScaler()),('m',models[i])])
    # evaluate the model and store results
    scores = evaluate_model(X, y, pipeline)
    results_accuracy.append(scores)
    # end time measurement
    elapsed_time = float("{:.0f}".format(time.time() - start_time))
    # summarize performance
    summary.append([models[i], mean(scores)])
    print('>%s: Accuracy = %.3f \xb1 %.3f, time: %s ' % 
     (names[i], mean(scores), std(scores), str(datetime.timedelta(seconds=elapsed_time))))


########################## get the best model and fit it #############################

# find model
summary=pd.DataFrame(summary, columns=['Models', 'Score'])      # Score and Model data in one df 
best_model = summary.loc[np.argmax(summary['Score']),'Models']  # choose the model with highest acurracy
print('Best Model: %s ' % (best_model))

# train and save the model
model = best_model.fit(X, y)
Pkl_Filename = "Lockdown_Classifier.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file) 


########################## end of training ################################


# plot the results
pyplot.boxplot(results_accuracy, 
               labels=names, 
               showmeans=True)
pyplot.title('Performance Analysis', fontweight='bold', fontsize=15)
#pyplot.show()