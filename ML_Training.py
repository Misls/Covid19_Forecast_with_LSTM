# Analysing different ML models 
# and pick the best for Training 

################## preamble##################

# data analysis:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import mean
from numpy import std
from sklearn.manifold import TSNE
import seaborn as sns

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
from sklearn.ensemble import AdaBoostClassifier

# preprocessing and evaluating:
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


# saving model:
import pickle

############## parameters ##################

Hyper_Opt = False # Hyperparameter optimization 
# if false: best model will be taken without further opimization 

################## load dataframe ##################

data = pd.read_csv('data.csv').dropna()

################## define help functions ##################

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
    # AdaBoostClassifier
    models.append(AdaBoostClassifier(n_estimators=100, random_state=0))
    names.append('AdaBoost')
    # GaussianNB
    models.append(GaussianNB())
    names.append('GaussianNB')
    # LogisticRegression
    models.append(LogisticRegression())
    names.append('LogisticRegression')
    # RidgeClassifier
    models.append(RidgeClassifier())
    names.append('RidgeClassifier')
    # MLPClassifier
    models.append(MLPClassifier(hidden_layer_sizes = 10, learning_rate='adaptive',tol=0.001,max_iter=2000))
    names.append('MLPC')
    # LinearDiscriminantAnalysis
    models.append(LinearDiscriminantAnalysis())
    names.append('LinearDiscriminant')
    # SVM
    models.append(SVC(gamma='scale'))
    names.append('SVC')
    # KNeighbors
    models.append(KNeighborsClassifier())
    names.append('KNeighbors')
    # CART
    models.append(DecisionTreeClassifier())
    names.append('DecisionTree')
    # Bagging
    #models.append(BaggingClassifier())
    #names.append('BAG')
    # RF
    models.append(RandomForestClassifier(n_estimators=100))
    names.append('RandomForest')
    # GBM
    models.append(GradientBoostingClassifier())
    names.append('GradientBoosting')
    return models, names

# define features and target
X, y = data.drop(['Date','Lockdown-Intensity'], axis=1), data['Lockdown-Intensity']
feature_names = [f'{col}' for col in X.columns]


# define models
models, names = get_models()
results_accuracy = list()
summary = list()

# calculate the base score
base_score = y.value_counts().max()/len(y)
print('Base score: %.2f' % (base_score))

y = LabelEncoder().fit_transform(y)
X = MinMaxScaler().fit_transform(X)

# main part of the code: evaluate the models by RepeatedStratifiedKFol 
# defined above in get_models() and evaluate()
for i in range(len(models)):
    # start time measurement
    start_time = time.time()
    # wrap the model in a pipeline
    pipeline = Pipeline(steps=[
                        #('n',MinMaxScaler()),
                        ('m',models[i])
                        ])
    # evaluate the model and store results
    scores = evaluate_model(X, y, pipeline)
    results_accuracy.append(scores)
    # end time measurement
    elapsed_time = float("{:.0f}".format(time.time() - start_time))
    # summarize performance
    summary.append([models[i], mean(scores)])
    print('>%s: Accuracy = %.3f \xb1 %.3f, time: %s ' % 
     (names[i], mean(scores), std(scores), str(datetime.timedelta(seconds=elapsed_time))))


############## get the best model and fit it ##############

# find model
summary=pd.DataFrame(summary, columns=['Models', 'Score'])      # Score and Model data in one df 
best_model = summary.loc[np.argmax(summary['Score']),'Models']  # choose the model with highest acurracy
print('Best Model: %s ' % (best_model))


############ hyperparameter oprimization ############

if Hyper_Opt == True:
    # start time measurement
    start_time = time.time()
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 1, stop = 150, num = 10)]
    # Number of features to consider at every split
    max_features = [
        #'auto',
        #'sqrt',
        'log2'
        ]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(100, 1000, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5, 15]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {
                    'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap
                }
    base_estimator = best_model
    sh = HalvingGridSearchCV(base_estimator, random_grid, cv=5,
                            factor=3,n_jobs=-1,#max_resources=50,
                            ).fit(X, y)
    # get accuracy
    pipeline = Pipeline(steps=[
        #('n',MinMaxScaler()),
        ('m',sh.best_estimator_)])
    # evaluate the model and store results
    best_model = sh.best_estimator_
    score = evaluate_model(X, y, pipeline)
    improvement = mean(score) - max(summary['Score'])
    # end time measurement
    elapsed_time = float("{:.0f}".format(time.time() - start_time))
    print('Best Model: %s with accuracy: %.3f \xb1 %.3f ' % (sh.best_estimator_,mean(score),std(score)))
    print('Accuracy improved by %.3f percent' % (improvement*100))
    print('Elapsed Time for hyperparameter optimization: %s' % (str(datetime.timedelta(seconds=elapsed_time))))

if Hyper_Opt == False:
     print('>>> No hyperparameter optimization. To run code with optimization set Hyper_Opt = True')

####################################################
# train and save the model
model = best_model.fit(X, y)
Pkl_Filename = "Lockdown_Classifier.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file) 


################## end of training ##################

# feature importance for RF from sklearn documentation
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

# feature_names = [f'{col}' for col in X.columns]
forest = best_model
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([
    tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
ax = plt.gca()
plt.gcf().autofmt_xdate() # Rotation
fig.tight_layout()
plt.savefig('Figures\Feature_Analysis\Feature_Importance_MDI.png')
#plt.show()

####################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y)
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=feature_names)


fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
ax = plt.gca()
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Feature_Analysis\Feature_Importance_Permutation.png')


# plot the performances of the evaluated models
fig, ax = plt.subplots()
pyplot.boxplot(results_accuracy, 
               labels=names, 
               showmeans=True)

ax.set_title('Performance Analysis', fontweight='bold', fontsize=15)
ax.set_ylabel('Accuracy')
fig.autofmt_xdate()
#ax.set_xticklabels(ax.get_xticks(),rotation=45,labels=names)
pyplot.savefig('Figures\Feature_Analysis\Performance Analysis.png')
#pyplot.show()

########### plot t-SNE graph ##################


sns.set(rc={'figure.figsize':(11.7,8.27)})

fig, ax = plt.subplots()
palette = sns.color_palette( 'bright',n_colors = 4  )
tsne = TSNE()
X_embedded = tsne.fit_transform(X)
y = data['Lockdown-Intensity']
sns.scatterplot(x = X_embedded[:,0], y = X_embedded[:,1], hue=y, legend='full',palette=palette)
ax.set_title('t_SNE Representation')
pyplot.savefig('Figures\Feature_Analysis\T-SNE_Analysis.png')
