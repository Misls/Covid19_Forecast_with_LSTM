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

# ignore warning messages
import warnings
warnings.filterwarnings("ignore")


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
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

# preprocessing and evaluating:
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import minmax_scale


# saving model:
import pickle

############## parameters ##################

n_ensemble = 5 # number of classifiers in the ensemble 
Hyper_Opt = False # Hyperparameter optimization 
# if False: best model will be taken without further opimization 

################## load dataframe ##################
data = pd.read_csv('data.csv').dropna()

# define features and target
X, y = data.drop(['Date','Lockdown-Intensity'], axis=1), data['Lockdown-Intensity']
feature_names = [f'{col}' for col in X.columns]
################## define help functions ##################
# evaluate a model
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
# Compute feature importance of Voting Classifier
def compute_feature_importance(voting_clf, weights):  
    feature_importance = dict()
    for est in voting_clf.estimators_:
        feature_importance[str(est)] = est.feature_importances_
    
    fe_scores = [0]*len(list(feature_importance.values())[0])
    for idx, imp_score in enumerate(feature_importance.values()):
        imp_score_with_weight = imp_score*weights[idx]
        fe_scores = list(np.add(fe_scores, list(imp_score_with_weight)))
    return fe_scores

# define models to test
def get_models():
    models, names = list(), list()
    # AdaBoostClassifier
    models.append(AdaBoostClassifier(n_estimators=100, random_state=0))
    names.append('AdaBoost')
    # GaussianNB
    models.append(GaussianNB())
    names.append('GaussianNB')
    #Perceptron
    models.append(Perceptron())
    names.append('Perceptron')
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
    models.append(SVC(gamma='scale',kernel='poly', tol = 0.01, probability = True))
    names.append('SVC')
    # KNeighbors
    models.append(KNeighborsClassifier())
    names.append('KNeighbors')
    # CART
    models.append(DecisionTreeClassifier())
    names.append('DecisionTree')
    # Bagging
    models.append(BaggingClassifier())
    names.append('Bagging')
    # RF
    models.append(RandomForestClassifier(n_estimators = 100))
    names.append('RandomForest')
    # GBM
    models.append(GradientBoostingClassifier())
    names.append('GradientBoosting')
    return models, names

# define models
models, names = get_models()
results_accuracy = list()
summary = list()

# calculate the base score
base_score = y.value_counts().max()/len(y)
print('Base score: %.2f' % (base_score))

y = LabelEncoder().fit_transform(y)
X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

# main part of the code: evaluate the models by RepeatedStratifiedKFol 
# defined above in get_models() and evaluate()
for i in range(len(models)):
    # start time measurement
    start_time = time.time()
    # wrap the model in a pipeline
    pipeline = Pipeline(steps=[
                        #('n',MinMaxScaler(feature_range=(0, 1))),
                        ('m',models[i])
                        ])
    # evaluate the model and store results
    scores = evaluate_model(X, y, pipeline)
    results_accuracy.append(scores)
    # end time measurement
    elapsed_time = float("{:.0f}".format(time.time() - start_time))
    # summarize performance
    summary.append([names[i], models[i], mean(scores)])
    print('>%s: Accuracy = %.3f \xb1 %.3f, time: %s ' % 
     (names[i], mean(scores), std(scores), str(datetime.timedelta(seconds=elapsed_time))))


############## get the best model and fit it ##############

# find best model
summary=pd.DataFrame(summary, columns=['Names','Models', 'Score'])      # Score and Model data in one df 
best_model = summary.loc[np.argmax(summary['Score']),'Models']  # choose the model with highest acurracy
print('Best Model: %s ' % (best_model))
# find Top MOdels:
summary.sort_values(by=['Score'], ascending = False, inplace = True)
top_models = summary.reset_index(drop = True, inplace=True)
top_models = summary[0:n_ensemble]
estimators = list(zip(*map(top_models.get, ['Names', 'Models'])))
print('>>> Top %i:' % (n_ensemble))
print(top_models)

############ hyperparameter oprimization ############

if Hyper_Opt:
    # start time measurement
    start_time = time.time()
    ########### RandomForest ###############
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 1, stop = 150, num = 75)]
    # Number of features to consider at every split
    max_features = [
        #'auto',
        'sqrt',
        'log2'
        ]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 1000, num = 100)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 4, 6, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    ############# Gadient Boost ###################
    # Loss function for GBC
    loss = ['deviance', 'exponential']
    # Learning Rate for GBC
    learning_rate = [0.1, 0.01]
    # The fraction of samples to be used for fitting the individual base learners
    subsample = [1, 0.9, 0.8]
    # The function to measure the quality of a split
    criterion = ['friedman_mse', 'squared_error', 'mse', 'mae']
    ############### KNeighbors ##################
    # Number of next neighbors
    n_neighbors = range(1,32)
    # weights
    weights_KN = ['uniform', 'distance']
    # Leaf size passed to BallTree or KDTree
    leaf_size = [int(x) for x in np.linspace(10, 200, num = 100)]

    
    estimators = []
    names_hyper_opt = top_models['Names']
    models_hyper_opt = top_models['Models']
    scores_hyper_opt = np.zeros(n_ensemble)
    for i in range(n_ensemble):
        if str(models_hyper_opt[i]) == 'RandomForestClassifier()':
            random_grid = {
                            'n_estimators': n_estimators,
                            'max_features': max_features,
                            #'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            #'bootstrap': bootstrap
                        }
        if str(models_hyper_opt[i]) == 'GradientBoostingClassifier()':
            random_grid = {
                            'n_estimators': n_estimators,
                            #'loss' : loss, 
                            'learning_rate' : learning_rate,
                            'subsample' : subsample,
                            #'criterion' : criterion
                        }
        if str(models_hyper_opt[i]) == 'KNeighborsClassifier()':
            random_grid = {
                            'n_neighbors': n_neighbors,
                            'weights' : weights_KN, 
                            'leaf_size' : leaf_size
                        }
        if str(models_hyper_opt[i]) == 'DecisionTreeClassifier()':
            random_grid = {
                            'criterion' : ['gini', 'entropy'],
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split
                        }
        if str(models_hyper_opt[i]) == 'SVC(probability=True)':
            random_grid = {
                            #'gamma' : ['scale', 'auto'],
                            'tol': [1e-2,1e-3,1e-4],
                            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                        }
        if str(models_hyper_opt[i]) == 'BaggingClassifier()':
            random_grid = {
                            'base_estimator' : [
                                Perceptron(tol=1e-3),
                                DecisionTreeClassifier(),
                                KNeighborsClassifier(),
                                SVC(probability=True),
                                LogisticRegression(),
                                #RandomForestClassifier(),
                                #KNeighborsClassifier(), 
                                #GradientBoostingClassifier()
                                ],
                            'n_estimators': n_estimators
                        }


        print('start hyperparameter optimization: %s' % (str(models_hyper_opt[i])))
        base_estimator = models_hyper_opt[i]
        sh = HalvingGridSearchCV(base_estimator, random_grid, cv=20,
                                factor=3,n_jobs=-1,#max_resources=50,
                                ).fit(X, y)
        # get accuracy
        pipeline = Pipeline(steps=[
            #('n',MinMaxScaler(feature_range=(0, 1))),
            ('m',sh.best_estimator_)])
        # evaluate the model and store results
        hyper_opt_model = (names_hyper_opt[i], sh.best_estimator_)
        score = evaluate_model(X, y, pipeline)
        improvement = mean(score) - top_models['Score'][i]
        scores_hyper_opt[i] = mean(score)
        estimators.append(hyper_opt_model)
        
        # end time measurement
        elapsed_time = float("{:.0f}".format(time.time() - start_time))
        print('Best Model: %s with accuracy: %.3f \xb1 %.3f ' % (sh.best_estimator_,mean(score),std(score)))
        print('Accuracy improved by %.3f percent' % (improvement*100))
        print('Elapsed Time for hyperparameter optimization: %s' % (str(datetime.timedelta(seconds=elapsed_time))))

else:
     print('>>> No hyperparameter optimization. To run code with optimization set Hyper_Opt = True')

############### build ensemble learner ####################
if Hyper_Opt:
    weights = list(scores_hyper_opt/(scores_hyper_opt.sum()))
else:
    weights = list(top_models['Score']/(top_models['Score'].sum()))
#weights = minmax_scale(weights, feature_range=(0.1, 1))
ensemble = VotingClassifier(estimators, voting='soft', n_jobs = -1, weights = weights)
ensemble_score = evaluate_model(X, y, ensemble)
ensemble.fit(X, y)
print('Accuracy of Top %i ensemble classifier: %.3f \xb1 %.3f' % (n_ensemble,mean(ensemble_score), std(ensemble_score)))
best_model = ensemble
#best_model = RandomForestClassifier()

####################################################
# train and save the model
model = best_model.fit(X, y)
Pkl_Filename = "Lockdown_Classifier.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file) 

################## end of training ##################

# feature importance for RF from sklearn documentation
forest = RandomForestClassifier()
forest.fit(X, y)
importances = forest.feature_importances_
sorted_idx = importances.argsort()
std = np.std([
    tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)[sorted_idx]

# if KNN or Bagging is part of the ensemble: comment the two lines out
#importances = compute_feature_importance(ensemble, weights)
#forest_importances = pd.Series(importances, index=feature_names)[sorted_idx]

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
ax = plt.gca()
plt.gcf().autofmt_xdate() # Rotation
fig.tight_layout()
plt.savefig('Figures\Feature_Analysis\Feature_Importance_MDI.png')
plt.close()

####################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y)
result = permutation_importance(
    forest, X_test, y_test, n_repeats=100, n_jobs=-1)
forest_importances = pd.Series(result.importances_mean, index=feature_names)
sorted_idx = forest_importances.argsort()

fig, ax = plt.subplots()
forest_importances[sorted_idx].plot.bar(yerr=result.importances_std[sorted_idx], ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
ax = plt.gca()
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Feature_Analysis\Feature_Importance_Permutation.png')
plt.close()


# plot the performances of the evaluated models
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots()
pyplot.boxplot(results_accuracy, 
               labels=names, 
               showmeans=True)
ax.set_title('Performance Analysis', fontweight='bold', fontsize=15)
ax.set_ylabel('Accuracy')
fig.autofmt_xdate()
#ax.set_xticklabels(ax.get_xticks(),rotation=45,labels=names)
pyplot.savefig('Figures\Feature_Analysis\Performance Analysis.png')
plt.close()

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
plt.close()