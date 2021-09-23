# Covid19_Forecast

# workflow:
Execute the following python scripts:
1. Data_Preprocessing.py
2. ML_training.py > ML_training.log
3. LSTM_Training.py > LSTM_Training.log
4. Lockdown_Prediction.py

## Caution: work in progress

ML for predicting a Lockdown based on RKI data

To do:
- get rid of outliers in ML_Training when fit model
- add DIVI and Vaccination data
- increase geografic resolution of data (Bundesland, Landkreis)
- check spline interpolation
- ML_Training: drop Bagging and search for the top 3 classifiers and put them in the Bagging classifier and do importance analysis on it.
- DONE: create Forecast algorithm giving all requested features to apply Lockdown:Classifier.pkl
- integrate features from different countries in LSTM_Training for better learning