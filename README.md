# Covid19_Forecast

# workflow:
Execute the following python scripts:
0. Activate venv:
    > Set-ExecutionPolicy Unrestricted -Scope Process (for MS Windows in some cases)
    > ./.venv/Scripts/Activate.ps1
1.  > Data_Preprocessing.py
2.  > ML_training.py > ML_training.log
3.  > LSTM_Training.py > LSTM_Training.log
4.  > Lockdown_Prediction.py

## Caution: work in progress
### since Omicron, the prediction is not accurate at all! more data has to be implemented

ML/DL for predicting a Lockdown based on RKI data

To do:
- substitude Vaccination number by immune status or resambling feature
- integrate VOC data
- increase geografic resolution of data (Bundesland, Landkreis)
- integrate features from different countries in LSTM_Training for better learning
- find a reliable way to define the dtrength of a Lockdown Measure