# predict Lockdown-Strength depending on trained Lockdown-Classifier and forecast by LSTM

################## preamble##################

# data analysis:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.dates as mdates
from numpy import mean
from numpy import std

# load model
import pickle

# load Lockdown-Classifier
Pkl_Filename = 'Lockdown_Classifier.pkl'  
with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

df = pd.read_csv('data_pred.csv')
dates = df['Date']
X = df.drop(['Date'],axis=1)

y_pred = Pickled_Model.predict_proba(X)


index = pd.to_datetime(dates)
plt.subplots()
plt.plot(index, y_pred)
plt.title("Probabilities for Lockdown")
plt.ylabel("Probability")
plt.xlabel('Time')
plt.legend(['No Lockdown','Light', 'Middle', 'Hard'])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Lockdown_Probability.png')