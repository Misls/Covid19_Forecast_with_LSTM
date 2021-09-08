# predict Lockdown-Strength depending on trained Lockdown-Classifier and forecast by LSTM

################## preamble##################

# data analysis:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
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

plt.subplots()
plt.plot(dates,y_pred)
plt.title("Probabilities for Lockdown")
plt.ylabel("Probability")
plt.xlabel('Time')
plt.legend(['No Lockdown','Green', 'Yellow', 'Red'])
plt.savefig('Figures\Lockdown_Probability.png')
#plt.show()

