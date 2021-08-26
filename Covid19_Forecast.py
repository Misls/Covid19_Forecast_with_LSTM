############################### preamble ##########################

# data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# imports
import requests
from xlrd import open_workbook
from io import BytesIO
from pandas import read_csv

# data preprocessing
from datetime import datetime
from datetime import date
from isoweek import Week


############################# data aquisition #####################

#data from RKI github repository:

#data_RWert = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung/blob/0c3a7b18078ca50f81b9002976801bb826a77197/Nowcast_R_aktuell.csv?raw=true')
#data_infections = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2_Infektionen_in_Deutschland/blob/master/Aktuell_Deutschland_SarsCov2_Infektionen.csv?raw=true')
#data_vaccinations = pd.read_csv('https://github.com/robert-koch-institut/COVID-19-Impfungen_in_Deutschland/blob/master/Aktuell_Deutschland_Bundeslaender_COVID-19-Impfungen.csv?raw=true')
#data_VOC = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2-Sequenzdaten_aus_Deutschland/raw/master/SARS-CoV-2-Sequenzdaten_Deutschland.csv.xz')

#excel sheet from weekly updated overview statistics provided by RKI including hospitalization (target value):

r = requests.get('https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Daten/Klinische_Aspekte.xlsx?__blob=publicationFile', headers={"User-Agent": "Chrome"})
data_all = pd.read_excel(BytesIO(r.content),header=2)
print(np.mean(data_all['Anteil Verstorben']))
for col in data_all.columns:
   print(col)


#print(data_all['MW'].head(50))

############################# data preprocessing ####################

# convert a week into a date (aim: inperpolate between the weekly data to get daily data)


def process(data_all):
    datetime = Week(data_all.loc[43, 'Meldejahr'],data_all.loc[43, 'MW']).thursday()
    return datetime

#data_all = data_all.drop([43])
print(data_all[['MW']].head(50))

dates = process(data_all)
print(dates)

#print(data_all.loc[data_all['Meldejahr'] == 2021, 'MW'])
#x = data_all.loc[data_all['Meldejahr'] == 2021, 'MW']
#y = data_all.loc[data_all['Meldejahr'] == 2021, 'Anzahl hospitalisiert']
#plt.plot(x,y)
#plt.show()

