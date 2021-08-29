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

data_RWert = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung/blob/0c3a7b18078ca50f81b9002976801bb826a77197/Nowcast_R_aktuell.csv?raw=true')
#data_infections = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2_Infektionen_in_Deutschland/blob/master/Aktuell_Deutschland_SarsCov2_Infektionen.csv?raw=true')
#data_vaccinations = pd.read_csv('https://github.com/robert-koch-institut/COVID-19-Impfungen_in_Deutschland/blob/master/Aktuell_Deutschland_Bundeslaender_COVID-19-Impfungen.csv?raw=true')
#data_VOC = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2-Sequenzdaten_aus_Deutschland/raw/master/SARS-CoV-2-Sequenzdaten_Deutschland.csv.xz')

#excel sheet from weekly updated overview statistics provided by RKI including hospitalization (target value):

r = requests.get('https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Daten/Klinische_Aspekte.xlsx?__blob=publicationFile', headers={"User-Agent": "Chrome"})
data_all = pd.read_excel(BytesIO(r.content),header=2)



############################# data preprocessing ####################

# convert a week into a date (aim: inperpolate between the weekly data to get daily data)
def process(df,year_key,week_key,day):
    dates = []
    for i in range(len(data_all['MW'])):
        date = Week(
            int(df.loc[i, year_key]),           # year values
            int(df.loc[i, week_key])            # week values
            ).day(day)
        dates.append(date.isoformat())          # create date in isoformat
    return np.reshape(dates, (1, len(dates))).T # vector of dates

# insert date values in current dataframe
data_all['Datum'] = process(data_all,'Meldejahr','MW',3)
print(data_all.head(5))

# merge dataframes on 'Datum' and interpolate missing data in data_all
data_merged = pd.merge(data_RWert,data_all,how='outer',on=['Datum'])
data = data_merged.interpolate(method= 'spline',order=5).dropna()
data.drop(data.loc[data['Fälle gesamt']<0].index,inplace=True) # drop unrealistic case numbers from interpolation
data.reset_index(drop=True, inplace=True)

# create a column for the prevailing trend: 0 = decreasing and 1 = increasing
trend = np.ones((len(data),1)).astype(int)
for i in range(len(data)):
    if i == 0:
        if data.loc[i,'Fälle gesamt']>data.loc[i+1,'Fälle gesamt']:
            trend[i]=0
    else:
        if data.loc[i,'Fälle gesamt']<data.loc[i-1,'Fälle gesamt']:
            trend[i]=0
data['Trend'] = trend



#print(data_all.loc[data_all['Meldejahr'] == 2021, 'MW'])
#x = data_all.loc[data_all['Meldejahr'] == 2021, 'MW']
#y = data_all.loc[data_all['Meldejahr'] == 2021, 'Anzahl hospitalisiert']
x = data['Datum']
y = data['Anzahl hospitalisiert']
plt.plot(x,data['PS_COVID_Faelle'])
plt.plot(x,data['Fälle gesamt']/7)
plt.show()

