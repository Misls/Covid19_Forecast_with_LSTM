################## preamble ##################

# data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ignore warning messages
import warnings
warnings.filterwarnings("ignore")

# data aquisition
import requests
from xlrd import open_workbook
from io import BytesIO
from pandas import read_csv

# data preprocessing
from datetime import datetime
from datetime import date
from isoweek import Week


################## data aquisition ##################

#data from RKI github repository:

data_RWert = pd.read_csv(
    'https://raw.githubusercontent.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung/main/Nowcast_R_aktuell.csv'
    )
#data_infections = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2_Infektionen_in_Deutschland/blob/master/Aktuell_Deutschland_SarsCov2_Infektionen.csv?raw=true')
#data_vaccinations = pd.read_csv('https://github.com/robert-koch-institut/COVID-19-Impfungen_in_Deutschland/blob/master/Aktuell_Deutschland_Bundeslaender_COVID-19-Impfungen.csv?raw=true')
#data_VOC = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2-Sequenzdaten_aus_Deutschland/raw/master/SARS-CoV-2-Sequenzdaten_Deutschland.csv.xz')
#data_DIVI = pd.read_csv('https://diviexchange.blob.core.windows.net/%24web/zeitreihe-tagesdaten.csv') # Intensivbettenbelegung

#print(data_VOC)

# excel sheet from weekly updated overview statistics provided by RKI 
# including hospitalization (important value):
r = requests.get(
    'https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Daten/Klinische_Aspekte.xlsx?__blob=publicationFile', 
    headers={"User-Agent": "Chrome"}
    )
data_all = pd.read_excel(BytesIO(r.content),header=2)


################## data preprocessing ##################


# filter for reasonable columns and rename them
data_RWert = data_RWert[['Datum', 'PS_7_Tage_R_Wert']].rename(columns={'Datum': 'Date', 'PS_7_Tage_R_Wert': 'R-value'})
data_all = data_all[['Meldejahr', 'MW', 'Fälle gesamt', 'Mittelwert Alter (Jahre)', 'Männer', 'Anzahl hospitalisiert', 'Anzahl Verstorben']]
data_all.rename(columns={'Meldejahr':'Year', 'MW':'Week', 'Fälle gesamt':'Cases', 'Mittelwert Alter (Jahre)':'Age',
         'Männer':'Gender', 'Anzahl hospitalisiert':'Hospitalization', 'Anzahl Verstorben':'Deaths'},inplace=True)


# convert a week into a date 
# (aim: inperpolate between the weekly data to get daily data)
def process(df,year_key,week_key,day):
    dates = []
    for i in range(len(data_all['Week'])):
        date = Week(
            int(df.loc[i, year_key]),           # year values
            int(df.loc[i, week_key])            # week values
            ).day(day)
        dates.append(date.isoformat())          # create date in isoformat
    return np.reshape(dates, (1, len(dates))).T # vector of dates

# insert date values in current dataframe
data_all['Date'] = process(data_all,'Year','Week',3)

# merge dataframes on 'Date' and interpolate missing data in data_all
data = pd.merge(data_RWert,data_all,how='outer',on=['Date'])
data[['Year','Week']] = data[['Year','Week']].fillna(method='ffill',axis=0)  # merge dataframes      
data.interpolate(method = 'spline',order = 2, inplace=True)             # interpolate missing data from weekly dataframes
data.drop(data.loc[data['Cases']<0].index,inplace=True)  # drop unrealistic case numbers from interpolation
data.drop(data.loc[data['Hospitalization']<0].index,inplace=True) # drop unrealistic case numbers from interpolation
data.reset_index(drop=True, inplace=True)

# correct data:
data[['Cases','Hospitalization']] = data[['Cases','Hospitalization']]/7



# reorder columns: time related columns to the left
data = data[['Date',  'Year', 'Week','R-value', 'Cases', 'Age', 'Gender',
       'Hospitalization', 'Deaths']]

# create a column for the prevailing trend: 0 = decreasing and 1 = increasing
trend = np.ones((len(data),1)).astype(int)
for i in range(len(data)):
    if i == 0:
        if data.loc[i,'Cases']>data.loc[i+1,'Cases']:
            trend[i]=0
    else:
        if data.loc[i,'Cases']<data.loc[i-1,'Cases']:
            trend[i]=0
#data['Trend'] = trend   # add trend column to the dataframe


################## last column ##################

# create target values: Lockdown calsses from 0 to 3
# the dates are from https://de.wikipedia.org/wiki/COVID-19-Pandemie_in_Deutschland 
# and categorized in 4 classes. Dates indicate a change. 
Lockdown = pd.DataFrame([
                   ('2020-03-02', '0'),
                   ('2020-03-08', '1'),
                   ('2020-03-17', '2'),
                   ('2020-03-22', '3'),
                   ('2020-05-06', '1'),
                   ('2020-10-28', '2'),
                   ('2020-12-13', '3'),
                   ('2021-03-03', '1'), 
                   ('2021-04-23', '2'),
                   ('2021-06-30', '1'),
                   ('2021-07-23', '0'),
                   ('2021-08-14', '1'),
                   ('2021-09-03', '2')],        
           columns=('Date', 'Lockdown-Strength')
                 )
        
data = pd.merge(data,Lockdown,how='outer',on=['Date'])       # merge dataframes
data['Lockdown-Strength'].fillna(method='ffill',inplace=True) # fill the missing data by using preceding values 
data.dropna(inplace=True)


################## save dataframe to csv ##################


data.to_csv('data.csv',index = False)
################## end of preprocessing ##################


#print(data_all.loc[data_all['Meldejahr'] == 2021, 'MW'])
index = pd.to_datetime(data['Date'])

plt.subplots()
plt.plot(index, data['Cases'])
plt.plot(index, data['Hospitalization']*10)
plt.title("Covid-19 in Germany")
plt.ylabel("Count")
plt.xlabel('Time')
plt.legend(['Daily Cases', 'Hospitalization x 10'])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Covid-Data-Cases.png')


plt.subplots()
plt.plot(index, data['R-value']*50)
plt.plot(index, data['Age'])
plt.title("Covid-19 in Germany")
plt.ylabel("Mean Age / R Value")
plt.xlabel('Time')
plt.legend(['R Value x 50','Mean Age'])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Covid-Data-Age-RValue.png')