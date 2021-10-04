################## preamble ##################

# data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ignore warning messages
import warnings
from pandas.core.base import DataError
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
data_infections = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2_Infektionen_in_Deutschland/blob/master/Aktuell_Deutschland_SarsCov2_Infektionen.csv?raw=true')
#data_vaccinations = pd.read_csv('https://github.com/robert-koch-institut/COVID-19-Impfungen_in_Deutschland/blob/master/Aktuell_Deutschland_Bundeslaender_COVID-19-Impfungen.csv?raw=true')
r = requests.get(
    'https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Daten/Impfquotenmonitoring.xlsx?__blob=publicationFile', 
    headers={"User-Agent": "Chrome"}
    )
data_vaccinations = pd.read_excel(BytesIO(r.content),header=0,sheet_name = 'Impfungen_proTag')
data_vaccinations.dropna(inplace=True)
data_vaccinations.drop(len(data_vaccinations)-1, axis = 0, inplace = True)
#data_VOC = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2-Sequenzdaten_aus_Deutschland/raw/master/SARS-CoV-2-Sequenzdaten_Deutschland.csv.xz')
data_DIVI = pd.read_csv('https://diviexchange.blob.core.windows.net/%24web/zeitreihe-tagesdaten.csv') # Intensivbettenbelegung

# excel sheet from weekly updated overview statistics provided by RKI 
# including hospitalization:
r = requests.get(
    'https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Daten/Klinische_Aspekte.xlsx?__blob=publicationFile', 
    headers={"User-Agent": "Chrome"}
    )
data_all = pd.read_excel(BytesIO(r.content),header=3)


################## data preprocessing ##################


# filter for reasonable columns and rename them
data_RWert = data_RWert[['Datum', 'PS_7_Tage_R_Wert']].rename(columns={'Datum': 'Date', 'PS_7_Tage_R_Wert': 'R-value'})
data_all = data_all[['Meldejahr', 'MW', 'Fälle gesamt', 'Mittelwert Alter (Jahre)', 'Männer', 'Anzahl hospitalisiert', 'Anzahl Verstorben']]
data_all.rename(columns={'Meldejahr':'Year', 'MW':'Week', 'Fälle gesamt':'Cases', 'Mittelwert Alter (Jahre)':'Age',
         'Männer':'Gender', 'Anzahl hospitalisiert':'Hospitalization', 'Anzahl Verstorben':'Deaths'},inplace=True)        
data_all['Year'].replace(2022, 2021, inplace = True)
data_vaccinations = data_vaccinations[['Datum', 'Erstimpfung', 'Zweitimpfung']].rename(
    columns={'Datum': 'Date', 'Erstimpfung': '1rst_Vac', 'Zweitimpfung' : '2nd_Vac'})
data_DIVI = data_DIVI[['date','faelle_covid_aktuell']].rename(columns={'date':'Date', 'faelle_covid_aktuell':'Intensive_Care'})
data_infections = data_infections[['Refdatum','AnzahlFall']].rename(columns = {'Refdatum':'Date','AnzahlFall':'Cases'})


################# bring data into daily format##################

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

# convert 'Date' object strings to datetime
data_all['Date'] = pd.to_datetime(data_all['Date'])
data_RWert['Date'] = pd.to_datetime(data_RWert['Date'])
data_vaccinations['Date'] = pd.to_datetime(data_vaccinations['Date'],format='%d.%m.%Y')
data_DIVI['Date'] = pd.to_datetime(data_DIVI['Date'],format='%Y-%m-%d')
data_infections['Date'] = pd.to_datetime(data_infections['Date'],format='%Y-%m-%d')

# sort DIVI data
data_DIVI_sort = pd.DataFrame(index=data_DIVI['Date'].unique(), columns=['Intensive_Care']).rename_axis('Date')
for day in data_DIVI['Date'].unique():
    data_DIVI_sort['Intensive_Care'][day]=data_DIVI[(data_DIVI['Intensive_Care']> -1)&(data_DIVI['Date']==day)]['Intensive_Care'].sum()
#data_DIVI_sort['Date'] = data_DIVI['Date'].unique()

# sort infection data
data_infections_sort = pd.DataFrame(index=data_infections['Date'].unique(), columns=['Cases2']).rename_axis('Date').sort_index()
for day in data_infections['Date'].unique():
    data_infections_sort['Cases2'][day]=data_infections[(data_infections['Date']==day)]['Cases'].sum()
data_infections_sort['Date'] = data_infections_sort.index

# 7 day incidince per 100.000:
incidince = list()
for i in range(len(data_infections_sort)+1):
    if i < 6:
        inc = 0
        incidince.append(inc)
    if i  > 6:
        inc = data_infections_sort['Cases2'][i-7:i].sum()/831
        incidince.append(inc)
data_infections_sort.reset_index(drop=True, inplace=True)
data_infections_sort['Incidince'] = pd.DataFrame(incidince)
print(data_infections_sort)

###################### merge dataframes on 'Date' ############################
data = pd.merge(data_RWert,data_all,how='outer',on=['Date'])
data = pd.merge(data,data_vaccinations,how='outer',on=['Date'])
data = pd.merge(data,data_DIVI_sort,how='outer',on=['Date'])
data = pd.merge(data,data_infections_sort,how='outer',on=['Date'])
#print(data)
#  fill and interpolate missing data in data_all
data[['Year','Week']] = data[['Year','Week']].fillna(method='ffill',axis=0)  # fill missing week and year numbers
data[['1rst_Vac', '2nd_Vac']] = data[['1rst_Vac', '2nd_Vac']].fillna(value = 0, axis=0).cumsum()  # fill missing vaccination numbers with zero
data[data.columns[1:]] = data[data.columns[1:]].interpolate(method = 'spline',order = 3, axis = 0)  # interpolate missing data from weekly dataframes
data.drop(data.loc[data['Cases']<0].index,inplace=True)  # drop unrealistic case numbers from interpolation
data.drop(data.loc[data['Hospitalization']<0].index,inplace=True) # drop unrealistic case numbers from interpolation
data.drop(data.loc[data['R-value']>1.6].index,inplace=True) # drop unrealistic R-Values from first days calculations
data.reset_index(drop=True, inplace=True)

# correct data:
data[['Cases','Hospitalization']] = data[['Cases','Hospitalization']]/7

# reorder columns: time related columns to the left
data = data[['Date',  'Year', 'Week', 'Cases','Age','R-value','Hospitalization', 
'Deaths', 'Gender','1rst_Vac', '2nd_Vac','Intensive_Care', 'Cases2', 'Incidince'
       ]]

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

################## target values ##################

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
                    ('2021-09-10', '2'),
                    ('2021-10-01', '1')],       
           columns=('Date', 'Lockdown-Strength')
                 )
Lockdown['Date'] = pd.to_datetime(Lockdown['Date'])   
data = pd.merge(data,Lockdown,how='outer',on=['Date'])       # merge dataframes
data['Lockdown-Strength'].fillna(method='ffill',inplace=True) # fill the missing data by using preceding values 
data.dropna(inplace=True)

################## save dataframe to csv ##################


data.to_csv('data.csv',index = False)
################## end of preprocessing ##################

index = pd.to_datetime(data['Date'])

plt.subplots()
plt.plot(index, data['Cases2'])
plt.plot(index, data['Incidince']*10)
plt.plot(index, data['Intensive_Care'])
plt.title("Covid-19 in Germany")
plt.ylabel("Cases")
plt.xlabel('Time')
plt.legend(['Daily Cases', '7 Day Incidinces x10', 'Intensive_Care'])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
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
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Covid-Data-Age-RValue.png')

plt.subplots()
plt.plot(index, data['1rst_Vac'])
plt.plot(index, data['2nd_Vac'])
plt.title("Covid-19 in Germany")
plt.ylabel("Vaccinations")
plt.xlabel('Time')
plt.legend(['1rst Vaccination','2nd Vaccination'])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Covid-Data-Vaccinations.png')

print(data)