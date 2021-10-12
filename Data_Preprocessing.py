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
    'https://raw.githubusercontent.com/robert-koch-institut/'
    +'SARS-CoV-2-Nowcasting_und_'+
    '-R-Schaetzung/main/Nowcast_R_aktuell.csv'
    )
data_infections = pd.read_csv('https://github.com/robert-koch-institut/'
    +'SARS-CoV-2_Infektionen_in_Deutschland/blob/master/'+
    'Aktuell_Deutschland_SarsCov2_Infektionen.csv?raw=true')

r = requests.get(
    'https://www.rki.de/DE/Content/InfAZ/N/'
    +'Neuartiges_Coronavirus/Daten/Impfquotenmonitoring.xlsx'+
    '?__blob=publicationFile', 
    headers={"User-Agent": "Chrome"}
    )
data_vaccinations = pd.read_excel(BytesIO(r.content),header=0,sheet_name = 'Impfungen_proTag')
data_vaccinations.dropna(inplace=True)
data_vaccinations.drop(len(data_vaccinations)-1, axis = 0, inplace = True)
#data_VOC = pd.read_csv('https://github.com/robert-koch-institut/'
#    +'SARS-CoV-2-Sequenzdaten_aus_Deutschland/'+
#    'raw/master/SARS-CoV-2-Sequenzdaten_Deutschland.csv.xz')

data_DIVI = pd.read_csv('https://diviexchange.blob.core.'
    +''+'windows.net/%24web/bundesland-zeitreihe.csv') # Intensivbettenbelegung

# excel sheet from weekly updated overview statistics provided by RKI 
# including hospitalization:
r = requests.get(
    'https://www.rki.de/DE/Content/InfAZ/N/'
    +'Neuartiges_Coronavirus/Daten/'+
    'Klinische_Aspekte.xlsx?__blob=publicationFile', 
    headers={"User-Agent": "Chrome"}
    )
data_all = pd.read_excel(BytesIO(r.content),header=2)

################## data preprocessing ##################

# filter for reasonable columns and rename them
data_RWert = data_RWert[['Datum', 'PS_7_Tage_R_Wert']].rename(columns={'Datum': 'Date', 'PS_7_Tage_R_Wert': 'R-value'})
data_all = data_all[['Meldejahr', 'MW', 'Mittelwert Alter (Jahre)', 'Männer', 'Anzahl hospitalisiert', 'Anzahl Verstorben']]
data_all.rename(columns={'Meldejahr':'Year', 'MW':'Week', 'Mittelwert Alter (Jahre)':'Age',
         'Männer':'Gender', 'Anzahl hospitalisiert':'Hospitalization', 'Anzahl Verstorben':'Deaths'},inplace=True)        
data_all['Year'].replace(2022, 2021, inplace = True)
data_vaccinations = data_vaccinations[['Datum', 'Erstimpfung', 'Zweitimpfung']].rename(
    columns={'Datum': 'Date', 'Erstimpfung': '1rst_Vac', 'Zweitimpfung' : '2nd_Vac'})
data_DIVI = data_DIVI[[
    'Datum',
    'Aktuelle_COVID_Faelle_Erwachsene_ITS'
    ]].rename(columns={
    'Datum':'Date', 
    'Aktuelle_COVID_Faelle_Erwachsene_ITS':'Intensive_Care'
    })
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
# get rid of the ugly utc datetime format from DIVI to set it to a datetime:
new_dates = list()
for i in range(len(data_DIVI)):
    new_dates.append(data_DIVI['Date'][i][0:10])
data_DIVI['Date'] = pd.DataFrame(new_dates)
data_DIVI['Date'] = pd.to_datetime(data_DIVI['Date'],format='%Y-%m-%d')
data_infections['Date'] = pd.to_datetime(data_infections['Date'],format='%Y-%m-%d')

# sort DIVI data
data_DIVI_sort = pd.DataFrame(index=data_DIVI['Date'].unique(), columns=['Intensive_Care']).rename_axis('Date')
for day in data_DIVI['Date'].unique():
    data_DIVI_sort['Intensive_Care'][day] = data_DIVI[(data_DIVI['Date']==day)]['Intensive_Care'].sum()

# sort infection data
data_infections_sort = pd.DataFrame(index=data_infections['Date'].unique(), columns=['Cases']).rename_axis('Date').sort_index()
for day in data_infections['Date'].unique():
    data_infections_sort['Cases'][day]=data_infections[
        (data_infections['Date']==day)]['Cases'].sum()
data_infections_sort['Date'] = data_infections_sort.index

# 7 day incidince per 100.000:
incidince = list()
for i in range(len(data_infections_sort)+1):
    if i < 6:
        inc = 0
        incidince.append(inc)
    if i  > 6:
        inc = data_infections_sort['Cases'][i-7:i].sum()/831
        incidince.append(inc)
data_infections_sort.reset_index(drop=True, inplace=True)
data_infections_sort['Incidince'] = pd.DataFrame(incidince)

###################### merge dataframes on 'Date' ############################
data = pd.merge(data_RWert,data_all,how='outer',on=['Date'])
data = pd.merge(data,data_vaccinations,how='outer',on=['Date'])
data = pd.merge(data,data_DIVI_sort,how='outer',on=['Date'])
data = pd.merge(data,data_infections_sort,how='outer',on=['Date'])

data.set_index('Date', inplace = True) # sort_values didn't work, so here we go with sort_index
data.rename_axis('Date_index', inplace = True)
data.sort_index(inplace = True)

#  fill, interpolate and smooth data:
data[['Year','Week']] = data[['Year','Week']].fillna(method='ffill',axis=0)  # fill missing week and year numbers
data[['1rst_Vac', '2nd_Vac']] = data[['1rst_Vac', '2nd_Vac']].fillna(value = 0, axis=0).cumsum()  # fill missing vaccination numbers with zero
data[data.columns[0:-1]] = data[data.columns[0:-1]].interpolate(method = 'spline',order = 3, axis = 0)  # interpolate missing data from weekly dataframes
data['Age'] = data['Age'].rolling(7).sum()/7 # smoothen
data['Gender'] = data['Gender'].rolling(7).sum()/7 # smoothen
data.drop(data.loc[data['Cases']<0].index,inplace=True)  # drop unrealistic case numbers from interpolation
data[['Intensive_Care']]=data[['Intensive_Care']].fillna(0)
data.drop(data.loc[data['Hospitalization']<0].index,inplace=True) # drop unrealistic case numbers from interpolation
data.drop(data.loc[data['R-value']>1.6].index,inplace=True) # drop unrealistic R-Values from first days calculations
data['Date'] = data.index # restore Data column
data.reset_index(drop=True, inplace=True)
# correct data:
data[['Hospitalization']] = data[['Hospitalization']]/7

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
# and based on https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus
# /Downloads/Stufenplan.pdf?__blob=publicationFile
# and categorized in 4 classes. Dates indicate a change. 
Lockdown = pd.DataFrame([
                    ('2020-01-01', '0'),
                    ('2020-03-08', '1'),
                    ('2020-03-17', '2'),
                    ('2020-03-22', '3'),
                    ('2020-05-06', '1'),
                    ('2020-10-28', '2'),
                    ('2020-12-13', '3'),
                    ('2021-03-03', '1'), 
                    ('2021-04-23', '2'),
                    ('2021-06-20', '0'),
                    # from here dates are based on RKI recommendations
                    ('2021-07-19', '1'),
                    ('2021-08-14', '2'),
                    ('2021-08-28', '3'),
                    ('2021-09-21', '2'),
                    ('2021-10-01', '1')],       
           columns=('Date', 'Lockdown-Intensity')
                 )
Lockdown['Date'] = pd.to_datetime(Lockdown['Date'])   
data = pd.merge(data,Lockdown,how='outer',on=['Date'])       # merge dataframes
data['Lockdown-Intensity'].fillna(method='ffill',inplace=True) # fill the missing data by using preceding values 
data['Date'] = pd.to_datetime(data['Date'])

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
data.drop(list(range(len(data)-2,len(data))), inplace = True) # drop the last days because RKI data are delayed for about 2 days

# select relevant columns:
data_final = data[[
    'Date',
    #'Year',
    #'Week',
    'Age',
    'R-value', 
    #'Cases',
    'Hospitalization',
    #'Incidince',
    'Intensive_Care', 
    'Gender',
    #'Deaths',
    #'1rst_Vac', 
    '2nd_Vac', 
    'Lockdown-Intensity'
       ]]

################## save dataframe to csv ##################

data_final.to_csv('data.csv',index = False)
################## end of preprocessing ##################

index = pd.to_datetime(data['Date'])

plt.subplots()
plt.plot(index, data['Cases'])
plt.plot(index, data['Incidince']*100)
plt.plot(index, data['Intensive_Care'])
plt.title("Covid-19 in Germany")
plt.ylabel("Cases")
plt.xlabel('Date')
plt.legend(['Daily Cases', '7 Day Incidinces x 100', 'Intensive_Care'])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Data_Graphs\Covid-Data-Cases.png')

plt.subplots()
plt.plot(index, data['R-value']*50)
plt.plot(index, data['Age'])
plt.title("Covid-19 in Germany")
plt.ylabel("Mean Age / R Value")
plt.xlabel('Date')
plt.legend(['R Value x 50','Mean Age'])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Data_Graphs\Covid-Data-Age-RValue.png')

plt.subplots()
plt.plot(index, data['1rst_Vac'])
plt.plot(index, data['2nd_Vac'])
plt.title("Covid-19 in Germany")
plt.ylabel("Vaccinations")
plt.xlabel('Date')
plt.legend(['1rst Vaccination','2nd Vaccination'])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Data_Graphs\Covid-Data-Vaccinations.png')