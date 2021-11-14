################## preamble ##################

# data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

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
    'https://raw.githubusercontent.com/robert-koch-institut/'
    +'SARS-CoV-2-Nowcasting_und_'+
    '-R-Schaetzung/main/Nowcast_R_aktuell.csv'
    )
data_Hosp = pd.read_csv(
    'https://raw.githubusercontent.com/robert-koch-institut/'
    +'COVID-19-Hospitalisierungen_in_Deutschland/master/'
    +'Aktuell_Deutschland_COVID-19-Hospitalisierungen.csv', 
    usecols=['Datum', '7T_Hospitalisierung_Faelle'])
data_infections = pd.read_csv(
    'https://github.com/robert-koch-institut/'
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
data_all = data_all[['Meldejahr', 'MW', 'Mittelwert Alter (Jahre)', 'Männer']]
data_all.rename(columns={'Meldejahr':'Year', 'MW':'Week', 'Mittelwert Alter (Jahre)':'Age',
         'Männer':'Gender'},inplace=True)        
data_all['Year'].replace(2022, 2021, inplace = True)
data_vaccinations = data_vaccinations[['Datum', 'Erstimpfung', 'Zweitimpfung','Auffrischimpfung']].rename(
    columns={'Datum': 'Date', 'Erstimpfung': '1rst_Vac', 'Zweitimpfung' : '2nd_Vac', 'Auffrischimpfung' : 'Booster'})
data_DIVI = data_DIVI[[
    'Datum',
    'Aktuelle_COVID_Faelle_Erwachsene_ITS'
    ]].rename(columns={
    'Datum':'Date', 
    'Aktuelle_COVID_Faelle_Erwachsene_ITS':'Intensive_Care'
    })
data_infections = data_infections[[
                    'Meldedatum',
                    'AnzahlFall',
                    'AnzahlTodesfall'
                    ]].rename(columns = {
                        'Meldedatum':'Date',
                        'AnzahlFall':'Cases', 
                        'AnzahlTodesfall':'Deaths'})
data_Hosp.rename(columns={'Datum': 'Date', '7T_Hospitalisierung_Faelle': 'Hospitalization'}, inplace=True)

################# bring data into daily format ##################
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
data_Hosp['Date'] = pd.to_datetime(data_Hosp['Date'])
data_vaccinations['Date'] = pd.to_datetime(data_vaccinations['Date'],format='%d.%m.%Y')
# get rid of the utc datetime format from DIVI to set it to a datetime:
new_dates = list()
for i in range(len(data_DIVI)):
    new_dates.append(data_DIVI['Date'][i][0:10])
data_DIVI['Date'] = pd.DataFrame(new_dates)
data_DIVI['Date'] = pd.to_datetime(data_DIVI['Date'],format='%Y-%m-%d')
data_infections['Date'] = pd.to_datetime(data_infections['Date'],format='%Y-%m-%d')


# sort DIVI data
data_DIVI_sort = pd.DataFrame(index=data_DIVI['Date'].unique(), columns=['Intensive_Care']).rename_axis('Date')
for day in data_DIVI['Date'].unique():
    data_DIVI_sort['Intensive_Care'][day] = data_DIVI[(data_DIVI['Date']==day)]['Intensive_Care'].sum()/2
data_DIVI_sort['Date'] = data_DIVI_sort.index
data_DIVI_sort.reset_index(drop=True, inplace=True)

# sort Hospitalization data
data_Hosp_sort = pd.DataFrame(index=data_Hosp['Date'].unique(), columns=['Hospitalization']).rename_axis('Date').sort_index()
for day in data_Hosp['Date'].unique():
    data_Hosp_sort['Hospitalization'][day]=data_Hosp[
        (data_Hosp['Date']==day)]['Hospitalization'].sum()/3
data_Hosp_sort['Date'] = data_Hosp_sort.index
data_Hosp_sort.reset_index(drop=True, inplace=True)
# drop last days because RKI data are delayed
data_Hosp_sort.drop(list(range(len(data_Hosp_sort)-4,
    len(data_Hosp_sort))), inplace = True) 

# sort infection data
data_infections_sort = pd.DataFrame(index=data_infections['Date'].unique(), columns=['Cases','Deaths']).rename_axis('Date').sort_index()
for day in data_infections['Date'].unique():
    data_infections_sort['Cases'][day]=data_infections[
        (data_infections['Date']==day)]['Cases'].sum()
    data_infections_sort['Deaths'][day]=data_infections[
        (data_infections['Date']==day)]['Deaths'].sum()
data_infections_sort['Date'] = data_infections_sort.index

# 7 day Incidence per 100.000:
Incidence = list()
for i in range(len(data_infections_sort)+1):
    if i < 6:
        inc = 0
        Incidence.append(inc)
    if i  > 6:
        inc = data_infections_sort['Cases'][i-7:i].sum()/831
        Incidence.append(inc)
data_infections_sort.reset_index(drop=True, inplace=True)
data_infections_sort['Incidence'] = pd.DataFrame(Incidence)
# drop last days because RKI data are delayed
data_infections_sort.drop(list(range(len(data_infections_sort)-3,
   len(data_infections_sort))), inplace = True) 

data_date = {'Date' : pd.date_range(start='2020-01-01',
                                end=date.today().strftime("%Y-%m-%d"), 
                                  freq='D')}
data_date= pd.DataFrame(data_date)
###################### merge dataframes on 'Date' ############################
dfs = [
        data_date,
        data_all,
        data_RWert,
        data_vaccinations,
        data_DIVI_sort,
        data_infections_sort,
        data_Hosp_sort
        ]
data = [df.set_index('Date',drop = True) for df in dfs]
data = pd.DataFrame(data[0].join(data[1:],how ='left'))
#data.drop(['Date_x','Date_y','Date'],axis=1,inplace=True)
data['Date'] = data.index
data.reset_index(drop=True, inplace=True)
data.drop(range(366,376),inplace = True)
data.reset_index(drop=True, inplace=True)

#  fill, interpolate and smooth data:
data[['Year','Week']] = data[['Year','Week']].fillna(method='ffill',axis=0)  # fill missing week and year numbers
data[['1rst_Vac', '2nd_Vac', 'Booster']] = data[['1rst_Vac', '2nd_Vac', 'Booster']].fillna(value = 0, axis=0).cumsum()/831292.85  # fill missing vaccination numbers with zero
data[['Gender','R-value']] = data[['Gender','R-value']].interpolate(method = 'linear', limit_direction = 'forward')
data[data.columns[0:-1]] = data[data.columns[0:-1]].interpolate(method = 'spline', axis = 0, order = 1, limit_direction ='forward').ffill()  # interpolate missing data from weekly dataframes
data['Hospitalization'] = data['Hospitalization'].rolling(7).sum()/7 # smoothen
data['Deaths'] = data['Deaths'].rolling(7).sum()/7 # smoothen
data['Age'] = data['Age'].rolling(7).sum()/7 # smoothen
data['Gender'] = data['Gender'].rolling(7).sum()/7 # smoothen
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
                    ('2021-08-14', '1'),
                    ('2021-08-26', '2'),
                    ('2021-11-07', '3')
                    ],       
           columns=('Date', 'Lockdown-Intensity')
                 )
Lockdown['Date'] = pd.to_datetime(Lockdown['Date'])
Lockdown.set_index('Date', inplace = True)
data.set_index('Date', inplace = True)
data = data.join(Lockdown)     
data['Lockdown-Intensity'].fillna(method='ffill',inplace=True) # fill the missing data by using preceding values 
data['Date'] = data.index
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
#data['Age + 2ndVac'] = data['Age']/max(data['Age'])+data['2nd_Vac']/max(data['2nd_Vac'])
data.to_csv('data_all.csv',index = False)
# select relevant columns:
data_final = data[[
    'Date',
    #'Year',
    'Week',
    #'R-value',
    'Age',
    'Gender', 
    #'Cases',
    #'Hospitalization',
    'Incidence',
    'Intensive_Care',
    'Deaths',
    #'1rst_Vac',
    #'2nd_Vac',
    #'Age + 2ndVac',
    #'Booster',#
    'Lockdown-Intensity'
       ]]
split = 0.8
data_train = data_final[0:int(split*len(data))]
data_test = data_final[int(split*len(data)):]

################## save dataframe to csv ##################
data_final.to_csv('data.csv',index = False)
data_train.to_csv('data_train.csv',index = False)
data_test.to_csv('data_test.csv',index = False)
################## end of preprocessing ##################

index = pd.to_datetime(data['Date'])

################## save figures ##########################
plt.rcParams.update({'font.size': 12})

plt.subplots()
#plt.plot(index, data['Cases'])
plt.plot(index, data['Incidence'])
plt.ylabel('Cases')
plt.xlabel('Date')
plt.legend(['7 Day Incidences'])
plt.tight_layout()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Data_Graphs\Covid-Data-Cases.png')

plt.subplots()
#plt.plot(index, data['Cases'])
plt.plot(index, data['Hospitalization'])
plt.plot(index, data['Intensive_Care'])
plt.plot(index, data['Deaths']*10)
plt.ylabel('Cases')
plt.xlabel('Date')
plt.legend(['Hospitalizations', 'Intensive_Care', 'Deaths x 10'])
ax = plt.gca()
plt.tight_layout()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Data_Graphs\Covid-Data-Cases2.png')

plt.subplots()
plt.plot(index, data['R-value'])
plt.ylabel('R-Value')
plt.xlabel('Date')
plt.legend(['R Value'])
ax = plt.gca()
plt.tight_layout()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Data_Graphs\Covid-Data-Age-RValue.png')

plt.subplots()
plt.plot(index, data['1rst_Vac'])
plt.plot(index, data['2nd_Vac'])
plt.plot(index, data['Booster'])
plt.ylabel('Vaccinations [%]')
plt.xlabel('Date')
plt.legend(['1rst Vaccination','2nd Vaccination', 'Booster'])
ax = plt.gca()
plt.tight_layout()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Data_Graphs\Covid-Data-Vaccinations.png')

plt.subplots()
plt.plot(index, data['Age'])
plt.plot(index, data['Gender']*100)
plt.ylabel('Mean Age/ Gender Ratio [%]')
plt.xlabel('Date')
plt.legend(['Age', 'Gender'])
ax = plt.gca()
plt.tight_layout()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Data_Graphs\Covid-Data-Gender.png')

plt.subplots()
plt.plot(index, data['Intensive_Care']/19574)
plt.ylabel('ITS Belegung [%]')
plt.xlabel('Date')
plt.legend(['ITS'])
ax = plt.gca()
plt.tight_layout()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Data_Graphs\Covid-Data-ITS.png')

########### plot correlation matrix ##################
corr = data_final.corr().round(1)
corr.style.background_gradient(cmap='PuOr')
fig,ax = plt.subplots()
sns.heatmap(corr, annot = True)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('Figures\Feature_Analysis\correlation.png')
plt.close()