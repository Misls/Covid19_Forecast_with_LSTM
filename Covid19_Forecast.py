import pandas as pd
import numpy as np

from pandas import read_csv

data_RWert = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung/blob/0c3a7b18078ca50f81b9002976801bb826a77197/Nowcast_R_aktuell.csv?raw=true')
data_infections = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2_Infektionen_in_Deutschland/blob/master/Aktuell_Deutschland_SarsCov2_Infektionen.csv?raw=true')
data_vaccinations = pd.read_csv('https://github.com/robert-koch-institut/COVID-19-Impfungen_in_Deutschland/blob/master/Aktuell_Deutschland_Bundeslaender_COVID-19-Impfungen.csv?raw=true')
data_VOC = pd.read_csv('https://github.com/robert-koch-institut/COVID-19-Impfungen_in_Deutschland/blob/master/Aktuell_Deutschland_Bundeslaender_COVID-19-Impfungen.csv?raw=true')

print(data_VOC.head(20))