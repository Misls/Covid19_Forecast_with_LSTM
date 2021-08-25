import pandas as pd
import numpy as np

from pandas import read_csv

data_online = pd.read_csv('https://github.com/robert-koch-institut/SARS-CoV-2-Nowcasting_und_-R-Schaetzung/blob/0c3a7b18078ca50f81b9002976801bb826a77197/Nowcast_R_aktuell.csv?raw=true')
data_local = pd.read_csv('../SARS-CoV-2-Nowcasting_und_-R-Schaetzung/Nowcast_R_aktuell.csv')
print(data_online.head(20))