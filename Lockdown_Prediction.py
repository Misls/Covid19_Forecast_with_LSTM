# predict Lockdown-Intensity depending on trained Lockdown-Classifier and forecast by LSTM

################## preamble##################

# data analysis:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from cycler import cycler

# predict timeseries
import torch
import torch.nn as nn
import datetime

# load model
import pickle

# smooth parameter
smooth = 3 # interval of days for rolling mean
# new prediction on current data with saved RNN
new_prediction = True
plt.rcParams.update({'font.size': 12})

# hyperparameter
fut_pred = 90 # how many days should be predicted
train_window = 14
dim = 1 # number of features in LSTM (dim >1 if more than 1 column is used for training)
hidden_layers =250
drop = 0.2
num_layers = 3
batch_size = 1

# define Long Short Term Memory Network (LSTM):
class LSTM(nn.Module):
    def __init__(self, input_size=dim, hidden_layer_size=hidden_layers):

        super().__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, dropout = drop)        
        self.linear = nn.Linear(hidden_layer_size, input_size)

        self.hidden_cell = (torch.zeros(
            num_layers,
            batch_size,
            self.hidden_layer_size),
                            torch.zeros(
                                num_layers,
                                batch_size,
                                self.hidden_layer_size))     
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        self.lstm.flatten_parameters()
        inpt = input_seq.view(len(input_seq) ,1, -1)
        lstm_out, self.hidden_cell = self.lstm(inpt, self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# load Lockdown-Classifier
Pkl_Filename = 'Lockdown_Classifier.pkl'  
with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

# load data
data_pred = pd.read_csv('data_pred.csv')
pred_dates = pd.to_datetime(data_pred['Date'])
data_current = pd.read_csv('data.csv')#.drop('Age + 2ndVac', axis = 1)
current_dates = pd.to_datetime(data_current['Date'])

################### start a new prediction on current data ######################
if new_prediction:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('This Computation is running on {}'.format(device))

    data_pred = pd.DataFrame()
    df = data_current.drop(['Date',
                'Week',
                'Lockdown-Intensity'],axis=1)
    
    dates = pd.to_datetime(data_current['Date'])
    last_date = dates.loc[len(dates)-1]  
    dateList = []
    for x in range (0, fut_pred):
        dateList.append(last_date + datetime.timedelta(days = x+1))
    pred_dates = pd.to_datetime(dateList)
    pred_dates = np.array(pred_dates)
    pred_dates = pd.DataFrame(data=pred_dates, columns = ['Date'])


    data_pred['Week'] = pred_dates['Date'].dt.isocalendar().week


    for col in df.columns:
        saved_epochs = list()
        df_temp = df[col].values.astype(float) # define dataset for training   
################## prepare test and train data ##################

        test_data_size = 90
        train_data = df_temp[:]
        test_data = df_temp[-test_data_size:]
    # normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data_normalized = scaler.fit_transform(train_data.reshape(-1, dim))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1,dim)
       
        # load best epoch
        Pkl_Filename = Pkl_Filename = 'LSTM-Models\LSTM-'+col+'.pkl' 
        with open(Pkl_Filename, 'rb') as file:  
            model = pickle.load(file)
            test_inputs = train_data_normalized[-train_window:].tolist()

        model.eval()
        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[-train_window:])
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(
                    num_layers,
                    batch_size,
                    model.hidden_layer_size).to(device),
                                torch.zeros(
                                num_layers,
                                batch_size,
                                model.hidden_layer_size).to(device))         
                test_inputs.append(
                model(seq.to(device)).detach().cpu().numpy().tolist())
        actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, dim))
        # save the forcast into dataframe
        data_pred[col]= pd.DataFrame(actual_predictions)

        plt.subplots()
        plt.title(col)
        plt.ylabel('Value')
        plt.xlabel('Date')
        plt.grid(True)
        #plt.autoscale(axis='x', tight=True)
        plt.plot(current_dates,data_current[col])
        plt.plot(pred_dates,actual_predictions[:,0])
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        plt.gcf().autofmt_xdate() # Rotation
        plt.savefig('Figures\Prdiction_Graphs_Current\Prediction_'
                    +col+'.png')
        plt.close()
############## end of prediction loop ##############################
#data_all = pd.read_csv('data_all.csv')
data = data_current.append(data_pred)
#data['Age + 2ndVac'] = data_all['Age']*data_all['2nd_Vac']
X = data.drop(['Date','Lockdown-Intensity'],axis=1)
for col in X.columns:
    X[col] = (X[col]-min(data_current[col]))/max(data_current[col])
#X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
y_pred = Pickled_Model.predict_proba(X)
y_pred = pd.DataFrame(y_pred).rolling(smooth).sum()/smooth
y_pred.drop(range(len(data_current)), inplace = True)

index = pred_dates
plt.rc('axes', prop_cycle=(cycler(color=['lightsteelblue', 'lime', 'darkorange', 'red']) 
                            + cycler(linestyle=['-','-','-','-']))
                           )
fig, ax = plt.subplots()
plt.plot(index, y_pred)
plt.title("Probabilities for SARS-CoV-2 Measures")
plt.ylabel("Probability")
plt.xlabel('Date')
plt.legend(['No Lockdown','Light', 'Middle', 'Hard'])
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate() # Rotation
plt.savefig('Figures\Prdiction_Graphs_Current\Lockdown_Probability.png')
plt.close()