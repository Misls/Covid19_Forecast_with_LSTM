# train LSTM neural networks based on data.csv and predict further trend


################## preamble ##################

# data analysis:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

# setup neural network
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter

# save models
import pickle
import matplotlib.dates as mdates


################## prepare time series data ##################

# parameters:
dim = 1 # number of features in LSTM (dim >1 if more than 1 column is used for training)
fut_pred = 90 # how many days should be predicted
train_window = 14
epochs = 50
hidden_layers = 200
drop = 0.1
num_layers = 2
lr = 0.001

# define cumpuation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('This Computation is running on {}'.format(device))

# load dataframe
data = pd.read_csv('data.csv')
#data = data[['Date', 'Year','Week','Cases','Lockdown-Intensity']]
# help function 
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def auto_save(loss,i,col):
    saved_epoch = list()
    if i == 0:
        print('saved epoch {} with loss {}' .format(i,loss))
        Pkl_Filename = 'LSTM-Models\LSTM-'+col+'.pkl'  
        saved_epoch = i
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(model, file)
        return saved_epoch

    if i >= 5 and loss[-1]<min(loss[4:-1]):
            print('saved epoch {} with loss {}' .format(i,loss[-1]))
            Pkl_Filename = 'LSTM-Models\LSTM-1'+col+'.pkl'  
            saved_epoch = i
            with open(Pkl_Filename, 'wb') as file:  
                pickle.dump(model, file)
    return saved_epoch


################## LSTM class ##################

# define Long Short Term Memory Network (LSTM):
class LSTM(nn.Module):
    def __init__(self, input_size=dim, hidden_layer_size=hidden_layers):

        super().__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, dropout = drop)
    

        self.linear = nn.Linear(hidden_layer_size, input_size)

        self.hidden_cell = (torch.zeros(num_layers,1,self.hidden_layer_size),
                            torch.zeros(num_layers,1,self.hidden_layer_size))

        #self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        self.lstm.flatten_parameters()
        inpt = input_seq.view(len(input_seq) ,1, -1)
        lstm_out, self.hidden_cell = self.lstm(inpt, self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))

        return predictions[-1]
################## initialze output dataframe ##################

# date column for prediction:
dates = pd.to_datetime(data['Date'])
last_date = dates.loc[len(dates)-1]  
dateList = []
for x in range (0, fut_pred):
    dateList.append(last_date + datetime.timedelta(days = x+1))
pred_dates = pd.to_datetime(dateList)
pred_dates = np.array(pred_dates)
df_pred = pd.DataFrame(data=pred_dates, columns = ['Date'])
df_pred['Year'] = df_pred['Date'].dt.year
df_pred['Week'] = df_pred['Date'].dt.isocalendar().week
# drop unimportant features for the training:
df = data.drop(['Date','Year', 'Week','Lockdown-Intensity'],axis=1)

#########################################################
################## start training loop ##################
#########################################################

for col in df.columns:
    start_time = time.time()
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

# sequence and labels:
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
# initialize model:
    model = LSTM().to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('>>>train feature: %s' % (col))
    print('model: %s' % (model))

    loss_summary = list()
    writer = SummaryWriter() # for Tensorboard
################## start learning for a feature ##################
  
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                            torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))
       
            y_pred = model(seq.to(device))
            y_pred = y_pred.view(-1,dim)

            single_loss = loss_function(y_pred.to(device), labels.to(device))
           
            single_loss.backward()
            optimizer.step()           
################## loss visualization ##################

        writer.add_scalar("Loss/train", single_loss, i) # for TensorBoard
        writer.flush() # for TensorBoard
        loss_summary.append(single_loss.item())
        saved_epoch = auto_save(loss_summary,i,col) # auto save best run
        saved_epochs.append(saved_epoch)    
        saved_epochs = [ele for ele in saved_epochs if ele != []] # remove empty elements from list   
    #print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')  
    # save the loss plot
    fig, ax = plt.subplots()
    plt.title(''+col+'-Loss')
    plt.ylabel('Loss')
    ax.set_yscale('log')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.plot(loss_summary)
    plt.savefig('Figures\Prediction-loss-1'+col+'.png')
    
################## predict data ##################

    test_inputs = train_data_normalized[-train_window:].tolist()
    # load best epoch
    #best_epoch = saved_epochs[-1]
    Pkl_Filename = Pkl_Filename = 'LSTM-Models\LSTM-1'+col+'.pkl' 
    with open(Pkl_Filename, 'rb') as file:  
        model = pickle.load(file)

    model.eval()
    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                            torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))       
      
            test_inputs.append(model(seq.to(device)).detach().cpu().numpy().tolist())
    actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, dim))
    # save the forcast into dataframe
    df_pred[col]= pd.DataFrame(actual_predictions)   
    # plot the forecast:
    #x = np.arange(len(train_data), len(train_data)+fut_pred, 1)  
    # save the forecast plot
    plt.subplots()
    plt.title(col)
    plt.ylabel('Value')
    plt.xlabel('Date')
    plt.grid(True)
    #plt.autoscale(axis='x', tight=True)
    plt.plot(dates,data[col])
    plt.plot(pred_dates,actual_predictions[:,0])
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gcf().autofmt_xdate() # Rotation
    plt.savefig('Figures\Prediction-1'+col+'.png')
   
    elapsed_time = float("{:.0f}".format(time.time() - start_time))
    print('elapsed time for feature training: %s' % (str(datetime.timedelta(seconds=elapsed_time))))

#########################################################
################## end training loop ####################
#########################################################

# save the results to data_pred as possible input for prediction 
df_pred.to_csv('data_pred1.csv',index = False)