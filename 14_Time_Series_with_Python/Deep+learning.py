#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

#### many inouts that resulted into one output . 

### neural networks , we have hidden layers

## flows input data * wights of the hidden layers , weight if output output 

input_data= np.array([12,3,65])
node1_weights= np.array([2,3,4])
node2_weights=np.array([3,5,-9])
node3_weights= np.array([2,4,2])
output_weights= np.array([4,4,2])



#### forward propgation


node1= (input_data*node1_weights).sum()
node2= (input_data*node2_weights).sum()
node3=(input_data*node3_weights).sum()

hidden_layer_output= np.array([node1,node2,node3])

output= (hidden_layer_output*output_weights).sum()


##actual 200 , not -632

def RELU(node_value):
    node= max(0,node_value)
    return node

##forward propgation

def predict(input_data,weights):
    node1= (input_data* weights['node1']).sum()
    node1_activated = RELU(node1)
    node2= (input_data* weights['node2']).sum()
    node2_activated = RELU(node2)
    node3= (input_data* weights['node3']).sum()
    node3_activated = RELU(node3)
    hidden_layer_output= np.array([node1_activated,node2_activated,node3_activated])
    output= (hidden_layer_output*weights['output']).sum()
    return output


input_data= np.array([12,3,65])
weights={ 'node1':np.array([2,3,4]),
          'node2':np.array([3,5,-9]),
          'node3':np.array([2,4,2]),
          'output': np.array([4,4,2])}


predict(input_data, weights)


#### back propgation 


def get_error(input_data,weights,target):
    prediction= predict(input_data,weights)
    error= prediction-target
    return error

get_error(input_data, weights, target)

def get_slope(input_data,target,weights):
    error= get_error(input_data, weights, target)
    slope= 2* input_data* error
    return slope

slope=get_slope(input_data, target, weights)

def get_mse(input_data,target,weights):
     error= get_error(input_data, weights, target)
     mse= error**2
     return error
    


import toolz

weights_updated= toolz.valmap(lambda x: x- (0.00001* slope), weights)


predict(input_data, weights_updated)

weights_list= []
mse_list= []
pred_list=[]

target=200
n_updates=100
learning_rate=0.01
for i in range(n_updates):
    pred= predict(input_data, weights)
    slope= get_slope(input_data, target, weights)
    weights=toolz.valmap(lambda x: x- (learning_rate* slope), weights)
    mse= get_mse(input_data, target, weights)
    pred_list.append(pred)
    mse_list.append(mse)
    
    
import matplotlib.pyplot as plt


plt.plot(mse_list)



from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
import os

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


sales= pd.read_csv('historical_retail_data.csv')

### training and testing

history=sales.iloc[:213,:]
    
history.drop(['Unnamed: 0','Date'],axis=1,inplace=True)    

history.columns
train_x= history.iloc[:170,8:].values
test_x= history.iloc[170:,8:].values

perf_train= history.iloc[:170,0].values
perf_test= history.iloc[170:,0].values


n_cols= (history.iloc[:170,8:].shape[1],)


model_ann= Sequential()

model_ann.add(Dense(100, input_shape= n_cols,activation='relu'))
model_ann.add(Dense(50, input_shape= n_cols,activation='relu'))

model_ann.add(Dense(1))

model_ann.compile(optimizer='adam',loss= 'mae')



#### fitting

hist= model_ann.fit(train_x,perf_train,epochs=300,validation_data= (test_x,perf_test),verbose= True,shuffle=False)


from sklearn.metrics import mean_squared_error

fitting= model_ann.predict(train_x)
testing= model_ann.predict(test_x)

mse_train= mean_squared_error(perf_train, fitting)
mse_testing= mean_squared_error(perf_test, testing)



plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()
 

train_x_rnn= history.iloc[:170,8:].values.reshape(history.iloc[:170,8:].shape[0],1,history.iloc[:170,8:].shape[1])
test_x_rnn= history.iloc[170:,8:].values.reshape(history.iloc[170:,8:].shape[0],1,history.iloc[170:,8:].shape[1])

perf_train= history.iloc[:170,0].values
perf_test= history.iloc[170:,0].values


model_RNN= Sequential()

model_RNN.add(LSTM(100, input_shape= (train_x_rnn.shape[1],train_x_rnn.shape[2])))

model_RNN.add(Dense(1))
model_RNN.compile(optimizer='adam',loss= 'mae')


hist= model_RNN.fit(train_x_rnn,perf_train,epochs=800,validation_data= (test_x_rnn,perf_test),verbose= True,shuffle=False)


from sklearn.metrics import mean_squared_error

fitting_rnn= model_RNN.predict(train_x_rnn)
testing_rnn= model_RNN.predict(test_x_rnn)

mse_train_rnn= mean_squared_error(perf_train, fitting_rnn)
mse_testing_rnn= mean_squared_error(perf_test, testing_rnn)




plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()
 

comparison= sales[['Date','Pefumes']]

comparison= comparison.iloc[0:213,:].set_index('Date')


train_ann= [ x[0] for x in fitting]
test_ann= [ x[0] for x in testing]

train_rnn= [ x[0] for x in fitting_rnn]
test_rnn= [ x[0] for x in testing_rnn]


comparison['Ann']= np.append(train_ann,test_ann)
comparison['Rnn']= np.append(train_rnn,test_rnn)

comparison.plot()


####Making a RNN function and a ANN function


def model_ann():
    model_ann= Sequential()

    model_ann.add(Dense(100, input_shape= n_cols,activation='relu'))
    model_ann.add(Dense(50, activation='relu'))
    model_ann.add(Dense(50,activation='relu'))


    model_ann.add(Dense(1))

    model_ann.compile(optimizer='adam',loss= 'mae')
    return model_ann

def model_rnn():
    model_RNN= Sequential()

    model_RNN.add(LSTM(300, input_shape= (train_x_rnn.shape[1],train_x_rnn.shape[2])))
    model_RNN.add(Dense(200))

    model_RNN.add(Dense(1))
    model_RNN.compile(optimizer='adam',loss= 'mae')
    return model_RNN
    


def forecast_nn(subfamily):
    train_ann= sales.loc[0:212,'trend':].values
    forecast_feat_ann= sales.loc[213:, 'trend':].values
    train_rnn=sales.loc[0:212,'trend':].values.reshape(213,1,66)
    forecast_feat_rnn=sales.loc[213:,'trend':].values.reshape(52,1,66)
    y= sales.loc[0:212,subfamily].values
    
    model1=model_ann()
    model2=model_rnn()
    hist_ann= model1.fit(train_ann,y,epochs=300,verbose=False, shuffle= False)
    hist_rnn= model2.fit(train_rnn,y,epochs=300,verbose=False, shuffle= False)

    comparison= sales.loc[:,['Date',subfamily]]
    comparison.set_index('Date',inplace=True)
    fitting_ann= model1.predict(train_ann)
    fitting_rnn= model2.predict(train_rnn)
    forecast_ann= model1.predict(forecast_feat_ann)
    forecast_rnn= model2.predict(forecast_feat_rnn)
    fitting_ann_list= [x[0] for x in fitting_ann]
    fitting_rnn_list= [x[0] for x in fitting_rnn]
    forecast_ann_list= [x[0] for x in forecast_ann]
    forecast_rnn_list= [x[0] for x in forecast_rnn]
    
    comparison['ann']= np.append(fitting_ann_list,forecast_ann_list)
    comparison['rnn']= np.append(fitting_rnn_list,forecast_rnn_list)
    
    comparison.plot()
    mse_ann= np.mean((comparison.loc[:'2020-01-26',subfamily]-comparison.loc[:'2020-01-26','ann'])**2)
    mse_rnn= np.mean((comparison.loc[:'2020-01-26',subfamily]-comparison.loc[:'2020-01-26','rnn'])**2)
    results= {'forecasts': comparison,'mse_ann': mse_ann,'mse_rnn':mse_rnn}

    return results

sales.columns
forecast_nn('pants')

subfamilies= sales.columns[2:10]

forecast_data= pd.DataFrame()
results= dict(list())

for subfamily in subfamilies:
     a= forecast_nn(subfamily)
     forecast_data= pd.concat([forecast_data,a['forecasts']],axis=1)
     results[subfamily]= [a['mse_ann'],a['mse_rnn']]
     
forecast_data.to_excel('forecasing_families.xlsx')






