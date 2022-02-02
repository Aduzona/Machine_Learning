#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 21:11:55 2020

@author: haythamomar
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


apparel= pd.read_csv('apparel.csv')

apparel['revenue']= apparel['price_paid']*apparel['Qty']
apparel['profit']= apparel['revenue']- (apparel['Cost']*apparel['Qty'])
apparel.info()

apparel['Date']= pd.to_datetime(apparel['Date'])
### a week and a year column

apparel['week']= apparel['Date'].dt.week
apparel['month']= apparel['Date'].dt.month

sub_family_grouped= apparel.groupby(['Date','subfamily'])['Qty'].sum().reset_index()


subfamilies= np.unique(apparel.subfamily)


#### spreading

spreaded= pd.pivot_table(sub_family_grouped,values='Qty',index='Date',columns='subfamily')

spreaded=spreaded.fillna(0)

spreaded.plot()


spreaded_weekly=spreaded.resample('W').sum()


spreaded_weekly.plot()
spreaded_weekly.shape
213*0.8

spreaded_weekly['trend']= range(0,spreaded_weekly.shape[0])


spreaded_weekly.reset_index(inplace=True)




spreaded_weekly.info()

spreaded_weekly['month']=spreaded_weekly['Date'].dt.month
spreaded_weekly['week']=spreaded_weekly['Date'].dt.week

spreaded_weekly['month']=spreaded_weekly['month'].astype('category')
spreaded_weekly['week']=spreaded_weekly['week'].astype('category')


### trend, week, month

independant= pd.get_dummies(spreaded_weekly)

independant.columns
subfamilies


x_train= independant.iloc[0:171,9:].values
x_test=independant.iloc[171:,9:].values

perf_train= independant.iloc[0:171,1].values
perf_test= independant.iloc[171:,1].values

model= LinearRegression()

model.fit(x_train, perf_train)

model.score(x_train, perf_train)


test_prediction= model.predict(x_test)

mae= np.mean(abs(perf_test-test_prediction))


###plotting performance

perfumes= spreaded_weekly.loc[:,['Date','Pefumes']].set_index('Date')

##TRAINING 

train_fitting= model.predict(x_train)

perfumes['prediction']= np.append(train_fitting,test_prediction)

perfumes.plot()

perfumes.iloc[0:171,:]


def forecast(subfamily_name):
    x_train= independant.iloc[0:171,9:].values
    x_test=independant.iloc[171:,9:].values

    train= independant.loc[0:170,subfamily_name].values
    test= independant.loc[171:,subfamily_name].values
    model= LinearRegression()

    model.fit(x_train, train)

    score=model.score(x_train,train)
    test_prediction= model.predict(x_test)

    mae= np.mean(abs(test-test_prediction))
    train_fitting= model.predict(x_train)
    comparison= spreaded_weekly.loc[:,['Date',subfamily_name]].set_index('Date')


    comparison['prediction']= np.append(train_fitting,test_prediction)

    comparison.plot()
    results_dict={ 'measure': mae,'results_data': comparison,'score': score}
    return results_dict

subfamilies

forecast('underwear')    



## forecasting 2020
spreaded_weekly

spreaded_weekly['Date']= pd.date_range(start='2016-01-03',periods= 213,freq='W')

dates= pd.DataFrame({'Date':pd.date_range(start='2020-02-03',periods=52,freq='W')})


spreaded_weekly= pd.concat([spreaded_weekly,dates],axis= 0)

spreaded_weekly.reset_index(inplace=True)



spreaded_weekly['trend']= range(0,spreaded_weekly.shape[0])



spreaded_weekly['month']=spreaded_weekly['Date'].dt.month
spreaded_weekly['week']=spreaded_weekly['Date'].dt.week

spreaded_weekly['month']=spreaded_weekly['month'].astype('category')
spreaded_weekly['week']=spreaded_weekly['week'].astype('category')


### trend, week, month

independant= pd.get_dummies(spreaded_weekly)

independant.drop('index',inplace=True,axis=1)
independant.to_csv('historical_retail_data.csv')



def forecast_future(subfamily_name):
    x_train= independant.iloc[0:213,9:].values
    x_test=independant.iloc[213:,9:].values

    train= independant.loc[0:212,subfamily_name].values
    model= LinearRegression()

    model.fit(x_train, train)

    score=model.score(x_train,train)
    test_prediction= model.predict(x_test)

    train_fitting= model.predict(x_train)
    comparison= spreaded_weekly.loc[:,['Date',subfamily_name]].set_index('Date')


    comparison['prediction']= np.append(train_fitting,test_prediction)

    comparison.plot()
    results_dict={ 'results_data': comparison,'score': score}
    return results_dict

subfamilies
forecast_future('jackets')

forecasting_linear_data= pd.DataFrame()
for subfamily in subfamilies:
    a= forecast_future(subfamily)
    a=a['results_data']
    forecasting_linear_data= pd.concat([forecasting_linear_data,a],axis=1)


forecasting_linear_data.to_excel('forecasting_linear.xlsx')











