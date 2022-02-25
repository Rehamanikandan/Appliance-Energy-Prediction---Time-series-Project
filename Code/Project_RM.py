import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.arima_model
import statsmodels.tsa.holtwinters as ets
from Toolbox import *
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from numpy import linalg as la
import warnings
warnings.filterwarnings('ignore')

#6.a: Preprocessing dataset
data=pd.read_csv('energydata_complete.csv')
print('Dataset:\n',data.head())
print('Dataset information:\n',data.info())
print('Dataset don\'t have null values')
data.date=pd.to_datetime(data.date)
data.set_index('date',inplace=True)
print('Let\'s look at the end of the data:\n',data.tail())

#6.b: dependant variable vs. time
print('Dependant variable:Appliances')
plt.plot(data.index,data.Appliances,label='Dependant variable')
plt.xlabel('Time')
plt.ylabel('Appliance Energy use in Wh')
plt.title('Plot of Dependant variable (Appliances) versus Time')
plt.xticks(data.index[::1000],rotation='vertical')
plt.tight_layout()
plt.legend()
plt.show()

#6c: ACF/PACF of dependent variable
ACF_PACF_Plot(data.Appliances,400)

print('ACF plot of dependent variable (symmetric)')
acf_data=ACF(data.Appliances,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(data.Appliances))
plt.stem(x,acf_data,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_data,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Original data')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#6d:Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient.
cor=data.corr()
sns.heatmap(cor,vmin=-1,vmax=1,center=0,cmap='PiYG')
plt.title("Heatmap of dataset", fontsize =20)
plt.show()

#6e:split the dataset into train, test set (80:20)
#split the entire dataset into train and test set
train,test=train_test_split(data,test_size=0.2,shuffle=False)

#7: stationarity test

#Check rolling mean and variance
cal_rolling_mean_var(data.Appliances,data.index)
#The mean and variance are having a flat plot but the mean is not at 0.
#It looks like data need to be differenced

#ADF test
ADF_Cal(data.Appliances)
#p-value is 0.000, the data is stationary

#kpss test
kpss_test(data.Appliances)
#Data is stationary

#seaonal Differencing is required on data to make the rolling mean at 0
#interval of 144 is used since data has spike at every 144 lags of ACF/PACF plot
diff=difference(data.Appliances,interval=144)
diff_df=pd.DataFrame(diff,index=data.index[144:])
diff_df.rename(columns={0:'Appliances'},inplace=True)
plt.plot(diff_df.index,diff_df.Appliances,label='Differenced')
plt.xlabel('Time')
plt.ylabel('Appliance Energy use in Wh')
plt.title('Plot of Dependant variable (differenced) versus Time')
plt.xticks(data.index[::1000],rotation='vertical')
plt.tight_layout()
plt.legend()
plt.show()

#ACF/PACF
ACF_PACF_Plot(diff_df,100)

#stationarity
cal_rolling_mean_var(diff_df,diff_df.index)
ADF_Cal(diff_df)
kpss_test(diff_df)
#The data is stationary

#8:Time series Decomposition
Appliances=data['Appliances']
Appliances=pd.Series(np.array(data['Appliances']),index = pd.date_range('2016-01-11 17:00:00',periods= len(Appliances)))
STL=STL(Appliances)
res=STL.fit()
fig=res.plot()
plt.xlabel("Iterations")
plt.tight_layout()
plt.show()

T=res.trend
S=res.seasonal
R=res.resid
plt.plot(T,label="Trend")
plt.plot(R,label="Residual")
plt.plot(S,label="Seasonal")
plt.xlabel("Iterations")
plt.ylabel("STL")
plt.legend()
plt.title("Trend, Seasonality and residuals of data")
plt.show()

#Strength of trend
var=1-(np.var(R)/np.var(T+R))
Ft=np.max([0,var])
print("Strength of trend:",Ft)

#Strength of seasonality
var1=1-(np.var(R)/np.var(S+R))
Fs=np.max([0,var1])
print("Strength of seasonality:",Fs)

#seasonally adjusted data
seasonally_adj=Appliances-S
plt.plot(data.index,data.Appliances,label="Original")
plt.plot(data.index,seasonally_adj,label="adjusted")
plt.xlabel("Time")
plt.xticks(data.index[::4000])
plt.ylabel("Appliances")
plt.title("Seasonally adjusted vs. Original")
plt.legend()
plt.show()

#detrended data
detrended=Appliances-T
plt.plot(data.index,data.Appliances,label="Original")
plt.plot(data.index,detrended,label="Detrended")
plt.xlabel("Time")
plt.xticks(data.index[::4000])
plt.ylabel("Appliances")
plt.title("Detrended vs. Original Data")
plt.legend()
plt.show()

# Holt-Winters method
print(data.columns)
X=data[['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4',
       'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9',
       'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility',
       'Tdewpoint', 'rv1', 'rv2']]
y=data['Appliances']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
holtw= ets.ExponentialSmoothing(y_train, damped_trend= True,trend='add', seasonal='add',seasonal_periods=144).fit()

#HW prediction on train set
holtw_predt=holtw.forecast(steps=len(y_train))
holtw_dft=pd.DataFrame(holtw_predt,columns=['Appliances']).set_index(y_train.index)

#HW prediction on test set
holtw_pred=holtw.forecast(steps=len(y_test))
holtw_df=pd.DataFrame(holtw_pred,columns=['Appliances']).set_index(y_test.index)

#plot of HW model
plt.figure(figsize=(8,8))
plt.plot(y_train.index,y_train, label='Train')
plt.plot(y_test.index,y_test, label='Test')
plt.plot(holtw_df.index,holtw_df.Appliances,label='Holts winter prediction')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Appliances")
plt.title("Holts winter model")
plt.show()

#Model performance on train and test data

#MSE
HW_train_mse=mean_squared_error(y_train,holtw_dft.Appliances)
print('MSE of Holts Winter method on train data:',HW_train_mse)
HW_test_mse=mean_squared_error(y_test,holtw_df.Appliances)
print('MSE of Holts Winter method on test data:',HW_test_mse)

#residual error
HW_reserror=y_train-holtw_dft.Appliances

#Forecast error
HW_foerror=y_test-holtw_df.Appliances

#ACF
acf_hw_res=ACF(HW_reserror.values,70)
x=np.arange(0,70)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_hw_res,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_hw_res,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals (HW)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_hw_fore=ACF(HW_foerror.values,70)
x=np.arange(0,70)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_hw_fore,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_hw_fore,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast error (HW)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Q-value
hotl_q_t=sm.stats.acorr_ljungbox(HW_reserror, lags=5,return_df=True)
print('Q-value (residual):',hotl_q_t)
lbvalue=sm.stats.acorr_ljungbox(HW_foerror,lags=5,return_df=True)
print('Q-value (Forecast):\n',lbvalue)

#Error mean and variance
print('Holts winter: Mean of residual error is',np.mean(HW_reserror),'and Forecast error is',np.mean(HW_foerror))
print('Holts winter: Variance of residual error is',np.var(HW_reserror),'and Forecast error is',np.var(HW_foerror))

#RMSE
HW_train_rmse=mean_squared_error(y_train,holtw_dft.Appliances,squared=False)
print('RMSE of Holts Winter method on train data:',HW_train_rmse)
HW_test_rmse=mean_squared_error(y_test,holtw_df.Appliances,squared=False)
print('RMSE of Holts Winter method on test data:',HW_test_rmse)

#10: Feature selection and collinearity
X_mat=x_train.values
Y=y_train.values
X_svd=sm.add_constant(X_mat)
H=np.matmul(X_svd.T,X_svd)
s,d,v=np.linalg.svd(H)
print('Singular Values: ',d)

#Condition number
print("The condition number is ",la.cond(X_svd))
#the condion number is high. Collinearity exists, so we need to remove the features using backward regression

#Feature selection
x_train_ols=sm.add_constant(x_train)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove rv1 and rv2 - high p-value
x_train_ols.drop(['rv1','rv2'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove const - high p-value
x_train_ols.drop(['const'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove T7 - high p-value
x_train_ols.drop(['T7'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove RH_7 - high p-value
x_train_ols.drop(['RH_7'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())


#Remove RH_5 - high p-value
x_train_ols.drop(['RH_5'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove T5 - high p-value
x_train_ols.drop(['T5'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove RH_out  - high p-value
x_train_ols.drop(['RH_out'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove Tdewpoint  - high p-value
x_train_ols.drop(['Tdewpoint'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove RH_9  - high p-value
x_train_ols.drop(['RH_9'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove T1-high std. error
x_train_ols.drop(['T1'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#Remove Press_mm_hg- high p-value
x_train_ols.drop(['Press_mm_hg'],axis=1,inplace=True)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#The collinearity is removed and the best model is defined above

#12. Multiple Linear Regression
print('The best features are',x_train_ols.columns)

#Prediction on train data
pred_train=model.predict(x_train[['lights', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T6',
       'RH_6', 'T8', 'RH_8', 'T9', 'T_out', 'Windspeed', 'Visibility']])

#Residual error
ml_res=y_train-pred_train

#Prediction on test data
pred_test=model.predict(x_test[['lights', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T6',
       'RH_6', 'T8', 'RH_8', 'T9', 'T_out', 'Windspeed', 'Visibility']])

#Forecast error
ml_fore=y_test-pred_test

#Plot of train data
plt.figure(figsize=(6,6))
plt.plot(y_train.index,y_train, label='Train')
plt.plot(pred_train.index,pred_train, label='Predicted')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Appliances")
plt.title("Predictions on Train set")
plt.show()

#Plot of test data
plt.figure(figsize=(6,6))
plt.plot(y_test.index,y_test, label='Test')
plt.plot(pred_test.index,pred_test, label='Forecasted')
plt.legend()
plt.xlabel("Time")
plt.xticks(rotation=70)
plt.ylabel("Appliances")
plt.title("Predictions on test set")
plt.show()

#Plot of train and test data
plt.figure(figsize=(8,8))
plt.plot(y_train.index,y_train, label='Train')
plt.plot(y_test.index,y_test, label='Test')
plt.plot(pred_test.index,pred_test, label='Forecasted')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Appliances")
plt.title("Train vs. Test vs. Forecasted Data")
plt.show()

#ACF
acf_ml_train=ACF(ml_res,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_ml_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_ml_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_ml_test=ACF(ml_fore.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_ml_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_ml_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (ML)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
ML_train_mse=mean_squared_error(y_train,pred_train)
print('MSE of MLR on train data:',ML_train_mse)
ML_test_mse=mean_squared_error(y_test,pred_test)
print('MSE of MLR on test data:',ML_test_mse)

#Q-value
q_ml_train=sm.stats.acorr_ljungbox(ml_res, lags=5, return_df=True)
print('Q-value (residual):',q_ml_train)
q_mltest=sm.stats.acorr_ljungbox(ml_fore,lags=5,return_df=True)
print('Q-value (Forecast):\n',q_mltest)

#Error mean and variance
print('MLR: Mean of residual error is',np.mean(ml_res),'and Forecast error is',np.mean(ml_fore))
print('MLR: Variance of residual error is',np.var(ml_res),'and Forecast error is',np.var(ml_fore))

#RMSE
ml_train_rmse=mean_squared_error(y_train,pred_train,squared=False)
print('RMSE of MLR method on train data:',ml_train_rmse)
ml_test_rmse=mean_squared_error(y_test,pred_test,squared=False)
print('RMSE of MLR method on test data:',ml_test_rmse)

#Base models
#Average method
train_pred_avg=avg_one(y_train)
test_pred_avg=avg_hstep(y_train,y_test)

#Plot of average method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_pred_avg,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('Average method predictions')
plt.legend()
plt.show()

#Plot of test vs predicted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_pred_avg,label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('Average method Forecast')
plt.legend()
plt.show()

#residual and forecast error
avg_res=y_train-train_pred_avg
avg_fore=y_test-test_pred_avg

#ACF
acf_avg_train=ACF(avg_res.values[1:],100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_avg_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_avg_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_avg_test=ACF(avg_fore.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_avg_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_avg_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (Average)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
avg_train_mse=mean_squared_error(y_train[1:],train_pred_avg[1:])
print('MSE of Average on train data:',avg_train_mse)
avg_test_mse=mean_squared_error(y_test,test_pred_avg)
print('MSE of Average on test data:',avg_test_mse)

#Q-value
q_avg_train=acorr_ljungbox(avg_res.values[1:], lags=5, boxpierce=True,return_df=True)
print('Q-value (residual):',q_avg_train)
q_avgtest=sm.stats.acorr_ljungbox(avg_fore.values[1:],lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_avgtest)

#Error mean and variance
print('Average: Mean of residual error is',np.mean(avg_res),'and Forecast error is',np.mean(avg_fore))
print('Average: Variance of residual error is',np.var(avg_res),'and Forecast error is',np.var(avg_fore))

#RMSE
avg_train_rmse=mean_squared_error(y_train[1:],pred_train[1:],squared=False)
print('RMSE of Average method on train data:',avg_train_rmse)
avg_test_rmse=mean_squared_error(y_test,pred_test,squared=False)
print('RMSE of Average method on test data:',avg_test_rmse)


#Naive method
train_naive=[]
for i in range(len(y_train[1:])):
    train_naive.append(y_train.values[i-1])

test_naive=[y_train.values[-1] for i in y_test]
naive_fore= pd.DataFrame(test_naive).set_index(y_test.index)

#Plot of naive method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_naive,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('Naive method predictions')
plt.legend()
plt.show()

#Plot of test vs predicted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_naive,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('Naive method predictions')
plt.legend()
plt.show()

#residual and forecast error
naive_res=y_train[1:]-train_naive
naive_fore=y_test-test_naive

#ACF
acf_naive_train=ACF(naive_res.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_naive_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_naive_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_naive_test=ACF(naive_fore.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_naive_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_naive_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (Naive)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
naive_train_mse=mean_squared_error(y_train[1:],train_naive)
print('MSE of Naive on train data:',naive_train_mse)
naive_test_mse=mean_squared_error(y_test,test_naive)
print('MSE of Naive on test data:',naive_test_mse)

#Q-value
q_naive_train=acorr_ljungbox(naive_res, lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_naive_train)
q_naivetest=acorr_ljungbox(naive_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_naivetest)

#Error mean and variance
print('Naive: Mean of residual error is',np.mean(naive_res),'and Forecast error is',np.mean(naive_fore))
print('Naive: Variance of residual error is',np.var(naive_res),'and Forecast error is',np.var(naive_fore))

#RMSE
naive_train_rmse=mean_squared_error(y_train[1:],train_naive,squared=False)
print('RMSE of Naive method on train data:',naive_train_rmse)
naive_test_rmse=mean_squared_error(y_test,test_naive,squared=False)
print('RMSE of Naive method on test data:',naive_test_rmse)

#Drift method
train_drift = []
value = 0
for i in range(len(y_train)):
    if i > 1:
        slope_val = (y_train[i - 1]-y_train[0]) / (i-1)
        y_predict = (slope_val * i) + y_train[0]
        train_drift.append(y_predict)
    else:
        continue

test_drift= []
for h in range(len(y_test)):
    slope_val = (y_train.values[-1] - y_train.values[0] ) /( len(y_train) - 1 )
    y_predict= y_train.values[-1] + ((h +1) * slope_val)
    test_drift.append(y_predict)

#Plot of drift method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_drift,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('Drift method predictions')
plt.legend()
plt.show()

#Plot of test vs predicted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_drift,label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('Drift method forecast')
plt.legend()
plt.show()

#residual and forecast error
drift_res=y_train[2:]-train_drift
drift_fore=y_test-test_drift

#ACF
acf_drift_train=ACF(drift_res.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_drift_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_drift_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_drift_test=ACF(drift_fore.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_drift_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_drift_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (Drift)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
drift_train_mse=mean_squared_error(y_train[2:],train_drift)
print('MSE of Drift on train data:',drift_train_mse)
drift_test_mse=mean_squared_error(y_test,test_drift)
print('MSE of Drift on test data:',drift_test_mse)

#Q-value
q_drift_train=acorr_ljungbox(drift_res, lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_drift_train)
q_drifttest=acorr_ljungbox(drift_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_drifttest)

#Error mean and variance
print('Drift: Mean of residual error is',np.mean(drift_res),'and Forecast error is',np.mean(drift_fore))
print('Drift: Variance of residual error is',np.var(drift_res),'and Forecast error is',np.var(drift_fore))

#RMSE
drift_train_rmse=mean_squared_error(y_train[2:],train_drift,squared=False)
print('RMSE of Drift method on train data:',drift_train_rmse)
drift_test_rmse=mean_squared_error(y_test,test_drift,squared=False)
print('RMSE of Drift method on test data:',drift_test_rmse)

#SES
ses= ets.ExponentialSmoothing(y_train,trend=None,damped_trend=False,seasonal=None).fit(smoothing_level=0.5)
train_ses= ses.forecast(steps=len(y_train))
train_ses=pd.DataFrame(train_ses).set_index(y_train.index)

test_ses= ses.forecast(steps=len(y_test))
test_ses=pd.DataFrame(test_ses).set_index(y_test.index)

#Plot of SES method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_ses[0],label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('SES method predictions')
plt.legend()
plt.show()

#Plot of test vs predicted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_ses[0],label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('SES method forecast')
plt.legend()
plt.show()

#residual and forecast error
ses_res=y_train[2:]-train_ses[0]
ses_fore=y_test-test_ses[0]

#ACF
acf_ses_train=ACF(ses_res.values[2:],100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_ses_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_ses_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_ses_test=ACF(ses_fore.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_ses_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_ses_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (SES)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
ses_train_mse=mean_squared_error(y_train,train_ses)
print('MSE of SES on train data:',ses_train_mse)
ses_test_mse=mean_squared_error(y_test,test_ses)
print('MSE of SES on test data:',ses_test_mse)

#Q-value
q_ses_train=acorr_ljungbox(ses_res[2:], lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_ses_train)
q_sestest=acorr_ljungbox(ses_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_sestest)

#Error mean and variance
print('SES: Mean of residual error is',np.mean(ses_res),'and Forecast error is',np.mean(ses_fore))
print('SES: Variance of residual error is',np.var(ses_res),'and Forecast error is',np.var(ses_fore))

#RMSE
ses_train_rmse=mean_squared_error(y_train,train_ses,squared=False)
print('RMSE of SES method on train data:',ses_train_rmse)
ses_test_rmse=mean_squared_error(y_test,test_ses,squared=False)
print('RMSE of SES method on test data:',ses_test_rmse)

#13: ARMA models

#Order determination
diff_train,diff_test=train_test_split(diff_df,test_size=0.2,shuffle=False)

#ACF
ACF_PACF_Plot(diff_df,50)
acf_gpac=ACF(diff_train.Appliances.values,50)
x=np.arange(0,50)
m=1.96/np.sqrt(len(diff_df))
plt.stem(x,acf_gpac,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_gpac,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot for GPAC')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#GPAC
GPAC(acf_gpac,10,10)

#order ARMA(3,0) and ARMA(3,1)

#14: LMA algorithm
na=3
nb=0
#ARMA model
arma_30=sm.tsa.ARMA(y_train,(na,nb)).fit(trend='nc',disp=0)

#Estimated parameters
for i in range(na):
    print(f"The AR coefficient a{i} is:",arma_30.params[i])
for i in range(nb):
    print(f"The MA coefficient a{i} is:",arma_30.params[i + na])
print(arma_30.summary())

#confidence interval
print('Confidence interval:\n',arma_30.conf_int())
#Estimators are significant does not include zero in them

#Prediction on train set
arma30_train=arma_30.predict(start=0,end=15787)
arma30_res=y_train-arma30_train

#Forecast
arma30_test=arma_30.predict(start=15788, end=19734)
arma30_fore=y_test-arma30_test


#ACF
acf_arma30_train=ACF(arma30_res,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_arma30_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_arma30_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_arma30_test=ACF(arma30_fore.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_arma30_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_arma30_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (ARMA(3,0))')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
arma30_train_mse=mean_squared_error(y_train,arma30_train)
print('MSE of ARMA(3,0) on train data:',arma30_train_mse)
arma30_test_mse=mean_squared_error(y_test,arma30_test)
print('MSE of ARMA(3,0) on test data:',arma30_test_mse)

#Q-value
q_arma30_train=acorr_ljungbox(arma30_res, lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_arma30_train)
q_ar30test=acorr_ljungbox(arma30_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_ar30test)

#Error mean and variance
print('ARMA(3,0): Mean of residual error is',np.mean(arma30_res),'and Forecast error is',np.mean(arma30_fore))
print('ARMA(3,0): Variance of residual error is',np.var(arma30_res),'and Forecast error is',np.var(arma30_fore))

#Covariance matrix
print('Covariance matrix\n',arma_30.cov_params())

#standard error
print('Standard error:',arma_30.bse)

#RMSE
ar30_train_rmse=mean_squared_error(y_train,arma30_train,squared=False)
print('RMSE of ARMA(3,0) method on train data:',ar30_train_rmse)
ar30_test_rmse=mean_squared_error(y_test,arma30_test,squared=False)
print('RMSE of ARMA(3,0) method on test data:',ar30_test_rmse)


#Plot of ARMA(3,0) method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,arma30_test,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('ARMA(3,0) method predictions')
plt.legend()
plt.show()

#Plot of test vs forecasted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,arma30_test,label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('ARMA(3,0) method forecast')
plt.legend()
plt.show()

#Plot of train vs predicted
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_train.index,arma30_train,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('ARMA(3,0) method predictions')
plt.legend()
plt.show()

#The model didn't work well on test data let's try other model.
na = 3
nb = 1
# ARMA model
arma31= sm.tsa.ARMA(y_train,(na, nb)).fit(trend='nc',disp=0)
print(arma31.summary())

#Estimated parameters
for i in range(na):
    print(f"The AR coefficient a{i} is:",arma31.params[i])
for i in range(nb):
    print(f"The MA coefficient a{i} is:",arma31.params[i + na])


#confidence interval
print('Confidence interval:\n',arma31.conf_int())
#Estimators are significant does not include zero in them

#Prediction on train set
arma31_train=arma31.predict(start=0,end=15787)
arma31_res=y_train-arma31_train

#Forecast
arma31_test=arma31.predict(start=15788, end=19734)
arma31_fore=y_test-arma31_test


#ACF
acf_arma31_train=ACF(arma31_res,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_arma31_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_arma31_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_arma31_test=ACF(arma31_fore.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_arma31_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_arma31_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (ARMA(3,1))')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
arma31_train_mse=mean_squared_error(y_train,arma31_train)
print('MSE of ARMA(3,1) on train data:',arma31_train_mse)
arma31_test_mse=mean_squared_error(y_test,arma31_test)
print('MSE of ARMA(3,1) on test data:',arma31_test_mse)

#Q-value
q_arma31_train=acorr_ljungbox(arma31_res, lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_arma31_train)
q_ar31test=acorr_ljungbox(arma30_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_ar31test)

#Error mean and variance
print('ARMA(3,1): Mean of residual error is',np.mean(arma31_res),'and Forecast error is',np.mean(arma31_fore))
print('ARMA(3,1): Variance of residual error is',np.var(arma31_res),'and Forecast error is',np.var(arma31_fore))

#Covariance matrix
print('Covariance matrix\n',arma31.cov_params())

#standard error
print('Standard error:',arma31.bse)

#RMSE
ar31_train_rmse=mean_squared_error(y_train,arma31_train,squared=False)
print('RMSE of ARMA(3,1) method on train data:',ar31_train_rmse)
ar31_test_rmse=mean_squared_error(y_test,arma31_test,squared=False)
print('RMSE of ARMA(3,1) method on test data:',ar31_test_rmse)


#Plot of ARMA(3,1) method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,arma31_test,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('ARMA(3,1) method predictions')
plt.legend()
plt.show()

#Plot of test vs forecasted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,arma31_test,label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('ARMA(3,1) method forecast')
plt.legend()
plt.show()

#Plot of train vs predicted
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_train.index,arma31_train,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('ARMA(3,1) method predictions')
plt.legend()
plt.show()


#Let's try SARIMA

#SARIMA= (3,0,0)X(0,1,0,12)
#Seasonality of 144 is not computational there exists a memory error so reduce it to 12
sarima= sm.tsa.statespace.SARIMAX(y_train,order=(3,0,0),seasonal_order=(0,1,0,12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
results=sarima.fit()
print(results.summary())

#predictions on train data
sarima_train=results.get_prediction(start=0, end=len(y_train), dynamic=False)
Sarima_pred=sarima_train.predicted_mean
Sarima_res= y_train-Sarima_pred.values[1:]

#forecast
sarima_test=results.predict(start=0, end=(len(y_test)))
sarima_fore=y_test-sarima_test.values[1:]

#ACF

#ACF
acf_sarima_train=ACF(Sarima_res.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_sarima_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_sarima_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_sarima_test=ACF(sarima_fore.values,100)
x=np.arange(0,100)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_sarima_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_sarima_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (SARMA(3,0,0)X(0,1,0,12))')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
sarima_train_mse=mean_squared_error(y_train,Sarima_pred[1:])
print('MSE of SARIMA on train data:',sarima_train_mse)
sarima_test_mse=mean_squared_error(y_test,sarima_test[1:])
print('MSE of SARIMA on test data:',sarima_test_mse)

#Q-value
q_sarima_train=acorr_ljungbox(Sarima_res, lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_sarima_train)
q_sarimatest=acorr_ljungbox(sarima_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_sarimatest)


#Error mean and variance
print('SARIMA: Mean of residual error is',np.mean(Sarima_res),'and Forecast error is',np.mean(sarima_fore))
print('SARIMA: Variance of residual error is',np.var(Sarima_res),'and Forecast error is',np.var(sarima_fore))

#Covariance matrix
print('Covariance matrix\n',results.cov_params())

#standard error
print('Standard error:',results.bse)

#RMSE
sarima_train_rmse=mean_squared_error(y_train,Sarima_pred[1:],squared=False)
print('RMSE of SARIMA method on train data:',sarima_train_rmse)
sarima_test_rmse=mean_squared_error(y_test,sarima_test[1:],squared=False)
print('RMSE of SARIMA method on test data:',sarima_test_rmse)


#Plot of SARIMA method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,sarima_test[1:],label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('SARIMA method predictions')
plt.legend()
plt.show()

#Plot of test vs forecasted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,sarima_test[1:],label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('SARIMA method forecast')
plt.legend()
plt.show()

#Plot of train vs predicted
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_train.index,Sarima_pred[1:],label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Appliances')
plt.title('SARIMA method predictions')
plt.legend()
plt.show()


#16:The best model with less MSE is Multiple Linear Regression

#17: Forecast function can be written as
print(model.summary())
#y_hat_t= 2.1396*lights + 13.8562*RH_1 + (-17.11)*T2 + (-13.2627)*RH_2 + 25.5010*T3
#         +8.6120*RH_3 + (-3.9367)*T4 + (-2.8629)*RH_4 + (8.3984)*T6 + (0.3553)*RH_6
#         + 8.5673*T8 + (-6.3745)*RH_8 + (-13.4323)*T9 + (-6.1772)*T_out
#         + 1.0866 *Windspeed + 0.2045*Visibility

#18: H-step ahead prediction

#predictions of test data for MLR was performed previously (see at MLR part of code)
#Let's plot the test vs forecasted values of MLR
plt.figure(figsize=(6,6))
plt.plot(y_test.index,y_test, label='Test')
plt.plot(pred_test.index,pred_test, label='Forecasted')
plt.legend()
plt.xlabel("Time")
plt.xticks(rotation=70)
plt.ylabel("Appliances")
plt.title("Predictions on test set")
plt.show()