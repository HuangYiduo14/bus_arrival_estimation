import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_record(linkid_interested=36632, train=True):
    if train:
        speed_record0 = pd.read_excel('data_car/data/link05m_lv5_20180601_西三环南向北.xlsx')
        speed_record0 = speed_record0.loc[speed_record0['linkid'] == linkid_interested]
        print('file 0 loaded')
        speed_record1 = pd.read_excel('data_car/data/link05m_lv5_20180602_20180607_西三环南向北.xlsx')
        speed_record1 = speed_record1.loc[speed_record1['linkid'] == linkid_interested]
        print('file 1 loaded')
        speed_record2 = pd.read_excel('data_car/data/link05m_lv5_20180608_20180614_西三环南向北.xlsx')
        speed_record2 = speed_record2.loc[speed_record2['linkid'] == linkid_interested]
        print('file 2 loaded')
        speed_record = pd.concat([speed_record0, speed_record1, speed_record2])
    else:
        speed_record3 = pd.read_csv('data_car/data/link05m_lv5_20180615_20180630_西三环南向北.csv')
        speed_record3 = speed_record3.loc[speed_record3['linkid'] == linkid_interested]
        print('file 3 loaded')
        speed_record = speed_record3

    speed_record['time'] = pd.to_datetime(speed_record['dt'].astype(str) + ' ' + speed_record['tm'].astype(str))
    speed_record = speed_record.drop(['dt', 'tm'], axis=1)
    # speed_record = speed_record.loc[speed_record['time'].dt.time>pd.to_datetime('05:00:00').time()]
    # speed_record = speed_record.loc[speed_record['time'].dt.time<pd.to_datetime('23:59:59').time()]
    speed_record['tm'] = speed_record['time'].dt.time.astype(str)
    speed_record['tm'] = speed_record['tm'].str[0].astype(int) * 10 * 3600 + speed_record['tm'].str[1].astype(
        int) * 3600 + speed_record['tm'].str[3].astype(int) * 10 * 60 + speed_record['tm'].str[4].astype(int) * 60
    speed_record['time1'] = speed_record['tm'] + speed_record['time'].dt.day * 24 * 3600
    speed_record.sort_values('time1', inplace=True)
    speed_record.reset_index(inplace=True, drop=True)

    mean_speed = speed_record['speed'].mean()
    speed_record['speed'] = speed_record['speed'] - mean_speed

    plt.figure()
    fxx, freqx = plt.psd(speed_record['speed'], Fs=60 / 5, pad_to=speed_record.shape[0], NFFT=512)
    plt.plot([1 / 24, 1 / 24], [0, 40], color='red', label='1/(24 hours)', alpha=0.5)
    plt.plot([1 / 12, 1 / 12], [0, 40], color='yellow', label='1/(12 hours)', alpha=0.5)
    plt.plot([1 / 6, 1 / 6], [0, 40], color='blue', label='1/(6 hours)', alpha=0.5)
    plt.plot([1 / 3, 1 / 3], [0, 40], color='green', label='1/(3 hours)', alpha=0.5)
    plt.legend()
    return speed_record,fxx,freqx,mean_speed



speed_recordx,fxx,freqx,mean_x = get_record(linkid_interested=36632)
speed_recordy,fyy,freqy,mean_y = get_record(linkid_interested=35965)
plt.figure()
csdyx, fcsd = plt.csd(speed_recordx['speed'],speed_recordy['speed'])
plt.figure()
csdxy, fcsd2 = plt.csd(speed_recordy['speed'],speed_recordx['speed'])

x=speed_recordx['speed'].values
y=speed_recordy['speed'].values
dx = np.fft.fft(x)
dy = np.fft.fft(y)

# low-pass filter
pass_index = dx.shape[0]//6
dx[pass_index+1:-pass_index-1]=0
xx = np.fft.ifft(dx)
yy = np.fft.ifft(dy)
dx2 = dx.copy()
pass_index2 = dx.shape[0]//12
dx2[pass_index2+1:-pass_index2-1]=0
xx2 = np.fft.ifft(dx2)
plt.figure()
plt.plot(5*np.arange(x.shape[0]),x,label='original data record')
plt.plot(5*np.arange(x.shape[0]),xx,label='low pass filter, threshold=1/2h',alpha=0.5)
plt.plot(5*np.arange(x.shape[0]),xx2,label='low pass filter, threshold=1/1h',alpha=0.5)
plt.xlabel('time (min)')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(dx2.shape[0])/dx2.shape[0] * 12,np.abs(dx2)/dx2.shape[0])
plt.xlabel('freq (Hz)')
plt.ylabel('abs(fx)')

z= x - xx.real

plt.plot(5*np.arange(x.shape[0]),z)
from statsmodels.graphics.tsaplots import  plot_acf,plot_pacf
plot_acf(z)
plot_pacf(z)

from statsmodels.tsa.arima_model import ARIMA

series = z
# fit model
model = ARIMA(series, order=(2,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
plt.figure()
residuals.plot()
plt.show()
plt.figure()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

speed_recordx_test,fxx_test,freqx_test,mean_x_test = get_record(linkid_interested=36632, train=False)
plt.figure()
x_test = speed_recordx_test['speed'].values
plt.plot(5*np.arange(x_test.shape[0]),x_test, label='real data')
extend_x = xx2 * np.exp((xx2.shape[0]-1.)*np.pi/2.*np.complex(0.,1.))
plt.plot(5*np.arange(extend_x.shape[0]),extend_x, label='prediction using fft (th), threshold=1/1h')
res_predict = model_fit.forecast(extend_x.shape[0])[0]
plt.plot(5*np.arange(extend_x.shape[0]),extend_x+np.array(res_predict), label='prediction fft+ar(2), threshold=1/1h')
plt.legend()
res_predict1 = np.ones_like(extend_x)
res_predict1[0] = x_test[0]-extend_x[0]
res_predict1[1] = x_test[1]-extend_x[1]
N_test = extend_x.shape[0]
x1 = x_test[1:N_test-1]-extend_x[1:N_test-1]
x2 = x_test[2:N_test] -extend_x[2:N_test]
res_predict1[2:] = model_fit.params[0]+x1* model_fit.params[1]+x2* model_fit.params[2]

plt.plot(5*np.arange(extend_x.shape[0]),extend_x+np.array(res_predict1), label='prediction fft+ar(2) one step, threshold=1/1h')
plt.legend()


plt.figure()
x_test = speed_recordx_test['speed'].values
plt.plot(5*np.arange(x_test.shape[0]),x_test, label='real data')
extend_x = xx * np.exp((xx2.shape[0]-1.)*np.pi/2.*np.complex(0.,1.))
plt.plot(5*np.arange(extend_x.shape[0]),extend_x, label='prediction using fft, threshold=1/2h')
res_predict = model_fit.forecast(extend_x.shape[0])[0]
plt.plot(5*np.arange(extend_x.shape[0]),extend_x+np.array(res_predict), label='prediction fft+ar(2), threshold=1/2h')
plt.legend()
res_predict1 = np.ones_like(extend_x)
res_predict1[0] = x_test[0]-extend_x[0]
res_predict1[1] = x_test[1]-extend_x[1]
N_test = extend_x.shape[0]
x1 = x_test[1:N_test-1]-extend_x[1:N_test-1]
x2 = x_test[2:N_test] -extend_x[2:N_test]
res_predict1[2:] = model_fit.params[0]+x1* model_fit.params[1]+x2* model_fit.params[2]

plt.plot(5*np.arange(extend_x.shape[0]),extend_x+np.array(res_predict1), label='prediction fft+ar(2) one step, threshold=1/2h')
plt.legend()
