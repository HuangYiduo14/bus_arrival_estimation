import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

arrival_time = pd.read_csv('arrival time matrix.csv',header=None)
arrival_time = arrival_time.values

#estimate random function
#for i in range(arrival_time.shape[1] - 1):
for i in range(10,11):
    time_diff = arrival_time[:, i+1] - arrival_time[:, i]
    time_diff = time_diff.reshape(-1)
    time_diff[abs(time_diff) > 1000] = 0
    mask = time_diff <= 0
    mask = ~mask
    valid_record = pd.DataFrame({
        'arrival_time':arrival_time[mask, i],
        'time_diff':time_diff[mask]
    })
    valid_record.sort_values(by='arrival_time',inplace=True)

    valid_record['arrival_time'] = valid_record['arrival_time'] % (3600 * 24)
    #valid_record.plot(x='arrival_time',y='time_diff')
    plt.scatter(valid_record['arrival_time'],valid_record['time_diff'])
    plt.show()
