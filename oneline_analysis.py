import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
from util import line_station2chinese, time2int, int2timedate, get_one_line, get_all_lines
mpl.rcParams['font.sans-serif'] = ['SimHei']

all_lines = get_all_lines()

def line_headway(line57_record, max_station, direction=1, line_id=57):
    mean_headway = np.zeros(100)
    std_headway = np.zeros(100)
    count_o = np.zeros(100)
    line57_record = line57_record.loc[line57_record['start_station'] <= max_station]
    line57_record = line57_record.loc[line57_record['start_station'] >= 1]
    for station in np.sort(line57_record['start_station'].unique()):
        print('line_id',line_id,' direction', direction,' station:', station)
        onestation = line57_record.loc[line57_record['start_station'] == station]
        onestation = onestation.loc[onestation['start_time'] > 0]
        if onestation.shape[0] < 10:
            continue
        if direction > 0:
            onestation = onestation.loc[onestation['start_station'] < onestation['end_station']]
        else:
            onestation = onestation.loc[onestation['start_station'] > onestation['end_station']]
        onestation.sort_values('start_time', inplace=True)
        onestation.reset_index(drop=1, inplace=True)
        diff_bus = onestation['bus_unique'] != onestation['bus_unique'].shift(1)
        diff_bus[0] = True
        onestation['diff_bus'] = diff_bus
        onestation['diff_bus'] = onestation['diff_bus'].cumsum()
        station_record = onestation.groupby('diff_bus').agg({'start_time': lambda x: x.iloc[0], 'bus_id': 'count'})
        station_record.rename(columns={'bus_id': 'count'}, inplace=True)
        station_record['diff_time'] = station_record['start_time'] - station_record['start_time'].shift()
        station_record = station_record.loc[
            station_record['diff_time'] < station_record['diff_time'].mean() + 3 * station_record['diff_time'].std()]
        station_record = station_record.loc[
            station_record['diff_time'] < 3600]
        mean_headway[station] = station_record['diff_time'].mean()
        std_headway[station] = station_record['diff_time'].std()
        count_o[station] = station_record['count'].mean()

    mean_headway = np.nan_to_num(mean_headway)
    std_headway = np.nan_to_num(std_headway)
    count_o = np.nan_to_num(count_o)
    return mean_headway, std_headway, count_o

def append2table(line_id, big_table, array1, array2):
    array = np.stack([array1,array2])
    small_table = pd.DataFrame(array, columns=np.arange(1, 101))
    newlineid = [line_id * 10 + 1, line_id * 10 + 0]
    small_table['line_id']= newlineid
    small_table.set_index('line_id',inplace=True)
    big_table = big_table.append(small_table)
    return big_table

mean_table = pd.DataFrame(columns=np.arange(1,101))
mean_table.index.rename('line_id',inplace=True)
std_table = pd.DataFrame(columns=np.arange(1,101))
std_table.index.rename('line_id',inplace=True)
count_o_table = pd.DataFrame(columns=np.arange(1,101))
count_o_table.index.rename('line_id',inplace=True)

cnt = 0
for line_id in all_lines['line_id'].values:
    print(cnt/len(all_lines),'-'*50)
    line57_record, le, max_station = get_one_line(line_id)
    direction = 1
    mean_headway1, std_headway1, count_o1 = line_headway(line57_record, max_station, direction, line_id)
    direction = -1
    mean_headway2, std_headway2, count_o2 = line_headway(line57_record, max_station, direction, line_id)
    mean_table = append2table(line_id, mean_table, mean_headway1, mean_headway2)
    std_table = append2table(line_id, std_table, std_headway1, std_headway2)
    count_o_table = append2table(line_id, count_o_table, count_o1, count_o2)
    cnt+=1
mean_table.to_csv('mean_headway.csv')
std_table.to_csv('std_headway.csv')
count_o_table.to_csv('count_o.csv')
