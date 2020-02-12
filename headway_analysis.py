import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
from util import line_station2chinese, line2chinese, line_bus_count

# analysis of headway information for line and station
mean_headway = pd.read_csv('mean_headway.csv', index_col=[0])
std_headway = pd.read_csv('std_headway.csv', index_col=[0])
count_o = pd.read_csv('count_o.csv', index_col=[0])
count_station_o = pd.read_csv('count_station2.csv')
count_station_o['line_id'] = count_station_o['line_id'] * 10 + 1 * (count_station_o['direction'] > 0)
# count_station_pivoted = count_station_o.pivot_table(index='line_id', columns='start_station',values='count_record',aggfunc=max)
# discard useless data
mean_headway_stk = mean_headway.stack()
mean_headway_stk = mean_headway_stk.reset_index()
std_headway_stk = std_headway.stack()
std_headway_stk = std_headway_stk.reset_index()

std_headway_stk.rename(columns={'level_1': 'start_station', 0: 'std_headway'}, inplace=True)
mean_headway_stk.rename(columns={'level_1': 'start_station', 0: 'mean_headway'}, inplace=True)
std_headway_stk['start_station'] = std_headway_stk['start_station'].astype('int64')
mean_headway_stk['start_station'] = mean_headway_stk['start_station'].astype('int64')
headway_new = pd.merge(mean_headway_stk, count_station_o, on=['line_id', 'start_station'])
headway_new = pd.merge(headway_new, std_headway_stk, on=['line_id', 'start_station'])



# discard records with insufficient data records
headway_new = headway_new.loc[(headway_new['count_record'] >= 1000) & (headway_new['mean_headway'] > 0)]
# station_num to number of station from start (according to direction)
headway_min_max_station = headway_new.groupby('line_id').agg({'start_station': ['min', 'max']})
headway_min_max_station.columns = ['_'.join(col).strip() for col in headway_min_max_station.columns]
headway_min_max_station.reset_index(inplace=True)
headway_new = pd.merge(headway_new, headway_min_max_station, how='left', on='line_id')
headway_new['num_station'] = (headway_new['start_station'] - headway_new['start_station_min']) * (
            headway_new['direction'] > 0) + (headway_new['start_station_max'] - headway_new['start_station']) * (
                                         headway_new['direction'] < 0)
headway_new['total_stations'] = headway_new['start_station_max']-headway_new['start_station_min']+1

# part 1. rank headway for each line
# mean headway for each line

# regularized headway pattern
headway_new['pax_headway'] = headway_new['count_record'] * headway_new['mean_headway']
headway_line = headway_new.groupby('line_id').agg({'pax_headway': 'sum', 'count_record': 'sum', 'linename': 'first'})
headway_line['avg_headway'] = headway_line['pax_headway'] / headway_line['count_record']

#line_bus = line_bus_count()
line_bus = pd.read_csv('line_bus_count.csv')
line_bus = line_bus.loc[line_bus['line_id']>0]
line_bus = line_bus.loc[line_bus['direction']!=0]
#line_bus['line_id'] = line_bus['line_id']*10 + 1*(line_bus['direction']>0)
line_bus['line_id'] = line_bus['line_id'].astype(int)
line_bus = line_bus.loc[line_bus['count_record']>10]

line_bus_mean = line_bus.groupby('line_id').agg({'count_record':'mean'}).reset_index()
line_bus = pd.merge(line_bus, line_bus_mean,how='left', on='line_id')
line_bus = line_bus.loc[line_bus['count_record_x']>line_bus['count_record_y']/2.]
line_bus = line_bus.groupby('line_id').agg({'bus_id':'count'})
headway_line = headway_line.join(line_bus)
headway_line['index0'] = headway_line['pax_headway']/headway_line['bus_id']
headway_line['reg_index'] = headway_line['index0']/headway_line['index0'].mean()
# find optimal solution
total_bus = headway_line['bus_id'].sum()
headway_line['trip_time']=headway_line['avg_headway']*headway_line['bus_id']
headway_line['prop_bus'] = (headway_line['count_record']*headway_line['trip_time'])**0.5
headway_line['m*']=headway_line['prop_bus']/headway_line['prop_bus'].sum()*total_bus
headway_line['m**']=headway_line['m*'].apply(np.floor)
available_bus = total_bus - headway_line['m**'].sum()
headway_line['score'] = headway_line['count_record']*headway_line['trip_time']*(1./headway_line['m**']-1./(headway_line['m**']+1))
headway_line.sort_values('score',ascending=False,inplace=True)
headway_line['is_up']=0
headway_line.iloc[:int(available_bus),-1]=1
headway_line['m***']=headway_line['m**']+headway_line['is_up']
headway_line['suggestion']=headway_line['m***']-headway_line['bus_id']
total_time_before = np.sum(headway_line['count_record']*headway_line['trip_time']/headway_line['bus_id'])/3600
total_time_opt = np.sum(headway_line['count_record']*headway_line['trip_time']/headway_line['m*'])/3600
total_time_opt_real = np.sum(headway_line['count_record']*headway_line['trip_time']/headway_line['m***'])/3600
headway_line.hist('suggestion',bins=100)
# part 2. headway pattern

headway_new.sort_values(['line_id','num_station'],inplace=True)
first_headway = headway_new.groupby('line_id').agg({'mean_headway':'first'})
first_headway.rename(columns={'mean_headway':'first_headway'},inplace=True)
first_headway.reset_index(inplace=True)
headway_new = pd.merge(headway_new, first_headway, on='line_id',how='left')

headway_new['reg_headway'] = headway_new['mean_headway']/headway_new['first_headway']
headway_new['reg_std'] = headway_new['std_headway']/headway_new['first_headway']
headway_new['reg_station'] = headway_new['num_station']/headway_new['total_stations']
l_lines = headway_new['line_id'].unique().tolist()

headway_new['h975'] =(headway_new['mean_headway']**2. + headway_new['std_headway']**2.)/2/headway_new['mean_headway']/headway_new['first_headway']

headway_new_rb = headway_new.groupby('line_id').agg({'linename':'first','h975':'max','reg_headway':'max','mean_headway':'mean','std_headway':'mean'})

plt.figure()
for line in l_lines:
    df_temp = headway_new.loc[headway_new['line_id']==line]
    if df_temp.shape[0]<3:
        continue
    if df_temp['reg_std'].max()<3.5:
        continue
    plt.plot(df_temp['reg_station'], df_temp['h975'],label='line_num={0},{1}'.format(line//10,'上行'*(line%10)+'下行'*(1-line%10) ))
    plt.legend()
plt.xlabel('regularized station number')
plt.ylabel('station-wise index of robustness')



headway_line_new = headway_line.join(headway_new_rb, lsuffix='_1',rsuffix='_2')
headway_line_new.to_csv('line_final.csv')



#headway_line_new.to_csv('line_final.csv')
headway_new['waiting_time'] = headway_new['h975']*headway_new['first_headway']
headway_new['pax_waiting_time'] = headway_new['waiting_time']*headway_new['count_record']
headway_station = headway_new.groupby('name').agg({'pax_waiting_time':'sum','count_record':'sum'})
headway_station['avg_waiting'] = headway_station['pax_waiting_time']/headway_station['count_record']
headway_station['Q'] = headway_station['count_record']/(30.*3600*24)*headway_station['avg_waiting']
headway_station.to_csv('temp_station.csv')


# part 3. station headway: station with long waiting time may result in overcrowded platform
# mean headway for each station
"""
headway_new['pax_headway'] = headway_new['count_record']*headway_new['mean_headway']
headway_station = headway_new.groupby(['name']).agg({'pax_headway':'sum','count_record':'sum'})
headway_station['avg_headway'] = headway_station['pax_headway']/headway_station['count_record']
headway_station['Q'] = headway_station['count_record']/(30.*3600*24)*headway_station['avg_headway']/2.
"""

# mean station headway vs demand
