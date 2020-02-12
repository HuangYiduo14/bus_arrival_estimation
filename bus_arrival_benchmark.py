import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import mpl
from util import line_station2chinese, time2int, int2timedate, get_one_line, aggregate_record_station
mpl.rcParams['font.sans-serif'] = ['SimHei']



def detect_round_info(df):
    # step 2. detect round information
    # if the direction changed, a new round starts, otherwise append record to current round
    print('start round decomposing')
    df_record = df.copy().reset_index(drop=1)
    # Identify direction by comparing consecutive station sequence
    l_new_station_ind = df_record.loc[(df_record['is_new_station']) == 1].index.tolist()
    l_new_station = df_record.loc[l_new_station_ind, 'end_station'].values
    # Greedy method is used to determine whether choose post direction or pre direction
    l_post_direction = l_new_station[1:] - l_new_station[:-1]
    l_pre_direction = np.insert(l_post_direction, 0, np.nan)
    l_post_direction = np.insert(l_post_direction, -1, np.nan)
    #l_new_station_time = df_record.loc[l_new_station_ind, 'trans_time'].values
    #l_post_time_gap = l_new_station_time[1:] - l_new_station_time[:-1]
    #l_pre_time_gap = np.insert(l_post_time_gap, 0, 3600)
    # we need to split record lines (not only station)
    #l_skew_post = l_pre_time_gap > np.insert(l_post_time_gap, -1, 3600)
    #l_pre_direction[l_skew_post] = np.insert(l_post_direction, -1, np.nan)[l_skew_post]
    df_record.loc[l_new_station_ind, 'prev_direction'] = l_pre_direction
    df_record.loc[l_new_station_ind,'succ_direction'] = l_post_direction
    df_record['time_diff'] = df_record['trans_time'] - df_record['trans_time'].shift(1)


    # Detect the round shift station by back-and-forth filling, denoted by 0
    df_record['prev_direction'] = df_record['prev_direction'].fillna(method='ffill').fillna(method='bfill')
    df_record['succ_direction'] = df_record['succ_direction'].fillna(method='ffill').fillna(method='bfill')
    df_record['prev_direction'] = df_record['prev_direction'].clip(lower=-1, upper=1)
    df_record['succ_direction'] = df_record['succ_direction'].clip(lower=-1, upper=1)
    df_record['direction'] = (df_record['prev_direction'] + df_record['succ_direction']) / 2
    df_record['is_new_round'] = 0
    # Extract the exact first record before and after round shift
    l_before_change = df_record[(
                                        df_record['direction'].shift(-1) != 0) & (
                                        df_record['direction'].shift(-2) == 0)].index.tolist()
    l_after_change = df_record[(
                                       df_record['direction'].shift(1) == 0) & (
                                       df_record['direction'] != 0)].index.tolist()
    if len(l_after_change)>len(l_before_change):
        l_after_change = l_after_change[1:]
    elif len(l_after_change)<len(l_before_change):
        l_before_change = l_before_change[:-1]
    # Calculate time difference and the spliting time
    df_change = pd.DataFrame()
    df_change['before_time'] = df_record.loc[l_before_change, 'trans_time'].reset_index(drop=1)
    df_change['after_time'] = df_record.loc[l_after_change, 'trans_time'].reset_index(drop=1)
    df_change['after_index'] = l_after_change
    df_change['median_time'] = df_change[['before_time', 'after_time']].mean(axis=1)
    # Find the exact record which starts a new round
    l_change = df_change['median_time'].apply(lambda x: df_record[df_record['trans_time'] > x].index[0])
    # Label the new record as a new station and fill all the other direction
    df_record.loc[l_change.tolist(), 'direction'] = df_record.loc[
        df_change['after_index'].tolist(), 'direction'].tolist()
    df_record.loc[l_change.tolist(), 'is_new_station'] = 1

    df_record.loc[(df_record['is_new_station']==1)&(df_record['prev_direction']!=df_record['succ_direction']),'is_new_round']=1
    df_record.loc[df_record['time_diff']>3600, 'is_new_round']=1

    df_record.loc[l_change.tolist(), 'is_new_round'] = 1
    df_record.loc[df_record['direction'] == 0, 'direction'] = np.nan
    df_record['direction'] = df_record['direction'].fillna(method='ffill')

    # Calibrate station number which is not consistent with direction
    l_need_cast = (df_record['start_station'] <= df_record['end_station']) * 2 - 1 != df_record['direction']
    df_record.loc[l_need_cast, 'start_station'] = df_record.loc[l_need_cast, 'end_station']
    #df_record.to_csv('temp2.csv')
    return df_record.drop(['prev_direction', 'succ_direction'], axis=1)

def merge_one_round(df_round):
    # step 3. merge possibly round trip
    # because some data point has error station information, some rounds are split into different rounds
    # 3.1. find 'small trip' (usually caused by data error) and merge 'small trip' to closest trip round
    print('start round merging')
    round_info = df_round.groupby('round_id').agg({'direction':'mean','trans_time':['min','max'],'end_station':['min','max', pd.Series.nunique]})
    round_info.columns = ['_'.join(i) for i in round_info.columns]
    small_round_ind = round_info.loc[round_info['end_station_nunique']<=3].index
    #round_info = round_info.loc[round_info['end_station_nunique'] >= 4]
    def common_station(df_round, trip1,trip2):
        stations1 = set(df_round.loc[df_round['round_id']==trip1,'end_station'])
        stations2 = set(df_round.loc[df_round['round_id']==trip2,'end_station'])
        return len(stations1&stations2)
    def merge2(df_round, trip1, trip2):
        # merge trip2(smaller) into trip1
        s1 = df_round.loc[df_round['round_id']==trip1, 'direction']
        direction1 = s1.iloc[0]
        df_round.loc[df_round['round_id'] == trip2,'direction'] = direction1
        df_round.loc[df_round['round_id'] == trip2,'round_id'] = trip1
        return df_round
    ind = round_info.index.min()
    while ind <= round_info.index.max()-2:
        if ind in small_round_ind:
            ind += 1
            continue
        ind2 = ind + 1
        while True:
            ind3 = ind2+1
            if (ind2 in small_round_ind) \
                    and (common_station(df_round,ind,ind3)<=3
                                              and (round_info.loc[ind,'direction_mean']==round_info.loc[ind3,'direction_mean']
                                                   and np.abs(round_info.loc[ind,'trans_time_max']-round_info.loc[ind3,'trans_time_min'])<20*60)) :
                df_round=merge2(df_round, ind, ind2)
                df_round=merge2(df_round, ind, ind3)
            else:
                ind = ind2
                break
            ind2+=2
    return df_round



def passenger_num_count(df, max_station):
    # step 4. figure out number of boarding and alighting for each round
    print('start counting passenger')
    df_record = df.copy().reset_index(drop=1)

    def cal_pax_num(df_record_round, max_station):
        round_board, round_alight = np.zeros(max_station + 1), np.zeros(max_station + 1)
        is_pos_direction = df_record_round['direction'].min() == 1

        # Find boarding number and alighting number for each station
        df_board = df_record_round.groupby('start_station').count()['direction']
        round_board[df_board.index.tolist()] = df_board.values
        df_alight = df_record_round.groupby('end_station').count()['direction']
        round_alight[df_alight.index.tolist()] = df_alight.values

        # Passenger number is calculated for different direction respectively in an inverse order.
        if is_pos_direction:
            round_pax_num = np.flip((np.flip(round_alight) - np.flip(round_board)).cumsum())
        else:
            round_pax_num = (round_alight - round_board).cumsum()
        return pd.DataFrame(
            {'end_station': range(1, max_station + 1), 'pax_num': round_pax_num[1:]}).set_index('end_station')

    df_pax_num = df_record.groupby('round_id').apply(lambda x: cal_pax_num(x, max_station))
    return df_pax_num

def estimate_arrival_time_local(df_round, df_pax_num):
    # setting the time range for outliers and record clustering
    print('start benchmark local arrival time estimation')
    df_record = df_round.copy().reset_index(drop=1)
    delta1 = 72  # threshold for outlier
    delta2 = 5  # threshold for clustering
    seat_number = 58  # this number is according to baike.baidu.com<<<<<<<<<<<<<<<<!!!!!!!

    # Detect outliers by threshold 1
    df_record['is_outlier'] = 0
    df_record.loc[(df_record['end_station'] == df_record['end_station'].shift(1))
                  & (df_record['trans_time'] - df_record['trans_time'].shift(1) > delta1), 'is_outlier'] = 1
    # Detect clusters by threshold 1
    df_record = df_record[df_record['is_outlier'] == 0].reset_index(drop=1).drop('is_outlier', axis=1)
    df_record['is_new_cluster'] = 0
    df_record.loc[(df_record['end_station'] == df_record['end_station'].shift(1))
                  & (df_record['trans_time'] - df_record['trans_time'].shift(1) > delta2), 'is_new_cluster'] = 1

    # Format all required data into the same round*station dataframe
    t1_record = df_record.groupby(['round_id', 'end_station']).max()['trans_time'].to_frame()
    i_val_record_tmp = df_record.groupby(['round_id', 'end_station']).count()['trans_time'].to_frame()
    j_val_record_tmp = df_record.groupby(['round_id', 'end_station']).sum()['is_new_cluster'].to_frame()
    arrival_time_record = pd.DataFrame(index=i_val_record_tmp.index, columns=['arrival_time']).fillna(0)

    # Case 1
    l_case1 = ((i_val_record_tmp > 2) & (j_val_record_tmp >= 3)).index.tolist()
    arrival_time_record.loc[l_case1, 'arrival_time'] = t1_record.loc[l_case1, 'trans_time'] - (
            1.17 * j_val_record_tmp.loc[l_case1, 'is_new_cluster'] - 2.27)
    # Case 2
    l_case2 = ((i_val_record_tmp > 2) & (j_val_record_tmp < 3)).index.tolist()
    Nm = 1. * df_pax_num.copy() / seat_number
    arrival_time_record.loc[l_case2, 'arrival_time'] = t1_record.loc[l_case2, 'trans_time'] - (
            -15.59 * (Nm.loc[l_case2, 'pax_num'] ** 2.) + 63.63 * Nm.loc[l_case2, 'pax_num'] - 68.)

    Nm['i_val'], Nm['j_val'], Nm['arr_time'] = i_val_record_tmp, j_val_record_tmp, arrival_time_record
    return Nm['arr_time'], Nm['i_val'], Nm['j_val']


def analysis_one_bus(line57_onebus_temp, max_station):
    # step 1. aggregate records into stations
    df_station = aggregate_record_station(line57_onebus_temp)
    # step 2. detect round information
    df_round = detect_round_info(df_station)
    # step 3. merge possibly round trip
    df_round = merge_one_round(df_round)
    df_round[['start_station', 'end_station']] = df_round[['start_station', 'end_station']].astype(int)
    # step 4. figure out number of boarding and alighting for each round
    df_passenger_number = passenger_num_count(df_round, max_station)
    # step 5. estimate arrival time (using only current data) for each station using empirical formula
    df_arrival_time, df_i_val, df_j_val = estimate_arrival_time_local(df_round, df_passenger_number)

    # convert the dataframe to array
    def df_to_array(df, column):
        return df.reset_index().pivot(index='round_id', columns='end_station', values=column).values

    round_direction_list = df_round.groupby('round_id').max()['direction'].values
    number_passenger_record = df_to_array(df_passenger_number, 'pax_num')
    arrival_time_record = df_to_array(df_arrival_time, 'arr_time')
    i_val_record = df_to_array(df_i_val, 'i_val')
    j_val_record = df_to_array(df_j_val, 'j_val')

    print('this bus done' + '=' * 50)
    return round_direction_list, number_passenger_record, arrival_time_record, i_val_record, j_val_record


def matrix_with_missing_construction(round_info):
    row_num = len(round_info)
    col_num = len(round_info[0].arrival_time_round)
    M = np.zeros((row_num, col_num))
    for i in range(row_num):
        M[i] = round_info[i].arrival_time_round
    return M

line_id=57
line57_record,le,max_station = get_one_line(line_id)
# testing: one bus
#for i in [0]:
for i in line57_record['bus_unique'].unique():
    print('start analysis of bus {0}, bus id{1}'.format(i, le.classes_[i]), '--' * 50)
    line57_onebus = line57_record.loc[line57_record['bus_unique'] == i]
    line57_onebus_temp = line57_onebus.sort_values('trans_time').drop(['bus_id', 'bus_unique'], axis=1)
    if line57_onebus.shape[0]<10:
        continue
    df_station = aggregate_record_station(line57_onebus_temp)
    df_round = detect_round_info(df_station)
    df_round['round_id'] = df_round['is_new_round'].cumsum()
    df_round = merge_one_round(df_round)
    df_round[['start_station', 'end_station']] = df_round[['start_station', 'end_station']].astype(int)
    df_passenger_number = passenger_num_count(df_round, max_station)
    #df_arrival_time, df_i_val, df_j_val = estimate_arrival_time_local(df_round, df_passenger_number)
    chinese_station_name =[line_station2chinese(line_id, k) for k in np.sort(df_round['end_station'].unique())]
    for j in df_round['round_id'].unique():
        trip_info = df_round.loc[df_round['round_id'] == j]
        trip_info.sort_values(by=['end_station', 'trans_time'])
        trip_info = trip_info.loc[trip_info['trans_time']<48*3600]
        if trip_info.shape[0]<10 or trip_info['direction'].mean()<=0:
           continue
        plt.scatter(trip_info['trans_time'].apply(int2timedate).to_list() + trip_info['start_time'].apply(int2timedate).to_list(),
                    trip_info['end_station'].to_list() + trip_info['start_station'].to_list(), alpha=0.2)
        plt.plot(trip_info['trans_time'].apply(int2timedate).to_list(), trip_info['end_station'])
        plt.yticks(np.sort(df_round['end_station'].unique()).tolist(), chinese_station_name)
        #plt.text(trip_info['trans_time'].apply(int2timedate).iloc[0], trip_info['end_station'].iloc[0]+0.01, trip_info['direction'].mean())
        #plt.text(trip_info['trans_time'].apply(int2timedate).iloc[-1], trip_info['end_station'].iloc[-1]+0.01, le.classes_[i])





"""
class Round:
    def __init__(self, bus_id, direction, i_val_round, j_val_round, arrival_time_round, number_passenger_round):
        self.bus_id = bus_id
        self.direction = direction
        self.i_val_round = i_val_round  # total number of passengers alighting at this station
        self.j_val_round = j_val_round  # total number of aggregated passengers
        self.arrival_time_round = np.nan_to_num(arrival_time_round)  # estimated arrival time
        self.number_passenger_round = number_passenger_round  # total of passenger on board at this station
        self.round_last_time = self.arrival_time_round.max()  # the last valid estimated arrival
        # we will also use
        # 1. weekdays
        # 2. weather? holidays?.....
        # as known features

    def __lt__(self, other):
        if self.direction < other.direction:
            return True
        elif self.round_last_time < other.round_last_time:
            return True
        return False

    def __str__(self):
        s0 = '''
        bus_id = {self.bus_id}
        direction = {self.direction}
        arrival_time_estimated = {self.arrival_time_round}
        '''.format(self=self)
        return s0
        
forward_round_info = []
backward_round_info = []
for i in range(1):
#for i in line57_record['bus_unique'].unique():
    line57_onebus = line57_record.loc[line57_record['bus_unique'] == i]
    # If there are only less than 5 data belongs to 1 vehicle, we consider it invalid.
    if len(line57_onebus) <= 5:
        continue
    bus_id = le.classes_[i]
    line57_onebus_temp = line57_onebus.sort_values('trans_time').drop(['bus_id', 'bus_unique'], axis=1)
    print('start analysis of bus {0}, bus id{1}'.format(i, le.classes_[i]), '--' * 50)
    round_direction_list, number_passenger_record, arrival_time_record, i_val_record, j_val_record = analysis_one_bus(
        line57_onebus_temp, max_station)
    # decompose each element and store them into different objects
    for j in range(len(round_direction_list)):
        # print(arrival_time_record[j])
        if arrival_time_record[j].sum() == 0:
            continue
        round_j = Round(bus_id, round_direction_list[j], i_val_record[j], j_val_record[j], arrival_time_record[j],
                        number_passenger_record[j])
        if round_j.direction > 0:
            forward_round_info.append(round_j)
        else:
            backward_round_info.append(round_j)

print('sorting different record into forward record and backward record')
# forward_round_info is a list of Round objects
# each Round object has estimated arrival time at each station
# if there is no estimation, it will be 0
forward_round_info.sort()
backward_round_info.sort()
import matplotlib.pyplot as plt
import seaborn as sns

ss = [s.round_last_time for s in forward_round_info]
M = matrix_with_missing_construction(forward_round_info)
plt.plot(ss)
plt.figure()
sns.heatmap(M)
plt.xlabel('station')
plt.ylabel('round of bus')
plt.title('before data imputation')
plt.show()

"""
# plt.figure()
