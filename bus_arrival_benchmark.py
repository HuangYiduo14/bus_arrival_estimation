import mysql.connector
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# initialize sql connector
print('getting sql data')
cnx = mysql.connector.connect(user='root', password='', database='beijing_bus_liuliqiao')
line_id = 57  # this line got second most records
sql_select_line57 = """
            select trans_time, trans_date, start_station, start_time, end_station, bus_id
            from ic_record
            where line_id = {0}
    """.format(line_id)
line57_record = pd.read_sql(sql_select_line57, cnx)
cnx.close()
print('database connection closed')


def aggregate_record_station(df):
    # step 1. aggregate consecutive end station and calibrate invalid data
    print('start aggregating record into station')
    df_record = df.copy().reset_index(drop=1)

    # is_new_station indicates whether the current record is the first passenger alighting at the station.
    # Initialized with all ones.
    df_record['is_new_station'] = df_record['end_station']
    # 1. If current station is the same as the last record, then set is_new_station to 0.
    df_record.loc[df_record['end_station'] == df_record['end_station'].shift(1), 'is_new_station'] = 0
    # 2. If the last station and the next station is the same, then this record is considered invalid.
    #    Set is_new_station to 0 and calibrate this record.
    df_new_station = df_record[df_record['is_new_station'] > 0].copy()
    l_err_station = df_new_station[((df_new_station['is_new_station'].shift(1) - df_new_station['is_new_station']) * (
            df_new_station['is_new_station'].shift(-1) - df_new_station['is_new_station']) > 0) &
                                   (df_new_station['trans_time'].shift(-1) - df_new_station['trans_time'].shift(
                                       1) < 600)].index.tolist()
    l_next_station = df_new_station[((
                                             df_new_station['is_new_station'].shift(2) - df_new_station[
                                         'is_new_station'].shift(
                                         1)) * (
                                             df_new_station['is_new_station'] - df_new_station['is_new_station'].shift(
                                         1)) > 0) &
                                    (df_new_station['trans_time'] - df_new_station['trans_time'].shift(
                                        2) < 600)].index.tolist()
    df_new_station.loc[l_err_station, 'is_new_station'] = 0
    df_new_station.loc[l_next_station, 'is_new_station'] = 0
    df_record.loc[df_record['is_new_station'] > 0, 'is_new_station'] = df_new_station['is_new_station']
    df_record['is_new_station'] = df_record['is_new_station'].clip(upper=1)
    df_record.loc[df_record['is_new_station'] == 0, 'end_station'] = np.nan
    df_record = df_record.fillna(method='ffill')
    return df_record


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

    l_new_station_time = df_record.loc[l_new_station_ind, 'trans_time'].values
    l_post_time_gap = l_new_station_time[1:] - l_new_station_time[:-1]
    l_pre_time_gap = np.insert(l_post_time_gap, 0, 0)
    l_skew_post = l_pre_time_gap > np.insert(l_post_time_gap, -1, 3600)
    l_pre_direction[l_skew_post] = np.insert(l_post_direction, -1, np.nan)[l_skew_post]

    df_record.loc[l_new_station_ind, 'direction'] = l_pre_direction
    df_record['direction'] = df_record['direction'].clip(lower=-1, upper=1)

    # Detect the round shift station by back-and-forth filling, denoted by 0
    df_record['prev_direction'] = df_record['direction'].fillna(method='ffill').fillna(method='bfill')
    df_record['succ_direction'] = df_record['direction'].fillna(method='bfill').fillna(method='ffill')
    df_record['direction'] = (df_record['prev_direction'] + df_record['succ_direction']) / 2
    # Extract the exact first record before and after round shift
    l_before_change = df_record[(
                                        df_record['direction'].shift(-1) != 0) & (
                                        df_record['direction'].shift(-2) == 0)].index.tolist()
    l_after_change = df_record[(
                                       df_record['direction'].shift(1) == 0) & (
                                       df_record['direction'] != 0)].index.tolist()
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
    df_record.loc[df_record['direction'] == 0, 'direction'] = np.nan
    df_record['direction'] = df_record['direction'].fillna(method='ffill')

    # Calibrate station number which is not consistent with direction
    l_need_cast = (df_record['start_station'] <= df_record['end_station']) * 2 - 1 != df_record['direction']
    df_record.loc[l_need_cast, 'start_station'] = df_record.loc[l_need_cast, 'end_station']

    df_record['is_new_round'] = (df_record['direction'].shift(1) != df_record['direction']).astype(int)

    return df_record.drop(['prev_direction', 'succ_direction'], axis=1)


def merge_one_round(df):
    # step 3. merge possibly round trip
    # because some data point has error station information, some rounds are split into different rounds
    # here we merge some rounds which are really short and have no significant seperation from previous round.
    print('start round merging')
    df_record = df.copy().reset_index(drop=1)

    # Find the start and end records for a proposed new round.
    l_start_round = df_record[df_record['is_new_round'] == 1].index.tolist()
    l_end_round = df_record[df_record['is_new_round'].shift(-1) == 1].index.tolist()
    df_round_time = pd.DataFrame()
    df_round_time['start_time'] = df_record.loc[l_start_round, 'trans_time'].reset_index(drop=1)
    df_round_time['end_time'] = df_record.loc[l_end_round, 'trans_time'].reset_index(drop=1)
    df_round_time['round_time'] = df_round_time['end_time'] - df_round_time['start_time']
    df_round_time['gap_time'] = df_round_time['start_time'] - df_round_time['end_time'].shift(1)

    # A valid merge requires the round is less than 10 min and has no significant seperation with the previous one.
    l_valid_merge = (
            (df_round_time['round_time'] < 600) & (df_round_time['gap_time'] < df_round_time['round_time'])).values
    l_start_round, l_end_round = np.array(l_start_round)[l_valid_merge], np.array(l_end_round)[l_valid_merge[:-1]]
    df_record.loc[l_start_round, 'is_new_round'] = 0

    # Label each round.
    df_record['round_id'] = df_record['is_new_round'].cumsum()

    return df_record


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
        s0 = """
        bus_id = {self.bus_id}
        direction = {self.direction}
        arrival_time_estimated = {self.arrival_time_round}
        """.format(self=self)
        return s0


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


# preprocess time data: convert yyyymmdd HHMMSS to integer: seconds from 20180601 00:00:00
line57_record['trans_time'] = (line57_record['trans_date'] - 20180601) * 24 * 3600 + line57_record[
    'trans_time'] // 10000 * 3600 + (
                                      line57_record[
                                          'trans_time'] % 10000) // 100 * 60 + \
                              line57_record[
                                  'trans_time'] % 100
le = LabelEncoder()
line57_record['bus_unique'] = le.fit_transform(line57_record['bus_id'])
station_unique = line57_record['end_station'].unique()
max_station = station_unique.max()

# Cast invalid station number into 1 and max_station
line57_record = line57_record.drop_duplicates()
line57_record['start_station'] = line57_record['start_station'].clip(lower=1, upper=max_station)
line57_record['end_station'] = line57_record['end_station'].clip(lower=1, upper=max_station)

# plt.figure()
forward_round_info = []
backward_round_info = []
# for i in range(len(le.classes_)):
for i in line57_record['bus_unique'].unique():
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
