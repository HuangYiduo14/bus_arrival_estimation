import mysql.connector
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# initialize sql connector
print('getting sql data')
cnx = mysql.connector.connect(user='root', password='lsss1122.lyd', database='beijing_bus_liuliqiao')
line_id = 57  # this line got second most records
sql_select_line57 = """
        select trans_time, trans_date, start_station, start_time, end_station, bus_id
        from ic_record
        where line_id = {0}0 or line_id = {0}1
""".format(line_id)
line57_record = pd.read_sql(sql_select_line57, cnx)
cnx.close()
print('database connection closed')


# def aggregate_record_station(line57_onebus_temp):
#     # step 1. aggregate record index into aggregate station table
#     # for each middle station, we get one list of record index
#     # each element is a list[[record_index_list]]
#     print('start aggregating record into station')
#     last_station = line57_onebus_temp.iloc[0]['end_station']
#     record_index_list = list()
#     aggregate_station_list = list()
#     for j in range(line57_onebus_temp.shape[0]):
#         this_station = line57_onebus_temp.iloc[j]['end_station']
#         if last_station == this_station:
#             record_index_list.append(line57_onebus_temp.index[j])
#         else:
#             if j + 1 < line57_onebus_temp.shape[0]:
#                 next_station = line57_onebus_temp.iloc[j + 1]['end_station']
#                 if last_station == next_station:
#                     # if we find error in station record, i.e. station i, station i+1, station i, then we make correction
#                     line57_onebus_temp.iloc[j]['end_station'] = last_station
#                     continue
#             # if we find that station is truly different, append current record_list and reinitialize record_list
#             aggregate_station_list.append(record_index_list)
#             record_index_list = [line57_onebus_temp.index[j]]
#             last_station = this_station
#     return aggregate_station_list


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
    df_new_station['direction'] = (df_new_station['end_station'].shift(-1) - df_new_station['end_station']).clip(
        lower=-1, upper=1)
    # 2.1 Error occurs at only 1 station
    l_err_station_1 = df_new_station[(df_new_station['direction'] == df_new_station['direction'].shift(-1)) &
                                     (df_new_station['direction'].shift(1) != df_new_station[
                                         'direction'])].index.tolist()
    # 2.2 Error occurs at consecutive several stations
    l_err_station_2 = df_new_station[(df_new_station['direction'] != df_new_station['direction'].shift(-1)) &
                                     (df_new_station['direction'].shift(1) != df_new_station['direction']) &
                                     (df_new_station['trans_time'].shift(1) - df_new_station['trans_time'].shift(
                                         -1) <= 300)].index.tolist()
    l_next_station = df_new_station[(df_new_station['direction'].shift(1) == df_new_station['direction']) &
                                    (df_new_station['end_station'].shift(2) == df_new_station['end_station']) &
                                    (df_new_station['direction'].shift(2) != df_new_station['direction'].shift(
                                        1))].index.tolist()
    df_new_station.loc[l_err_station_1, 'is_new_station'] = 0
    df_new_station.loc[l_err_station_2, 'is_new_station'] = 0
    df_new_station.loc[l_next_station, 'is_new_station'] = 0
    df_new_station.loc[0, 'is_new_station'] = 1
    df_record.loc[df_record['is_new_station'] > 0, 'is_new_station'] = df_new_station['is_new_station']
    df_record['is_new_station'] = df_record['is_new_station'].clip(upper=1)
    df_record.loc[df_record['is_new_station'] == 0, 'end_station'] = np.nan
    df_record = df_record.fillna(method='ffill')
    df_record.loc[df_record['end_station'] == df_record['end_station'].shift(1), 'is_new_station'] = 0


# def detect_round_info(aggregate_station_list, line57_onebus_temp, max_station):
#     # step 2. detect round information
#     # if the direction changed, a new round starts, otherwise append record to current round
#     print('start round decomposing')
#     total_round = list()
#     round_direction_list = list()
#     num_station_in_record = len(aggregate_station_list)
#     i = 0
#     first_next_round = []
#     first_next_station = 999
#     while True:
#         if i + 1 >= num_station_in_record:
#             break
#         this_round = [[] for k in range(max_station)]
#         if first_next_station < 900:
#             this_round[int(first_next_station) - 1] = first_next_round
#         flag = False
#         while True:
#             first_next_station = 999
#             first_next_round = []
#             if i + 1 >= num_station_in_record:
#                 break
#             this_element = aggregate_station_list[i]
#             next_element = aggregate_station_list[i + 1]
#             this_station = line57_onebus_temp.loc[this_element[0], 'end_station']
#             next_station = line57_onebus_temp.loc[next_element[0], 'end_station']
#             this_direction = np.sign(next_station - this_station)
#             if flag and this_direction != last_direction:
#                 # if the direction is not same, it indicated that we need a different round
#                 # we split the end/start station according to their time distance to last round or next round record
#                 min_time = line57_onebus_temp.loc[aggregate_station_list[i - 1][-1], 'trans_time']
#                 max_time = line57_onebus_temp.loc[next_element[0], 'trans_time']
#                 divide = sum(line57_onebus_temp.loc[this_element, 'trans_time'] <= (min_time + max_time) / 2.)
#                 last_this_round = this_element[:divide]
#                 first_next_round = this_element[divide:]
#                 this_round[int(this_station) - 1] = last_this_round
#                 first_next_station = this_station
#                 i += 1
#                 break
#             this_round[int(this_station) - 1] = this_element
#             i += 1
#             last_direction = this_direction
#             flag = True
#         total_round.append(this_round)
#         round_direction_list.append(last_direction)
#     return total_round, round_direction_list

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


# def merge_one_round(total_round, round_direction_list, max_station):
#     # step 3. merge possibly round trip
#     # because some data point has error station information, some rounds are split into different rounds
#     # here we merge two rounds if they have the same direction and
#     print('start round merging')
#     ind = 0
#     while ind + 1 < len(total_round):
#         if round_direction_list[ind] != round_direction_list[ind + 1]:
#             ind += 1
#             continue
#         # print('one possible')
#         this_round_filled = [not (not (station)) for station in total_round[ind]]
#         next_round_filled = [not (not (station)) for station in total_round[ind + 1]]
#         common_station = sum([this_round_filled[i] and next_round_filled[i] for i in range(len(this_round_filled))])
#         if common_station <= 3:
#             new_round = [total_round[ind][i] + total_round[ind + 1][i] for i in range(max_station)]
#             total_round[ind] = new_round
#             del total_round[ind + 1]
#             del round_direction_list[ind + 1]
#             # print('one element del')
#         else:
#             ind += 1
#     return total_round, round_direction_list

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


# def passenger_number_count(total_round, round_direction_list, line57_onebus_temp, max_station):
#     # step 4. figure out number of boarding and alighting for each round
#     print('start counting passenger')
#     alighting_record = []
#     boarding_record = []
#     valid_alighting_record = []
#     number_passenger_record = []
#     for ind_round, round in enumerate(total_round):
#         round_alighting = np.zeros(max_station)
#         round_valid_alighting = np.zeros(max_station)
#         round_boarding = np.zeros(max_station)
#         round_passenger_number = np.zeros(max_station)
#         for key, station in enumerate(round):
#             round_alighting[key] += len(station)
#             for record in station:
#                 start_station = line57_onebus_temp.loc[record, 'start_station']
#                 if round_direction_list[ind_round] * np.sign(
#                         1. * key + 1. - start_station) > 0 and start_station > 0 and start_station <= max_station:
#                     round_boarding[int(start_station) - 1] += 1
#                     round_valid_alighting[key] += 1
#         alighting_record.append(round_alighting)
#         boarding_record.append(round_boarding)
#         valid_alighting_record.append(round_valid_alighting)
#         # calculate passenger number on board at each station for each round
#         s = 0
#         for key in range(len(round)):
#             if round_direction_list[ind_round] > 0:
#                 real_key = key
#             else:
#                 real_key = len(round) - 1 - key
#             s += round_boarding[real_key]
#             s -= round_valid_alighting[real_key]
#             round_passenger_number[real_key] = s
#         number_passenger_record.append(round_passenger_number)
#     return alighting_record, boarding_record, valid_alighting_record, number_passenger_record

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
        self.arrival_time_round = arrival_time_round  # estimated arrival time
        self.number_passenger_round = number_passenger_round  # total of passenger on board at this station
        self.round_last_time = arrival_time_round.max()  # the last valid estimated arrival
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


# def estimate_arrival_time_local(total_round, line57_onebus_temp, number_passenger_record, max_station):
#     # setting the time range for outliers and record clustering
#     print('start benchmark local arrival time estimation')
#     delta1 = 72  # threshold for outlier
#     delta2 = 5  # threshold for clustering
#     seat_number = 58  # this number is according to baike.baidu.com<<<<<<<<<<<<<<<<!!!!!!!
#     arrival_time_record = np.zeros((len(total_round), max_station))
#     # arrival time record is a round*station matrix <double>, this matrix records all inferred station-time information
#     i_val_record = np.zeros((len(total_round), max_station))
#     j_val_record = np.zeros((len(total_round), max_station))
#     for round_ind, round in enumerate(total_round):
#         for station_ind, station in enumerate(round):
#             time_sequence = line57_onebus_temp.loc[station, 'trans_time'].values
#             current_i = len(station)
#             if current_i == 0:
#                 continue
#             tl = time_sequence[-1]
#             while True:
#                 lag_judge = [time_sequence[i] - time_sequence[i + 1] <= delta1 for i in range(len(time_sequence) - 1)]
#                 if sum(lag_judge) >= len(lag_judge):
#                     break
#                 time_sequence = time_sequence[lag_judge]
#             lag_judge = [time_sequence[i] - time_sequence[i + 1] <= delta2 for i in range(len(time_sequence) - 1)]
#             current_j = sum(lag_judge)
#             if current_i > 2:
#                 if current_j >= 3:
#                     arrival_time_record[round_ind, station_ind] = tl - (1.17 * current_j - 2.27)
#                 else:
#                     Nm = 1. * number_passenger_record[round_ind][station_ind] / seat_number
#                     arrival_time_record[round_ind, station_ind] = tl - (-15.59 * Nm + 63.63 * Nm - 68.)
#             j_val_record[round_ind, station_ind] = current_j
#             i_val_record[round_ind, station_ind] = current_i
#     return arrival_time_record, i_val_record, j_val_record

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
        -15.59 * Nm.loc[l_case2, 'pax_num'] + 63.63 * Nm.loc[l_case2, 'pax_num'] - 68.)

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


def benchmark_matrix_filling(round_info, threshold=30 * 60, forward=True):
    for ind_round, round in enumerate(round_info):
        neighbor_round = []
        previous_round_ind = ind_round - 1
        after_round_ind = ind_round + 1
        while previous_round_ind >= 0:
            if round.round_last_time - round_info[previous_round_ind].round_last_time > threshold:
                break
            neighbor_round.append(previous_round_ind)
            previous_round_ind -= 1

        while after_round_ind < len(round_info):
            if round_info[after_round_ind].round_last_time - round.round_last_time > threshold:
                break
            neighbor_round.append(after_round_ind)
            after_round_ind += 1

        for ind_station in range(len(round.arrival_time_round) - 1):
            if forward:
                true_ind_station = ind_station
                direction = 1
            else:
                true_ind_station = len(round.arrival_time_round) - 1 - ind_station
                direction = -1
            if round.arrival_time_round[true_ind_station] <= 0. or round.arrival_time_round[
                        true_ind_station + direction] > 0.:
                continue
            neighbor_record = []
            for possible_round_ind in neighbor_round:
                neighbor_this = round_info[possible_round_ind].arrival_time_round[true_ind_station]
                neighbor_next = round_info[possible_round_ind].arrival_time_round[true_ind_station + direction]
                if neighbor_this > 0 and neighbor_next > 0:
                    neighbor_record.append(neighbor_next - neighbor_this)
            if len(neighbor_record) > 0:
                round_info[ind_round].arrival_time_round[true_ind_station + direction] = round.arrival_time_round[
                                                                                             true_ind_station] + sum(
                    neighbor_record) / len(neighbor_record)
    return round_info


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
for i in range(3):
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

M = matrix_with_missing_construction(forward_round_info)
print(np.count_nonzero(M))
forward_round_info = benchmark_matrix_filling(forward_round_info)
M1 = matrix_with_missing_construction(forward_round_info)
print(np.count_nonzero(M1))


# plt.legend()
# plt.show()
