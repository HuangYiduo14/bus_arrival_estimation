import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import itertools

# initialize sql connector
print('getting sql data')
cnx = mysql.connector.connect(user='root', password='', database='beijing_bus_liuliqiao')
line_id = 57 # this line got second most records
sql_select_line57 = """
        select trans_time, trans_date, start_station, start_time, end_station, bus_id
        from ic_record
        where line_id = {0}0 or line_id = {0}1
""".format(line_id)
line57_record = pd.read_sql(sql_select_line57, cnx)
cnx.close()
print('database connection closed')

def aggregate_record_station(line57_onebus_temp):
    # step 1. aggregate record index into aggregate station table
    # for each middle station, we get one list of record index
    # each element is a list[[record_index_list]]
    print('start aggregating record into station')
    last_station = line57_onebus_temp.iloc[0]['end_station']
    record_index_list = list()
    aggregate_station_list = list()
    for j in range(line57_onebus_temp.shape[0]):
        this_station = line57_onebus_temp.iloc[j]['end_station']
        if last_station == this_station:
            record_index_list.append(line57_onebus_temp.index[j])
        else:
            if j + 1 < line57_onebus_temp.shape[0]:
                next_station = line57_onebus_temp.iloc[j + 1]['end_station']
                if last_station == next_station:
                    # if we find error in station record, i.e. station i, station i+1, station i, then we make correction
                    line57_onebus_temp.iloc[j]['end_station'] = last_station
                    continue
            # if we find that station is truly different, append current record_list and reinitialize record_list
            aggregate_station_list.append(record_index_list)
            record_index_list = [line57_onebus_temp.index[j]]
            last_station = this_station
    return aggregate_station_list


def detect_round_info(aggregate_station_list, line57_onebus_temp, max_station):
    # step 2. detect round information
    # if the direction changed, a new round starts, otherwise append record to current round
    print('start round decomposing')
    total_round = list()
    round_direction_list = list()
    num_station_in_record = len(aggregate_station_list)
    i = 0
    first_next_round = []
    first_next_station = 999
    while True:
        if i + 1 >= num_station_in_record:
            break
        this_round = [[] for k in range(max_station)]
        if first_next_station < 900:
            this_round[int(first_next_station) - 1] = first_next_round
        flag = False
        while True:
            first_next_station = 999
            first_next_round = []
            if i + 1 >= num_station_in_record:
                break
            this_element = aggregate_station_list[i]
            next_element = aggregate_station_list[i + 1]
            this_station = line57_onebus_temp.loc[this_element[0], 'end_station']
            next_station = line57_onebus_temp.loc[next_element[0], 'end_station']
            this_direction = np.sign(next_station - this_station)
            if flag and this_direction != last_direction:
                # if the direction is not same, it indicated that we need a different round
                # we split the end/start station according to their time distance to last round or next round record
                min_time = line57_onebus_temp.loc[aggregate_station_list[i - 1][-1], 'trans_time']
                max_time = line57_onebus_temp.loc[next_element[0], 'trans_time']
                divide = sum(line57_onebus_temp.loc[this_element, 'trans_time'] <= (min_time + max_time) / 2.)
                last_this_round = this_element[:divide]
                first_next_round = this_element[divide:]
                this_round[int(this_station) - 1] = last_this_round
                first_next_station = this_station
                i += 1
                break
            this_round[int(this_station) - 1] = this_element
            i += 1
            last_direction = this_direction
            flag = True
        total_round.append(this_round)
        round_direction_list.append(last_direction)
    return total_round, round_direction_list


def merge_one_round(total_round, round_direction_list, max_station):
    # step 3. merge possibly round trip
    # because some data point has error station information, some rounds are split into different rounds
    # here we merge two rounds if they have the same direction and
    print('start round merging')
    ind = 0
    while ind + 1 < len(total_round):
        if round_direction_list[ind] != round_direction_list[ind + 1]:
            ind += 1
            continue
        # print('one possible')
        this_round_filled = [not (not (station)) for station in total_round[ind]]
        next_round_filled = [not (not (station)) for station in total_round[ind + 1]]
        common_station = sum([this_round_filled[i] and next_round_filled[i] for i in range(len(this_round_filled))])
        if common_station <= 3:
            new_round = [total_round[ind][i] + total_round[ind + 1][i] for i in range(max_station)]
            total_round[ind] = new_round
            del total_round[ind + 1]
            del round_direction_list[ind + 1]
            # print('one element del')
        else:
            ind += 1
    return total_round, round_direction_list


def passenger_number_count(total_round, round_direction_list, line57_onebus_temp, max_station):
    # step 4. figure out number of boarding and alighting for each round
    print('start counting passenger')
    alighting_record = []
    boarding_record = []
    valid_alighting_record = []
    number_passenger_record = []
    for ind_round, round in enumerate(total_round):
        round_alighting = np.zeros(max_station)
        round_valid_alighting = np.zeros(max_station)
        round_boarding = np.zeros(max_station)
        round_passenger_number = np.zeros(max_station)
        for key, station in enumerate(round):
            round_alighting[key] += len(station)
            for record in station:
                start_station = line57_onebus_temp.loc[record, 'start_station']
                if round_direction_list[ind_round] * np.sign(
                        1. * key + 1. - start_station) > 0 and start_station > 0 and start_station <= max_station:
                    round_boarding[int(start_station) - 1] += 1
                    round_valid_alighting[key] += 1
        alighting_record.append(round_alighting)
        boarding_record.append(round_boarding)
        valid_alighting_record.append(round_valid_alighting)
        # calculate passenger number on board at each station for each round
        s = 0
        for key in range(len(round)):
            if round_direction_list[ind_round] > 0:
                real_key = key
            else:
                real_key = len(round) - 1 - key
            s += round_boarding[real_key]
            s -= round_valid_alighting[real_key]
            round_passenger_number[real_key] = s
        number_passenger_record.append(round_passenger_number)
    return alighting_record, boarding_record, valid_alighting_record, number_passenger_record


class Round:
    def __init__(self, bus_id, direction, i_val_round, j_val_round, arrival_time_round, number_passenger_round):
        self.bus_id = bus_id
        self.direction = direction
        self.i_val_round = i_val_round # total number of passengers alighting at this station
        self.j_val_round = j_val_round # total number of aggregated passengers
        self.arrival_time_round = arrival_time_round # estimated arrival time
        self.number_passenger_round = number_passenger_round # total of passenger on board at this station
        self.round_last_time = arrival_time_round.max() # the last valid estimated arrival
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


def estimate_arrival_time_local(total_round, line57_onebus_temp, number_passenger_record, max_station):
    # setting the time range for outliers and record clustering
    print('start benchmark local arrival time estimation')
    delta1 = 72  # threshold for outlier
    delta2 = 5  # threshold for clustering
    seat_number = 58  # this number is according to baike.baidu.com<<<<<<<<<<<<<<<<!!!!!!!
    arrival_time_record = np.zeros((len(total_round), max_station))
    # arrival time record is a round*station matrix <double>, this matrix records all inferred station-time information
    i_val_record = np.zeros((len(total_round), max_station))
    j_val_record = np.zeros((len(total_round), max_station))
    for round_ind, round in enumerate(total_round):
        for station_ind, station in enumerate(round):
            time_sequence = line57_onebus_temp.loc[station, 'trans_time'].values
            current_i = len(station)
            if current_i == 0:
                continue
            tl = time_sequence[-1]
            while True:
                lag_judge = [time_sequence[i] - time_sequence[i + 1] <= delta1 for i in range(len(time_sequence) - 1)]
                if sum(lag_judge) >= len(lag_judge):
                    break
                time_sequence = time_sequence[lag_judge]
            lag_judge = [time_sequence[i] - time_sequence[i + 1] <= delta2 for i in range(len(time_sequence) - 1)]
            current_j = sum(lag_judge)
            if current_i > 2:
                if current_j >= 3:
                    arrival_time_record[round_ind, station_ind] = tl - (1.17 * current_j - 2.27)
                else:
                    Nm = 1. * number_passenger_record[round_ind][station_ind] / seat_number
                    arrival_time_record[round_ind, station_ind] = tl - (-15.59 * Nm + 63.63 * Nm - 68.)
            j_val_record[round_ind, station_ind] = current_j
            i_val_record[round_ind, station_ind] = current_i
    return arrival_time_record, i_val_record, j_val_record


def analysis_one_bus(line57_onebus_temp, max_station):
    # step 1. aggregate records into stations
    aggregate_station_list = aggregate_record_station(line57_onebus_temp)
    # step 2. detect round information
    total_round, round_direction_list = detect_round_info(aggregate_station_list, line57_onebus_temp, max_station)
    # step 3. merge possibly round trip
    total_round, round_direction_list = merge_one_round(total_round, round_direction_list, max_station)
    # step 4. figure out number of boarding and alighting for each round
    alighting_record, boarding_record, valid_alighting_record, number_passenger_record = passenger_number_count(
        total_round, round_direction_list, line57_onebus_temp, max_station)
    # step 5. estimate arrival time (using only current data) for each station using empirical formula
    arrival_time_record, i_val_record, j_val_record = estimate_arrival_time_local(total_round, line57_onebus_temp,
                                                                                  number_passenger_record, max_station)
    print('this bus done' + '=' * 50)
    return round_direction_list, number_passenger_record, arrival_time_record, i_val_record, j_val_record


def matrix_with_missing_construction(round_info):
    row_num = len(round_info)
    col_num = len(round_info[0].arrival_time_round)
    M = np.zeros((row_num,col_num))
    for i in range(row_num):
        M[i] = round_info[i].arrival_time_round
    return M

def benchmark_matrix_filling(round_info, threshold=30*60, forward=True):
    for ind_round, round in enumerate(round_info):
        neighbor_round = []
        previous_round_ind = ind_round -1
        after_round_ind = ind_round + 1
        while previous_round_ind>=0:
            if round.round_last_time - round_info[previous_round_ind].round_last_time>threshold:
                break
            neighbor_round.append(previous_round_ind)
            previous_round_ind-=1

        while after_round_ind < len(round_info):
            if round_info[after_round_ind].round_last_time - round.round_last_time> threshold:
                break
            neighbor_round.append(after_round_ind)
            after_round_ind+=1

        for ind_station in range(len(round.arrival_time_round)-1):
            if forward:
                true_ind_station = ind_station
                direction = 1
            else:
                true_ind_station = len(round.arrival_time_round) - 1 - ind_station
                direction = -1
            if round.arrival_time_round[true_ind_station]<=0. or round.arrival_time_round[true_ind_station+direction]>0.:
                continue
            neighbor_record = []
            for possible_round_ind in neighbor_round:
                neighbor_this = round_info[possible_round_ind].arrival_time_round[true_ind_station]
                neighbor_next = round_info[possible_round_ind].arrival_time_round[true_ind_station+direction]
                if neighbor_this>0 and neighbor_next>0:
                    neighbor_record.append(neighbor_next-neighbor_this)
            if len(neighbor_record)>0:
                round_info[ind_round].arrival_time_round[true_ind_station+direction] = round.arrival_time_round[true_ind_station] + sum(neighbor_record)/len(neighbor_record)
    return round_info

# preprocess time data: convert yyyymmdd HHMMSS to integer: seconds from 20180601 00:00:00
line57_record['trans_time'] = (line57_record['trans_date'] - 20180601) * 24 * 3600 + line57_record[
    'trans_time'] // 10000 * 3600 + (line57_record['trans_time'] % 10000) // 100 * 60 + line57_record[
                                  'trans_time'] % 100
le = LabelEncoder()
line57_record['bus_unique'] = le.fit_transform(line57_record['bus_id'])
station_unique = line57_record['end_station'].unique()
max_station = station_unique.max()
# plt.figure()
forward_round_info = []
backward_round_info = []
for i in range(3):
    line57_onebus = line57_record.loc[line57_record['bus_unique'] == i]
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
        if round_j.direction>0:
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
