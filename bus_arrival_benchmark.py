import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import itertools

# initialize sql connector
cnx = mysql.connector.connect(user='root', password='a2=b2=c2', database='beijing_bus_liuliqiao')
line_id = 57
sql_select_line57 = """
        select trans_time, trans_date, start_station, start_time, end_station, bus_id
        from ic_record
        where line_id = {0}0 or line_id = {0}1
""".format(line_id)
line57_record = pd.read_sql(sql_select_line57, cnx)
cnx.close()
# preprocess time data: convert yyyymmdd HHMMSS to integer: seconds from 20180601 00:00:00
line57_record['trans_time'] = (line57_record['trans_date'] - 20180601) * 24 * 3600 + line57_record[
    'trans_time'] // 10000 * 3600 + (line57_record['trans_time'] % 10000) // 100 * 60 + line57_record[
                                  'trans_time'] % 100
le = LabelEncoder()
line57_record['bus_unique'] = le.fit_transform(line57_record['bus_id'])
station_unique = line57_record['end_station'].unique()
max_station = station_unique.max()
i = 0
line57_onebus = line57_record.loc[line57_record['bus_unique'] == i]
line57_onebus_temp = line57_onebus.sort_values('trans_time').drop(['bus_id', 'bus_unique'], axis=1)

# step 1. aggregate record index into aggregate station table
# for each middle station, we get one tuple of record index; for each start or end station, we get two tuples
# each element is a list[[record_index_list]]
print('end of sql, start aggregating record into station')
last_station = line57_onebus_temp.iloc[0]['end_station']
record_index_list = list()
aggregate_station_list = list()
station_j = 0
for j in range(line57_onebus_temp.shape[0]):
    this_station = line57_onebus_temp.iloc[j]['end_station']
    if last_station == this_station:
        record_index_list.append(line57_onebus_temp.index[j])
    else:
        if j+1 < line57_onebus_temp.shape[0]:
            next_station = line57_onebus_temp.iloc[j+1]['end_station']
            if last_station == next_station:
                # if we find error in station record, i.e. station i, station i+1, station i, then we make correction
                line57_onebus_temp.iloc[j]['end_station'] = last_station
                continue
        aggregate_station_list.append(record_index_list)
        record_index_list = [line57_onebus_temp.index[j]]
        last_station = this_station
# step 2. detect round information
# for the first element, if we have direction information, then we can construct one round and match direction
# otherwise, detect next 2 elements to see if there is a trend
# then initialize an array with the first element being the direction,
# others being the matched index in aggregate station table
total_round = list()
round_direction_list = list()
num_station_in_record = len(aggregate_station_list)
i = 0
first_next_round = []
first_next_station = 999
while True:
    if i+1 >= num_station_in_record:
        break
    this_round = [[] for k in range(max_station)]
    if first_next_station<900:
        this_round[int(first_next_station)-1] = first_next_round
    flag = False
    flag_next2 = False
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
            min_time = line57_onebus_temp.loc[aggregate_station_list[i-1][-1], 'trans_time']
            max_time = line57_onebus_temp.loc[next_element[0], 'trans_time']
            divide = sum(line57_onebus_temp.loc[this_element, 'trans_time']<= (min_time+max_time)/2.)
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
# step 3. merge possibly round trip
ind = 0
while ind+1 < len(total_round):
    if round_direction_list[ind]!= round_direction_list[ind+1]:
        ind += 1
        continue
    #print('one possible')
    this_round_filled = [not (not (station)) for station in total_round[ind]]
    next_round_filled = [not (not (station)) for station in total_round[ind + 1]]
    common_station = sum([this_round_filled[i] and next_round_filled[i] for i in range(len(this_round_filled))])
    if common_station <= 3:
        new_round = [ total_round[ind][i] + total_round[ind+1][i] for i in range(max_station)]
        total_round[ind] = new_round
        del total_round[ind+1]
        del round_direction_list[ind+1]
        #print('one element del')
    else:
        ind+=1
# step 4. figure out number of boarding and alighting for each round
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
            if round_direction_list[ind_round]*np.sign(1.*key+1. - start_station)>0 and start_station>0 and start_station <= max_station:
                round_boarding[int(start_station)-1] += 1
                round_valid_alighting[key]+=1
    alighting_record.append(round_alighting)
    boarding_record.append(round_boarding)
    valid_alighting_record.append(round_valid_alighting)
    # calculate passenger number on board at each station for each round
    s = 0
    for key in range(len(round)):
        if round_direction_list[ind_round]>0:
            real_key = key
        else:
            real_key = len(round)-1-key
        s += round_boarding[real_key]
        s -= round_valid_alighting[real_key]
        round_passenger_number[real_key] = s
    number_passenger_record.append(round_passenger_number)
# step 5. estimate arrival time (using only current data) for each station using empirical formula

# step 6. calculate historical data for this line

# plot to verify
plt.figure()
for ind,round in enumerate(total_round):
    one_round = list(itertools.chain.from_iterable(round))
    plt.scatter(line57_onebus_temp.loc[one_round,'trans_time'], line57_onebus_temp.loc[one_round,'end_station'])
    plt.text(line57_onebus_temp.loc[one_round[0],'trans_time'], line57_onebus_temp.loc[one_round[0],'end_station'], str(round_direction_list[ind]>0))
    plt.plot(line57_onebus_temp.loc[one_round, 'trans_time'], line57_onebus_temp.loc[one_round, 'end_station'],alpha=0.2,color='black')
plt.show()










'''
# then we find outliers and cluster
delta1 = 72
delta2 = 5

station_i = np.zeros(len(aggregate_station_list))
station_j = np.zeros(len(aggregate_station_list))

for ind, element in enumerate(aggregate_station_list):
    this_station = line57_onebus_temp.loc[element[0],'end_station']
    if ind+1<len(aggregate_station_list):
        0
'''


# then we calculate potential direction of each element







