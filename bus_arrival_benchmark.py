import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


def find_max_count(cnx):
    # first we select the line with most record
    sql_rank_trans_count = """
    select count(*) as count_line_record, line_id
    from ic_record
    group by line_id
    order by count_line_record DESC
    """
    line_count = pd.read_sql(sql_rank_trans_count, cnx)
    line_count['line_id'] = line_count['line_id'] // 10
    line_merged = line_count.groupby('line_id').sum()
    line_merged.sort_values('count_line_record', ascending=False, inplace=True)
    print('the top 5 line is:')
    print(line_merged.index[:5])
    # after this we found that top line is (1,57,52,890,52300,...)
    return line_merged


def plot_record(cnx, line_id=57):
    # we can plot the arrival and departure record
    sql_select2 = """
        select trans_time, trans_date, start_station, end_station, bus_id
        from ic_record
        where (line_id = {0}0 or line_id = {0}1)
        and trans_date = 20180601
        """.format(line_id)
    line57_record = pd.read_sql(sql_select2, cnx)
    line57_record['trans_time'] = (line57_record['trans_date'] - 20180601) * 24 * 3600 + line57_record[
        'trans_time'] // 10000 * 3600 + (line57_record['trans_time'] % 10000) // 100 * 60 + line57_record[
                                      'trans_time'] % 100
    le = LabelEncoder()
    line57_record['bus_unique'] = le.fit_transform(line57_record['bus_id'])
    for i in le.classes_[:5]:
        line57_onebus = line57_record.loc[line57_record['bus_id'] == i]
        line57_onebus = line57_onebus.sort_values('trans_time')
        plt.scatter(line57_onebus['trans_time'], line57_onebus['end_station'], alpha=0.1, label=i)
        plt.plot(line57_onebus['trans_time'], line57_onebus['end_station'], alpha=0.1)
    plt.legend()
    plt.show()
    return


cnx = mysql.connector.connect(user='root', password='a2=b2=c2', database='beijing_bus_liuliqiao')
line_id = 57
sql_select_line57 = """
        select trans_time, trans_date, start_station, start_time, end_station, bus_id
        from ic_record
        where line_id = {0}0 or line_id = {0}1
""".format(line_id)
line57_record = pd.read_sql(sql_select_line57, cnx)
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
        aggregate_station_list.append(record_index_list)
        if last_station == 1 or last_station== max_station:
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
while True:
    if i+1 >= num_station_in_record:
        break
    this_round = [[] for k in range(max_station)]
    last_direction = 0
    flag = 0
    while True:
        if i + 1 >= num_station_in_record:
            break
        this_element = aggregate_station_list[i]
        next_element = aggregate_station_list[i + 1]
        this_station = line57_onebus_temp.loc[this_element[0], 'end_station']
        next_station = line57_onebus_temp.loc[next_element[0], 'end_station']

        this_direction = np.sign(next_station - this_station)
        this_round[int(this_station)-1] = this_element
        if this_direction == 1 and this_station == max_station:
            break
        if this_direction == -1 and this_station == 1:
            break



        last_direction = this_direction
        i+=1










# then we find outliers and cluster
delta1 = 72
delta2 = 5

station_i = np.zeros(len(aggregate_station_list))
station_j = np.zeros(len(aggregate_station_list))

for ind, element in enumerate(aggregate_station_list):
    this_station = line57_onebus_temp.loc[element[0],'end_station']
    if ind+1<len(aggregate_station_list):
        0



# then we calculate potential direction of each element






cnx.close()
