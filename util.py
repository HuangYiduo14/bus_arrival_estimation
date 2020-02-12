import pandas as pd
import mysql.connector
from sklearn.preprocessing import LabelEncoder

busstation = pd.read_csv(
    'beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/附件 刷卡数据与GIS对应规则/站点静态表.txt', encoding='gbk')
password = ''



def line_station2chinese(line_id, station_id):
    '''
    get Chinese name for line_id and station_id
    :param line_id: linenum
    :param station_id: num
    :return: Chinese station name
    '''
    if line_id in busstation['linenum']:
        line_info = busstation.loc[busstation['linenum'] == line_id]
        if sum(line_info['num'] == station_id) > 0:
            name = line_info.loc[line_info['num'] == station_id, 'stationname'].values[0]
        else:
            name = station_id
    else:
        name = station_id
    return name
def line2chinese(line_id, direction):
    if line_id in busstation['linenum']:
        if direction >0:
            line_info = busstation.loc[(busstation['linenum'] == line_id)&(busstation['direction'] == '上行'),'linename']
        else:
            line_info = busstation.loc[(busstation['linenum'] == line_id)&(busstation['direction'] == '下行'),'linename']
        if line_info.shape[0]>0:
            name = line_info.iloc[0]
        else:
            name=line_id
    else:
        name=line_id
    return name


def time2int(time):
    '''
    time like 20180601121140 to seconds from 20180601000000
    :param time: int
    :return: int
    '''
    return time // 10000 * 3600 + (time % 10000) // 100 * 60 + time % 100


def int2timedate(time_s):
    '''
    time in seconds to datetime
    :param time_s: seconds int
    :return: datetime
    '''
    day = time_s // (24 * 3600)
    day = 1 + min(day, 30)
    time_s = time_s % (24 * 3600)
    hour = time_s // 3600
    time_s = time_s % 3600
    minute = time_s // 60
    time_s = time_s % 60
    sec = time_s
    return pd.to_datetime('2018-06-{0} {1}:{2}:{3}'.format(day, hour, minute, sec))

def get_all_lines():
    cnx = mysql.connector.connect(user='root', password=password, database='beijing_bus_liuliqiao')
    sql_select_lines = """
                    select distinct line_id
                    from ic_record
    """
    all_lines = pd.read_sql(sql_select_lines, cnx)
    cnx.close()
    return all_lines


def line_bus_count():
    cnx = mysql.connector.connect(user='root', password=password, database='beijing_bus_liuliqiao')
    sql_select_lines = """
                        select line_id, bus_id, direction, count(1) as count_record
                        from ic_record
                        group by line_id, bus_id, direction
        """
    line_bus = pd.read_sql(sql_select_lines, cnx)
    cnx.close()
    return line_bus



def count_line_station():
    print('getting sql data')
    cnx = mysql.connector.connect(user='root', password=password, database='beijing_bus_liuliqiao')
    sql_select_line57 = """
                    select count(1) as count_record, line_id, start_station, direction
                    from ic_record
                    where direction != 0
                    group by line_id, start_station, direction
            """
    line_station_count = pd.read_sql(sql_select_line57, cnx)
    cnx.close()
    return line_station_count


def get_one_line(line_id=57):
    '''
    get bus information of one line from db
    :param line_id: int
    :return: dataframe
    '''
    # initialize sql connector
    print('getting sql data')
    cnx = mysql.connector.connect(user='root', password=password, database='beijing_bus_liuliqiao')
    sql_select_line57 = """
                select trans_time, trans_date, start_station, start_time, end_station, bus_id
                from ic_record
                where line_id = {0}
        """.format(line_id)
    line57_record = pd.read_sql(sql_select_line57, cnx)
    cnx.close()
    print('database connection closed')
    # preprocess time data: convert yyyymmdd HHMMSS to integer: seconds from 20180601 00:00:00
    line57_record['trans_time'] = (line57_record['trans_date'] - 20180601) * 24 * 3600 + line57_record[
        'trans_time'].apply(time2int)
    line57_record['start_time'] = (line57_record['trans_date'] - 20180601) * 24 * 3600 + line57_record[
        'start_time'].apply(time2int)
    le = LabelEncoder()
    line57_record['bus_unique'] = le.fit_transform(line57_record['bus_id'])
    station_unique = line57_record['end_station'].unique()
    max_station = station_unique.max()
    # Cast invalid station number into 1 and max_station
    line57_record = line57_record.drop_duplicates()
    # line57_record['start_station'] = line57_record['start_station'].clip(lower=1, upper=max_station)
    line57_record['end_station'] = line57_record['end_station'].clip(lower=1, upper=max_station)

    return line57_record, le, max_station


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
