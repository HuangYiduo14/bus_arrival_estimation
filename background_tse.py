# -*- coding: utf-8 -*-
import geopandas
import pandas as pd
from sqlalchemy import create_engine

xsh_s2n = geopandas.read_file('data_car/data/西三环南向北/西三环南向北.SHP', encoding='gbk')
speed_record = pd.read_excel('data_car/data/link05m_lv5_20180601_西三环南向北.xlsx')
speed_record_37592 = speed_record.loc[speed_record['linkid'] == 37592]

engine = create_engine('mysql+mysqlconnector://root:******@localhost/beijing_bus_liuliqiao', echo=False)
cnx = engine.raw_connection()
count_start_end = pd.read_csv('count_start_end.csv', encoding='utf-8')
linenum_list = count_start_end['linenum'].unique()

for line_id in linenum_list[:1]:
    station_intereted = tuple(count_start_end.loc[count_start_end['linenum'] == line_id, 'num'].values.tolist())
    direction = count_start_end.loc[count_start_end['linenum'] == line_id, 'direction'].values[0]
    read_interested_start = '''
    select * from ic_record
    where direction={0}
    and line_id ={1}
    and start_station in {2}
    '''.format(direction, int(line_id), station_intereted)
    print('read sql start')
    start_interested = pd.read_sql(sql=read_interested_start, con=engine)
    read_interested_start = '''
        select * from ic_record
        where direction={0}
        and line_id ={1}
        and end_station in {2}
    '''.format(direction, int(line_id), station_intereted)
    end_interested = pd.read_sql(sql=read_interested_start, con=engine)
    print('read sql end')
    start_interested['time'] = start_interested['start_time'] + start_interested['trans_date'] * 1000000
    end_interested['time'] = end_interested['trans_time'] + end_interested['trans_date'] * 1000000

    end_interested.sort_values(['bus_id', 'end_station', 'time'], inplace=True)
    end_interested.set_index(['bus_id', 'end_station'], inplace=True)
    start_interested.sort_values(['bus_id', 'start_station', 'time'], inplace=True)
    start_interested.set_index(['bus_id', 'start_station'], inplace=True)
    # here we try to find the last time of each bus at each station
    for bus_id_station in start_interested.index.unique():
        print(bus_id_station)
        start_interested.loc[bus_id_station, 'diff_time'] = start_interested.loc[bus_id_station, 'time'] - \
                                                            start_interested.loc[bus_id_station, 'time'].shift()
        end_interested.loc[bus_id_station, 'diff_time'] = end_interested.loc[bus_id_station, 'time'] - \
                                                          end_interested.loc[bus_id_station, 'time'].shift()
    end_interested['diff_time'].fillna(2000, inplace=True)
    start_interested['diff_time'].fillna(2000, inplace=True)
    start_interested['last_record'] = start_interested['diff_time'] >= 2000
    end_interested['last_record'] = end_interested['diff_time'] >= 2000
    start_interested['new_idx'] = start_interested['last_record'].cumsum()
    end_interested['new_idx'] = end_interested['last_record'].cumsum()
    start_interested.reset_index(inplace=True)
    end_interested.reset_index(inplace=True)
    new_start = start_interested.groupby(['bus_id','start_station','new_idx']).agg(
        {
            'time':['min','max','count','mean']
        }
    )
    new_end = end_interested.groupby(['bus_id', 'end_station', 'new_idx']).agg(
        {
            'time': ['min', 'max', 'count', 'mean']
        }
    )
    new_start.columns = new_start.columns.get_level_values(1)
    new_end.columns = new_end.columns.get_level_values(1)
    new_start.reset_index(inplace=True)
    new_start.sort_values('max',inplace=True)
    new_start['max_start'] = new_start['max']
    new_end.reset_index(inplace=True)
    new_end.sort_values('max', inplace=True)
    new_end['max_end']=new_end['max']
    start_mg= pd.merge_asof(
        new_start,new_end,
        left_on='max',
        right_on='max',
        left_by=['bus_id','start_station'],
        right_by=['bus_id','end_station'],
        suffixes=['_start','_end'],
        direction='nearest'
    )
    start_mg.sort_values(['bus_id','max'],inplace=True)
    end_mg = pd.merge_asof(
        new_end,new_start,
        left_on='max',
        right_on='max',
        left_by=['bus_id','end_station'],
        right_by=['bus_id','start_station'],
        suffixes=['_end','_start'],
        direction='nearest'
    )


cnx.close()
