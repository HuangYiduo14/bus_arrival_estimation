# -*- coding: utf-8 -*-
import geopandas
import pandas as pd
from sqlalchemy import create_engine

xsh_s2n = geopandas.read_file('data_car/data/西三环南向北/西三环南向北.SHP', encoding='gbk')
speed_record = pd.read_excel('data_car/data/link05m_lv5_20180601_西三环南向北.xlsx')
speed_record_37592 = speed_record.loc[speed_record['linkid'] == 37592]

engine = create_engine('mysql+mysqlconnector://root:a2=b2=c2@localhost/beijing_bus_liuliqiao', echo=False)
cnx = engine.raw_connection()
count_start_end = pd.read_csv('count_start_end.csv', encoding='utf-8')


stop_pair=['公主坟南', '六里桥北里']
count_start_end_local = count_start_end.loc[count_start_end['NAME_y'].isin(stop_pair)]
linenum_list = count_start_end_local['linenum'].unique()

def max_min(series):
    return series.max()-series.loc[series<0].max()

result_diff = pd.DataFrame(columns=['diff_max','count','max_pivot','line_id'])
for line_id in linenum_list:
    station_intereted = tuple(count_start_end_local.loc[count_start_end['linenum'] == line_id, 'num'].values.tolist())
    if(len(station_intereted)==1):
        continue
    direction = count_start_end.loc[count_start_end['linenum'] == line_id, 'direction'].values[0]
    if direction==1:
        station1 = min(station_intereted)
        station2 = max(station_intereted)
    else:
        station1 = max(station_intereted)
        station2 = min(station_intereted)
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

    start_interested['time0'] = start_interested['start_time'] + start_interested['trans_date'] * 1000000
    start_interested['time'] = pd.to_datetime(start_interested['time0'].astype(str))
    end_interested['time0'] = end_interested['trans_time'] + end_interested['trans_date'] * 1000000
    end_interested['time'] = pd.to_datetime(end_interested['time0'].astype(str))

    end_interested.sort_values(['bus_id', 'end_station', 'time'], inplace=True)
    end_interested.set_index(['bus_id', 'end_station'], inplace=True)
    start_interested.sort_values(['bus_id', 'start_station', 'time'], inplace=True)
    start_interested.set_index(['bus_id', 'start_station'], inplace=True)
    # here we try to find the last time of each bus at each station
    for bus_id_station in start_interested.index.unique():
        start_interested.loc[bus_id_station, 'diff_time'] = start_interested.loc[bus_id_station, 'time'] - \
                                                            start_interested.loc[bus_id_station, 'time'].shift()
        end_interested.loc[bus_id_station, 'diff_time'] = end_interested.loc[bus_id_station, 'time'] - \
                                                          end_interested.loc[bus_id_station, 'time'].shift()
    end_interested['diff_time'].fillna(pd.to_timedelta('21min'), inplace=True)
    start_interested['diff_time'].fillna(pd.to_timedelta('21min'), inplace=True)
    start_interested['last_record'] = start_interested['diff_time'] >= pd.to_timedelta('20min')
    end_interested['last_record'] = end_interested['diff_time'] >= pd.to_timedelta('20min')
    start_interested['new_idx'] = start_interested['last_record'].cumsum()
    end_interested['new_idx'] = end_interested['last_record'].cumsum()
    start_interested.reset_index(inplace=True)
    end_interested.reset_index(inplace=True)
    new_start = start_interested.groupby(['bus_id','start_station','new_idx']).agg(
        {
            'time0':['min','max','count','mean']
        }
    )
    new_end = end_interested.groupby(['bus_id', 'end_station', 'new_idx']).agg(
        {
            'time0': ['min', 'max', 'count', 'mean']
        }
    )
    new_start.columns = new_start.columns.get_level_values(1)
    new_end.columns = new_end.columns.get_level_values(1)
    new_start.reset_index(inplace=True)
    new_end.reset_index(inplace=True)
    new_start.rename(columns={'start_station':'station'},inplace=True)
    new_end.rename(columns={'end_station': 'station'}, inplace=True)
    new_start['is_start']=1
    new_end['is_start']=0
    new_start_end = pd.concat([new_start,new_end],sort=True)
    new_start_end.sort_values(['bus_id','max'],inplace=True)
    new_start_end.reset_index(drop=True,inplace=True)
    new_start_end['diff_station'] = new_start_end['station'] - new_start_end['station'].shift()
    new_start_end['diff_station'].fillna(-direction,inplace=True)
    new_start_end['run_idx'] = new_start_end['diff_station']==-direction
    new_start_end['run_idx'] =new_start_end['run_idx'].cumsum()
    new_start_end.set_index('run_idx',inplace=True)
    pivot_record = new_start_end.loc[new_start_end['diff_station']==direction,'max']
    new_start_end = new_start_end.join(pivot_record,rsuffix='_pivot')
    time2sec = lambda x: x%100 + 60*((x//100)%100) + 3600*((x//10000)%100) + 3600*24*((x//1000000)%100)
    new_start_end['diff_max'] = new_start_end['max'].apply(time2sec) - new_start_end['max_pivot'].apply(time2sec)
    new_start_end = new_start_end.loc[new_start_end['diff_max']>-30*60]
    new_start_end = new_start_end.loc[new_start_end['diff_max'] < 10*60]
    print('start pivoting bus line: ',line_id)
    result_diff0 = new_start_end.groupby(level=0).agg({'diff_max': max_min, 'count': 'sum','max_pivot':'mean'}).dropna()
    result_diff0['line_id']=line_id
    result_diff = pd.concat([result_diff,result_diff0])

result_diff.to_csv('result_{0}_{1}_diff.csv'.format(stop_pair[0],stop_pair[1]))
result_diff[['diff_max','count']].hist(bins=100)



cnx.close()
