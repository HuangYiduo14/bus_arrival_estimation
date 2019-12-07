# -*- coding: utf-8 -*-
import geopandas
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

holiday_list = [2,3,9,10,16,17,18,23,24,30]
holiday_list = pd.DataFrame({'day':holiday_list})

xsh_s2n = geopandas.read_file('data_car/data/西三环南向北/西三环南向北.SHP', encoding='gbk')

count_start_end = pd.read_csv('count_start_end.csv', encoding='utf-8')
stop_pair=['公主坟南', '六里桥北里']
count_start_end_local = count_start_end.loc[count_start_end['NAME_y'].isin(stop_pair)]
busstation_shp = geopandas.read_file(
    'beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/02 区域内公交站点GIS信息/station.shp', encoding='gbk')
busstation_shp['STATION_ID'] = pd.to_numeric(busstation_shp['STATION_ID'])
busstation_shp = busstation_shp.loc[busstation_shp['STATION_ID'].isin(count_start_end_local['STATION_ID'])]
stop_set = busstation_shp['geometry'].unary_union.envelope
maxy = max(stop_set.exterior.xy[1])
miny = min(stop_set.exterior.xy[1])
length = maxy-miny
find_y_max = lambda x: max(x.coords.xy[1])
find_y_min = lambda x: min(x.coords.xy[1])
xsh_s2n['max_y'] = xsh_s2n['geometry'].apply(find_y_max)
xsh_s2n['min_y'] = xsh_s2n['geometry'].apply(find_y_min)
xsh_s2n.sort_values('max_y',inplace=True)
xsh_s2n.reset_index(inplace=True,drop=True)

xsh_s2n1 = xsh_s2n.loc[(xsh_s2n['max_y']>miny)&(xsh_s2n['min_y']<maxy)]
xsh_s2n2 = xsh_s2n.loc[xsh_s2n1.index.min()-2:xsh_s2n1.index.max()+2]
base = xsh_s2n1.plot(color='red')
busstation_shp.plot(ax=base)
xsh_s2n.plot(alpha=0.3,ax=base)
xsh_s2n2.plot(alpha=0.5,ax=base,color='green')


speed_record0 = pd.read_excel('data_car/data/link05m_lv5_20180601_西三环南向北.xlsx')
speed_record0 = speed_record0.loc[speed_record0['linkid'].isin(xsh_s2n2.LinkID)]
print('file 0 loaded')
speed_record1 = pd.read_excel('data_car/data/link05m_lv5_20180602_20180607_西三环南向北.xlsx')
speed_record1 = speed_record1.loc[speed_record1['linkid'].isin(xsh_s2n2.LinkID)]
print('file 1 loaded')
speed_record2 = pd.read_excel('data_car/data/link05m_lv5_20180608_20180614_西三环南向北.xlsx')
speed_record2 = speed_record2.loc[speed_record2['linkid'].isin(xsh_s2n2.LinkID)]
print('file 2 loaded')
speed_record3 = pd.read_csv('data_car/data/link05m_lv5_20180615_20180630_西三环南向北.csv')
speed_record3 = speed_record3.loc[speed_record3['linkid'].isin(xsh_s2n2.LinkID)]
print('file 3 loaded')
speed_record = pd.concat([speed_record0,speed_record1,speed_record2,speed_record3])
speed_record['time'] = pd.to_datetime(speed_record['dt'].astype(str)+' '+speed_record['tm'].astype(str))
speed_record=speed_record.drop(['dt','tm'],axis=1)
speed_record = speed_record.loc[speed_record['time'].dt.time>pd.to_datetime('05:00:00').time()]
speed_record = speed_record.loc[speed_record['time'].dt.time<pd.to_datetime('23:59:59').time()]


speed_record['tm'] = speed_record['time'].dt.time.astype(str)
speed_record['tm'] = speed_record['tm'].str[0].astype(int)*10*3600 + speed_record['tm'].str[1].astype(int)*3600 +speed_record['tm'].str[3].astype(int)*10*60+speed_record['tm'].str[4].astype(int)*60

speed_record['base_pattern'] = 0
holiday_record = speed_record.loc[speed_record['time'].dt.day.isin(holiday_list['day'])]
workday_record = speed_record.loc[~(speed_record['time'].dt.day.isin(holiday_list['day']))]

from statsmodels.nonparametric.smoothers_lowess import lowess
def one_link_analysis(record, linkid=35972):
    speed_link = record.loc[record['linkid']==linkid]

    #plt.figure()
    #for day in speed_link['time'].dt.date.unique():
    #    speed_day = speed_link.loc[speed_link['time'].dt.date==day]
    #    plt.scatter(speed_day['tm'], speed_day['speed'])
    x = speed_link['tm'].values
    y = speed_link['speed']
    new = lowess(y,x,frac=0.15)


    return new

plt.figure()
for ind, linkid in enumerate(xsh_s2n2.LinkID.values):
    lowess_result = one_link_analysis(holiday_record, linkid)
    plt.plot(lowess_result[:,0],lowess_result[:,1], label='sequence={0},linkid={1}'.format(ind,linkid))
plt.legend()
plt.xlabel('time(s)')
plt.ylabel('lowess speed')
plt.title('holiday base pattern')

plt.figure()
for ind, linkid in enumerate(xsh_s2n2.LinkID.values):
    lowess_result = one_link_analysis(workday_record, linkid)
    plt.plot(lowess_result[:,0],lowess_result[:,1], label='sequence={0},linkid={1}'.format(ind,linkid))
plt.legend()
plt.xlabel('time(s)')
plt.ylabel('lowess speed')
plt.title('workday base pattern')


"""
def max_min(series):
    return series.max()-series.loc[series<0].max()
engine = create_engine('mysql+mysqlconnector://root:a2=b2=c2@localhost/beijing_bus_liuliqiao', echo=False)
cnx = engine.raw_connection()
result_diff = pd.DataFrame(columns=['diff_max','count','max_pivot','line_id'])
print('line includes', linenum_list)
for line_id in linenum_list:
    print('line_id:', line_id,'-'*50)
    station_intereted = tuple(count_start_end_local.loc[count_start_end_local['linenum'] == line_id, 'num'].values.tolist())
    print('station interested:',station_intereted)
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
    for bus_id_station in end_interested.index.unique():
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
result_diff.hist(column=['diff_max'],bins=100,density=True)
plt.xlabel('time diff(s)')
cnx.close()
"""
