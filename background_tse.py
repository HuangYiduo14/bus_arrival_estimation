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

from statsmodels.nonparametric.smoothers_lowess import lowess
def one_link_analysis(record, linkid=35972,day=1):
    speed_link = record.loc[record['linkid']==linkid]
    speed_link = speed_link.loc[speed_link['time'].dt.day==day]
    x = speed_link['tm'].values
    y = speed_link['speed']
    new = lowess(y,x,frac=0.15)
    return new

result_diff = pd.read_csv('result_公主坟南_六里桥北里_diff_2.csv')
result_diff.drop('Unnamed: 0',axis=1,inplace=True)
result_diff.sort_values('max_pivot',inplace=True)
result_diff['tm']=result_diff['max_pivot']%1000000
result_diff['tm'] = (result_diff['tm']//100)%100*60 + (result_diff['tm']//10000)*60*60+result_diff['tm']%100
result_diff['max_pivot'] = pd.to_datetime(result_diff['max_pivot'].astype(str))
d_length_last = xsh_s2n1.iloc[-1,-2]-maxy
d_length_first = miny -  xsh_s2n1.iloc[0,-1]

new_result = pd.DataFrame(columns=['diff_max','count','max_pivot','board_count','alight_count','line_id','tm']+['speed_{0}'.format(linkid) for linkid in xsh_s2n1['LinkID'].values])
for day in range(1,30):
    print(day,'-'*50)
    result_day = result_diff.loc[result_diff['max_pivot'].dt.day==day]
    for linkid in xsh_s2n1['LinkID'].values:
        print(linkid)
        low_result = pd.DataFrame(one_link_analysis(speed_record,linkid,day),columns=['tm','speed'])
        low_result['tm'] = low_result['tm'].astype('int64')
        result_day = pd.merge_asof(
            result_day, low_result, on='tm'
        )
        result_day.rename(columns={'speed':'speed_{0}'.format(linkid)},inplace=True)
    new_result = pd.concat([new_result,result_day])

new_result['free_time'] = 0
for ind in xsh_s2n1.index:
    linkid = xsh_s2n1.loc[ind,'LinkID']
    if xsh_s2n1.loc[ind,'min_y']<miny:
        len_link = xsh_s2n1.loc[ind,'LENGTH'] - d_length_first
    elif xsh_s2n1.loc[ind,'max_y']>maxy:
        len_link = xsh_s2n1.loc[ind, 'LENGTH'] - d_length_last
    else:
        len_link = xsh_s2n1.loc[ind, 'LENGTH']
    # here 36 is the ratio from km/h to m/s
    new_result['free_time'] = new_result['free_time'] + 3.6*len_link/new_result['speed_{0}'.format(linkid)]
new_result['diff_time'] = new_result['diff_max']-new_result['free_time']-new_result['board_count']*2.-new_result['alight_count']*1.
new_result = new_result.loc[new_result['diff_max']>=60]

plt.figure()
new_result['diff_max'].hist(bins=100,density=True)
plt.xlabel('travel time(s)')
#plt.title('distribution of queue delay')