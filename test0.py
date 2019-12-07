# -*- coding: utf-8 -*-
import geopandas
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
info_list =['connector','link','node','zone_centroid']
xsh_shp = {name: geopandas.read_file('data_car/西三环_version 3/西三环___{0}.SHP'.format(name)) for name in info_list}
xsh_s2n = geopandas.read_file('data_car/data/西三环南向北/西三环南向北.SHP',encoding='gbk')

point_shp = geopandas.read_file('beijing_data/2014版北京路网_接点_六里桥区-2018/2014版北京路网_接点_六里桥区.shp', encoding='gbk')
link_shp = geopandas.read_file('beijing_data/2014版北京路网_接点_六里桥区-2018/新建文件夹/六里桥区域(1)/shp线/六里桥区域.shp', encoding='gbk')
busline_shp = geopandas.read_file(
    'beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/03-1 区域内线路GIS信息/line.shp', encoding='gbk')
busstation_shp = geopandas.read_file(
    'beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/02 区域内公交站点GIS信息/station.shp', encoding='gbk')
linestation = pd.read_csv('beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/附件 刷卡数据与GIS对应规则/站点静态表.txt',encoding='gbk')

region_buffer = xsh_s2n['geometry'].buffer(5)

xsh_buffered = xsh_s2n.copy()
xsh_buffered['geometry'] = region_buffer
s1 = geopandas.sjoin(xsh_buffered, busstation_shp)
station_interest_index = s1['index_right'].unique()

station_interest_data = busstation_shp.loc[station_interest_index]

# 1. here we use station id and linenum to match station and line
station_interest_data['STATION_ID'] = pd.to_numeric(station_interest_data['STATION_ID'],errors='coerce')
station_interest_data = station_interest_data.merge(linestation, on='STATION_ID')

# filter those line from south to north
def find_s2n(station_interest):
    for lid in station_interest['LINE_ID_x'].unique():
        temp = station_interest.loc[station_interest['LINE_ID_x']==lid]
        temp_min = temp.loc[temp['num'].idxmin()]
        temp_max = temp.loc[temp['num'].idxmax()]
        if temp_min['geometry'].y==temp_max['geometry'].y:
            # if there is only one station in this area, do not keep this data
            continue
        # if this bus goes from south to north and have more than 2 stations in this area,
        # keep this bus line and its stations
        if temp_max['direction'] == '上行' and (temp_max['geometry'].y>temp_min['geometry'].y):
            station_interest.loc[station_interest['LINE_ID_x']==lid,'interested'] = True
        elif temp_max['direction'] == '下行' and (temp_max['geometry'].y<temp_min['geometry'].y):
            station_interest.loc[station_interest['LINE_ID_x'] == lid, 'interested'] = True
    return station_interest
station_interest_data['interested']=False
station_interest_data = find_s2n(station_interest_data)
station_interest_data = station_interest_data.loc[station_interest_data['interested']==True]

count_line_station = station_interest_data['LINE_ID_x'].value_counts()
station_interest_data.set_index('LINE_ID_x',inplace=True)
busline_shp.set_index('LINE_ID',inplace=True)
for lid in count_line_station[count_line_station>1].index:
    if not(lid in busline_shp.index):
        station_interest_data = station_interest_data.drop(lid)
        print(lid, 'not found LINE_ID')

line_interest = busline_shp.join(count_line_station[count_line_station>1],how='inner')
line_interest.reset_index(inplace=True)
station_interest_data.reset_index(inplace=True)
station_interest_data['coord'] = station_interest_data['geometry'].apply(lambda x: x.representative_point().coords[:][0])

base = station_interest_data.plot()
xsh_s2n.plot(ax=base,color='red',linewidth=4)
line_interest.plot(ax=base,alpha=0.2)
for idx,row in station_interest_data.iterrows():
    plt.annotate(s=row['linenum'],xy=(row['coord'][0],row['coord'][1]))
# make direction and linenum to int
station_interest_data['num'] = station_interest_data['num'].astype(int)
new_station_interest_data = station_interest_data[['linenum','num','LINE_ID_y','NAME_y','direction','STATION_ID']].copy()
clean_nums ={
    'direction':{'上行':1,'下行':-1}
}
new_station_interest_data = new_station_interest_data.replace(clean_nums)

# initialize sql connector
print('getting sql data')
engine = create_engine('mysql+mysqlconnector://root:******@localhost/beijing_bus_liuliqiao', echo=False)
cnx =engine.raw_connection()

"""
#new_station_interest_data.to_sql(con=engine, name='station_interest_data',if_exists='replace')
#print('table insert to mysql')
read_grouped_start = '''
select line_id, direction, start_station, count(*) as count_record from ic_record
where direction!=0
and line_id in (57300,60631,55008,51300,43007,821,67,820,840,43002)
group by line_id, direction, start_station;
'''


read_grouped_end = '''
select line_id, direction, end_station, count(*) as count_record from ic_record
where direction!=0
and line_id in (57300,60631,55008,51300,43007,821,67,820,840,43002)
group by line_id, direction, end_station;
'''

print('start grouping')
start_grouped = pd.read_sql(sql=read_grouped_start,con=engine)
print('end grouping')
end_grouped = pd.read_sql(sql=read_grouped_end,con=engine)
start_grouped.to_csv('start_grouped.csv')
end_grouped.to_csv('end_grouped.csv')


count_start = new_station_interest_data.merge(start_grouped, left_on=['linenum','num','direction'],right_on=['line_id','start_station','direction'])
count_start.rename(columns={'count_record':'start_count'},inplace=True)
count_end = new_station_interest_data.merge(end_grouped, left_on=['linenum','num','direction'],right_on=['line_id','end_station','direction'])
count_end.rename(columns={'count_record':'end_count'},inplace=True)
count_start['end_count'] = count_end['end_count']
count_start.to_csv('count_start_end.csv')
"""




start_record=dict()
end_record=dict()
'''
k=0
for idx in station_interest_data.index:
    print(k)
    k+=1
    station_id = int(station_interest_data.loc[idx,'num'])
    line_id = int(station_interest_data.loc[idx,'linenum'])
    sql_select_start = """
                select trans_time, trans_date, start_station, start_time, end_station, bus_id
                from ic_record
                where
                (line_id ={0}1 or line_id = {0}0)
                and start_station ={1};
    """.format(line_id, station_id)
    sql_select_end = """
                select trans_time, trans_date, start_station, start_time, end_station, bus_id
                from ic_record
                where
                (line_id ={0}1 or line_id = {0}0)
                and end_station ={1};
    """.format(line_id, station_id)
    start_record[idx] = pd.read_sql(sql_select_start, cnx)
    end_record[idx] = pd.read_sql(sql_select_end, cnx)
cnx.close()
print('database connection closed')
'''


'''
import numpy as np
dx = np.array([[0,0]])
for i in xsh_s2n.index:
    print(i)
    g1 = xsh_s2n.loc[i,'geometry']
    lid =  xsh_s2n.loc[i,'LinkID']
    s =  xsh_shp['link'].loc[xsh_shp['link']['NO']==lid,'geometry']

    if s.shape[0]==0:
        print(i,': this link is missing')
        print('missing link ID:', lid)
        xsh_s2n.loc[[i]].plot(ax=base, color='red',linewidth=3)
        continue

    g2 = s.values[0]
    dx0 = np.array([[g1.xy[0][i] - g2.xy[0][i], g1.xy[1][i] - g2.xy[1][i]] for i in range(len(g1.xy[0]))])

    dx = np.concatenate((dx,dx0),axis=0)
    dx_mean = dx0.mean(axis=0)

    s = s.translate(xoff=dx_mean[0], yoff=dx_mean[1])
    s.plot(ax=base, color='green')
dx = np.delete(dx,0,0)
ddx = dx.copy()
dx = dx.mean(axis=0)
#for name in info_list:
#    xsh_shp[name]['geometry'] = xsh_shp[name]['geometry'].translate(xoff=dx[0],yoff=dx[1])
#for name in info_list:
#    xsh_shp[name].plot(ax=base,alpha=0.5)
'''




cnx.close()