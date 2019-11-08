# -*- coding: utf-8 -*-
import geopandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

region_buffer = xsh_s2n['geometry'].buffer(50)

xsh_buffered = xsh_s2n.copy()
xsh_buffered['geometry'] = region_buffer
s1 = geopandas.sjoin(xsh_buffered, busstation_shp)
station_interest_index = s1['index_right'].unique()

station_interest_data = busstation_shp.loc[station_interest_index]

# 1. here we use station id and linenum to match station and line
station_interest_data['STATION_ID'] = pd.to_numeric(station_interest_data['STATION_ID'],errors='coerce')
station_interest_data = station_interest_data.merge(linestation, on='STATION_ID')
#station_interest_data['LINE_ID_x']=pd.to_numeric(station_interest_data['LINE_ID_x'])
#assert sum(station_interest_data['LINE_ID_x']!=station_interest_data['LINE_ID_y'])==0
# after checking if LINE_ID is the same, we find lines that have >=2 stations in our area
count_line_station = station_interest_data['LINE_ID_x'].value_counts()
station_interest_data.set_index('LINE_ID_x',inplace=True)
station_interest_data = station_interest_data.loc[count_line_station[count_line_station>1].index]

busline_shp.set_index('LINE_ID',inplace=True)
for lid in count_line_station[count_line_station>1].index:
    if not(lid in busline_shp.index):
        station_interest_data = station_interest_data.drop(lid)
        print(lid, 'not found LINE_ID')
line_interest = busline_shp.join(count_line_station[count_line_station>1],how='inner')
line_interest.reset_index(inplace=True)
station_interest_data.reset_index(inplace=True)

base = station_interest_data.plot()
xsh_s2n.plot(ax=base,color='red',linewidth=4)
line_interest.plot(ax=base,alpha=0.2)

#station_interest_data['LINE_ID'].value_counts()
#busstation_shp.plot(ax=base,color='green')
#busline_shp.plot(ax=base, alpha = 0.1,color='red')


'''
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
    #s.plot(ax=base, color='green')
dx = np.delete(dx,0,0)
ddx = dx.copy()
dx = dx.mean(axis=0)
#for name in info_list:
#    xsh_shp[name]['geometry'] = xsh_shp[name]['geometry'].translate(xoff=dx[0],yoff=dx[1])
#for name in info_list:
#    xsh_shp[name].plot(ax=base,alpha=0.5)
'''


