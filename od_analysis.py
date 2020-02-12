# -*- coding: utf-8 -*-
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

od_df = pd.read_csv('od_demand_everyday.csv',encoding='utf-8')
od_df.drop('Unnamed: 0',axis=1,inplace=True)
od_df.drop('date',axis=1,inplace=True)
od_df = od_df.groupby(['start_name','end_name','time0']).sum()
od_df.reset_index(inplace=True)


od_df = od_df.loc[(od_df['start_name']!='Unknown') &(od_df['end_name']!='Unknown')]
od_df = od_df.loc[od_df['start_name']!=od_df['end_name']]

od_df['avg_people_per_bus'] = od_df['demand_count']/od_df['supply_bus']
#od_df.hist('avg_people_per_bus',bins=100)
od_df = od_df.loc[od_df['demand_count']>10]
od_df['relative_occupancy'] = od_df['avg_people_per_bus']/od_df['avg_people_per_bus'].max()
od_dfa = od_df.loc[od_df['time0']==8]

busstation_shp = geopandas.read_file(
    'beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/02 区域内公交站点GIS信息/station.shp', encoding='gbk')
busstation_shp['STATION_ID'] = pd.to_numeric(busstation_shp['STATION_ID'])
station_loc_x = busstation_shp[['NAME','geometry']].groupby('NAME').agg(lambda geos: geos.unary_union.centroid.x)
station_loc_y = busstation_shp[['NAME','geometry']].groupby('NAME').agg(lambda geos: geos.unary_union.centroid.y)
station_loc = station_loc_x.join(station_loc_y, lsuffix='x',rsuffix='y')
station_loc.rename(columns={'geometryx':'x','geometryy':'y'},inplace=True)

for ind in station_loc.index:
    plt.scatter(station_loc.loc[ind,'x'], station_loc.loc[ind,'y'],color='blue')
    plt.text(x=station_loc.loc[ind,'x'],y=station_loc.loc[ind,'y'],s=ind)

for ind in od_dfa.index:
    name1 = od_dfa.loc[ind,'start_name']
    name2 = od_dfa.loc[ind, 'end_name']
    avg_occ = od_dfa.loc[ind,'avg_people_per_bus']
    color_float = od_dfa.loc[ind,'relative_occupancy']
    x1 = station_loc.loc[name1,'x']
    x2 = station_loc.loc[name2,'x']

    y1 = station_loc.loc[name1,'y']
    y2 = station_loc.loc[name2,'y']
    if avg_occ>5:
        plt.arrow(x1,y1,(x2-x1),(y2-y1),width=avg_occ*4,alpha=0.6,length_includes_head=True)
