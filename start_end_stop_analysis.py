# -*- coding: utf-8 -*-
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from pylab import mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']

start_df = pd.read_csv('start_station_demand.csv',encoding='utf-8')
end_df = pd.read_csv('end_station_demand.csv',encoding='utf-8')
start_df.drop('Unnamed: 0',axis=1,inplace=True)
end_df.drop('Unnamed: 0',axis=1,inplace=True)

start_df = start_df.loc[start_df['start_name']!='Unknown']
end_df = end_df.loc[end_df['end_name']!='Unknown']

start_df['avg_people_per_bus'] = start_df['demand_count']/start_df['supply_bus']
end_df['avg_people_per_bus'] = end_df['demand_count']/end_df['supply_bus']
start_df['relative_occupancy'] = start_df['avg_people_per_bus']/start_df['avg_people_per_bus'].max()
end_df['relative_occupancy'] = end_df['avg_people_per_bus']/end_df['avg_people_per_bus'].max()
start_df.rename(columns={'start_name':'name'},inplace=True)
end_df.rename(columns={'end_name':'name'},inplace=True)

start_info = pd.pivot(start_df,index='name',columns='time0',values='demand_count')
end_info = pd.pivot(end_df,index='name',columns='time0',values='demand_count')
station_info = start_info.join(end_info, lsuffix='_start', rsuffix='_end')
station_info = station_info.fillna(0)

from sklearn.decomposition import PCA
import numpy as np
x = station_info.values
scale =np.amax(x, axis=1)
x = np.matmul(np.diag(1./scale),x)

from sklearn.cluster import KMeans
# create kmeans object
kmeans = KMeans(n_clusters=3)
# fit kmeans object to data
kmeans.fit(x)
station_info['label'] = kmeans.labels_
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
plt.figure()
colormap=['red','green','blue']
for i in range(3):
    plt.plot(kmeans.cluster_centers_[i][:24], label='group={0}'.format(i+1),color=colormap[i])
    plt.plot(-kmeans.cluster_centers_[i][24:], color=colormap[i])
plt.legend()
plt.xlabel('time (hour)')
plt.ylabel('demand pattern (+:departure -:arrival)')
plt.show()

busstation_shp = geopandas.read_file(
    'beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/02 区域内公交站点GIS信息/station.shp', encoding='gbk')
busstation_shp['STATION_ID'] = pd.to_numeric(busstation_shp['STATION_ID'])
station_loc_x = busstation_shp[['NAME','geometry']].groupby('NAME').agg(lambda geos: geos.unary_union.centroid.x)
station_loc_y = busstation_shp[['NAME','geometry']].groupby('NAME').agg(lambda geos: geos.unary_union.centroid.y)
station_loc = station_loc_x.join(station_loc_y, lsuffix='x',rsuffix='y')
station_loc.rename(columns={'geometryx':'x','geometryy':'y'},inplace=True)

plt.figure()
for ind in station_loc.index:
    if ind in station_info.index:
        label = station_info.loc[ind, 'label']
        plt.scatter(station_loc.loc[ind,'x'], station_loc.loc[ind,'y'],color=colormap[label])
        plt.text(x=station_loc.loc[ind,'x'],y=station_loc.loc[ind,'y'],s=ind)

start_df.sort_values('avg_people_per_bus',ascending=False,inplace=True)
end_df.sort_values('avg_people_per_bus',ascending=False,inplace=True)
start_df = start_df.join(station_info[['label']], on='name')
end_df = end_df.join(station_info[['label']],on='name')
'''
for ind in od_dfa.index:
    name1 = od_dfa.loc[ind,'start_name']
    name2 = od_dfa.loc[ind, 'end_name']
    avg_occ = od_dfa.loc[ind,'avg_people_per_bus']
    color_float = od_dfa.loc[ind,'relative_occupancy']
    x1 = station_loc.loc[name1,'x']
    x2 = station_loc.loc[name2,'x']
    y1 = station_loc.loc[name1,'y']
    y2 = station_loc.loc[name2,'y']
    if avg_occ>10:
        plt.arrow(x1,y1,(x2-x1),(y2-y1),width=avg_occ,alpha=color_float,length_includes_head=True)
'''