# -*- coding: utf-8 -*-
import geopandas

point_shp = geopandas.read_file('beijing_data/2014版北京路网_接点_六里桥区-2018/2014版北京路网_接点_六里桥区.shp', encoding='gbk')
link_shp = geopandas.read_file('beijing_data/2014版北京路网_接点_六里桥区-2018/新建文件夹/六里桥区域(1)/shp线/六里桥区域.shp', encoding='gbk')
busline_shp = geopandas.read_file(
    'beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/03-1 区域内线路GIS信息/line.shp', encoding='gbk')
busstation_shp = geopandas.read_file(
    'beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/02 区域内公交站点GIS信息/station.shp', encoding='gbk')
base = busstation_shp.plot()

bus57_shp = busline_shp.loc[busline_shp['线路号'].astype('str').str[:5]=='00057']
bus57_shp.plot(ax=base)
link_shp.plot(ax=base, color='red',alpha=0.1)