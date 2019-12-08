import pandas as pd
count_start_end = pd.read_csv('count_start_end.csv', encoding='utf-8')
linestation = pd.read_csv('beijing_data/2014版北京路网_接点_六里桥区-2018/北京局部区域公交相关数据/北京局部区域公交相关数据/附件 刷卡数据与GIS对应规则/站点静态表.txt',encoding='gbk')
ss = count_start_end.merge(linestation.set_index('STATION_ID'), on='STATION_ID')
ss.to_csv('requested_data.csv',encoding='utf-8')