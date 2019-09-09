import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
cnx = mysql.connector.connect(user='root', password='a2=b2=c2', database='beijing_bus_liuliqiao')

# dd = pd.read_sql('select * from ic_record limit 100',cnx)
sql_select2 = """
select trans_time, start_time, start_station, end_station
from ic_record
where bus_id = 22055
and trans_date = 20180601
and start_time > 80000
and start_time < 230000
"""
bus_22055 = pd.read_sql(sql_select2, cnx)
bus_22055['start_time'] = bus_22055['start_time']//10000*3600 + (bus_22055['start_time']%10000)//100*60 + bus_22055['start_time']%100
bus_22055['trans_time'] = bus_22055['trans_time']//10000*3600 + (bus_22055['trans_time']%10000)//100*60 + bus_22055['trans_time']%100
plt.scatter(bus_22055['start_time'],bus_22055['start_station'],label='boarding')
plt.scatter(bus_22055['trans_time'],bus_22055['end_station'],marker='+', label = 'alighting')
for idx in bus_22055.index:
    plt.plot([bus_22055.loc[idx,'start_time'], bus_22055.loc[idx,'trans_time']],
             [bus_22055.loc[idx,'start_station'], bus_22055.loc[idx,'end_station']],color='black',alpha=0.1)
plt.legend()
plt.show()

cnx.close()
