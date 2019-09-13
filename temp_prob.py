import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def find_max_count(cnx):
    # first we select the line with most record
    sql_rank_trans_count = """
    select count(*) as count_line_record, line_id
    from ic_record
    group by line_id
    order by count_line_record DESC
    """
    line_count = pd.read_sql(sql_rank_trans_count, cnx)
    line_count['line_id'] = line_count['line_id'] // 10
    line_merged = line_count.groupby('line_id').sum()
    line_merged.sort_values('count_line_record', ascending=False, inplace=True)
    print('the top 5 line is:')
    print(line_merged.index[:5])
    # after this we found that top line is (1,57,52,890,52300,...)
    return line_merged


def plot_record(cnx, line_id=57):
    # we can plot the arrival and departure record
    sql_select2 = """
        select trans_time, trans_date, start_station, end_station, bus_id
        from ic_record
        where (line_id = {0}0 or line_id = {0}1)
        and trans_date = 20180601
        """.format(line_id)
    line57_record = pd.read_sql(sql_select2, cnx)
    line57_record['trans_time'] = (line57_record['trans_date'] - 20180601) * 24 * 3600 + line57_record[
        'trans_time'] // 10000 * 3600 + (line57_record['trans_time'] % 10000) // 100 * 60 + line57_record[
                                      'trans_time'] % 100
    le = LabelEncoder()
    line57_record['bus_unique'] = le.fit_transform(line57_record['bus_id'])
    for j in le.classes_[:5]:
        line57_onebus = line57_record.loc[line57_record['bus_id'] == j]
        line57_onebus = line57_onebus.sort_values('trans_time')
        plt.scatter(line57_onebus['trans_time'], line57_onebus['end_station'], alpha=0.1, label=i)
        plt.plot(line57_onebus['trans_time'], line57_onebus['end_station'], alpha=0.1)
    plt.legend()
    plt.show()
    return