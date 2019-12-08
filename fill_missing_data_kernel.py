import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# before kernel based method
arrival = pd.read_csv('arrival time matrix.csv', header=None)
arrival = arrival.fillna(0)
arrival = arrival.values
# rand = np.arange(1000,1300)
plt.figure()
sns.heatmap(arrival)
plt.xlabel('station')
plt.ylabel('round of bus')
plt.title('before data imputation')
plt.show()
M = arrival

def kernel_func(t1t2):
    if np.abs(t1t2) <= 30*60:
        return 1
    return 0
def matrix_fill(M, threshold = 30*60):
    M1 = M.copy()
    M2 = M.copy()
    # left
    M_diff = M[:, :-1] - M[:, 1:]
    M_zero = M <= 0
    M_nzero = ~M_zero
    M_valid = M_nzero[:, :-1] & M_nzero[:, 1:]
    M_left_valid = np.insert(M_valid, 0, False, axis=1)
    M_right_valid = np.insert(M_valid, M_valid.shape[1], False, axis=1)
    M_left = M_nzero[:, :-1] & M_zero[:, 1:]
    M_left = np.insert(M_left, 0, False, axis=1)
    M_right = M_zero[:, :-1] & M_nzero[:, 1:]
    M_right = np.insert(M_right, M_right.shape[1], False, axis=1)
    vkernel = np.vectorize(kernel_func)

    for col_ind in range(M.shape[1]):

        if col_ind >= 1:
            #print('left: station number:', col_ind)
            left_col = M[:, col_ind - 1]
            valid_diff = M_diff[M_left_valid[:, col_ind], col_ind - 1]
            left_support = left_col[M_left[:, col_ind]]
            left_other = left_col[M_left_valid[:, col_ind]]
            diff = np.outer(left_support, np.ones(left_other.shape[0])) - \
                   np.outer(np.ones(left_support.shape[0]), left_other)
            # here we only use threshold kernel
            K = 1. * (np.abs(diff) <=threshold)
            total_diff_left = np.dot(K, valid_diff)
            valid_total = total_diff_left != 0
            valid_estimate = np.dot(K[valid_total, :], valid_diff) / np.dot(K[valid_total, :],
                                                                            np.ones_like(valid_diff))
            total_diff_left[valid_total] = valid_estimate
            M1[M_left[:, col_ind], col_ind] = (M1[M_left[:, col_ind], col_ind - 1] - total_diff_left)\
                                              *(1.*(total_diff_left!=0))

        if col_ind < M.shape[1] - 1:
            #print('right: station number:', col_ind)
            right_col = M[:, col_ind + 1]
            valid_right_diff = M_diff[M_right_valid[:, col_ind], col_ind]
            right_support = right_col[M_right[:, col_ind]]
            right_other = right_col[M_right_valid[:, col_ind]]
            right_diff = np.outer(right_support, np.ones(right_other.shape[0])) - \
                         np.outer(np.ones(right_support.shape[0]), right_other)
            # here we only use threshold kernel
            K = 1. * (np.abs(right_diff) <= threshold)
            total_diff_right = np.dot(K, valid_right_diff)
            valid_right_total = total_diff_right != 0
            valid_right_estimate = np.dot(K[valid_right_total, :], valid_right_diff) / np.dot(K[valid_right_total, :],
                                                                                              np.ones_like(
                                                                                                  valid_right_diff))
            total_diff_right[valid_right_total] = valid_right_estimate
            M2[M_right[:, col_ind], col_ind] = (M2[M_right[:, col_ind], col_ind + 1] + total_diff_right)\
                                               *(1.*(total_diff_right!=0))
    new_arrival = (M1 + M2) / (1. * (M1 > 0) + 1. * (M2 > 0) + 1E-12)
    return new_arrival
missing_rate = []
for i in range(100):
    print(i,'-'*50)
    M = matrix_fill(M)
    missing_rate.append(1 - np.count_nonzero(M)/M.size)
    print('missing_rate',missing_rate[-1])
    if missing_rate[-1] == 0:
        break
plt.figure()
plt.plot(missing_rate)
plt.xlabel('iter')
plt.ylabel('missing rate')
plt.show()

plt.figure()
sns.heatmap(M)
plt.xlabel('station')
plt.ylabel('round of bus')
plt.title('after {0} round of data imputation'.format(i+1))
plt.show()

ss = pd.DataFrame(M)
ss.to_csv('kernel_filled_forward.csv',header=False,index=False)


# estimate random function
# for i in range(arrival.shape[1] - 1):
# for i in range(10,11):
#    time_diff = arrival[:, i+1] - arrival[:, i]
#    time_diff = time_diff.reshape(-1)
#    time_diff[abs(time_diff) > 1000] = 0
#    mask = time_diff <= 0
#    mask = ~mask
#    valid_record = pd.DataFrame({
#        'arrival':arrival[mask, i],
#        'time_diff':time_diff[mask]
#    })
#    valid_record.sort_values(by='arrival',inplace=True)

#    valid_record['arrival'] = valid_record['arrival'] % (3600 * 24)
# valid_record.plot(x='arrival',y='time_diff')
#    plt.scatter(valid_record['arrival'],valid_record['time_diff'])
#    plt.show()
