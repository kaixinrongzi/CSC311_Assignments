import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def l_one_distance(pt1, pt2):
    n = len(pt1)
    sum = 0
    for i in range(n):
        sum += np.abs(pt1[i] - pt2[i])
    return sum

def l_two_distance(pt1, pt2):
    n = len(pt1)
    sum = 0
    for i in range(n):
        sum += np.square(pt1[i] - pt2[i])
    return sum

def shit():
    print("kakaka")



# generate d
d = []
for i in range(11):
    d.append(2**i)
    
l1_distance_mean_lst = []
l1_distance_std_lst = []
l2_distance_mean_lst = []
l2_distance_std_lst = []
for i in range(11):
    dimension = d[i]
    # print("dimension=", dimension)
    lst = []
    for j in range(100):
        pt = []
        for k in range(dimension):
            pt.append(random.rand())
        lst.append(pt)
    # calculate distance
    l1_distance = []
    l2_distance = []
    sum2 = 0
    for pt1 in lst:
        for pt2 in lst:
            if pt1 == pt2:
                continue
            l1_distance.append(l_one_distance(pt1, pt2))
            l2_distance.append(l_two_distance(pt1, pt2))
    l1_distance_mean = np.mean(l1_distance)
    l1_distance_std = np.std(l1_distance)
    l2_distance_mean = np.mean(l2_distance)
    l2_distance_std = np.std(l2_distance)
    
    l1_distance_mean_lst.append(l1_distance_mean)
    l1_distance_std_lst.append(l1_distance_std)
    l2_distance_mean_lst.append(l2_distance_mean)
    l2_distance_std_lst.append(l2_distance_std)
    
    # print("d = ", dimension, " mean of l1_distance is: ", np.mean(l1_distance),"\n")
    # print("d = ", dimension, " std of l1_distance is: ", np.std(l1_distance),"\n")
    # print("d = ", dimension, " mean of l2_distance is: ", np.mean(l2_distance),"\n")
    # print("d = ", dimension, " std of l2_distance is: ", np.std(l2_distance),"\n")
            

# plt.plot(d, l1_distance_mean_lst, label = 'l1_distance_mean vs. d')
# plt.legend()
# plt.xlabel('d')
# plt.ylabel('statistics')
# plt.title("function l1_distance_mean of d")
# plt.show()

# plt.plot(d, l1_distance_std_lst, label = 'l1_distance_std vs. d')
# plt.legend()
# plt.xlabel('d')
# plt.ylabel('statistics')
# plt.title("function l1_distance_std of d")
# plt.show()

# plt.plot(d, l2_distance_mean_lst, label = 'l2_distance_mean vs. d')
# plt.legend()
# plt.xlabel('d')
# plt.ylabel('statistics')
# plt.title("function l2_distance_mean of d")
# plt.show()

plt.plot(d, l2_distance_std_lst, label = 'l2_distance_std vs. d')
plt.legend()
plt.xlabel('d')
plt.ylabel('statistics')
plt.title("function l2_distance_std of d")
plt.show()



