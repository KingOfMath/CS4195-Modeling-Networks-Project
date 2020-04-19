
import pandas as pd
import collections
import numpy as npy
import networkx as nx
import matplotlib.pyplot as plt

#load excel files
data = pd.read_excel(r'C:\Users\花儿对我笑笑笑\Desktop\modeling final\workspace\test 2.xlsx')
#select column and put into numpy arrays
User = data.user.to_numpy()
Movie = data.movie.to_numpy()
Date = data.date.to_numpy()
Rating = data.rating.to_numpy()
#numbers in the sheet is at +2 row (Node_1[10] is at row 12)

def simplify(user,movie,rating,date):
    rate_num = 0
    for i in range(rating.size):
        if(rating[i]>3):
            rate_num += 1
    
    user_s = npy.zeros(rate_num)
    movie_s = npy.zeros(rate_num)
    date_s = npy.zeros(rate_num)
    rating_s = npy.zeros(rate_num)

    non_zero_count = 0
    for i in range(rating.size):
        if(rating[i]>3):
            user_s[non_zero_count] = user[i]
            movie_s[non_zero_count] = movie[i]
            date_s[non_zero_count] = date[i]
            rating_s[non_zero_count] = rating[i]
            non_zero_count += 1

    return non_zero_count, user_s, movie_s, date_s, rating_s

def single_list(arr, target):
    I = arr.tolist()

    return I.count(target)



(non_zero_count, user_s, movie_s, date_s, rating_s) = simplify(User,Movie,Rating,Date)

print(non_zero_count)
print(user_s)
size = movie_s.size
graph_size = max(movie_s)
print(size)
movie_nodes = npy.arange(1,graph_size+1,1)
G = nx.Graph()
G.add_nodes_from(movie_nodes)
print(graph_size)

"""

user_seq = 0
link_weight = npy.zeros([1682,1682])
user_sum = 0

len_xlsx_need = 0

for i in range(size-1):

    if (i < size):
        if (user_seq != user_s[i]):
#            I = npy.zeros(single_list(user_s,user_s[i]))
            user_seq = user_s[i]
            j = 0
#            I[j] = movie_s[i]
            j += 1
        elif (user_seq == user_s[i]) and (user_s[i] == user_s[i+1]):
#            I[j] = movie_s[i]
            j += 1
        elif (user_seq == user_s[i]) and (user_s[i] != user_s[i+1]):
#            I[j] = movie_s[i]
            for x in range(user_sum,user_sum + single_list(user_s,user_s[i])):
                for y in range(user_sum,user_sum + single_list(user_s,user_s[i])):
                    m = int(movie_s[x]) - 1
                    n = int(movie_s[y]) - 1
                    if(m != n):
#                        G.add_edge(m, n, weight = link_weight[m][n]+1)
                        link_weight[m][n] += 1
                        len_xlsx_need += 1
            user_sum += single_list(user_s,user_s[i])
            print(user_sum)
    elif (i == size):
            for x in range(user_sum,user_sum + single_list(user_s,user_s[i])):
                for y in range(user_sum,user_sum + single_list(user_s,user_s[i])):
                    m = int(movie_s[x]) - 1
                    n = int(movie_s[y]) - 1
                    if(m != n):
#                        G.add_edge(m, n, weight = link_weight[m][n]+1)
                        link_weight[m][n] += 1
                        len_xlsx_need += 1


print(link_weight)


data = pd.DataFrame(link_weight)

writer = pd.ExcelWriter('3.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()

writer.close()


#D_non_count = 0

#for i in range(0,1681):
#    if G.degree(i)>0:
#        D_non_count+=1

#print(D_non_count)


weight_threshold = npy.percentile(link_weight, 40)

print(weight_threshold)

#print(len_xlsx_need)

for i in range(1682):
    for j in range(1682):
        if (link_weight[i-1][j-1] >= 4):
#            link_weight[i-1][j-1] = 1
            G.add_edge(i, j)
        else:
            link_weight[i-1][j-1] = 0
            pass


degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color="b")

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

# draw graph in inset
plt.axes([0.4, 0.4, 0.5, 0.5])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(G)
plt.axis("off")
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, alpha=0.4)
plt.show()

#nx.draw(G)
#plt.show()

"""

#infection

def infection(nodes,timestamp,seed):
    seed_time = 0
    Infe_i = npy.zeros(1682)
    Infe_count = npy.zeros(1682)
    
    for i in range(0,timestamp.size-1):
        if (seed == nodes[i]):
            seed_time = timestamp[i]
            break
    if (seed_time == 0):
        return Infe_i
    else:
        for j in range(0,timestamp.size-1):
            if (timestamp[j]>=seed_time):
                if (Infe_count[int(nodes[j])] == 0):
                    Infe_i[int(nodes[j])] = timestamp[j]-seed_time+1
                    Infe_count[int(nodes[j])]=1
        return Infe_i

    





timedelay = npy.zeros([1682,1682])

for i in range(1,1682):

    T = infection(movie_s,user_s,i)
    for j in range(1,1682):
        timedelay[i-1][j-1] = T[j-1]
    print(i)

data = pd.DataFrame(timedelay)

writer = pd.ExcelWriter('timedelay2.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()

writer.close()
