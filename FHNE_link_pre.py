# -*- coding: utf-8 -*-
import networkx as nx
from multiprocessing import Pool
import csv
import numpy as np
import math
import gc
import os
import datetime
import math
import random
import time
import copy
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from tqdm import tqdm
def k_core_mgnrl(graphfile,k_layer,p,para_a,para_b):
    G=nx.read_edgelist(graphfile,delimiter=' ') #生成网络
    source_g=copy.deepcopy(G)  #复制网络
    all_nodes=list(nx.nodes(G))  #所有节点
    k_inter=int(len(all_nodes)/k_layer)
    k_core_dict=nx.core_number(G)
    sort_k_core=sorted(k_core_dict.items(),key=lambda x:x[1])
    k_value_list=[1]
    # for index,i in enumerate(sort_k_core):
    #     if index==k_inter:
    #         k_value_list.append(i[1])
    #         k_inter+=int(len(all_nodes)/k_layer)
    #     if len(k_value_list)==k_layer:
    #         break
    max_k_core=sort_k_core[-1][1]
    gap=int(max_k_core/k_layer)
    num=1
    l=1
    while num<max_k_core and l<k_layer:
        num+=gap
        l+=1
        k_value_list.append(num)

    print('k_value_list:',k_value_list)
    #计算增益：每个节点i邻居间的边数
    nl_dict=neighbor_node_links(G)
    # print(nl_dict)
    #计算每个节点是否构成环形：
    loop_nodes=[node for node in nx.k_core(G,2).nodes if node not in nl_dict]
    # print(loop_nodes)

    multi_simi=[]  #相似性
    layer_nodes=[]  #每一层的节点
    G_list=[]   #图层列表
    num=0
    for index,k in enumerate(k_value_list):
        if k==1:
            G1=G
        else:
            G1=fuzzy_k_core(source_g,G,k,2*k-1,nl_dict,loop_nodes)  #模糊k-core划分网络   2k表示公式中的b
        G2 = copy.deepcopy(G1)  # 深度复制
        G_list.append(G2)
    print('k_layer:',[len(x.nodes) for x in G_list])
    #G_list=G_list1[::-1]  #图层求逆，最上层的
    k=len(G_list)
    for index,G in enumerate(G_list):
        nodes,dict_simi=Jump_matrix(G,index,para_a,para_b) #每一粒度层的节点，节点相似性
        multi_simi.append(dict_simi)  #
        layer_nodes.append(nodes)
    multi_simi1=[]
    for index,jump_matrix in enumerate(multi_simi):
        if index==0:
            for node in jump_matrix.keys():
                if node in multi_simi[index+1]:
                    jump_matrix[node]['jump_up']=multi_simi[index+1][node]['aver']
                    jump_matrix[node]['sum'] +=jump_matrix[node]['jump_up']
        elif index==len(multi_simi)-1:
            for node in jump_matrix.keys():
                if node in multi_simi[index-1]:
                    jump_matrix[node]['jump_down']=multi_simi[index-1][node]['aver']
                    jump_matrix[node]['sum'] += jump_matrix[node]['jump_down']
        else:
            for node in jump_matrix.keys():
                if node in multi_simi[index-1].keys():
                    if node in multi_simi[index - 1]:
                        jump_matrix[node]['jump_down']=multi_simi[index-1][node]['aver']
                        jump_matrix[node]['sum'] += jump_matrix[node]['jump_down']
                if node in multi_simi[index+1].keys():
                    if node in multi_simi[index + 1]:
                        jump_matrix[node]['jump_up']=multi_simi[index+1][node]['aver']
                        jump_matrix[node]['sum'] += jump_matrix[node]['jump_up']
        multi_simi1.append(jump_matrix)
    return  all_nodes,multi_simi1,layer_nodes,k
def neighbor_node_links(G):
    nl_dict={}
    for node in G.nodes:
        if nx.degree(G,node)>1:
            nl_dict[node]=0
            i_neighbors = list(nx.neighbors(G, node))
            for index,neightbor_node_1 in enumerate(i_neighbors[:-1]):
                for neightbor_node_2 in i_neighbors[index+1:]:
                    if (neightbor_node_1,neightbor_node_2) in nx.edges(G):
                        nl_dict[node]+=1
    return nl_dict
def fuzzy_k_core(source_G,G,k,b,nl_dict,loop_nodes):
    all_nodes=list(G.nodes)
    for node in all_nodes:
        d_i=source_G.degree(node)
        if d_i<k and node in G.nodes:
            G.remove_node(node)
        else:
            if d_i>=b:
                membership=1
            else:
                #增益1
                # p=0
                # if node in loop_nodes:
                #     p=1
                # if node in nl_dict:
                #     p=float(nl_dict[node])/(d_i)
                # membership=float(d_i-k+1+p)/(b-k)
                membership = float(d_i - k + 1 ) / (b - k)
            if membership<0.9:
                G.remove_node(node)
    return G

#层内跳转和层间跳转
def Jump_matrix(G,k,para_a,para_b):
    # 重要相似性矩阵
    nodes=list(G.nodes)
    nodes_degree=list(nx.degree(G))  #所有节点的度数
    dic_simi={}
    for index,i in enumerate(nodes_degree):
        num=0
        sum_simi=0
        dic_simi[i[0]]={}
        max_simi=0
        min_simi=0
        for j in nodes_degree:
            if i==j:
                continue
            edge=(i[0],j[0])
            # struc_simi=math.exp((min(i[1],j[1])-max(i[1],j[1]))/max(i[1],j[1]))  #结构相似性
            struc_simi=min(i[1]+1,j[1]+1)/max(i[1]+1,j[1]+1)   #结构相似性
            # if struc_simi<float(k)/(k+1):
            #     struc_simi=float(k)/(k+1)
            if edge in nx.edges(G):
                link_simi=1.0   #邻接矩阵，链接相似性
            else:
                link_simi=0
            simi=round(struc_simi*para_a+link_simi*para_b,3) #综合相似性
            if simi!=0:
                num+=1
            if max_simi<simi:
                max_simi=simi
            if min_simi>simi:
                min_simi=simi
            dic_simi[i[0]][j[0]]=simi
            sum_simi+=simi
        aver_simi=sum_simi/num
        dic_simi[i[0]]['aver']=aver_simi  #跳转平均概率
        dic_simi[i[0]]['max']=max_simi
        dic_simi[i[0]]['min'] = min_simi
        dic_simi[i[0]]['sum']=sum_simi  #跳转概率之和
    return nodes,dic_simi
def single_random(k_th,multi_simi,layer_nodes,random_times,walk_len,randomfile):
    all_random=[]
    random_w = csv.writer(open(randomfile, 'a', encoding='utf-8', newline=''), delimiter=' ')
    all_node_seq=locals()  #所有节点列表
    name_th=copy.deepcopy(k_th)  #第k层
    copy_simi=copy.deepcopy(multi_simi)
    use_nodes=layer_nodes[name_th]

    # if name_th<len(multi_simi)-1:
    #     use_nodes=[node for node in layer_nodes[name_th] if node not in layer_nodes[name_th+1]]
    # else:
    #     use_nodes=layer_nodes[name_th]
    for i in tqdm(use_nodes):  #遍历当前层的所有节点
        for j in range(random_times): #迭代次数
            k_th=name_th  #当前层
            node_seq=[]  #一次的游走序列
            seed=i
            node_seq.append(seed) #加入种子节点
            iter_length=int(walk_len*math.exp(-j/random_times))
            while True:
                next_node=node_choice1(node_seq,multi_simi[k_th])
                while True:
                    if next_node=='jump_down':  #节点要跳转到下一层
                        k_th=k_th-1  #层数减1
                        next_node=node_choice1(node_seq,multi_simi[k_th])
                    elif next_node=='jump_up': #节点跳转到上一层
                        k_th=k_th+1  #层数加一
                        next_node=node_choice1(node_seq,multi_simi[k_th])
                    else:  #后继节点选择成功
                        break
                t=random.uniform(0,multi_simi[k_th][node_seq[-1]]['max'])
                if multi_simi[k_th][node_seq[-1]][next_node]<t:
                    multi_simi[k_th][node_seq[-1]]['sum']-=multi_simi[k_th][node_seq[-1]][next_node]/2
                    multi_simi[k_th][node_seq[-1]][next_node]-=multi_simi[k_th][node_seq[-1]][next_node]/2
                else:
                    node_seq.append(next_node)
                if len(node_seq)==iter_length:
                    break
            all_random.append(node_seq)
    random_w.writerows(all_random)
def node_choice1(node_seq,layer_simi):
    node=node_seq[-1]  #当前节点
    simi_sum=layer_simi[node]['sum']
    t = random.uniform(0,simi_sum)
    s=0
    for simi_node,simi in layer_simi[node].items():
        if simi_node!='aver' and simi_node!='sum' and simi_node!='max'and simi_node!='min':
            s+=simi
            if s>t:
                return simi_node
def skip_gram(randomfile,all_nodes,size,window,workers,savetrain):
    walks = LineSentence(randomfile)
    model = Word2Vec(walks, sg=1,hs=1, size=size,window=window,workers=workers)
    #model.save('train.model')
    csv_writer=csv.writer(open(savetrain,'w',encoding='utf-8',newline=''),delimiter=' ')
    all_vecs=[]
    print(len(all_nodes))
    for i in all_nodes:
        train=[]
        train.append(int(i))
        say_vector = model[str(i)]  # get vector for word
        for j in list(map(float,say_vector)):
            train.append(round(j,7))
        all_vecs.append(train)
    csv_writer.writerows(all_vecs)

if __name__=="__main__":
    dir1, dir2, dir3 = '../graph/', '../random/', '../../对比实验/source_emb/'
    # edgefile="karate-mirrored.edgelist"
    # randomfile='random.csv'
    # trainsave='karate_fhne.emb'
    # edgefile="blog_edges.csv"
    # randomfile='random.csv'
    # trainsave='blog_fhne.emb'
    # edgefile = "europe-flights.edgelist"
    # randomfile = 'random.csv'
    # trainsave = 'europe_fhne.emb'
    # edgefile = "brazil-flights.edgelist"
    # randomfile = 'random.csv'
    # trainsave = 'brazil_fhne.emb'
    # edgefile = "blog_edges.csv"
    # randomfile = 'random.csv'
    # trainsave = 'blog_fhne.emb'
    edgefile = "ppi_edges.txt"
    randomfile = 'random.csv'
    trainsave = 'ppi_fhne.emb'
    layer=4
    p=1
    all_nodes,multi_simi,layer_nodes,k=k_core_mgnrl(dir1+edgefile,layer,p,0,1)  #标准
    print([len(x) for x in layer_nodes])
    print('开启游走')
    if os.access(dir2+randomfile,os.F_OK):
        os.remove(dir2+randomfile)
    start = datetime.datetime.now()
    p = Pool(layer)
    for i in range(layer):
        p.apply_async(single_random, args=(i, multi_simi, layer_nodes, 10, 120, dir2+randomfile))
    p.close()
    p.join()
    # for i in range(4):
    #     single_random(i, multi_simi, layer_nodes,  10, 100, dir2+randomfile)
    # single_random(1, multi_simi, layer_nodes,  10, 100, dir2+randomfile)
    # single_random(layer-1, multi_simi, layer_nodes,  10, 100, dir2+randomfile)
    # time.sleep(5)
    # # skip-gram训练
    skip_gram(dir2+randomfile, all_nodes, 128, 10, 4, dir3+trainsave)
    # os.remove(dir2+randomfile)