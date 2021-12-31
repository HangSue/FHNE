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
def k_core_mgnrl(graphfile,k_layer,para_a,para_b):
    G=nx.read_edgelist(graphfile,delimiter=' ') #生成网络
    source_g=copy.deepcopy(G)  #复制网络
    all_nodes=list(nx.nodes(G))  #所有节点
    k_inter=int(len(all_nodes)/k_layer)
    k_core_dict=nx.core_number(G)
    sort_k_core=sorted(k_core_dict.items(),key=lambda x:x[1])
    k_value_list=[1]
    for index,i in enumerate(sort_k_core):
        if index==k_inter:
            k_value_list.append(i[1])
            k_inter+=int(len(all_nodes)/k_layer)
        if len(k_value_list)==k_layer:
            break
    print(k_value_list)
    multi_simi=[]  #相似性
    layer_nodes=[]  #每一层的节点
    G_list=[]   #图层列表
    num=0
    for index,i in enumerate(k_value_list):
        if i==1:
            G1=G
        else:
            G1=fuzzy_k_core(source_g,G,i,k_value_list[index]-k_value_list[index-1])  #模糊k-core划分网络
        # G1=nx.k_core(G,i)  #标准k-core划分
        G2=copy.deepcopy(G1)  #深度复制
        G_list.append(G2)
    k=len(G_list)  #图层数
    G_list=G_list[::-1]  #图层求逆，最上层的
    for index,G in enumerate(G_list):
        nodes,dict_simi=Jump_matrix(G,G_list,index,para_a,para_b) #每一粒度层的节点，节点相似性
        multi_simi.append(dict_simi)  #
        layer_nodes.append(nodes)
    return  all_nodes,multi_simi,layer_nodes,k
def fuzzy_k_core(source_G,G,d,k_num):
    all_nodes=list(source_G.nodes)
    remain_nodes=[]
    cycles=1
    while True:
        num_delete=0
        if cycles==1:
            for i in all_nodes:
                if source_G.degree(i)<d and i in G.nodes:
                    G.remove_node(i)
                    num_delete+=1
        else:
            for i in all_nodes:
                if i not in G.nodes:
                    continue
                if i in remain_nodes:
                    continue
                if G.degree(i)<d:  #如果有小于k度数的节点，继续用隶属度函数判断。
                    membership=0
                    differ_num=source_G.degree(i)-G.degree(i)
                    if differ_num>=k_num:
                        membership=1
                    if differ_num<d:
                        membership=math.pow(float(differ_num)/k_num,1)
                        # membership=differ_num/k_num
                    if membership>=0.5:
                        remain_nodes.append(i)
                    else:
                        G.remove_node(i)
                        num_delete+=1
        cycles+=1
        # if cycles==4:
        #     break
        if num_delete==0:
            break
    # print(d)
    # print(nx.degree(G))
    return G

#层内跳转和层间跳转
def Jump_matrix(G,all_G,num,para_a,para_b):
    # 重要相似性矩阵
    nodes=list(G.nodes)
    if num==0:  #第一层，只往下一层跳转
        up_nodes=[]
        down_nodes=nodes
    elif num==len(all_G)-1: #最后一层，只往上一层跳转
        down_nodes=[]
        up_nodes=list(all_G[num-1].nodes)
    else:  #中间层，上下跳转
        up_nodes=list(all_G[num-1].nodes)
        down_nodes=nodes
    nodes_degree=list(nx.degree(G))  #所有节点的度数
    dic_simi={}
    for index,i in enumerate(nodes_degree):
        num=0
        sum_simi=0
        dic_simi[i[0]]={}
        max_simi=0
        for j in nodes_degree:
            if i==j:
                continue
            edge=(i[0],j[0])
            # struc_simi=math.exp((min(i[1],j[1])-max(i[1],j[1]))/max(i[1],j[1]))  #结构相似性
            struc_simi=min(i[1],j[1])/max(i[1],j[1])   #结构相似性
            if edge in nx.edges(G):
                link_simi=1   #邻接矩阵，链接相似性
            else:
                link_simi=0
            simi=round(struc_simi*para_a+link_simi*para_b,4) #综合相似性
            if simi>0:
                num+=1
            if simi>max_simi:
                max_simi=simi
            dic_simi[i[0]][j[0]]=simi
            sum_simi+=simi
        aver_simi=sum_simi/num
        dic_simi[i[0]]['jump_up']=0
        dic_simi[i[0]]['jump_down']=0
        if i[0] in up_nodes:
            dic_simi[i[0]]['jump_up']=aver_simi
        if i[0] in down_nodes:
            dic_simi[i[0]]['jump_down']=1-aver_simi
        dic_simi[i[0]]['aver']=aver_simi  #跳转平均概率
        dic_simi[i[0]]['max']=max_simi
        dic_simi[i[0]]['sum']=sum_simi+dic_simi[i[0]]['jump_up']+dic_simi[i[0]]['jump_down']  #跳转概率之和
    return nodes,dic_simi
def single_random(k_th,multi_simi,layer_nodes,random_times,walk_len,randomfile):
    all_random=[]
    random_w = csv.writer(open(randomfile, 'a', encoding='utf-8', newline=''), delimiter=' ')
    all_node_seq=locals()  #所有节点列表
    name_th=copy.deepcopy(k_th)  #第k层
    copy_simi=copy.deepcopy(multi_simi)
    use_nodes=layer_nodes[name_th]
    # if name_th>=1:
    #     use_nodes=[node for node in layer_nodes[name_th] if node not in layer_nodes[name_th-1]]
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
                        k_th=k_th+1  #层数加一
                        next_node=node_choice1(node_seq,multi_simi[k_th])
                    elif next_node=='jump_up': #节点跳转到上一层
                        k_th=k_th-1  #层数减1
                        next_node=node_choice1(node_seq,multi_simi[k_th])
                    else:  #后继节点选择成功
                        break
                t=random.uniform(0,multi_simi[k_th][node_seq[-1]]['max'])
                if multi_simi[k_th][node_seq[-1]][next_node]<t:
                    multi_simi[k_th][node_seq[-1]]['sum']-=multi_simi[k_th][node_seq[-1]][next_node]
                    multi_simi[k_th][node_seq[-1]][next_node]=0
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
        if simi_node!='aver' and simi_node!='sum' and simi_node!='max':
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
    dir1,dir2,dir3='../graph/','../random/','E:/实验/对比实验/source_emb/'
    edgefile="karate-mirrored.edgelist"
    randomfile='random.csv'
    trainsave='karate_fhne.emb'
    # edgefile="brazil-flights.edgelist"
    # randomfile='random.csv'
    # trainsave='brazil_fhne.emb'
    layer=3
    all_nodes,multi_simi,layer_nodes,k=k_core_mgnrl(dir1+edgefile,layer,1,0.1)  #标准
    print([len(x) for x in layer_nodes])

    print('开启游走')
    if os.access(dir2+randomfile,os.F_OK):
        os.remove(dir2+randomfile)
    start = datetime.datetime.now()
    p = Pool(layer)
    for i in range(layer):
        p.apply_async(single_random, args=(i, multi_simi, layer_nodes, 5, 25, dir2+randomfile))
    p.close()
    p.join()
    # for i in range(4):
    #     single_random(i, multi_simi, layer_nodes,  10, 80, dir2+randomfile)
    # single_random(2, multi_simi, layer_nodes,  10, 100, dir2+randomfile)
    # single_random(layer-1, multi_simi, layer_nodes,  10, 100, dir2+randomfile)
    # time.sleep(5)
    # # skip-gram训练
    skip_gram(dir2+randomfile, all_nodes, 2, 3, 4, dir3+trainsave)
    # os.remove(dir2+randomfile)