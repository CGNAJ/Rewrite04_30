
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import random
import math
import csv
from scipy.optimize import root, fsolve

PATH = "./data2021/04_30/rewrite.csv"
row = 6
column = 7

#mu+
event_num1 = 15000
#mu-
event_num2 = 15000
momentum1= 2
momentum2= 85
#入射角度的最大值
max_anglep = 40
#入射角度的最小值
min_anglep = 10
#无磁场区域的高度（m）
mag_radius = 6.8
#噪声概率
noise_probability = 0.001
#单个strip的接收概率
efficiency = 0.7

#随机生成动量和入射角度
def ptc_init(ptc_type):
    pai=3.14159265
    #将角度转化为弧度
    anglep = round(random.uniform(min_anglep, max_anglep),4)
    angle = round(anglep/180*pai,4)

    if ptc_type == 1:
        momentum = round (random.uniform(momentum1,momentum2),4)
    elif ptc_type == -1:
        momentum = round(random.uniform(momentum1,momentum2), 4)
    elif ptc_type == 0:
        momentum = efficiency_curve_momentum
    else:
        print("wrong particle type")

    #动量与半径的关系为固定值约为6.6，已考虑相对论效应
    particle_r = round(momentum * 6.6,4)

    # 数学计算，求出particle的x和y，即圆心的x,y坐标
    if ptc_type == -1:
        particle_x = round(mag_radius * math.tan(angle) + particle_r * math.cos(angle),5)
        particle_y = round(mag_radius - particle_r * math.sin(angle),5)
    elif ptc_type == 1:
        particle_x = round(mag_radius * math.tan(angle) - particle_r * math.cos(angle),5)
        particle_y = round(mag_radius + particle_r * math.sin(angle),5)
    return [particle_x, particle_y, particle_r,anglep]
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#



class RPC:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2



# RPC是平行于x轴的。返回float类型x坐标。
def hit_single(RPC, ptc,ptc_type):
    x1, y1, x2, y2 = RPC.x1, RPC.y1, RPC.x2, RPC.y2
    particle_x, particle_y, particle_r = ptc[0], ptc[1], ptc[2]
    # 计算hit的横坐标
    delta = particle_r ** 2 - (y1 - particle_y) ** 2
    if delta < 0:
        return -1

    if ptc_type == -1:
        x = round(particle_x - pow(delta, 0.5),5)
    elif ptc_type == 1:
        x = round(particle_x + pow(delta, 0.5),5)

    return x

# RPC参数设定,(x1,y1,x2,y2)
def RPC_init():
    RPC_3_2 = RPC(0, 9.838, 12.267, 9.838)
    RPC_3_1 = RPC(0, 9.832, 12.267, 9.832)
    RPC_2_2 = RPC(0, 7.484, 9.66, 7.484)
    RPC_2_1 = RPC(0, 7.478, 9.66, 7.478)
    RPC_1_2 = RPC(0, 6.806, 9.147, 6.806)
    RPC_1_1 = RPC(0, 6.8, 9.147, 6.8)
    return (RPC_1_1, RPC_1_2, RPC_2_1, RPC_2_2, RPC_3_1, RPC_3_2)

# 生成主hit，把x坐标转化为strip序号。
# 次级重要
# x==mainhit和 y==nearby_hit
def mainhit_all(RPC_all, ptc,ptc_type,i):
    strip_list = []
    for k in range(len(RPC_all)):
        x = hit_single(RPC_all[k], ptc, ptc_type)
        xstrip = main_strip(x)
        ystrip = side_effect(x, xstrip)
        ypro = near_ratio(x,xstrip)
        x = round(main_efficiency_bias(xstrip), 4)
        y = round(near_efficiency(ystrip, ypro), 4)
        strip_list.append(x)
        strip_list.append(y)
        noise_arr = write_noise(x, y, k, RPC_all)
        noise_arr.sort(reverse=True)
        strip_list.extend(noise_arr)
    #print(strip_list)
    return (strip_list)

#将主hit转变为strip序号,0.03是strip的width
def main_strip(mainhit_position):
    if mainhit_position > 0:
        mainhit_strip = int(mainhit_position / 0.03)
    else:
        mainhit_strip = -1
    return (mainhit_strip)

# 根据主hit在其strip上的位置判断nearby_hit的strip序号
# 入射角为(10,40),未考虑打出RPClayer的情况
def side_effect(mainhit_position, mainhit_strip):
    bia = mainhit_position - mainhit_strip * 0.03
    if bia >= 0.015:
        nearhit_strip = mainhit_strip + 1
    else:
        nearhit_strip = mainhit_strip - 1
    return (nearhit_strip)

#根据主hit在其strip上位置判断生成nearby_hit的概率
def near_ratio(mainhit_position, mainhit_strip):
    bia = mainhit_position - mainhit_strip * 0.03
    if bia >= 0.015:
        near_probability = bia / 0.015 - 1
    else:
        near_probability = 1- bia / 0.015
    return (near_probability)

#对主hit作eff判断,
def main_efficiency_bias(mainhit_strip):
    eff = random.uniform(0, 1)
    if eff < efficiency:
        biaspostion = mainhit_strip
    else:
        biaspostion = -1
    return(biaspostion)

#对nearhit作eff判断,      ####并用strip的中心坐标代替具体坐标
def near_efficiency(nearhit_strip, near_probability):
    near_eff = random.uniform(0, 1)
    if near_eff < efficiency * near_probability:
        nearposition = nearhit_strip
    else:
        nearposition = -1
    return(nearposition)

#随机写入noise, #新增查重功能
def write_noise(x, y, layer_number, RPC_all):
    noise_arr = [0, 0, 0, 0, 0]
    k = 0
    stripmax = int(RPC_all[layer_number].x2 / 0.03)
    for i in range (stripmax):
        noise_pa = random.uniform(0, 1)
        if noise_pa < noise_probability:
            if i !=x and i !=y:
                noise_arr[k] = i
                k = k + 1
                if k == 5:
                    break
        else:
            continue
    return(noise_arr)

#cluster的重建
def cluster_reconstruction(strip_list):
    hit_check = [0, 0, 0, 0, 0, 0]
    cluster_strip = [[], [], [], [], [], []]
    cluster_number = [0, 0, 0, 0, 0, 0]
    for i in range (6):
        check_strip = []
        hit_number = 0
        for k in range (7):
            if strip_list[i * 7 + k] == -1 or strip_list[i * 7 + k] > 0:
                hit_number = hit_number +1
                check_strip.append(strip_list[i * 7 + k])
            else:
                break
        #检索信号数量
        hit_check[i] = hit_number
        #无视信号种类降序排列信号
        check_strip.sort(reverse=True)
        check_strip = np.array(check_strip)
        print(check_strip)
        if hit_number > 0:
            m = 0
            while m < hit_number:
                #通过自适应的cluster边界[min,max],实现可延展的cluster
                max = check_strip[m]
                min = check_strip[m]
                sum = check_strip[m]
                hits_in_cluster = 1
                #repeat = 0
                for n in range(m+1, hit_number):
                    if check_strip [n] > 0:
                        if (min - check_strip[n]) == 1:
                            hits_in_cluster = hits_in_cluster + 1
                            sum = sum + check_strip[n]
                            min = check_strip[n]
                        else:
                            break
                        # 已添加查重功能
                        '''elif (min - check_strip[n]) == 0:
                            repeat = repeat + 1
                            continue'''
                        #print(check_strip,check_strip[n])
                    else:
                        continue
                #cluster的边界[min,max], 若未能展开说明没有连续信号
                if max > min:
                    cluster_strip[i].append([sum/hits_in_cluster, hits_in_cluster, i])
                    cluster_number[i] = cluster_number[i] + 1
                m = m + 1
                #延展cluster中的信号不再进入cluster的搜索循环
                if hits_in_cluster > 2:
                    m = m + hits_in_cluster-2
            #当不存在天然cluster时, 使用单点信号完成cluster的reconstruction
            if cluster_number [i] == 0:
                for m in range(hit_number):
                    if check_strip[m]>0:
                        cluster_strip[i].append([check_strip[m],1, i])
                        cluster_number[i] = cluster_number[i] + 1
                if cluster_number[i] == 0:
                    cluster_strip[i].append([-2,-2])
    return ([cluster_strip,cluster_number])

#按RPC层数组合super_cluster(故技重施)
def super_cluster(cluster_strip,cluster_number,detector_number):
    super_list=[]
    super_strip = []
    cluster_number = np.array(cluster_number)
    super_cluster_number = cluster_number[detector_number*2-2]+cluster_number[detector_number*2-1]
    for i in range(cluster_number[detector_number*2-2]):
        super_list.append(cluster_strip[detector_number*2-2][i])
    for i in range(cluster_number[detector_number*2-1]):
        super_list.append(cluster_strip[detector_number*2-1][i])
    super_list.sort(reverse=True)
    super_list= np.array(super_list)
    #print(super_list)
    if super_cluster_number > 0:
        m = 0
        while m < super_cluster_number:
            # 通过自适应的super_luster边界[min,max],实现可延展的super_cluster
            max = super_list[m][0]
            min = super_list[m][0]
            sum = 0
            clusters_in_super_cluster = 0
            hits_in_super_cluster = 0
            layer_number_cluster = 1
            for n in range(m , super_cluster_number):
                if super_list[n][0] > 0:
                    if 0 <= (min - super_list[n][0]) <= 1 :
                        clusters_in_super_cluster = clusters_in_super_cluster + 1
                        hits_in_super_cluster =hits_in_super_cluster + super_list[n][1]
                        sum = sum + super_list[n][0]
                        min = super_list[n][0]
                        if super_list[n][2] != super_list[m][2]:
                            layer_number_cluster = 2
                    else:
                        break
                else:
                    continue
            # super_cluster的边界没什么意义
            if max >= min:
                super_strip.append([sum / clusters_in_super_cluster, clusters_in_super_cluster,int(hits_in_super_cluster),layer_number_cluster])
            m = m + 1
            # 延展super_cluster中的cluster不再进入super_cluster的搜索循环
            if clusters_in_super_cluster > 1:
                m = m + clusters_in_super_cluster - 1
    else:
            super_strip.append([-1, -1, -1])
    return (super_strip)

#找到基准线在RPC3和RPC1上的交点
def seed_corresponding(candidate_seed):
    seed_x = candidate_seed[0] * 0.03 + 0.015
    seed_RPC3_x = seed_x * 9.835 / 7.481
    seed_RPC3_strip = int(seed_RPC3_x / 0.03)
    seed_RPC1_x = seed_x * 6.803 / 7.481
    seed_RPC1_strip = int(seed_RPC1_x / 0.03)
    return([seed_RPC1_strip,seed_RPC3_strip])

#通过比对super_cluster的距离选取最合适的构成candidate_event
def super_cluster_selection(seed_strip, RPC_cluster):
    difference_const = 33
    selection = -1
    for i in range(len(RPC_cluster)):
        if RPC_cluster[i][0]>0:
            difference = seed_strip - RPC_cluster[i][0]
            #scanning_window的左右界
            if -32 <=  difference <= 32:
                if abs(difference) < difference_const:
                    difference_const = abs(difference)
                    selection = i
                else:
                    continue
            else:
                continue
        else:
            continue
    if selection >=0:
        #print(RPC_cluster[selection])
        return (RPC_cluster[selection])

def write_into(event_number, ptc_type):
    i = 0
    v = 0
    w = 0
    while i < event_number:
        number_of_candidate_events = 0
        ptc = ptc_init(ptc_type)
        RPC_all = RPC_init()
        strip_list = mainhit_all(RPC_all, ptc, ptc_type,i)
        [cluster_strip,cluster_number] =cluster_reconstruction(strip_list)
        #print(cluster_strip,cluster_number)
        # RPC1上super_cluster
        RPC1_cluster = super_cluster(cluster_strip, cluster_number, 1)
        # RPC2上super_cluster
        RPC2_cluster = super_cluster(cluster_strip, cluster_number, 2)
        # RPC3上super_cluster
        RPC3_cluster = super_cluster(cluster_strip, cluster_number, 3)
        #print(1, RPC1_cluster)
        #print(2, RPC2_cluster)
        #print(3, RPC3_cluster)
        for k in range(len(RPC2_cluster)):
            candidate_seed = RPC2_cluster[k]
            if candidate_seed[0]>0:
                [seed_RPC1_strip,seed_RPC3_strip] = seed_corresponding(candidate_seed)
                #print(seed_RPC1_strip,seed_RPC3_strip)
                RPC1_selection = super_cluster_selection(seed_RPC1_strip, RPC1_cluster)
                RPC3_selection = super_cluster_selection(seed_RPC3_strip, RPC3_cluster)
                if RPC1_selection != None and RPC3_selection != None:
                    output_list = []
                    candidate_event = []
                    number_of_super_clusters = 3
                    hits_of_event = RPC1_selection[2] +candidate_seed[2] + RPC3_selection[2]
                    layers_with_hits_of_event = RPC1_selection[3] +candidate_seed[3] + RPC3_selection[3]
                    number_of_single_clusters = RPC1_selection[1] +candidate_seed[1] + RPC3_selection[1]
                    interval = abs(RPC1_selection[0]-seed_RPC1_strip)+abs(RPC3_selection[0]-seed_RPC3_strip)
                    candidate_event.append(RPC1_selection)
                    candidate_event.append(candidate_seed)
                    candidate_event.append(RPC3_selection)
                    candidate_event.append(number_of_super_clusters)
                    candidate_event.append(hits_of_event)
                    candidate_event.append(layers_with_hits_of_event)
                    candidate_event.append(number_of_single_clusters)
                    candidate_event.append(interval)
                    print(candidate_event)
                    number_of_candidate_events = number_of_candidate_events + 1
                    output_list.extend(strip_list)
                    output_list.append(RPC1_selection[0])
                    output_list.append(candidate_seed[0])
                    output_list.append(RPC3_selection[0])
                    w = w + 1
                    with open(PATH, "a") as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        writer.writerow(output_list)
        if number_of_candidate_events > 0:
            v = v + 1
        i = i + 1
    print(i, v , w)




# 遍历strip，得到输出
def output():
    write_into(event_num1, 1)
    write_into(event_num2, -1)

if __name__ == "__main__":
    output()
