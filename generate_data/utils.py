# 获取文件名
import pandas as pd 
from statsmodels import robust
import numpy as np

def MAD3(data, columns):
    '''
    input: 
            data：待检测的DataFrame
            columns: 需要进行绝对中位偏差调整的列值
    return：
            data:pd.DataFrame: MAD调整好之后的数据表段
    '''
    for c in columns:
        m = data[c].median()
        mad_c = robust.mad(data[c])
        data[c] -= m
        print(c)
        print(m - 3*mad_c, m, m + 3*mad_c)
        print(m - 3.5*mad_c, m, m + 3.5*mad_c)
        print(3*mad_c)
        ''' 将在3mad之外的全部拉回到3.5mad内'''
        b3i = data[data[c] > 3*mad_c].index
        l3i = data[data[c] < -3*mad_c].index
        data.loc[b3i, c] = (data.loc[b3i, c] - 3*mad_c)*((0.5*mad_c)/((data[c].max()) - 3*mad_c)) + 3*mad_c
        data.loc[l3i, c] = (data.loc[l3i, c] + 3*mad_c)*((0.5*mad_c)/(-3*mad_c - data[c].min())) - 3*mad_c
        data[c] += m
    return data

''' 用于计算z_score的函数'''
def z_score(data, columns):
    '''
    input:
            data：待检测的DataFrame
            columns: 需要进行z_score调整的列值
    return:
            data:pd.DataFrame 列值进行z_score之后的data
    '''
    for c in columns:
        data[c] = (data[c] - data[c].mean())/data[c].std()
    return data

def get_select_dts(start:int=2009, end:int=2021):
    ''' 
    return：
        sel_dt: ndarray 所有的selectdt时间节点
    '''
    sel_dt = []
    for yr in range(end, start, -1):
        sel_dt.append(str(yr) + '1101')
        sel_dt.append(str(yr) + '0901')
        sel_dt.append(str(yr) + '0501')
        sel_dt.append(str(yr) + '0401')
    sel_dt = np.array(sel_dt)
    return sel_dt

def get_analy_dts(start:int=2008, end:int=2021):
    ''' 
    return：
        analy_dt: ndarray 所有的selectdt时间节点
    '''
    analy_dt = []
    for yr in range(end, start, -1):
        analy_dt.append(str(yr) + '1120')
        analy_dt.append(str(yr) + '0901')
        analy_dt.append(str(yr) + '0520')
        analy_dt.append(str(yr) + '0401')
    analy_dt = np.array(analy_dt)
    return analy_dt

def get_sel_seasn_dict(start:int=2009, end:int=2021):
    ''' 
    return：
        ssdict: selectdt 和 seasndt的字典
    '''
    ssdict = {}
    for yr in range(end, start, -1):
        ssdict[str(yr) + '1101'] = str(yr) + '0930'
        ssdict[str(yr) + '0901'] = str(yr) + '0630'
        ssdict[str(yr) + '0501'] = str(yr) + '0331'
        ssdict[str(yr) + '0401'] = str(yr) + '1231'
    return ssdict