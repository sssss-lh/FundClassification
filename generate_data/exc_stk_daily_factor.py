#%%
''' 处理dailyfactor '''
import numpy as np
import pandas as pd
import sys
import time
import os
from utils import *
#%%
class stkfactor_exc():
    def __init__(self, stk_daily_factor:pd.DataFrame, is_raw:bool = True) -> None:
        '''
        类参数初始化
        input:
                stk_daily_factor: pd.DataFrame ['ffmv','totmv','toteqyconmin','toteqyexmin','npexnonreglpttm','opencfttm','operevttm','tradedt','stockcode','famv','seasndt','anndt']
                is_raw: bool 如果为False，则不对输入的数据进行处理
        '''
        self.stk_daily_factor = stk_daily_factor
        # 导出所有的seasndt 
        self.seasn_dts = np.sort(self.stk_daily_factor['seasndt'].unique())
        self.seasn_dts_df = pd.DataFrame(self.seasn_dts, columns=['seasndt'])
        # 保存seasndt
        # 回滚计算近三年的增长 时间往前回滚四个季度 作为dict
        self.dtdict = dict(zip(self.seasn_dts[:-4],self.seasn_dts[4:]))
        # 生成sel_dt 每个季度在sel_dt进行持仓分析
        self.sel_dt = []
        for yr in range(2021, 2009, -1):
            self.sel_dt.extend([str(yr) + '1101', str(yr) + '0901', str(yr) + '0501', str(yr) + '0401'])
        self.sel_dt = np.array(self.sel_dt)
        # 对数据预处理
        if(is_raw == True):
            self.exc_raw_data()
        print('raw_data_excute finished')
        self.stk_factor = self.cal_stk_factor()
        print('cal_stk_factor finished')
        self.stk_factor_f = self.cal_stk_factor_f()
        print('cal_stk_factor_f finished')

    def exc_raw_data(self) -> pd.DataFrame:
        '''
        将原始数据进行处理
        '''
        # 含少数股东的所有者权益 不能为0
        self.stk_daily_factor = self.stk_daily_factor[~(self.stk_daily_factor['toteqyconmin'].isin([0]))]
        # 不含少数股东的所有者权益 不能为0
        self.stk_daily_factor = self.stk_daily_factor[~(self.stk_daily_factor['toteqyexmin'].isin([0]))]
        # 营业收入ttm 不能为0
        self.stk_daily_factor = self.stk_daily_factor[~(self.stk_daily_factor['operevttm'].isin([0]))]
        # 往前1年
        ''' stk_daily_factor_ 为把数据都全部处理好的表段(只含val和grw 不含原始数据) stk_daily_factor为原始数据'''
        ''' 将1年前/2年前/3年前的表段与今年的merge 左连接 产生空缺的值用前面的填充 '''
        last = self.stk_daily_factor[['stockcode','seasndt','npexnonreglpttm','toteqyconmin','operevttm','opencfttm']].drop_duplicates()
        last = last.rename(columns={'npexnonreglpttm':'npexnonreglpttm1','toteqyconmin':'toteqyconmin1','operevttm':'operevttm1','opencfttm':'opencfttm1'})
        last['seasndt'] = last['seasndt'].map(self.dtdict)
        self.stk_daily_factor = pd.merge(self.stk_daily_factor, last, on=['stockcode', 'seasndt'] , how='left')
        # 往前2年
        last2 = last.copy()
        last2['seasndt'] = last2['seasndt'].map(self.dtdict) 
        last2 = last2.rename(columns={'npexnonreglpttm1':'npexnonreglpttm2','toteqyconmin1':'toteqyconmin2','operevttm1':'operevttm2','opencfttm1':'opencfttm2'})
        self.stk_daily_factor = pd.merge(self.stk_daily_factor, last2, on=['stockcode', 'seasndt'], how='left')
        # 往前3年
        last3 = last2.copy()
        last3['seasndt'] = last3['seasndt'].map(self.dtdict)
        last3 = last3.rename(columns={'npexnonreglpttm2':'npexnonreglpttm3','toteqyconmin2':'toteqyconmin3','operevttm2':'operevttm3','opencfttm2':'opencfttm3'})
        self.stk_daily_factor = pd.merge(self.stk_daily_factor, last3, on=['stockcode', 'seasndt'], how='left')
        # 如果有nan值 则用该股票前面一个数据补齐
        self.stk_daily_factor = self.stk_daily_factor.groupby('stockcode', as_index=False).apply(pd.DataFrame.fillna, method = 'pad')
        


    def cal_stk_factor(self) -> pd.DataFrame:
        '''
        计算没有进行MAD和z_score之前的因子值
        '''
        ''' 市值取对数 '''
        stk_daily_factor_  = self.stk_daily_factor.copy()
        # print(stk_daily_factor_)
        stk_daily_factor_['log_mv'] = stk_daily_factor_['ffmv'].apply(np.log)
        ''' 账面净值/总市值 '''
        stk_daily_factor_['bp'] = stk_daily_factor_['toteqyconmin']/stk_daily_factor_['totmv']
        ''' 盈利ttm/总市值 '''
        stk_daily_factor_['ep'] = stk_daily_factor_['npexnonreglpttm']/stk_daily_factor_['totmv']
        ''' 营业收入ttm/总市值'''
        stk_daily_factor_['sp'] = stk_daily_factor_['operevttm']/stk_daily_factor_['totmv']
        ''' 经营现金流ttm/总市值 '''
        stk_daily_factor_['cfp'] = stk_daily_factor_['opencfttm']/stk_daily_factor_['totmv']

        ''''''
        ''' ROE '''
        stk_daily_factor_['roe'] = stk_daily_factor_['npexnonreglpttm']/stk_daily_factor_['toteqyconmin']
        ''' 近三年净资产增长率平均值 '''
        stk_daily_factor_['growth_equity'] = (stk_daily_factor_['toteqyconmin']/abs(stk_daily_factor_['toteqyconmin1']) + stk_daily_factor_['toteqyconmin1']/abs(stk_daily_factor_['toteqyconmin2']) + stk_daily_factor_['toteqyconmin2']/abs(stk_daily_factor_['toteqyconmin3']))/3 -1
        ''' 近三年归母扣非净利润增长率平均值 '''
        stk_daily_factor_['growth_or'] = (stk_daily_factor_['npexnonreglpttm']/abs(stk_daily_factor_['npexnonreglpttm1']) + stk_daily_factor_['npexnonreglpttm1']/abs(stk_daily_factor_['npexnonreglpttm2']) + stk_daily_factor_['npexnonreglpttm2']/abs(stk_daily_factor_['npexnonreglpttm3']))/3 -1
        ''' 近三年主营业务收入增长率平均值 '''
        stk_daily_factor_['growth_profit'] = (stk_daily_factor_['operevttm']/abs(stk_daily_factor_['operevttm1']) + stk_daily_factor_['operevttm1']/abs(stk_daily_factor_['operevttm2']) + stk_daily_factor_['operevttm2']/abs(stk_daily_factor_['operevttm3']))/3 -1
        ''' 近三年经营现金流增长率平均值 '''
        stk_daily_factor_['growth_ncf'] = (stk_daily_factor_['opencfttm']/abs(stk_daily_factor_['opencfttm1']) + stk_daily_factor_['opencfttm1']/abs(stk_daily_factor_['opencfttm2']) + stk_daily_factor_['opencfttm2']/abs(stk_daily_factor_['opencfttm3']))/3 -1
        ''' 取因子logmv bp ep sp cfp growth_equity growth_or growth_profit growth_ncf '''
        stk_daily_factor_ = stk_daily_factor_[['stockcode','STK_NAME','tradedt','log_mv','bp','ep','sp','cfp','growth_equity','growth_or','growth_profit','growth_ncf']]

        ''' 生成sel_dt 每个季度在sel_dt进行持仓分析 '''
        ''' 每一天对应的selectdt 是大于该tradedt的第一个selectdt '''
        stk_daily_factor_['selectdt'] = stk_daily_factor_['tradedt'].apply(lambda x : self.sel_dt[self.sel_dt > '%s'%x][-1])
        ''' 如果有nan值 则用该股票的前一个数据补齐 '''
        stk_daily_factor_ = stk_daily_factor_.groupby('stockcode', as_index=False).apply(pd.DataFrame.fillna, method = 'pad')
        ''' 在每一个selectdt 选取股票小于该selectdt 的最后一个交易日的因子值 '''
        stk_factor = stk_daily_factor_.sort_values('tradedt').groupby(['stockcode','selectdt']).tail(1)
        return stk_factor

    def cal_stk_factor_f(self) -> pd.DataFrame:
        '''
        对因子截面上做MAD 和 z_score
        '''
        stk_factor_f = self.stk_factor.dropna().copy()
        ''' MAD '''
        stk_factor_f = stk_factor_f.groupby('selectdt').apply(MAD3, ['bp','ep','sp','cfp','growth_equity','growth_or','growth_profit','growth_ncf']).reset_index(drop = True)
        ''' z_score '''
        stk_factor_f = stk_factor_f.groupby('selectdt').apply(z_score, ['log_mv','bp','ep','sp','cfp','growth_equity','growth_or','growth_profit','growth_ncf']).reset_index(drop = True)
        ''' 计算成长指标 '''
        stk_factor_f['growth'] = stk_factor_f.growth_equity*0.25 + stk_factor_f.growth_or*0.25 + stk_factor_f.growth_profit*0.25 + stk_factor_f.growth_ncf*0.25
        ''' 计算价值指标 '''
        stk_factor_f['value'] = stk_factor_f.bp*0.4 + stk_factor_f.ep*0.2 + stk_factor_f.sp*0.2 + stk_factor_f.cfp*0.2
        ''' 成长-价值 = 风格'''
        stk_factor_f['g_v'] = stk_factor_f.growth - stk_factor_f.value
        ''' g_v 做z_score'''
        stk_factor_f = stk_factor_f.groupby('selectdt').apply(z_score, ['g_v'])
        return stk_factor_f

