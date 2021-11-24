# %%
import numpy as np
import pandas as pd
import sys
import time
import os
from typing import Tuple
#%%
class nav_exc():
    def __init__(self, fundnav:pd.DataFrame, indexs:pd.DataFrame, is_raw:bool = True) -> None:
        self.fundnav = fundnav
        self.indexs = indexs
        self.sel_dt = []
        for yr in range(2022, 2009, -1):
            self.sel_dt.extend([str(yr) + '1101', str(yr) + '0901', str(yr) + '0501', str(yr) + '0401'])
        self.sel_dt = np.array(self.sel_dt)
        self.fundret, self.ff3_indexs, self.reg4_indexs = self.exc_raw_data()
        self.corr_tbl = self.cal_corr_table()

    def exc_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        将原始数据进行处理
        '''
        # 处理基金净值
        # selectdt 为大于该交易日的第一个selectdt
        print('select date')
        self.fundnav['selectdt'] = self.fundnav['tradedt'].apply(lambda x : self.sel_dt[self.sel_dt > '%s'%x][-1])
        self.fundnav.sort_values('tradedt', inplace=True)
        # 每只基金计算每天收益率
        self.fundnav['ret'] = self.fundnav.groupby(['fundcode']).nav.pct_change()
        self.fundnav.dropna(inplace = True)
        self.fundnav.reset_index(drop = True, inplace = True)
        fundret = self.fundnav[['fundcode','tradedt','selectdt','ret']]

        self.indexs['selectdt'] = self.indexs['tradedt'].apply(lambda x : self.sel_dt[self.sel_dt > '%s'%x][-1])
        self.indexs['mkt_ret'] = self.indexs.mkt.pct_change()
        self.indexs['smb_ret'] = self.indexs.smb.pct_change()
        self.indexs['hml_ret'] = self.indexs.hml.pct_change()
        self.indexs['sml_ret'] = self.indexs['sml'].pct_change()
        self.indexs['big_ret'] = self.indexs['big'].pct_change()
        self.indexs['val_ret'] = self.indexs['val'].pct_change()
        self.indexs['grw_ret'] = self.indexs['grw'].pct_change()
        self.indexs.dropna(inplace= True)
        self.indexs.rename(columns= {'TRADE_DT':'tradedt'}, inplace=True)
        self.indexs = self.indexs.query("tradedt > '20100101'")
        ''' fama french 3因子数据'''
        ff3_indexs = self.indexs[['tradedt','selectdt','mkt_ret','smb_ret','hml_ret']].reset_index(drop=True)
        ''' 巨潮小盘 巨潮大盘 巨潮价值 巨潮成长 '''
        reg4_indexs = self.indexs[['tradedt','selectdt','sml_ret','big_ret','val_ret','grw_ret']].reset_index(drop=True)
        return fundret, ff3_indexs, reg4_indexs

    def cal_corr_table(self) -> pd.DataFrame:
        ''' 计算每一个期间 每只基金净值收益率 和四个指数收益率的相关系数 '''
        merge_info_ = pd.merge(self.fundret, self.reg4_indexs, on = ['tradedt','selectdt'])
        corr_tbl = merge_info_.groupby(['fundcode','selectdt']).apply(lambda x: x.corrwith(x['ret']))
        corr_tbl.drop('ret',axis=1,inplace=True)
        corr_tbl['max_idx'] = corr_tbl.idxmax(axis=1) #求一行的最大值对应的索引
        corr_tbl['max_val']= corr_tbl.max(axis=1) #取出该最大值
        corr_tbl['mv_style'] = corr_tbl[['sml_ret','big_ret']].idxmax(axis=1) #求一行的最大值对应的索引
        corr_tbl['g_v_style'] = corr_tbl[['val_ret','grw_ret']].idxmax(axis=1) #求一行的最大值对应的索引
        mv_dict = {'sml_ret':-1,'big_ret':1}
        g_v_dict = {'val_ret':-1,'grw_ret':1}
        corr_tbl['mv_label'] = corr_tbl.mv_style.map(mv_dict)
        corr_tbl['g_v_label'] = corr_tbl.g_v_style.map(g_v_dict)
        corr_tbl[['c_mv_change','c_g_v_change']] = corr_tbl.groupby('fundcode')['mv_label','g_v_label'].diff()
        return corr_tbl
