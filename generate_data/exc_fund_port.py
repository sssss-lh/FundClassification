#%%
''' 处理基金持仓数据 '''
import numpy as np
import pandas as pd
import sys
import time
import os
from utils import *
#%%
class fundport_exc():
    def __init__(self, fund_portf:pd.DataFrame) -> None:
        self.fund_portf = fund_portf
        self.sel_dt = []
        for yr in range(2021, 2009, -1):
            self.sel_dt.extend([str(yr) + '1101', str(yr) + '0901', str(yr) + '0501', str(yr) + '0401'])
        self.sel_dt = np.array(self.sel_dt)
        self.exc_raw_data()

    def exc_raw_data(self) -> pd.DataFrame:
        '''
        将原始数据进行处理
        '''
        # 计算selectdt 每个基金的selectdt 是大于该条记录的anndt的第一个selectdt
        print('cal_sel_dt')
        self.fund_portf['selectdt'] = self.fund_portf.anndt.apply(lambda x : self.sel_dt[self.sel_dt > '%s'%x][-1])
        # 由于2019年12.31季度的年报出现了大规模推迟 将2019年12.31 推迟披露的年报转移到2020年4月1日进行考虑 引入了部分未来信息
        idx = self.fund_portf.query("(seasndt == '20191231') & (selectdt == '20200501')").index
        self.fund_portf.loc[idx, 'selectdt'] = '20200401'
        # 去掉持仓占比为0的记录
        self.fund_portf = self.fund_portf[~(self.fund_portf['proportion'].isin([0]))]
        # 计算每个基金每个季度的总披露持仓
        tot_reveal = self.fund_portf.groupby(['fundcode','seasndt']).proportion.sum().to_frame().rename(columns = {'proportion':'tot_reveal'}).reset_index()
        self.fund_portf = pd.merge(self.fund_portf, tot_reveal, on = ['fundcode','seasndt'])
        # 调整占比，调整后每只基金每个selectdt的持仓比 合为1
        self.fund_portf['proportion_adj'] = self.fund_portf.proportion / self.fund_portf.tot_reveal
        # 计算披露占总仓位之比
        self.fund_portf['reveal_rate'] = self.fund_portf.tot_reveal / self.fund_portf.position
