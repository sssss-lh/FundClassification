
data_dir = '../raw_data' #数据文件夹路径
res_dir = '../data'
isExists=os.path.exists(res_dir)
if not isExists:
    os.makedirs(res_dir)
pd.set_option('display.max_columns', None)
# %%
''' 导入数据 删除anndt为NaN的行 '''
stk_daily_factor = pd.read_pickle("%s/stkdailyfactor.pkl"%data_dir)
stk_daily_factor = stk_daily_factor.query("anndt != 'NaN'").sort_values(['stockcode','tradedt'])
# %%
''' 导出所有的seasndt '''
seasn_dts = np.sort(stk_daily_factor['seasndt'].unique())
seasn_dts_df = pd.DataFrame(seasn_dts, columns=['seasndt'])
seasn_dts_df.to_pickle("%s/seasn_dts_df.pkl"%res_dir)
''' 回滚计算近三年的增长 时间往前回滚四个季度 作为dict '''
dtdict = dict(zip(seasn_dts[:-4],seasn_dts[4:]))
# %%
''' 添加股票简称 '''
stk_name_map = pd.read_excel("%s/stk_map.xlsx"%data_dir).rename(columns= {'WindCodes':'stockcode','SEC_NAME':'STK_NAME'})
stk_daily_factor = pd.merge(stk_daily_factor, stk_name_map, on = 'stockcode')

''' 含少数股东的所有者权益 不能为0'''
stk_daily_factor = stk_daily_factor[~(stk_daily_factor['toteqyconmin'].isin([0]))]
''' 不含少数股东的所有者权益 不能为0'''
stk_daily_factor = stk_daily_factor[~(stk_daily_factor['toteqyexmin'].isin([0]))]
''' 营业收入ttm 不能为0'''
stk_daily_factor = stk_daily_factor[~(stk_daily_factor['operevttm'].isin([0]))]

# %%
# 往前1年
''' stk_daily_factor_ 为把数据都全部处理好的表段(只含val和grw 不含原始数据) stk_daily_factor为原始数据'''
''' 将1年前/2年前/3年前的表段与今年的merge 左连接 产生空缺的值用前面的填充 '''
last = stk_daily_factor[['stockcode','seasndt','npexnonreglpttm','toteqyconmin','operevttm','opencfttm']].drop_duplicates()
last = last.rename(columns={'npexnonreglpttm':'npexnonreglpttm1','toteqyconmin':'toteqyconmin1','operevttm':'operevttm1','opencfttm':'opencfttm1'})
last['seasndt'] = last['seasndt'].map(dtdict)
stk_daily_factor_ = pd.merge(stk_daily_factor, last, on=['stockcode', 'seasndt'] , how='left')
# 往前2年
last2 = last.copy()
last2['seasndt'] = last2['seasndt'].map(dtdict) 
last2 = last2.rename(columns={'npexnonreglpttm1':'npexnonreglpttm2','toteqyconmin1':'toteqyconmin2','operevttm1':'operevttm2','opencfttm1':'opencfttm2'})
stk_daily_factor_ = pd.merge(stk_daily_factor_, last2, on=['stockcode', 'seasndt'], how='left')
# 往前3年
last3 = last2.copy()
last3['seasndt'] = last3['seasndt'].map(dtdict)
last3 = last3.rename(columns={'npexnonreglpttm2':'npexnonreglpttm3','toteqyconmin2':'toteqyconmin3','operevttm2':'operevttm3','opencfttm2':'opencfttm3'})
stk_daily_factor_ = pd.merge(stk_daily_factor_, last3, on=['stockcode', 'seasndt'], how='left')

#%%
''' 如果有nan值 则用该股票前面一个数据补齐 '''
stk_daily_factor_ = stk_daily_factor_.groupby('stockcode', as_index=False).apply(pd.DataFrame.fillna, method = 'pad')

#%%
''' 市值取对数 '''
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

# %%
''' 取因子logmv bp ep sp cfp growth_equity growth_or growth_profit growth_ncf '''
stk_daily_factor_ = stk_daily_factor_[['stockcode','STK_NAME','tradedt','log_mv','bp','ep','sp','cfp','growth_equity','growth_or','growth_profit','growth_ncf']]
# %%
''' 生成sel_dt 每个季度在sel_dt进行持仓分析 '''
sel_dt = []
for yr in range(2021, 2009, -1):
    sel_dt.append(str(yr) + '1101')
    sel_dt.append(str(yr) + '0901')
    sel_dt.append(str(yr) + '0501')
    sel_dt.append(str(yr) + '0401')
sel_dt = np.array(sel_dt)
# %%
''' 每一天对应的selectdt 是大于该tradedt的第一个selectdt '''
stk_daily_factor_['selectdt'] = stk_daily_factor_['tradedt'].apply(lambda x : sel_dt[sel_dt > '%s'%x][-1])
''' 如果有nan值 则用该股票的前一个数据补齐 '''
stk_daily_factor_ = stk_daily_factor_.groupby('stockcode', as_index=False).apply(pd.DataFrame.fillna, method = 'pad')
''' 每一个基金在每一个selectdt 选取小于该selectdt 的最后一个交易日 '''
stk_factor = stk_daily_factor_.sort_values('tradedt').groupby(['stockcode','selectdt']).tail(1)
# %%
stk_factor.to_pickle("%s/stkfactor.pkl"%res_dir)
# %%
''' 处理因子 '''
''' 对前面的数据进行截面MAD 和 z_score '''
from utils import *
# %%
stk_factor_f = stk_factor.dropna().copy()
# %%
''' MAD '''
stk_factor_f = stk_factor_f.groupby('selectdt').apply(MAD3, ['bp','ep','sp','cfp','growth_equity','growth_or','growth_profit','growth_ncf']).reset_index(drop = True)
''' z_score '''
stk_factor_f = stk_factor_f.groupby('selectdt').apply(z_score, ['log_mv','bp','ep','sp','cfp','growth_equity','growth_or','growth_profit','growth_ncf']).reset_index(drop = True)
#%%
''' 计算成长指标 '''
stk_factor_f['growth'] = stk_factor_f.growth_equity*0.25 + stk_factor_f.growth_or*0.25 + stk_factor_f.growth_profit*0.25 + stk_factor_f.growth_ncf*0.25
''' 计算价值指标 '''
stk_factor_f['value'] = stk_factor_f.bp*0.4 + stk_factor_f.ep*0.2 + stk_factor_f.sp*0.2 + stk_factor_f.cfp*0.2
''' 成长-价值 = 风格'''
stk_factor_f['g_v'] = stk_factor_f.growth - stk_factor_f.value
''' g_v 做z_score'''
stk_factor_f = stk_factor_f.groupby('selectdt').apply(z_score, ['g_v'])
# %%
''' 保存 '''
stk_factor_f.to_pickle("%s/stkfactor_f.pkl"%res_dir)

# %%
# 处理基金的持仓数据
data_dir = '../raw_data' #数据文件夹路径
res_dir = '../data'
isExists = os.path.exists(res_dir)
if not isExists:
    os.makedirs(res_dir)
pd.set_option('display.max_columns', None)
#%%
''' 导入基金持仓 '''
fund_portf = pd.read_pickle("%s/CHINAMUTUALFUNDSTOCKPORTFOLIO.pkl"%data_dir)
''' 基金代码 截止日期 股票代码 持仓占比 披露日期'''
fund_portf_ = fund_portf[['S_INFO_WINDCODE','F_PRT_ENDDATE','S_INFO_STOCKWINDCODE','F_PRT_STKVALUETONAV','ANN_DATE']]
fund_portf_.rename(columns = {'S_INFO_WINDCODE':'fundcode','F_PRT_ENDDATE':'seasndt','S_INFO_STOCKWINDCODE':'stockcode','F_PRT_STKVALUETONAV':'proportion','ANN_DATE':'anndt'}, inplace = True)
''' 添加基金名 '''
''' 基金资产类型数据 '''
''' 基金代码 截止日期 '''
fund_asset = pd.read_pickle("%s/chinamutualfundassetportfolio.pkl"%data_dir)
fund_asset_ = fund_asset[['S_INFO_WINDCODE','F_PRT_ENDDATE','F_PRT_STOCKTONAV']]
fund_asset_.rename(columns = {'S_INFO_WINDCODE':'fundcode','F_PRT_ENDDATE':'seasndt','F_PRT_STOCKTONAV':'position'}, inplace = True)
fund_asset_.dropna(inplace = True)
#%%
# 添加基金名称
fnd_dscr = pd.read_pickle("%s/CHINAMUTUALFUNDDESCRIPTION.pkl"%data_dir)[['F_INFO_WINDCODE','F_INFO_NAME']].rename(columns = {'F_INFO_WINDCODE':'fundcode','F_INFO_NAME':'fundname'})
fund_portf_ = pd.merge(fund_portf_, fnd_dscr, on = 'fundcode')
''' 导入每个季度的时间点 '''
seasn_dts_df = pd.read_pickle("%s/seasn_dts_df.pkl"%res_dir)
''' 仅选取季度节点作为披露日期的基金 '''
fund_portf_ = pd.merge(fund_portf_, seasn_dts_df, on =['seasndt'])
fund_portf_ = pd.merge(fund_portf_, fund_asset_, on = ['fundcode','seasndt'])
# %%
''' 生成sel_dt 每个季度在sel_dt进行持仓分析 '''
sel_dt = []
for yr in range(2021, 2001, -1):
    sel_dt.append(str(yr) + '1101')
    sel_dt.append(str(yr) + '0901')
    sel_dt.append(str(yr) + '0501')
    sel_dt.append(str(yr) + '0401')
sel_dt = np.array(sel_dt)
# %%
''' 计算selectdt 每个基金的selectdt 是大于该条记录的anndt的第一个selectdt '''
fund_portf_['selectdt'] = fund_portf_.anndt.apply(lambda x : sel_dt[sel_dt > '%s'%x][-1])
''' 由于2019年12.31季度的年报出现了大规模推迟 将2019年12.31 推迟披露的年报转移到2020年4月1日进行考虑 引入了部分未来信息'''
idx = fund_portf_.query("(seasndt == '20191231') & (selectdt == '20200501')").index
fund_portf_.loc[idx, 'selectdt'] = '20200401'
#%%
''' 去掉持仓占比为0的记录 '''
fund_portf_ = fund_portf_[~(fund_portf_['proportion'].isin([0]))]
''' 计算每个基金每个季度的总披露持仓 '''
tot_reveal = fund_portf_.groupby(['fundcode','seasndt']).proportion.sum().to_frame().rename(columns = {'proportion':'tot_reveal'}).reset_index()
fund_portf_ = pd.merge(fund_portf_, tot_reveal, on = ['fundcode','seasndt'])
# %%
''' 调整占比，调整后每只基金每个selectdt的持仓比 合为1 '''
fund_portf_['proportion_adj'] = fund_portf_.proportion / fund_portf_.tot_reveal
#%%
# 计算披露占总仓位之比
fund_portf_['reveal_rate'] = fund_portf_.tot_reveal / fund_portf_.position
# %%
''' 保存 '''
fund_portf_.to_pickle("%s/fund_portf_.pkl"%res_dir)


''' 处理基金净值 和指数的净值 '''
#%%
import numpy as np
import pandas as pd
import sys
import time
import os
#%%
data_dir = '../raw_data' #数据文件夹路径
res_dir = '../data' # 生成数据的文件夹
isExists=os.path.exists(res_dir)
if not isExists:
    os.makedirs(res_dir)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', None)
# %%
''' 生成sel_dt 每个季度在sel_dt进行持仓分析 '''
sel_dt = []
for yr in range(2022, 2009, -1):
    sel_dt.append(str(yr) + '1101')
    sel_dt.append(str(yr) + '0901')
    sel_dt.append(str(yr) + '0501')
    sel_dt.append(str(yr) + '0401')
sel_dt = np.array(sel_dt)
# %%
''' 导入基金的净值 '''
fundnav = pd.read_pickle("%s/CHINAMUTUALFUNDNAV.pkl"%data_dir)
''' 选取基金代码fundcode 交易日tradedt 净值nav '''
fundnav_ = fundnav[['F_INFO_WINDCODE','PRICE_DATE', 'F_NAV_ADJUSTED']].sort_values('PRICE_DATE').rename(columns = {'F_INFO_WINDCODE':'fundcode','PRICE_DATE':'tradedt','F_NAV_ADJUSTED':'nav'})
fundnav_ = fundnav_.query("tradedt > '20100101'")

fundnav_supply = pd.read_pickle("%s/chinamutualfundnav.pkl"%data_dir).query("PRICE_DATE > '20210411'")
fundnav_supply = fundnav_supply[['F_INFO_WINDCODE','PRICE_DATE', 'F_NAV_ADJUSTED']].sort_values('PRICE_DATE').rename(columns = {'F_INFO_WINDCODE':'fundcode','PRICE_DATE':'tradedt','F_NAV_ADJUSTED':'nav'})

fundnav_ = pd.concat([fundnav_, fundnav_supply], axis = 0)
#%%
#按照时间排序
fundnav_.sort_values('tradedt', inplace=True)
''' 每只基金计算每天收益率 '''
fundnav_['ret'] = fundnav_.groupby(['fundcode']).nav.pct_change()
# %%
''' selectdt 为大于该交易日的第一个selectdt '''
fundnav_['selectdt'] = fundnav_['tradedt'].apply(lambda x : sel_dt[sel_dt > '%s'%x][-1])
fundnav_.dropna(inplace = True)
fundnav_.reset_index(drop=True, inplace = True)
fundret_ = fundnav_[['fundcode','tradedt','selectdt','ret']]

# %%
''' 市场指数数据 '''
mkt_index = pd.read_pickle("%s/marketindex.pkl"%data_dir)
mkt_index_ = mkt_index[['TRADE_DT','S_DQ_CLOSE']]
''' 风格指数数据 '''
style_index = pd.read_pickle("%s/styleindex.pkl"%data_dir)
style_index_ = style_index.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_CLOSE')
'''所有指数汇总 并按照TRADE_DT排序'''
indexs = pd.merge(mkt_index_, style_index_, on='TRADE_DT').sort_values('TRADE_DT').reset_index(drop= True)
indexs['mkt'] = indexs['S_DQ_CLOSE']/indexs['S_DQ_CLOSE'].iloc[0]
indexs['smb'] = indexs['399316.SZ']/indexs['399314.SZ']
indexs['hml'] = indexs['399371.SZ']/indexs['399370.SZ']
indexs['mkt_ret'] = indexs.mkt.pct_change()
indexs['smb_ret'] = indexs.smb.pct_change()
indexs['hml_ret'] = indexs.hml.pct_change()
indexs['sml_ret'] = indexs['399316.SZ'].pct_change()
indexs['big_ret'] = indexs['399314.SZ'].pct_change()
indexs['val_ret'] = indexs['399371.SZ'].pct_change()
indexs['grw_ret'] = indexs['399370.SZ'].pct_change()
indexs.dropna(inplace= True)
indexs.rename(columns= {'TRADE_DT':'tradedt'}, inplace=True)
indexs = indexs.query("tradedt > '20100101'")

''' fama french 3因子数据'''
ff3_indexs = indexs[['tradedt','selectdt','mkt_ret','smb_ret','hml_ret']].reset_index(drop=True)
''' 巨潮小盘 巨潮大盘 巨潮价值 巨潮成长 '''
reg4_indexs = indexs[['tradedt','selectdt','sml_ret','big_ret','val_ret','grw_ret']].reset_index(drop=True)

#%%
''' 计算每一个期间 每只基金净值收益率 和四个指数收益率的相关系数 '''
merge_info_ = pd.merge(fundret_, reg4_indexs, on = ['tradedt','selectdt'])
corr_tbl = merge_info_.groupby(['fundcode','selectdt']).apply(lambda x: x.corrwith(x['ret']))
corr_tbl.drop('ret',axis=1,inplace=True)
corr_tbl['max_idx'] = corr_tbl.idxmax(axis=1) #求一行的最大值对应的索引
corr_tbl['max_val']= corr_tbl.max(axis=1) #取出该最大值
#%%
corr_tbl['mv_style'] = corr_tbl[['sml_ret','big_ret']].idxmax(axis=1) #求一行的最大值对应的索引
corr_tbl['g_v_style'] = corr_tbl[['val_ret','grw_ret']].idxmax(axis=1) #求一行的最大值对应的索引
# corr_tbl[['sml_change','big_change','val_change','grw_change']] = corr_tbl.groupby('fundcode')['sml_ret','big_ret','val_ret','grw_ret'].diff()
# corr_tbl['c_mv_change'] = corr_tbl.big_change - corr_tbl.sml_change
# corr_tbl['c_g_v_change'] =corr_tbl.grw_change - corr_tbl.val_change
mv_dict = {'sml_ret':-1,'big_ret':1}
g_v_dict = {'val_ret':-1,'grw_ret':1}
corr_tbl['mv_label'] = corr_tbl.mv_style.map(mv_dict)
corr_tbl['g_v_label'] = corr_tbl.g_v_style.map(g_v_dict)
corr_tbl[['c_mv_change','c_g_v_change']] = corr_tbl.groupby('fundcode')['mv_label','g_v_label'].diff()
# ''' 仅保留相关系数大于0.7的条目 '''
# corr_tbl = corr_tbl.query("max_val > 0.7").reset_index()
#%%
''' 保存基金日收益率 '''
fundnav_.to_pickle("%s/fundnav_.pkl"%res_dir)
fundret_.to_pickle("%s/fundret_.pkl"%res_dir)
''' 保存 '''
# ff3_indexs.to_pickle("%s/ff3_indexs.pkl"%res_dir)
# reg4_indexs.to_pickle("%s/reg4_indexs.pkl"%res_dir)
corr_tbl.to_pickle("%s/corr_tbl.pkl"%res_dir)
