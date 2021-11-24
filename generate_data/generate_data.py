#%%
import os
import pandas as pd
from utils import *
from excuters import *
# %%
# 基础设置
data_dir = '../raw_data' #数据文件夹路径
res_dir = '../data'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
pd.set_option('display.max_columns', None)
#%%
''' 导入股票因子原始数据 '''
# 导入每日数据
stk_daily_factor = pd.read_pickle("%s/stkdailyfactor.pkl"%data_dir).query("anndt != 'NaN'").sort_values(['stockcode','tradedt'])
# 添加股票简称
stk_name_map = pd.read_excel("%s/stk_map.xlsx"%data_dir).rename(columns= {'WindCodes':'stockcode','SEC_NAME':'STK_NAME'})
# merge
stk_daily_factor = pd.merge(stk_daily_factor, stk_name_map, on = 'stockcode')
#%%
''' 处理基金持仓数据 '''
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

# 添加基金名称
fnd_dscr = pd.read_pickle("%s/CHINAMUTUALFUNDDESCRIPTION.pkl"%data_dir)[['F_INFO_WINDCODE','F_INFO_NAME']].rename(columns = {'F_INFO_WINDCODE':'fundcode','F_INFO_NAME':'fundname'})
fund_portf_ = pd.merge(fund_portf_, fnd_dscr, on = 'fundcode')
''' 导入每个季度的时间点 '''
seasn_dts_df = pd.read_pickle("%s/seasn_dts_df.pkl"%res_dir)
''' 仅选取季度节点作为披露日期的基金 '''
fund_portf_ = pd.merge(fund_portf_, seasn_dts_df, on =['seasndt'])
fund_portf_ = pd.merge(fund_portf_, fund_asset_, on = ['fundcode','seasndt'])
# seasn_dts_df.to_pickle("%s/seasn_dts_df.pkl"%res_dir)

#%%
''' 导入基金的净值 '''
fundnav = pd.read_pickle("%s/CHINAMUTUALFUNDNAV.pkl"%data_dir)
''' 选取基金代码fundcode 交易日tradedt 净值nav '''
fundnav_ = fundnav[['F_INFO_WINDCODE','PRICE_DATE', 'F_NAV_ADJUSTED']].sort_values('PRICE_DATE').rename(columns = {'F_INFO_WINDCODE':'fundcode','PRICE_DATE':'tradedt','F_NAV_ADJUSTED':'nav'})
fundnav_ = fundnav_.query("tradedt > '20100101'")
fundnav_supply = pd.read_pickle("chinamutualfundnav.pkl").query("PRICE_DATE > '20210411'")
fundnav_supply = fundnav_supply[['F_INFO_WINDCODE','PRICE_DATE', 'F_NAV_ADJUSTED']].sort_values('PRICE_DATE').rename(columns = {'F_INFO_WINDCODE':'fundcode','PRICE_DATE':'tradedt','F_NAV_ADJUSTED':'nav'})
fundnav_ = pd.concat([fundnav_, fundnav_supply], axis = 0)

''' 市场指数数据 '''
mkt_index = pd.read_pickle("%s/marketindex.pkl"%data_dir)
mkt_index_ = mkt_index[['TRADE_DT','S_DQ_CLOSE']]
''' 风格指数数据 '''
style_index = pd.read_pickle("%s/styleindex.pkl"%data_dir)
style_index_ = style_index.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_CLOSE')
'''所有指数汇总 并按照TRADE_DT排序'''
indexs = pd.merge(mkt_index_, style_index_, on='TRADE_DT').sort_values('TRADE_DT').reset_index(drop= True)
indexs.rename(columns={'TRADE_DT':'tradedt','S_DQ_CLOSE':'mkt','399314.SZ':'big','399316.SZ':'sml','399371.SZ':'val','399370.SZ':'grw'}, inplace=True)
indexs['mkt'] = indexs['mkt']/indexs['mkt'].iloc[0]
indexs['smb'] = indexs['sml']/indexs['big']
indexs['hml'] = indexs['val']/indexs['grw']
#%%
stk_excuter = stkfactor_exc(stk_daily_factor)
seasn_dts_df = stk_excuter.seasn_dts_df
stk_factor = stk_excuter.stk_factor
stk_factor_f = stk_excuter.stk_factor_f
#%%
fund_excuter = fundport_exc(fund_portf_)
fund_portf = fund_excuter.fund_portf
#%%
nav_excuter = nav_exc(fundnav_, indexs)
fundret_ = nav_excuter.fundret_
ff3_indexs = nav_excuter.ff3_indexs
reg4_indexs = nav_excuter.reg4_indexs
corr_tbl = nav_excuter.corr_tbl
#%%
# 保存数据
seasn_dts_df.to_pickle("%s/seasn_dts_df.pkl"%res_dir)
stk_factor.to_pickle("%s/stkfactor.pkl"%res_dir)
stk_factor_f.to_pickle("%s/stkfactor_f.pkl"%res_dir)
fund_portf.to_pickle("%s/fund_portf.pkl"%res_dir)
ff3_indexs.to_pickle("%s/ff3_indexs.pkl"%res_dir)
reg4_indexs.to_pickle("%s/reg4_indexs.pkl"%res_dir)
corr_tbl.to_pickle("%s/corr_tbl.pkl"%res_dir)

#%%

#%%
# ''' 处理每日基金因子数据 '''
# os.system("python ./exc_stk_daily_factor.py")
# print("stock factor finished")

# ''' 处理季度基金 '''
# os.system("python ./exc_fund_port.py")
# print("fund holding finished")

# ''' 处理基金和指数的净值 '''
# os.system("python ./exc_nav.py")
# print("nav finished")

# ''' 计算基金调仓能力 '''
# os.system("python ./exc_swith_ability.py")
# print("dev finished")

