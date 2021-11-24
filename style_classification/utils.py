import pandas as pd 
import os
import time
import numpy as np
from sklearn import preprocessing
from statsmodels import robust
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.formula.api import ols #加载ols模型
import matplotlib
import sys
import matplotlib.colors as mcolors

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
        # print(c)
        # print(m - 3*mad_c, m, m + 3*mad_c)
        # print(m - 3.5*mad_c, m, m + 3.5*mad_c)
        # print(3*mad_c)
        ''' 将在3mad之外的全部拉回到3.5mad内'''
        b3i = data[data[c] > 3*mad_c].index
        l3i = data[data[c] < -3*mad_c].index
        data.loc[b3i, c] = (data.loc[b3i, c] - 3*mad_c)*((0.5*mad_c)/((data[c].max()) - 3*mad_c)) + 3*mad_c
        data.loc[l3i, c] = (data.loc[l3i, c] + 3*mad_c)*((0.5*mad_c)/(-3*mad_c - data[c].min())) - 3*mad_c
        data[c] += m
    return data

''' 用于计算z_score的函数'''
def z_score(data:pd.DataFrame, columns:list):
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

''' 用于筛选符合条件的基金 '''
def get_valid(fundtype:list = ['普通股票型基金', '偏股混合型基金','灵活配置型基金'], entrydate:str = '20200930', enddate:str = '20201231', data_dir = "../raw_data"):
    '''
    input:
            fundtype: 筛选的基金类型
            entrydate: 基金的进入时间
            enddate: 基金的结束时间
            data_dir: 数据的读取路径
    return:
            valid_fund_:pd.DataFrame [fundcode, INDUSTRIESNAME]
    '''
    '''设置筛选条件'''
    ''' 筛选符合条件的基金（基金类型，成立时间，结束时间）'''
    selector = pd.read_pickle("%s/CHINAMUTUALFUNDSECTOR.pkl"%data_dir)
    selector['S_INFO_SECTOR'] = selector['S_INFO_SECTOR'].str[0:10]
    industry_sel = pd.read_pickle("%s/ASHAREINDUSTRIESCODE.pkl"%data_dir)
    industry_sel['INDUSTRIESCODE'] = industry_sel['INDUSTRIESCODE'].str[0:10]
    fund_describe = pd.read_pickle("%s/CHINAMUTUALFUNDDESCRIPTION.pkl"%data_dir) 
    valid_fund = pd.merge(selector[['F_INFO_WINDCODE','S_INFO_SECTOR','S_INFO_SECTORENTRYDT','S_INFO_SECTOREXITDT']], industry_sel[['INDUSTRIESCODE','INDUSTRIESNAME']], left_on = 'S_INFO_SECTOR', right_on= 'INDUSTRIESCODE')
    valid_fund = pd.merge(valid_fund, fund_describe[['F_INFO_WINDCODE','F_INFO_ISINITIAL']], on = 'F_INFO_WINDCODE')
    if1 = "(INDUSTRIESNAME in %s)"%str(fundtype)
    if2 = "(S_INFO_SECTORENTRYDT <= '%s')"%entrydate
    #结束时间在多久之后，或者结束时间为空（空的值自身不等于自身）
    if3 = "((S_INFO_SECTOREXITDT >= '%s') or (S_INFO_SECTOREXITDT != S_INFO_SECTOREXITDT))"%enddate
    # 剔除非初始基金
    if4 = "(F_INFO_ISINITIAL == 1)"
    valid_fund_ = valid_fund.query("&".join([if1, if2, if3, if4]))[['F_INFO_WINDCODE','INDUSTRIESNAME']].rename(columns= {'F_INFO_WINDCODE':'fundcode'}).copy()
    #剔除含港股的股票
    hk_fund = valid_fund.query("INDUSTRIESNAME == '陆港通基金'")
    valid_fund_ = valid_fund_[~valid_fund_['fundcode'].isin(hk_fund['F_INFO_WINDCODE'])]
    #只选取初始基金

    return valid_fund_

# def get_windstyle():



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



def cal_fund_factor(fund_stk_factor_:pd.DataFrame):
    '''
    计算市场所有基金在每个截面上的因子值
    input:
            fund_stk_factor_ [fundcode, seasndt, stockcode, proportion, anndt, fundname, selectdt, reveal_rate, proportion_adj, log_mv, g_v]
    return:
            fund_factor:pd.DataFrame []
    '''
    ''' fund_factor 表段计算好每只基金在某个时间节点上的最终披露风格 将所有股票加权之后的贡献相加（成长、价值、市值）'''
    fund_stk_factor_['log_mv_p'] = fund_stk_factor_.log_mv* fund_stk_factor_.proportion_adj
    fund_stk_factor_['g_v_p'] = fund_stk_factor_.g_v* fund_stk_factor_.proportion_adj
    # 每个基金在每个截面上加权求和
    fund_factor = fund_stk_factor_.groupby(by = ['fundcode','fundname','seasndt','selectdt'])[['log_mv_p','g_v_p']].sum().reset_index()
    ''' 截面上 MAD + z_score'''
    fund_factor = fund_factor.groupby('selectdt').apply(MAD3, ['log_mv_p','g_v_p'])
    fund_factor = fund_factor.groupby('selectdt').apply(z_score, ['log_mv_p','g_v_p'])
    fund_factor.rename(columns = {'log_mv_p':'mv','g_v_p':'g_v'},inplace = True)
    # 计算风格的排名
    fund_factor['mv_rank'] = fund_factor.groupby('seasndt').mv.rank(pct = True)
    fund_factor['g_v_rank'] = fund_factor.groupby('seasndt').g_v.rank(pct = True)
    # 添加仓位
    fund_factor = pd.merge(fund_factor, fund_stk_factor_[['fundcode','seasndt','reveal_rate']].drop_duplicates(), on = ['fundcode','seasndt'])
    padidx = fund_factor[(fund_factor.seasndt.str[-4:].isin(['0331','0930'])) & (fund_factor.reveal_rate < 0.4)].index
    adjidx = fund_factor[(fund_factor.seasndt.str[-4:].isin(['0331','0930'])) & (fund_factor.reveal_rate > 0.4)].index
    # 如果是全持仓则正常计算
    fund_factor['flag'] = 'normal'
    # 如果是前十大重仓且披露小于40则用前一个全持仓进行填充
    fund_factor.loc[padidx, 'flag'] = 'pad'
    # 如果是前十大重仓且披露大于40则进行动态调整
    fund_factor.loc[adjidx, 'flag'] = 'adj'
    return fund_factor

def cal_cluster_thrs(corr_tbl:pd.DataFrame):
    '''
    input:
            corr_tbl: 含有基金 [fundcode, selectdt, sml_ret, big_ret, val_ret, grw_ret, max_idx, max_val, fundname, seasndt, mv, g_v]
    return:
            cluster_thrs_ 每个季度的6个分类的阈值: [selectdt, big_thr, sml_thr, grw_thr, val_thr, mid_thr, bal_thr]
    '''
    ''' 每一期在每一类中计算中位数作为聚类中心 '''
    cluster_thrs = corr_tbl.groupby(['selectdt','max_idx'])[['mv','g_v']].median()
    cluster_thrs = cluster_thrs.reset_index().pivot(index='selectdt', columns='max_idx', values=['mv','g_v'])
    cluster_thrs['big_thr'] = cluster_thrs['mv']['big_ret']
    cluster_thrs['sml_thr'] = cluster_thrs['mv']['sml_ret']
    cluster_thrs['grw_thr'] = cluster_thrs['g_v']['grw_ret']
    cluster_thrs['val_thr'] = cluster_thrs['g_v']['val_ret']
    cluster_thrs_ = cluster_thrs[['big_thr','sml_thr','grw_thr','val_thr']]
    cluster_thrs_.columns = cluster_thrs_.columns.droplevel(1)
    cluster_thrs_.fillna(method = 'pad', inplace=True)
    cluster_thrs_.dropna(inplace=True)
    ''' 取两个阈值的中间点作为中间类别的聚类中心 '''
    cluster_thrs_['mid_thr'] = (cluster_thrs_.big_thr + cluster_thrs_.sml_thr)/2
    cluster_thrs_['bal_thr'] = (cluster_thrs_.grw_thr + cluster_thrs_.val_thr)/2
    return cluster_thrs_
    
def cal_train_center(cluster_thrs_:pd.DataFrame):
    '''
    input:
            cluster_thrs_ [selectdt, big_thr, sml_thr, grw_thr, val_thr, mid_thr, bal_thr]
    return:
            train_c:pd.DataFrame [selectdt, mv, label, g_v]
    '''
    mv_t = cluster_thrs_[['sml_thr','mid_thr','big_thr']].T
    mv_t['label'] = 0
    g_v_t = cluster_thrs_[['val_thr','bal_thr','grw_thr']].T
    g_v_t['label'] = 0
    train_c = pd.merge(mv_t, g_v_t, on = 'label')
    train_c.columns = ['mv','label','g_v']
    # print(train)
    train_c.label = [0,1,2,3,4,5,6,7,8]
    return train_c

def knn_byreg(data:pd.DataFrame, train_c:pd.DataFrame):
    '''
    使用knn的聚类函数，对每个季度的基金需要apply一下
    input:
            data [fundcode, mv, g_v]
    return:
            data [fundcode, mv, g_v, label]
    '''
    t = data.selectdt.unique()[0]
    X_train = train_c.loc['%s'%t][['mv','g_v']]
    Y_train = train_c.loc['%s'%t][['label']]
    model =KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, Y_train)
    data.label = model.predict(data[['mv','g_v']])
    return data

def classification(fund_factor:pd.DataFrame, method_:int, corr_tbl:pd.DataFrame, cluster_thrs_:pd.DataFrame):
    ''' 
    对基金进行分类 返回基金分类结果
    input: 
            fund_factor: [fundcode, fundname, seasndt, selectdt, mv, g_v, mv_rank, g_v_rank, tot_reveal, flag]
            method_: 指定分类方法，method_ == 0 knn，method_ == 1 简单阈值法
            corr_tbl：相关系数的变化趋势表段
            # train_c: 每个季度有9个聚类中心 [selectdt, mv, label, g_v]
            cluster_thrs_: 每个季度的6个指标的阈值 [selectdt, big_thr, sml_thr, grw_thr, val_thr, mid_thr, bal_thr]
    return : 
            inte_tbl:pd.DataFrame 每个基金在每个季度的市值暴露，成长价值暴露，分类类别等信息 [fundcode, fundname, seasndt, selectdt, mv, g_v, label, category, my_style, g_v_style]
    '''
    # knn方法
    if(method_ == 0):
        # 计算聚类中心
        train_c = cluster_thrs_.groupby('selectdt').apply(cal_train_center)
        inte_tbl = fund_factor.query("seasndt > '20110101'")
        inte_tbl['label'] = 0
        inte_tbl = inte_tbl.groupby('seasndt',as_index = False).apply(knn_byreg, train_c)
        ''' 设置风格对应表 '''
        style_dict = {0:'小盘价值',1:'小盘均衡',2:'小盘成长',3:'中盘价值',4:'中盘均衡',5:'中盘成长',6:'大盘价值',7:'大盘均衡',8:'大盘成长'}
        ''' inte_tbl 记录每一只基金在某一季度时间点上的所有信息 '''
        inte_tbl['category'] = inte_tbl.label.map(style_dict)
        inte_tbl['mv_style'] = inte_tbl['category'].str[0:2]
        inte_tbl['g_v_style'] = inte_tbl['category'].str[2:4]
    # 简单阈值法
    if(method_ == 1):
        def cal_style(data:pd.DataFrame):
            if(data.mv<data.sml_thr):
                data['mv_style'] = '小盘'
            if(data.mv>data.big_thr):
                data['mv_style'] = '大盘'
            if(data.g_v<data.val_thr):
                data['g_v_style'] = '价值'
            if(data.g_v>data.grw_thr):
                data['g_v_style'] = '成长'
            data['category'] = data['mv_style'] + data['g_v_style']
            return data
        inte_tbl = fund_factor.query("seasndt > '20110101'")
        inte_tbl = pd.merge(inte_tbl, cluster_thrs_[['sml_thr','big_thr','val_thr','grw_thr']].reset_index(), on = 'selectdt')
        inte_tbl['mv_style'] = '中盘'
        inte_tbl['g_v_style'] = '均衡'
        inte_tbl['category'] = 0
        ''' 设置风格对应表 '''
        style_dict = {'小盘价值':0,'小盘均衡':1,'小盘成长':2,'中盘价值':3,'中盘均衡':4,'中盘成长':5,'大盘价值':6,'大盘均衡':7,'大盘成长':8}
        inte_tbl = inte_tbl.apply(cal_style ,axis=1)
        inte_tbl['label'] = inte_tbl.category.map(style_dict)
    
    # 修正
    # 需要用前一个全持仓数据进行填充的index
    padidx = inte_tbl.query("flag == 'pad'").index
    # 先把需要填充的季度时间点数据全部清空 然后用前一个该基金前一个季度的数据填充
    inte_tbl.loc[padidx, ['label','category','mv_style','g_v_style']] = np.nan
    inte_tbl = inte_tbl.groupby(['fundcode']).apply(pd.DataFrame.fillna, method = 'pad')
    mv_dict = {'小盘':-1,'中盘':0,'大盘':1}
    g_v_dict = {'价值':-1,'均衡':0,'成长':1}
    inte_tbl['mv_label'] = inte_tbl.mv_style.map(mv_dict)
    inte_tbl['g_v_label'] = inte_tbl.g_v_style.map(g_v_dict)
    inte_tbl.dropna(inplace = True)
    # 计算市值和g_v风格两期之间的变化
    inte_tbl[['mv_change','g_v_change']] = inte_tbl.groupby('fundcode')['mv_label','g_v_label'].diff()

    inte_tbl = pd.merge(inte_tbl, corr_tbl[['fundcode','seasndt','c_mv_change','c_g_v_change']], on = ['fundcode','seasndt'])
    # 如果两个变化趋势相同，则本期的pred为正，pred为负或者pred为0则不进行调整
    inte_tbl['mv_pred'] = inte_tbl.mv_change * inte_tbl.c_mv_change
    inte_tbl['g_v_pred'] = inte_tbl.g_v_change * inte_tbl.c_g_v_change
    # 如果没有变化（pred ！= 1），则用上一期的进行填充
    mv_remidx = inte_tbl.query("(mv_pred <= 0) & (flag == 'adj')").index
    g_v_remidx = inte_tbl.query("(g_v_pred <= 0) & (flag == 'adj')").index
    # 先填Nan，然后用该基金的前一期进行填充
    inte_tbl.loc[mv_remidx, ['mv_style']] = np.nan
    inte_tbl = inte_tbl.groupby(['fundcode']).apply(pd.DataFrame.fillna, method = 'pad')
    inte_tbl.loc[g_v_remidx, ['g_v_style']] = np.nan
    inte_tbl = inte_tbl.groupby(['fundcode']).apply(pd.DataFrame.fillna, method = 'pad')
    # 合并市值和成长价值风格
    inte_tbl.category = inte_tbl.mv_style + inte_tbl.g_v_style
    # 计算风格漂移的基金（rolling 4期 计算mv_rank的样本标准差、g_v_rank的样本标准差）
    inte_tbl['mv_shift'] = inte_tbl.groupby('fundcode').mv_rank.rolling(4).std().values
    inte_tbl['g_v_shift'] = inte_tbl.groupby('fundcode').g_v_rank.rolling(4).std().values
    # 最开始几期视为没有漂移 用0填充
    inte_tbl.fillna(value = 0, inplace = True)
    # 计算漂移系数在截面上的排名
    inte_tbl['mv_shift_rank'] = inte_tbl.groupby('seasndt').mv_shift.rank(pct =True)
    inte_tbl['g_v_shift_rank'] = inte_tbl.groupby('seasndt').g_v_shift.rank(pct =True)
    inte_tbl['shift_rate'] = inte_tbl.mv_shift + inte_tbl.g_v_shift
    inte_tbl['shift_rank'] = inte_tbl.groupby('seasndt').shift_rate.rank(pct = True)
    # 如果当期漂移系数百分位在前5% 则视为风格漂移基金
    shift_idx = inte_tbl.query("shift_rank > 0.95").index
    inte_tbl.loc[shift_idx, 'category'] = '风格漂移'
    return inte_tbl

# def reg_FF3(data, yvar,	xvars):
#     Y =	data[yvar]
#     X =	data[xvars]
#     X['intercept']	=	1.
#     result = sm.OLS(Y, X).fit() # 计算回归系数
#     res = result.params
#     res['R2'] = result.rsquared # 添加R方
#     return res 

def strategy_performance(portfolio:pd.DataFrame, fund_ret_:pd.DataFrame, port_name:str='nav'):
    '''
    查看策略的走势
    input:
            portfolio: pd.DataFrame [fundcode, seasndt, selectdt] 在某一个季度selectdt截面上的持有的基金
            fund_ret_: pd.DataFrame [fundcode, tradedt, selectdt, ret] 基金收益率表段
            port_name: str 策略的名称
    return: 
            port_anly [tradedt, ret, selectdt, nav, DateTime, hs300, dt, rate, rf, mkt]
    '''
    # ''' 每个季度选取隐形交易因子值最小的几个 '''
    # portfolio = fund_dev.sort_values('%s'%factor).groupby(['selectdt','label']).head(num_per_style)
    ''' 添加 '''
    portfolio_ = pd.merge(portfolio[['fundcode','fundname','seasndt','selectdt','category']], fund_ret_, on = ['fundcode','selectdt'])
    ''' 每一天计算持有股票的收益率平均值 '''
    port_behav = portfolio_.groupby('tradedt').ret.mean().to_frame().reset_index()
    port_behav = pd.merge(port_behav, portfolio_[['tradedt', 'selectdt']].drop_duplicates(), on = 'tradedt')
    port_behav['ret'] = port_behav.ret + 1
    ''' 需要计算手续费 '''
    ''' 按照千五 万2进行计算 '''
    fee = port_behav.groupby('selectdt').head(1) #需要计算手续费的节点，每个selectdt的第一天
    ''' 每个selectdt 计算手续费 '''
    port_behav.loc[fee.index, 'ret'] *= 0.995 #卖出
    port_behav.loc[fee.index, 'ret'] /= 1.0002 #买入
    port_behav['nav'] = port_behav.ret.cumprod() #计算累积净值
    port_behav['date'] = pd.to_datetime(port_behav['tradedt'], format = '%Y%m%d')
    port_behav.nav /= port_behav.nav.loc[0]
    port_behav.rename(columns={'nav':'%s'%port_name}, inplace=True)
    return port_behav

def evaluation(portfolio:pd.DataFrame, rf:int):
    '''
    评价构建的策略
    input: 
            portfolio [tradedt, ret, selectdt, nav, DateTime, hs300, dt, rate, rf, mkt]
            rf 无风险收益率
    return: 
            maxmum_drawdown(最大回撤), start_row(最大回撤的开始行), end_row(最大回撤结束行)
            port_return(期间收益), index_return(对比指数期间收益), return_excess(期间超额收益)
            ann_excess_return(年化超额收益), ann_tracking_error(年化跟踪误差)
            information_ratio(信息比率), sharp_ratio(夏普比)
    '''
    portfolio['port_yield'] = portfolio['nav'].pct_change()
    portfolio['index_yeild'] = portfolio['mkt'].pct_change()
    portfolio['excess_yeild'] = portfolio['index_yeild'] - portfolio['port_yield'] 

    port_nav = portfolio['nav'].values
    index_nav = portfolio['mkt'].values
    ''' 夏普比 '''
    sharp_ratio = (portfolio.port_yield.mean() - portfolio.rf.mean())*np.sqrt(244)/portfolio.port_yield.std()
    '''最大回撤率'''
    end_row = np.argmax((np.maximum.accumulate(port_nav) - port_nav) / np.maximum.accumulate(port_nav))  # 结束位置
    if end_row == 0:
        maxmum_drawdown = 0
    start_row = np.argmax(port_nav[:end_row])  # 开始位置
    maxmum_drawdown = (port_nav[start_row] - port_nav[end_row]) / (port_nav[start_row])
    '''策略期间收益'''
    port_return = (port_nav[-1] - port_nav[0])/port_nav[0]
    '''指数期间收益'''
    index_return = (index_nav[-1] - index_nav[0])/index_nav[0]
    '''超额收益'''
    return_excess = port_return - index_return
    '''年化超额收益'''
    ann_port_return = np.power(port_return + 1, 244/len(port_nav)) - 1 #策略年化收益
    ann_index_return = np.power(index_return + 1, 244/len(port_nav)) - 1 #指数年化收益
    print('策略年化收益： %f, 指数年化收益：%f'%(ann_port_return, ann_index_return))
    ann_excess_return = ann_port_return - ann_index_return #年化超额收益
    '''年化跟踪误差'''
    ann_tracking_error = portfolio['excess_yeild'].std() * np.sqrt(244)
    '''信息比率'''
    information_ratio = ann_excess_return/ann_tracking_error
    print('最大回撤：%f 最大回撤开始：%s 最大回撤结束：%s'%\
        (maxmum_drawdown, portfolio.tradedt.iloc[start_row], portfolio.tradedt.iloc[end_row]))
    print('策略期间收益：%f 指数期间收益：%f'%(port_return,index_return))
    print('超额收益：%f 年化超额收益：%f 年化跟踪误差：%f 信息比率：%f 夏普比：%f'%\
        (return_excess,ann_excess_return,ann_tracking_error,information_ratio,sharp_ratio))

    return maxmum_drawdown, start_row, end_row, port_return, index_return, return_excess, ann_excess_return, ann_tracking_error, information_ratio, sharp_ratio


def plot_strategy(port_anly:pd.DataFrame, start_row:int, end_row:int, res_dir:str):
    ''' 
    将策略的走势画出来
    input:
            port_anly:  [tradedt, ret, selectdt, nav, DateTime, hs300, dt, rate, rf, mkt]
            res_dir: 保存文件夹目录
    return:
            None
    '''
    startdt = port_anly['dt'].iloc[start_row]
    enddt = port_anly['dt'].iloc[end_row]
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    # ax2画组合净值和对照指数净值
    ax2.plot(port_anly['dt'], port_anly['nav'], label = u'port', c = 'black')
    ax2.plot(port_anly['dt'], port_anly['hs300'], label = u'hs300', c = 'red')
    # 标注最大回撤开始点
    ax2.annotate("$maxmum drawdown\ start$",
    xy=(startdt,port_anly.nav.iloc[start_row]),xycoords='data',xytext=(startdt,port_anly.nav.iloc[start_row]+1),arrowprops=dict(facecolor = "black", arrowstyle="simple"))
    # 标注最大回撤结束点
    ax2.annotate("$maxmum drawdown\ end$",
    xy=(enddt,port_anly.nav.iloc[end_row]),xycoords='data',xytext=(enddt,port_anly.nav.iloc[end_row]+1),arrowprops=dict(facecolor = "black", arrowstyle="simple"))
    ax2.set_ylim(0.5, 4)
    ax2.set_xlabel("tradedt")
    ax2.set_ylabel("nav")
    ax2.legend()

    # ax画组合与参照指数的比例的阴影区域
    ax.fill_between(port_anly['dt'], 0, port_anly['rate'], facecolor = 'pink', label = u'rate')
    ax.set_ylabel("rate")
    ax.set_ylim(0.8, 2.5)
    ax.legend()

    plt.title('portfolio behavior vs hs300')
    ''' 保存图像 '''
    f = plt.gcf()  #获取当前图像
    f.savefig("%s/portfolio behavior vs mkt.jpg"%res_dir)
    plt.show()
    f.clear()  #释放内存

def cal_RankIC():
    return 

# ''' 中性化股票因子 '''

# ''' 是否对原始因子进行中性化 '''
# ''' 导入行业数据 '''
# if(n_flag == 1):
#     ''' 进行行业中性化 '''
#     ''' 定义行业中性化函数 '''
#     def neutral(data:pd.DataFrame, yvar):
#         Y =	data[yvar]
#         X =	data.iloc[:,4:]
#         # X['intercept']	=	1.
#         result = sm.OLS(Y, X).fit() # 计算回归系数
#         return Y -result.fittedvalues # 返回残差

#     industry = pd.read_pickle("%s/industry.pkl"%data_dir)
#     industry.seasndt = industry.seasndt.apply(str)
#     stk_factor_ = pd.merge(stk_factor_, industry, on = ['S_INFO_STOCKWINDCODE','seasndt'])
#     stk_factor_ = stk_factor_.join(pd.get_dummies(stk_factor_.INDUSTRY_CITIC))
#     stk_factor_.drop('INDUSTRY_CITIC',axis=1,inplace=True)
#     stk_factor_neutral = stk_factor_[['S_INFO_STOCKWINDCODE','seasndt','log_mv']].copy()
#     ''' 中性化之后的表段'''
#     stk_by_seasn = stk_factor_.groupby(['S_INFO_STOCKWINDCODE','seasndt'])
#     ''' 回归做中性化 要运行很久'''
#     ''' 价值因子 进行中性化'''
#     stk_factor_neutral['val_n'] = stk_by_seasn.apply(neutral,'value').values
#     ''' z_score 标准化 '''
#     stk_factor_neutral.val_n = stk_factor_neutral.groupby('seasndt').val_n.apply(z_score)
#     ''' 成长因子 进行中性化'''
#     stk_factor_neutral['grw_n'] = stk_by_seasn.apply(neutral,'growth').values
#     ''' z_score 标准化'''
#     stk_factor_neutral.grw_n = stk_factor_neutral.groupby('seasndt').grw_n.apply(z_score)
