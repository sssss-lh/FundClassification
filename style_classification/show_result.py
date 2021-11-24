#%%
from utils import *
#%%
# 基本参数设置
pd.set_option('display.max_columns', None)# pandas显示列数
pd.set_option('display.max_rows', 30)# pandas显示行数
sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']}) # 设置sns的输出格式
matplotlib.rcParams['axes.unicode_minus'] =False #正常显示负号
res_dir = '../result/style_classification' #存储文件夹路径
data_dir = '../data' #数据文件夹路径
if not os.path.exists(res_dir):
    os.makedirs(res_dir) #如果目录不存在则创建文件夹
#%%
# 导入数据
classification_result = pd.read_pickle("%s/classification_result.pkl"%res_dir)
fundnav = pd.read_pickle("%s/fundnav.pkl"%data_dir)
fundret = pd.read_pickle("%s/fundret.pkl"%data_dir)
corr_tbl = pd.read_pickle("%s/corr_tbl.pkl"%data_dir)
fund_range = get_valid(fundtype = ['普通股票型基金', '偏股混合型基金','灵活配置型基金'], entrydate = '20200930', enddate = '20201231', data_dir = "../raw_data")
fund_range['INDUSTRIESNAME'].value_counts().plot.pie(autopct='%.2f%%')
# 导入股票每个季度的因子值
stk_factor = pd.read_pickle("%s/stkfactor_f.pkl"%data_dir)
# 导入每个季度的基金持仓
fund_portf = pd.read_pickle("%s/fund_portf.pkl"%data_dir)
fund_portf = fund_portf.query("(seasndt > '20110101') & (seasndt < '20210101')").copy()# 选取分析的时间片段
# 相关系数表
corr_tbl = pd.read_pickle("%s/corr_tbl.pkl"%data_dir)
# 基金因子表
fund_factor = pd.read_pickle("%s/fund_factor.pkl"%data_dir)
#%%
# 选取过去一年的收益率因子
fund_year_ret = fundnav.sort_values('tradedt').copy()
fund_year_ret = pd.merge(fund_year_ret, fund_range, on = 'fundcode')
# 滚动计算一年的收益率
fund_year_ret['year_ret'] = fund_year_ret.groupby('fundcode').nav.pct_change(244)
fund_year_ret = fund_year_ret.sort_values('tradedt').groupby(['fundcode','selectdt']).tail(1)
fund_year_ret = pd.merge(fund_year_ret, classification_result[['fundcode','selectdt','seasndt','fundname','category','mv_style','g_v_style']], on = ['fundcode','selectdt']).sort_values('year_ret')
#%%
select_dts = fundret.selectdt.unique()
fundret_ = fundret.query("selectdt > '20110101'")
''' 此时selectdt 为大于该交易日的第一个selectdt '''
fundret_['selectdt'] =  fundret_['tradedt'].apply(lambda x : select_dts[select_dts < '%s'%x][-1])
#%%
# 在不同风格中构建策略
fund_year_ret.dropna(inplace= True)
# 每个风格选五个
strategy_by_style = fund_year_ret.groupby(['selectdt','category']).tail(5)
# 不区分风格
strategy_simp = fund_year_ret.groupby('selectdt').tail(45)
# 选所有为价值风格的基金
strategy_val = fund_year_ret.query("g_v_style == '价值'").groupby(['selectdt']).tail(45)
# 选所有为成长风格的基金
strategy_grw = fund_year_ret.query("g_v_style == '成长'").groupby(['selectdt']).tail(45)
# 选取风格漂移的基金
strategy_shift =  fund_year_ret.query("category == '风格漂移'").groupby(['selectdt']).tail(45)

#%%
behav_simp = strategy_performance(strategy_simp, fundret_, 'simp')
behav_style = strategy_performance(strategy_by_style, fundret_, 'style')
behav_val = strategy_performance(strategy_val, fundret_, 'val')
behav_grw = strategy_performance(strategy_grw, fundret_, 'grw')
behav_shift = strategy_performance(strategy_shift, fundret_, 'shift')
hs300 = pd.read_excel("%s/hs300.xlsx"%data_dir).rename(columns={'CLOSE':'hs300','DateTime':'date'})
hs300['tradedt'] = hs300['date'].apply(lambda x:x.strftime('%Y%m%d'))

#%%
mg = (behav_simp[['tradedt','date','selectdt','simp']].merge(behav_style[['tradedt','style']], on='tradedt')
            .merge(behav_val[['tradedt','val']], on='tradedt')
            .merge(behav_grw[['tradedt','grw']], on='tradedt')
            .merge(behav_shift[['tradedt','shift']], on='tradedt')
            .merge(hs300[['tradedt','hs300']], on = 'tradedt'))

mg[['simp','style','val','grw','hs300']] /= mg[['simp','style','val','grw','hs300']].loc[0]
mg.plot(x = 'date', y = ['simp','style','val','grw','shift','hs300'], figsize = (20,10))

# %%
''' 绘制图表 '''
# PART3.1 查看每个季度有多少只股票 
t = stk_factor.groupby('selectdt').stockcode.unique().to_frame().stockcode.apply(lambda x: x.shape[0])
t.plot()
#%%
# PART3.2 查看某一时间截面市场上股票的因子分布
t = stk_factor.query("selectdt == '20160901'")
t.hist(bins = 100, figsize = (20, 10))
t.describe()
# PART4.1 查看每个季度有多少只基金
t = fund_portf.groupby('selectdt').fundcode.unique().to_frame().fundcode.apply(lambda x: x.shape[0])
t.plot()
# PART4.3 查看每个季度每只基金的持仓数
# t = fund_stk_factor.groupby(['seasndt','fundcode']).count()[['stockcode']]
# t
#%%
fgp = fund_factor.groupby('seasndt')
''' PART5.1 查看每个季度有多少基金 '''
fgp.fundcode.unique().to_frame().fundcode.apply(lambda x: x.shape[0]).plot()
''' PART5.2 查看某一期基金截面的因子分布 '''
fgp.get_group('20180630').hist(bins = 100, figsize = (12,8))
#%%
''' PART6.1 查看某一截面上被划分出来相应类别的基金个数 '''
t = corr_tbl.query("(selectdt == '20180501')")
t['max_idx'].value_counts().plot.pie(autopct='%.2f%%', figsize = (6,6))
#%%
''' PART6.2 从时间维度看变化 '''
t = corr_tbl.groupby(['selectdt','max_idx']).count()[['fundcode']].reset_index()
t = t.pivot(index='selectdt', columns='max_idx', values='fundcode').fillna(method = 'pad')
t.plot(figsize = (20,8))
#%%
''' PART6.2 从时间维度看变化 （占比） '''
t = corr_tbl.groupby(['selectdt','max_idx']).count()[['fundcode']].reset_index()
t = t.pivot(index='selectdt', columns='max_idx', values='fundcode').fillna(method = 'pad').dropna()
t = t.apply(lambda x:x/x.sum(),axis =1)
t.plot(figsize = (20,8))
# PART7.4查看每个季度的基金个数
t = classification_result.groupby('seasndt').count().fundcode
t.plot()
#%%
# PART7.5 画某一时间截面市场上基金的风格分布饼状图
t = classification_result.query("seasndt == '20201231'").category.value_counts()
t.plot.pie(autopct='%.2f%%', figsize = (12,8))
#%%
# PART7.6 选取一个时间节点 画这个节点上的基金风格分布散点图
test_time = '20180630'
section = classification_result.query("seasndt == '%s'"%test_time)
y_pre = section.label
colors=list(mcolors.TABLEAU_COLORS.keys())
plt.figure(figsize=(10, 8))
plt.xlabel("Style Index")
plt.ylabel("Market Value Index")
# plt.title("%s knn"%(test_time))
for j in range(0,10):
    index_set = np.where(y_pre==j)
    cluster = section[['g_v','mv']].iloc[index_set]
    plt.scatter(cluster.iloc[:,0],cluster.iloc[:,1],c=mcolors.TABLEAU_COLORS[colors[j]],marker='.')  
    # plt.plot(centers[j][0],centers[j][1],'o',markerfacecolor=colors[j],markeredgecolor='k',markersize=8)  #画类别中心
f = plt.gcf()  #获取当前图像
f.savefig("%s/%sknn.jpg"%(res_dir,test_time))
plt.show()
plt.show()
f.clear()  #释放内存
#%%
''' PART7.7 查看基金风格数量随时间变化的曲线 '''
style_nums = classification_result.groupby(['seasndt','category']).count().fundname.to_frame().reset_index()
style_nums = style_nums.pivot(index='seasndt', columns='category', values='fundname')
style_nums.plot(figsize = (20,8))
style_nums = style_nums.apply(lambda x:x/x.sum(),axis =1)
style_nums.plot(figsize = (20,8))
#%%
''' PART7.7.1 查看基金市值风格数量随时间变化的曲线 '''
style_nums = classification_result.groupby(['seasndt','mv_style']).count().fundname.to_frame().reset_index()
style_nums = style_nums.pivot(index='seasndt', columns='mv_style', values='fundname')
style_nums.plot(figsize = (20,8))
style_nums = style_nums.apply(lambda x:x/x.sum(),axis =1)
style_nums.plot(figsize = (20,8))
#%%
''' PART7.7.2 查看基金价值成长风格数量随时间变化的曲线 '''
style_nums = classification_result.groupby(['seasndt','g_v_style']).count().fundname.to_frame().reset_index()
style_nums = style_nums.pivot(index='seasndt', columns='g_v_style', values='fundname')
style_nums.plot(figsize = (20,8))
style_nums = style_nums.apply(lambda x:x/x.sum(),axis =1)
style_nums.plot(figsize = (20,8))
#%%
# PART7.8 查看选中的基金的基金风格
# check_index = ['博时产业新动力A','南方新优享A','兴全趋势投资','中欧新蓝筹A',\
# '汇添富价值精选A','国泰价值经典','国富弹性市值','前海开源清洁能源A','安信价值精选',\
# '圆信永丰优加生活','交银新成长','中银收益A','大成互联网思维','新华鑫益','鹏兴新兴产业',\
# '华安策略精选','前海开源国家比较优势','易方达中小盘','诺安先进制造','东方红睿丰']
check_index = ['上投摩根科技前沿','华安逆向策略','招商制造业转型A','工银瑞信灵活配置A','华宝先进成长','交银经济新动力','光大新增长','银化心诚','嘉实泰和','富国高端制造行业']
check_fund = str(check_index)
sel_res = section.query("fundname in %s"%check_fund)
sel_res.set_index(sel_res.fundname, inplace=True)
sel_res.reindex(check_index).dropna()