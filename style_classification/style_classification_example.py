#%%
'''
对基金进行分类的例子，所需要的数据为：stk_factor_、fund_portf_、corr_tbl_、fund_range_
stk_factor_：为股票的每季度的因子值 所必须的表段为：['stockcode','STK_NAME','selectdt','log_mv','g_v']
fund_portf_：为基金的每个季度的持仓 所必须的表段为：['fundcode','seasndt','stockcode','selectdt','proportion_adj','reveal_rate']
corr_tbl_：为基金每个季度风格偏移的方向 所必须的表段为：[c_mv_change, c_g_v_change]
fund_range_: 为选中的基金列表 pd.DataFrame [fundcode]
'''
# 导入包
from classifier import fund_classifier
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
seasn_dts = pd.read_pickle("%s/seasn_dts_df.pkl"%data_dir).seasndt.unique() #导入seasndt 时间节点
#%%
# 筛选符合条件的基金
# 选取范围：'普通股票型基金', '偏股混合型基金','灵活配置型基金'；成立时间：早于20200930；结束时间：晚于20201231 且为初始基金
fund_range_ = get_valid(fundtype = ['普通股票型基金', '偏股混合型基金','灵活配置型基金'], entrydate = '20200930', enddate = '20201231', data_dir = "../raw_data")
fund_range_['INDUSTRIESNAME'].value_counts().plot.pie(autopct='%.2f%%')
# 导入股票每个季度的因子值
stk_factor_ = pd.read_pickle("%s/stkfactor_f.pkl"%data_dir)
# 导入每个季度的基金持仓
fund_portf_ = pd.read_pickle("%s/fund_portf.pkl"%data_dir)
fund_portf_ = fund_portf_.query("(seasndt > '20110101') & (seasndt < '20210101')").copy()# 选取分析的时间片段
# 相关系数表
corr_tbl_ = pd.read_pickle("%s/corr_tbl.pkl"%data_dir)
#%%
clsf = fund_classifier(stk_factor_, fund_portf_, corr_tbl_, fund_range_)
#%%
# 保存分类结果和基金因子
clsf.classification_result.to_pickle("%s/classification_result.pkl"%res_dir)
clsf.fund_factor.to_pickle("%s/fund_factor.pkl"%data_dir)
