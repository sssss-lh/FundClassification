#%%
from utils import *
#%%
class fund_classifier():
    def __init__(self, stk_factor:pd.DataFrame, fund_portf:pd.DataFrame, corr_tbl:pd.DataFrame, fund_range:pd.DataFrame, method:int = 0) -> None:
        '''
        初始类参数定义 及必需表段：
        input:
                stk_factor: pd.DataFrame ['stockcode','selectdt','log_mv','g_v']
                fund_portf: pd.DataFrame ['fundcode','seasndt','stockcode','selectdt','proportion_adj','reveal_rate']
                corr_tbl: pd.DataFrame [c_mv_change, c_g_v_change] 在市值上相比上一期变化的趋势，在g-v上相比上一期变化的趋势
                fund_range: pd.DataFrame [fundcode] 需要进行分析的基金列表
                method: int 1为knn分类 0为简单阈值法分类
        '''
        self.method = method
        self.stk_factor = stk_factor
        self.fund_range = fund_range
        self.fund_portf = pd.merge(fund_portf, fund_range[['fundcode']], on=['fundcode']) # 仅分析选中的基金
        self.fund_stk_factor = pd.merge(fund_portf, stk_factor[['stockcode','STK_NAME','selectdt','log_mv','g_v']], on = ['stockcode','selectdt'])# 得到每只基金的每个季度的暴露值
        self.sel_dt = self.get_select_dts()
        print("cal_fund_factor")
        self.fund_factor = self.cal_fund_factor(self.fund_stk_factor)# 基金因子计算
        print("import corr_tbl")
        self.corr_tbl = corr_tbl = pd.merge(corr_tbl, self.fund_factor[['fundcode','seasndt','selectdt','mv','g_v']], on = ['fundcode','selectdt'])
        print("cal_cluster_thrs")
        self.cluster_thrs = self.cal_cluster_thrs(self.corr_tbl)# 分类阈值
        print("classification")
        # 分类结果（complicated）
        self.classification_result_merge = self.classification(self.fund_factor, self.method, self.corr_tbl[['fundcode','selectdt','seasndt','c_mv_change','c_g_v_change']], self.cluster_thrs)
        # 分类结果（simplified）
        self.classification_result = self.classification_result_merge[['fundcode','fundname','seasndt','selectdt','mv','g_v','category','mv_style','g_v_style','label']]

    def get_select_dts(self, start:int=2009, end:int=2022):
        ''' 
        生成select_dt
        return：
            sel_dt: ndarray 所有的selectdt时间节点
        '''
        sel_dt = []
        for yr in range(end, start, -1):
            sel_dt.extend([str(yr) + '1101', str(yr) + '0901', str(yr) + '0501', str(yr) + '0401'])
        sel_dt = np.array(sel_dt)
        return sel_dt

    def cal_fund_factor(self, fund_stk_factor_:pd.DataFrame):
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

    def cal_cluster_thrs(self, corr_tbl:pd.DataFrame):
        '''
        通过相关系数表，计算每一期的聚类中心
        input:
                corr_tbl: 含有基金 [fundcode, selectdt, sml_ret, big_ret, val_ret, grw_ret, max_idx, max_val, fundname, seasndt, mv, g_v]
        return:
                cluster_thrs_ 每个季度的6个分类的阈值: [selectdt, big_thr, sml_thr, grw_thr, val_thr, mid_thr, bal_thr]
        '''
        ''' 每一期在每一类中计算中位数作为聚类中心 '''
        # 仅保留相关系数大于0.7的条目,用作选取聚类中心
        corr_tbl_ = corr_tbl.query("max_val > 0.7").reset_index()[['fundcode','selectdt','sml_ret','big_ret','val_ret','grw_ret','max_idx','max_val','mv','g_v']]
        cluster_thrs = corr_tbl_.groupby(['selectdt','max_idx'])[['mv','g_v']].median()
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
    
    def classification(self, fund_factor:pd.DataFrame, method_:int, corr_tbl:pd.DataFrame, cluster_thrs_:pd.DataFrame):
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

        # 对季度节点的风格进行修正
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