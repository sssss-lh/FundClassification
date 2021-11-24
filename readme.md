# 基金风格分类

- 每个股票取selectdt前一个工作日的指标作为当季的因子值
- 每只基金在每个selectdt根据持仓股票，计算其市值和成长价值暴露
- 每个selectdt，每只基金分别计算与巨潮大盘，巨潮小盘，巨潮成长，巨潮价值的相关系数
- 基金归类到相关系数最大的一类，在大盘、小盘池中取市值中位数作为市值阈值big_thr，sml_thr，成长、价值池中取g-v中位数作为g-v阈值grw_thr，val_thr
- 分类方式1：取big_thr和sml_thr的平均值mid_thr，作为中盘聚类点，取grw_thr和val_thr的平均值bal_thr，作为均衡聚类点，构建9个聚类中心，用knn（k=1）聚类
- 分类方式2：简单取big_thr，sml_thr， grw_thr，val_thr作为划分的阈值
- 每一个selectdt，在每个类别中选取”隐形交易能力”因子值最小的4只基金作为下一期

## 文件夹

### raw_data 文件夹:

存放原始数据，目录下应有如下文件：

ASHAREINDUSTRIESCODE.pkl

chinamutualfundassetportfolio.pkl

CHINAMUTUALFUNDDESCRIPTION.pkl

CHINAMUTUALFUNDNAV.pkl

chinamutualfundnavs.pkl

CHINAMUTUALFUNDSECTOR.pkl

CHINAMUTUALFUNDSTOCKPORTFOLIO.pkl

marketindex.pkl

stk_map.xlsx

stkdailyfactor.pkl

styleindex.pkl



### generate_data文件夹：

generate_data.py文件用于处理raw_data里的数据，运行generate_data.py之后将运行所有用于处理数据的.py文件

exc_stk_daily_factor.py 定义了处理股票因子的类：

```python
class stkfactor_exc():
    def __init__(self, stk_daily_factor:pd.DataFrame, is_raw:bool = True) -> None:
```

 处理每日的股票因子，对每个因子进行MAD和z_score处理，并计算成长和价值因子，变成每个季度的股票因子

exc_fund_port.py 处理基金每个季度的持仓

### data文件夹：

在运行generate_data.py之后，将处理好的文件生成到该目录，处理好的文件如下：

seasn_dts_df.pkl 记录了每个季度的季度日期

stkfactor_f.pkl 记录了每个季度股票的因子值

fund_portf_.pkl 记录了修改后的基金持仓信息

corr_tbl.pkl 记录了基金与四个巨潮指数的相关性列表

ret_tbl.pkl 记录了每个季度每只基金的季末后30天的真实收益率和模拟收益率，以及两个收益率的排名差

fundret_.pkl 记录了每只基金所有交易日的收益率

**已经存放的文件：**

CHINAMUTUALFUNDSECTOR.pkl

ASHAREINDUSTRIESCODE.pkl

CHINAMUTUALFUNDDESCRIPTION.pkl

### result/result文件夹：

记录所生成的所有结果

### style_classification文件夹：

**classifier.py** 定义了分类器类，对分类器进行合适初始化，即可对基金类别进行分类

```python
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
```

**style_classification_example.py** 给出了使用classifier类对基金进行分类的一个例子，并保存了基金分类结果。

## 运行代码

- 运行generate_data/generate_data.py
- 运行style_classification/style_classification_example.py，对基金进行分类
- 分段运行style_classification/show_result.py，查看研报中间数据的生成

问题 20191231 有的是在3.31之后才公布

