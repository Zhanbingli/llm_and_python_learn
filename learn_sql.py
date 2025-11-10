# # import sqlite3
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # ============================================================================
# # # 场景：分析电商销售数据
# # # ============================================================================

# # conn = sqlite3.connect('ecommerce.db')

# # # 创建销售数据
# # sales_data = {
# #     'order_id': range(1, 101),
# #     'customer_id': [f'C{i%20:03d}' for i in range(100)],
# #     'product': ['手机', '电脑', '平板', '音响', '手表'] * 20,
# #     'amount': [5999, 8999, 3999, 1299, 2999] * 20,
# #     'quantity': [1, 1, 2, 3, 1] * 20,
# #     'order_date': pd.date_range('2024-01-01', periods=100, freq='D')
# # }

# # df_sales = pd.DataFrame(sales_data)

# # # 写入数据库
# # df_sales.to_sql('sales', conn, if_exists='replace', index=False)

# # # ============================================================================
# # # SQL分析查询
# # # ============================================================================

# # # 1. 每日销售额
# # query1 = '''
# # SELECT
# #     DATE(order_date) as date,
# #     SUM(amount * quantity) as daily_revenue,
# #     COUNT(*) as order_count
# # FROM sales
# # GROUP BY DATE(order_date)
# # ORDER BY date
# # '''

# # df_daily = pd.read_sql_query(query1, conn)
# # print("每日销售统计:")
# # print(df_daily.head())

# # # 2. 产品销售排名
# # query2 = '''
# # SELECT
# #     product,
# #     COUNT(*) as sales_count,
# #     SUM(amount * quantity) as total_revenue,
# #     AVG(amount) as avg_price
# # FROM sales
# # GROUP BY product
# # ORDER BY total_revenue DESC
# # '''

# # df_products = pd.read_sql_query(query2, conn)
# # print("\n产品销售排名:")
# # print(df_products)

# # # 3. 客户消费分析
# # query3 = '''
# # SELECT
# #     customer_id,
# #     COUNT(*) as purchase_count,
# #     SUM(amount * quantity) as total_spent,
# #     MAX(order_date) as last_purchase
# # FROM sales
# # GROUP BY customer_id
# # HAVING purchase_count > 3
# # ORDER BY total_spent DESC
# # LIMIT 10
# # '''

# # df_customers = pd.read_sql_query(query3, conn)
# # print("\nVIP客户(购买超过3次):")
# # print(df_customers)

# # # 4. 使用窗口函数 - 计算累计销售额
# # query4 = '''
# # SELECT
# #     DATE(order_date) as date,
# #     SUM(amount * quantity) OVER (ORDER BY DATE(order_date)) as cumulative_revenue
# # FROM sales
# # GROUP BY DATE(order_date)
# # '''

# # df_cumulative = pd.read_sql_query(query4, conn)

# # # 可视化
# # plt.figure(figsize=(12, 6))
# # plt.plot(df_daily['date'], df_daily['daily_revenue'], marker='o')
# # plt.title('每日销售额趋势')
# # plt.xlabel('日期')
# # plt.ylabel('销售额(¥)')
# # plt.xticks(rotation=45)
# # plt.tight_layout()
# # plt.savefig('sales_trend.png')
# # print("\n✅ 图表已保存为 sales_trend.png")

# # conn.close()


# # import pandas as pd
# # import numpy as np

#  data = {
#      'name':['zhans', 'lisi', 'zhoului',None],
#      'age':[20, 30, None, 32],
#      'monay':[3000, 2000, 6000, None,]
# }

#  df = pd.DataFrame(data)

# # #isnull
# # print(df.isnull().sum())

# # #dropna
# # df_dropna = df.dropna()

# # #filled na
# # de_filled = df.copy()
# # de_filled['age'] = de_filled['age'].fillna(de_filled['age'].mean())
# # de_filled['name'] = de_filled['name'].fillna('unknown')

# # #data translate type
# # df_cleaned = de_filled.copy()
# # df_cleaned['age'] = df_cleaned['age'].astype(int)

# # #duplicates
# # df_with_duplicates = pd.concat([df_cleaned, df_cleaned.iloc[[0]]], ignore_index=True)

# # #drop_duplicate
# # df_no_duplicate = df_with_duplicates.drop_duplicates()

# # #unnomer
# # q1 = df_cleaned['monay'].quantile(0.25)
# # q2 = df_cleaned['monay'].quantile(0.75)
# # iqr = q2 -q1
# # lower_bound = q1 - 0.5*iqr
# # upper_bound = q2 - 0.5*iqr
# # df_cleaned['monay_unnormore'] = (df_cleaned['monay'] < lower_bound) | (df_cleaned['monay'] > upper_bound)

# # df_string = df_cleaned.copy()
# # df_string['name'] = df_string['name'].str.strip()

# # df_string['name'] = df_string['name'].str.upper()
# def data_cleaning_pipeline(df):
#     """
#     完整的数据清洗流程
#     """
#     # 1. 复制原始数据
#     cleaned_df = df.copy()

#     # 2. 处理缺失值
#     cleaned_df = cleaned_df.fillna({
#         '姓名': '未知',
#         '年龄': cleaned_df['年龄'].mean(),
#         '工资': cleaned_df['工资'].median(),
#         '入职日期': '1900-01-01'
#     })

#     # 3. 数据类型转换
#     cleaned_df['年龄'] = cleaned_df['年龄'].astype(int)
#     cleaned_df['入职日期'] = pd.to_datetime(cleaned_df['入职日期'])

#     # 4. 去除重复值
#     cleaned_df = cleaned_df.drop_duplicates()

#     # 5. 字符串清洗
#     cleaned_df['姓名'] = cleaned_df['姓名'].str.strip().str.title()
#     cleaned_df['部门'] = cleaned_df['部门'].str.strip()

#     # 6. 添加清洗标记
#     cleaned_df['数据状态'] = '已清洗'

#     return cleaned_df

# # 使用清洗流程
# final_cleaned_df = data_cleaning_pipeline(df)
# print("\n最终清洗完成的数据:")
# print(final_cleaned_df)
import pandas as pd
import numpy as np

@dataclass
class columnrule:
    required: bool = False
    dtype: Optional[str] = None
    default: Any = None

def validata_columns(df: pd.DataFrame, config: CleaningCofig) -> None:
    missing = [name for name, rule in config.column_rules.items()
               if rule.required and name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def filled_defaults(df: pd.DataFrame, config: CleaningCofig) -> pd.DataFrame:
    result = df.copy()
    for column, rule in config.columns_rules.items():
        if column not in result.columns or result.default is None:
            continue

        result[column] = result[column].fillna(rule.dafault)
    return result

def cast_types(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    result = df.copy()
    for column, rule in config.column_rules.items():
        if column not in result.columns or not rule.dtype:
            continue
        result[column] = result[column].astype(rule.dtype, errors='ignore')
    return result

def clip_numeric_range(df: pd.DataFrame, config: cleaningConfig) -> pd.DataFrame:
    result = df.copy()
    for column, rule in config.column_rules.items():
        if column not in config.columns:
            continue
        seriers = result[column]
        if np.issubdtype(seriers.dtypt, np.number):
            if rule.min_value is not None:
                seriers = np.maximum(seriers, rule.main_value)
            if rule.max_value is not None:

                seriers = np.minimum(seriers, rule.max_value)

            result[column] = seriers

    return result

def normalize_text_columns(df: pd.DataFrame, config:CleaningConfig) -> pd.DataFrame:
    result = df.copy()
    for column, rule in config.column_rules.items():
        if column not in result.columns:
            continue
        if not pd.api.types.is_string_dtype(result[column]):
            continue

        seriers = result[column].astype(str)
        if rule.strip_text:
            seriers = seriers.str.strip()
        if rule.title_case:
            seriers = seriers.str.title()
        result[column] = seriers

    return result

def prune_outliers_zscore(df: pd.DataFrame, config:CleaningConfig) -> pd.DataFrame:
    if not np.isfinite(config.outlizer_zscore_threshold):
        return df

    threshold = config.outlier_zscore_threshold
    result = df.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        series = result[column]
        z_scores = (series - series.mean()) / series.std(ddof=0)
        mask = z_scores.abs() > threshold
        if mask.any():
            result.loc[mask,column] = np.nan

    return result

def parse_datas(df: pd.DataFrame, data_calumns:list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in data_calumns:
        if column in result.comlumns:
            result[column] = pd.to_datetime(result[column], errors='coerce')
    return result

def __init__(self, normalize: bool = True):
    self.normalize = normalize
    self._vectors: Optional[np.ndarray] = None
    self.documents: list[Document] = []

def add(self, documents: Iterable[Document]) -> None:
    embeddings = []
    docs: list[documents] = []
    for doc in documents:
        if doc.embedding is None:
            raise ValueError(f"Document {doc.doc_id} missing embedding")
        vector = np.asanyarray(doc.embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("embedding must be 1-D")
        if self.normalize:
            vector = _l2_normalize(vector)
        embeddings.append(vector)
        docs.append(doc)

    new_vectors = np.vstack(embeddings)
    if self._vector is None:
        self._vectors = new_vectors
        self._documents = docs

    else:
        if new_vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError("embeddings")
        self._vectors = np.vstack([self._vectors, new_vectors])
        self._documents.extend(docs)

def search(self, query_embedding:Vector, top_k: int = 5,
           metadata_filter: Optional[FilterFn] = None) -> list[searchResult]:
    query = np.asarray(query_embedding, dtype=np.float32)
    if self.normalize:
        query = _l1_normalize(query)

    scores = self._vectors @ query
    ranked_indices = np.argsort(scores)[::-1]



