"""
scikit-learn 快速参考指南
常用代码片段和最佳实践
"""

# ============================================================================
# 1. 数据加载和准备
# ============================================================================

# 从CSV加载数据
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20%作为测试集
    random_state=42     # 固定随机种子,确保可复现
    # ← 回归问题不使用stratify!
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# ============================================================================
# 步骤6: (可选) 特征缩放
# ============================================================================
# ============================================================================
# 2. 数据预处理
# ============================================================================

# 标准化（均值0，方差1）- 适用于大多数算法
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit在训练集
X_test_scaled = scaler.transform(X_test)        # 只transform测试集

# 归一化（缩放到0-1）- 适用于神经网络
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 处理缺失值
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # 'median', 'most_frequent', 'constant'
X_imputed = imputer.fit_transform(X)

# One-Hot编码（类别变量）
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# X_encoded = encoder.fit_transform(X_categorical)

# 使用pandas的get_dummies更简单
# X_encoded = pd.get_dummies(X, columns=['category_col'], drop_first=True)

# ============================================================================
# 3. 回归模型
# ============================================================================

# 线性回归
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Ridge回归（L2正则化）
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)  # alpha越大，正则化越强

# Lasso回归（L1正则化，可以特征选择）
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)

# ElasticNet（L1+L2）
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=1.0, l1_ratio=0.5)

# 决策树回归
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5, min_samples_split=10)

# 随机森林回归
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=100,     # 树的数量
    max_depth=10,         # 树的最大深度
    min_samples_split=5,  # 分裂所需最小样本数
    random_state=42
)

# 梯度提升回归
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

# 支持向量回归
from sklearn.svm import SVR
model = SVR(kernel='rbf', C=1.0, gamma='scale')

# ============================================================================
# 4. 分类模型
# ============================================================================

# 逻辑回归
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # 预测概率

# 决策树分类
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_split=10)

# 随机森林分类
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 梯度提升分类
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# 支持向量机分类
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, probability=True)  # probability=True可以预测概率

# K近邻
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# ============================================================================
# 5. 聚类模型（无监督学习）
# ============================================================================

# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# DBSCAN（基于密度）
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# 层次聚类
from sklearn.cluster import AgglomerativeClustering
hierarchical = AgglomerativeClustering(n_clusters=3)
labels = hierarchical.fit_predict(X)

# ============================================================================
# 6. 降维
# ============================================================================

# PCA（主成分分析）
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  # 降到2维
X_reduced = pca.fit_transform(X)
print(f"方差解释比例: {pca.explained_variance_ratio_}")

# t-SNE（可视化）
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

# ============================================================================
# 7. 模型评估 - 回归
# ============================================================================

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ============================================================================
# 8. 模型评估 - 分类
# ============================================================================

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

# 基本指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# 详细报告
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)

# AUC-ROC（需要概率预测）
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# ============================================================================
# 9. 交叉验证
# ============================================================================

from sklearn.model_selection import cross_val_score, cross_validate

# 简单交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"交叉验证分数: {scores}")
print(f"平均分数: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 多指标交叉验证
scoring = ['accuracy', 'precision_weighted', 'recall_weighted']
scores = cross_validate(model, X, y, cv=5, scoring=scoring)
print(scores)

# K折交叉验证
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    # 训练和验证

# 分层K折（保持类别比例）
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
# 10. 超参数调优
# ============================================================================

# 网格搜索（全面搜索）
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,  # 使用所有CPU核心
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_

# 随机搜索（更快）
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10, 20]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,  # 尝试20种组合
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# ============================================================================
# 11. Pipeline（工作流）
# ============================================================================

from sklearn.pipeline import Pipeline, make_pipeline

# 方法1：使用Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 方法2：使用make_pipeline（自动命名）
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

# Pipeline可以直接用于交叉验证和网格搜索
scores = cross_val_score(pipeline, X, y, cv=5)

# 在Pipeline中调优参数
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)

# ============================================================================
# 12. ColumnTransformer（不同列不同处理）
# ============================================================================

from sklearn.compose import ColumnTransformer

# 定义数值列和类别列
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'gender']

# 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 结合Pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

full_pipeline.fit(X_train, y_train)

# ============================================================================
# 13. 处理不平衡数据
# ============================================================================

# 方法1：类权重
model = RandomForestClassifier(class_weight='balanced')

# 方法2：SMOTE（需要安装imbalanced-learn）
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 方法3：手动设置样本权重
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)

# ============================================================================
# 14. 特征选择
# ============================================================================

# 方法1：基于模型的特征重要性
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'  # 选择重要性高于中位数的特征
)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# 方法2：单变量特征选择
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)  # 选择前10个最好的特征
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 方法3：递归特征消除
from sklearn.feature_selection import RFE

selector = RFE(
    estimator=RandomForestClassifier(random_state=42),
    n_features_to_select=10
)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)

# ============================================================================
# 15. 模型保存和加载
# ============================================================================

import joblib

# 保存模型
joblib.dump(model, 'model.pkl')

# 加载模型
loaded_model = joblib.load('model.pkl')

# 使用加载的模型
predictions = loaded_model.predict(X_new)

# 保存Pipeline（推荐）
joblib.dump(pipeline, 'pipeline.pkl')
loaded_pipeline = joblib.load('pipeline.pkl')

# ============================================================================
# 16. 学习曲线（诊断过拟合/欠拟合）
# ============================================================================

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# 计算平均值和标准差
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# ============================================================================
# 17. 验证曲线（调整单个参数）
# ============================================================================

from sklearn.model_selection import validation_curve

param_range = [10, 50, 100, 200, 500]
train_scores, val_scores = validation_curve(
    estimator=RandomForestClassifier(random_state=42),
    X=X,
    y=y,
    param_name='n_estimators',
    param_range=param_range,
    cv=5,
    scoring='accuracy'
)

# ============================================================================
# 18. 集成学习
# ============================================================================

# Voting分类器（投票）
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True))
    ],
    voting='soft'  # 'hard'是硬投票，'soft'是软投票（概率平均）
)
voting_clf.fit(X_train, y_train)

# Bagging
from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,  # 每个基模型使用80%的样本
    bootstrap=True,   # 有放回采样
    random_state=42
)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

adaboost_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0
)

# Stacking（需要sklearn >= 0.22）
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True))
    ],
    final_estimator=LogisticRegression()
)

# ============================================================================
# 19. 常用技巧
# ============================================================================

# 1. 快速查看模型参数
print(model.get_params())

# 2. 查看特征重要性（树模型）
importances = model.feature_importances_
feature_names = X.columns
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

# 3. 设置随机种子（确保结果可复现）
import numpy as np
np.random.seed(42)

# 4. 并行处理（加速训练）
model = RandomForestClassifier(n_jobs=-1)  # -1使用所有CPU核心

# 5. 早停（对于某些模型）
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.1,
    n_iter_no_change=10,  # 10轮没有改善就停止
    random_state=42
)

# ============================================================================
# 20. 常见错误和解决方案
# ============================================================================

"""
错误1: ValueError: could not convert string to float
解决: 需要对类别变量进行编码（OneHotEncoder或LabelEncoder）

错误2: NotFittedError
解决: 调用predict前必须先fit模型

错误3: 数据泄露
解决:
- 在分割数据前不要fit预处理器
- 使用Pipeline确保正确的fit/transform顺序

错误4: 过拟合
解决:
- 使用交叉验证
- 增加正则化
- 减少模型复杂度
- 获取更多数据

错误5: 训练太慢
解决:
- 使用n_jobs=-1并行处理
- 减少数据量或特征数
- 使用更简单的模型
- 考虑使用增量学习（SGD）

错误6: 内存不足
解决:
- 使用sparse矩阵
- 批量处理数据
- 使用增量学习算法
"""

print("scikit-learn快速参考指南 - 完成！")

