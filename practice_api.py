"""
实用机器学习项目：客户流失预测
完整的项目流程示例，包含数据清洗、特征工程、模型训练和评估
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
np.random.seed(42)

print("=" * 80)
print("实用项目：客户流失预测系统")
print("=" * 80)

# ============================================================================
# 第一步：生成模拟数据（实际项目中会从CSV或数据库读取）
# ============================================================================
print("\n第一步：数据加载和探索")
print("-" * 80)

# 创建模拟数据集
n_samples = 1000

data = {
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 70, n_samples),
    'monthly_charges': np.random.uniform(20, 150, n_samples),
    'total_charges': np.random.uniform(100, 8000, n_samples),
    'tenure_months': np.random.randint(1, 72, n_samples),
    'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples),
    'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check'], n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_samples),
    'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
}

# 创建目标变量（流失与否）
# 流失概率与合同类型、使用时长等相关
churn_prob = (
    (data['contract_type'] == 'Month-to-Month').astype(int) * 0.4 +
    (data['tenure_months'] < 12).astype(int) * 0.3 +
    (data['payment_method'] == 'Electronic Check').astype(int) * 0.2 +
    np.random.uniform(0, 0.1, n_samples)
)
data['churn'] = (churn_prob > 0.5).astype(int)

df = pd.DataFrame(data)

# 添加一些缺失值（模拟真实情况）
df.loc[np.random.choice(df.index, 50), 'total_charges'] = np.nan
df.loc[np.random.choice(df.index, 30), 'tenure_months'] = np.nan

print(f"数据集大小: {df.shape}")
print(f"\n前5行数据:")
print(df.head())

print(f"\n数据类型:")
print(df.dtypes)

print(f"\n缺失值统计:")
print(df.isnull().sum())

print(f"\n目标变量分布:")
print(df['churn'].value_counts())
print(f"流失率: {df['churn'].mean():.2%}")

# ============================================================================
# 第二步：数据清洗和预处理
# ============================================================================
print("\n" + "=" * 80)
print("第二步：数据清洗和预处理")
print("-" * 80)

# 1. 处理缺失值
print("\n1. 处理缺失值...")
# 对数值型变量用中位数填充
numeric_features = ['age', 'monthly_charges', 'total_charges', 'tenure_months']
imputer = SimpleImputer(strategy='median')
df[numeric_features] = imputer.fit_transform(df[numeric_features])
print(f"缺失值已填充")

# 2. 编码类别变量
print("\n2. 编码类别变量...")
categorical_features = ['contract_type', 'payment_method', 'internet_service',
                        'online_security', 'tech_support']

# 使用one-hot编码
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
print(f"编码后的特征数量: {df_encoded.shape[1]}")

# 3. 准备特征和目标变量
X = df_encoded.drop(['customer_id', 'churn'], axis=1)
y = df_encoded['churn']

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
print(f"\n特征列表:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# ============================================================================
# 第三步：特征工程
# ============================================================================
print("\n" + "=" * 80)
print("第三步：特征工程")
print("-" * 80)

# 创建新特征
print("\n创建衍生特征...")

# 平均月消费
X['avg_monthly_charge'] = df['total_charges'] / (df['tenure_months'] + 1)

# 消费增长率
X['charge_growth_rate'] = (df['monthly_charges'] - X['avg_monthly_charge']) / (X['avg_monthly_charge'] + 1)

# 客户价值分数
X['customer_value'] = df['tenure_months'] * df['monthly_charges']

print(f"新增特征后的形状: {X.shape}")
print(f"新增特征:")
print("  - avg_monthly_charge: 平均月消费")
print("  - charge_growth_rate: 消费增长率")
print("  - customer_value: 客户价值分数")

# ============================================================================
# 第四步：数据分割和标准化
# ============================================================================
print("\n" + "=" * 80)
print("第四步：数据分割和标准化")
print("-" * 80)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"训练集流失率: {y_train.mean():.2%}")
print(f"测试集流失率: {y_test.mean():.2%}")

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n特征已标准化")

# ============================================================================
# 第五步：模型训练和比较
# ============================================================================
print("\n" + "=" * 80)
print("第五步：模型训练和比较")
print("-" * 80)

models = {
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    "梯度提升": GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

for name, model in models.items():
    print(f"\n训练 {name}...")

    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 评估指标
    print(f"\n{name} 评估结果:")
    print(classification_report(y_test, y_pred, target_names=['留存', '流失']))

    # AUC分数
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC分数: {auc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(cm)
    print(f"真负例(TN): {cm[0,0]}, 假正例(FP): {cm[0,1]}")
    print(f"假负例(FN): {cm[1,0]}, 真正例(TP): {cm[1,1]}")

    # 交叉验证
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"\n5折交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    results[name] = {
        'model': model,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'cv_scores': cv_scores
    }

# ============================================================================
# 第六步：特征重要性分析
# ============================================================================
print("\n" + "=" * 80)
print("第六步：特征重要性分析")
print("-" * 80)

# 使用随机森林的特征重要性
rf_model = results['随机森林']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n前10个最重要的特征:")
print(feature_importance.head(10))

# ============================================================================
# 第七步：模型优化建议
# ============================================================================
print("\n" + "=" * 80)
print("第七步：模型优化建议")
print("-" * 80)

print("""
1. 数据方面：
   - 收集更多历史数据
   - 增加更多特征（如客户互动历史、投诉记录等）
   - 处理数据不平衡（SMOTE、类权重调整）

2. 模型方面：
   - 使用GridSearchCV或RandomizedSearchCV调优超参数
   - 尝试集成学习（Stacking、Voting）
   - 考虑使用XGBoost、LightGBM等高级算法

3. 特征工程：
   - 创建更多交互特征
   - 特征选择（去除不重要的特征）
   - 使用多项式特征

4. 业务应用：
   - 设置合适的决策阈值（根据业务成本）
   - 构建客户分群策略
   - 设计挽留措施的优先级
""")

# ============================================================================
# 第八步：模型保存和部署准备
# ============================================================================
print("\n" + "=" * 80)
print("第八步：模型保存和部署")
print("-" * 80)

import joblib

# 选择最佳模型（基于AUC分数）
best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
best_model = results[best_model_name]['model']

print(f"\n最佳模型: {best_model_name}")
print(f"AUC分数: {results[best_model_name]['auc']:.4f}")

# 保存模型和预处理器
model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'imputer': imputer
}

# joblib.dump(model_artifacts, 'churn_prediction_model.pkl')
print("\n模型已准备好保存")
print("使用方法: joblib.dump(model_artifacts, 'model.pkl')")

# ============================================================================
# 第九步：预测新客户
# ============================================================================
print("\n" + "=" * 80)
print("第九步：预测新客户流失概率")
print("-" * 80)

# 模拟一个新客户
new_customer = X_test.iloc[[0]].copy()
print("\n新客户数据（部分特征）:")
print(new_customer[['age', 'monthly_charges', 'total_charges', 'tenure_months']].to_string())

# 标准化
new_customer_scaled = scaler.transform(new_customer)

# 预测
churn_probability = best_model.predict_proba(new_customer_scaled)[0][1]
churn_prediction = best_model.predict(new_customer_scaled)[0]

print(f"\n预测结果:")
print(f"  流失概率: {churn_probability:.2%}")
print(f"  预测结果: {'流失' if churn_prediction == 1 else '留存'}")

if churn_probability > 0.7:
    print(f"  风险等级: 高风险")
    print(f"  建议: 立即采取挽留措施")
elif churn_probability > 0.4:
    print(f"  风险等级: 中风险")
    print(f"  建议: 加强客户关怀")
else:
    print(f"  风险等级: 低风险")
    print(f"  建议: 维持正常服务")

print("\n" + "=" * 80)
print("项目完成！")
print("=" * 80)

print("""
完整的机器学习项目流程：
1. ✓ 数据加载和探索
2. ✓ 数据清洗和预处理
3. ✓ 特征工程
4. ✓ 模型训练和比较
5. ✓ 模型评估和优化
6. ✓ 特征重要性分析
7. ✓ 模型保存和部署
8. ✓ 实际预测应用

下一步：
- 将模型部署为API服务（Flask/FastAPI）
- 构建监控dashboard
- 设置自动化再训练pipeline
""")
