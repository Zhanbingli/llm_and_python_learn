"""
scikit-learn 完整教程
涵盖从数据加载到模型优化的完整机器学习流程
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("scikit-learn 机器学习完整教程")
print("=" * 80)

# ============================================================================
# 第一部分：回归问题示例 (预测房价)
# ============================================================================
print("\n" + "=" * 80)
print("第一部分：回归问题 - 波士顿房价预测")
print("=" * 80)

# 1. 加载数据
print("\n1. 加载数据...")
# 使用加利福尼亚房价数据集（波士顿数据集已被弃用）
california = datasets.fetch_california_housing()
X = california.data
y = california.target

print(f"数据集形状: {X.shape}")
print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"特征名称: {california.feature_names}")

# 2. 数据分割
print("\n2. 分割训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 3. 数据标准化
print("\n3. 数据标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("标准化完成 (均值=0, 方差=1)")

# 4. 训练多个回归模型
print("\n4. 训练多个回归模型...")

models = {
    "线性回归": LinearRegression(),
    "Ridge回归": Ridge(alpha=1.0),
    "Lasso回归": Lasso(alpha=0.1)
}

results = {}

for name, model in models.items():
    print(f"\n训练 {name}...")

    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MSE": mse, "RMSE": rmse, "R²": r2}

    print(f"  均方误差 (MSE): {mse:.4f}")
    print(f"  均方根误差 (RMSE): {rmse:.4f}")
    print(f"  R² 分数: {r2:.4f}")

    # 显示部分预测结果
    print(f"  前5个预测值: {y_pred[:5]}")
    print(f"  前5个真实值: {y_test[:5]}")

# 5. 交叉验证
print("\n5. 使用交叉验证评估模型稳定性...")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                 cv=5, scoring='r2')
    print(f"{name} - 5折交叉验证 R² 分数: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 第二部分：分类问题示例 (鸢尾花分类)
# ============================================================================
print("\n" + "=" * 80)
print("第二部分：分类问题 - 鸢尾花分类")
print("=" * 80)

# 1. 加载数据
print("\n1. 加载鸢尾花数据集...")
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(f"数据集形状: {X.shape}")
print(f"类别数量: {len(np.unique(y))}")
print(f"类别名称: {iris.target_names}")
print(f"特征名称: {iris.feature_names}")

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled[0][0])
# 4. 训练多个分类模型
print("\n2. 训练多个分类模型...")

classifiers = {
    "逻辑回归": LogisticRegression(max_iter=200),
    "决策树": DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "支持向量机": SVC(kernel='rbf', random_state=42)
}

for name, clf in classifiers.items():
    print(f"\n训练 {name}...")

    # 训练
    clf.fit(X_train_scaled, y_train)

    # 预测
    y_pred = clf.predict(X_test_scaled)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  准确率: {accuracy:.4f}")

    # 详细分类报告
    print(f"\n  分类报告:")
    print(classification_report(y_test, y_pred,
                                target_names=iris.target_names))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"  混淆矩阵:")
    print(cm)

# 5. 交叉验证
print("\n3. 交叉验证评估...")
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"{name} - 5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 第三部分：超参数调优
# ============================================================================
print("\n" + "=" * 80)
print("第三部分：超参数调优 (网格搜索)")
print("=" * 80)

print("\n使用网格搜索优化随机森林...")

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# 创建网格搜索对象
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 执行搜索
print("开始网格搜索...")
grid_search.fit(X_train_scaled, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# ============================================================================
# 第四部分：实用技巧和最佳实践
# ============================================================================
print("\n" + "=" * 80)
print("第四部分：实用技巧")
print("=" * 80)

print("""
1. 数据预处理的重要性
   - 标准化/归一化可以提高模型性能
   - 缺失值处理：SimpleImputer
   - 类别变量编码：OneHotEncoder, LabelEncoder

2. 模型选择建议
   - 线性问题：LinearRegression, LogisticRegression
   - 非线性问题：RandomForest, SVM, GradientBoosting
   - 大数据集：SGDClassifier, SGDRegressor (在线学习)

3. 评估指标选择
   回归：MSE, RMSE, MAE, R²
   分类：Accuracy, Precision, Recall, F1-Score, AUC-ROC

4. 避免过拟合
   - 使用交叉验证
   - 正则化 (Ridge, Lasso)
   - 减少模型复杂度
   - 增加训练数据

5. 超参数调优
   - GridSearchCV：网格搜索（全面但慢）
   - RandomizedSearchCV：随机搜索（快速）
""")

# ============================================================================
# 第五部分：完整的Pipeline示例
# ============================================================================
print("\n" + "=" * 80)
print("第五部分：使用Pipeline构建完整工作流")
print("=" * 80)

from sklearn.pipeline import Pipeline

# 创建Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("\nPipeline的优点：")
print("- 自动化预处理和建模流程")
print("- 防止数据泄露")
print("- 代码更简洁")

# 训练Pipeline
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPipeline准确率: {accuracy:.4f}")

print("\n" + "=" * 80)
print("教程完成！")
print("=" * 80)
print("\n下一步学习建议：")
print("1. 尝试其他数据集 (kaggle.com)")
print("2. 学习特征工程技术")
print("3. 探索深度学习 (TensorFlow, PyTorch)")
print("4. 实践端到端的机器学习项目")
