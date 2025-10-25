# 医学知识学习系统 - Python实战项目

## 项目简介

本项目包含两个完整的医学学习系统，帮助你通过Python学习医学知识：

1. **医学知识学习系统** (`medical_learning_system.py`)
   - 症状诊断助手
   - 药物信息查询
   - 医学术语学习

2. **心脏病预测系统** (`heart_disease_prediction.py`)
   - 基于机器学习的疾病预测
   - 使用真实医学特征
   - 风险评估和建议

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行项目

**运行医学学习系统:**
```bash
python medical_learning_system.py
```

**运行心脏病预测系统:**
```bash
python heart_disease_prediction.py
```

## 项目功能详解

### 一、医学知识学习系统

#### 功能1：症状诊断助手
- 输入症状，系统自动分析可能的疾病
- 提供匹配度评分
- 显示疾病详情（症状、原因、治疗、预防）

**示例使用:**
```
输入症状: 发热,咳嗽,流鼻涕
输出:
  - 感冒 (匹配度: 50%)
  - 症状、治疗建议、预防措施
```

#### 功能2：药物信息查询
- 查询药物详细信息
- 了解用法用量、副作用、禁忌症
- 学习药物分类

**支持的药物:**
- 阿司匹林
- 青霉素
- 布洛芬
- (可自行扩展)

#### 功能3：医学术语学习
- 随机测验医学术语
- 记录学习进度
- 统计学习效果

### 二、心脏病预测系统

#### 核心功能
- 基于13个医学特征进行预测
- 使用随机森林算法
- 提供风险概率和置信度
- 给出健康建议

#### 医学特征说明

| 特征     | 说明       | 范围             |
| -------- | ---------- | ---------------- |
| age      | 年龄       | 29-80岁          |
| sex      | 性别       | 0=女, 1=男       |
| cp       | 胸痛类型   | 0-3              |
| trestbps | 静息血压   | 90-200 mm Hg     |
| chol     | 胆固醇     | 120-400 mg/dl    |
| fbs      | 空腹血糖   | 0=否, 1=是(>120) |
| thalach  | 最大心率   | 70-200           |
| exang    | 运动心绞痛 | 0=否, 1=是       |
| oldpeak  | ST段压低   | 0-6              |

#### 使用流程
1. 系统自动训练模型
2. 输入患者信息
3. 获得预测结果和建议

## 扩展学习资源

### 推荐数据集

1. **UCI Machine Learning Repository**
   - Heart Disease Dataset
   - Diabetes Dataset
   - Breast Cancer Dataset

2. **Kaggle医学数据集**
   - Heart Disease UCI
   - Pima Indians Diabetes
   - COVID-19 Dataset

3. **MIMIC-III/IV**
   - 重症监护数据库（需申请）
   - 真实患者数据

### 推荐Python医学库

```python
# 医学文本处理
pip install medspacy
pip install scispacy

# 生物信息学
pip install biopython

# 医学术语
pip install pymedtermino2

# DICOM图像处理
pip install pydicom

# 药物数据库
pip install chembl-webresource-client
```

### 进阶项目建议

1. **医学图像分析**
   - X光片分析
   - CT/MRI图像分类
   - 使用深度学习(CNN)

2. **电子病历分析**
   - 自然语言处理
   - 疾病编码(ICD-10)
   - 症状提取

3. **药物推荐系统**
   - 基于协同过滤
   - 药物相互作用检测
   - 个性化用药建议

4. **疫情预测模型**
   - 时间序列分析
   - 传染病模型(SIR)
   - 地理分布可视化

## 学习路线图

### 初级阶段（当前项目）
- ✅ 基础医学知识库
- ✅ 简单的症状诊断
- ✅ 机器学习预测模型

### 中级阶段
- [ ] 接入真实医学数据集
- [ ] 自然语言处理(医学文本)
- [ ] 深度学习模型(疾病分类)
- [ ] 数据可视化

### 高级阶段
- [ ] 医学图像识别(X光、CT)
- [ ] 电子病历系统
- [ ] 药物发现和分子模拟
- [ ] 临床决策支持系统

## 数据获取方式

### 1. 公开数据集
```python
# Kaggle数据集
# 需要先安装kaggle CLI: pip install kaggle
# 然后配置API token

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# 下载心脏病数据集
api.dataset_download_files('ronitf/heart-disease-uci',
                           path='./data',
                           unzip=True)
```

### 2. PubMed文献
```python
from Bio import Entrez

Entrez.email = "your_email@example.com"

# 搜索糖尿病相关文献
handle = Entrez.esearch(db="pubmed", term="diabetes", retmax=10)
record = Entrez.read(handle)
print(record["IdList"])
```

### 3. DrugBank API
访问 https://go.drugbank.com/ 注册账号获取API密钥

## 注意事项

⚠️ **重要提醒:**

1. **本项目仅供学习使用**
   - 不可用于真实医疗诊断
   - 所有结果仅供参考

2. **医学伦理**
   - 尊重患者隐私
   - 遵守数据使用协议
   - 不传播未经验证的医学信息

3. **数据安全**
   - 不使用真实患者数据练习
   - 保护个人健康信息
   - 遵守HIPAA等法规

## 常见问题

**Q1: 如何获取更多疾病数据？**
A: 可以扩展knowledge_base中的疾病字典，或从公开数据集导入

**Q2: 模型准确率如何提高？**
A:
- 使用更多真实数据
- 特征工程优化
- 尝试其他算法(XGBoost, Neural Networks)
- 超参数调优

**Q3: 如何添加新功能？**
A: 项目采用面向对象设计，可轻松扩展新模块

## 贡献与反馈

如果你有好的想法或建议，欢迎：
- 添加更多疾病知识
- 优化预测算法
- 增加可视化功能
- 改进用户体验

## 学习资源

**在线课程:**
- Coursera: AI for Medicine Specialization
- edX: Data Science for Healthcare

**书籍推荐:**
- 《医学统计学》
- 《机器学习在医学中的应用》
- 《生物信息学导论》

**GitHub优秀项目:**
- medspacy: 医学NLP
- PyHealth: 医疗深度学习
- ClinicalBERT: 临床文本BERT模型

---

## 快速开始示例

```python
# 示例1: 快速症状检查
from medical_learning_system import MedicalLearningSystem

system = MedicalLearningSystem()
system.run_symptom_diagnosis()

# 示例2: 心脏病风险预测
from heart_disease_prediction import HeartDiseasePredictionSystem

predictor = HeartDiseasePredictionSystem()
result = predictor.predict({
    'age': 55, 'sex': 1, 'cp': 2,
    'trestbps': 140, 'chol': 250,
    # ... 其他特征
})
print(f"风险概率: {result['risk_probability']:.2f}%")
```

开始你的医学AI学习之旅吧！🏥💻
