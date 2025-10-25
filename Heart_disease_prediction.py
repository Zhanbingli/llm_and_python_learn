"""
心脏病预测系统 - 机器学习实战
使用真实医学数据进行疾病预测
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

class HeartDiseasePredictionSystem:
    """心脏病预测系统"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.feature_descriptions = {
            'age': '年龄',
            'sex': '性别 (1=男, 0=女)',
            'cp': '胸痛类型 (0-3)',
            'trestbps': '静息血压 (mm Hg)',
            'chol': '血清胆固醇 (mg/dl)',
            'fbs': '空腹血糖>120 mg/dl (1=是, 0=否)',
            'restecg': '静息心电图结果 (0-2)',
            'thalach': '最大心率',
            'exang': '运动诱发心绞痛 (1=是, 0=否)',
            'oldpeak': 'ST段压低',
            'slope': 'ST段斜率 (0-2)',
            'ca': '荧光透视的主要血管数 (0-3)',
            'thal': '地中海贫血 (0=正常, 1=固定缺陷, 2=可逆缺陷)'
        }

    def create_sample_dataset(self, n_samples=300):
        """
        创建模拟心脏病数据集
        在实际应用中，应该使用真实的UCI心脏病数据集
        """
        np.random.seed(42)

        data = {
            'age': np.random.randint(29, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(120, 400, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(70, 200, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 3, n_samples)
        }

        df = pd.DataFrame(data)

        # 创建目标变量（基于一些医学规则的模拟）
        # 实际情况更复杂，这里简化处理
        target = ((df['age'] > 55).astype(int) +
                 (df['cp'] > 1).astype(int) +
                 (df['trestbps'] > 140).astype(int) +
                 (df['chol'] > 240).astype(int) +
                 (df['thalach'] < 120).astype(int) +
                 (df['exang'] == 1).astype(int)) >= 3

        df['target'] = target.astype(int)

        return df

    def train_model(self, X, y):
        """训练模型"""
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.model.fit(X_train_scaled, y_train)

        # 评估模型
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"模型训练完成！")
        print(f"准确率: {accuracy*100:.2f}%")
        print(f"\n分类报告:")
        print(classification_report(y_test, y_pred,
                                   target_names=['健康', '有风险']))

        # 特征重要性
        feature_importance = pd.DataFrame({
            '特征': self.feature_names,
            '重要性': self.model.feature_importances_
        }).sort_values('重要性', ascending=False)

        print(f"\n特征重要性（前5项）:")
        for idx, row in feature_importance.head().iterrows():
            feature_name = self.feature_descriptions[row['特征']]
            print(f"  {feature_name}: {row['重要性']:.4f}")

        return accuracy

    def predict(self, patient_data: dict):
        """
        预测单个患者的心脏病风险
        """
        if self.model is None:
            raise Exception("模型尚未训练，请先训练模型！")

        # 准备输入数据
        X = pd.DataFrame([patient_data])[self.feature_names]
        X_scaled = self.scaler.transform(X)

        # 预测
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]

        return {
            'prediction': '有心脏病风险' if prediction == 1 else '健康',
            'risk_probability': probability[1] * 100,
            'confidence': max(probability) * 100
        }

    def interactive_prediction(self):
        """交互式预测"""
        print("\n" + "="*50)
        print("心脏病风险评估")
        print("="*50)
        print("\n请输入患者信息：\n")

        patient_data = {}

        # 收集患者信息
        questions = {
            'age': '年龄',
            'sex': '性别 (1=男性, 0=女性)',
            'cp': '胸痛类型 (0=无症状, 1=非典型心绞痛, 2=非心绞痛, 3=典型心绞痛)',
            'trestbps': '静息血压 (mm Hg)',
            'chol': '血清胆固醇 (mg/dl)',
            'fbs': '空腹血糖是否>120 mg/dl (1=是, 0=否)',
            'restecg': '静息心电图 (0=正常, 1=ST-T异常, 2=左心室肥大)',
            'thalach': '最大心率',
            'exang': '运动是否诱发心绞痛 (1=是, 0=否)',
            'oldpeak': 'ST段压低值 (0-6)',
            'slope': 'ST段斜率 (0=上升, 1=平坦, 2=下降)',
            'ca': '荧光透视显示的主要血管数 (0-3)',
            'thal': '地中海贫血 (0=正常, 1=固定缺陷, 2=可逆缺陷)'
        }

        for key, question in questions.items():
            while True:
                try:
                    value = float(input(f"{question}: "))
                    patient_data[key] = value
                    break
                except ValueError:
                    print("请输入有效的数字！")

        # 进行预测
        result = self.predict(patient_data)

        print("\n" + "="*50)
        print("预测结果")
        print("="*50)
        print(f"\n诊断: {result['prediction']}")
        print(f"风险概率: {result['risk_probability']:.2f}%")
        print(f"置信度: {result['confidence']:.2f}%")

        # 给出建议
        if result['risk_probability'] > 50:
            print("\n⚠️  建议:")
            print("  - 尽快就医进行详细检查")
            print("  - 控制血压和胆固醇")
            print("  - 保持健康的生活方式")
            print("  - 规律运动，健康饮食")
        else:
            print("\n✓ 建议:")
            print("  - 保持当前健康的生活方式")
            print("  - 定期体检")
            print("  - 注意心血管健康")

        print("\n⚠️  注意：此预测仅供参考，请咨询专业医生！")

def main():
    """主程序"""
    print("="*60)
    print("心脏病预测系统 - 机器学习实战项目")
    print("="*60)

    # 创建系统实例
    system = HeartDiseasePredictionSystem()

    # 创建数据集
    print("\n正在创建模拟数据集...")
    df = system.create_sample_dataset(n_samples=300)
    print(f"数据集大小: {len(df)} 条记录")
    print(f"特征数量: {len(system.feature_names)}")

    # 训练模型
    print("\n" + "="*60)
    print("开始训练模型...")
    print("="*60)

    X = df[system.feature_names]
    y = df['target']

    accuracy = system.train_model(X, y)

    # 交互式预测
    while True:
        print("\n" + "="*60)
        print("选项:")
        print("1. 进行心脏病风险预测")
        print("2. 查看数据集统计")
        print("3. 退出")
        print("="*60)

        choice = input("\n请选择 (1-3): ")

        if choice == "1":
            system.interactive_prediction()
        elif choice == "2":
            print("\n数据集统计:")
            print(df.describe())
            print(f"\n心脏病患者比例: {(df['target'].sum() / len(df) * 100):.2f}%")
        elif choice == "3":
            print("\n感谢使用！")
            break
        else:
            print("无效选择，请重新输入。")

if __name__ == "__main__":
    main()
