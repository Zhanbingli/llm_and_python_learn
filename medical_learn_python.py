"""
医学知识学习系统 - 实战项目
包含：疾病诊断助手、药物查询、症状分析
"""

import json
import random
from datetime import datetime
from typing import List, Dict, Tuple

class MedicalKnowledgeBase:
    """医学知识库"""

    def __init__(self):
        # 疾病知识库（示例数据）
        self.diseases = {
            "感冒": {
                "symptoms": ["发热", "咳嗽", "流鼻涕", "喉咙痛", "头痛", "乏力"],
                "causes": ["病毒感染", "免疫力下降", "天气变化"],
                "treatments": ["休息", "多喝水", "对症治疗药物", "维生素C"],
                "prevention": ["勤洗手", "戴口罩", "增强免疫力", "避免人群密集"],
                "severity": "轻度"
            },
            "高血压": {
                "symptoms": ["头晕", "头痛", "心悸", "视力模糊", "耳鸣"],
                "causes": ["遗传因素", "高盐饮食", "肥胖", "压力", "缺乏运动"],
                "treatments": ["降压药", "低盐饮食", "减肥", "规律运动", "减压"],
                "prevention": ["健康饮食", "适量运动", "控制体重", "减少压力"],
                "severity": "中度"
            },
            "糖尿病": {
                "symptoms": ["多饮", "多尿", "多食", "体重下降", "乏力", "视力模糊"],
                "causes": ["胰岛素分泌不足", "胰岛素抵抗", "遗传", "肥胖", "不良饮食"],
                "treatments": ["胰岛素注射", "口服降糖药", "饮食控制", "运动疗法"],
                "prevention": ["控制体重", "健康饮食", "规律运动", "定期体检"],
                "severity": "中重度"
            },
            "胃炎": {
                "symptoms": ["上腹痛", "恶心", "呕吐", "食欲不振", "腹胀"],
                "causes": ["幽门螺杆菌感染", "不规律饮食", "压力", "药物刺激"],
                "treatments": ["抑酸药", "抗生素(如有感染)", "饮食调理", "减压"],
                "prevention": ["规律饮食", "避免刺激性食物", "减少压力", "戒烟限酒"],
                "severity": "轻中度"
            }
        }

        # 药物知识库
        self.drugs = {
            "阿司匹林": {
                "category": "解热镇痛抗炎药",
                "uses": ["解热", "镇痛", "抗炎", "抗血小板"],
                "dosage": "成人：每次0.3-0.6g，每日3次",
                "side_effects": ["胃肠道反应", "出血倾向", "过敏反应"],
                "contraindications": ["胃溃疡", "出血性疾病", "孕妇"]
            },
            "青霉素": {
                "category": "β-内酰胺类抗生素",
                "uses": ["细菌感染", "链球菌感染", "肺炎"],
                "dosage": "根据感染严重程度和医嘱",
                "side_effects": ["过敏反应", "胃肠道反应", "皮疹"],
                "contraindications": ["青霉素过敏者"]
            },
            "布洛芬": {
                "category": "非甾体抗炎药",
                "uses": ["解热", "镇痛", "抗炎"],
                "dosage": "成人：每次0.2-0.4g，每日2-3次",
                "side_effects": ["胃肠道不适", "头晕", "皮疹"],
                "contraindications": ["胃溃疡", "严重肝肾功能不全"]
            }
        }

        # 医学术语库
        self.medical_terms = {
            "高血压": "Hypertension - 动脉血压持续升高的疾病",
            "糖尿病": "Diabetes Mellitus - 胰岛素分泌或作用缺陷导致的代谢性疾病",
            "心肌梗死": "Myocardial Infarction - 心肌缺血性坏死",
            "脑卒中": "Stroke - 脑血管突然破裂或阻塞导致的脑功能障碍",
            "肺炎": "Pneumonia - 肺部感染性炎症",
            "胃溃疡": "Gastric Ulcer - 胃黏膜溃疡性病变"
        }

class SymptomChecker:
    """症状检查器 - 基于症状推断可能疾病"""

    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.kb = knowledge_base

    def check_symptoms(self, symptoms: List[str]) -> List[Tuple[str, float]]:
        """
        根据症状列表返回可能的疾病及匹配度
        """
        results = []

        for disease_name, disease_info in self.kb.diseases.items():
            disease_symptoms = set(disease_info["symptoms"])
            input_symptoms = set(symptoms)

            # 计算匹配度
            if len(disease_symptoms) == 0:
                continue

            match_count = len(input_symptoms & disease_symptoms)
            match_rate = match_count / len(disease_symptoms)

            if match_count > 0:
                results.append((disease_name, match_rate, match_count))

        # 按匹配度排序
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)

        return [(name, rate) for name, rate, _ in results]

    def get_disease_info(self, disease_name: str) -> Dict:
        """获取疾病详细信息"""
        return self.kb.diseases.get(disease_name, {})

class DrugDatabase:
    """药物数据库查询"""

    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.kb = knowledge_base

    def search_drug(self, drug_name: str) -> Dict:
        """搜索药物信息"""
        return self.kb.drugs.get(drug_name, None)

    def list_drugs_by_category(self, category: str) -> List[str]:
        """按类别列出药物"""
        return [name for name, info in self.kb.drugs.items()
                if info["category"] == category]

class MedicalTermLearning:
    """医学术语学习系统"""

    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.kb = knowledge_base
        self.learning_history = []

    def quiz(self) -> Tuple[str, str]:
        """随机测验"""
        term, definition = random.choice(list(self.kb.medical_terms.items()))
        return term, definition

    def add_to_history(self, term: str, correct: bool):
        """添加学习记录"""
        self.learning_history.append({
            "term": term,
            "correct": correct,
            "timestamp": datetime.now().isoformat()
        })

    def get_statistics(self) -> Dict:
        """获取学习统计"""
        if not self.learning_history:
            return {"total": 0, "correct": 0, "accuracy": 0}

        total = len(self.learning_history)
        correct = sum(1 for record in self.learning_history if record["correct"])

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0
        }

class MedicalLearningSystem:
    """医学学习系统主类"""

    def __init__(self):
        self.kb = MedicalKnowledgeBase()
        self.symptom_checker = SymptomChecker(self.kb)
        self.drug_db = DrugDatabase(self.kb)
        self.term_learning = MedicalTermLearning(self.kb)

    def run_symptom_diagnosis(self):
        """运行症状诊断模块"""
        print("\n" + "="*50)
        print("症状诊断助手")
        print("="*50)

        print("\n可用症状：发热、咳嗽、流鼻涕、喉咙痛、头痛、头晕、")
        print("心悸、多饮、多尿、上腹痛、恶心、呕吐、乏力等")
        print("\n请输入您的症状（用逗号分隔）：")
        symptoms_input = input("> ")

        symptoms = [s.strip() for s in symptoms_input.split(",")]

        results = self.symptom_checker.check_symptoms(symptoms)

        if not results:
            print("\n未找到匹配的疾病，建议就医咨询。")
            return

        print(f"\n根据您的症状，可能的疾病（按匹配度排序）：\n")

        for i, (disease, match_rate) in enumerate(results[:3], 1):
            print(f"{i}. {disease} - 匹配度: {match_rate*100:.1f}%")

            disease_info = self.symptom_checker.get_disease_info(disease)
            print(f"   严重程度: {disease_info.get('severity', '未知')}")
            print(f"   常见症状: {', '.join(disease_info.get('symptoms', []))}")
            print(f"   可能原因: {', '.join(disease_info.get('causes', []))}")
            print(f"   治疗建议: {', '.join(disease_info.get('treatments', []))}")
            print(f"   预防措施: {', '.join(disease_info.get('prevention', []))}")
            print()

        print("⚠️  注意：此结果仅供参考，请及时就医获取专业诊断！")

    def run_drug_query(self):
        """运行药物查询模块"""
        print("\n" + "="*50)
        print("药物信息查询")
        print("="*50)

        print(f"\n可查询药物：{', '.join(self.kb.drugs.keys())}")
        print("\n请输入药物名称：")
        drug_name = input("> ")

        drug_info = self.drug_db.search_drug(drug_name)

        if not drug_info:
            print(f"\n未找到药物 '{drug_name}' 的信息。")
            return

        print(f"\n药物名称: {drug_name}")
        print(f"类别: {drug_info['category']}")
        print(f"用途: {', '.join(drug_info['uses'])}")
        print(f"用法用量: {drug_info['dosage']}")
        print(f"副作用: {', '.join(drug_info['side_effects'])}")
        print(f"禁忌症: {', '.join(drug_info['contraindications'])}")
        print("\n⚠️  请遵医嘱用药，不要自行用药！")

    def run_term_learning(self):
        """运行术语学习模块"""
        print("\n" + "="*50)
        print("医学术语学习")
        print("="*50)

        print("\n开始医学术语测验（输入 'q' 退出）\n")

        while True:
            term, definition = self.term_learning.quiz()

            print(f"请解释医学术语: {term}")
            print("(输入 's' 显示答案, 'q' 退出)")
            answer = input("> ")

            if answer.lower() == 'q':
                break
            elif answer.lower() == 's':
                print(f"\n答案: {definition}\n")
                print("您是否理解了？(y/n)")
                understood = input("> ").lower() == 'y'
                self.term_learning.add_to_history(term, understood)
            else:
                print(f"\n正确答案: {definition}\n")
                print("您的回答是否正确？(y/n)")
                correct = input("> ").lower() == 'y'
                self.term_learning.add_to_history(term, correct)

        # 显示学习统计
        stats = self.term_learning.get_statistics()
        print(f"\n学习统计:")
        print(f"总题数: {stats['total']}")
        print(f"正确数: {stats['correct']}")
        print(f"正确率: {stats['accuracy']*100:.1f}%")

    def run(self):
        """运行主程序"""
        print("\n" + "="*50)
        print("欢迎使用医学知识学习系统")
        print("="*50)

        while True:
            print("\n请选择功能：")
            print("1. 症状诊断助手")
            print("2. 药物信息查询")
            print("3. 医学术语学习")
            print("4. 退出系统")

            choice = input("\n请输入选项 (1-4): ")

            if choice == "1":
                self.run_symptom_diagnosis()
            elif choice == "2":
                self.run_drug_query()
            elif choice == "3":
                self.run_term_learning()
            elif choice == "4":
                print("\n感谢使用！祝您学习愉快！")
                break
            else:
                print("\n无效选项，请重新选择。")

if __name__ == "__main__":
    system = MedicalLearningSystem()
    system.run()
