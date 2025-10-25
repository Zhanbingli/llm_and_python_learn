"""
妇产科学高频考点学习系统
适用于医学考试复习
"""

import random
import json
from datetime import datetime
from typing import List, Dict, Tuple

class ObstetricsGynecologyKnowledge:
    """妇产科学知识库"""

    def __init__(self):
        # 产科高频考点
        self.obstetrics = {
            "正常妊娠": {
                "考点等级": "⭐⭐⭐⭐⭐",
                "核心内容": {
                    "早孕诊断": [
                        "停经史（最重要的早孕症状）",
                        "早孕反应：恶心、呕吐、嗜睡（6周出现，12周消失）",
                        "尿频（增大子宫压迫膀胱）",
                        "乳房变化：增大、胀痛、乳晕着色",
                        "妇科检查：子宫增大变软、宫颈着色（紫蓝色）",
                        "辅助检查：尿hCG（+）、血hCG升高、超声见孕囊"
                    ],
                    "预产期计算": [
                        "末次月经第1日算起",
                        "月份：+9或-3",
                        "日期：+7",
                        "例：末次月经2024/1/10 → 预产期2024/10/17"
                    ],
                    "胎动": [
                        "初产妇：18-20周自觉胎动",
                        "经产妇：16-18周",
                        "正常胎动：≥30次/12小时，≥3次/小时",
                        "胎动减少：提示胎儿缺氧"
                    ],
                    "产前检查时间": [
                        "≤28周：每4周1次",
                        "28-36周：每2周1次",
                        "≥36周：每周1次",
                        "高危孕妇酌情增加次数"
                    ]
                },
                "必背数值": {
                    "孕周计算": "280天（40周）",
                    "早期妊娠": "≤13周+6天",
                    "中期妊娠": "14-27周+6天",
                    "晚期妊娠": "≥28周",
                    "足月妊娠": "37-41周+6天",
                    "过期妊娠": "≥42周"
                }
            },

            "异常妊娠": {
                "考点等级": "⭐⭐⭐⭐⭐",
                "核心内容": {
                    "流产": [
                        "先兆流产：少量阴道流血，轻微腹痛，宫口未开",
                        "难免流产：阴道流血增多，阵发腹痛，宫口已开",
                        "不全流产：部分组织排出，出血多，宫口开",
                        "完全流产：组织完全排出，出血停止，宫口闭",
                        "治疗原则：先兆→保胎；难免/不全→清宫",
                        "★连续3次及以上自然流产=习惯性流产"
                    ],
                    "异位妊娠（宫外孕）": [
                        "最常见部位：输卵管妊娠（95%，壶腹部最多）",
                        "三联征：停经+腹痛+阴道流血",
                        "破裂型：突发剧烈腹痛、内出血、休克",
                        "辅助检查：血hCG升高但低于正常、超声宫内无孕囊、后穹窿穿刺抽出不凝血",
                        "治疗：手术为主（输卵管切除或保守性手术）",
                        "★保守治疗条件：生命体征平稳、包块<3cm、无腹腔内出血、血hCG<2000U/L"
                    ],
                    "前置胎盘": [
                        "定义：28周后胎盘附着于子宫下段",
                        "分类：完全性、部分性、边缘性、低置",
                        "典型症状：妊娠晚期无痛性阴道流血",
                        "三大禁忌：禁止肛查、禁止阴道检查、禁止性生活",
                        "确诊：超声检查",
                        "处理：期待疗法（<36周）、终止妊娠（≥36周或大出血）"
                    ],
                    "胎盘早剥": [
                        "定义：20周后胎盘在胎儿娩出前从子宫壁剥离",
                        "高危因素：妊高征、腹部外伤、脐带过短",
                        "临床表现：突发持续性腹痛、阴道流血、子宫板状硬",
                        "并发症：DIC、产后出血、急性肾衰",
                        "处理：立即终止妊娠"
                    ]
                }
            },

            "妊娠期高血压": {
                "考点等级": "⭐⭐⭐⭐⭐",
                "分类": [
                    "妊娠期高血压：BP≥140/90mmHg，20周后出现，产后12周恢复",
                    "子痫前期：BP+蛋白尿或器官损害",
                    "  - 轻度：BP≥140/90，尿蛋白(+)",
                    "  - 重度：BP≥160/110或器官损害",
                    "子痫：子痫前期+抽搐",
                    "慢性高血压并发子痫前期",
                    "妊娠合并慢性高血压"
                ],
                "处理原则": [
                    "解痉：硫酸镁（首选，治疗和预防）",
                    "  - 负荷量：4-6g，15-20分钟静推",
                    "  - 维持量：1-2g/h静滴",
                    "  - 中毒表现：膝反射消失、呼吸<12次/分",
                    "  - 解毒剂：10%葡萄糖酸钙10ml静推",
                    "降压：收缩压≥160或舒张压≥110时",
                    "  - 首选：拉贝洛尔、硝苯地平",
                    "  - 禁用：ACEI、ARB类",
                    "利尿：仅用于急性心衰、肺水肿",
                    "终止妊娠：唯一根治方法"
                ],
                "终止妊娠指征": [
                    "子痫控制后2小时",
                    "子痫前期≥34周",
                    "子痫前期<34周但病情加重",
                    "胎盘早剥、胎儿窘迫"
                ]
            },

            "正常分娩": {
                "考点等级": "⭐⭐⭐⭐⭐",
                "产程分期": {
                    "第一产程（宫颈扩张期）": [
                        "从规律宫缩至宫口开全(10cm)",
                        "初产妇：11-12小时",
                        "经产妇：6-8小时",
                        "潜伏期：宫口开大0-3cm（慢）",
                        "活跃期：宫口开大3-10cm（快，≥1.2cm/h）"
                    ],
                    "第二产程（胎儿娩出期）": [
                        "从宫口开全至胎儿娩出",
                        "初产妇：1-2小时（≤2小时）",
                        "经产妇：数分钟至1小时（≤1小时）",
                        "保护会阴：控制胎头娩出速度"
                    ],
                    "第三产程（胎盘娩出期）": [
                        "从胎儿娩出至胎盘娩出",
                        "正常：5-15分钟（≤30分钟）",
                        "胎盘剥离征象：①宫体变硬呈球形，上升至脐上；②阴道少量流血；③脐带自行延长",
                        "检查胎盘：面积、完整性"
                    ]
                },
                "必背数值": {
                    "骨盆外测量": {
                        "髂棘间径（IS）": "23-26cm",
                        "髂嵴间径（IC）": "25-28cm",
                        "骶耻外径（EC）": "18-20cm",
                        "坐骨结节间径（IT）": "8.5-10cm",
                        "出口后矢状径": "8-9cm"
                    },
                    "胎心率": "110-160次/分",
                    "羊水量": "800-1000ml（足月）",
                    "羊水过多": ">2000ml",
                    "羊水过少": "<300ml"
                }
            },

            "产后出血": {
                "考点等级": "⭐⭐⭐⭐⭐",
                "定义": [
                    "胎儿娩出后24小时内出血≥500ml（阴道分娩）",
                    "或≥1000ml（剖宫产）",
                    "产后出血：产科首位死亡原因"
                ],
                "四大原因": {
                    "1. 子宫收缩乏力（70%，最常见）": [
                        "原因：产程延长、羊水过多、多胎、巨大儿",
                        "表现：子宫软、出血多、持续性",
                        "处理：按摩子宫、缩宫素、前列腺素、宫腔填塞"
                    ],
                    "2. 胎盘因素（20%）": [
                        "胎盘滞留：>30分钟未娩出",
                        "胎盘粘连/植入",
                        "胎盘残留",
                        "处理：徒手剥离、刮宫"
                    ],
                    "3. 软产道裂伤（10%）": [
                        "表现：子宫收缩好，但持续出血",
                        "处理：缝合止血"
                    ],
                    "4. 凝血功能障碍": [
                        "DIC、血小板减少等",
                        "处理：针对病因治疗"
                    ]
                },
                "处理原则": [
                    "1. 迅速止血",
                    "2. 纠正休克（边抢救边查找原因）",
                    "3. 防治感染",
                    "药物：缩宫素→前列腺素→麦角新碱",
                    "手术：宫腔填塞→B-Lynch缝合→子宫动脉栓塞/结扎→子宫切除"
                ]
            }
        }

        # 妇科高频考点
        self.gynecology = {
            "功能失调性子宫出血": {
                "考点等级": "⭐⭐⭐⭐⭐",
                "定义": "由于HPO轴功能失调引起的异常子宫出血",
                "分类": {
                    "无排卵型（90%）": [
                        "多见于青春期和绝经过渡期",
                        "表现：月经周期紊乱、经期长短不一",
                        "原因：雌激素持续作用，无孕激素对抗",
                        "子宫内膜：增生期改变，无分泌期",
                        "基础体温：单相型"
                    ],
                    "有排卵型（10%）": [
                        "多见于育龄期",
                        "黄体功能不足：月经周期缩短、经期延长",
                        "黄体萎缩不全：经期延长",
                        "基础体温：双相型"
                    ]
                },
                "治疗": [
                    "青春期：止血→调整周期→促排卵",
                    "育龄期：促排卵为主",
                    "绝经过渡期：止血→调整周期→必要时内膜切除/切除子宫",
                    "止血药物：雌激素、孕激素、雄激素、刮宫"
                ]
            },

            "子宫肌瘤": {
                "考点等级": "⭐⭐⭐⭐⭐",
                "特点": [
                    "育龄妇女最常见的良性肿瘤",
                    "雌激素依赖性肿瘤",
                    "多发性多见"
                ],
                "分类": {
                    "肌壁间肌瘤（60%）": "最常见",
                    "浆膜下肌瘤（20%）": "对月经影响小",
                    "粘膜下肌瘤（10%）": "月经改变明显，易感染、坏死"
                },
                "临床表现": [
                    "经量增多、经期延长（粘膜下最明显）",
                    "腹部包块",
                    "压迫症状：尿频、便秘",
                    "不孕、流产",
                    "变性：红色变性（妊娠期多见）、玻璃样变、囊性变、恶变（<1%）"
                ],
                "治疗": [
                    "观察：无症状、小肌瘤",
                    "药物治疗：GnRH激动剂（术前准备）",
                    "手术指征：",
                    "  - 月经过多导致贫血",
                    "  - 肌瘤>5cm或>妊娠8周子宫大小",
                    "  - 压迫症状明显",
                    "  - 不孕或反复流产",
                    "  - 疑有恶变",
                    "手术方式：肌瘤剔除（年轻）、子宫切除（无生育要求）"
                ]
            },

            "子宫内膜异位症": {
                "考点等级": "⭐⭐⭐⭐⭐",
                "定义": "子宫内膜腺体和间质出现在子宫体以外的部位",
                "好发部位": "卵巢（最常见，形成巧克力囊肿）、子宫骶韧带",
                "三大症状": [
                    "1. 继发性痛经，进行性加重",
                    "2. 性交痛、肛门坠胀",
                    "3. 不孕（约50%）"
                ],
                "体征": [
                    "子宫后位、固定",
                    "子宫骶韧带触痛、结节",
                    "附件区包块、触痛"
                ],
                "辅助检查": [
                    "CA125升高",
                    "超声：卵巢囊性包块（巧囊）",
                    "确诊：腹腔镜检查+活检"
                ],
                "治疗": [
                    "药物治疗：",
                    "  - GnRH激动剂（假绝经疗法）",
                    "  - 孕激素",
                    "  - 口服避孕药（假孕疗法）",
                    "  - 达那唑（假绝经+雄激素）",
                    "手术治疗：",
                    "  - 保守性手术：年轻、要求生育",
                    "  - 根治性手术：年龄大、无生育要求"
                ]
            },

            "卵巢肿瘤": {
                "考点等级": "⭐⭐⭐⭐",
                "良性肿瘤": {
                    "成熟性囊性畸胎瘤": [
                        "最常见的卵巢生殖细胞肿瘤",
                        "好发年龄：20-40岁",
                        "X线：牙齿、骨骼影像（特征性）",
                        "并发症：扭转（最常见）",
                        "恶变率：1-2%"
                    ],
                    "浆液性囊腺瘤": "最常见的卵巢良性肿瘤"
                },
                "恶性肿瘤": {
                    "上皮性卵巢癌": [
                        "卵巢恶性肿瘤中最常见（70-80%）",
                        "浆液性囊腺癌最多见",
                        "CA125：重要肿瘤标志物（>35U/ml）",
                        "转移途径：腹腔种植、淋巴、血行",
                        "手术分期：FIGO分期"
                    ]
                },
                "并发症": [
                    "蒂扭转：突发下腹痛、恶心呕吐",
                    "破裂：外伤、性交后",
                    "感染：发热、腹痛"
                ],
                "治疗": [
                    "良性：手术切除",
                    "恶性：手术+化疗（紫杉醇+卡铂）"
                ]
            },

            "宫颈癌": {
                "考点等级": "⭐⭐⭐⭐⭐",
                "病因": [
                    "HPV感染（主要病因）",
                    "高危型：16、18、31、33、45等",
                    "HPV16：最常见",
                    "HPV18：腺癌相关"
                ],
                "筛查": [
                    "21-29岁：每3年1次细胞学检查",
                    "30-65岁：细胞学+HPV联合筛查，每5年1次",
                    "或单独细胞学，每3年1次",
                    ">65岁：既往筛查正常可停止"
                ],
                "临床表现": [
                    "早期：接触性出血（同房后、妇检后）",
                    "晚期：不规则阴道流血、恶臭白带",
                    "晚期症状：尿频、血尿、肛门坠胀、输尿管梗阻"
                ],
                "病理类型": [
                    "鳞状细胞癌（85-90%，最常见）",
                    "腺癌（10-15%）",
                    "腺鳞癌（<5%）"
                ],
                "分期": [
                    "I期：局限于宫颈",
                    "II期：超出宫颈但未达盆壁",
                    "III期：达盆壁或肾积水",
                    "IV期：超出真骨盆或侵犯膀胱、直肠"
                ],
                "治疗": [
                    "手术治疗：IA2-IIA期",
                    "放疗：IIB-IVA期",
                    "化疗：晚期或复发",
                    "宫颈锥切：IA1期、要求保留生育功能"
                ]
            },

            "子宫内膜癌": {
                "考点等级": "⭐⭐⭐⭐",
                "高危因素": [
                    "肥胖、高血压、糖尿病（三联征）",
                    "未婚、未育、不孕",
                    "绝经延迟",
                    "长期雌激素治疗无孕激素对抗",
                    "多囊卵巢综合征"
                ],
                "临床表现": [
                    "绝经后阴道流血（主要症状）",
                    "绝经前：月经紊乱、经期延长",
                    "阴道排液：血性、浆液性、脓性"
                ],
                "确诊": [
                    "分段诊刮（金标准）",
                    "宫腔镜检查+活检"
                ],
                "治疗": [
                    "首选：手术治疗（全子宫+双附件切除）",
                    "年轻要求保留生育功能：孕激素治疗",
                    "术后辅助：放疗、化疗"
                ]
            },

            "子宫脱垂": {
                "考点等级": "⭐⭐⭐",
                "病因": [
                    "分娩损伤（最重要）",
                    "腹压增高：慢性咳嗽、便秘",
                    "雌激素下降",
                    "先天性盆底组织发育不良"
                ],
                "分度": [
                    "I度轻型：宫颈外口距处女膜<4cm",
                    "I度重型：宫颈外口达处女膜缘",
                    "II度轻型：宫颈脱出阴道口，宫体仍在阴道内",
                    "II度重型：宫颈及部分宫体脱出",
                    "III度：宫颈及宫体全部脱出"
                ],
                "治疗": [
                    "非手术：盆底肌训练（Kegel）、子宫托",
                    "手术：曼氏手术、阴式子宫切除+盆底修补"
                ]
            }
        }

        # 常用药物
        self.medications = {
            "缩宫素": {
                "作用": "促进子宫收缩",
                "用途": "引产、产后出血",
                "禁忌": "产前（胎儿未娩出）静脉注射"
            },
            "硫酸镁": {
                "作用": "解痉、降压、抗惊厥",
                "用途": "妊娠期高血压、子痫前期/子痫、早产（抑制宫缩）",
                "中毒表现": "膝反射消失、呼吸抑制、心跳骤停",
                "解毒": "10%葡萄糖酸钙"
            },
            "米非司酮+米索前列醇": {
                "用途": "药物流产（≤49天）",
                "方法": "米非司酮200mg→36-48小时后米索前列醇600μg"
            },
            "甲氨蝶呤（MTX）": {
                "用途": "异位妊娠保守治疗、滋养细胞疾病",
                "条件": "生命体征平稳、包块<3cm、血hCG<2000"
            }
        }

class ObGynQuizSystem:
    """妇产科学自测系统"""

    def __init__(self, knowledge: ObstetricsGynecologyKnowledge):
        self.kb = knowledge
        self.score = 0
        self.total = 0
        self.wrong_answers = []

    def generate_question(self, category: str, topic: str):
        """生成题目"""
        questions = {
            "正常妊娠": [
                {
                    "question": "初产妇自觉胎动的时间一般是？",
                    "options": ["A. 16-18周", "B. 18-20周", "C. 20-22周", "D. 22-24周"],
                    "answer": "B",
                    "explanation": "初产妇18-20周，经产妇16-18周"
                },
                {
                    "question": "预产期计算方法中，月份的计算是？",
                    "options": ["A. +7或-5", "B. +9或-3", "C. +10或-2", "D. +8或-4"],
                    "answer": "B",
                    "explanation": "月份+9或-3，日期+7"
                },
                {
                    "question": "正常胎心率范围是？",
                    "options": ["A. 100-140次/分", "B. 110-160次/分", "C. 120-160次/分", "D. 110-150次/分"],
                    "answer": "B",
                    "explanation": "胎心率正常范围110-160次/分"
                }
            ],
            "异常妊娠": [
                {
                    "question": "异位妊娠最常见的部位是？",
                    "options": ["A. 卵巢", "B. 腹腔", "C. 输卵管", "D. 宫颈"],
                    "answer": "C",
                    "explanation": "95%为输卵管妊娠，其中壶腹部最多"
                },
                {
                    "question": "前置胎盘的典型临床表现是？",
                    "options": ["A. 有痛性阴道流血", "B. 无痛性阴道流血", "C. 持续性腹痛", "D. 子宫强直性收缩"],
                    "answer": "B",
                    "explanation": "前置胎盘：无痛性阴道流血；胎盘早剥：有痛性阴道流血"
                },
                {
                    "question": "胎盘早剥最严重的并发症是？",
                    "options": ["A. 感染", "B. DIC", "C. 贫血", "D. 休克"],
                    "answer": "B",
                    "explanation": "胎盘早剥可致DIC、产后出血、急性肾衰"
                }
            ],
            "妊娠期高血压": [
                {
                    "question": "硫酸镁中毒时，首先出现的表现是？",
                    "options": ["A. 呼吸抑制", "B. 心跳骤停", "C. 膝反射消失", "D. 血压下降"],
                    "answer": "C",
                    "explanation": "中毒顺序：膝反射消失→呼吸抑制→心跳骤停"
                },
                {
                    "question": "子痫前期解痉的首选药物是？",
                    "options": ["A. 地西泮", "B. 硫酸镁", "C. 苯巴比妥", "D. 氯丙嗪"],
                    "answer": "B",
                    "explanation": "硫酸镁是解痉首选，用于治疗和预防子痫"
                }
            ],
            "产后出血": [
                {
                    "question": "产后出血最常见的原因是？",
                    "options": ["A. 宫缩乏力", "B. 胎盘因素", "C. 软产道裂伤", "D. 凝血功能障碍"],
                    "answer": "A",
                    "explanation": "宫缩乏力占70%，是产后出血最常见原因"
                },
                {
                    "question": "胎盘娩出后持续出血，子宫收缩良好，最可能的原因是？",
                    "options": ["A. 宫缩乏力", "B. 胎盘残留", "C. 软产道裂伤", "D. 凝血障碍"],
                    "answer": "C",
                    "explanation": "子宫收缩好但出血多，考虑软产道裂伤"
                }
            ],
            "子宫肌瘤": [
                {
                    "question": "子宫肌瘤最常见的类型是？",
                    "options": ["A. 浆膜下肌瘤", "B. 肌壁间肌瘤", "C. 粘膜下肌瘤", "D. 宫颈肌瘤"],
                    "answer": "B",
                    "explanation": "肌壁间肌瘤占60%，最常见"
                },
                {
                    "question": "对月经影响最大的子宫肌瘤是？",
                    "options": ["A. 浆膜下肌瘤", "B. 肌壁间肌瘤", "C. 粘膜下肌瘤", "D. 阔韧带肌瘤"],
                    "answer": "C",
                    "explanation": "粘膜下肌瘤月经改变最明显"
                }
            ],
            "宫颈癌": [
                {
                    "question": "宫颈癌最常见的病理类型是？",
                    "options": ["A. 腺癌", "B. 鳞状细胞癌", "C. 腺鳞癌", "D. 未分化癌"],
                    "answer": "B",
                    "explanation": "鳞癌占85-90%"
                },
                {
                    "question": "宫颈癌早期最常见的症状是？",
                    "options": ["A. 阴道排液", "B. 接触性出血", "C. 不规则阴道流血", "D. 下腹痛"],
                    "answer": "B",
                    "explanation": "接触性出血是早期最常见症状"
                }
            ]
        }

        if topic in questions:
            return random.choice(questions[topic])
        return None

    def run_quiz(self, num_questions: int = 10):
        """运行测验"""
        print("\n" + "="*60)
        print("妇产科学随机测验")
        print("="*60)

        all_topics = []
        for category in [self.kb.obstetrics, self.kb.gynecology]:
            all_topics.extend(category.keys())

        for i in range(num_questions):
            topic = random.choice(all_topics)
            question_data = self.generate_question("", topic)

            if not question_data:
                continue

            print(f"\n【题目 {i+1}/{num_questions}】 {topic}")
            print(f"{question_data['question']}")
            for option in question_data['options']:
                print(f"  {option}")

            user_answer = input("\n请输入答案(A/B/C/D): ").strip().upper()

            self.total += 1
            if user_answer == question_data['answer']:
                print("✓ 正确！")
                self.score += 1
            else:
                print(f"✗ 错误！正确答案是: {question_data['answer']}")
                print(f"解析: {question_data['explanation']}")
                self.wrong_answers.append({
                    "topic": topic,
                    "question": question_data['question'],
                    "your_answer": user_answer,
                    "correct_answer": question_data['answer'],
                    "explanation": question_data['explanation']
                })

            input("\n按回车继续...")

        self.show_results()

    def show_results(self):
        """显示测验结果"""
        print("\n" + "="*60)
        print("测验结果")
        print("="*60)
        print(f"总题数: {self.total}")
        print(f"正确数: {self.score}")
        print(f"正确率: {(self.score/self.total*100):.1f}%")

        if self.wrong_answers:
            print("\n错题回顾：")
            for i, item in enumerate(self.wrong_answers, 1):
                print(f"\n{i}. [{item['topic']}] {item['question']}")
                print(f"   你的答案: {item['your_answer']}")
                print(f"   正确答案: {item['correct_answer']}")
                print(f"   解析: {item['explanation']}")

class ObGynLearningSystem:
    """妇产科学学习系统"""

    def __init__(self):
        self.kb = ObstetricsGynecologyKnowledge()
        self.quiz = ObGynQuizSystem(self.kb)

    def show_topic_list(self, category_data: dict, category_name: str):
        """显示分类下的主题列表"""
        print(f"\n{category_name}高频考点：")
        topics = list(category_data.keys())
        for i, topic in enumerate(topics, 1):
            level = category_data[topic].get("考点等级", "")
            print(f"{i}. {topic} {level}")
        return topics

    def show_topic_detail(self, topic_data: dict, topic_name: str):
        """显示主题详细内容"""
        print("\n" + "="*60)
        print(f"{topic_name}")
        print("="*60)

        if "考点等级" in topic_data:
            print(f"\n重要程度: {topic_data['考点等级']}")

        if "核心内容" in topic_data:
            print(f"\n【核心内容】")
            for key, value in topic_data['核心内容'].items():
                print(f"\n▶ {key}:")
                if isinstance(value, list):
                    for item in value:
                        print(f"  • {item}")
                else:
                    print(f"  {value}")

        if "必背数值" in topic_data:
            print(f"\n【必背数值】")
            for key, value in topic_data['必背数值'].items():
                if isinstance(value, dict):
                    print(f"\n▶ {key}:")
                    for k, v in value.items():
                        print(f"  • {k}: {v}")
                else:
                    print(f"  • {key}: {value}")

        for key in topic_data:
            if key not in ["考点等级", "核心内容", "必背数值"]:
                print(f"\n【{key}】")
                value = topic_data[key]
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"\n▶ {k}:")
                        if isinstance(v, list):
                            for item in v:
                                print(f"  • {item}")
                        else:
                            print(f"  {v}")
                elif isinstance(value, list):
                    for item in value:
                        print(f"  • {item}")

    def run(self):
        """运行主程序"""
        print("="*60)
        print("妇产科学高频考点学习系统")
        print("="*60)

        while True:
            print("\n请选择学习模块：")
            print("1. 产科学习")
            print("2. 妇科学习")
            print("3. 常用药物")
            print("4. 随机测验")
            print("5. 退出系统")

            choice = input("\n请输入选项 (1-5): ").strip()

            if choice == "1":
                topics = self.show_topic_list(self.kb.obstetrics, "产科")
                topic_choice = input("\n请选择主题序号（0返回）: ").strip()
                if topic_choice.isdigit() and 1 <= int(topic_choice) <= len(topics):
                    topic_name = topics[int(topic_choice) - 1]
                    self.show_topic_detail(self.kb.obstetrics[topic_name], topic_name)
                    input("\n按回车返回...")

            elif choice == "2":
                topics = self.show_topic_list(self.kb.gynecology, "妇科")
                topic_choice = input("\n请选择主题序号（0返回）: ").strip()
                if topic_choice.isdigit() and 1 <= int(topic_choice) <= len(topics):
                    topic_name = topics[int(topic_choice) - 1]
                    self.show_topic_detail(self.kb.gynecology[topic_name], topic_name)
                    input("\n按回车返回...")

            elif choice == "3":
                print("\n" + "="*60)
                print("常用药物")
                print("="*60)
                for drug_name, drug_info in self.kb.medications.items():
                    print(f"\n▶ {drug_name}")
                    for key, value in drug_info.items():
                        print(f"  • {key}: {value}")
                input("\n按回车返回...")

            elif choice == "4":
                num = input("\n请输入题目数量（默认10题）: ").strip()
                num = int(num) if num.isdigit() else 10
                self.quiz.run_quiz(num)
                input("\n按回车返回...")

            elif choice == "5":
                print("\n祝您考试顺利！加油！💪")
                break

            else:
                print("\n无效选项，请重新选择。")

if __name__ == "__main__":
    system = ObGynLearningSystem()
    system.run()
