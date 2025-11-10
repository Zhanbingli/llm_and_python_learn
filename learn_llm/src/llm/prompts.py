"""
Prompt utilities: declarative templates + helper builders for medical LLM flows.

Design goals:
1. Keep prompt text in one place (no string soup in business logic).
2. Provide light validation (missing variables raise errors early).
3. Offer ready-made builders for RAG, symptom triage, medication lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


# ---------------------------------------------------------------------------
# Core abstractions
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    """
    Text template with named variables, inspired by LangChain-style prompts.
    """

    template: str
    input_variables: Sequence[str]
    partials: Dict[str, str] = field(default_factory=dict)

    def format(self, **kwargs: str) -> str:
        data = {**self.partials, **kwargs}
        missing = [
            name for name in self.input_variables
            if name not in data or data[name] in (None, "")
        ]
        if missing:
            raise ValueError(f"Missing variables for prompt: {missing}")
        return self.template.format(**data)


@dataclass
class ChatPrompt:
    """
    Minimal chat prompt representation for provider-agnostic usage.
    """

    system: str
    user: str
    context: str = ""

    def to_messages(self) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system}]
        if self.context:
            messages.append({"role": "system", "content": self.context})
        messages.append({"role": "user", "content": self.user})
        return messages


# ---------------------------------------------------------------------------
# Base system instructions
# ---------------------------------------------------------------------------


SYSTEM_PROMPTS = {
    "clinical_assistant": (
        "You are a cautious clinical study assistant. "
        "Always cite evidence chunks and highlight uncertainty. "
        "Never provide diagnosis or treatment without disclaimer."
    ),
    "data_cleaner": (
        "You are a data quality analyst. Respond with concise bullet points, "
        "highlighting anomalies and recommending fixes."
    ),
}


# ---------------------------------------------------------------------------
# Template instances
# ---------------------------------------------------------------------------


SYMPTOM_TRIAGE_PROMPT = PromptTemplate(
    template=(
        "患者症状: {symptoms}\n"
        "已知病史: {history}\n"
        "上下文证据:\n{context}\n\n"
        "任务: 1) 提出3个可能原因并给出置信度 (低/中/高)。\n"
        "     2) 建议下一步检查或处理。\n"
        "     3) 加入免责声明: '仅供医学学习参考,勿替代临床决策'."
    ),
    input_variables=["symptoms", "history", "context"],
)


MEDICATION_GUIDE_PROMPT = PromptTemplate(
    template=(
        "药物: {drug_name}\n"
        "适应症: {indications}\n"
        "患者背景: {patient_context}\n\n"
        "请输出:\n"
        "- 推荐剂量范围\n"
        "- 主要不良反应\n"
        "- 禁忌或慎用情况\n"
        "- 参考来源 (若有)\n"
    ),
    input_variables=["drug_name", "indications", "patient_context"],
)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def build_rag_prompt(question: str, context_chunks: Sequence[str]) -> ChatPrompt:
    """
    Formats a retrieval-augmented prompt that quotes provided context verbatim.
    """

    context = "\n---\n".join(context_chunks) if context_chunks else "无可用证据。"
    system = SYSTEM_PROMPTS["clinical_assistant"]
    user = (
        "根据下列医学知识片段回答问题，优先引用内容并标注段落编号。\n"
        f"问题: {question}\n"
        f"证据:\n{context}\n"
    )
    return ChatPrompt(system=system, user=user)


def build_data_clean_prompt(dataset_summary: str) -> ChatPrompt:
    """Quick helper for LLM-based data QA reports."""

    system = SYSTEM_PROMPTS["data_cleaner"]
    user = f"数据摘要:\n{dataset_summary}\n\n请列出三条最关键的清洗建议。"
    return ChatPrompt(system=system, user=user)


__all__ = [
    "PromptTemplate",
    "ChatPrompt",
    "SYMPTOM_TRIAGE_PROMPT",
    "MEDICATION_GUIDE_PROMPT",
    "build_rag_prompt",
    "build_data_clean_prompt",
    "SYSTEM_PROMPTS",
]
