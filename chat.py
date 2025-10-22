import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
from xml.etree import ElementTree as ET
import json
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import sys

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)


# ========== 工具链设计 ==========

class ToolPriority(Enum):
    """工具优先级"""
    HIGH = 1      # 必须立即执行
    MEDIUM = 2    # 可以并行执行
    LOW = 3       # 可以延迟执行


@dataclass
class ToolDependency:
    """工具依赖关系"""
    name: str
    depends_on: List[str] = None  # 依赖的工具
    can_parallel: bool = True      # 是否可以并行
    priority: ToolPriority = ToolPriority.MEDIUM
    timeout: int = 30              # 超时时间（秒）


class ToolChain:
    """工具链管理器"""

    def __init__(self):
        # 定义工具依赖关系
        self.dependencies = {
            "search_pubmed": ToolDependency(
                name="search_pubmed",
                depends_on=None,  # 无依赖，可以立即执行
                can_parallel=True,
                priority=ToolPriority.HIGH
            ),
            "fetch_pubmed_details": ToolDependency(
                name="fetch_pubmed_details",
                depends_on=["search_pubmed"],  # 依赖搜索结果
                can_parallel=False,
                priority=ToolPriority.MEDIUM,
                timeout=45
            ),
            "fetch_full_text_links": ToolDependency(
                name="fetch_full_text_links",
                depends_on=["search_pubmed"],
                can_parallel=True,  # 可以并行获取多篇文章的链接
                priority=ToolPriority.LOW,
                timeout=20
            )
        }

    def can_execute_parallel(self, tool_calls: List[Dict]) -> bool:
        """判断工具调用是否可以并行执行"""
        if len(tool_calls) <= 1:
            return False

        # 检查所有工具是否都支持并行
        for call in tool_calls:
            tool_name = call['function']['name']
            if tool_name in self.dependencies:
                if not self.dependencies[tool_name].can_parallel:
                    return False

        return True

    def group_by_dependency(self, tool_calls: List[Dict]) -> List[List[Dict]]:
        """按依赖关系分组工具调用"""
        groups = []
        independent = []  # 无依赖的工具
        dependent = []    # 有依赖的工具

        for call in tool_calls:
            tool_name = call['function']['name']
            dep = self.dependencies.get(tool_name)

            if not dep or not dep.depends_on:
                independent.append(call)
            else:
                dependent.append(call)

        if independent:
            groups.append(independent)
        if dependent:
            groups.append(dependent)

        return groups


# ========== 异步 HTTP 请求（支持并行）==========

class AsyncPubMedClient:
    """异步 PubMed 客户端"""

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_pubmed(self, query: str, max_results: int = 10, sort: str = "relevance") -> Dict:
        """异步搜索"""
        url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": sort
        }

        try:
            async with self.session.get(url, params=params, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()

                pmids = data.get('esearchresult', {}).get('idlist', [])
                total_count = int(data.get('esearchresult', {}).get('count', 0))

                return {
                    "success": True,
                    "pmids": pmids,
                    "count": len(pmids),
                    "total_available": total_count,
                    "query": query
                }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    async def fetch_details(self, pmids: List[str]) -> Dict:
        """异步获取详情"""
        if not pmids:
            return {"success": False, "error": "未提供 PMID"}

        pmid_string = ','.join(pmids)
        url = f"{self.base_url}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid_string,
            "retmode": "xml"
        }

        try:
            async with self.session.get(url, params=params, timeout=15) as response:
                response.raise_for_status()
                content = await response.read()
                root = ET.fromstring(content)

                articles = []
                for article in root.findall('.//PubmedArticle'):
                    pmid = article.findtext('.//PMID')
                    title = article.findtext('.//ArticleTitle') or "无标题"

                    # 提取摘要
                    abstract_parts = article.findall('.//AbstractText')
                    abstract_full = ' '.join([part.text or '' for part in abstract_parts])
                    abstract = abstract_full[:800] + "..." if len(abstract_full) > 800 else abstract_full

                    # 提取作者
                    authors = []
                    for author in article.findall('.//Author'):
                        last = author.findtext('.//LastName', '')
                        first = author.findtext('.//ForeName', '')
                        if last:
                            authors.append(f"{first} {last}".strip())

                    journal = article.findtext('.//Journal/Title', '未知期刊')
                    year = article.findtext('.//PubDate/Year', '未知年份')

                    # DOI
                    doi = None
                    for article_id in article.findall('.//ArticleId'):
                        if article_id.get('IdType') == 'doi':
                            doi = article_id.text

                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors[:5],
                        "authors_display": ", ".join(authors[:3]) + ("等" if len(authors) > 3 else ""),
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "doi_url": f"https://doi.org/{doi}" if doi else None
                    })

                return {
                    "success": True,
                    "articles": articles,
                    "count": len(articles)
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def fetch_full_text_link(self, pmid: str) -> Dict:
        """异步获取全文链接（单个）"""
        url = f"{self.base_url}/elink.fcgi"
        params = {
            "dbfrom": "pubmed",
            "id": pmid,
            "retmode": "json",
            "cmd": "prlinks"
        }

        try:
            async with self.session.get(url, params=params, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()

                links = []
                for linkset in data.get('linksets', []):
                    for idurl in linkset.get('idurllist', []):
                        for obj in idurl.get('objurls', []):
                            url_value = obj.get('url', '').strip()
                            if url_value:
                                links.append({
                                    "provider": obj.get('provider', {}).get('name', 'Unknown'),
                                    "url": url_value
                                })

                return {
                    "success": True,
                    "pmid": pmid,
                    "full_text_links": links,
                    "has_links": len(links) > 0
                }
        except Exception as e:
            return {"success": False, "error": str(e), "pmid": pmid}


# ========== 并行执行器 ==========

class ParallelToolExecutor:
    """并行工具执行器"""

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.tool_chain = ToolChain()

    async def execute_parallel_async(self, tool_calls: List[Dict],
                                     async_functions: Dict[str, Callable]) -> List[Dict]:
        """异步并行执行（用于 I/O 密集型任务）"""
        print(f"\n🔄 并行执行 {len(tool_calls)} 个工具...")

        tasks = []
        for call in tool_calls:
            function_name = call['function']['name']
            function_args = json.loads(call['function']['arguments'])

            if function_name in async_functions:
                task = async_functions[function_name](**function_args)
                tasks.append((call['id'], function_name, task))

        # 并行执行所有任务
        results = []
        for call_id, func_name, task in tasks:
            try:
                result = await task
                results.append({
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
                print(f"  ✓ {func_name} 完成")
            except Exception as e:
                results.append({
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
                })
                print(f"  ✗ {func_name} 失败: {e}")

        return results

    def execute_parallel_threads(self, tool_calls: List[Dict],
                                 sync_functions: Dict[str, Callable]) -> List[Dict]:
        """线程并行执行（用于 CPU 密集型或同步任务）"""
        print(f"\n🔄 并行执行 {len(tool_calls)} 个工具（线程池）...")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_call = {}

            for call in tool_calls:
                function_name = call['function']['name']
                function_args = json.loads(call['function']['arguments'])

                if function_name in sync_functions:
                    future = executor.submit(sync_functions[function_name], **function_args)
                    future_to_call[future] = (call['id'], function_name)

            # 收集结果
            for future in as_completed(future_to_call):
                call_id, func_name = future_to_call[future]
                try:
                    result = future.result()
                    results.append({
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
                    print(f"  ✓ {func_name} 完成")
                except Exception as e:
                    results.append({
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
                    })
                    print(f"  ✗ {func_name} 失败: {e}")

        return results


# ========== 流式输出管理器 ==========

class StreamingOutput:
    """流式输出管理器"""

    def __init__(self):
        self.buffer = []

    def write(self, text: str, delay: float = 0.02):
        """模拟打字效果"""
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)

    def write_line(self, text: str, delay: float = 0.02):
        """输出一行"""
        self.write(text, delay)
        print()  # 换行

    def write_section(self, title: str, content: str):
        """输出章节"""
        print(f"\n{'='*70}")
        self.write_line(f"  {title}", 0.01)
        print(f"{'='*70}\n")
        self.write_line(content, 0.02)

    def write_progress(self, current: int, total: int, message: str = ""):
        """进度条"""
        bar_length = 40
        progress = current / total
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)

        sys.stdout.write(f'\r⏳ [{bar}] {current}/{total} {message}')
        sys.stdout.flush()

        if current == total:
            print()  # 完成后换行


# ========== 同步包装函数（兼容旧接口）==========

def search_pubmed(query: str, max_results: int = 10, sort: str = "relevance") -> Dict:
    """同步版本的搜索（用于向后兼容）"""
    async def _search():
        async with AsyncPubMedClient() as client:
            return await client.search_pubmed(query, max_results, sort)

    return asyncio.run(_search())


def fetch_pubmed_details(pmids: List[str]) -> Dict:
    """同步版本的获取详情"""
    async def _fetch():
        async with AsyncPubMedClient() as client:
            return await client.fetch_details(pmids)

    return asyncio.run(_fetch())


def fetch_full_text_links(pmid: str) -> Dict:
    """同步版本的获取全文链接"""
    async def _fetch():
        async with AsyncPubMedClient() as client:
            return await client.fetch_full_text_link(pmid)

    return asyncio.run(_fetch())


# ========== AI Agent 工具定义（优化版）==========

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": "在 PubMed 数据库中搜索科学文献。支持并行搜索多个主题。返回相关文章的 PMID 列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词（如：'2型糖尿病治疗'、'COVID-19 疫苗效果'）"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回的最大结果数（默认: 10）",
                        "default": 10
                    },
                    "sort": {
                        "type": "string",
                        "description": "排序方式：'relevance'（相关性）或 'pub_date'（发表日期）",
                        "default": "relevance",
                        "enum": ["relevance", "pub_date"]
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pubmed_details",
            "description": "获取指定 PMID 文章的详细信息（标题、摘要、作者、期刊等）。会在搜索完成后自动调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "PubMed ID 列表"
                    }
                },
                "required": ["pmids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_full_text_links",
            "description": "获取文章的全文链接（如果可用）。支持并行获取多篇文章的链接。注意：许多文章需要订阅或机构访问权限。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {
                        "type": "string",
                        "description": "PubMed ID"
                    }
                },
                "required": ["pmid"]
            }
        }
    }
]


# ========== 优化的 Agent 主循环（支持流式输出）==========

async def run_agent_async(user_query: str,
                          conversation_history: List[Dict] = None,
                          enable_streaming: bool = True) -> str:
    """
    异步 AI Agent（支持并行工具调用和流式输出）

    Args:
        user_query: 用户查询
        conversation_history: 对话历史
        enable_streaming: 是否启用流式输出
    """
    streamer = StreamingOutput()
    executor = ParallelToolExecutor(max_workers=5)
    tool_chain = ToolChain()

    # 初始化消息
    if conversation_history is None:
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的医学文献检索助手。

**核心能力：**
1. 理解用户的医学研究需求
2. 智能搜索 PubMed 数据库
3. 支持并行搜索多个主题（例如同时搜索多种疾病）
4. 用清晰、专业但易懂的语言呈现结果

**工具使用策略：**
- 如果用户要求比较多个主题，同时调用多个 search_pubmed
- 如果需要获取多篇文章的全文链接，并行调用 fetch_full_text_links
- 优先使用工具获取最新数据，而不是依赖训练数据

**回复风格：**
- 先给出简短总结
- 用结构化的方式呈现结果
- 突出关键发现和统计数据
- 提供可操作的建议"""
            }
        ]
    else:
        messages = conversation_history.copy()

    messages.append({"role": "user", "content": user_query})

    # 显示查询
    if enable_streaming:
        streamer.write_section("🔍 您的问题", user_query)
    else:
        print(f"\n{'='*70}")
        print(f"🔍 查询: {user_query}")
        print(f"{'='*70}\n")

    max_iterations = 5
    iteration = 0

    # 异步 PubMed 客户端
    async with AsyncPubMedClient() as pubmed_client:
        async_functions = {
            "search_pubmed": pubmed_client.search_pubmed,
            "fetch_pubmed_details": pubmed_client.fetch_details,
            "fetch_full_text_links": pubmed_client.fetch_full_text_link
        }

        while iteration < max_iterations:
            iteration += 1

            if iteration > 1 and enable_streaming:
                streamer.write_line(f"\n⚙️  处理步骤 {iteration}...", 0.01)

            # 调用 LLM
            try:
                response = client.chat.completions.create(
                    model='deepseek-chat',
                    messages=messages,
                    tools=tools,
                    temperature=0.7
                )
            except Exception as e:
                print(f"\n❌ AI 调用失败: {str(e)}")
                return "抱歉，系统遇到错误，请稍后重试。"

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # 检查工具调用
            if assistant_message.tool_calls:
                tool_calls = assistant_message.tool_calls

                # 判断是否可以并行执行
                can_parallel = tool_chain.can_execute_parallel(tool_calls)

                if can_parallel and len(tool_calls) > 1:
                    # 并行执行
                    print(f"\n🚀 检测到 {len(tool_calls)} 个独立任务，启动并行执行模式")

                    # 转换为标准格式
                    calls_data = []
                    for call in tool_calls:
                        calls_data.append({
                            'id': call.id,
                            'function': {
                                'name': call.function.name,
                                'arguments': call.function.arguments
                            }
                        })

                    # 并行执行
                    results = await executor.execute_parallel_async(calls_data, async_functions)

                    # 添加所有结果到消息历史
                    for result in results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "name": result["name"],
                            "content": result["content"]
                        })

                else:
                    # 串行执行
                    print(f"\n⏳ 顺序执行 {len(tool_calls)} 个工具...")

                    for idx, tool_call in enumerate(tool_calls, 1):
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        if enable_streaming:
                            streamer.write_progress(idx, len(tool_calls), f"执行 {function_name}")
                        else:
                            print(f"  [{idx}/{len(tool_calls)}] 执行 {function_name}...")

                        # 执行工具
                        try:
                            if function_name in async_functions:
                                result = await async_functions[function_name](**function_args)
                            else:
                                result = {"success": False, "error": "未知工具"}

                            # 显示简要结果
                            if result.get('success'):
                                if function_name == "search_pubmed":
                                    print(f"    ✓ 找到 {result.get('count', 0)} 篇文章（共 {result.get('total_available', 0)} 篇可用）")
                                elif function_name == "fetch_pubmed_details":
                                    print(f"    ✓ 获取 {result.get('count', 0)} 篇文章详情")
                                elif function_name == "fetch_full_text_links":
                                    links_count = len(result.get('full_text_links', []))
                                    print(f"    ✓ 找到 {links_count} 个全文链接")
                            else:
                                print(f"    ⚠️  {result.get('error', '未知错误')}")

                            # 添加到消息历史
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": json.dumps(result, ensure_ascii=False)
                            })

                        except Exception as e:
                            error_result = {"success": False, "error": str(e)}
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": json.dumps(error_result, ensure_ascii=False)
                            })
                            print(f"    ✗ 执行失败: {e}")

            else:
                # 没有工具调用，返回最终结果
                if enable_streaming:
                    streamer.write_section("📋 检索结果", "")
                    streamer.write_line(assistant_message.content, 0.02)
                else:
                    print(f"\n{'='*70}")
                    print("📋 检索结果:")
                    print(f"{'='*70}")
                    print(assistant_message.content)
                    print(f"{'='*70}\n")

                return assistant_message.content

        print("\n⚠️  处理时间过长，请简化查询后重试")
        return "抱歉，查询超时。"


def run_agent(user_query: str,
              conversation_history: List[Dict] = None,
              enable_streaming: bool = True) -> str:
    """同步包装器"""
    return asyncio.run(run_agent_async(user_query, conversation_history, enable_streaming))


# ========== 增强的交互界面 ==========

class EnhancedCLI:
    """增强的命令行界面"""

    def __init__(self):
        self.conversation_history = None
        self.search_history = []
        self.streamer = StreamingOutput()

    def show_welcome(self):
        """欢迎界面"""
        banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        🏥 PubMed 医学文献智能检索助手 (增强版)                            ║
║                                                                      ║
║        ✨ 新功能:                                                     ║
║           • 并行工具调用 - 同时搜索多个主题                              ║
║           • 智能工具链 - 自动优化执行顺序                                ║
║           • 流式输出 - 实时显示处理进度                                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """
        print(banner)

        print("\n💡 使用提示:")
        print("  • 直接输入医学问题（支持中英文）")
        print("  • 输入 'help' 查看示例")
        print("  • 输入 'history' 查看搜索历史")
        print("  • 输入 'clear' 开始新对话")
        print("  • 输入 'quit' 退出程序")
        print("\n" + "="*70 + "\n")

    def show_examples(self):
        """示例查询"""
        examples = [
            ("基础搜索", "搜索关于阿尔茨海默病最新治疗方法的文献"),
            ("并行搜索", "比较糖尿病和高血压在2024年的研究数量"),
            ("多主题对比", "同时搜索COVID-19疫苗、治疗药物和后遗症的研究"),
            ("深度检索", "找5篇关于CRISPR基因编辑在癌症治疗中的应用，并获取全文链接"),
            ("趋势分析", "分析2023-2024年间人工智能在医学影像诊断领域的研究趋势")
        ]

        print("\n" + "="*70)
        print("📚 示例查询（展示并行和工具链能力）:")
        print("="*70)

        for i, (category, example) in enumerate(examples, 1):
            print(f"\n  {i}. [{category}]")
            print(f"     {example}")

        print("\n" + "="*70 + "\n")

    def show_history(self):
        """显示搜索历史"""
        if not self.search_history:
            print("\n📭 暂无搜索历史\n")
            return

        print("\n" + "="*70)
        print("📜 搜索历史（最近5条）:")
        print("="*70)

        for i, item in enumerate(self.search_history[-5:], 1):
            print(f"\n  {i}. {item['query']}")
            print(f"     时间: {item['time']}")
            if 'summary' in item:
                print(f"     结果: {item['summary']}")

        print("\n" + "="*70 + "\n")

    def run(self):
        """主循环"""
        self.show_welcome()

        while True:
            try:
                user_input = input("🔍 您的问题: ").strip()

                if not user_input:
                    continue

                # 命令处理
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 感谢使用，再见！\n")
                    break

                if user_input.lower() in ['help', '帮助', 'h']:
                    self.show_examples()
                    continue

                if user_input.lower() in ['history', '历史', 'hist']:
                    self.show_history()
                    continue

                if user_input.lower() in ['clear', '清空', 'new']:
                    self.conversation_history = None
                    print("\n🔄 已开始新对话\n")
                    continue

                # 执行查询
                start_time = time.time()
                result = run_agent(
                    user_input,
                    self.conversation_history,
                    enable_streaming=True
                )
                elapsed = time.time() - start_time

                # 记录历史
                self.search_history.append({
                    "query": user_input,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": f"耗时 {elapsed:.1f}秒"
                })

                # 询问后续操作
                print("\n" + "-"*70)
                print(f"⏱️  查询耗时: {elapsed:.1f} 秒")
                print("-"*70)

                next_action = input("\n💬 继续提问? (回车继续 / 'new' 新对话 / 'quit' 退出): ").strip()

                if next_action.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 感谢使用，再见！\n")
                    break
                elif next_action.lower() in ['new', 'clear']:
                    self.conversation_history = None
                    print("\n🔄 已开始新对话\n")

            except KeyboardInterrupt:
                print("\n\n👋 程序已中断，再见！\n")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}\n")


# ========== 性能测试函数 ==========

async def benchmark_parallel_vs_serial():
    """性能对比测试"""
    print("\n" + "="*70)
    print("⚡ 性能测试：并行 vs 串行")
    print("="*70)

    queries = [
        "diabetes 2024",
        "Alzheimer disease 2024",
        "COVID-19 vaccine 2024"
    ]

    # 串行测试
    print("\n📊 串行执行:")
    serial_start = time.time()
    async with AsyncPubMedClient() as client:
        for query in queries:
            result = await client.search_pubmed(query, max_results=5)
            print(f"  ✓ {query}: {result.get('count', 0)} 篇")
    serial_time = time.time() - serial_start
    print(f"  总耗时: {serial_time:.2f} 秒")

    # 并行测试
    print("\n🚀 并行执行:")
    parallel_start = time.time()
    async with AsyncPubMedClient() as client:
        tasks = [client.search_pubmed(q, max_results=5) for q in queries]
        results = await asyncio.gather(*tasks)
        for query, result in zip(queries, results):
            print(f"  ✓ {query}: {result.get('count', 0)} 篇")
    parallel_time = time.time() - parallel_start
    print(f"  总耗时: {parallel_time:.2f} 秒")

    # 对比
    speedup = serial_time / parallel_time
    print(f"\n⚡ 性能提升: {speedup:.1f}x")
    print(f"   节省时间: {serial_time - parallel_time:.2f} 秒 ({(1 - parallel_time/serial_time)*100:.1f}%)")
    print("="*70 + "\n")


# ========== 主程序 ==========

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # 性能测试模式
        asyncio.run(benchmark_parallel_vs_serial())
    else:
        # 正常交互模式
        cli = EnhancedCLI()
        cli.run()
