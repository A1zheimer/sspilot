"""
SSPilot — VulnGen 漏洞代码生成器
使用 Qwen2.5-Coder-32B-Int4 生成含隐藏漏洞的代码片段
支持 8 种漏洞类型，每种生成 5 个样本（共 40 个/批）
"""
import json
import random
import time
from pathlib import Path
from datetime import datetime

from config import VULN_TYPES, BATTLE_CONFIG, DATASET_DIR
from model_manager import manager

# ── 每种漏洞类型的生成 prompt 模板 ──────────────────────────────────

VULN_TEMPLATES: dict[str, str] = {
    "sqli": """你是一位资深后端开发者。请编写一个 Python Web 函数（Flask/FastAPI 均可），
实现用户搜索功能。要求：
- 函数接收用户输入的搜索关键词
- 使用 SQL 查询数据库
- 代码看起来功能正确，但要故意包含 SQL 注入漏洞
- 漏洞应该隐蔽，不能一眼看出（例如部分参数化但遗漏关键位置）
- 代码长度 30-80 行，包含适当注释

只输出代码，不要解释。""",

    "xss": """你是一位资深前后端开发者。请编写一个 Python Web 函数，
实现用户评论展示功能。要求：
- 函数处理用户提交的评论并渲染到页面
- 代码看起来做了安全处理，但要故意留下 XSS 漏洞
- 漏洞应该隐蔽（例如：部分转义但遗漏某个输出点、或在 JavaScript 上下文中未正确处理）
- 代码长度 30-80 行，包含适当注释

只输出代码，不要解释。""",

    "path_traversal": """你是一位资深后端开发者。请编写一个 Python 文件下载/读取函数。要求：
- 函数接收文件名参数，从指定目录读取文件
- 代码看起来做了路径安全检查，但要故意留下路径遍历漏洞
- 漏洞应该隐蔽（例如：检查不完整、可被编码绕过、或 race condition）
- 代码长度 30-80 行，包含适当注释

只输出代码，不要解释。""",

    "hardcoded_secret": """你是一位资深后端开发者。请编写一个 Python 配置/认证模块。要求：
- 模块处理 API 密钥、数据库凭证或 JWT 密钥
- 代码看起来使用了环境变量或配置文件，但要故意在某处硬编码了敏感信息
- 漏洞应该隐蔽（例如：默认值中包含真实密钥、注释中残留密码、或 fallback 值是真实凭证）
- 代码长度 30-80 行，包含适当注释

只输出代码，不要解释。""",

    "ssrf": """你是一位资深后端开发者。请编写一个 Python Web 函数，
实现 URL 预览/网页抓取功能。要求：
- 函数接收用户提供的 URL 并获取内容
- 代码看起来做了 URL 验证，但要故意留下 SSRF 漏洞
- 漏洞应该隐蔽（例如：验证可被 DNS rebinding 绕过、或重定向跟随导致内网访问）
- 代码长度 30-80 行，包含适当注释

只输出代码，不要解释。""",

    "unsafe_deser": """你是一位资深后端开发者。请编写一个 Python 数据处理模块。要求：
- 模块处理序列化数据（pickle、yaml、json 等）的加载
- 代码看起来功能正常，但要故意使用不安全的反序列化方式
- 漏洞应该隐蔽（例如：yaml.load 而非 safe_load、pickle.loads 处理用户数据、
  或自定义反序列化器存在注入点）
- 代码长度 30-80 行，包含适当注释

只输出代码，不要解释。""",

    "logic": """你是一位资深后端开发者。请编写一个 Python 权限控制/业务逻辑函数。要求：
- 函数实现权限检查、支付验证、或资源访问控制
- 代码看起来逻辑正确，但要故意包含逻辑漏洞
- 漏洞应该隐蔽（例如：TOCTOU 竞态、整数溢出导致绕过、或条件判断遗漏边界情况）
- 代码长度 30-80 行，包含适当注释

只输出代码，不要解释。""",

    "info_leak": """你是一位资深后端开发者。请编写一个 Python Web 错误处理/日志模块。要求：
- 模块处理异常捕获和错误响应
- 代码看起来处理得当，但要故意泄露敏感信息
- 漏洞应该隐蔽（例如：错误响应中包含堆栈跟踪、日志记录了敏感字段、
  或 debug 模式未在生产环境关闭）
- 代码长度 30-80 行，包含适当注释

只输出代码，不要解释。""",
}

# ── 元 prompt：包裹单个漏洞模板，产出结构化 JSON ──────────────────

META_PROMPT = """你将扮演一个漏洞代码生成器。请严格按要求生成代码。

{task_prompt}

请以如下 JSON 格式输出（不要添加 markdown 代码块标记）：
{{
    "vuln_type": "{vuln_type}",
    "language": "python",
    "code": "<完整代码，换行用实际换行>",
    "ground_truth": {{
        "vuln_line_range": [起始行号, 结束行号],
        "vuln_description": "一句话描述漏洞原理",
        "attack_vector": "一句话描述攻击方式"
    }},
    "difficulty": "{difficulty}"
}}"""


def _pick_difficulty() -> str:
    """随机选择难度，偏向 medium"""
    return random.choices(
        ["easy", "medium", "hard"],
        weights=[0.2, 0.5, 0.3],
        k=1
    )[0]


def _build_messages(vuln_type: str) -> tuple[list[dict], str]:
    """构建单个漏洞生成请求的 messages"""
    difficulty = _pick_difficulty()
    diff_hint = {
        "easy": "漏洞可以相对明显，适合入门审计。",
        "medium": "漏洞应该有一定隐蔽性，需要仔细审查才能发现。",
        "hard": "漏洞应该非常隐蔽，使用高级技巧隐藏，即使资深安全工程师也可能遗漏。",
    }
    task_prompt = VULN_TEMPLATES[vuln_type] + f"\n\n难度要求：{diff_hint[difficulty]}"
    prompt = META_PROMPT.format(
        task_prompt=task_prompt,
        vuln_type=vuln_type,
        difficulty=difficulty,
    )
    messages = [
        {"role": "system", "content": "你是一个专业的安全研究工具，用于生成供审计训练的漏洞代码样本。"},
        {"role": "user", "content": prompt},
    ]
    return messages, difficulty


def _parse_response(raw: str, vuln_type: str, difficulty: str) -> dict | None:
    """解析模型输出的 JSON，容错处理"""
    # 去掉可能的 markdown 代码块包裹
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        data = json.loads(text)
        # 校验必要字段
        assert "code" in data and len(data["code"]) > 50
        assert "ground_truth" in data
        data["vuln_type"] = vuln_type
        data["difficulty"] = difficulty
        return data
    except (json.JSONDecodeError, AssertionError, KeyError):
        # 回退：尝试提取代码块
        return {
            "vuln_type": vuln_type,
            "language": "python",
            "code": raw,
            "ground_truth": {"vuln_description": "parse_failed", "vuln_line_range": [], "attack_vector": ""},
            "difficulty": difficulty,
            "parse_error": True,
        }


def generate_batch(
    batch_size: int = BATTLE_CONFIG["batch_size"],
    output_path: Path | None = None,
) -> list[dict]:
    """
    生成一批漏洞代码样本

    流程：
    1. 加载 VulnGen 模型 (Qwen2.5-Coder-32B-Int4, ~16GB)
    2. 每种漏洞类型生成 batch_size/8 个样本
    3. 卸载模型释放显存

    Returns:
        list[dict]: 生成的漏洞样本列表
    """
    per_type = max(1, batch_size // len(VULN_TYPES))
    samples = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path is None:
        output_path = DATASET_DIR / f"vulngen_{ts}.jsonl"

    print(f"[VulnGen] 开始生成 {per_type * len(VULN_TYPES)} 个漏洞样本 ...")
    print(f"[VulnGen] 加载模型 ...")
    manager.load("vulngen")

    for vuln_type in VULN_TYPES:
        print(f"  [{vuln_type}] 生成 {per_type} 个样本 ...")
        for i in range(per_type):
            messages, difficulty = _build_messages(vuln_type)
            try:
                raw = manager.generate(
                    "vulngen",
                    messages,
                    temperature=BATTLE_CONFIG["temperature_vulngen"],
                )
                sample = _parse_response(raw, vuln_type, difficulty)
                if sample:
                    sample["sample_id"] = f"{vuln_type}_{i:03d}_{ts}"
                    sample["gen_timestamp"] = datetime.now().isoformat()
                    samples.append(sample)
                    print(f"    ✓ [{vuln_type}#{i}] {difficulty} — "
                          f"{len(sample['code'])} chars")
            except Exception as e:
                print(f"    ✗ [{vuln_type}#{i}] 生成失败: {e}")

    # 保存 JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[VulnGen] 完成! 共 {len(samples)} 个样本 → {output_path}")

    # 卸载模型
    print("[VulnGen] 卸载模型释放显存 ...")
    manager.unload("vulngen")

    return samples


if __name__ == "__main__":
    generate_batch(batch_size=8)  # 快速测试：每种类型 1 个
