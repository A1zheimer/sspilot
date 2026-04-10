"""
SSPilot — AuditAgent 安全审计模块
使用 Qwen2.5-Coder-32B-Instruct (GPTQ INT4) + LoRA 对代码进行安全审计
支持单步推理和工具增强 (Tool-Augmented) 两种模式
输出结构化的审计报告
"""
import json
from datetime import datetime
from config import BATTLE_CONFIG
from model_manager import manager

# ── AuditAgent System Prompt ─────────────────────────────────────

AUDIT_SYSTEM_PROMPT = """你是 SSPilot AuditAgent，一位顶级代码安全审计专家。
你的任务是对给定代码进行全面安全审计，找出所有潜在漏洞。

审计要求：
1. 逐行分析代码逻辑
2. 识别所有安全漏洞（包括但不限于：注入、XSS、路径遍历、硬编码密钥、SSRF、不安全反序列化、逻辑漏洞、信息泄露）
3. 对每个发现的漏洞给出：位置、类型、严重程度、攻击向量、修复建议
4. 如果代码是安全的，也要明确说明

你必须输出 JSON 格式的审计报告。"""

AUDIT_USER_TEMPLATE = """请对以下代码进行全面安全审计：

```python
{code}
```

请以如下 JSON 格式输出审计报告（不要添加 markdown 代码块标记）：
{{
    "overall_risk": "critical|high|medium|low|safe",
    "findings": [
        {{
            "vuln_type": "漏洞类型（如 sqli, xss, path_traversal 等）",
            "severity": "critical|high|medium|low|info",
            "location": "受影响的代码行号或函数名",
            "description": "漏洞的详细描述",
            "attack_vector": "具体的攻击方式和示例",
            "remediation": "详细的修复建议和安全代码示例"
        }}
    ],
    "summary": "一段话总结审计结果",
    "secure_patterns_found": ["代码中使用的安全编码模式（如有）"]
}}"""


def _parse_audit_response(raw: str) -> dict:
    """解析审计模型的输出"""
    text = raw.strip()
    # 去掉可能的 markdown 包裹
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        data = json.loads(text)
        assert "findings" in data
        return data
    except (json.JSONDecodeError, AssertionError):
        return {
            "overall_risk": "unknown",
            "findings": [],
            "summary": raw[:500],
            "secure_patterns_found": [],
            "parse_error": True,
            "raw_output": raw,
        }


def audit_code(code: str, sample_id: str = "", use_tools: bool | None = None) -> dict:
    """
    对单段代码执行安全审计

    Args:
        code: 待审计的代码
        sample_id: 样本 ID（用于日志）
        use_tools: 是否启用工具增强模式，None 时读取 BATTLE_CONFIG

    Returns:
        dict: 结构化审计报告
    """
    if use_tools is None:
        use_tools = BATTLE_CONFIG.get("use_tools", False)

    if use_tools:
        from tool_agent import tool_augmented_audit
        return tool_augmented_audit(code, sample_id=sample_id)

    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": AUDIT_USER_TEMPLATE.format(code=code)},
    ]

    raw = manager.generate(
        "agent",
        messages,
        temperature=BATTLE_CONFIG["temperature_agent"],
        max_new_tokens=BATTLE_CONFIG["max_new_tokens"],
    )

    report = _parse_audit_response(raw)
    report["sample_id"] = sample_id
    report["audit_timestamp"] = datetime.now().isoformat()
    report["model"] = "Qwen2.5-Coder-32B-Int4"
    report["audit_mode"] = "single_shot"

    return report


def audit_batch(samples: list[dict], use_tools: bool | None = None) -> list[dict]:
    """
    批量审计代码样本

    Args:
        samples: VulnGen 生成的样本列表
        use_tools: 是否启用工具增强模式，None 时读取 BATTLE_CONFIG

    Returns:
        list[dict]: 每个样本附带审计报告
    """
    if use_tools is None:
        use_tools = BATTLE_CONFIG.get("use_tools", False)
    mode_label = "Tool-Augmented" if use_tools else "Single-Shot"
    print(f"[AuditAgent] 加载审计模型 ... (模式: {mode_label})")
    manager.load("agent")
    results = []

    for idx, sample in enumerate(samples):
        sid = sample.get("sample_id", f"sample_{idx}")
        print(f"  [{idx+1}/{len(samples)}] 审计 {sid} ({sample.get('vuln_type', '?')}) ...")

        try:
            report = audit_code(sample["code"], sample_id=sid, use_tools=use_tools)
            results.append({
                **sample,
                "audit_report": report,
            })

            # 简要打印结果
            n_findings = len(report.get("findings", []))
            risk = report.get("overall_risk", "?")
            print(f"    ✓ 风险: {risk}, 发现: {n_findings} 个问题")

        except Exception as e:
            print(f"    ✗ 审计失败: {e}")
            results.append({
                **sample,
                "audit_report": {"error": str(e)},
            })

    print(f"[AuditAgent] 批量审计完成, {len(results)}/{len(samples)}")
    return results


if __name__ == "__main__":
    # 快速测试
    test_code = '''
from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route("/search")
def search():
    keyword = request.args.get("q", "")
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    # 拼接查询（存在 SQL 注入）
    query = f"SELECT * FROM products WHERE name LIKE '%{keyword}%'"
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return {"results": results}
'''
    print("[AuditAgent] 测试模式")
    manager.load("agent")
    report = audit_code(test_code, "test_sqli_001")
    print(json.dumps(report, ensure_ascii=False, indent=2))
