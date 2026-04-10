"""
SSPilot — Judge 漏洞裁判模块
使用 NVIDIA Nemotron-3-Nano-30B-A3B (NVFP4) 评估审计质量
通过 vLLM 推理服务进行深度评估，输出 4 维度评分
"""
import json
from datetime import datetime
from config import BATTLE_CONFIG
from model_manager import manager

# ── Judge System Prompt ──────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """你是 SSPilot Judge，一位权威的代码安全审计评估专家。
你的任务是评估 AuditAgent 对漏洞代码的审计质量。

你会收到：
1. 原始代码（含已知漏洞）
2. 漏洞的 ground truth（真实漏洞信息）
3. AuditAgent 的审计报告

请从以下 4 个维度进行评分（每项 0-10 分）：

**Detection（检出能力）**：是否正确识别了代码中的漏洞？
- 10: 精准定位所有漏洞
- 7-9: 找到主要漏洞，可能遗漏次要问题
- 4-6: 部分检出，有重要遗漏
- 1-3: 大部分漏洞未检出
- 0: 完全未检出或判断为安全

**Precision（精确度）**：审计结果是否准确，有无误报？
- 10: 无误报，所有发现均为真实问题
- 7-9: 极少误报
- 4-6: 有一定误报但主要发现准确
- 1-3: 误报较多
- 0: 全是误报

**Depth（深度）**：对漏洞原理和攻击向量的分析是否深入？
- 10: 深入分析原理，给出完整攻击链
- 7-9: 分析较好，攻击向量准确
- 4-6: 表面分析，缺乏深度
- 1-3: 仅泛泛而谈
- 0: 无有效分析

**Remediation（修复建议）**：修复建议是否可行且完整？
- 10: 给出完整、可直接使用的修复代码
- 7-9: 修复方向正确，建议具体
- 4-6: 有修复建议但不够完整
- 1-3: 修复建议含糊或不正确
- 0: 无修复建议

请深入思考后给出评分。"""

JUDGE_USER_TEMPLATE = """## 原始代码

```python
{code}
```

## 漏洞 Ground Truth

- 漏洞类型: {vuln_type}
- 漏洞描述: {vuln_description}
- 攻击向量: {attack_vector}
- 漏洞位置: {vuln_line_range}
- 难度: {difficulty}

## AuditAgent 审计报告

```json
{audit_report}
```

---

请以如下 JSON 格式输出评估结果（不要添加 markdown 代码块标记）：
{{
    "scores": {{
        "detection": <0-10>,
        "precision": <0-10>,
        "depth": <0-10>,
        "remediation": <0-10>
    }},
    "total_score": <0-40>,
    "grade": "S|A|B|C|D|F",
    "thinking_summary": "你的评估推理过程摘要",
    "strengths": ["审计报告的优点"],
    "weaknesses": ["审计报告的不足"],
    "missed_vulns": ["AuditAgent 遗漏的漏洞（如有）"],
    "false_positives": ["AuditAgent 的误报（如有）"]
}}

其中 grade 按总分分级：
S: 36-40, A: 28-35, B: 20-27, C: 12-19, D: 4-11, F: 0-3"""


def _parse_judge_response(raw: str) -> dict:
    """解析 Judge 模型的输出"""
    text = raw.strip()

    # Qwen3-thinking 可能输出 <think>...</think> 块
    thinking = ""
    if "<think>" in text:
        think_start = text.index("<think>") + len("<think>")
        think_end = text.index("</think>") if "</think>" in text else len(text)
        thinking = text[think_start:think_end].strip()
        text = text[think_end:]
        if text.startswith("</think>"):
            text = text[len("</think>"):]
        text = text.strip()

    # 去掉 markdown 包裹
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        data = json.loads(text)
        assert "scores" in data
        # 确保分数在范围内
        for key in ["detection", "precision", "depth", "remediation"]:
            data["scores"][key] = max(0, min(10, int(data["scores"].get(key, 0))))
        data["total_score"] = sum(data["scores"].values())
        data["thinking_chain"] = thinking  # 保留 thinking chain
        return data
    except (json.JSONDecodeError, AssertionError, ValueError):
        return {
            "scores": {"detection": 0, "precision": 0, "depth": 0, "remediation": 0},
            "total_score": 0,
            "grade": "F",
            "thinking_summary": thinking or "parse_failed",
            "thinking_chain": thinking,
            "parse_error": True,
            "raw_output": raw[:1000],
        }


def _compute_grade(total: int) -> str:
    """根据总分计算等级"""
    if total >= 36: return "S"
    if total >= 28: return "A"
    if total >= 20: return "B"
    if total >= 12: return "C"
    if total >= 4:  return "D"
    return "F"


def judge_audit(sample: dict, audit_report: dict) -> dict:
    """
    评估单次审计结果

    Args:
        sample: VulnGen 样本（含 code, ground_truth 等）
        audit_report: AuditAgent 的审计报告

    Returns:
        dict: Judge 评分结果
    """
    gt = sample.get("ground_truth", {})

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            code=sample.get("code", ""),
            vuln_type=sample.get("vuln_type", "unknown"),
            vuln_description=gt.get("vuln_description", "N/A"),
            attack_vector=gt.get("attack_vector", "N/A"),
            vuln_line_range=str(gt.get("vuln_line_range", [])),
            difficulty=sample.get("difficulty", "medium"),
            audit_report=json.dumps(audit_report, ensure_ascii=False, indent=2)[:3000],
        )},
    ]

    raw = manager.generate(
        "judge",
        messages,
        temperature=BATTLE_CONFIG["temperature_judge"],
        max_new_tokens=BATTLE_CONFIG["max_new_tokens"],
    )

    result = _parse_judge_response(raw)
    result["grade"] = _compute_grade(result["total_score"])
    result["judge_timestamp"] = datetime.now().isoformat()
    result["model"] = "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"

    return result


def judge_batch(battle_results: list[dict]) -> list[dict]:
    """
    批量评估审计结果

    Args:
        battle_results: 含 audit_report 的样本列表

    Returns:
        list[dict]: 附带 judge 评分的完整结果
    """
    print(f"[Judge] 加载评估模型 ...")
    manager.load("judge")
    results = []

    for idx, item in enumerate(battle_results):
        sid = item.get("sample_id", f"sample_{idx}")
        print(f"  [{idx+1}/{len(battle_results)}] 评估 {sid} ...")

        try:
            audit = item.get("audit_report", {})
            if audit.get("error"):
                print(f"    ⊘ 跳过（审计失败的样本）")
                results.append({**item, "judge_result": {"skipped": "audit_failed"}})
                continue

            judge_result = judge_audit(item, audit)
            results.append({
                **item,
                "judge_result": judge_result,
            })

            score = judge_result["total_score"]
            grade = judge_result["grade"]
            scores = judge_result["scores"]
            print(f"    ✓ {grade} ({score}/40) "
                  f"D:{scores['detection']} P:{scores['precision']} "
                  f"Dp:{scores['depth']} R:{scores['remediation']}")

        except Exception as e:
            print(f"    ✗ 评估失败: {e}")
            results.append({
                **item,
                "judge_result": {"error": str(e)},
            })

    # 统计
    valid = [r for r in results if "scores" in r.get("judge_result", {})]
    if valid:
        avg = sum(r["judge_result"]["total_score"] for r in valid) / len(valid)
        print(f"[Judge] 完成! {len(valid)}/{len(battle_results)} 有效, 平均分: {avg:.1f}/40")

    print("[Judge] 卸载评估模型 ...")
    manager.unload("judge")

    return results
