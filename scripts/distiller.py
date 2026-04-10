"""
SSPilot — Distiller 训练数据提取器
从 trace 日志中提取 SFT 和 DPO 训练数据

SFT 数据：高分审计样本 → 直接作为 positive example
DPO 数据：同类型不同分数的审计对 → chosen/rejected pair
"""
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from config import TRACE_DIR, DATASET_DIR
from trace import load_all_traces


# ── SFT 数据提取 ─────────────────────────────────────────────────

def extract_sft_data(
    traces: list[dict],
    min_score: int = 28,  # Grade A 以上
    output_path: Path | None = None,
) -> list[dict]:
    """
    提取 SFT 训练数据

    选择标准：Judge 总分 >= min_score (默认 28，即 A 级以上)
    格式：标准 chat messages (system + user + assistant)

    Args:
        traces: trace 记录列表
        min_score: 最低总分阈值
        output_path: 输出路径

    Returns:
        list[dict]: SFT 训练样本
    """
    sft_samples = []

    for t in traces:
        jr = t.get("judge_result", {})
        if "scores" not in jr:
            continue
        if jr["total_score"] < min_score:
            continue

        sample = t.get("sample", {})
        audit = t.get("audit_report", {})

        if not sample.get("code") or audit.get("error") or audit.get("parse_error"):
            continue

        # 构建 SFT 样本：system prompt + 用户代码 → 高质量审计报告
        sft_item = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一位顶级代码安全审计专家。请对给定代码进行全面安全审计，"
                        "找出所有潜在漏洞，并给出详细的分析和修复建议。"
                        "以 JSON 格式输出审计报告。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"请对以下代码进行安全审计：\n\n```python\n{sample['code']}\n```",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(audit, ensure_ascii=False, indent=2),
                },
            ],
            "metadata": {
                "vuln_type": t.get("vuln_type", "unknown"),
                "difficulty": t.get("difficulty", "medium"),
                "judge_score": jr["total_score"],
                "judge_grade": jr.get("grade", "?"),
                "round_id": t.get("round_id", 0),
                "trace_id": t.get("trace_id", ""),
            },
        }
        sft_samples.append(sft_item)

    # 保存
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DATASET_DIR / f"sft_data_{ts}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in sft_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Distiller] SFT 数据: {len(sft_samples)} 样本 → {output_path}")
    return sft_samples


# ── DPO 数据提取 ─────────────────────────────────────────────────

def extract_dpo_data(
    traces: list[dict],
    score_gap: int = 8,  # chosen/rejected 最小分差
    output_path: Path | None = None,
) -> list[dict]:
    """
    提取 DPO 训练数据

    策略：对同一 vuln_type 的审计，配对高分(chosen)和低分(rejected)
    要求：分差 >= score_gap

    DPO 格式：
    {
        "prompt": "system + user messages",
        "chosen": "高分审计报告",
        "rejected": "低分审计报告"
    }
    """
    # 按 vuln_type 分组
    by_type: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        jr = t.get("judge_result", {})
        if "scores" not in jr:
            continue
        audit = t.get("audit_report", {})
        if audit.get("error") or audit.get("parse_error"):
            continue
        vt = t.get("vuln_type", "unknown")
        by_type[vt].append(t)

    dpo_pairs = []

    for vt, items in by_type.items():
        # 按分数排序
        scored = sorted(
            items,
            key=lambda x: x["judge_result"]["total_score"],
            reverse=True,
        )

        # 配对：高分 vs 低分
        i, j = 0, len(scored) - 1
        while i < j:
            high = scored[i]
            low = scored[j]
            high_score = high["judge_result"]["total_score"]
            low_score = low["judge_result"]["total_score"]

            if high_score - low_score >= score_gap:
                # 构建 prompt（相同的 system + user）
                prompt_messages = [
                    {
                        "role": "system",
                        "content": (
                            "你是一位顶级代码安全审计专家。请对给定代码进行全面安全审计，"
                            "找出所有潜在漏洞，并给出详细的分析和修复建议。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"请对以下代码进行安全审计：\n\n```python\n{high['sample']['code']}\n```",
                    },
                ]

                dpo_pair = {
                    "prompt": prompt_messages,
                    "chosen": json.dumps(high["audit_report"], ensure_ascii=False),
                    "rejected": json.dumps(low["audit_report"], ensure_ascii=False),
                    "metadata": {
                        "vuln_type": vt,
                        "chosen_score": high_score,
                        "rejected_score": low_score,
                        "score_gap": high_score - low_score,
                        "chosen_grade": high["judge_result"].get("grade", "?"),
                        "rejected_grade": low["judge_result"].get("grade", "?"),
                    },
                }
                dpo_pairs.append(dpo_pair)
                i += 1
                j -= 1
            else:
                j -= 1

    # 保存
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DATASET_DIR / f"dpo_data_{ts}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[Distiller] DPO 数据: {len(dpo_pairs)} 对 → {output_path}")
    return dpo_pairs


# ── 统一提取入口 ─────────────────────────────────────────────────

def extract_training_data(
    round_ids: list[int] | None = None,
    sft_min_score: int = 28,
    dpo_score_gap: int = 8,
) -> dict:
    """
    从 trace 中提取所有训练数据

    Returns:
        dict: {
            "sft_path": str,
            "dpo_path": str,
            "sft_count": int,
            "dpo_count": int,
            "stats": { ... }
        }
    """
    traces = load_all_traces(round_ids)
    print(f"[Distiller] 加载了 {len(traces)} 条 trace 记录")

    if not traces:
        return {"sft_count": 0, "dpo_count": 0, "sft_path": "", "dpo_path": ""}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sft_path = DATASET_DIR / f"sft_data_{ts}.jsonl"
    dpo_path = DATASET_DIR / f"dpo_data_{ts}.jsonl"

    sft_samples = extract_sft_data(traces, min_score=sft_min_score, output_path=sft_path)
    dpo_pairs = extract_dpo_data(traces, score_gap=dpo_score_gap, output_path=dpo_path)

    # 统计
    all_scores = [
        t["judge_result"]["total_score"]
        for t in traces
        if "scores" in t.get("judge_result", {})
    ]
    stats = {
        "total_traces": len(traces),
        "total_scored": len(all_scores),
        "avg_score": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
        "max_score": max(all_scores) if all_scores else 0,
        "min_score": min(all_scores) if all_scores else 0,
    }

    return {
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "sft_count": len(sft_samples),
        "dpo_count": len(dpo_pairs),
        "stats": stats,
    }


# ── 数据质量分析 ─────────────────────────────────────────────────

def analyze_training_data(sft_path: str = "", dpo_path: str = "") -> dict:
    """分析训练数据的质量和分布"""
    analysis = {}

    if sft_path and Path(sft_path).exists():
        sft_data = []
        with open(sft_path) as f:
            for line in f:
                sft_data.append(json.loads(line.strip()))

        vuln_dist = defaultdict(int)
        diff_dist = defaultdict(int)
        score_dist = defaultdict(int)

        for item in sft_data:
            meta = item.get("metadata", {})
            vuln_dist[meta.get("vuln_type", "?")] += 1
            diff_dist[meta.get("difficulty", "?")] += 1
            grade = meta.get("judge_grade", "?")
            score_dist[grade] += 1

        analysis["sft"] = {
            "total": len(sft_data),
            "vuln_distribution": dict(vuln_dist),
            "difficulty_distribution": dict(diff_dist),
            "grade_distribution": dict(score_dist),
        }

    if dpo_path and Path(dpo_path).exists():
        dpo_data = []
        with open(dpo_path) as f:
            for line in f:
                dpo_data.append(json.loads(line.strip()))

        gaps = [p["metadata"]["score_gap"] for p in dpo_data]
        analysis["dpo"] = {
            "total": len(dpo_data),
            "avg_score_gap": round(sum(gaps) / len(gaps), 2) if gaps else 0,
            "max_gap": max(gaps) if gaps else 0,
        }

    return analysis


if __name__ == "__main__":
    result = extract_training_data()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["sft_path"] or result["dpo_path"]:
        analysis = analyze_training_data(result["sft_path"], result["dpo_path"])
        print("\n数据质量分析:")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
