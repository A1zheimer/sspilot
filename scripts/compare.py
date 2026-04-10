"""
SSPilot — Compare 多轮对比分析
生成进化趋势报告，对比训练前后的 Agent 表现
"""
import json
from pathlib import Path
from collections import defaultdict
from config import REPORT_DIR, TRACE_DIR
from trace import load_all_traces, compare_rounds


def generate_evolution_report(max_round: int | None = None) -> dict:
    """
    生成完整的进化趋势报告

    Returns:
        dict: 包含每轮分数、各维度分析、各漏洞类型分析
    """
    # 发现所有轮次
    trace_files = sorted(TRACE_DIR.glob("battle_round*.jsonl"))
    if not trace_files:
        return {"error": "no_traces_found"}

    all_traces = load_all_traces()
    if max_round:
        all_traces = [t for t in all_traces if t.get("round_id", 0) <= max_round]

    # 按轮次分组
    by_round: dict[int, list] = defaultdict(list)
    for t in all_traces:
        rid = t.get("round_id", 0)
        by_round[rid].append(t)

    report = {
        "total_rounds": len(by_round),
        "total_traces": len(all_traces),
        "rounds": [],
        "dimension_trends": {
            "detection": [],
            "precision": [],
            "depth": [],
            "remediation": [],
        },
        "vuln_type_trends": defaultdict(list),
    }

    for rid in sorted(by_round.keys()):
        traces = by_round[rid]
        scored = [
            t for t in traces
            if "scores" in t.get("judge_result", {})
        ]

        if not scored:
            continue

        scores = [t["judge_result"]["total_score"] for t in scored]
        avg = sum(scores) / len(scores)

        # 各维度平均
        dim_avgs = {}
        for dim in ["detection", "precision", "depth", "remediation"]:
            vals = [t["judge_result"]["scores"][dim] for t in scored]
            dim_avgs[dim] = round(sum(vals) / len(vals), 2)
            report["dimension_trends"][dim].append({
                "round": rid, "avg": dim_avgs[dim]
            })

        # 各漏洞类型平均
        by_vuln: dict[str, list] = defaultdict(list)
        for t in scored:
            vt = t.get("vuln_type", "unknown")
            by_vuln[vt].append(t["judge_result"]["total_score"])

        vuln_avgs = {}
        for vt, vscores in by_vuln.items():
            vuln_avgs[vt] = round(sum(vscores) / len(vscores), 2)
            report["vuln_type_trends"][vt].append({
                "round": rid, "avg": vuln_avgs[vt]
            })

        # 等级分布
        grade_dist = defaultdict(int)
        for t in scored:
            grade_dist[t["judge_result"].get("grade", "?")] += 1

        round_info = {
            "round_id": rid,
            "sample_count": len(traces),
            "scored_count": len(scored),
            "avg_score": round(avg, 2),
            "max_score": max(scores),
            "min_score": min(scores),
            "dimension_avgs": dim_avgs,
            "vuln_type_avgs": vuln_avgs,
            "grade_distribution": dict(grade_dist),
        }
        report["rounds"].append(round_info)

    # 计算整体进化量
    if len(report["rounds"]) >= 2:
        first = report["rounds"][0]["avg_score"]
        last = report["rounds"][-1]["avg_score"]
        report["total_improvement"] = round(last - first, 2)
        report["improvement_pct"] = round((last - first) / max(first, 1) * 100, 1)

    return report


def print_evolution_report(report: dict):
    """以格式化文本打印进化报告"""
    if "error" in report:
        print(f"错误: {report['error']}")
        return

    print("\n" + "=" * 70)
    print("  SSPilot 进化趋势报告")
    print("=" * 70)

    print(f"\n总轮次: {report['total_rounds']}, 总样本: {report['total_traces']}")

    if "total_improvement" in report:
        imp = report["total_improvement"]
        pct = report["improvement_pct"]
        arrow = "↑" if imp > 0 else "↓"
        print(f"总进化: {arrow} {abs(imp):.1f} 分 ({pct:+.1f}%)")

    # 每轮分数
    print(f"\n{'轮次':>6} {'样本':>6} {'平均分':>8} {'最高':>6} {'最低':>6} {'等级分布'}")
    print("-" * 70)
    for r in report["rounds"]:
        grades = " ".join(f"{g}:{c}" for g, c in sorted(r["grade_distribution"].items()))
        print(f"  R{r['round_id']:<4} {r['scored_count']:>5} "
              f"{r['avg_score']:>7.1f} {r['max_score']:>5} {r['min_score']:>5}  {grades}")

    # 维度趋势
    print(f"\n{'维度':<12}", end="")
    for r in report["rounds"]:
        print(f"  R{r['round_id']:<5}", end="")
    print()
    print("-" * (12 + 7 * len(report["rounds"])))
    for dim in ["detection", "precision", "depth", "remediation"]:
        print(f"  {dim:<10}", end="")
        for r in report["rounds"]:
            val = r["dimension_avgs"].get(dim, 0)
            print(f"  {val:<5.1f}", end="")
        print()

    # 各漏洞类型趋势
    all_vtypes = set()
    for r in report["rounds"]:
        all_vtypes.update(r["vuln_type_avgs"].keys())

    if all_vtypes:
        print(f"\n{'漏洞类型':<18}", end="")
        for r in report["rounds"]:
            print(f"  R{r['round_id']:<5}", end="")
        print()
        print("-" * (18 + 7 * len(report["rounds"])))
        for vt in sorted(all_vtypes):
            print(f"  {vt:<16}", end="")
            for r in report["rounds"]:
                val = r["vuln_type_avgs"].get(vt, 0)
                print(f"  {val:<5.1f}", end="")
            print()

    print()


def save_report(report: dict, filename: str = "evolution_report.json"):
    """保存报告到文件"""
    path = REPORT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {path}")
    return path


if __name__ == "__main__":
    report = generate_evolution_report()
    print_evolution_report(report)
    save_report(report)
