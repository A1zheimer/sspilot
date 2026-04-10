"""
SSPilot — Trace Logger
记录每轮对战的完整 trace，用于后续 SFT/DPO 数据提取
"""
import json
import time
from pathlib import Path
from datetime import datetime
from config import TRACE_DIR


class TraceLogger:
    """
    JSONL 格式的 trace 记录器

    每条 trace 记录一次完整的 VulnGen → AuditAgent → Judge 流程：
    {
        "trace_id": "round_001_sqli_000_20260330_120000",
        "round_id": 1,
        "timestamp": "2026-03-30T12:00:00",
        "sample": { ... },           # VulnGen 输出
        "audit_report": { ... },      # AuditAgent 输出
        "judge_result": { ... },      # Judge 评分
        "metadata": {
            "stage": "battle",
            "duration_sec": 45.2,
            "models": { "vulngen": "...", "agent": "...", "judge": "..." }
        }
    }
    """

    def __init__(self, round_id: int = 1, prefix: str = "battle"):
        self.round_id = round_id
        self.prefix = prefix
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_path = TRACE_DIR / f"{prefix}_round{round_id:03d}_{ts}.jsonl"
        self.trace_count = 0
        self._start_time = time.time()
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Trace] 日志文件: {self.trace_path}")

    def log(self, item: dict, duration_sec: float = 0.0):
        """记录一条 trace"""
        trace = {
            "trace_id": f"round_{self.round_id:03d}_{item.get('sample_id', self.trace_count)}",
            "round_id": self.round_id,
            "timestamp": datetime.now().isoformat(),
            "vuln_type": item.get("vuln_type", "unknown"),
            "difficulty": item.get("difficulty", "medium"),
            "sample": {
                "sample_id": item.get("sample_id", ""),
                "code": item.get("code", ""),
                "ground_truth": item.get("ground_truth", {}),
            },
            "audit_report": item.get("audit_report", {}),
            "judge_result": item.get("judge_result", {}),
            "metadata": {
                "stage": self.prefix,
                "round_id": self.round_id,
                "duration_sec": duration_sec,
            },
        }

        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        self.trace_count += 1

    def log_batch(self, items: list[dict]):
        """批量记录"""
        for item in items:
            self.log(item)

    def summary(self) -> dict:
        """生成本轮 trace 摘要"""
        traces = self.load_traces()
        if not traces:
            return {"total": 0, "round_id": self.round_id}

        scores = []
        grade_dist = {}
        vuln_dist = {}

        for t in traces:
            jr = t.get("judge_result", {})
            if "scores" in jr:
                total = jr["total_score"]
                scores.append(total)
                grade = jr.get("grade", "?")
                grade_dist[grade] = grade_dist.get(grade, 0) + 1

            vt = t.get("vuln_type", "?")
            vuln_dist[vt] = vuln_dist.get(vt, 0) + 1

        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "round_id": self.round_id,
            "total_traces": len(traces),
            "scored_traces": len(scores),
            "avg_score": round(avg_score, 2),
            "grade_distribution": grade_dist,
            "vuln_distribution": vuln_dist,
            "trace_path": str(self.trace_path),
            "total_duration_sec": round(time.time() - self._start_time, 1),
        }

    def load_traces(self) -> list[dict]:
        """加载本轮所有 trace"""
        if not self.trace_path.exists():
            return []
        traces = []
        with open(self.trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
        return traces


def load_all_traces(round_ids: list[int] | None = None) -> list[dict]:
    """
    加载多轮 trace 数据

    Args:
        round_ids: 指定轮次，None 表示全部

    Returns:
        list[dict]: 所有 trace 记录
    """
    all_traces = []
    for trace_file in sorted(TRACE_DIR.glob("battle_round*.jsonl")):
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trace = json.loads(line)
                if round_ids is None or trace.get("round_id") in round_ids:
                    all_traces.append(trace)
    return all_traces


def compare_rounds(round_a: int, round_b: int) -> dict:
    """
    比较两轮对战的表现差异

    Returns:
        dict: 包含分数变化、等级变化等
    """
    traces_a = load_all_traces([round_a])
    traces_b = load_all_traces([round_b])

    def _avg(traces):
        scores = [t["judge_result"]["total_score"]
                  for t in traces if "scores" in t.get("judge_result", {})]
        return sum(scores) / len(scores) if scores else 0

    avg_a = _avg(traces_a)
    avg_b = _avg(traces_b)

    return {
        "round_a": round_a,
        "round_b": round_b,
        "avg_score_a": round(avg_a, 2),
        "avg_score_b": round(avg_b, 2),
        "improvement": round(avg_b - avg_a, 2),
        "improvement_pct": round((avg_b - avg_a) / max(avg_a, 1) * 100, 1),
        "count_a": len(traces_a),
        "count_b": len(traces_b),
    }
