#!/usr/bin/env python3
"""
battle_patched.py — SSPilot Battle Engine with Diagnostic Staging
Drop-in replacement for battle.py, adds:
  - Per-stage memory snapshots (GPU + system RAM for UVM)
  - Per-stage timing with StageTimer context manager
  - Inference timeout protection (signal.alarm)
  - Force cleanup between stages
  - Detailed diagnostic report

Usage (same CLI as battle.py):
  python battle_patched.py battle --round 3 --skip-train
  python battle_patched.py evolve --rounds 5
  python battle_patched.py generate --batch-size 20
  python battle_patched.py diag              # NEW: diagnostic-only mode
"""
from __future__ import annotations

import os
import sys
import gc
import json
import time
import signal
import argparse
import logging
log = logging.getLogger("sspilot")
from trace import TraceLogger
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from typing import Optional

# ── SSPilot existing imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_manager import manager
from vulngen import generate_batch
from audit_agent import audit_batch
from distiller import extract_training_data
from judge import (
    judge_batch,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    _parse_judge_response,
    _compute_grade,
)
import requests as _requests
import os as _os
# === vLLM Auto Judge Helper ===
import subprocess as _sp
import time as _time

def _ensure_vllm_judge(timeout=600):
    """Auto-start Nemotron vLLM Judge if not running."""
    import requests as _req
    url = "http://localhost:8001/v1/models"
    try:
        r = _req.get(url, timeout=5)
        if r.status_code == 200:
            log.info("vLLM Judge already online")
            return True
    except Exception:
        pass
    log.info("Starting vLLM Nemotron Judge container...")
    _sp.run(["docker", "rm", "-f", "vllm-nemotron"], capture_output=True)
    _time.sleep(5)
    cmd = [
        "docker", "run", "-d", "--gpus", "all", "--ipc=host",
        "--name", "vllm-nemotron",
        "-v", "/home/xsuper/models:/models",
        "-p", "8001:8001",
        "nvcr.io/nvidia/vllm:26.02-py3",
        "vllm", "serve", "/models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        "--host", "0.0.0.0", "--port", "8001",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.40",
        "--max-num-seqs", "16",
        "--trust-remote-code",
    ]
    result = _sp.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Docker run failed: {result.stderr}")
        return False
    start = _time.time()
    while _time.time() - start < timeout:
        try:
            r = _req.get(url, timeout=5)
            if r.status_code == 200:
                log.info(f"vLLM Judge online in {_time.time()-start:.0f}s")
                return True
        except Exception:
            pass
        _time.sleep(15)
        elapsed = int(_time.time() - start)
        if elapsed % 60 < 16:
            log.info(f"  Waiting for vLLM Judge... {elapsed}s/{timeout}s")
    log.error("vLLM Judge startup timeout!")
    return False

def _stop_vllm_judge():
    """Stop vLLM Judge to free GPU memory."""
    log.info("Stopping vLLM Judge container...")
    _sp.run(["docker", "rm", "-f", "vllm-nemotron"], capture_output=True)
    _time.sleep(10)
    log.info("vLLM Judge stopped, GPU memory freed")
# === End vLLM Auto Judge Helper ===



# ─────────────────────────────────────────────
# Diagnostic Infrastructure
# ─────────────────────────────────────────────

@dataclass
class MemSnapshot:
    """Memory snapshot for DGX Spark UVM (unified CPU+GPU memory)."""
    timestamp: str
    stage: str
    gpu_used_gb: float = 0.0
    gpu_total_gb: float = 0.0
    gpu_free_gb: float = 0.0
    sys_used_gb: float = 0.0
    sys_total_gb: float = 0.0
    sys_avail_gb: float = 0.0

    @staticmethod
    def capture(stage: str) -> "MemSnapshot":
        snap = MemSnapshot(
            timestamp=datetime.now().isoformat(),
            stage=stage,
        )
        # GPU memory via nvidia-smi (more reliable than torch on UVM)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip()
            parts = [float(x) for x in out.split(",")]
            snap.gpu_used_gb = parts[0] / 1024
            snap.gpu_total_gb = parts[1] / 1024
            snap.gpu_free_gb = parts[2] / 1024
        except Exception:
            pass
        # System RAM (critical for UVM — GPU borrows from system)
        try:
            import psutil
            vm = psutil.virtual_memory()
            snap.sys_total_gb = vm.total / (1024**3)
            snap.sys_used_gb = vm.used / (1024**3)
            snap.sys_avail_gb = vm.available / (1024**3)
        except ImportError:
            pass
        return snap

    def report(self) -> str:
        return (
            f"[MEM {self.stage}] "
            f"GPU: {self.gpu_used_gb:.1f}/{self.gpu_total_gb:.1f}GB "
            f"(free {self.gpu_free_gb:.1f}GB) | "
            f"SYS: {self.sys_used_gb:.1f}/{self.sys_total_gb:.1f}GB "
            f"(avail {self.sys_avail_gb:.1f}GB)"
        )


class InferenceTimeout(Exception):
    """Raised when inference exceeds time budget."""
    pass


def _timeout_handler(signum, frame):
    raise InferenceTimeout("Inference timeout — possible KV cache fragmentation or UVM thrashing")


@contextmanager
def stage_timer(name: str, timeout_sec: int = 0, snapshots: list = None):
    """Context manager: timing + optional timeout + memory snapshots."""
    snap_before = MemSnapshot.capture(f"{name}:before")
    log.info(snap_before.report())
    if snapshots is not None:
        snapshots.append(snap_before)

    old_handler = None
    if timeout_sec > 0:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)

    t0 = time.perf_counter()
    error = None
    try:
        yield
    except InferenceTimeout as e:
        error = e
        log.error(f"⏱ TIMEOUT in {name} after {timeout_sec}s")
        raise
    except Exception as e:
        error = e
        raise
    finally:
        elapsed = time.perf_counter() - t0
        if timeout_sec > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

        snap_after = MemSnapshot.capture(f"{name}:after")
        if snapshots is not None:
            snapshots.append(snap_after)

        delta_gpu = snap_after.gpu_used_gb - snap_before.gpu_used_gb
        status = "✗ FAIL" if error else "✓ OK"
        log.info(
            f"[STAGE {name}] {status} | {elapsed:.1f}s | "
            f"GPU Δ{delta_gpu:+.1f}GB | {snap_after.report()}"
        )


def force_cleanup():
    """Aggressive memory cleanup for UVM environment."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


# ─────────────────────────────────────────────
# Patched Battle Stages
# ─────────────────────────────────────────────

# Default timeouts (seconds) — adjust based on DGX Spark observed timings
TIMEOUT_GENERATE = 600   # 10 min for VulnGen (Qwen 32B-Int4, ~16GB)
TIMEOUT_AUDIT    = 1200  # 20 min for AuditAgent (Qwen2.5-Coder-32B-Int4, ~16GB)
TIMEOUT_JUDGE    = 900   # 15 min for Judge (Nemotron-3-Nano-30B NVFP4, ~40GB via vLLM)
TIMEOUT_TRAIN    = 1800  # 30 min for SFT training
BATCH_SIZE       = 20

# Diagnostic accumulator
diag_snapshots: list[MemSnapshot] = []
diag_timings: dict[str, float] = {}


def stage1_generate(
    batch_size: int = BATCH_SIZE,
    output_dir: str = "datasets",
    round_id: int | str = 1,
) -> list[dict]:
    """Stage 1: VulnGen generates vulnerable code samples."""
    log.info(f"═══ STAGE 1: GENERATE (batch={batch_size}, round={round_id}) ═══")
    t0 = time.perf_counter()

    with stage_timer("1_generate", timeout_sec=TIMEOUT_GENERATE, snapshots=diag_snapshots):
        rid = int(str(round_id).replace("r", "")) if str(round_id).replace("r", "").isdigit() else 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"vulngen_round{rid:03d}_{ts}.jsonl"
        samples = generate_batch(batch_size, output_path)
        log.info(f"Generated {len(samples)} samples → {output_path}")

    diag_timings["stage1_generate"] = time.perf_counter() - t0
    return samples



def api_judge_batch(
    audited_samples: list[dict],
    api_url: str = "http://localhost:8001/v1/chat/completions",
    model_name: str = "/models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
) -> list[dict]:
    """Judge via vLLM OpenAI-compatible API with canonical judge schema."""
    results = []
    for sample in audited_samples:
        audit_report = sample.get("audit_report", {})
        if audit_report.get("error"):
            results.append({**sample, "judge_result": {"skipped": "audit_failed"}})
            continue

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
        try:
            resp = _requests.post(api_url, json={
                "model": model_name,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.1
            }, timeout=120)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            judge_result = _parse_judge_response(raw)
            judge_result["grade"] = _compute_grade(judge_result["total_score"])
            judge_result["judge_timestamp"] = datetime.now().isoformat()
            judge_result["model"] = "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4(vLLM)"
            results.append({
                **sample,
                "judge_result": judge_result,
            })
        except Exception as e:
            log.warning(f"API judge failed for sample: {e}")
            results.append({
                **sample,
                "judge_result": {"error": str(e)},
            })
    return results

def stage2_battle(
    samples: list[dict],
    round_id: str = "r0",
    skip_judge: bool = False,
) -> list[dict]:
    """Stage 2: Audit + Judge battle with memory management."""
    log.info(f"═══ STAGE 2: BATTLE (round={round_id}, samples={len(samples)}) ═══")
    t0 = time.perf_counter()
    trace = TraceLogger(round_id=int(round_id.replace("r", "")))

    # ── 2a: Audit ──
    with stage_timer("2a_audit", timeout_sec=TIMEOUT_AUDIT, snapshots=diag_snapshots):
        audited = audit_batch(samples)
        log.info(f"Audited {len(audited)} samples")

    # ── Memory transition: unload agent, prepare for judge ──
    log.info("── Memory transition: agent → judge ──")
    snap_pre_unload = MemSnapshot.capture("pre_unload_agent")
    log.info(snap_pre_unload.report())

    manager.safe_unload("agent", required_free_gb=45.0)
    force_cleanup()

    snap_post_unload = MemSnapshot.capture("post_unload_agent")
    freed = snap_pre_unload.gpu_used_gb - snap_post_unload.gpu_used_gb
    log.info(snap_post_unload.report())
    log.info(f"🧹 Freed {freed:.1f}GB GPU memory after agent unload")

    if freed < 30.0:
        log.warning(
            f"⚠️  Only freed {freed:.1f}GB (expected ~60GB). "
            "Possible reference leak or UVM not releasing. "
            "Check model_manager.safe_unload() and torch tensor refs."
        )

    diag_snapshots.extend([snap_pre_unload, snap_post_unload])

    if skip_judge:
        log.info("Skipping judge (--skip-judge)")
        diag_timings["stage2_battle"] = time.perf_counter() - t0
        return audited

    # ── 2b: Judge ──
    with stage_timer("2b_judge", timeout_sec=TIMEOUT_JUDGE, snapshots=diag_snapshots):
        if os.environ.get("VLLM_JUDGE_API"):
            if not _ensure_vllm_judge():
                log.error("vLLM Judge failed to start, falling back to local judge")
                results = judge_batch(audited)
            else:
                results = api_judge_batch(audited)
            if os.environ.get("VLLM_JUDGE_AUTOSTOP"):
                _stop_vllm_judge()
        else:
            results = judge_batch(audited)
        trace.log_batch(results)
        log.info(f"Judged {len(results)} samples")

    manager.unload("judge")
    force_cleanup()

    summary = trace.summary()
    log.info(f"Round {round_id} summary: {json.dumps(summary, indent=2)}")

    diag_timings["stage2_battle"] = time.perf_counter() - t0
    return results


def stage3_train(round_ids: list[str], skip: bool = False) -> Optional[str]:
    """Stage 3: Distill traces to SFT/DPO files. Run `python scripts/sft_v5.py` to train LoRA."""
    if skip:
        log.info("═══ STAGE 3: TRAIN (skipped) ═══")
        return None

    log.info(f"═══ STAGE 3: TRAIN (rounds={round_ids}) ═══")
    t0 = time.perf_counter()

    with stage_timer("3_train", timeout_sec=TIMEOUT_TRAIN, snapshots=diag_snapshots):
        manager.unload_all()
        force_cleanup()

        rid_ints: list[int] = []
        for r in round_ids:
            s = str(r).replace("r", "").strip()
            if s.isdigit():
                rid_ints.append(int(s))
        training_data = extract_training_data(round_ids=rid_ints or None)
        log.info(
            "Distiller: sft_count=%s dpo_count=%s sft_path=%s",
            training_data.get("sft_count"),
            training_data.get("dpo_count"),
            training_data.get("sft_path"),
        )
        log.info(
            "LoRA SFT: run `cd scripts && python sft_v5.py` (or `training/train_audit_lora.py`) after updating data paths."
        )
        ckpt_hint = training_data.get("sft_path") or None

    diag_timings["stage3_train"] = time.perf_counter() - t0
    return ckpt_hint


# ─────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────

def run_battle(
    round_id: str = "r1",
    batch_size: int = BATCH_SIZE,
    skip_train: bool = False,
    skip_judge: bool = False,
    samples_path: str | None = None,
) -> dict:
    """Run a single battle round with full diagnostics."""
    log.info(f"{'='*60}")
    log.info(f"BATTLE ROUND {round_id} — {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"{'='*60}")

    diag_snapshots.clear()
    diag_timings.clear()
    t_total = time.perf_counter()

    result = {"round_id": round_id, "status": "unknown"}

    try:
        if samples_path and Path(samples_path).exists():
            log.info(f"Loading samples from {samples_path}")
            samples = []
            with open(samples_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line.strip()))
            log.info(f"Loaded {len(samples)} samples from file")
        else:
            rid_num = int(str(round_id).replace("r", "")) if str(round_id).replace("r", "").isdigit() else 1
            samples = stage1_generate(batch_size, round_id=rid_num)
        results = stage2_battle(samples, round_id, skip_judge)
        checkpoint = stage3_train([round_id], skip=skip_train)

        result.update({
            "status": "success",
            "samples_generated": len(samples),
            "samples_judged": len(results),
            "checkpoint": checkpoint,
        })

    except InferenceTimeout as e:
        result.update({"status": "timeout", "error": str(e)})
        log.error(f"🔥 Battle round {round_id} TIMEOUT: {e}")
        # Emergency cleanup
        try:
            manager.unload_all()
        except Exception:
            pass
        force_cleanup()

    except Exception as e:
        result.update({"status": "error", "error": str(e), "traceback": traceback.format_exc()})
        log.error(f"🔥 Battle round {round_id} ERROR: {e}")
        try:
            manager.unload_all()
        except Exception:
            pass
        force_cleanup()

    finally:
        total_time = time.perf_counter() - t_total
        result["total_time_sec"] = round(total_time, 1)
        result["timings"] = {k: round(v, 1) for k, v in diag_timings.items()}
        result["memory_snapshots"] = [asdict(s) for s in diag_snapshots]

        # Save diagnostic report
        diag_path = f"diag_round_{round_id}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(diag_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"📊 Diagnostic report → {diag_path}")
        log.info(f"Total time: {total_time:.1f}s")

    return result


def run_evolution(num_rounds: int = 5, batch_size: int = BATCH_SIZE):
    """Run multiple battle rounds (evolution loop)."""
    log.info(f"Starting evolution: {num_rounds} rounds")
    all_results = []

    for i in range(1, num_rounds + 1):
        round_id = f"r{i}"
        log.info(f"\n{'#'*60}")
        log.info(f"# EVOLUTION ROUND {i}/{num_rounds}")
        log.info(f"{'#'*60}")

        result = run_battle(
            round_id=round_id,
            batch_size=batch_size,
            skip_train=(i == num_rounds),  # skip train on last round
        )
        all_results.append(result)

        if result["status"] == "timeout":
            log.error(f"Evolution stopped at round {i} due to timeout")
            break

        # Cool-down between rounds
        if i < num_rounds:
            log.info("Cooling down 30s between rounds...")
            force_cleanup()
            time.sleep(30)

    # Summary
    log.info(f"\n{'='*60}")
    log.info("EVOLUTION SUMMARY")
    for r in all_results:
        log.info(f"  {r['round_id']}: {r['status']} ({r.get('total_time_sec', '?')}s)")
    log.info(f"{'='*60}")

    return all_results


def run_diagnostic():
    """
    Diagnostic-only mode: test each stage individually with detailed reporting.
    Helps identify which stage causes timeouts.
    """
    log.info("=" * 60)
    log.info("SSPilot DIAGNOSTIC MODE")
    log.info("=" * 60)

    diag_snapshots.clear()
    diag_timings.clear()
    report = {"mode": "diagnostic", "stages": {}}

    # ── Baseline ──
    snap = MemSnapshot.capture("baseline")
    log.info(snap.report())
    report["baseline"] = asdict(snap)

    # ── Stage 1: Generate (small batch) ──
    log.info("\n--- DIAG: Stage 1 Generate (batch=5) ---")
    try:
        with stage_timer("diag_generate", timeout_sec=300, snapshots=diag_snapshots):
            samples = generate_batch(5, "/tmp/diag_vulns.jsonl")
        report["stages"]["generate"] = {
            "status": "ok", "count": len(samples),
            "time": diag_timings.get("diag_generate", 0),
        }
    except Exception as e:
        report["stages"]["generate"] = {"status": "fail", "error": str(e)}
        log.error(f"Stage 1 diag failed: {e}")
        samples = []

    # ── Memory after generate ──
    snap = MemSnapshot.capture("after_generate")
    log.info(snap.report())

    if not samples:
        log.warning("No samples generated, creating dummy for audit test")
        samples = [{"code": "import os\nos.system(user_input)", "category": "injection", "id": "diag_0"}]

    # ── Stage 2a: Audit (small batch) ──
    log.info("\n--- DIAG: Stage 2a Audit (batch=5) ---")
    try:
        with stage_timer("diag_audit", timeout_sec=600, snapshots=diag_snapshots):
            audited = audit_batch(samples[:5])
        report["stages"]["audit"] = {"status": "ok", "count": len(audited)}
    except Exception as e:
        report["stages"]["audit"] = {"status": "fail", "error": str(e)}
        log.error(f"Stage 2a diag failed: {e}")
        audited = samples

    # ── Memory transition test ──
    log.info("\n--- DIAG: Memory Transition (agent → judge) ---")
    snap_pre = MemSnapshot.capture("pre_unload")
    log.info(snap_pre.report())

    try:
        manager.safe_unload("agent", required_free_gb=45.0)
        force_cleanup()
        time.sleep(5)  # Let UVM settle
        force_cleanup()  # Second pass
    except Exception as e:
        log.error(f"Unload failed: {e}")

    snap_post = MemSnapshot.capture("post_unload")
    freed = snap_pre.gpu_used_gb - snap_post.gpu_used_gb
    log.info(snap_post.report())
    log.info(f"Memory freed: {freed:.1f}GB")

    report["stages"]["memory_transition"] = {
        "gpu_before": snap_pre.gpu_used_gb,
        "gpu_after": snap_post.gpu_used_gb,
        "freed_gb": round(freed, 1),
        "expected_gb": 60.0,
        "healthy": freed >= 45.0,
    }

    if freed < 45.0:
        log.warning(f"⚠️  INSUFFICIENT MEMORY RELEASE: {freed:.1f}GB < 45GB expected")
        log.warning("Likely root cause of battle timeouts!")
        log.warning("Checking for leaked references...")

        # Reference leak check
        try:
            import torch
            for obj in gc.get_objects():
                if isinstance(obj, torch.nn.Module):
                    rc = sys.getrefcount(obj)
                    if rc > 3:  # self + gc list + getrefcount arg
                        log.warning(f"  Leaked module: {type(obj).__name__} refcount={rc}")
        except Exception:
            pass

    # ── Stage 2b: Judge ──
    log.info("\n--- DIAG: Stage 2b Judge (batch=5) ---")
    try:
        with stage_timer("diag_judge", timeout_sec=600, snapshots=diag_snapshots):
            results = vllm_judge_batch(audited[:5]) if _os.environ.get("USE_VLLM_JUDGE") else judge_batch(audited[:5])
        report["stages"]["judge"] = {"status": "ok", "count": len(results)}
    except Exception as e:
        report["stages"]["judge"] = {"status": "fail", "error": str(e)}
        log.error(f"Stage 2b diag failed: {e}")

    # ── Final cleanup ──
    try:
        manager.unload_all()
    except Exception:
        pass
    force_cleanup()

    snap_final = MemSnapshot.capture("final")
    log.info(snap_final.report())
    report["final_memory"] = asdict(snap_final)

    # ── Save report ──
    diag_path = f"diag_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(diag_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'='*60}")
    log.info("DIAGNOSTIC REPORT SUMMARY")
    log.info(f"{'='*60}")
    for stage, info in report["stages"].items():
        status_icon = "✅" if info.get("status") == "ok" or info.get("healthy", True) else "❌"
        log.info(f"  {status_icon} {stage}: {info}")
    log.info(f"Report saved → {diag_path}")
    log.info(f"{'='*60}")

    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SSPilot Battle Engine (Patched)")
    sub = parser.add_subparsers(dest="command")

    # battle
    p_battle = sub.add_parser("battle", help="Run single battle round")
    p_battle.add_argument("--round", default="r1", help="Round id, e.g. 1 or r1")
    p_battle.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p_battle.add_argument("--skip-train", action="store_true")
    p_battle.add_argument("--skip-judge", action="store_true")
    p_battle.add_argument("--use-vllm-judge", action="store_true", help="Use vLLM Judge")
    p_battle.add_argument("--samples", type=str, default=None, help="Load JSONL samples, skip Stage 1")

    # evolve
    p_evolve = sub.add_parser("evolve", help="Run evolution loop")
    p_evolve.add_argument("--rounds", type=int, default=5)
    p_evolve.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    # generate
    p_gen = sub.add_parser("generate", help="Generate only")
    p_gen.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p_gen.add_argument("--round", type=int, default=1, help="Round number for output filename")

    # diag (NEW)
    p_diag = sub.add_parser("diag", help="Run diagnostic (small batch, detailed report)")

    args = parser.parse_args()

    if args.command == "battle":
        rid = args.round
        if isinstance(rid, int) or (isinstance(rid, str) and rid.isdigit()):
            rid = f"r{int(rid)}"
        run_battle(
            round_id=str(rid),
            batch_size=args.batch_size,
            skip_train=args.skip_train,
            skip_judge=args.skip_judge,
            samples_path=args.samples,
        )
    elif args.command == "evolve":
        run_evolution(num_rounds=args.rounds, batch_size=args.batch_size)
    elif args.command == "generate":
        stage1_generate(args.batch_size, round_id=args.round)
    elif args.command == "diag":
        run_diagnostic()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
