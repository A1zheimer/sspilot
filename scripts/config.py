"""
SSPilot 全局配置
双模型流水线：Qwen2.5-Coder (VulnGen + AuditAgent) → Nemotron-3-Nano-30B (Judge via vLLM)
"""
from pathlib import Path

# ===== 项目路径 =====
PROJECT_ROOT = Path("/home/xsuper/sspilot")
TRACE_DIR = PROJECT_ROOT / "traces"
DATASET_DIR = PROJECT_ROOT / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"

for d in [TRACE_DIR, DATASET_DIR, CHECKPOINT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== 模型路径 =====
MODELS = {
    "vulngen": {
        "path": "/home/xsuper/models/Qwen2.5-Coder-32B-Int4",
        "name": "Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
        "role": "VulnGen 漏洞代码生成器",
        "quantization": "gptq",
        "trust_remote_code": False,
    },
    "agent": {
        "path": "/home/xsuper/models/Qwen2.5-Coder-32B-Int4",
        "name": "Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
        "role": "AuditAgent 安全审计选手",
        "quantization": "gptq",
        
        "trust_remote_code": True,
        "lora_path": "/home/xsuper/sspilot/checkpoints/audit_sft_v5/final",
    },
    "judge": {
        "path": "/home/xsuper/models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        "name": "nvidia/Nemotron-3-Nano-30B-A3B-NVFP4",
        "role": "Judge 漏洞裁判",
        "quantization": "nvfp4",  # 需要 INT4 量化加载
        "trust_remote_code": True,
    },
}

# ===== 备用模型 =====
FALLBACK_JUDGE = "/home/xsuper/models/Llama3.3-70b-insturct"

# ===== 漏洞类型 =====
VULN_TYPES = [
    "sqli", "xss", "path_traversal", "hardcoded_secret",
    "ssrf", "unsafe_deser", "logic", "info_leak",
]

# ===== 对战参数 =====
BATTLE_CONFIG = {
    "max_retries": 1,        # Agent 每题最多重试次数
    "batch_size": 40,        # 每轮对战题目数
    "max_new_tokens": 2048,  # 生成最大 token 数
    "temperature_vulngen": 0.7,
    "temperature_agent": 0.3,
    "temperature_judge": 0.1,
    "use_tools": False,           # 启用工具增强审计模式 (Tool-Augmented Audit)
    "max_tool_rounds": 5,         # 工具调用最大轮次
}

# ===== 显存安全参数 =====
MEMORY_SAFETY = {
    "min_free_before_load": 5.0,   # 加载前最低空闲 GB
    "unload_wait_max": 30,           # 卸载后最大等待秒数
    "unload_wait_interval": 2,       # 检查间隔
}
