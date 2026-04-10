#!/bin/bash
# ═══════════════════════════════════════════════════════════
# SSPilot — 自我进化的代码安全审计 Agent
# 主入口脚本
# 硬件: NVIDIA DGX Spark (GB10, 119GB)
# ═══════════════════════════════════════════════════════════

set -e

PROJECT_ROOT="/home/xsuper/sspilot"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
CONDA_ENV="sspilot"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║       SSPilot — 自我进化的代码安全审计 Agent           ║"
    echo "║       NVIDIA Hackathon 2026                              ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_env() {
    echo -e "${YELLOW}[检查环境]${NC}"
    
    # 检查 conda 环境
    if ! conda env list | grep -q "$CONDA_ENV"; then
        echo -e "${RED}✗ conda 环境 '$CONDA_ENV' 不存在${NC}"
        echo "  请运行: conda create -n $CONDA_ENV python=3.11 -y"
        exit 1
    fi
    
    # 检查 GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo -e "${GREEN}✓ GPU 可用 (${GPU_MEM}MB)${NC}"
    else
        echo -e "${RED}✗ nvidia-smi 不可用${NC}"
        exit 1
    fi
    
    # 检查模型
    for model_dir in "nemotron-3-nano-30b-a3b" "Qwen3-next-80b-a3b-thinking" "Qwen2.5-Coder-32B-Int4"; do
        if [ -d "/home/xsuper/models/$model_dir" ]; then
            echo -e "${GREEN}✓ 模型: $model_dir${NC}"
        else
            echo -e "${YELLOW}⚠ 模型缺失: $model_dir${NC}"
        fi
    done
    
    echo ""
}

# ── 命令 ──────────────────────────────────────────────────

cmd_generate() {
    echo -e "${BLUE}[Stage 1] VulnGen 生成漏洞代码${NC}"
    BATCH_SIZE=${1:-8}
    ROUND=${2:-1}
    cd "$SCRIPTS_DIR"
    python battle_patched.py generate --batch-size "$BATCH_SIZE" --round "$ROUND"
}

cmd_battle() {
    echo -e "${BLUE}[Battle] 运行单轮对战${NC}"
    ROUND=${1:-1}
    BATCH_SIZE=${2:-40}
    cd "$SCRIPTS_DIR"
    python battle_patched.py battle --round "$ROUND" --batch-size "$BATCH_SIZE" "$@"
}

cmd_evolve() {
    echo -e "${BLUE}[Evolve] 运行多轮进化${NC}"
    ROUNDS=${1:-5}
    BATCH_SIZE=${2:-40}
    cd "$SCRIPTS_DIR"
    python battle_patched.py evolve --rounds "$ROUNDS" --batch-size "$BATCH_SIZE"
}

cmd_compare() {
    echo -e "${BLUE}[Compare] 对比分析${NC}"
    cd "$SCRIPTS_DIR"
    python compare.py
}

cmd_distill() {
    echo -e "${BLUE}[Distill] 提取训练数据${NC}"
    cd "$SCRIPTS_DIR"
    python distiller.py
}

cmd_test() {
    echo -e "${BLUE}[Test] 快速测试流水线${NC}"
    cd "$SCRIPTS_DIR"
    
    # 检查是否已有 VulnGen 数据，有则跳过 Stage 1
    LATEST=$(ls -t "$PROJECT_ROOT/datasets/vulngen_round000"*.jsonl 2>/dev/null | head -1)
    
    if [ -n "$LATEST" ]; then
        echo -e "${GREEN}✓ 已有测试样本: $LATEST，跳过 Stage 1${NC}"
    else
        echo -e "${YELLOW}1/3 测试 VulnGen (每种类型 1 个, 共 8 个) ...${NC}"
        python battle_patched.py generate --batch-size 8 --round 0
        LATEST=$(ls -t "$PROJECT_ROOT/datasets/vulngen_round000"*.jsonl 2>/dev/null | head -1)
    fi
    
    echo -e "${YELLOW}2/3 测试 AuditAgent + Judge ...${NC}"
    if [ -n "$LATEST" ]; then
        python battle_patched.py battle --round 0 --skip-train --samples "$LATEST"
    else
        echo -e "${RED}未找到测试样本${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}3/3 测试数据提取 ...${NC}"
    python distiller.py
    
    echo -e "${GREEN}✓ 测试完成${NC}"
}

# ── 主入口 ────────────────────────────────────────────────

print_banner

case "${1:-help}" in
    generate)
        check_env
        cmd_generate "${@:2}"
        ;;
    battle)
        check_env
        cmd_battle "${@:2}"
        ;;
    evolve)
        check_env
        cmd_evolve "${@:2}"
        ;;
    compare)
        cmd_compare
        ;;
    distill)
        cmd_distill
        ;;
    test)
        check_env
        cmd_test
        ;;
    help|--help|-h)
        echo "用法: $0 <command> [options]"
        echo ""
        echo "命令:"
        echo "  generate [batch_size] [round]  生成漏洞代码样本"
        echo "  battle [round] [batch_size]    运行单轮对战"
        echo "  evolve [rounds] [batch_size]   运行多轮进化循环"
        echo "  compare                        对比分析所有轮次"
        echo "  distill                        提取 SFT/DPO 训练数据"
        echo "  test                           快速流水线测试"
        echo ""
        echo "示例:"
        echo "  $0 test                        快速测试 (8 个样本)"
        echo "  $0 battle 1 40                 第 1 轮对战, 40 个样本"
        echo "  $0 evolve 5 40                 5 轮进化, 每轮 40 个样本"
        echo "  $0 compare                     查看进化趋势"
        ;;
    *)
        echo -e "${RED}未知命令: $1${NC}"
        echo "运行 '$0 help' 查看帮助"
        exit 1
        ;;
esac
