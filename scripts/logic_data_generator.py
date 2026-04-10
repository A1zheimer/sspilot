#!/usr/bin/env python3
"""
from __future__ import annotations
SSPilot Logic 漏洞训练数据生成器 — 高多样性 SFT + DPO 数据

解决的问题:
  当前 logic 类 17.2/40，是 8 类漏洞中得分最低的。
  根因: v1~v4 训练数据重复度高（只改路径），导致过拟合于特定模式，
  TOCTOU/竞态条件完全检测不出。

策略:
  1. 覆盖 5 种 logic 子类: TOCTOU / 认证绕过 / 权限提升 / 业务逻辑 / 不安全默认
  2. 每种包含 code + ground_truth + 高质量审计推理链 (SFT 示范)
  3. 包含 negative examples (安全代码) 降低误报
  4. 可选: 生成 DPO 对 (chosen=好审计 vs rejected=差审计)

用法:
  python scripts/logic_data_generator.py --output datasets/sft_logic_v6.jsonl --format sft
  python scripts/logic_data_generator.py --output datasets/dpo_logic_v6.jsonl --format dpo
"""

import json
import argparse
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
#  Logic 漏洞代码模板库 — 5 大子类 × 多种变体
# ──────────────────────────────────────────────────────────────────────

LOGIC_SAMPLES = [

    # ══════════════════════════════════════════════════════════════════
    #  D1. TOCTOU / 竞态条件 (CWE-367)
    # ══════════════════════════════════════════════════════════════════

    {
        "id": "logic-toctou-001",
        "sub_type": "toctou",
        "code": '''import os
import hashlib

def safe_file_processor(filepath, user_id):
    """Process uploaded file with security checks"""
    # Security check: verify file exists and is not a symlink
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    
    if os.path.islink(filepath):
        return {"error": "Symlinks not allowed"}
    
    # Check file size limit (10MB)
    file_size = os.path.getsize(filepath)
    if file_size > 10 * 1024 * 1024:
        return {"error": "File too large"}
    
    # ← TOCTOU gap: file can be replaced between check and read
    
    # Process the file
    with open(filepath, 'rb') as f:
        content = f.read()
    
    checksum = hashlib.sha256(content).hexdigest()
    
    # Save processed result
    output_path = f"/app/processed/{user_id}/{checksum}.dat"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(content)
    
    return {"status": "ok", "checksum": checksum, "size": len(content)}
''',
        "ground_truth": {
            "vuln_type": "logic",
            "cwe_id": "CWE-367",
            "description": "TOCTOU race condition: os.path.exists/islink/getsize checks happen before open(). Between the security checks (lines 7-14) and the actual file read (line 18), an attacker can replace the file with a symlink to /etc/passwd or a different file, bypassing all checks.",
            "vulnerable_lines": [7, 10, 14, 18],
            "attack_vector": "1. Upload normal file → passes checks. 2. After islink() check but before open(), atomically replace file with symlink to /etc/shadow. 3. open() reads /etc/shadow, bypassing symlink protection."
        },
        "good_audit": '''## 分析过程

### 数据流分析
- source: filepath (用户可控的文件路径参数)
- transform: os.path.exists → os.path.islink → os.path.getsize (安全检查)
- sink: open(filepath, 'rb') (文件读取操作)

### 控制流分析
- 正常路径: 检查存在 → 检查非symlink → 检查大小 → 读取文件
- 所有检查逻辑本身是正确的

### 时序分析 ⚠️
- 关键发现: 检查操作(line 7-14)和使用操作(line 18)之间存在时间窗口
- 在多进程/多线程环境下，攻击者可以在 islink() 返回 False 后、open() 执行前，将文件替换为 symlink
- 这是经典的 TOCTOU (Time-of-Check to Time-of-Use) 竞态条件

## 发现

```json
{
  "overall_risk": "high",
  "findings": [
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-367",
      "severity": "high",
      "location": "safe_file_processor() 第 7-18 行",
      "description": "TOCTOU 竞态条件: 安全检查(exists/islink/getsize)和文件操作(open)之间存在时间窗口，攻击者可在此窗口内替换文件为 symlink",
      "attack_vector": "1) 上传正常文件通过检查 2) 在 islink() 和 open() 之间将文件原子替换为指向 /etc/shadow 的 symlink 3) open() 读取敏感文件内容",
      "remediation": "使用 os.open() 配合 O_NOFOLLOW 标志打开文件，或先 open() 后用 os.fstat() 在文件描述符上做检查: fd = os.open(filepath, os.O_RDONLY | os.O_NOFOLLOW); fstat = os.fstat(fd)"
    }
  ],
  "safe_patterns": ["hashlib.sha256 用于完整性校验是安全的", "os.makedirs 的 exist_ok=True 是安全实践"]
}
```''',
        "bad_audit": '''## 分析

代码对文件进行了存在性检查和大小限制，看起来安全。

## 发现

```json
{
  "overall_risk": "low",
  "findings": [],
  "safe_patterns": ["文件存在性检查", "symlink 检查", "文件大小限制"]
}
```

代码有完善的安全检查机制，未发现漏洞。'''
    },

    {
        "id": "logic-toctou-002",
        "sub_type": "toctou",
        "code": '''import threading
import time

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.transaction_log = []

    def get_balance(self):
        return self.balance

    def transfer(self, to_account, amount):
        """Transfer money to another account"""
        if amount <= 0:
            return {"error": "Invalid amount"}
        
        # Check sufficient balance
        current_balance = self.get_balance()
        if current_balance < amount:
            return {"error": "Insufficient funds"}
        
        # ← Race condition window: another thread can also pass the check
        
        # Perform transfer
        self.balance -= amount
        to_account.balance += amount
        
        # Log transaction
        self.transaction_log.append({
            "type": "transfer",
            "to": to_account.owner,
            "amount": amount,
            "remaining": self.balance
        })
        
        return {"status": "ok", "remaining": self.balance}

    def withdraw(self, amount):
        """Withdraw money"""
        if amount <= 0:
            return {"error": "Invalid amount"}
        
        if self.balance >= amount:
            time.sleep(0.01)  # Simulate processing delay
            self.balance -= amount
            return {"status": "ok", "remaining": self.balance}
        
        return {"error": "Insufficient funds"}
''',
        "ground_truth": {
            "vuln_type": "logic",
            "cwe_id": "CWE-367",
            "description": "Double-spend race condition: Both transfer() and withdraw() check balance then modify it without any locking mechanism. Two concurrent calls can both pass the balance check and double-deduct.",
            "vulnerable_lines": [20, 21, 26, 41, 42, 43],
            "attack_vector": "Send two simultaneous transfer requests of the full balance amount. Both threads read balance=1000, both pass the check, both deduct → balance goes to -1000."
        },
        "good_audit": '''## 分析过程

### 数据流分析
- source: amount 参数 (外部输入)
- transform: get_balance() 读取余额, 条件判断
- sink: self.balance -= amount (直接修改共享状态)

### 控制流分析
- transfer(): 检查余额(line 20-21) → 扣款(line 26) — 无原子性保证
- withdraw(): 检查余额(line 41) → sleep(line 42) → 扣款(line 43) — 故意的延迟使竞态更明显

### 时序分析 ⚠️
- BankAccount.balance 是共享可变状态，无任何同步原语(lock/mutex/semaphore)
- transfer() 的 check-then-act 模式是经典的 TOCTOU
- withdraw() 中的 time.sleep(0.01) 进一步放大了竞态窗口
- 两个并发 transfer 可以同时通过余额检查，导致双重扣款(余额变负)

## 发现

```json
{
  "overall_risk": "critical",
  "findings": [
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-367",
      "severity": "critical",
      "location": "transfer() 第 20-26 行, withdraw() 第 41-43 行",
      "description": "双重支付竞态条件: balance 读取和修改之间无原子性保证。并发请求可同时通过余额检查并扣款，导致余额为负。",
      "attack_vector": "并发发送两个 transfer(account_b, 1000) 请求，两个线程同时读到 balance=1000，同时通过检查，各扣 1000，最终 balance=-1000",
      "remediation": "使用 threading.Lock 保护 balance 操作: with self._lock: if self.balance >= amount: self.balance -= amount。或使用数据库事务 SELECT ... FOR UPDATE。"
    }
  ],
  "safe_patterns": ["amount <= 0 的输入验证是正确的"]
}
```''',
        "bad_audit": '''## 分析

检查了代码中的输入验证和业务逻辑。

## 发现

```json
{
  "overall_risk": "medium",
  "findings": [
    {
      "vuln_type": "info_leak",
      "severity": "low",
      "location": "transaction_log",
      "description": "交易日志可能泄露账户信息",
      "remediation": "对日志进行脱敏处理"
    }
  ]
}
```

代码有基本的输入验证（金额检查），整体安全性一般。'''
    },

    # ══════════════════════════════════════════════════════════════════
    #  D2. 认证绕过 (CWE-287)
    # ══════════════════════════════════════════════════════════════════

    {
        "id": "logic-authbypass-001",
        "sub_type": "auth_bypass",
        "code": '''import jwt
import os
from functools import wraps
from flask import request, jsonify, g

SECRET_KEY = os.environ.get("JWT_SECRET", "dev-secret-key-change-in-prod")
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"

def require_auth(f):
    """Authentication decorator for API endpoints"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Skip auth in debug mode for easier testing
        if DEBUG_MODE or request.headers.get("X-Debug-Mode") == "true":
            g.user = {"id": "debug-user", "role": "admin"}
            return f(*args, **kwargs)
        
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token required"}), 401
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256", "none"])
            g.user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        
        return f(*args, **kwargs)
    return decorated

@require_auth
def delete_user(user_id):
    """Admin-only: delete a user account"""
    if g.user.get("role") != "admin":
        return jsonify({"error": "Admin required"}), 403
    
    # ... delete user logic ...
    return jsonify({"status": "deleted", "user_id": user_id})
''',
        "ground_truth": {
            "vuln_type": "logic",
            "cwe_id": "CWE-287",
            "description": "Multiple auth bypass vectors: 1) X-Debug-Mode header grants admin without any auth. 2) jwt.decode allows algorithms=['none'] which means unsigned tokens are accepted. 3) Hardcoded fallback secret 'dev-secret-key-change-in-prod'.",
            "vulnerable_lines": [6, 14, 15, 24],
            "attack_vector": "Send request with header 'X-Debug-Mode: true' to bypass auth entirely and get admin role. Alternatively, craft JWT with alg=none."
        },
        "good_audit": '''## 分析过程

### 数据流分析
- source: request.headers (Authorization, X-Debug-Mode)
- transform: jwt.decode 验证 token
- sink: g.user 设置 → 影响后续授权决策

### 控制流分析 ⚠️
正常路径: 获取 token → jwt.decode → 设置 g.user
绕过路径 1: X-Debug-Mode header = "true" → 直接获得 admin (line 14-15)
绕过路径 2: JWT algorithms=["HS256", "none"] → alg=none 绕过签名 (line 24)
绕过路径 3: 环境变量未设置时使用硬编码密钥 (line 6)

### 时序分析
- 非并发问题，但 DEBUG_MODE 可能在运行时被改变

## 发现

```json
{
  "overall_risk": "critical",
  "findings": [
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-287",
      "severity": "critical",
      "location": "require_auth() 第 14-15 行",
      "description": "认证绕过: X-Debug-Mode 请求头可被任何客户端设置，值为 'true' 时直接跳过认证并授予 admin 权限",
      "attack_vector": "curl -H 'X-Debug-Mode: true' http://target/admin/delete_user/123",
      "remediation": "移除 X-Debug-Mode 头部检查。如需调试模式，仅依赖服务端 DEBUG_MODE 环境变量，且生产环境必须为 false"
    },
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-327",
      "severity": "critical",
      "location": "jwt.decode() 第 24 行",
      "description": "JWT 'none' 算法攻击: algorithms 列表包含 'none'，攻击者可构造无签名的 JWT token 通过验证",
      "attack_vector": "构造 JWT header: {'alg': 'none', 'typ': 'JWT'}，payload 设置 role=admin，签名留空",
      "remediation": "从 algorithms 列表中移除 'none': jwt.decode(token, SECRET_KEY, algorithms=['HS256'])"
    },
    {
      "vuln_type": "hardcoded_secret",
      "cwe_id": "CWE-798",
      "severity": "high",
      "location": "第 6 行",
      "description": "硬编码的 JWT 密钥回退值 'dev-secret-key-change-in-prod'，如果环境变量未设置则使用此弱密钥",
      "remediation": "移除默认值，环境变量未设置时应抛出异常: SECRET_KEY = os.environ['JWT_SECRET']"
    }
  ],
  "safe_patterns": ["jwt.ExpiredSignatureError 的过期检查是正确的"]
}
```''',
        "bad_audit": '''## 分析

检查了 JWT 认证流程。

## 发现

```json
{
  "overall_risk": "medium",
  "findings": [
    {
      "vuln_type": "hardcoded_secret",
      "severity": "medium",
      "location": "第 6 行",
      "description": "硬编码的默认密钥",
      "remediation": "使用环境变量"
    }
  ]
}
```

认证流程使用了 JWT，有 token 验证和过期检查，基本安全。'''
    },

    # ══════════════════════════════════════════════════════════════════
    #  D3. 权限提升 (CWE-269)
    # ══════════════════════════════════════════════════════════════════

    {
        "id": "logic-privesc-001",
        "sub_type": "privilege_escalation",
        "code": '''from flask import Flask, request, jsonify, session
import sqlite3

app = Flask(__name__)
app.secret_key = "super-secret-key"

def get_db():
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/profile', methods=['GET'])
def get_profile():
    """Get user profile by user_id parameter"""
    user_id = request.args.get('user_id', session.get('user_id'))
    
    db = get_db()
    user = db.execute('SELECT id, name, email, role FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify(dict(user))

@app.route('/api/profile', methods=['PUT'])
def update_profile():
    """Update user profile"""
    data = request.get_json()
    user_id = session.get('user_id')
    
    # Build update query from provided fields
    allowed_fields = ['name', 'email', 'bio', 'avatar', 'role']
    updates = []
    values = []
    
    for field in allowed_fields:
        if field in data:
            updates.append(f"{field} = ?")
            values.append(data[field])
    
    if not updates:
        return jsonify({"error": "No fields to update"}), 400
    
    values.append(user_id)
    query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
    
    db = get_db()
    db.execute(query, values)
    db.commit()
    
    return jsonify({"status": "updated"})
''',
        "ground_truth": {
            "vuln_type": "logic",
            "cwe_id": "CWE-269",
            "description": "Two vulnerabilities: 1) IDOR in get_profile - user_id comes from request.args, any user can view others' profiles. 2) Mass assignment / privilege escalation in update_profile - 'role' is in allowed_fields, users can set their own role to 'admin'.",
            "vulnerable_lines": [16, 33],
            "attack_vector": "PUT /api/profile with body {\"role\": \"admin\"} to escalate to admin. GET /api/profile?user_id=1 to access any user's profile."
        },
        "good_audit": '''## 分析过程

### 数据流分析
- get_profile: source=request.args.get('user_id') → sink=SQL query (参数化✓)
- update_profile: source=request.get_json() → transform=allowed_fields 过滤 → sink=SQL UPDATE

### 控制流分析 ⚠️
- get_profile: 无权限检查！user_id 来自请求参数而非 session
- update_profile: allowed_fields 包含 'role' 字段 → 用户可自行修改角色

## 发现

```json
{
  "overall_risk": "critical",
  "findings": [
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-639",
      "severity": "high",
      "location": "get_profile() 第 16 行",
      "description": "IDOR (不安全的直接对象引用): user_id 参数来自 request.args，任何认证用户可以通过修改 user_id 查看其他用户的 profile (包括 email, role 等敏感信息)",
      "attack_vector": "GET /api/profile?user_id=1 — 遍历所有 user_id 获取全部用户信息",
      "remediation": "移除 request.args.get('user_id')，强制使用 session 中的 user_id: user_id = session['user_id']"
    },
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-269",
      "severity": "critical",
      "location": "update_profile() 第 33 行",
      "description": "权限提升: allowed_fields 包含 'role'，普通用户可以通过 PUT 请求将自己的 role 改为 'admin'",
      "attack_vector": "PUT /api/profile -d '{\"role\": \"admin\"}' — 直接提升为管理员",
      "remediation": "从 allowed_fields 中移除 'role': allowed_fields = ['name', 'email', 'bio', 'avatar']。角色修改应有独立的管理员 API"
    }
  ],
  "safe_patterns": ["SQL 查询使用参数化 (?) 防止注入", "session 管理使用 Flask 内置机制"]
}
```''',
        "bad_audit": '''## 分析

代码使用了参数化 SQL 查询，防止了 SQL 注入。

```json
{
  "overall_risk": "low",
  "findings": [
    {
      "vuln_type": "hardcoded_secret",
      "severity": "medium",
      "location": "第 5 行",
      "description": "硬编码的 secret_key",
      "remediation": "使用环境变量"
    }
  ],
  "safe_patterns": ["参数化 SQL 查询", "session 管理"]
}
```'''
    },

    # ══════════════════════════════════════════════════════════════════
    #  D4. 业务逻辑缺陷 (CWE-840)
    # ══════════════════════════════════════════════════════════════════

    {
        "id": "logic-bizlogic-001",
        "sub_type": "business_logic",
        "code": '''from flask import Flask, request, jsonify, session
from decimal import Decimal
import uuid

app = Flask(__name__)
orders = {}  # In-memory order store

@app.route('/api/order', methods=['POST'])
def create_order():
    """Create a new order"""
    data = request.get_json()
    items = data.get('items', [])
    
    total = Decimal('0')
    for item in items:
        price = Decimal(str(item['price']))
        quantity = item['quantity']
        total += price * quantity
    
    # Apply discount code
    discount_code = data.get('discount_code')
    if discount_code == "WELCOME50":
        total = total * Decimal('0.5')
    elif discount_code == "SAVE20":
        total = total * Decimal('0.8')
    
    order_id = str(uuid.uuid4())
    orders[order_id] = {
        'id': order_id,
        'user_id': session.get('user_id'),
        'items': items,
        'total': float(total),
        'status': 'pending_payment',
        'discount_code': discount_code,
    }
    
    return jsonify({"order_id": order_id, "total": float(total)})

@app.route('/api/order/<order_id>/pay', methods=['POST'])
def pay_order(order_id):
    """Process payment for an order"""
    order = orders.get(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404
    
    payment_amount = request.get_json().get('amount', 0)
    
    if payment_amount >= order['total']:
        order['status'] = 'paid'
        return jsonify({"status": "paid"})
    
    return jsonify({"error": "Insufficient payment"}), 400

@app.route('/api/order/<order_id>/update', methods=['PUT'])
def update_order(order_id):
    """Update order items before payment"""
    order = orders.get(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404
    
    data = request.get_json()
    
    # Recalculate total
    items = data.get('items', order['items'])
    total = Decimal('0')
    for item in items:
        price = Decimal(str(item['price']))
        quantity = item['quantity']
        total += price * quantity
    
    order['items'] = items
    order['total'] = float(total)
    
    return jsonify({"status": "updated", "new_total": float(total)})
''',
        "ground_truth": {
            "vuln_type": "logic",
            "cwe_id": "CWE-840",
            "description": "Multiple business logic flaws: 1) Price from client input - item prices come from request, not server-side catalog. 2) Negative quantity not checked - can result in negative total. 3) update_order doesn't re-apply discount but recalculates total. 4) update_order can modify after payment status check is missing. 5) No idempotency - pay_order can be called multiple times.",
            "vulnerable_lines": [14, 17, 45, 68, 69],
            "attack_vector": "1) Send items with price=0.01 to get goods at any price. 2) Send quantity=-100 to get a negative total (refund). 3) Create order with discount → update order (discount lost) → items still ship."
        },
        "good_audit": '''## 分析过程

### 数据流分析
- create_order: source=request.get_json() → items[].price 和 items[].quantity 均来自客户端 ⚠️
- pay_order: source=request.get_json().get('amount') → 与 order['total'] 比较
- update_order: source=request.get_json() → 直接替换 items 和 total

### 控制流分析 ⚠️
- create_order: 无服务端价格验证 → 客户端可设任意价格
- pay_order: 无状态检查 → 已支付订单可重复支付？已取消订单也能支付？
- update_order: 无状态检查 → 已支付订单也能修改 items！折扣码在更新时丢失

### 时序分析
- 无并发保护 → 理论上可能存在竞态，但主要问题是业务逻辑

## 发现

```json
{
  "overall_risk": "critical",
  "findings": [
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-840",
      "severity": "critical",
      "location": "create_order() 第 14-17 行",
      "description": "价格篡改: item price 来自客户端请求而非服务端商品目录，攻击者可以设置任意价格购买商品",
      "attack_vector": "POST /api/order -d '{\"items\": [{\"name\": \"iPhone\", \"price\": 0.01, \"quantity\": 1}]}'",
      "remediation": "价格必须从服务端商品目录获取: price = catalog.get_price(item['product_id'])，不信任客户端传来的价格"
    },
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-840",
      "severity": "high",
      "location": "create_order() 第 17 行",
      "description": "负数数量攻击: quantity 没有做正整数校验，负数 quantity 导致负数 total（相当于退款）",
      "attack_vector": "POST /api/order -d '{\"items\": [{\"name\": \"Gift Card\", \"price\": 100, \"quantity\": -10}]}'",
      "remediation": "添加数量校验: if not isinstance(quantity, int) or quantity <= 0: return error"
    },
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-840",
      "severity": "high",
      "location": "update_order() 第 56-69 行",
      "description": "订单状态检查缺失: 已支付的订单仍可修改 items 和 total。攻击者可先支付低价订单，再更新为高价商品",
      "attack_vector": "1) 创建低价订单并支付 2) PUT /api/order/{id}/update 替换为高价商品",
      "remediation": "添加状态检查: if order['status'] != 'pending_payment': return error('Cannot modify paid order')"
    },
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-840",
      "severity": "medium",
      "location": "update_order() 第 63-69 行",
      "description": "折扣码丢失: update_order 重算 total 时不再应用原始折扣码，用户可能通过更新操作使折扣失效或绕过限制",
      "remediation": "更新时重新应用原始折扣: discount = order.get('discount_code'); 重算后再应用折扣"
    }
  ],
  "safe_patterns": ["使用 Decimal 进行金额计算避免浮点精度问题", "使用 uuid4 生成订单 ID 不可预测"]
}
```''',
        "bad_audit": '''代码使用了 Decimal 进行金额计算，uuid 生成订单 ID。

```json
{
  "overall_risk": "low",
  "findings": [],
  "safe_patterns": ["Decimal 精确计算", "UUID 订单 ID", "折扣码机制"]
}
```

未发现明显安全漏洞。'''
    },

    # ══════════════════════════════════════════════════════════════════
    #  D5. 不安全默认配置 (CWE-1188)
    # ══════════════════════════════════════════════════════════════════

    {
        "id": "logic-insecure-default-001",
        "sub_type": "insecure_default",
        "code": '''from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import string
import logging

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app.config['DEBUG'] = True
app.config['SECRET_KEY'] = ''.join(random.choices(string.ascii_letters, k=32))

def generate_reset_token(user_email):
    """Generate password reset token"""
    token = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
    logger.debug(f"Generated reset token for {user_email}: {token}")
    return token

@app.route('/api/password-reset', methods=['POST'])
def password_reset():
    email = request.json.get('email')
    if not email:
        return jsonify({"error": "Email required"}), 400
    
    token = generate_reset_token(email)
    
    # Send email with reset link
    reset_link = f"https://example.com/reset?token={token}&email={email}"
    logger.info(f"Password reset requested for {email}")
    
    # ... send_email(email, reset_link) ...
    
    return jsonify({"message": "Reset email sent"})

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "details": str(e),
        "traceback": __import__('traceback').format_exc()
    }), 500
''',
        "ground_truth": {
            "vuln_type": "logic",
            "cwe_id": "CWE-1188",
            "description": "Multiple insecure defaults: 1) CORS allows all origins '*'. 2) DEBUG=True exposes internals. 3) random.choices for security token (not cryptographically secure). 4) Debug log contains plaintext token. 5) Error handler returns full traceback. 6) SECRET_KEY uses random (predictable) not secrets.",
            "vulnerable_lines": [8, 14, 15, 20, 21, 43, 44],
            "attack_vector": "1) Predict reset token via random seed. 2) Trigger 500 error to get traceback with secrets. 3) CORS * allows any website to call the API."
        },
        "good_audit": '''## 分析过程

### 数据流分析
- source: request.json.get('email') → 用于生成 reset token
- transform: random.choices() 生成 token ⚠️ (非密码学安全随机)
- sink: logger.debug() 明文记录 token ⚠️

### 控制流分析
- 无认证检查 → 任何人可触发密码重置
- 500 错误返回完整 traceback ⚠️

### 时序分析
- 非并发问题

## 发现

```json
{
  "overall_risk": "high",
  "findings": [
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-330",
      "severity": "critical",
      "location": "generate_reset_token() 第 20 行",
      "description": "使用 random.choices() 生成安全令牌，random 模块使用 Mersenne Twister PRNG，不适用于安全场景。攻击者可通过观察输出预测后续 token",
      "attack_vector": "收集足够的 token 样本后，使用 randcrack 库恢复 PRNG 状态，预测下一个 reset token",
      "remediation": "使用 secrets.token_urlsafe(32) 替代 random.choices()"
    },
    {
      "vuln_type": "info_leak",
      "cwe_id": "CWE-532",
      "severity": "high",
      "location": "第 21 行, 第 43-44 行",
      "description": "明文密码重置 token 写入日志(line 21)，500 错误返回完整 traceback 和异常详情(line 43-44)",
      "attack_vector": "访问日志获取 reset token；触发 500 错误获取服务器内部信息(文件路径、依赖版本等)",
      "remediation": "1) 移除 token 日志 2) 生产环境 error handler 只返回 error ID，不返回 details 和 traceback"
    },
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-1188",
      "severity": "high",
      "location": "第 8 行, 第 14 行",
      "description": "不安全默认配置: CORS 允许所有来源 (origins='*')，DEBUG=True 暴露调试信息",
      "remediation": "1) CORS 指定具体域名 2) 生产环境设置 DEBUG=False"
    },
    {
      "vuln_type": "logic",
      "cwe_id": "CWE-330",
      "severity": "medium",
      "location": "第 15 行",
      "description": "SECRET_KEY 使用 random 模块生成，应用重启后 key 变化导致所有 session 失效，且 random 不是密码学安全的",
      "remediation": "使用 secrets.token_hex(32) 并从环境变量加载固定值"
    }
  ],
  "safe_patterns": []
}
```''',
        "bad_audit": '''代码使用了 Flask 框架和 CORS 配置。

```json
{
  "overall_risk": "medium",
  "findings": [
    {
      "vuln_type": "info_leak",
      "severity": "low",
      "location": "error handler",
      "description": "错误信息可能包含敏感内容",
      "remediation": "简化错误响应"
    }
  ]
}
```'''
    },

    # ══════════════════════════════════════════════════════════════════
    #  NEGATIVE EXAMPLE — 安全代码 (降低误报)
    # ══════════════════════════════════════════════════════════════════

    {
        "id": "logic-safe-001",
        "sub_type": "safe_code",
        "code": '''import os
import secrets
import hashlib
import threading
from contextlib import contextmanager

class SecureFileProcessor:
    def __init__(self):
        self._lock = threading.Lock()
    
    def process_file(self, filepath):
        """Securely process a file with proper TOCTOU prevention"""
        try:
            # Open first, then check — prevents TOCTOU
            fd = os.open(filepath, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                stat = os.fstat(fd)
                
                # Check file size on the opened descriptor
                if stat.st_size > 10 * 1024 * 1024:
                    return {"error": "File too large"}
                
                # Read from the verified descriptor
                with os.fdopen(fd, 'rb') as f:
                    content = f.read()
                    fd = -1  # Mark as consumed
                
            finally:
                if fd >= 0:
                    os.close(fd)
        
        except OSError as e:
            return {"error": f"Cannot open file: {e}"}
        
        checksum = hashlib.sha256(content).hexdigest()
        return {"status": "ok", "checksum": checksum}

class SecureTransfer:
    def __init__(self):
        self._lock = threading.Lock()
        self._balance = 0
    
    def transfer(self, amount):
        """Thread-safe transfer with proper locking"""
        with self._lock:
            if self._balance >= amount:
                self._balance -= amount
                return True
            return False
''',
        "ground_truth": {
            "vuln_type": "none",
            "cwe_id": None,
            "description": "This code is secure. It uses O_NOFOLLOW to prevent symlink attacks, fstat on file descriptor (not path) to prevent TOCTOU, and threading.Lock for atomic balance operations.",
            "vulnerable_lines": [],
            "attack_vector": None
        },
        "good_audit": '''## 分析过程

### 数据流分析
- source: filepath → sink: os.open() (with O_NOFOLLOW)
- 数据流经过正确的安全检查

### 控制流分析
- SecureFileProcessor: 先 open → 再 fstat → 正确防止了 TOCTOU
- SecureTransfer: 使用 threading.Lock 保护余额操作

### 时序分析
- 文件操作: O_NOFOLLOW + fstat on fd → 无竞态
- 余额操作: Lock 保护 → 无竞态

## 发现

```json
{
  "overall_risk": "safe",
  "findings": [],
  "safe_patterns": [
    "os.O_NOFOLLOW 防止 symlink 攻击",
    "os.fstat(fd) 在文件描述符上检查，而非路径上检查 — 正确防止 TOCTOU",
    "threading.Lock 保护共享状态的原子操作",
    "hashlib.sha256 用于完整性校验"
  ]
}
```''',
        "bad_audit": '''检查了代码逻辑。

```json
{
  "overall_risk": "medium",
  "findings": [
    {
      "vuln_type": "logic",
      "severity": "medium",
      "location": "process_file()",
      "description": "文件操作可能存在竞态条件",
      "remediation": "添加文件锁"
    }
  ]
}
```

代码中的文件操作可能在并发环境下存在风险。'''
    },
]


# ──────────────────────────────────────────────────────────────────────
#  数据集生成函数
# ──────────────────────────────────────────────────────────────────────

def generate_sft_data(samples: list[dict] = None) -> list[dict]:
    """生成 SFT 训练数据 — 高质量审计推理链示范"""
    if samples is None:
        samples = LOGIC_SAMPLES
    
    sft_data = []
    for sample in samples:
        sft_data.append({
            "instruction": "对以下代码进行全面安全审计，包括数据流分析、控制流分析和时序分析。输出 JSON 格式的审计报告。",
            "input": f"**业务上下文:** 代码安全审计\n\n```python\n{sample['code']}\n```",
            "output": sample["good_audit"],
            "metadata": {
                "sample_id": sample["id"],
                "sub_type": sample["sub_type"],
                "cwe_id": sample["ground_truth"].get("cwe_id"),
            }
        })
    return sft_data


def generate_dpo_data(samples: list[dict] = None) -> list[dict]:
    """生成 DPO 训练数据 — chosen (好审计) vs rejected (差审计)"""
    if samples is None:
        samples = LOGIC_SAMPLES
    
    dpo_data = []
    for sample in samples:
        if "bad_audit" not in sample:
            continue
        dpo_data.append({
            "prompt": f"对以下代码进行全面安全审计：\n\n```python\n{sample['code']}\n```",
            "chosen": sample["good_audit"],
            "rejected": sample["bad_audit"],
            "metadata": {
                "sample_id": sample["id"],
                "sub_type": sample["sub_type"],
                "cwe_id": sample["ground_truth"].get("cwe_id"),
            }
        })
    return dpo_data


def save_jsonl(data: list[dict], output_path: str):
    """保存为 JSONL 格式"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SSPilot Logic Training Data Generator")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--format", choices=["sft", "dpo", "both"], default="sft", help="Data format")
    args = parser.parse_args()

    if args.format in ("sft", "both"):
        sft_data = generate_sft_data()
        sft_path = args.output if args.format == "sft" else args.output.replace(".jsonl", "_sft.jsonl")
        save_jsonl(sft_data, sft_path)

        # 统计
        by_type = {}
        for item in sft_data:
            st = item["metadata"]["sub_type"]
            by_type[st] = by_type.get(st, 0) + 1
        print(f"  Sub-type distribution: {by_type}")

    if args.format in ("dpo", "both"):
        dpo_data = generate_dpo_data()
        dpo_path = args.output if args.format == "dpo" else args.output.replace(".jsonl", "_dpo.jsonl")
        save_jsonl(dpo_data, dpo_path)

    print(f"\n📊 Summary:")
    print(f"  Total logic samples: {len(LOGIC_SAMPLES)}")
    print(f"  Sub-types covered: toctou, auth_bypass, privilege_escalation, business_logic, insecure_default, safe_code")
    print(f"  Each sample includes: code + ground_truth + good_audit (reasoning chain) + bad_audit (for DPO)")


if __name__ == "__main__":
    main()
