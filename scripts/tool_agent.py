"""
SSPilot — Tool-Augmented AuditAgent
赋予 AuditAgent 工具调用能力：LLM 在审计过程中可主动调用安全分析工具，
将工具结果纳入推理上下文，经过多轮 reason-act 循环后输出最终审计报告。

工具清单：
  - regex_scan   : 正则匹配常见漏洞模式
  - ast_analyze  : Python AST 静态分析（危险调用 + 污点追踪）
  - cwe_lookup   : CWE 知识库查询
  - dependency_check : 导入模块风险检查
"""
import ast
import json
import re
from datetime import datetime
from typing import Any

from config import BATTLE_CONFIG
from model_manager import manager

# ═══════════════════════════════════════════════════════════════════
# Security Analysis Tools
# ═══════════════════════════════════════════════════════════════════

VULN_PATTERNS: dict[str, list[tuple[str, str, str]]] = {
    "sqli": [
        (r"""f['\"].*(?:SELECT|INSERT|UPDATE|DELETE|DROP)\b""", "F-string SQL query", "CWE-89"),
        (r"""['\"].*%s.*['\"].*%""", "%-format SQL query", "CWE-89"),
        (r"""\.format\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)""", ".format() SQL query", "CWE-89"),
        (r"""execute\(\s*[f'\"]""", "Direct string in execute()", "CWE-89"),
    ],
    "xss": [
        (r"""return\s+f['\"].*<.*\{""", "Unescaped f-string in HTML", "CWE-79"),
        (r"""render_template_string\(""", "render_template_string usage", "CWE-79"),
        (r"""innerHTML\s*=""", "Direct innerHTML assignment", "CWE-79"),
        (r"""Markup\(""", "Markup() without escaping", "CWE-79"),
    ],
    "path_traversal": [
        (r"""open\(.*\+.*\)""", "User input in file open()", "CWE-22"),
        (r"""open\(f['\"]""", "F-string in file path", "CWE-22"),
        (r"""os\.path\.join\(.*request""", "Request data in path join", "CWE-22"),
    ],
    "ssrf": [
        (r"""requests\.(?:get|post|put|delete)\(.*request""", "User URL in requests call", "CWE-918"),
        (r"""urlopen\(.*request""", "User URL in urlopen", "CWE-918"),
        (r"""urllib.*request\.args""", "Request args in urllib", "CWE-918"),
    ],
    "cmdi": [
        (r"""os\.(?:system|popen)\(""", "os.system/popen usage", "CWE-78"),
        (r"""subprocess.*shell\s*=\s*True""", "subprocess with shell=True", "CWE-78"),
        (r"""eval\(""", "eval() usage", "CWE-94"),
        (r"""exec\(""", "exec() usage", "CWE-94"),
    ],
    "unsafe_deser": [
        (r"""pickle\.loads?\(""", "pickle deserialization", "CWE-502"),
        (r"""yaml\.load\(""", "yaml.load without SafeLoader", "CWE-502"),
        (r"""marshal\.loads?\(""", "marshal deserialization", "CWE-502"),
        (r"""shelve\.open\(""", "shelve with pickle backend", "CWE-502"),
    ],
    "hardcoded_secret": [
        (r"""(?:password|passwd|secret|api_key|token)\s*=\s*['\"][^'\"]{8,}""",
         "Hardcoded credential", "CWE-798"),
        (r"""(?:AWS|AKIA)[A-Z0-9]{12,}""", "AWS key pattern", "CWE-798"),
    ],
    "info_leak": [
        (r"""traceback\.format_exc\(\)""", "Traceback in response", "CWE-209"),
        (r"""debug\s*=\s*True""", "Debug mode enabled", "CWE-209"),
        (r"""\.format_exc\(""", "Exception details exposed", "CWE-209"),
    ],
}

DANGEROUS_CALLS = {
    "eval": ("Code injection", "CWE-94", "critical"),
    "exec": ("Code injection", "CWE-94", "critical"),
    "compile": ("Dynamic code compilation", "CWE-94", "high"),
    "os.system": ("OS command execution", "CWE-78", "critical"),
    "os.popen": ("OS command execution", "CWE-78", "critical"),
    "subprocess.call": ("Subprocess execution", "CWE-78", "high"),
    "subprocess.Popen": ("Subprocess execution", "CWE-78", "high"),
    "pickle.loads": ("Unsafe deserialization", "CWE-502", "critical"),
    "pickle.load": ("Unsafe deserialization", "CWE-502", "critical"),
    "yaml.load": ("Unsafe YAML parsing", "CWE-502", "high"),
    "marshal.loads": ("Unsafe deserialization", "CWE-502", "high"),
    "sqlite3.execute": ("Direct SQL — check for injection", "CWE-89", "medium"),
    "__import__": ("Dynamic import", "CWE-94", "medium"),
}

RISKY_MODULES = {
    "pickle": ("Arbitrary code execution via deserialization", "CWE-502", "critical"),
    "shelve": ("Pickle-backed storage", "CWE-502", "high"),
    "marshal": ("Unsafe deserialization", "CWE-502", "high"),
    "subprocess": ("OS command execution", "CWE-78", "high"),
    "os": ("File system and command access", "CWE-78", "medium"),
    "xml.etree": ("XML parsing — XXE risk", "CWE-611", "medium"),
    "lxml": ("XML parsing — XXE risk if misconfigured", "CWE-611", "medium"),
    "ctypes": ("Native memory access", "CWE-120", "medium"),
    "tempfile": ("Temp file race conditions", "CWE-377", "low"),
}

CWE_DB: dict[str, dict[str, str]] = {
    "CWE-22":  {"name": "Path Traversal", "severity": "high",
                "desc": "Improper limitation of a pathname to a restricted directory",
                "fix": "Validate and canonicalize paths; reject '..' sequences; use os.path.realpath() + startswith check"},
    "CWE-78":  {"name": "OS Command Injection", "severity": "critical",
                "desc": "Improper neutralization of special elements in OS commands",
                "fix": "Use subprocess with list args (no shell=True); never interpolate user input into commands"},
    "CWE-79":  {"name": "Cross-site Scripting (XSS)", "severity": "high",
                "desc": "Improper neutralization of input during web page generation",
                "fix": "Use template engine auto-escaping (Jinja2); html.escape() for manual output"},
    "CWE-89":  {"name": "SQL Injection", "severity": "critical",
                "desc": "Improper neutralization of special elements in SQL commands",
                "fix": "Use parameterized queries / prepared statements; never concatenate user input into SQL"},
    "CWE-94":  {"name": "Code Injection", "severity": "critical",
                "desc": "Improper control of generation of code",
                "fix": "Avoid eval/exec; use AST literal_eval for safe parsing; sandbox if code execution is required"},
    "CWE-120": {"name": "Buffer Overflow", "severity": "high",
                "desc": "Buffer copy without checking size of input",
                "fix": "Validate buffer sizes; use safe memory APIs; bounds checking"},
    "CWE-209": {"name": "Information Exposure via Error Message", "severity": "medium",
                "desc": "Generation of error messages containing sensitive information",
                "fix": "Return generic error messages to users; log detailed errors server-side only"},
    "CWE-377": {"name": "Insecure Temporary File", "severity": "low",
                "desc": "Creating temporary files in insecure manner",
                "fix": "Use tempfile.mkstemp() or NamedTemporaryFile with appropriate permissions"},
    "CWE-502": {"name": "Deserialization of Untrusted Data", "severity": "critical",
                "desc": "Deserializing data from untrusted sources can execute arbitrary code",
                "fix": "Use JSON instead of pickle; if pickle needed, verify HMAC signature first"},
    "CWE-611": {"name": "XXE (XML External Entity)", "severity": "high",
                "desc": "Improper restriction of XML external entity reference",
                "fix": "Disable external entity resolution; use defusedxml library"},
    "CWE-798": {"name": "Hardcoded Credentials", "severity": "high",
                "desc": "Use of hard-coded credentials in source code",
                "fix": "Use environment variables or secrets manager; never commit credentials"},
    "CWE-918": {"name": "Server-Side Request Forgery (SSRF)", "severity": "high",
                "desc": "Server makes requests to user-controlled URLs",
                "fix": "Allowlist valid domains; block private IP ranges; validate URL scheme"},
}


def tool_regex_scan(code: str, patterns: list[str] | None = None) -> dict:
    """Scan code with security-focused regex patterns."""
    if patterns is None:
        patterns = list(VULN_PATTERNS.keys())

    matches = []
    lines = code.split("\n")
    for cat in patterns:
        cat_patterns = VULN_PATTERNS.get(cat, [])
        for regex, desc, cwe in cat_patterns:
            try:
                for i, line in enumerate(lines, 1):
                    if re.search(regex, line, re.IGNORECASE):
                        matches.append({
                            "line": i,
                            "category": cat,
                            "pattern": desc,
                            "cwe": cwe,
                            "code_line": line.strip(),
                        })
            except re.error:
                continue

    return {"matches": matches, "total": len(matches), "categories_scanned": patterns}


def tool_ast_analyze(code: str) -> dict:
    """Analyze code with Python AST for dangerous calls and taint sources."""
    result = {"dangerous_calls": [], "imports": [], "taint_sources": [], "parse_error": None}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["parse_error"] = f"SyntaxError: {e}"
        return result

    for node in ast.walk(tree):
        # Dangerous function calls
        if isinstance(node, ast.Call):
            fname = _resolve_call_name(node)
            if fname in DANGEROUS_CALLS:
                desc, cwe, sev = DANGEROUS_CALLS[fname]
                result["dangerous_calls"].append({
                    "function": fname,
                    "line": getattr(node, "lineno", 0),
                    "description": desc,
                    "cwe": cwe,
                    "severity": sev,
                })

        # Import tracking
        if isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result["imports"].append(node.module)

        # Taint sources (web framework input patterns)
        if isinstance(node, ast.Attribute):
            attr_chain = _resolve_attr_chain(node)
            taint_patterns = [
                "request.args", "request.form", "request.data",
                "request.json", "request.values", "request.cookies",
                "request.headers", "request.files",
            ]
            for tp in taint_patterns:
                if tp in attr_chain:
                    result["taint_sources"].append({
                        "source": attr_chain,
                        "line": getattr(node, "lineno", 0),
                        "type": "web_input",
                    })
                    break

    return result


def tool_cwe_lookup(cwe_id: str) -> dict:
    """Look up a CWE entry by ID."""
    cwe_id = cwe_id.upper().strip()
    if not cwe_id.startswith("CWE-"):
        cwe_id = "CWE-" + cwe_id.lstrip("CWE").lstrip("-")
    entry = CWE_DB.get(cwe_id)
    if entry:
        return {"cwe_id": cwe_id, "found": True, **entry}
    return {"cwe_id": cwe_id, "found": False, "desc": "CWE not in local database"}


def tool_dependency_check(code: str) -> dict:
    """Check imported modules for known security risks."""
    findings = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"findings": [], "error": "Could not parse code"}

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    for mod in imported:
        if mod in RISKY_MODULES:
            desc, cwe, sev = RISKY_MODULES[mod]
            findings.append({
                "module": mod,
                "risk": desc,
                "cwe": cwe,
                "severity": sev,
            })

    return {"imports_checked": sorted(imported), "findings": findings, "total_risky": len(findings)}


# ── AST helpers ──────────────────────────────────────────────────

def _resolve_call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return _resolve_attr_chain(func)
    return ""


def _resolve_attr_chain(node: ast.Attribute, depth: int = 0) -> str:
    if depth > 8:
        return ""
    if isinstance(node.value, ast.Attribute):
        return _resolve_attr_chain(node.value, depth + 1) + "." + node.attr
    if isinstance(node.value, ast.Name):
        return node.value.id + "." + node.attr
    return node.attr


# ═══════════════════════════════════════════════════════════════════
# Tool Registry & Executor
# ═══════════════════════════════════════════════════════════════════

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "regex_scan": {
        "fn": tool_regex_scan,
        "description": "Scan code with security regex patterns. Finds common vulnerability signatures like SQLi, XSS, path traversal, SSRF, command injection, etc.",
        "parameters": {
            "code": "The source code to scan (string, required)",
            "patterns": "List of categories to check: sqli, xss, path_traversal, ssrf, cmdi, unsafe_deser, hardcoded_secret, info_leak. Default: all.",
        },
    },
    "ast_analyze": {
        "fn": tool_ast_analyze,
        "description": "Static analysis via Python AST. Finds dangerous function calls (eval, exec, os.system, pickle.loads ...), tracks imports, and identifies taint sources (web framework request objects).",
        "parameters": {
            "code": "The source code to analyze (string, required)",
        },
    },
    "cwe_lookup": {
        "fn": tool_cwe_lookup,
        "description": "Look up CWE (Common Weakness Enumeration) details. Returns name, severity, description, and recommended fix for a given CWE ID.",
        "parameters": {
            "cwe_id": "CWE identifier, e.g. 'CWE-89' or '89' (string, required)",
        },
    },
    "dependency_check": {
        "fn": tool_dependency_check,
        "description": "Check imported Python modules for known security risks. Identifies risky modules like pickle, subprocess, os, xml parsers, etc.",
        "parameters": {
            "code": "The source code to analyze (string, required)",
        },
    },
}


def execute_tool(name: str, args: dict, code_context: str = "") -> dict:
    """Execute a registered tool by name."""
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return {"error": f"Unknown tool: {name}", "available": list(TOOL_REGISTRY.keys())}

    fn = tool["fn"]

    # Auto-inject code context for tools that need it
    if "code" in tool["parameters"] and "code" not in args:
        args["code"] = code_context

    try:
        return fn(**args)
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}


# ═══════════════════════════════════════════════════════════════════
# Tool-Augmented Audit Loop
# ═══════════════════════════════════════════════════════════════════

def _build_tool_descriptions() -> str:
    """Build a formatted tool list for the system prompt."""
    parts = []
    for name, meta in TOOL_REGISTRY.items():
        params = ", ".join(f"{k}: {v}" for k, v in meta["parameters"].items())
        parts.append(f"  - {name}({params})\n    {meta['description']}")
    return "\n".join(parts)


TOOL_SYSTEM_PROMPT = """你是 SSPilot AuditAgent，一位具有工具调用能力的顶级代码安全审计专家。

你可以在审计过程中调用以下安全分析工具来辅助你的判断：

{tools}

每一轮你可以选择：
1. 调用工具获取更多信息 — 输出 JSON:
   {{"tool_call": {{"name": "工具名", "args": {{...}}}}}}

2. 完成审计并输出最终报告 — 输出 JSON:
   {{"final_report": {{
       "overall_risk": "critical|high|medium|low|safe",
       "findings": [
           {{
               "vuln_type": "漏洞类型",
               "severity": "critical|high|medium|low|info",
               "location": "行号或函数名",
               "description": "详细描述",
               "attack_vector": "攻击方式",
               "remediation": "修复建议",
               "cwe": "CWE-XXX",
               "tool_evidence": "支持该发现的工具输出摘要（如有）"
           }}
       ],
       "tools_used": ["本次审计调用过的工具列表"],
       "summary": "审计总结"
   }}}}

规则：
- 每次只能输出一个 JSON 对象（tool_call 或 final_report）
- 不要输出 markdown 代码块，只输出纯 JSON
- 先调用工具收集证据，再综合分析形成最终报告
- 工具结果仅作参考，你的专业判断才是最终依据
- 最多调用 {max_rounds} 轮工具后必须输出 final_report"""


def _parse_tool_output(raw: str) -> dict:
    """Parse LLM output as either a tool_call or final_report."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        return {"parse_error": True, "raw": raw[:500]}


def tool_augmented_audit(
    code: str,
    sample_id: str = "",
    max_rounds: int | None = None,
) -> dict:
    """
    Execute a tool-augmented security audit.

    The LLM iteratively calls tools and reasons about the results before
    producing a final structured audit report.

    Args:
        code: Source code to audit
        sample_id: Sample identifier for logging
        max_rounds: Max tool-calling rounds (default from config)

    Returns:
        dict: Structured audit report compatible with standard audit_report schema
    """
    if max_rounds is None:
        max_rounds = BATTLE_CONFIG.get("max_tool_rounds", 5)

    tool_descs = _build_tool_descriptions()
    system_prompt = TOOL_SYSTEM_PROMPT.format(tools=tool_descs, max_rounds=max_rounds)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请对以下代码进行安全审计：\n\n```python\n{code}\n```"},
    ]

    tools_used = []
    tool_log = []

    for round_idx in range(max_rounds):
        raw = manager.generate(
            "agent",
            messages,
            temperature=BATTLE_CONFIG["temperature_agent"],
            max_new_tokens=BATTLE_CONFIG["max_new_tokens"],
        )

        parsed = _parse_tool_output(raw)

        if "final_report" in parsed:
            report = parsed["final_report"]
            report["sample_id"] = sample_id
            report["audit_timestamp"] = datetime.now().isoformat()
            report["model"] = "Qwen2.5-Coder-32B-Int4"
            report["audit_mode"] = "tool_augmented"
            report["tool_rounds"] = round_idx
            report["tool_log"] = tool_log
            if "tools_used" not in report:
                report["tools_used"] = tools_used
            return report

        if "tool_call" in parsed:
            tc = parsed["tool_call"]
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})

            result = execute_tool(tool_name, tool_args, code_context=code)
            tools_used.append(tool_name)
            tool_log.append({
                "round": round_idx + 1,
                "tool": tool_name,
                "args": {k: v[:100] if isinstance(v, str) else v for k, v in tool_args.items()},
                "result_summary": _summarize_result(result),
            })

            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"工具 {tool_name} 返回结果：\n{json.dumps(result, ensure_ascii=False, indent=2)[:2000]}\n\n请继续分析，或调用其他工具，或输出最终报告。",
            })

            print(f"    [Tool R{round_idx + 1}] {tool_name} → {_summarize_result(result)}")
            continue

        # Parse error or unexpected format — force final report on next round
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": "请直接输出 final_report JSON 格式的最终审计报告。",
        })

    # Exhausted rounds — force a final answer
    messages.append({
        "role": "user",
        "content": "工具调用轮次已用完，请立即输出 final_report JSON 格式的最终审计报告。",
    })
    raw = manager.generate(
        "agent", messages,
        temperature=BATTLE_CONFIG["temperature_agent"],
        max_new_tokens=BATTLE_CONFIG["max_new_tokens"],
    )
    parsed = _parse_tool_output(raw)
    if "final_report" in parsed:
        report = parsed["final_report"]
    else:
        report = {
            "overall_risk": "unknown",
            "findings": [],
            "summary": raw[:500],
            "parse_error": True,
        }

    report["sample_id"] = sample_id
    report["audit_timestamp"] = datetime.now().isoformat()
    report["model"] = "Qwen2.5-Coder-32B-Int4"
    report["audit_mode"] = "tool_augmented"
    report["tool_rounds"] = max_rounds
    report["tool_log"] = tool_log
    report["tools_used"] = tools_used
    return report


def _summarize_result(result: dict) -> str:
    """Create a short summary of a tool result for logging."""
    if "error" in result:
        return f"error: {result['error']}"
    if "total" in result:
        return f"{result['total']} matches"
    if "dangerous_calls" in result:
        n = len(result["dangerous_calls"])
        return f"{n} dangerous calls, {len(result.get('taint_sources', []))} taint sources"
    if "found" in result:
        return f"{'found' if result['found'] else 'not found'}: {result.get('name', result.get('cwe_id', ''))}"
    if "findings" in result:
        return f"{result.get('total_risky', len(result['findings']))} risky modules"
    return str(result)[:80]
