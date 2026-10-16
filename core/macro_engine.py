"""酒馆宏变量（Macros）与轻量 Handlebars 模板渲染引擎。

支持：
- 基础变量：``{{char}}`` ``{{user}}`` ``{{description}}`` ``{{personality}}``
  ``{{scenario}}`` ``{{persona}}`` ``{{mesExamples}}`` ``{{input}}`` ``{{original}}``
  ``{{wiBefore}}`` ``{{wiAfter}}`` ``{{lastUserMessage}}`` 等（由调用方注入 vars）；
- 时间变量：``{{time}}`` ``{{date}}`` ``{{weekday}}`` ``{{isotime}}`` ``{{isodate}}``；
- 随机函数：``{{random:a,b,c}}`` ``{{pick:a,b,c}}`` ``{{roll:2d6+1}}``；
- 注释：``{{// 任意内容 }}`` —— 整块从提示词中移除（酒馆即如此，实测某些真实
  预设定整条条目就是一段注释，若不剥离会把数千字作者说明发给模型）；
- 变量宏：``{{setvar::名::值}}`` ``{{addvar::名::值}}`` ``{{getvar::名}}``
  ``{{incvar::名}}`` ``{{decvar::名}}``（含 ``setglobalvar`` 等同义写法）。
  变量按会话持久化，可被 ``/预设 变量`` 命令手动改写——真实预设常用它做功能开关；
- Handlebars 子集：``{{#if var}}...{{else}}...{{/if}}``（支持嵌套）与 ``{{trim}}``。

已知但为空的变量替换为空串；未识别变量原样保留，方便排查笔误。
纯标准库实现，可脱离 AstrBot 单独测试。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime

_FUNC_RE = re.compile(r"\{\{\s*(random|pick|roll)\s*:\s*([^{}]+?)\s*\}\}", re.IGNORECASE)
_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][\w]*)\s*\}\}")
# 匹配最内层 if 块（内部不再含 if），循环执行以支持嵌套
_IF_INNERMOST_RE = re.compile(
    r"\{\{#if\s+([A-Za-z_][\w]*)\s*\}\}((?:(?!\{\{#if|\{\{/if\}\}).)*?)\{\{/if\}\}",
    re.DOTALL,
)
_ELSE_RE = re.compile(r"\{\{\s*else\s*\}\}")
_TRIM_RE = re.compile(r"\{\{\s*trim\s*\}\}")
_ROLL_RE = re.compile(r"(\d*)\s*d\s*(\d+)\s*([+-]\s*\d+)?", re.IGNORECASE)

# 注释宏：{{// 任意内容 }}（不含 }} 即可）
_COMMENT_RE = re.compile(r"\{\{\s*//.*?\}\}", re.DOTALL)
# 变量宏（:: 分隔的标准写法）
_VAR_MACRO_RE = re.compile(
    r"\{\{\s*(setvar|addvar|incvar|decvar|getvar|setglobalvar|addglobalvar|getglobalvar)"
    r"\s*::\s*([^{}]*?)\s*\}\}",
    re.IGNORECASE,
)
# 变量宏（旧式单冒号写法，仅用于 setvar:name:value / getvar:name）
_VAR_MACRO_LEGACY_RE = re.compile(
    r"\{\{\s*(setvar|addvar|incvar|decvar|getvar)\s*:\s*([A-Za-z_][\w]*)\s*"
    r"(?::\s*([^{}]*?))?\s*\}\}",
    re.IGNORECASE,
)

_MAX_IF_ROUNDS = 50
_MUTATING_MACROS = {"setvar", "addvar", "incvar", "decvar", "setglobalvar", "addglobalvar"}
_READING_MACROS = {"getvar", "getglobalvar"}



def time_vars(now: datetime = None) -> dict:
    now = now or datetime.now()
    return {
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%Y-%m-%d"),
        "weekday": "星期" + "一二三四五六日"[now.weekday()],
        "isotime": now.isoformat(timespec="seconds"),
        "isodate": now.strftime("%Y-%m-%d"),
    }


@dataclass
class MacroContext:
    vars: dict = field(default_factory=dict)          # 内置变量（char/user/…）
    rng: random.Random = field(default_factory=random.Random)
    variables: dict = field(default_factory=dict)     # 酒馆变量（setvar/getvar 读写）

    def truthy(self, name: str) -> bool:
        if name in self.vars:
            return bool(str(self.vars.get(name) or "").strip())
        return bool(str(self.variables.get(name) or "").strip())


def _render_variable_macros(text: str, ctx: MacroContext) -> str:
    """处理 setvar/addvar/incvar/decvar/getvar 宏，产出空串或变量值。"""

    def _apply(name_parts: list, raw: str) -> str:
        macro = name_parts[0].strip().lower()
        var_name = name_parts[1].strip() if len(name_parts) > 1 else ""
        value = name_parts[2] if len(name_parts) > 2 else ""
        if not var_name:
            return ""
        if macro in _READING_MACROS:
            return str(ctx.variables.get(var_name, "") or "")
        if macro in ("setvar", "setglobalvar"):
            ctx.variables[var_name] = value
            return ""
        if macro in ("addvar", "addglobalvar"):
            ctx.variables[var_name] = str(ctx.variables.get(var_name, "") or "") + value
            return ""
        # incvar / decvar
        try:
            current = int(str(ctx.variables.get(var_name, "0") or "0").strip() or 0)
        except ValueError:
            current = 0
        step = 1
        if value.strip():
            try:
                step = int(value.strip())
            except ValueError:
                step = 1
        ctx.variables[var_name] = str(current + (step if macro == "incvar" else -step))
        return ""

    def _sub_std(m: re.Match) -> str:
        return _apply([m.group(1), *m.group(2).split("::")], m.group(0))

    def _sub_legacy(m: re.Match) -> str:
        parts = [m.group(1), m.group(2), m.group(3) or ""]
        return _apply(parts, m.group(0))

    out = _VAR_MACRO_RE.sub(_sub_std, text)
    out = _VAR_MACRO_LEGACY_RE.sub(_sub_legacy, out)
    return out


def _render_functions(text: str, ctx: MacroContext) -> str:
    def _sub(m: re.Match) -> str:
        func = m.group(1).lower()
        spec = m.group(2)
        if func in ("random", "pick"):
            # 兼容 {{random:a,b,c}} 与 {{random::a::b::c}}（后者多出的前导冒号要剥掉）
            raw = spec.replace("::", ",")
            options = [x.strip().lstrip(":").strip() for x in raw.split(",")]
            options = [x for x in options if x]
            return ctx.rng.choice(options) if options else ""
        # roll: NdM±K
        rm = _ROLL_RE.search(spec)
        if not rm:
            return m.group(0)
        n = int(rm.group(1)) if rm.group(1) else 1
        sides = int(rm.group(2))
        bonus = int(rm.group(3).replace(" ", "")) if rm.group(3) else 0
        if n < 1 or sides < 1 or n > 100:
            return m.group(0)
        return str(sum(ctx.rng.randint(1, sides) for _ in range(n)) + bonus)

    return _FUNC_RE.sub(_sub, text)


def _render_vars(text: str, ctx: MacroContext) -> str:
    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name in ctx.vars:
            return str(ctx.vars.get(name) or "")
        if name in ctx.variables:      # getvar 结果之外，也允许直接用 {{名}} 引用
            return str(ctx.variables.get(name) or "")
        return m.group(0)  # 未识别变量原样保留

    return _VAR_RE.sub(_sub, text)


def render_text(text: str, ctx: MacroContext) -> str:
    """渲染普通提示词条目：注释剥离 + 变量宏 + 函数 + 变量。"""
    if not text:
        return ""
    out = _COMMENT_RE.sub("", text)
    out = _render_variable_macros(out, ctx)
    out = _render_functions(out, ctx)
    out = _render_vars(out, ctx)
    return out


def render_template(template: str, ctx: MacroContext) -> str:
    """渲染 story_string 级别的 Handlebars 子集：注释、变量宏、if/else、trim、函数、变量。"""
    if not template:
        return ""
    out = _COMMENT_RE.sub("", template)
    out = _render_variable_macros(out, ctx)
    for _ in range(_MAX_IF_ROUNDS):
        m = _IF_INNERMOST_RE.search(out)
        if not m:
            break
        var_name, body = m.group(1), m.group(2)
        parts = _ELSE_RE.split(body, maxsplit=1)
        keep = parts[0] if ctx.truthy(var_name) else (parts[1] if len(parts) > 1 else "")
        out = out[: m.start()] + keep + out[m.end():]
    trim = bool(_TRIM_RE.search(out))
    out = _TRIM_RE.sub("", out)
    out = _render_functions(out, ctx)
    out = _render_vars(out, ctx)
    return out.strip() if trim else out

