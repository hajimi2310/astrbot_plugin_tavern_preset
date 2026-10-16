"""酒馆（SillyTavern）预设解析器。

识别并归一化酒馆导出的三类预设 JSON：

1. Chat Completion 预设（含 ``prompts`` 列表 + ``prompt_order`` 编排）—— 完整支持；
2. Context 预设（``story_string`` Handlebars 模板）—— 作为「信息装配模板」支持；
3. Sysprompt 预设（``name`` + ``content`` + ``post_history``）—— 归一化为
   主提示 + 一条深度 0 注入。

本模块只依赖标准库，可脱离 AstrBot 单独测试。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

TYPE_CHAT = "chat"
TYPE_CONTEXT = "context"
TYPE_SYSPROMPT = "sysprompt"
TYPE_UNKNOWN = "unknown"

TYPE_LABELS = {
    TYPE_CHAT: "Chat Completion 预设",
    TYPE_CONTEXT: "Context 上下文预设",
    TYPE_SYSPROMPT: "Sysprompt 系统提示预设",
    TYPE_UNKNOWN: "未识别",
}

# 酒馆采样字段 -> 通用请求字段
SAMPLING_KEY_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "top_a": "top_a",
    "repetition_penalty": "repetition_penalty",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "openai_max_tokens": "max_tokens",
    "seed": "seed",
}

VALID_ROLES = ("system", "user", "assistant")


@dataclass
class PresetEntry:
    """预设中的一条提示词项（对应酒馆 prompts[] 的一项）。"""

    identifier: str
    name: str = ""
    role: str = "system"
    content: str = ""
    marker: bool = False            # True = 占位标记（chatHistory/charDescription 等）
    enabled: bool = True            # 条目自身的启用状态（prompt_order 未收录时的兜底）
    injection_position: int = 0     # 0 = 按相对顺序；1 = In-Chat @ Depth 深度注入
    injection_depth: int = 4
    injection_order: int = 100      # 同深度注入排序：数值越大越靠近生成位置
    injection_role: Optional[int] = None  # 0 system / 1 user / 2 assistant，None=跟随 role

    def depth_role_str(self) -> str:
        if self.injection_role is None:
            return self.role
        return VALID_ROLES[self.injection_role] if self.injection_role in (0, 1, 2) else "system"


@dataclass
class TavernPreset:
    """归一化后的预设模型。"""

    name: str
    type: str
    raw: dict = field(default_factory=dict)
    source_path: str = ""
    entries: list = field(default_factory=list)          # list[PresetEntry]
    prompt_orders: list = field(default_factory=list)    # 原始 prompt_order 列表
    # context 预设字段
    story_string: str = ""
    story_string_position: int = 0   # 0 = 头部默认位置；1 = In-Chat @ Depth
    story_string_depth: int = 1
    story_string_role: int = 0
    example_separator: str = ""
    chat_start: str = ""
    # 生成参数
    sampling: dict = field(default_factory=dict)
    assistant_prefill: str = ""
    squash_system_messages: bool = False
    utility_prompts: dict = field(default_factory=dict)  # impersonation/continue_nudge 等

    def ordered_entries(self, prefer_character_id: int = 100000):
        """按 prompt_order 排序，返回 [(PresetEntry, enabled)]。

        无 prompt_order 时按 prompts 原序（沿用条目自身的 enabled）；
        prompt_order 未收录的条目追加在末尾，同样沿用条目自身的 enabled——
        真实预设（如泉此方预设）会在 prompts[] 里写 enabled:false。
        """
        if self.type == TYPE_SYSPROMPT:
            return [(e, True) for e in self.entries]
        if not self.prompt_orders:
            return [(e, e.enabled) for e in self.entries]
        chosen = None
        for po in self.prompt_orders:
            if po.get("character_id") == prefer_character_id:
                chosen = po
                break
        if chosen is None:
            chosen = self.prompt_orders[0]
        by_id = {e.identifier: e for e in self.entries}
        result, used = [], set()
        for item in chosen.get("order", []):
            ident = item.get("identifier")
            if ident in by_id:
                result.append((by_id[ident], bool(item.get("enabled", True))))
                used.add(ident)
        for e in self.entries:
            if e.identifier not in used:
                result.append((e, e.enabled))
        return result

    @property
    def type_label(self) -> str:
        return TYPE_LABELS.get(self.type, self.type)


def detect_type(data: Any) -> str:
    """识别预设 JSON 类型。"""
    if not isinstance(data, dict):
        return TYPE_UNKNOWN
    if isinstance(data.get("prompts"), list):
        return TYPE_CHAT
    if "story_string" in data:
        return TYPE_CONTEXT
    if "content" in data and "name" in data:
        return TYPE_SYSPROMPT
    return TYPE_UNKNOWN


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_sampling(raw: dict) -> dict:
    out = {}
    for src, dst in SAMPLING_KEY_MAP.items():
        v = raw.get(src)
        if isinstance(v, bool):  # JSON 里 bool 是 int 子类，排除
            continue
        if isinstance(v, (int, float)):
            if dst == "seed" and v < 0:
                continue
            out[dst] = v
    return out


def _parse_entry(item: dict) -> PresetEntry:
    role = str(item.get("role") or "system").lower()
    if role not in VALID_ROLES:
        role = "system"
    inj_role = item.get("injection_role")
    if inj_role is not None:
        inj_role = _to_int(inj_role, 0)
        if inj_role not in (0, 1, 2):
            inj_role = None
    return PresetEntry(
        identifier=str(item.get("identifier") or item.get("name") or ""),
        name=str(item.get("name") or item.get("identifier") or ""),
        role=role,
        content=str(item.get("content") or ""),
        marker=bool(item.get("marker", False)),
        enabled=bool(item.get("enabled", True)),
        injection_position=_to_int(item.get("injection_position", 0), 0),
        injection_depth=_to_int(item.get("injection_depth", 4), 4),
        injection_order=_to_int(item.get("injection_order", 100), 100),
        injection_role=inj_role,
    )


def parse_preset(data: dict, *, fallback_name: str = "", source_path: str = "") -> TavernPreset:
    """把一份酒馆预设 JSON 归一化为 TavernPreset。"""
    ptype = detect_type(data)
    if ptype == TYPE_UNKNOWN:
        raise ValueError(
            "无法识别的预设格式：既不是 Chat 预设(prompts)、Context 预设(story_string)，"
            "也不是 Sysprompt 预设(name+content)"
        )

    name = str(data.get("name") or fallback_name or "未命名预设").strip()
    preset = TavernPreset(name=name, type=ptype, raw=data, source_path=source_path)

    if ptype == TYPE_CHAT:
        preset.entries = [_parse_entry(x) for x in data.get("prompts", []) if isinstance(x, dict)]
        preset.prompt_orders = [po for po in data.get("prompt_order", []) if isinstance(po, dict)]
        preset.sampling = _extract_sampling(data)
        preset.assistant_prefill = str(data.get("assistant_prefill") or "")
        preset.squash_system_messages = bool(data.get("squash_system_messages", False))
        preset.utility_prompts = {
            k: str(data.get(k) or "")
            for k in (
                "impersonation_prompt",
                "continue_nudge_prompt",
                "new_chat_prompt",
                "new_group_chat_prompt",
                "new_example_chat_prompt",
                "group_nudge_prompt",
                "send_if_empty",
            )
            if data.get(k)
        }
    elif ptype == TYPE_SYSPROMPT:
        # 归一化：content -> main 主提示；post_history -> 深度 0 注入（酒馆语义即贴末尾）
        preset.entries = [
            PresetEntry(
                identifier="main",
                name=name,
                role="system",
                content=str(data.get("content") or ""),
            )
        ]
        post = str(data.get("post_history") or "")
        if post.strip():
            preset.entries.append(
                PresetEntry(
                    identifier="postHistory",
                    name="Post-History",
                    role="system",
                    content=post,
                    injection_position=1,
                    injection_depth=0,
                    injection_role=0,
                )
            )
    else:  # context
        preset.story_string = str(data.get("story_string") or "")
        preset.story_string_position = _to_int(data.get("story_string_position", 0), 0)
        preset.story_string_depth = _to_int(data.get("story_string_depth", 1), 1)
        preset.story_string_role = _to_int(data.get("story_string_role", 0), 0)
        if preset.story_string_role not in (0, 1, 2):
            preset.story_string_role = 0
        preset.example_separator = str(data.get("example_separator") or "")
        preset.chat_start = str(data.get("chat_start") or "")

    return preset


class PresetLibrary:
    """扫描目录、加载并缓存全部预设。支持文件变动自动重载。"""

    def __init__(self, dirs):
        self.dirs = [Path(d) for d in dirs]
        self.presets: dict = {}
        self.errors: list = []  # [(文件名, 原因)]
        self._last_check = 0.0
        self._last_sig: Optional[list] = None
        self.reload()

    def reload(self):
        self.presets = {}
        self.errors = []
        for d in self.dirs:
            if not d.is_dir():
                continue
            for fp in sorted(d.glob("*.json")):
                try:
                    data = json.loads(fp.read_text(encoding="utf-8-sig"))
                    preset = parse_preset(data, fallback_name=fp.stem, source_path=str(fp))
                except Exception as exc:
                    self.errors.append((fp.name, str(exc)))
                    continue
                name = preset.name
                i = 2
                while name in self.presets:
                    name = f"{preset.name}~{i}"
                    i += 1
                preset.name = name
                self.presets[name] = preset
        self._last_sig = self._signature()
        self._last_check = time.monotonic()

    def _signature(self):
        sig = []
        for d in self.dirs:
            if not d.is_dir():
                continue
            for fp in d.glob("*.json"):
                try:
                    st = fp.stat()
                    sig.append((str(fp), int(st.st_mtime), st.st_size))
                except OSError:
                    pass
        return sig

    def maybe_reload(self, interval: float = 3.0) -> bool:
        """目录内容有变化时自动重载，返回是否发生了重载。"""
        now = time.monotonic()
        if now - self._last_check < interval:
            return False
        self._last_check = now
        sig = self._signature()
        if sig != self._last_sig:
            self.reload()
            return True
        return False

    def get(self, name: str) -> Optional[TavernPreset]:
        return self.presets.get(name)

    def find(self, query: str) -> Optional[TavernPreset]:
        """精确 -> 前缀 -> 包含 的模糊查找。"""
        query = (query or "").strip()
        if not query:
            return None
        if query in self.presets:
            return self.presets[query]
        for name, p in self.presets.items():
            if name.startswith(query):
                return p
        for name, p in self.presets.items():
            if query in name:
                return p
        return None

    def import_json(self, name: str, data: dict, target_dir) -> Path:
        """把一份预设 JSON 落盘到指定目录并立即注册。"""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in name if c not in '\\/:*?"<>|').strip() or "未命名预设"
        fp = target_dir / f"{safe}.json"
        i = 2
        while fp.exists():
            fp = target_dir / f"{safe}~{i}.json"
            i += 1
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.reload()
        return fp
