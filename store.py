"""插件持久化存储：会话绑定、全局绑定、槽位内容。

数据文件：``<数据目录>/tavern_preset_store.json``，结构::

    {
      "version": 1,
      "global":   {"preset": null, "context_preset": null, "slots": {}},
      "sessions": {"<umo>": {"preset": null, "context_preset": null,
                              "slots": {}, "disabled": false}}
    }

槽位（slots）用于替代酒馆中的角色卡 / 世界书 / 用户人设：
char_name, user_name, description, personality, scenario,
persona, mes_examples, wi_before, wi_after。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_VERSION = 1

# 槽位中文别名 -> 内部 key
SLOT_ALIASES = {
    "角色名": "char_name",
    "角色": "char_name",
    "用户名": "user_name",
    "用户": "user_name",
    "描述": "description",
    "角色描述": "description",
    "性格": "personality",
    "场景": "scenario",
    "人设": "persona",
    "用户人设": "persona",
    "例句": "mes_examples",
    "对话例句": "mes_examples",
    "世界前": "wi_before",
    "世界书前": "wi_before",
    "世界后": "wi_after",
    "世界书后": "wi_after",
}

SLOT_LABELS = {
    "char_name": "角色名 {{char}}",
    "user_name": "用户名 {{user}}",
    "description": "角色描述 charDescription",
    "personality": "角色性格 charPersonality",
    "scenario": "场景 scenario",
    "persona": "用户人设 personaDescription",
    "mes_examples": "对话例句 dialogueExamples",
    "wi_before": "世界书·前置 worldInfoBefore",
    "wi_after": "世界书·后置 worldInfoAfter",
}


@dataclass
class ResolvedBinding:
    """一次请求的解析结果：会话配置覆盖全局配置。"""

    preset: Optional[str] = None
    context_preset: Optional[str] = None
    slots: dict = field(default_factory=dict)
    disabled: bool = False
    session_name: Optional[str] = None   # 会话级绑定的预设名（用于展示）
    global_name: Optional[str] = None


class PresetStore:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {"version": DATA_VERSION, "global": {}, "sessions": {}}
        self.load()

    # ---------------- 持久化 ----------------

    def load(self):
        try:
            if self.path.is_file():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.data["global"] = dict(raw.get("global") or {})
                    self.data["sessions"] = dict(raw.get("sessions") or {})
        except Exception:
            # 数据损坏不致命，退回空库
            pass

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self.path)
        except Exception:
            pass

    # ---------------- 读取 ----------------

    def _session(self, umo: str) -> dict:
        return self.data["sessions"].get(umo) or {}

    def resolve(self, umo: str) -> ResolvedBinding:
        g = self.data["global"]
        s = self._session(umo)
        merged_slots = dict(g.get("slots") or {})
        merged_slots.update({k: v for k, v in (s.get("slots") or {}).items() if v != ""})
        return ResolvedBinding(
            preset=s.get("preset") or g.get("preset"),
            context_preset=s.get("context_preset") or g.get("context_preset"),
            slots=merged_slots,
            disabled=bool(s.get("disabled", False)),
            session_name=s.get("preset"),
            global_name=g.get("preset"),
        )

    def session_view(self, umo: str) -> dict:
        """返回会话原始配置（展示用）。"""
        return dict(self._session(umo))

    def global_view(self) -> dict:
        return dict(self.data["global"])

    # ---------------- 酒馆变量（setvar/getvar 的持久化载体） ----------------

    def get_session_vars(self, umo: str) -> dict:
        return dict(self._session(umo).get("vars") or {})

    def set_session_vars(self, umo: str, variables: dict):
        self._session_mut(umo)["vars"] = dict(variables or {})
        self.save()

    def del_session_var(self, umo: str, name: str) -> bool:
        variables = self._session_mut(umo).setdefault("vars", {})
        existed = name in variables
        variables.pop(name, None)
        self.save()
        return existed

    def clear_session_vars(self, umo: str):
        self._session_mut(umo)["vars"] = {}
        self.save()

    # ---------------- 写入 ----------------

    def _session_mut(self, umo: str) -> dict:
        return self.data["sessions"].setdefault(
            umo,
            {"preset": None, "context_preset": None, "slots": {}, "disabled": False, "vars": {}},
        )

    def set_session_preset(self, umo: str, name: Optional[str]):
        s = self._session_mut(umo)
        s["preset"] = name
        if name:
            s["disabled"] = False
        self.save()

    def set_session_context(self, umo: str, name: Optional[str]):
        self._session_mut(umo)["context_preset"] = name
        self.save()

    def set_session_disabled(self, umo: str, disabled: bool):
        self._session_mut(umo)["disabled"] = disabled
        self.save()

    def set_session_slot(self, umo: str, key: str, value: str):
        self._session_mut(umo).setdefault("slots", {})[key] = value
        self.save()

    def del_session_slot(self, umo: str, key: str) -> bool:
        slots = self._session_mut(umo).setdefault("slots", {})
        existed = key in slots
        slots.pop(key, None)
        self.save()
        return existed

    def clear_session_slots(self, umo: str):
        self._session_mut(umo)["slots"] = {}
        self.save()

    def clear_session(self, umo: str):
        self.data["sessions"].pop(umo, None)
        self.save()

    def set_global_preset(self, name: Optional[str]):
        self.data["global"]["preset"] = name
        self.save()

    def set_global_context(self, name: Optional[str]):
        self.data["global"]["context_preset"] = name
        self.save()

    def set_global_slot(self, key: str, value: str):
        self.data["global"].setdefault("slots", {})[key] = value
        self.save()

    def clear_global_slots(self):
        self.data["global"]["slots"] = {}
        self.save()
