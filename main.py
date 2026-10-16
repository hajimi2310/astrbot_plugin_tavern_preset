"""酒馆预设适配器 —— AstrBot 插件主入口。

直接导入酒馆（SillyTavern）导出的预设 JSON，在 ``on_llm_request`` 阶段按
prompt_order + In-Chat @ Depth 规则重构请求，并完成宏变量替换与采样参数同步。

针对 AstrBot「无角色卡、无世界书」的环境差异，提供槽位化适配：
- 角色卡描述/性格/场景 -> 槽位（/预设 设定）；
- 世界书前置/后置     -> 静态槽位 wi_before / wi_after；
- {{char}} / {{user}} -> 槽位 / 配置默认值 / 发送者昵称。

v1.1.0 修复（对照 AstrBot v4.27.x 源码）：
1. ``original_system_prompt_mode``（默认 drop）：预设激活时不再把 AstrBot 的
   Agent 提示/人格/工具提示/安全提示搬进 payload——这是此前"预设看似没生效"
   的根因（框架身份与预设角色争夺身份，模型会选择框架身份）。
2. 采样参数改走 provider 的 ``custom_extra_body``：v4.27 的 ProviderRequest
   没有 temperature/top_p/max_tokens 字段，text_chat 的多余 kwargs 也会被忽略，
   原来写 req 属性的做法不会到达 API。
3. 新增 ``/预设 自检`` 与"最近一次真实请求"快照，让注入结果可验证。
"""

from __future__ import annotations

import copy
import inspect
import json
import re
import sys
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

try:  # AstrBot 以包方式加载时走相对导入
    from .core.parser import (
        TYPE_CHAT,
        TYPE_CONTEXT,
        TYPE_LABELS,
        PresetLibrary,
        detect_type,
        parse_preset,
    )
    from .core.pipeline import (
        MODE_CHAR_DESC,
        MODE_DROP,
        MODE_PREPEND,
        apply_to_request,
        assemble,
        format_preview,
    )
    from .store import SLOT_ALIASES, SLOT_LABELS, PresetStore
except ImportError:  # 直接加载 main.py 时退回路径导入
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core.parser import (
        TYPE_CHAT,
        TYPE_CONTEXT,
        TYPE_LABELS,
        PresetLibrary,
        detect_type,
        parse_preset,
    )
    from core.pipeline import (
        MODE_CHAR_DESC,
        MODE_DROP,
        MODE_PREPEND,
        apply_to_request,
        assemble,
        format_preview,
    )
    from store import SLOT_ALIASES, SLOT_LABELS, PresetStore

PLUGIN_NAME = "astrbot_plugin_tavern_preset"
PLUGIN_VERSION = "1.1.0"

# 可同步到 provider.custom_extra_body 的酒馆采样字段
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "top_a",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "max_tokens",
    "seed",
)

SYSTEM_PROMPT_MODES = {
    "drop": MODE_DROP,
    "to_char_description": MODE_CHAR_DESC,
    "prepend_to_head": MODE_PREPEND,
}

HELP_TEXT = """【酒馆预设适配器 · 命令一览】
/预设 列表 —— 查看已导入的预设（⭐会话 🌐全局）
/预设 导入 [名称] <JSON> —— 直接粘贴酒馆导出的预设 JSON 导入
/预设 重载 —— 重新扫描预设目录（也可直接把 .json 放入 presets 目录）
/预设 使用 <名称> —— 本会话启用该预设
/预设 停用 / 恢复 —— 本会话暂停跟随预设 / 恢复
/预设 全局 <名称> / 取消全局 —— 设置/清除全局默认预设
/预设 上下文 <名称> / 取消上下文 —— 为本会话叠加 Context(story_string) 预设
/预设 详情 <名称> —— 查看预设编排链与采样参数
/预设 设定 <槽位> <内容> —— 填写角色卡/世界书替代槽位
/预设 设定查看 / 设定清除 [槽位|全部]
/预设 预览 —— 查看装配结构（含最近一次真实请求快照）
/预设 自检 —— 诊断预设是否真的生效（强烈建议首次使用先跑一次）
/预设 变量 <名> <值> —— 改写预设内部变量（多数中文预设靠它做功能开关）
/预设 变量查看 [前缀] / 变量清除 [名|全部]
槽位：角色名/用户名/描述/性格/场景/人设/例句/世界前/世界后
预设文件目录：插件数据目录下 presets/（支持直接丢入酒馆导出的 .json）"""



class _Cfg:
    """配置读取封装：兼容分组/扁平两种结构。"""

    def __init__(self, raw):
        self.raw = raw or {}

    def get(self, key, default=None):
        cfg = self.raw
        if isinstance(cfg, dict):
            if key in cfg:
                return cfg[key]
            for v in cfg.values():
                if isinstance(v, dict) and key in v:
                    return v[key]
        else:
            return getattr(cfg, key, default)
        return default

    def bool(self, key, default=False):
        v = self.get(key, default)
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on", "是")
        return bool(v)

    def int(self, key, default=0):
        try:
            return int(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def str(self, key, default=""):
        v = self.get(key, default)
        return default if v is None else str(v)


@register(PLUGIN_NAME, "AstrBotDev", "酒馆（SillyTavern）预设直接导入与完美适配", PLUGIN_VERSION, "")
class TavernPresetPlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.cfg = _Cfg(config)

        # 数据目录（优先框架分配，兜底插件目录）
        try:
            self.data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        except Exception:
            self.data_dir = Path(__file__).resolve().parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.user_presets_dir = self.data_dir / "presets"
        self.user_presets_dir.mkdir(parents=True, exist_ok=True)
        self.builtin_presets_dir = Path(__file__).resolve().parent / "presets"

        # 用户目录优先，内置示例兜底
        self.library = PresetLibrary([self.user_presets_dir, self.builtin_presets_dir])
        self.store = PresetStore(self.data_dir / "tavern_preset_store.json")

        # ---- 诊断状态 ----
        self._stats = {
            "hook_calls": 0,          # on_llm_request 触发次数
            "applied": 0,             # 真正完成注入的次数
            "skipped": 0,             # 因未绑定/停用而跳过的次数
            "errors": 0,
            "last_error": "",
            "last_ts": 0.0,
            "last_umo": "",
        }
        self._last_request: dict = {}          # 最近一次真实请求的装配快照
        self._sampling_backup: dict = {}       # provider_id -> 采样写入前快照
        self.sampling_status = "未同步（尚无预设请求）"
        self._legacy_warned = False            # 旧配置项告警去重

    async def initialize(self):
        self.library.reload()
        logger.info(
            f"[tavern_preset] v{PLUGIN_VERSION} 已加载 {len(self.library.presets)} 个预设，"
            f"用户目录：{self.user_presets_dir}；原系统提示处理="
            f"{self.cfg.str('original_system_prompt_mode', 'drop')}"
        )

    async def terminate(self):
        try:
            self.store.save()
        except Exception:
            pass
        self._restore_sampling_all()

    # ================================================================
    # LLM 请求钩子：核心装配入口
    # ================================================================

    @filter.on_llm_request()
    async def on_tavern_request(self, event: AstrMessageEvent, req):
        self._stats["hook_calls"] += 1
        try:
            await self._handle_llm_request(event, req)
        except Exception as exc:
            self._stats["errors"] += 1
            self._stats["last_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning(f"[tavern_preset] 装配失败，已放行原始请求：{exc}")

    # ---- 配置读取 ----

    def _system_prompt_mode(self) -> str:
        mode = self.cfg.str("original_system_prompt_mode", "").strip()
        if mode in SYSTEM_PROMPT_MODES:
            return SYSTEM_PROMPT_MODES[mode]
        # 新键缺失（例如实例上还留着 v1.0 的配置文件）时一律按 drop 处理：
        # 绝不能因为旧配置里的 use_persona_as_char=true 就把框架身份塞回 payload，
        # 那正是 v1.0 "预设看着没生效" 的根因。需要旧行为请显式设置新配置项。
        if self.cfg.bool("use_persona_as_char", False):
            if not self._legacy_warned:
                self._legacy_warned = True
                logger.warning(
                    "[tavern_preset] 检测到旧配置项 use_persona_as_char=true，已忽略并改用默认 drop。"
                    "如需把 AstrBot 人格填进 charDescription，请把「原系统提示处理」设为 to_char_description。"
                )
        return MODE_DROP

    def _sampling_mode(self) -> str:
        mode = self.cfg.str("sampling_sync_mode", "").strip().lower()
        if mode in ("off", "custom_extra_body", "legacy_req_attrs"):
            return mode
        return "custom_extra_body" if self.cfg.bool("apply_sampling", True) else "off"

    async def _handle_llm_request(self, event: AstrMessageEvent, req):
        if not self.cfg.bool("enable", True):
            return
        self.library.maybe_reload()

        umo = self._umo(event)
        binding = self.store.resolve(umo)
        if binding.disabled:
            self._stats["skipped"] += 1
            return

        chat_preset = self.library.get(binding.preset) if binding.preset else None
        ctx_preset = (
            self.library.get(binding.context_preset) if binding.context_preset else None
        )
        if ctx_preset is not None and ctx_preset.type != TYPE_CONTEXT:
            ctx_preset = None
        if chat_preset is None and ctx_preset is None:
            self._stats["skipped"] += 1
            return

        original_system = getattr(req, "system_prompt", "") or ""
        user_input = getattr(req, "prompt", "") or ""
        user_name = binding.slots.get("user_name") or self._sender_name(event)

        assembled = assemble(
            chat_preset,
            ctx_preset,
            slots=binding.slots,
            original_system_prompt=original_system,
            user_input=user_input,
            user_name=user_name,
            char_name=self.cfg.str("default_char_name", "Assistant"),
            prefer_character_id=self.cfg.int("prefer_prompt_order_char_id", 100000),
            original_system_prompt_mode=self._system_prompt_mode(),
            variables=self.store.get_session_vars(umo),
        )
        # 预设里的 {{setvar}} 会改写变量，变化时落盘（供 /预设 变量 查看与手动开关）
        if assembled.variables != self.store.get_session_vars(umo):
            self.store.set_session_vars(umo, assembled.variables)
        notes = apply_to_request(
            assembled,
            req,
            depth0_mode=self.cfg.str("depth0_mode", "prompt_suffix"),
            write_legacy_sampling=self._sampling_mode() == "legacy_req_attrs",
        )

        # ---- 采样参数：req 上没有对应字段，改走 provider.custom_extra_body ----
        sampling_mode = self._sampling_mode()
        if sampling_mode == "off":
            assembled.sampling = {}
        elif sampling_mode == "legacy_req_attrs":
            if not self.cfg.bool("apply_sampling", True):
                assembled.sampling = {}
        elif assembled.sampling:
            provider = await self._resolve_provider(umo)
            status = self._apply_sampling_to_provider(provider, assembled.sampling)
            self.sampling_status = status
            if status:
                notes.append(status)

        # ---- 记录真实请求快照（/预设 预览 与 自检 会展示）----
        self._stats["applied"] += 1
        self._stats["last_ts"] = time.time()
        self._stats["last_umo"] = umo
        try:
            contexts = getattr(req, "contexts", None) or []
        except Exception:
            contexts = []
        self._last_request = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "umo": umo,
            "presets": list(assembled.used_names),
            "system_prompt_len": len(assembled.system_prompt or ""),
            "system_prompt_head": (assembled.system_prompt or "")[:400],
            "contexts": len(contexts),
            "prompt_len": len(getattr(req, "prompt", "") or ""),
            "original_system_len": assembled.original_system_len,
            "original_system_dropped": assembled.original_system_dropped,
            "depth_injections": [
                {"depth": d, "order": o, "role": r, "chars": len(c)}
                for d, o, r, c in assembled.depth_injections
            ],
            "sampling": dict(assembled.sampling),
            "sampling_mode": sampling_mode,
            "system_prompt_mode": self._system_prompt_mode(),
            "notes": assembled.notes + notes,
        }

        if self.cfg.bool("debug_log", False):
            logger.info(
                f"[tavern_preset] umo={umo} 预设={'+'.join(assembled.used_names)} "
                f"head={len(assembled.system_prompt)}字符 历史={len(contexts)}条 "
                f"pre={len(assembled.pre_contexts)} post={len(assembled.post_contexts)} "
                f"depth={[(d, o, r) for d, o, r, _ in assembled.depth_injections]} "
                f"原系统提示={assembled.original_system_len}字符"
                f"({'已丢弃' if assembled.original_system_dropped else '已并入'})"
            )
            for note in assembled.notes + notes:
                logger.info(f"[tavern_preset]   - {note}")

    # ================================================================
    # 采样参数同步（写 provider.custom_extra_body，解绑时还原）
    # ================================================================

    async def _resolve_provider(self, umo: str):
        ctx = self.context
        for name in ("get_using_provider_async", "get_using_provider"):
            fn = getattr(ctx, name, None)
            if not callable(fn):
                continue
            try:
                res = fn(umo=umo)
            except TypeError:
                try:
                    res = fn()
                except Exception:
                    continue
            except Exception:
                continue
            try:
                if inspect.isawaitable(res):
                    res = await res
            except Exception:
                continue
            if res is not None:
                return res
        return None

    def _apply_sampling_to_provider(self, provider, sampling: dict) -> str:
        """把酒馆采样参数写进 provider 的 custom_extra_body（未知键由 SDK 原样透传）。"""
        if provider is None:
            return "采样未同步：未取到当前 Provider（可忽略，或改用 provider 设置手动同步）"
        cfg = getattr(provider, "provider_config", None)
        if not isinstance(cfg, dict):
            return "采样未同步：该 Provider 不支持 provider_config 覆盖"

        pid = str(cfg.get("id") or f"obj:{id(provider)}")
        backup = self._sampling_backup.get(pid)
        if backup is None:
            backup = {
                "provider": provider,
                "had_key": "custom_extra_body" in cfg,
                "extra_body": copy.deepcopy(cfg.get("custom_extra_body") or {}),
                "values": {},
            }
            self._sampling_backup[pid] = backup

        current = dict(cfg.get("custom_extra_body") or {})
        for key in backup["values"]:          # 撤销上一次本插件写入的键
            current.pop(key, None)
        values = {k: v for k, v in sampling.items() if k in SAMPLING_KEYS}
        current.update(values)
        cfg["custom_extra_body"] = current
        backup["values"] = values
        if not values:
            return ""
        kv = ", ".join(f"{k}={v}" for k, v in values.items())
        return f"采样已同步到 provider({pid}).custom_extra_body：{kv}"

    def _restore_sampling(self, pid: str):
        info = self._sampling_backup.pop(pid, None)
        if not info:
            return
        provider = info.get("provider")
        cfg = getattr(provider, "provider_config", None)
        if not isinstance(cfg, dict):
            return
        current = dict(cfg.get("custom_extra_body") or {})
        for key in info.get("values", {}):
            current.pop(key, None)
        if info.get("had_key"):
            merged = dict(current)
            merged.update(info.get("extra_body") or {})
            cfg["custom_extra_body"] = merged
        elif current:
            cfg["custom_extra_body"] = current
        else:
            cfg.pop("custom_extra_body", None)

    def _restore_sampling_all(self):
        for pid in list(self._sampling_backup):
            try:
                self._restore_sampling(pid)
            except Exception:
                pass
        self.sampling_status = "未同步（已还原）"


    # ================================================================
    # 命令
    # ================================================================

    @filter.command("预设", alias={"酒馆预设", "preset"})
    async def cmd_preset(self, event: AstrMessageEvent):
        text = (event.message_str or "").replace("　", " ").strip()
        parts = text.split(None, 2)
        sub = parts[1] if len(parts) >= 2 else ""
        rest = parts[2] if len(parts) >= 3 else ""
        umo = self._umo(event)

        handlers = {
            "": self._help,
            "帮助": self._help,
            "help": self._help,
            "状态": self._status,
            "列表": self._list,
            "重载": self._reload,
            "导入": self._import,
            "使用": self._use,
            "停用": self._disable,
            "恢复": self._enable_back,
            "全局": self._use_global,
            "取消全局": self._clear_global,
            "上下文": self._use_context,
            "取消上下文": self._clear_context,
            "详情": self._detail,
            "设定": self._set_slot,
            "设定查看": self._view_slots,
            "设定清除": self._clear_slot,
            "清除设定": self._clear_slot,
            "预览": self._preview,
            "自检": self._doctor,
            "诊断": self._doctor,
            "变量": self._set_var,
            "变量查看": self._view_vars,
            "变量清除": self._clear_var,
        }
        handler = handlers.get(sub)
        if handler is None:
            yield event.plain_result(f"未知子命令「{sub}」。\n\n{HELP_TEXT}")
            return
        async for out in handler(event, umo, rest):
            yield out

    # ---------------- 基础 ----------------

    async def _help(self, event, umo, rest):
        yield event.plain_result(HELP_TEXT)

    async def _status(self, event, umo, rest):
        b = self.store.resolve(umo)
        lines = ["【酒馆预设 · 当前状态】"]
        lines.append(f"会话预设：{b.preset or '（未绑定）'}" + ("（已停用）" if b.disabled else ""))
        g = self.store.global_view()
        lines.append(f"全局预设：{g.get('preset') or '（未设置）'}")
        lines.append(f"上下文叠加：{b.context_preset or '（无）'}")
        filled = [SLOT_LABELS.get(k, k) for k, v in b.slots.items() if str(v).strip()]
        lines.append(f"已填槽位：{'、'.join(filled) if filled else '（无）'}")
        lines.append(f"预设库：{len(self.library.presets)} 个预设，{len(self.library.errors)} 个解析失败")
        yield event.plain_result("\n".join(lines))

    async def _list(self, event, umo, rest):
        self.library.maybe_reload(interval=0)
        b = self.store.resolve(umo)
        if not self.library.presets:
            text = "预设库为空。\n把酒馆导出的 .json 放入数据目录 presets/ 后用 /预设 重载，或用 /预设 导入 直接粘贴 JSON。"
            if self.library.errors:
                text += "\n\n解析失败：\n" + "\n".join(f"- {n}：{e}" for n, e in self.library.errors)
            yield event.plain_result(text)
            return
        lines = ["【酒馆预设 · 列表】"]
        for name, p in self.library.presets.items():
            marks = []
            if b.session_name == name:
                marks.append("⭐会话")
            if b.global_name == name:
                marks.append("🌐全局")
            if b.context_preset == name:
                marks.append("📎上下文")
            mark = (" " + "".join(marks)) if marks else ""
            extra = ""
            if p.type == TYPE_CHAT:
                n_inj = sum(1 for e in p.entries if e.injection_position == 1)
                extra = f"，{len(p.entries)} 项" + (f"，含 {n_inj} 条深度注入" if n_inj else "")
            lines.append(f"- {name}〔{p.type_label}{extra}〕{mark}")
        if self.library.errors:
            lines.append("解析失败：")
            lines += [f"- {n}：{e}" for n, e in self.library.errors]
        lines.append("\n使用 /预设 使用 <名称> 绑定，/预设 详情 <名称> 查看编排链")
        yield event.plain_result("\n".join(lines))

    async def _reload(self, event, umo, rest):
        self.library.reload()
        yield event.plain_result(
            f"已重新扫描。共 {len(self.library.presets)} 个预设，{len(self.library.errors)} 个解析失败。\n"
            f"目录：{self.user_presets_dir}"
        )

    # ---------------- 导入 ----------------

    async def _import(self, event, umo, rest):
        body = (rest or "").strip()
        if not body:
            yield event.plain_result(
                "用法：/预设 导入 [名称] <酒馆预设JSON>\n"
                "直接在命令后粘贴从酒馆导出的完整 JSON 内容即可；"
                "也可以把 .json 文件放入数据目录 presets/ 后用 /预设 重载。"
            )
            return
        # 剥掉可能附带的 markdown 代码围栏
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body).strip()

        name = ""
        if not body.startswith("{"):
            head, _, tail = body.partition(" ")
            if tail.strip().startswith("{"):
                name, body = head.strip(), tail.strip()
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            yield event.plain_result(f"JSON 解析失败：第 {exc.lineno} 行 {exc.msg}。\n请粘贴酒馆导出的完整预设文件内容。")
            return
        ptype = detect_type(data)
        if ptype == "unknown":
            yield event.plain_result(
                "无法识别该 JSON：支持三类酒馆预设——\n"
                "1. Chat Completion 预设（含 prompts + prompt_order）\n"
                "2. Context 预设（含 story_string）\n"
                "3. Sysprompt 预设（含 name + content）"
            )
            return
        try:
            preset = parse_preset(data, fallback_name=name or "聊天导入")
            if name:
                preset.name = name
        except Exception as exc:
            yield event.plain_result(f"预设解析失败：{exc}")
            return
        fp = self.library.import_json(preset.name, data, self.user_presets_dir)
        yield event.plain_result(
            f"导入成功：{preset.name}〔{preset.type_label}〕\n"
            f"已保存到 {fp.name}\n"
            f"发送 /预设 使用 {preset.name} 即可在本会话启用。"
        )

    # ---------------- 绑定 ----------------

    async def _use(self, event, umo, rest):
        preset = self.library.find(rest)
        if preset is None:
            yield event.plain_result(f"未找到预设「{rest}」。发送 /预设 列表 查看可用预设。")
            return
        if preset.type == TYPE_CONTEXT:
            self.store.set_session_context(umo, preset.name)
            yield event.plain_result(
                f"「{preset.name}」是 Context 预设，已作为本会话的信息装配模板叠加。\n"
                f"（story_string 将接管角色描述/性格/场景等标记位）"
            )
            return
        self.store.set_session_preset(umo, preset.name)
        tips = []
        slots = self.store.resolve(umo).slots
        idents = {e.identifier for e in preset.entries if e.marker}
        mode = self._system_prompt_mode()
        if mode == MODE_DROP:
            tips.append("已启用预设接管：AstrBot 的 Agent/人格/工具提示将不再进入请求（如需保留见配置「原系统提示处理」）")
        else:
            tips.append(f"提示：当前「原系统提示处理」= {mode}，框架提示会与角色设定同时进入请求，可能影响入戏")
        if "charDescription" in idents and not slots.get("description"):
            tips.append("角色描述槽位为空：可用 /预设 设定 描述 <角色卡内容> 填入（最推荐）")
        if {"worldInfoBefore", "worldInfoAfter"} & idents and not (
            slots.get("wi_before") or slots.get("wi_after")
        ):
            tips.append("该预设含世界书标记位：AstrBot 无世界书，可用 /预设 设定 世界前/世界后 填写静态内容")
        if not slots.get("char_name"):
            tips.append(f"{{{{char}}}} 当前为默认角色名「{self.cfg.str('default_char_name', 'Assistant')}」，可用 /预设 设定 角色名 修改")
        msg = f"本会话已启用预设：{preset.name}〔{preset.type_label}〕"
        if tips:
            msg += "\n\n适配提示：\n" + "\n".join(f"- {t}" for t in tips)
        msg += "\n发送 /预设 自检 确认是否真的生效，/预设 预览 查看实际装配结构。"
        yield event.plain_result(msg)

    async def _disable(self, event, umo, rest):
        self.store.set_session_disabled(umo, True)
        self._restore_sampling_all()
        yield event.plain_result("本会话已停用预设接管（不影响全局设置），采样参数已还原。/预设 恢复 可重新启用。")

    async def _enable_back(self, event, umo, rest):
        self.store.set_session_disabled(umo, False)
        b = self.store.resolve(umo)
        yield event.plain_result(f"本会话已恢复预设接管。当前生效：{b.preset or '（无，仅跟随全局/未绑定）'}")

    async def _use_global(self, event, umo, rest):
        preset = self.library.find(rest)
        if preset is None:
            yield event.plain_result(f"未找到预设「{rest}」。")
            return
        if preset.type == TYPE_CONTEXT:
            self.store.set_global_context(preset.name)
            yield event.plain_result(f"已将 Context 预设「{preset.name}」设为全局信息装配模板。")
            return
        self.store.set_global_preset(preset.name)
        yield event.plain_result(f"已设置全局预设：{preset.name}〔{preset.type_label}〕（各会话可用 /预设 使用 覆盖）")

    async def _clear_global(self, event, umo, rest):
        self.store.set_global_preset(None)
        self._restore_sampling_all()
        yield event.plain_result("已清除全局预设，采样参数已还原。")

    async def _use_context(self, event, umo, rest):
        preset = self.library.find(rest)
        if preset is None:
            yield event.plain_result(f"未找到预设「{rest}」。")
            return
        if preset.type != TYPE_CONTEXT:
            yield event.plain_result(f"「{preset.name}」不是 Context 预设（〔{preset.type_label}〕）。该命令用于叠加 story_string 模板预设。")
            return
        self.store.set_session_context(umo, preset.name)
        yield event.plain_result(f"本会话已叠加 Context 预设：{preset.name}")

    async def _clear_context(self, event, umo, rest):
        self.store.set_session_context(umo, None)
        yield event.plain_result("已取消本会话的 Context 预设叠加。")

    # ---------------- 详情 / 预览 ----------------

    async def _detail(self, event, umo, rest):
        preset = self.library.find(rest)
        if preset is None:
            yield event.plain_result(f"未找到预设「{rest}」。")
            return
        lines = [f"【{preset.name}】〔{preset.type_label}〕"]
        if preset.source_path:
            lines.append(f"文件：{Path(preset.source_path).name}")
        max_chars = self.cfg.int("preview_max_chars", 160)

        if preset.type in (TYPE_CHAT,):
            lines.append("编排链（按 prompt_order）：")
            i = 1
            for entry, enabled in preset.ordered_entries(
                self.cfg.int("prefer_prompt_order_char_id", 100000)
            ):
                mark = "✔" if enabled else "✘"
                if entry.injection_position == 1:
                    pos = (
                        f"@深度{entry.injection_depth}"
                        f"(order{entry.injection_order})/{entry.depth_role_str()}"
                    )
                elif entry.identifier == "chatHistory":
                    pos = "〔历史标记〕"
                elif entry.marker:
                    pos = f"〔标记:{entry.role}〕"
                else:
                    pos = f"〔{entry.role}〕"
                content = entry.content.replace("\n", "\\n")
                if len(content) > max_chars:
                    content = content[:max_chars] + "…"
                tail = f"：{content}" if content else ""
                lines.append(f"{i}. {mark} {entry.name or entry.identifier} {pos}{tail}")
                i += 1
        elif preset.type == TYPE_CONTEXT:
            ss = preset.story_string
            if len(ss) > max_chars * 2:
                ss = ss[: max_chars * 2] + "…"
            lines.append(f"story_string：{ss}")
            lines.append(
                f"装配位置：{'In-Chat @深度' + str(preset.story_string_depth) if preset.story_string_position == 1 else '头部（默认）'}"
            )
        else:
            content = preset.entries[0].content if preset.entries else ""
            if len(content) > max_chars * 2:
                content = content[: max_chars * 2] + "…"
            lines.append(f"主提示：{content}")
            if len(preset.entries) > 1:
                lines.append(f"后置注入（深度 0）：{preset.entries[1].content[:max_chars]}")

        if preset.sampling:
            lines.append("采样参数：" + ", ".join(f"{k}={v}" for k, v in preset.sampling.items()))
        if preset.assistant_prefill:
            lines.append(f"assistant 预填：{preset.assistant_prefill[:max_chars]}")
        if preset.utility_prompts:
            lines.append("含工具提示词：" + "、".join(preset.utility_prompts.keys()))
        yield event.plain_result("\n".join(lines))

    async def _preview(self, event, umo, rest):
        b = self.store.resolve(umo)
        max_chars = self.cfg.int("preview_max_chars", 160)
        blocks = []

        # 1) 最近一次真实请求（最可信的证据）
        blocks.append(self._format_last_request(max_chars))

        # 2) 按当前配置做的模拟装配
        if b.disabled:
            blocks.append("【模拟装配】本会话已停用预设接管（/预设 恢复 可重新启用）。")
        else:
            chat_preset = self.library.get(b.preset) if b.preset else None
            ctx_preset = self.library.get(b.context_preset) if b.context_preset else None
            if ctx_preset is not None and ctx_preset.type != TYPE_CONTEXT:
                ctx_preset = None
            if chat_preset is None and ctx_preset is None:
                blocks.append("【模拟装配】当前会话未绑定任何预设。/预设 使用 <名称> 或 /预设 全局 <名称>。")
            else:
                assembled = assemble(
                    chat_preset,
                    ctx_preset,
                    slots=b.slots,
                    original_system_prompt="（此处为 AstrBot Agent/人格/工具系统提示）",
                    user_input="（用户当前输入）",
                    user_name=self._sender_name(event),
                    char_name=self.cfg.str("default_char_name", "Assistant"),
                    prefer_character_id=self.cfg.int("prefer_prompt_order_char_id", 100000),
                    original_system_prompt_mode=self._system_prompt_mode(),
                )
                text = format_preview(assembled, max_chars=max_chars)
                head = (
                    f"【模拟装配】{' + '.join(assembled.used_names)}\n"
                    f"深度0落点：{self.cfg.str('depth0_mode', 'prompt_suffix')}　"
                    f"原系统提示处理：{self.cfg.str('original_system_prompt_mode', 'drop')}\n"
                )
                if assembled.notes:
                    text += "\n\n适配说明：\n" + "\n".join(f"- {n}" for n in assembled.notes)
                blocks.append(head + text)

        yield event.plain_result("\n\n".join(blocks))

    def _format_last_request(self, max_chars: int) -> str:
        snap = self._last_request
        if not snap:
            return (
                "【最近一次真实请求】尚无记录。\n"
                "说明：绑定预设后任意发一条消息给机器人，这里就会显示那次请求的"
                "真实装配结果；若始终为空，说明插件的 on_llm_request 从未被触发。"
            )
        head = (snap.get("system_prompt_head") or "").replace("\n", "\\n")
        if len(head) > max_chars * 3:
            head = head[: max_chars * 3] + "…"
        dep = "、".join(
            f"@{d['depth']}(order{d['order']},{d['role']},{d['chars']}字)"
            for d in snap.get("depth_injections", [])
        ) or "无"
        lines = [
            "【最近一次真实请求】" + snap.get("ts", ""),
            f"预设：{' + '.join(snap.get('presets', []))}　会话：{snap.get('umo', '')}",
            f"system 头部：{snap.get('system_prompt_len', 0)} 字符",
            f"历史消息：{snap.get('contexts', 0)} 条　当前输入：{snap.get('prompt_len', 0)} 字符",
            f"深度注入：{dep}",
            f"原系统提示：{snap.get('original_system_len', 0)} 字符 · "
            + ("已丢弃（预设接管）" if snap.get("original_system_dropped") else "已并入/未提供"),
            f"采样：{snap.get('sampling') or '无'}（模式 {snap.get('sampling_mode', '-')}）",
            f"头部内容预览：{head}",
        ]
        return "\n".join(lines)

    async def _doctor(self, event, umo, rest):
        b = self.store.resolve(umo)
        g = self.store.global_view()
        mode = self.cfg.str("original_system_prompt_mode", "drop")
        sampling_mode = self._sampling_mode()
        lines = [f"【酒馆预设 · 自检】插件 v{PLUGIN_VERSION}"]

        # --- 配置 ---
        lines.append(
            f"配置：enable={self.cfg.bool('enable', True)}　原系统提示处理={mode}　"
            f"深度0落点={self.cfg.str('depth0_mode', 'prompt_suffix')}　采样同步={sampling_mode}"
        )
        lines.append(f"数据目录：{self.data_dir}")
        lines.append(
            f"预设库：{len(self.library.presets)} 个（解析失败 {len(self.library.errors)}）"
        )
        if self.library.errors:
            lines += [f"　⚠️ {n}：{e}" for n, e in self.library.errors[:5]]

        # --- 绑定 ---
        lines.append(
            f"绑定：会话={b.session_name or '无'}　全局={g.get('preset') or '无'}　"
            f"生效={b.preset or '无'}　上下文叠加={b.context_preset or '无'}"
            + ("　（本会话已停用）" if b.disabled else "")
        )
        filled = [SLOT_LABELS.get(k, k) for k, v in b.slots.items() if str(v).strip()]
        lines.append(f"槽位：{'、'.join(filled) if filled else '（未填写）'}")

        # --- 钩子统计 ---
        s = self._stats
        last = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s["last_ts"])) if s["last_ts"] else "从未"
        lines.append(
            f"钩子：on_llm_request 触发 {s['hook_calls']} 次　成功注入 {s['applied']} 次　"
            f"跳过 {s['skipped']} 次　异常 {s['errors']} 次　最近注入：{last}"
        )
        if s["last_error"]:
            lines.append(f"最近异常：{s['last_error']}")

        # --- 结论 ---
        diagnose = []
        if s["hook_calls"] == 0:
            diagnose.append(
                "❌ 钩子从未触发：说明这个 AstrBot 实例没有走本插件的请求链路。"
                "请确认插件已启用、且消息确实触发了 LLM 请求。"
            )
        elif s["applied"] == 0:
            diagnose.append(
                "⚠️ 钩子触发了但没有注入：当前会话/全局都没有绑定预设，或本会话被 /预设 停用。"
                "先发 /预设 全局 <名称> 或 /预设 使用 <名称>。"
            )
        else:
            diagnose.append("✅ 预设已成功注入 system prompt（详见 /预设 预览 的'最近一次真实请求'）。")

        if self.cfg.str("original_system_prompt_mode", "drop") == "drop":
            diagnose.append(
                "ℹ️ 当前为 drop：AstrBot 的 Agent 提示/人格/工具提示/安全提示不进入 payload，"
                "酒馆预设完全接管。若模型仍顽固自称 AstrBot 助手，多半是历史消息里残留的旧身份"
                "自述——建议 /reset 开新会话，或先用 /预设 设定 描述 填写角色卡内容。"
            )
        else:
            diagnose.append(
                f"⚠️ 当前为 {mode}：AstrBot 的框架提示会与预设角色设定同时进入 payload，"
                "实测容易导致模型拒绝入戏。介意的话请把'原系统提示处理'改回 drop。"
            )

        if sampling_mode == "custom_extra_body":
            diagnose.append(f"ℹ️ 采样：{self.sampling_status}")
        elif sampling_mode == "off":
            diagnose.append("ℹ️ 采样同步已关闭：请在 AstrBot 提供商设置里手动设置 temperature 等。")
        else:
            diagnose.append(
                "⚠️ legacy_req_attrs：v4.27 的 ProviderRequest 没有 temperature/top_p 字段，"
                "该模式不会真正改变 API 请求，建议改用 custom_extra_body。"
            )

        if self._last_request.get("notes"):
            diagnose.append("最近一次装配说明：")
            diagnose += [f"　- {n}" for n in self._last_request["notes"][:8]]

        lines.append("\n诊断结论：\n" + "\n".join(diagnose))
        yield event.plain_result("\n".join(lines))


    # ---------------- 槽位 ----------------

    async def _set_var(self, event, umo, rest):
        name, _, value = (rest or "").partition(" ")
        name = name.strip()
        if not name:
            yield event.plain_result(
                "用法：/预设 变量 <名> <值>\n"
                "真实中文预设普遍用 {{setvar}}/{{getvar}} 做开关（如 字数/格式/强化 等），"
                "可用此命令手动改写；/预设 变量查看 可看当前值。"
            )
            return
        variables = self.store.get_session_vars(umo)
        variables[name] = value.strip()
        self.store.set_session_vars(umo, variables)
        shown = value.strip() or "（空）"
        yield event.plain_result(f"已设置变量 {name} = {shown}")

    async def _view_vars(self, event, umo, rest):
        variables = self.store.get_session_vars(umo)
        prefix = (rest or "").strip()
        items = {k: v for k, v in variables.items() if not prefix or k.startswith(prefix)}
        if not items:
            yield event.plain_result(
                "当前没有记录到预设变量。"
                + ("（前缀 " + prefix + "）" if prefix else "")
                + "\n提示：绑定预设并发送一条消息后，预设里的 {{setvar}} 初始化就会写入这里。"
            )
            return
        max_chars = self.cfg.int("preview_max_chars", 160)
        lines = [f"【预设变量】{len(items)} 项（会话级）"]
        for k, v in sorted(items.items()):
            v = str(v).replace("\n", "\\n")
            if len(v) > max_chars:
                v = v[:max_chars] + "…"
            lines.append(f"· {k} = {v or '（空）'}")
        lines.append("改写：/预设 变量 <名> <值>　清除：/预设 变量清除 <名|全部>")
        yield event.plain_result("\n".join(lines))

    async def _clear_var(self, event, umo, rest):
        name = (rest or "").strip()
        if name in ("", "全部", "所有"):
            self.store.clear_session_vars(umo)
            yield event.plain_result("已清空本会话全部预设变量（下次请求会由预设重新初始化）。")
            return
        if self.store.del_session_var(umo, name):
            yield event.plain_result(f"已清除变量 {name}。")
        else:
            yield event.plain_result(f"未找到变量 {name}。")

    async def _set_slot(self, event, umo, rest):
        key_alias, _, value = (rest or "").partition(" ")
        key = SLOT_ALIASES.get(key_alias.strip())
        if key is None:
            yield event.plain_result(
                "用法：/预设 设定 <槽位> <内容>\n可用槽位：" + "、".join(SLOT_LABELS.values())
            )
            return
        value = value.strip()
        if not value:
            yield event.plain_result("内容为空。如需清除请用：/预设 设定清除 <槽位>")
            return
        self.store.set_session_slot(umo, key, value)
        yield event.plain_result(f"已设置本会话槽位「{SLOT_LABELS[key]}」（{len(value)} 字符）。")

    async def _view_slots(self, event, umo, rest):
        b = self.store.resolve(umo)
        if not b.slots:
            yield event.plain_result("当前没有已填写的槽位。用 /预设 设定 <槽位> <内容> 填写。")
            return
        max_chars = self.cfg.int("preview_max_chars", 160)
        session_slots = (self.store.session_view(umo).get("slots") or {})
        lines = ["【槽位一览】（🌐=全局，⭐=本会话）"]
        for key, value in b.slots.items():
            src = "⭐" if key in session_slots else "🌐"
            v = str(value).replace("\n", "\\n")
            if len(v) > max_chars:
                v = v[:max_chars] + "…"
            lines.append(f"{src} {SLOT_LABELS.get(key, key)}：{v}")
        yield event.plain_result("\n".join(lines))

    async def _clear_slot(self, event, umo, rest):
        alias = (rest or "").strip()
        if alias in ("全部", "所有", ""):
            self.store.clear_session_slots(umo)
            yield event.plain_result("已清除本会话全部槽位（全局槽位不受影响）。")
            return
        key = SLOT_ALIASES.get(alias)
        if key is None:
            yield event.plain_result("未知槽位。可用：" + "、".join(SLOT_LABELS.values()))
            return
        if self.store.del_session_slot(umo, key):
            yield event.plain_result(f"已清除本会话槽位「{SLOT_LABELS[key]}」。")
        else:
            yield event.plain_result("本会话未设置该槽位（可能来自全局，全局槽位请在配置中清理）。")

    # ---------------- 工具 ----------------

    @staticmethod
    def _umo(event) -> str:
        umo = getattr(event, "unified_msg_origin", "") or ""
        if not umo:
            try:
                umo = str(event.get_sender_id())
            except Exception:
                umo = "__unknown__"
        return umo

    @staticmethod
    def _sender_name(event) -> str:
        try:
            name = event.get_sender_name()
            return str(name).strip() if name else "User"
        except Exception:
            return "User"
