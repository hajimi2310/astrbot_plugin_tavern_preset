"""上下文重构管线：按酒馆 prompt_order + In-Chat @ Depth 规则重排消息。

装配流程（对应酒馆 Chat Completion 流水线）：

1. 依 prompt_order 取得启用条目序列；
2. 以 ``chatHistory`` 标记为界，前段进「头部」，后段进「尾部」；
3. ``injection_position == 1`` 的条目进入深度注入队列；
4. 头部中 role=system 的内容合并为 system_prompt，其余角色放在历史之前；
5. 标记位（marker）内容由槽位解析：charDescription/charPersonality/scenario/
   personaDescription/dialogueExamples/worldInfoBefore/worldInfoAfter；
6. Context 预设的 story_string 作为「信息装配模板」接管角色信息类标记位。

AstrBot 适配说明（无角色卡 / 无世界书）：
- 角色卡各字段 -> 会话/全局槽位；
- 世界书前后置 -> 静态槽位 wi_before / wi_after；
- 深度 0 注入 -> 由 depth0_mode 决定落点（默认追加到当前用户输入末尾，
  这是 AstrBot 管线中最接近酒馆「最后一条消息」语义的位置）；
- AstrBot 侧原有的 system prompt（Agent 提示 / 人格 / 工具提示 / 安全提示）
  由 ``original_system_prompt_mode`` 决定去留，默认 drop（见该参数说明）。

纯标准库实现，可脱离 AstrBot 单独测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .macro_engine import MacroContext, render_template, render_text, time_vars
from .parser import TavernPreset, TYPE_CHAT, TYPE_CONTEXT, TYPE_SYSPROMPT

# 角色信息类标记位（可被 Context 预设 story_string 整体接管）
CHAR_INFO_MARKERS = ("charDescription", "charPersonality", "scenario", "personaDescription")

VALID_ROLES = ("system", "user", "assistant")

# 原 system prompt（AstrBot 的 Agent 提示/人格/工具/安全提示）处理方式
MODE_DROP = "drop"                          # 丢弃，只由 {{original}} 宏引用（默认）
MODE_CHAR_DESC = "to_char_description"      # 塞进 charDescription 标记位
MODE_PREPEND = "prepend_to_head"            # 前置到预设头部最前面


@dataclass
class Assembled:
    """装配结果（与具体框架无关）。"""

    system_prompt: str = ""
    pre_contexts: list = field(default_factory=list)    # 历史之前的非 system 头部消息
    post_contexts: list = field(default_factory=list)   # 历史之后的尾部消息
    depth_injections: list = field(default_factory=list)  # [(depth, order, role, content)]
    keep_history: bool = True
    squash: bool = False
    assistant_prefill: str = ""
    sampling: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)           # 装配说明（预览/调试用）
    used_names: list = field(default_factory=list)      # 实际参与装配的预设名
    variables: dict = field(default_factory=dict)       # 装配后被 setvar 更新过的酒馆变量
    original_system_len: int = 0                        # 原 system prompt 长度
    original_system_dropped: bool = False               # 原 system prompt 是否被丢弃


def _marker_content(
    identifier: str,
    slots: dict,
    char_desc_fallback: str,
    notes: list,
) -> str:
    """解析标记位内容。返回空串表示该条目应被跳过。"""
    if identifier == "worldInfoBefore":
        return slots.get("wi_before", "")
    if identifier == "worldInfoAfter":
        return slots.get("wi_after", "")
    if identifier == "charDescription":
        desc = slots.get("description", "")
        if not desc and char_desc_fallback.strip():
            notes.append("charDescription：由 AstrBot 原系统提示填充")
            return char_desc_fallback
        return desc
    if identifier == "charPersonality":
        return slots.get("personality", "")
    if identifier == "scenario":
        return slots.get("scenario", "")
    if identifier == "personaDescription":
        return slots.get("persona", "")
    if identifier == "dialogueExamples":
        return slots.get("mes_examples", "")
    return ""


def _build_macro_vars(
    slots: dict,
    original_system: str,
    user_input: str,
    user_name: str,
    char_name: str,
) -> dict:
    mes = slots.get("mes_examples", "")
    return {
        "char": char_name,
        "user": user_name,
        "charIfNotGroup": char_name,
        "description": slots.get("description", ""),
        "personality": slots.get("personality", ""),
        "scenario": slots.get("scenario", ""),
        "persona": slots.get("persona", ""),
        "mesExamples": mes,
        "mesExamplesRaw": mes,
        "wiBefore": slots.get("wi_before", ""),
        "wiAfter": slots.get("wi_after", ""),
        "anchorBefore": slots.get("wi_before", ""),
        "anchorAfter": slots.get("wi_after", ""),
        "system": original_system,
        "original": original_system,
        "input": user_input,
        "group": "",
        **time_vars(),
    }


def assemble(
    chat_preset: Optional[TavernPreset],
    context_preset: Optional[TavernPreset] = None,
    *,
    slots: dict,
    original_system_prompt: str = "",
    user_input: str = "",
    user_name: str = "User",
    char_name: str = "Assistant",
    prefer_character_id: int = 100000,
    original_system_prompt_mode: str = MODE_DROP,
    use_persona_as_char: Optional[bool] = None,
    variables: Optional[dict] = None,
) -> Assembled:
    """装配一份请求。chat_preset 与 context_preset 至少给一个。

    ``original_system_prompt_mode`` 决定 AstrBot 原有 system prompt 的去留：

    - ``drop``（默认）：不进入 payload，只由 ``{{original}}`` 宏引用。这是
      酒馆预设"接管大脑"的正确语义——否则 AstrBot 的 Agent 提示/人格会与
      预设角色设定争夺身份，实测会让模型拒绝入戏。
    - ``to_char_description``：当作角色卡描述填入 charDescription 标记位。
    - ``prepend_to_head``：前置到预设头部最前面（保留框架身份，仅作兼容）。

    ``variables`` 为按会话持久化的酒馆变量表（setvar/getvar 读写），
    装配结束后可从 ``Assembled.variables`` 取回写盘。
    """
    result = Assembled()
    original_system = original_system_prompt or ""
    slots = dict(slots or {})
    # 兼容旧配置：use_persona_as_char=True 等价于 to_char_description
    if use_persona_as_char is True and original_system_prompt_mode == MODE_DROP:
        original_system_prompt_mode = MODE_CHAR_DESC
    mode = original_system_prompt_mode if original_system_prompt_mode in (
        MODE_DROP,
        MODE_CHAR_DESC,
        MODE_PREPEND,
    ) else MODE_DROP
    result.original_system_len = len(original_system)
    char_desc_fallback = original_system if mode == MODE_CHAR_DESC else ""

    # 主预设类型兜底：只允许 chat/sysprompt 走主装配
    if chat_preset is not None and chat_preset.type not in (TYPE_CHAT, TYPE_SYSPROMPT):
        if chat_preset.type == TYPE_CONTEXT and context_preset is None:
            chat_preset, context_preset = None, chat_preset
        else:
            chat_preset = None
    if context_preset is not None and context_preset.type != TYPE_CONTEXT:
        context_preset = None

    char_name = (slots.get("char_name") or char_name or "Assistant").strip() or "Assistant"
    user_name = (slots.get("user_name") or user_name or "User").strip() or "User"
    macro_vars = _build_macro_vars(slots, original_system, user_input, user_name, char_name)
    ctx = MacroContext(vars=macro_vars, variables=dict(variables or {}))

    # ---- Context 预设：渲染 story_string 作为「信息装配模板」----
    story_block = ""
    if context_preset is not None:
        story_block = render_template(context_preset.story_string, ctx).strip()
        result.used_names.append(context_preset.name)
        if story_block:
            if context_preset.story_string_position == 1:
                # In-Chat @ Depth 装配
                role = VALID_ROLES[context_preset.story_string_role]
                result.depth_injections.append(
                    (context_preset.story_string_depth, 100, role, story_block)
                )
                result.notes.append(
                    f"story_string 按 depth@{context_preset.story_string_depth}/{role} 注入"
                )
            else:
                result.notes.append("story_string 已接管角色信息类标记位")
        else:
            result.notes.append("story_string 渲染结果为空（相关槽位均未填写）")

    # ---- 主预设：按 prompt_order 装配 ----
    head_system_parts: list = []
    story_inserted = context_preset is None or context_preset.story_string_position == 1

    if mode == MODE_PREPEND and original_system.strip():
        head_system_parts.append(original_system.strip())
        result.notes.append(
            f"原系统提示（{len(original_system)} 字符，AstrBot Agent/人格/工具/安全提示）已前置到头部"
        )

    if chat_preset is not None:
        result.used_names.insert(0, chat_preset.name)
        result.sampling = dict(chat_preset.sampling)
        result.assistant_prefill = chat_preset.assistant_prefill.strip()
        result.squash = chat_preset.squash_system_messages

        in_tail = False
        history_marker_seen = False
        for entry, enabled in chat_preset.ordered_entries(prefer_character_id):
            if not enabled:
                continue
            ident = entry.identifier

            # 深度注入条目：收集后统一处理
            if entry.injection_position == 1:
                content = render_text(entry.content, ctx).strip()
                if content:
                    result.depth_injections.append(
                        (
                            max(0, entry.injection_depth),
                            entry.injection_order,
                            entry.depth_role_str(),
                            content,
                        )
                    )
                continue

            # 历史标记位
            if ident == "chatHistory":
                in_tail = True
                history_marker_seen = True
                continue

            # 其余标记位 / 普通条目
            if entry.marker:
                if ident in CHAR_INFO_MARKERS and story_block and not story_inserted:
                    # story_string 接管：在第一个角色信息标记位输出整块
                    _emit(result, head_system_parts, "system", story_block, in_tail)
                    story_inserted = True
                    result.notes.append(f"角色信息已由 story_string 装配（位于 {ident} 处）")
                    continue
                if ident in CHAR_INFO_MARKERS and story_block:
                    # story 已注入，跳过剩余角色信息标记位
                    continue
                content = _marker_content(ident, slots, char_desc_fallback, result.notes)
                if not content.strip():
                    result.notes.append(f"标记位 {ident} 内容为空，已跳过")
                    continue
            else:
                content = render_text(entry.content, ctx)
                if not content.strip():
                    continue

            _emit(result, head_system_parts, entry.role, content.strip(), in_tail)

        if not history_marker_seen:
            result.notes.append("预设中未找到 chatHistory 标记，历史记录默认保留在原位置")
        # story_block 启用但预设没有任何角色信息标记位：放到头部末尾
        if story_block and not story_inserted:
            _emit(result, head_system_parts, "system", story_block, False)
            result.notes.append("story_string 追加到头部末尾（预设无角色信息标记位）")
    else:
        # 仅 Context 预设：story_block 作为整个头部
        if story_block and context_preset.story_string_position != 1:
            head_system_parts.append(story_block)

    result.system_prompt = "\n\n".join(p for p in head_system_parts if p.strip())
    result.variables = ctx.variables

    # 原 system prompt 去向说明（drop 模式下若还填了 charDescription 会另行说明）
    if original_system.strip():
        if mode == MODE_DROP:
            if result.system_prompt.strip():
                result.original_system_dropped = True
                result.notes.append(
                    f"原系统提示（{len(original_system)} 字符）已丢弃：预设完全接管 system prompt，"
                    "如需保留请把「原系统提示处理」改为 prepend_to_head 或 to_char_description"
                )
            else:
                result.system_prompt = original_system.strip()
                result.notes.append("预设头部为空，已保留原系统提示（避免空 system prompt）")
        elif mode == MODE_CHAR_DESC and not any(
            e.identifier == "charDescription" for e, _ in (chat_preset.ordered_entries(prefer_character_id) if chat_preset else [])
        ):
            head_system_parts.append(original_system.strip())
            result.notes.append("预设无 charDescription 标记位，原系统提示追加到头部")
            result.system_prompt = "\n\n".join(
                p for p in [result.system_prompt, original_system.strip()] if p.strip()
            )

    return result


def _emit(result: Assembled, head_system_parts: list, role: str, content: str, in_tail: bool):
    if in_tail:
        result.post_contexts.append({"role": role, "content": content})
    elif role == "system":
        head_system_parts.append(content)
    else:
        result.pre_contexts.append({"role": role, "content": content})


def squash_messages(messages: list) -> list:
    """合并连续同角色消息（对应酒馆 squash_system_messages）。"""
    merged: list = []
    for msg in messages:
        role = msg.get("role", "system")
        content = msg.get("content", "")
        if (
            merged
            and merged[-1].get("role") == role
            and isinstance(merged[-1].get("content"), str)
            and isinstance(content, str)
        ):
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged


def apply_to_request(
    assembled: Assembled,
    req,
    *,
    depth0_mode: str = "prompt_suffix",
    write_legacy_sampling: bool = False,
) -> list:
    """把装配结果写入 AstrBot ProviderRequest。返回附加说明列表。

    只依赖 req 的 ``system_prompt`` / ``contexts`` / ``prompt`` 三个通用属性，
    全部做防御式读写。

    采样参数默认**不**写 req：AstrBot v4.27 的 ProviderRequest 没有
    temperature/top_p/max_tokens 字段，写属性不会到达 API（真正的落地方式是
    provider.custom_extra_body，由 main.py 负责）。``write_legacy_sampling=True``
    时保留旧行为，仅为兼容某些改造过的 Provider。
    """
    notes: list = []
    try:
        orig = list(getattr(req, "contexts", None) or [])
    except Exception:
        orig = []
    if not assembled.keep_history:
        orig = []
        notes.append("chatHistory 在预设中被禁用：已清空历史")

    new_ctx = list(assembled.pre_contexts) + orig

    # 深度 >= 1：depth 以「含当前用户输入的完整消息列表」倒数计数，
    # 因此在 contexts 中的下标 = len - (depth - 1)。
    # 同深度按 injection_order 升序依次插入：每插入一条长度 +1，后来的（order 更大）
    # 会落在更深处——与酒馆「order 越大越靠近生成位置」的语义一致。
    deep = [d for d in assembled.depth_injections if d[0] >= 1]
    deep.sort(key=lambda x: (-x[0], x[1]))
    for depth, _order, role, content in deep:
        idx = len(new_ctx) - (depth - 1)
        idx = max(0, min(len(new_ctx), idx))
        new_ctx.insert(idx, {"role": role, "content": content})
        notes.append(f"深度注入 @{depth}（{role}）：{content[:40]}…" if len(content) > 40 else f"深度注入 @{depth}（{role}）：{content}")

    new_ctx.extend(assembled.post_contexts)

    if assembled.assistant_prefill:
        new_ctx.append({"role": "assistant", "content": assembled.assistant_prefill})
        notes.append("assistant_prefill 已附加到历史末尾")

    if assembled.squash:
        new_ctx = squash_messages(new_ctx)

    # ---- 深度 0：AstrBot 管线适配 ----
    prompt = getattr(req, "prompt", "") or ""
    d0 = [d for d in assembled.depth_injections if d[0] == 0]
    if d0:
        if depth0_mode == "context_tail":
            for _, _order, role, content in d0:
                new_ctx.append({"role": role, "content": content})
            notes.append("深度 0 注入 -> 历史末尾（context_tail）")
        elif depth0_mode == "system_prompt":
            extra = "\n\n".join(c for _, _, _, c in d0)
            assembled.system_prompt = (assembled.system_prompt + "\n\n" + extra).strip()
            notes.append("深度 0 注入 -> 并入系统提示（system_prompt）")
        else:  # prompt_suffix（默认，最接近酒馆语义）
            suffix_parts = []
            for _, _order, role, content in d0:
                if role == "assistant":
                    new_ctx.append({"role": "assistant", "content": content})
                    notes.append("深度 0 assistant 注入 -> 历史末尾")
                else:
                    suffix_parts.append(content)
            if suffix_parts:
                prompt = (prompt + "\n\n" + "\n\n".join(suffix_parts)).strip()
                notes.append("深度 0 注入 -> 追加到当前输入末尾（prompt_suffix）")

    # ---- 写回 req ----
    try:
        req.contexts = new_ctx
    except Exception as exc:
        notes.append(f"写入 contexts 失败：{exc}")
    if assembled.system_prompt.strip():
        try:
            req.system_prompt = assembled.system_prompt
        except Exception as exc:
            notes.append(f"写入 system_prompt 失败：{exc}")
    try:
        req.prompt = prompt
    except Exception as exc:
        notes.append(f"写入 prompt 失败：{exc}")

    # ---- 采样参数：legacy 模式下才写 req 属性（v4.27 通常无效）----
    if write_legacy_sampling:
        for key, value in assembled.sampling.items():
            try:
                setattr(req, key, value)
            except Exception:
                pass
        extra = getattr(req, "extra", None)
        if isinstance(extra, dict):
            try:
                extra.update(assembled.sampling)
            except Exception:
                pass

    return notes


def format_preview(assembled: Assembled, history_count: int = 0, max_chars: int = 160) -> str:
    """把装配结果格式化为人类可读的预览文本。"""

    def cut(s: str) -> str:
        s = s.replace("\n", "\\n")
        return s if len(s) <= max_chars else s[:max_chars] + "…"

    lines = []
    idx = 1
    if assembled.system_prompt:
        lines.append(f"{idx}. [system 头部] {cut(assembled.system_prompt)}")
        idx += 1
    for msg in assembled.pre_contexts:
        lines.append(f"{idx}. [{msg['role']} 头部] {cut(msg['content'])}")
        idx += 1
    if assembled.keep_history:
        lines.append(f"—— 历史对话（原样保留{('，当前 ' + str(history_count) + ' 条') if history_count else ''}）——")
    else:
        lines.append("—— 历史对话（预设已禁用，不发送）——")
    for depth, _order, role, content in sorted(assembled.depth_injections, key=lambda x: (x[0], x[1])):
        if depth >= 1:
            lines.append(f"{idx}. [depth@{depth} {role}] {cut(content)}")
            idx += 1
    for msg in assembled.post_contexts:
        lines.append(f"{idx}. [{msg['role']} 尾部] {cut(msg['content'])}")
        idx += 1
    if assembled.assistant_prefill:
        lines.append(f"{idx}. [assistant 预填] {cut(assembled.assistant_prefill)}")
        idx += 1
    d0 = [d for d in assembled.depth_injections if d[0] == 0]
    for _, _order, role, content in d0:
        lines.append(f"{idx}. [depth@0 {role}] {cut(content)}（落点见配置 depth0_mode）")
        idx += 1
    if assembled.original_system_len:
        state = "已丢弃（预设接管）" if assembled.original_system_dropped else "已并入"
        lines.append(f"原 system prompt：{assembled.original_system_len} 字符 · {state}")
    if assembled.sampling:
        kv = ", ".join(f"{k}={v}" for k, v in assembled.sampling.items())
        lines.append(f"采样参数：{kv}")
    if assembled.squash:
        lines.append("已启用连续同角色消息合并（squash）")
    return "\n".join(lines) if lines else "（装配结果为空）"
