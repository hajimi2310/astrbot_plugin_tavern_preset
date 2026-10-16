"""核心模块离线测试：不依赖 AstrBot，直接运行 `python tests/test_core.py`。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.macro_engine import MacroContext, render_template, render_text
from core.parser import (
    TYPE_CHAT,
    TYPE_CONTEXT,
    TYPE_SYSPROMPT,
    PresetLibrary,
    detect_type,
    parse_preset,
)
from core.pipeline import apply_to_request, assemble, format_preview, squash_messages
from store import PresetStore

CHAT_PRESET = {
    "temperature": 0.9,
    "top_p": 0.95,
    "openai_max_tokens": 800,
    "seed": -1,
    "squash_system_messages": False,
    "prompts": [
        {"identifier": "main", "name": "Main", "role": "system",
         "content": "Write {{char}}'s next reply for {{user}}."},
        {"identifier": "charDescription", "name": "Desc", "marker": True},
        {"identifier": "charPersonality", "name": "Pers", "marker": True},
        {"identifier": "chatHistory", "name": "History", "marker": True},
        {"identifier": "jailbreak", "name": "JB", "role": "system",
         "content": "[JB: stay in character]",
         "injection_position": 1, "injection_depth": 0, "injection_role": 0},
        {"identifier": "reminder", "name": "Rem", "role": "system",
         "content": "[Reminder: depth2]",
         "injection_position": 1, "injection_depth": 2, "injection_role": 0},
    ],
    "prompt_order": [
        {"character_id": 100000, "order": [
            {"identifier": "main", "enabled": True},
            {"identifier": "charDescription", "enabled": True},
            {"identifier": "charPersonality", "enabled": False},
            {"identifier": "chatHistory", "enabled": True},
            {"identifier": "reminder", "enabled": True},
            {"identifier": "jailbreak", "enabled": True},
        ]}
    ],
}

CONTEXT_PRESET = {
    "name": "Ctx",
    "story_string": "{{#if system}}{{system}}\n{{/if}}{{#if description}}### 描述\n{{description}}\n{{/if}}{{#if scenario}}### 场景\n{{scenario}}\n{{/if}}{{trim}}",
    "story_string_position": 0,
}

SYSPROMPT_PRESET = {
    "name": "Roleplay",
    "content": "Stay in character as {{char}}.",
    "post_history": "[Post: do not break immersion]",
}


class TestParser(unittest.TestCase):
    def test_detect_type(self):
        self.assertEqual(detect_type(CHAT_PRESET), TYPE_CHAT)
        self.assertEqual(detect_type(CONTEXT_PRESET), TYPE_CONTEXT)
        self.assertEqual(detect_type(SYSPROMPT_PRESET), TYPE_SYSPROMPT)
        self.assertEqual(detect_type({"foo": 1}), "unknown")

    def test_chat_order_and_enabled(self):
        p = parse_preset(CHAT_PRESET, fallback_name="T")
        seq = [(e.identifier, en) for e, en in p.ordered_entries(100000)]
        self.assertEqual([s[0] for s in seq],
                         ["main", "charDescription", "charPersonality", "chatHistory", "reminder", "jailbreak"])
        self.assertIn(("charPersonality", False), seq)

    def test_chat_sampling(self):
        p = parse_preset(CHAT_PRESET)
        self.assertEqual(p.sampling["temperature"], 0.9)
        self.assertEqual(p.sampling["max_tokens"], 800)
        self.assertNotIn("seed", p.sampling)  # seed=-1 被忽略

    def test_sysprompt_normalized(self):
        p = parse_preset(SYSPROMPT_PRESET)
        self.assertEqual(p.type, TYPE_SYSPROMPT)
        self.assertEqual(p.entries[0].identifier, "main")
        self.assertEqual(p.entries[1].injection_position, 1)
        self.assertEqual(p.entries[1].injection_depth, 0)

    def test_library_scan_and_import(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "a.json").write_text(json.dumps(CHAT_PRESET), encoding="utf-8")
            lib = PresetLibrary([d])
            self.assertEqual(len(lib.presets), 1)
            self.assertIsNotNone(lib.find("a"))
            lib.import_json("新预设", SYSPROMPT_PRESET, d)
            self.assertIn("Roleplay", lib.presets)  # sysprompt 用内部 name


class TestMacroEngine(unittest.TestCase):
    def test_vars_and_unknown(self):
        ctx = MacroContext(vars={"char": "爱丽丝", "user": "小明"})
        out = render_text("{{char}} 对 {{user}} 说，{{unknown}} 保留", ctx)
        self.assertEqual(out, "爱丽丝 对 小明 说，{{unknown}} 保留")

    def test_comment_macro_removed(self):
        """{{// ... }} 是注释，整块必须剥离（真实预设常用它写作者说明）。"""
        ctx = MacroContext(vars={})
        out = render_text("{{//作者：某人\n禁止二传\n}}正文开始", ctx)
        self.assertNotIn("禁止二传", out)
        self.assertIn("正文开始", out)
        # 整条都是注释时结果为空串
        self.assertEqual(render_text("{{// 只有注释 }}", ctx).strip(), "")

    def test_setvar_getvar(self):
        ctx = MacroContext(vars={})
        self.assertEqual(render_text("{{setvar::zishu::500}}字数={{getvar::zishu}}", ctx).strip(), "字数=500")
        self.assertEqual(ctx.variables["zishu"], "500")
        # 未定义变量取空串（酒馆行为），不留 {{getvar::x}} 原文
        self.assertEqual(render_text("[{{getvar::never_set}}]", ctx), "[]")

    def test_setvar_empty_and_legacy_colon(self):
        ctx = MacroContext(vars={})
        self.assertEqual(render_text("{{setvar::a::}}{{setvar::b::on}}", ctx), "")
        self.assertEqual(ctx.variables, {"a": "", "b": "on"})
        ctx2 = MacroContext(vars={})
        self.assertEqual(render_text("{{setvar:flag:yes}}{{getvar:flag}}", ctx2).strip(), "yes")

    def test_addvar_incvar_decvar(self):
        ctx = MacroContext(vars={})
        render_text("{{setvar::n::10}}", ctx)
        render_text("{{incvar::n}}", ctx)
        render_text("{{incvar::n::5}}", ctx)
        self.assertEqual(ctx.variables["n"], "16")
        render_text("{{decvar::n::6}}", ctx)
        self.assertEqual(ctx.variables["n"], "10")
        render_text("{{addvar::s::abc}}{{addvar::s::def}}", ctx)
        self.assertEqual(ctx.variables["s"], "abcdef")

    def test_random_and_roll(self):
        ctx = MacroContext(vars={})
        out = render_text("{{random:a,b,c}}", ctx)
        self.assertIn(out, ("a", "b", "c"))
        out = render_text("{{roll:1d1+5}}", ctx)
        self.assertEqual(out, "6")
        # :: 分隔的 random 也要支持
        self.assertIn(render_text("{{random::x::y::z}}", ctx), ("x", "y", "z"))

    def test_if_nested_and_else(self):
        ctx = MacroContext(vars={"description": "D", "scenario": ""})
        tpl = "{{#if description}}[{{#if scenario}}S{{else}}无场景{{/if}}:{{description}}]{{/if}}"
        self.assertEqual(render_template(tpl, ctx), "[无场景:D]")

    def test_if_uses_preset_variables(self):
        ctx = MacroContext(vars={}, variables={"kk": "1"})
        self.assertEqual(render_template("{{#if kk}}on{{else}}off{{/if}}", ctx), "on")

    def test_trim(self):
        ctx = MacroContext(vars={"a": "x"})
        self.assertEqual(render_template("  {{a}}  \n{{trim}}", ctx), "x")


class TestPipeline(unittest.TestCase):
    def _assemble(self, **kw):
        p = parse_preset(CHAT_PRESET)
        context_preset = kw.pop("context_preset", None)
        args = dict(
            slots={"description": "角色设定文本"},
            original_system_prompt="",
            user_input="你好",
            user_name="小明",
            char_name="爱丽丝",
        )
        args.update(kw)
        return assemble(p, context_preset, **args)

    def test_basic_assembly(self):
        a = self._assemble()
        # main + charDescription 进 system 头部；charPersonality 被禁用
        self.assertIn("Write 爱丽丝's next reply for 小明.", a.system_prompt)
        self.assertIn("角色设定文本", a.system_prompt)
        self.assertNotIn("Pers", a.system_prompt)
        # 深度注入：depth0 + depth2（4 元组：depth, order, role, content）
        depths = sorted(d for d, _, _, _ in a.depth_injections)
        self.assertEqual(depths, [0, 2])
        self.assertEqual(a.sampling["temperature"], 0.9)

    def test_original_system_prompt_drop_is_default(self):
        """v1.1 默认 drop：AstrBot 框架提示不进入 payload。"""
        a = self._assemble(original_system_prompt="AstrBot Agent 系统提示", slots={})
        self.assertNotIn("AstrBot Agent 系统提示", a.system_prompt)
        self.assertEqual(a.original_system_len, len("AstrBot Agent 系统提示"))
        self.assertTrue(a.original_system_dropped)
        self.assertTrue(any("已丢弃" in n for n in a.notes))

    def test_original_system_prompt_to_char_description(self):
        a = self._assemble(
            original_system_prompt="人格提示词内容",
            slots={},
            original_system_prompt_mode="to_char_description",
        )
        self.assertIn("人格提示词内容", a.system_prompt)
        self.assertFalse(a.original_system_dropped)

    def test_original_system_prompt_prepend(self):
        a = self._assemble(
            original_system_prompt="框架提示",
            original_system_prompt_mode="prepend_to_head",
        )
        self.assertTrue(a.system_prompt.startswith("框架提示"))

    def test_drop_keeps_original_when_head_empty(self):
        """预设头部为空时不能把 system prompt 清空。"""
        raw = json.loads(json.dumps(CHAT_PRESET))
        raw["prompts"] = []           # 无任何可装配条目
        raw["prompt_order"] = []
        p = parse_preset(raw)
        a = assemble(p, slots={}, original_system_prompt="保底系统提示", user_input="x")
        self.assertEqual(a.system_prompt, "保底系统提示")

    def test_injection_order_same_depth(self):
        """同深度按 injection_order 从大到小插入：order 大的更靠近生成位置。"""
        raw = json.loads(json.dumps(CHAT_PRESET))
        raw["prompts"].extend([
            {"identifier": "inj_a", "name": "A", "role": "system", "content": "[A-order0]",
             "injection_position": 1, "injection_depth": 3, "injection_order": 0},
            {"identifier": "inj_b", "name": "B", "role": "system", "content": "[B-order999]",
             "injection_position": 1, "injection_depth": 3, "injection_order": 999},
        ])
        raw["prompt_order"][0]["order"].extend([
            {"identifier": "inj_a", "enabled": True},
            {"identifier": "inj_b", "enabled": True},
        ])
        p = parse_preset(raw)
        a = assemble(p, slots={}, user_input="u")
        req = SimpleNamespace(system_prompt="", contexts=[{"role": "user", "content": f"h{i}"} for i in range(6)], prompt="u")
        apply_to_request(a, req)
        contents = [m["content"] for m in req.contexts]
        # 6 条历史 + 2 条注入：order999 应在 order0 之后（更靠近末尾）
        self.assertIn("[A-order0]", contents)
        self.assertIn("[B-order999]", contents)
        self.assertLess(contents.index("[A-order0]"), contents.index("[B-order999]"))

    def test_entry_level_enabled_when_order_missing(self):
        """prompt_order 未收录的条目沿用其自身 enabled 字段。"""
        raw = json.loads(json.dumps(CHAT_PRESET))
        raw["prompts"].append({
            "identifier": "extra_off", "name": "Extra", "role": "system",
            "content": "[不应出现]", "enabled": False,
        })
        p = parse_preset(raw)
        a = assemble(p, slots={}, user_input="u")
        self.assertNotIn("[不应出现]", a.system_prompt)

    def test_variables_flow_through_assemble(self):
        """预设里的 setvar 初始化应被采集，并供后续条目 getvar 使用。"""
        raw = json.loads(json.dumps(CHAT_PRESET))
        raw["prompts"].insert(0, {
            "identifier": "initvars", "name": "初始化变量", "role": "system",
            "content": "{{setvar::zishu::800}}", "enabled": True,
        })
        raw["prompts"].insert(1, {
            "identifier": "uservar", "name": "用变量", "role": "system",
            "content": "字数限制：{{getvar::zishu}}", "enabled": True,
        })
        raw["prompt_order"][0]["order"].insert(0, {"identifier": "initvars", "enabled": True})
        raw["prompt_order"][0]["order"].insert(1, {"identifier": "uservar", "enabled": True})
        p = parse_preset(raw)
        a = assemble(p, slots={}, user_input="u", variables={"zishu": "300"})
        self.assertIn("字数限制：800", a.system_prompt)
        self.assertEqual(a.variables["zishu"], "800")
        # 初始化条目本身不产出正文
        self.assertNotIn("setvar", a.system_prompt)

    def test_comment_entry_produces_nothing(self):
        """整条都是 {{// 注释 }} 的条目在酒馆里不发送任何内容。"""
        raw = json.loads(json.dumps(CHAT_PRESET))
        raw["prompts"].insert(0, {
            "identifier": "notes", "name": "说明", "role": "system",
            "content": "{{//作者：某人\n禁止二传，贩子死妈}}", "enabled": True,
        })
        raw["prompt_order"][0]["order"].insert(0, {"identifier": "notes", "enabled": True})
        p = parse_preset(raw)
        a = assemble(p, slots={}, user_input="u")
        self.assertNotIn("贩子死妈", a.system_prompt)

    def test_empty_markers_skipped(self):
        a = self._assemble(slots={})
        self.assertNotIn("charPersonality", a.system_prompt)
        self.assertTrue(any("charDescription" in n for n in a.notes))

    def test_context_preset_takes_over(self):
        cp = parse_preset(CONTEXT_PRESET)
        a = self._assemble(
            context_preset=cp,
            slots={"description": "D文本", "scenario": "S文本"},
        )
        self.assertIn("### 描述\nD文本", a.system_prompt)
        self.assertIn("### 场景\nS文本", a.system_prompt)
        self.assertTrue(any("story_string" in n for n in a.notes))

    def test_apply_to_request_depth(self):
        a = self._assemble()
        req = SimpleNamespace(
            system_prompt="原系统",
            contexts=[{"role": "user", "content": "h1"},
                      {"role": "assistant", "content": "h2"},
                      {"role": "user", "content": "h3"},
                      {"role": "assistant", "content": "h4"}],
            prompt="当前输入",
        )
        apply_to_request(a, req, depth0_mode="prompt_suffix")
        # depth2 插到 contexts 倒数第 1 位之前：len=4 -> idx=3
        self.assertEqual(len(req.contexts), 5)
        self.assertEqual(req.contexts[3]["content"], "[Reminder: depth2]")
        self.assertEqual(req.contexts[3]["role"], "system")
        # depth0 -> prompt 后缀
        self.assertTrue(req.prompt.endswith("[JB: stay in character]"))
        self.assertTrue(req.prompt.startswith("当前输入"))
        # system_prompt 被替换为头部
        self.assertIn("Write 爱丽丝", req.system_prompt)
        # 默认不往 req 写采样属性（v4.27 无对应字段）
        self.assertFalse(hasattr(req, "temperature"))
        self.assertEqual(a.sampling["max_tokens"], 800)

    def test_legacy_sampling_attrs(self):
        a = self._assemble()
        req = SimpleNamespace(system_prompt="", contexts=[], prompt="x")
        apply_to_request(a, req, write_legacy_sampling=True)
        self.assertEqual(req.temperature, 0.9)
        self.assertEqual(req.max_tokens, 800)

    def test_apply_depth0_context_tail(self):
        a = self._assemble()
        req = SimpleNamespace(system_prompt="", contexts=[], prompt="hi")
        apply_to_request(a, req, depth0_mode="context_tail")
        self.assertEqual(req.contexts[-1]["content"], "[JB: stay in character]")
        self.assertEqual(req.prompt, "hi")

    def test_history_disabled(self):
        raw = json.loads(json.dumps(CHAT_PRESET))
        for item in raw["prompt_order"][0]["order"]:
            if item["identifier"] == "chatHistory":
                item["enabled"] = False
        p = parse_preset(raw)
        a = assemble(p, slots={}, user_input="x")
        req = SimpleNamespace(system_prompt="", contexts=[{"role": "user", "content": "h"}], prompt="x")
        # chatHistory 被禁用 -> 标记为未看到历史标记? 禁用条目被跳过,
        # 此时 pipeline 认为标记缺失但历史仍保留;此处验证请求未被破坏
        apply_to_request(a, req)
        self.assertIsInstance(req.contexts, list)

    def test_squash(self):
        msgs = [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        merged = squash_messages(msgs)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["content"], "a\n\nb")

    def test_preview_format(self):
        a = self._assemble()
        text = format_preview(a, history_count=5, max_chars=50)
        self.assertIn("system 头部", text)
        self.assertIn("depth@2", text)
        self.assertIn("depth@0", text)
        self.assertIn("temperature=0.9", text)


class TestStore(unittest.TestCase):
    def test_resolve_merge(self):
        with tempfile.TemporaryDirectory() as td:
            st = PresetStore(Path(td) / "s.json")
            st.set_global_preset("G")
            st.set_global_slot("char_name", "全局角色")
            st.set_session_preset("umo1", "S")
            st.set_session_slot("umo1", "scenario", "会话场景")
            b = st.resolve("umo1")
            self.assertEqual(b.preset, "S")
            self.assertEqual(b.slots["char_name"], "全局角色")
            self.assertEqual(b.slots["scenario"], "会话场景")
            b2 = st.resolve("umo2")
            self.assertEqual(b2.preset, "G")
            st.set_session_disabled("umo2", True)
            self.assertTrue(st.resolve("umo2").disabled)
            # 持久化
            st2 = PresetStore(Path(td) / "s.json")
            self.assertEqual(st2.resolve("umo1").preset, "S")


if __name__ == "__main__":
    unittest.main(verbosity=2)
