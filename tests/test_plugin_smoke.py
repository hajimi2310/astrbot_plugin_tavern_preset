"""主入口冒烟测试：stub 掉 astrbot API，真实执行 main.py 的钩子与命令。

运行：python tests/test_plugin_smoke.py
覆盖用户实际报障路径：绑定预设 -> on_llm_request -> 检查 req 是否被真正改写。
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import shutil
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
# 每进程独立的数据根目录（tempfile.mkdtemp 已保证唯一）
_SMOKE_ROOT = Path(tempfile.mkdtemp(prefix="tavern_preset_smoke_"))


def install_astrbot_stubs():
    """把 astrbot.api.* 换成最小可用替身，便于离线驱动插件。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot_stub")
    api.AstrBotConfig = dict

    event_mod = types.ModuleType("astrbot.api.event")

    class _Filter:
        """把 @filter.xxx(...) 一律变成透传装饰器，仅记录注册信息。"""

        registered: list = []

        def __getattr__(self, name):
            def deco(*args, **kwargs):
                _Filter.registered.append((name, args, kwargs))
                if args and callable(args[0]) and not isinstance(args[0], str):
                    return args[0]

                def wrap(fn):
                    return fn

                return wrap

            return deco

    class AstrMessageEvent:  # noqa: D401 - 占位类型
        pass

    event_mod.filter = _Filter()
    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.MessageChain = object

    star_mod = types.ModuleType("astrbot.api.star")

    class Star:
        def __init__(self, context=None):
            self.context = context

    class StarTools:
        @staticmethod
        def get_data_dir(name):
            # 每次运行使用独立目录，避免上一轮的会话绑定/停用状态污染本次断言
            base = _SMOKE_ROOT / name
            base.mkdir(parents=True, exist_ok=True)
            return str(base)

    def register(*_a, **_kw):
        def wrap(cls):
            return cls

        return wrap

    star_mod.Context = object
    star_mod.Star = Star
    star_mod.StarTools = StarTools
    star_mod.register = register

    astrbot_pkg = types.ModuleType("astrbot")
    astrbot_pkg.api = api
    sys.modules.update(
        {
            "astrbot": astrbot_pkg,
            "astrbot.api": api,
            "astrbot.api.event": event_mod,
            "astrbot.api.star": star_mod,
        }
    )


def load_plugin_module():
    spec = importlib.util.spec_from_file_location(
        "tavern_preset_under_test", ROOT / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_event(text: str, umo: str = "aiocqhttp:default:FriendMessage:10001"):
    return SimpleNamespace(
        message_str=text,
        unified_msg_origin=umo,
        get_sender_name=lambda: "主人",
        get_sender_id=lambda: "10001",
        plain_result=lambda t: t,
    )


async def run_command(plugin, text, umo="aiocqhttp:default:FriendMessage:10001"):
    out = []
    async for item in plugin.cmd_preset(make_event(text, umo)):
        out.append(item)
    return "\n".join(str(x) for x in out)


async def main():
    install_astrbot_stubs()
    mod = load_plugin_module()

    # 找一个真实预设喂进去（优先用上传目录里的泉此方预设）
    candidates = [
        p
        for d in (
            ROOT.parent / ".dsh-uploads",
            ROOT / "presets",
        )
        for p in d.rglob("*.json")
    ]
    assert candidates, "找不到任何预设 JSON 用于测试"
    src = candidates[0]

    plugin = mod.TavernPresetPlugin(context=SimpleNamespace(), config={})
    shutil.copy2(src, plugin.user_presets_dir / src.name)
    plugin.library.reload()
    assert plugin.library.presets, "预设库为空"

    name = next(iter(plugin.library.presets))
    print(f"[smoke] 使用预设：{name}（源文件 {src.name}）")

    # --- 1) 绑定前：钩子应跳过 ---
    req = SimpleNamespace(
        system_prompt="You are AstrBot, an AI agent running on this server.",
        contexts=[{"role": "user", "content": "h1"}],
        prompt="你是谁",
    )
    await plugin.on_tavern_request(make_event("你是谁"), req)
    assert req.system_prompt.startswith("You are AstrBot"), "未绑定时不应改动请求"
    assert plugin._stats["skipped"] == 1, plugin._stats

    # --- 2) 全局绑定后：钩子应改写 system prompt / contexts / prompt ---
    out = await run_command(plugin, "/预设 全局 " + name)
    assert "已设置全局预设" in out, out

    original = req.system_prompt
    req = SimpleNamespace(
        system_prompt=original,
        contexts=[{"role": "user", "content": f"h{i}"} for i in range(6)],
        prompt="你是谁",
    )
    await plugin.on_tavern_request(make_event("你是谁"), req)
    assert plugin._stats["applied"] == 1, plugin._stats

    print(f"[smoke] 原 system prompt {len(original)} 字符 -> 新 {len(req.system_prompt)} 字符")
    assert req.system_prompt != original, "system prompt 未被替换"
    assert "AstrBot, an AI agent running" not in req.system_prompt, (
        "默认 drop 模式下框架身份不应出现在 payload 中"
    )
    assert len(req.contexts) >= 6, "历史被破坏"
    assert plugin._last_request.get("system_prompt_len"), "未记录真实请求快照"

    # --- 3) 自检 / 预览 / 变量 命令必须能跑通 ---
    for cmd in ("/预设 自检", "/预设 预览", "/预设 状态", "/预设 列表",
                "/预设 变量查看", "/预设 变量 demo 1", "/预设 详情 " + name):
        out = await run_command(plugin, cmd)
        assert out and not out.startswith("Traceback"), f"{cmd} 输出异常：{out[:200]}"
        print(f"[smoke] {cmd:<18} -> {out.splitlines()[0][:70]}")

    out = await run_command(plugin, "/预设 自检")
    assert "已成功注入" in out or "钩子" in out, out

    # --- 4) 采样同步应写进 provider.custom_extra_body ---
    provider = SimpleNamespace(provider_config={"id": "test-provider"})
    plugin._apply_sampling_to_provider(provider, {"temperature": 0.9, "max_tokens": 800})
    assert provider.provider_config["custom_extra_body"]["temperature"] == 0.9
    plugin._restore_sampling_all()
    assert "custom_extra_body" not in provider.provider_config, "还原失败"

    # --- 5) 停用后应放行 ---
    await run_command(plugin, "/预设 停用")
    req2 = SimpleNamespace(system_prompt="keep-me", contexts=[], prompt="x")
    await plugin.on_tavern_request(make_event("x"), req2)
    assert req2.system_prompt == "keep-me", "停用后不应改动请求"

    print("\n[smoke] 全部通过 ✅")
    print(f"[smoke] 钩子统计：{plugin._stats}")
    print(f"[smoke] 最近请求：presets={plugin._last_request.get('presets')} "
          f"contexts={plugin._last_request.get('contexts')} "
          f"原系统提示={plugin._last_request.get('original_system_len')}字符"
          f"({ '已丢弃' if plugin._last_request.get('original_system_dropped') else '已并入'})")
    print(f"[smoke] 预设变量回写：{len(plugin.store.get_session_vars('aiocqhttp:default:FriendMessage:10001'))} 项")


if __name__ == "__main__":
    asyncio.run(main())
