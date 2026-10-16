> [!WARNING]
> ## ⚠️ 纯 AI 生成 · 凑数演示项目
>
> 本插件的**全部代码、文档与示例预设均由大语言模型自动生成**，仅用于演示 / 凑数，
> **未经人工充分审查，也缺乏长期实机验证**。可能存在 bug、边界情况未覆盖、设计不合理等问题。
> **不建议用于生产环境，使用风险自负。**
> 若你需要可靠的酒馆预设适配器，请优先选择人工维护的项目。
>
> ---

# 酒馆预设适配器（astrbot_plugin_tavern_preset）

让 AstrBot **直接导入并完整使用酒馆（SillyTavern）预设**的插件。
不是世界书插件、不是角色卡插件——它还原的是酒馆的**预设（Preset）**体系：
Prompt Order 编排链、In-Chat @ Depth 深度注入、宏变量替换（含 `setvar/getvar`）、
Handlebars 模板、采样参数同步。

> **v1.1.0 重要修复**：修掉「导入成功但预设看起来没生效」的根因。见下方
> [为什么以前不生效](#为什么以前不生效v110-修复说明)。

## 支持的预设类型（直接导入酒馆导出的 JSON）

| 类型 | 特征字段 | 插件行为 |
| --- | --- | --- |
| Chat Completion 预设 | `prompts` + `prompt_order` | **完整还原**：编排链、启用开关、深度注入（含 `injection_order`）、角色伪装、squash、assistant_prefill、采样参数 |
| Context 预设 | `story_string` | 作为「信息装配模板」叠加到主预设上，接管角色描述/性格/场景等标记位（支持 `{{#if}}`/`{{trim}}`） |
| Sysprompt 预设 | `name` + `content` + `post_history` | 归一化为：主提示词 + 深度 0 注入 |

## 安装与导入

1. 把整个 `astrbot_plugin_tavern_preset` 目录放入 AstrBot 的 `data/plugins/` 下，在 WebUI 启用。
2. 导入预设有两种方式：
   - **聊天直导**：`/预设 导入 [名称]` 后**直接粘贴酒馆导出的完整 JSON**（支持带 ```json 围栏）；
   - **文件导入**：把 `.json` 文件丢进 `数据目录/astrbot_plugin_tavern_preset/presets/`，发送 `/预设 重载`（目录有变动也会自动重载）。
3. `/预设 使用 <名称>` 绑定当前会话；`/预设 全局 <名称>` 设置全局默认。
4. **`/预设 自检`** 确认真的生效（钩子触发次数、真实请求快照、采样状态一目了然）。

插件内置 4 个示例预设（含酒馆官方 Default 预设和一个深度注入演示），装好后发 `/预设 列表` 即可看到。

## 命令

```
/预设 列表 / 状态 / 重载
/预设 导入 [名称] <JSON>
/预设 使用 <名称> / 停用 / 恢复
/预设 全局 <名称> / 取消全局
/预设 上下文 <名称> / 取消上下文      # 叠加 Context(story_string) 预设
/预设 详情 <名称>                     # 查看编排链/注入点/采样表
/预设 设定 <槽位> <内容>              # 填写角色卡/世界书替代槽位
/预设 设定查看 / 设定清除 [槽位|全部]
/预设 变量 <名> <值>                  # 改写预设内部变量（多数中文预设靠它做开关）
/预设 变量查看 [前缀] / 变量清除 [名|全部]
/预设 预览                            # 最近一次真实请求 + 模拟装配
/预设 自检                            # 诊断预设是否真的生效
```

## 为什么以前不生效（v1.1.0 修复说明）

对照 AstrBot v4.27.x 源码逐行核对后，确认有三个叠加原因：

### 1. 框架身份与预设角色争夺身份（主因）

AstrBot v4 的 **Agent 模式**（`agent_runner_type=local`，默认）在 `build_main_agent`
里会把一大堆内容拼进 `req.system_prompt`：Agent 系统提示、人格（Persona）、技能、
工具调用提示、本地环境提示、安全模式提示、时间提示……

v1.0 的默认配置 `use_persona_as_char=true` 会把这**整坨**内容搬进预设的
`charDescription` 标记位。于是模型同时看到「你是运行在服务器上的 AstrBot 助手，
可以执行 shell/Python」和「你是泉此方」——它选择前者，并回一句
「那些是消息里附带的设定文本，我不会拿它替换自己的身份」。

**修复**：新增「原系统提示处理」配置（`original_system_prompt_mode`），默认 **`drop`**：
预设激活时框架提示不再进入 payload（仍可用 `{{original}}` 宏手动引用）。
需要保留工具/身份说明时，可改成 `prepend_to_head` 或 `to_char_description`。

### 2. 采样参数根本没到 API

v4.27 的 `ProviderRequest` 只有 `prompt / contexts / system_prompt / image_urls /
audio_urls / func_tool / conversation / model` 等字段，**没有** `temperature`、
`top_p`、`max_tokens`；而且 `text_chat(**kwargs)` 会把多余参数直接忽略。
v1.0 用 `setattr(req, "temperature", …)` 写属性，纯属空转。

**修复**：改用 Provider 真正读取的通道——写 `provider.provider_config["custom_extra_body"]`
（OpenAI 适配器会把它合并进请求体）。解绑预设 / 停用 / 卸载插件时自动还原原值。
配置项 `sampling_sync_mode`（默认 `custom_extra_body`；`off` 关闭；`legacy_req_attrs`
保留旧行为）。

### 3. 宏保真度不足，导致噪声与逻辑失效

真实中文预设（如「Izumi 1002」）里：`{{setvar}}` × 219、`{{getvar}}` × 243、
`{{//注释}}` × 13。v1.0 把未支持的宏**原样发给模型**，还把整条都是注释的作者说明
（2000+ 字）当成正文发出去。

**修复**：

- `{{// 任意内容 }}` 整块剥离（与酒馆一致）；
- `{{setvar::名::值}}` / `{{addvar}}` / `{{incvar}}` / `{{decvar}}` / `{{getvar::名}}`
  按编排顺序求值，变量**按会话持久化**，可用 `/预设 变量` 查看与改写；
- `{{random::a::b}}` 双冒号写法、`{{#if}}` 里引用预设变量也已支持。

实测效果（泉此方预设 + 模拟 v4.27 Agent 请求）：

| 指标 | v1.0 | v1.1（drop） |
| --- | --- | --- |
| system 头部长度 | 6720 字符 | 1216 字符 |
| 残留未解析宏 | 多处 | 0 处 |
| 作者说明被发送 | 是 | 否 |
| 框架身份进入 payload | 是 | 否 |
| 预设变量跟踪 | 无 | 92 项 |

## 适配说明：AstrBot 没有角色卡和世界书，怎么办？

| 酒馆概念 | 标记位 | AstrBot 适配 |
| --- | --- | --- |
| 角色卡·描述 | `charDescription` | 槽位「描述」（最推荐：把角色卡正文贴进去） |
| 角色卡·性格 | `charPersonality` | 槽位「性格」 |
| 角色卡·场景 | `scenario` | 槽位「场景」 |
| 用户人设 | `personaDescription` | 槽位「人设」 |
| 对话例句 | `dialogueExamples` | 槽位「例句」 |
| 世界书·前置/后置 | `worldInfoBefore/After` | 槽位「世界前/世界后」（静态内容） |
| `{{char}}` | — | 槽位「角色名」→ 配置默认角色名 |
| `{{user}}` | — | 槽位「用户名」→ 发送者昵称 |

槽位按「会话 > 全局 > 默认」合并；内容为空的标记位会被**自动跳过**（不会发出空消息），
并在 `/预设 预览` 的适配说明中列出。

### 深度注入（In-Chat @ Depth）的落点

- `depth >= 1`：完整还原——插入到历史记录倒数第 N 条之前，角色按 `injection_role`
  伪装；同深度按 `injection_order` 排序（order 越大越靠近生成位置）；
- `depth = 0`：酒馆语义是「整条消息列表的最后一条」。AstrBot 管线中最后一条固定为当前
  用户输入，因此提供三种落点（配置 `depth0_mode`）：
  - `prompt_suffix`（默认）：追加到当前用户输入末尾——**离生成位置最近**；
  - `context_tail`：作为独立消息放在历史末尾；
  - `system_prompt`：并入系统提示。

## 宏变量支持

`{{char}}` `{{user}}` `{{description}}` `{{personality}}` `{{scenario}}` `{{persona}}`
`{{mesExamples}}` `{{input}}` `{{original}}` `{{wiBefore}}` `{{wiAfter}}`
`{{time}}` `{{date}}` `{{weekday}}` `{{isotime}}` `{{isodate}}`
`{{random:a,b,c}}` `{{pick}}` `{{roll:2d6+1}}` `{{//注释}}`
`{{setvar::k::v}}` `{{addvar::k::v}}` `{{incvar::k}}` `{{decvar::k}}` `{{getvar::k}}`

Handlebars 子集：`{{#if var}}…{{else}}…{{/if}}`（支持嵌套）、`{{trim}}`。
已知但为空的变量替换为空串；未识别变量原样保留以便排查。

## 排障

| 现象 | 处理 |
| --- | --- |
| `/预设 自检` 显示「钩子从未触发」 | 插件未启用，或该实例没走本插件的请求链路 |
| 自检显示「钩子触发但未注入」 | 没绑定预设：`/预设 全局 <名称>` 或 `/预设 使用 <名称>` |
| 已注入但模型仍自称 AstrBot 助手 | ① 把「原系统提示处理」确认为 `drop`；② **`/reset` 开新会话**——旧历史里保留了身份自述，模型会跟着旧历史走；③ 用 `/预设 设定 描述 <角色卡正文>` 强化角色锚定 |
| 采样参数似乎没变 | 看 `/预设 自检` 的采样同步状态；若 Provider 是自定义实现，请在提供商设置里手动同步 |
| 预设里的开关（字数/格式等）不起作用 | 用 `/预设 变量查看` 看变量是否初始化；注意预设自身的 `{{setvar}}` 初始化会覆盖手动值（与酒馆行为一致） |
| 世界书条目为空 | AstrBot 无世界书：用 `/预设 设定 世界前/世界后` 填静态内容 |

## 目录结构

```
astrbot_plugin_tavern_preset/
├── metadata.yaml / _conf_schema.json / requirements.txt
├── main.py               # 插件入口：命令分派 + on_llm_request 钩子 + 采样同步 + 自检
├── core/
│   ├── parser.py         # 预设识别/归一化 + 预设库扫描（纯标准库）
│   ├── macro_engine.py   # 宏变量/变量宏/注释/Handlebars 子集（纯标准库）
│   └── pipeline.py       # prompt_order 编排 + 深度注入装配（纯标准库）
├── store.py              # 会话绑定/槽位/预设变量持久化（JSON）
├── presets/              # 内置示例预设（用户预设放数据目录 presets/）
└── tests/
    ├── test_core.py          # 离线单测（32 项，纯标准库）
    └── test_plugin_smoke.py  # 主入口冒烟：stub astrbot 后真实跑钩子与命令
```

## 提示

- 会话级停用用 `/预设 停用`（不影响全局），恢复用 `/预设 恢复`。
- 多个会话可绑定不同预设；未绑定的会话跟随全局预设。
- 打开配置的「输出装配日志」可在 AstrBot 日志中查看每次请求的装配摘要。
- 预设 JSON 里 `prompt_order` 可含多个 `character_id` 方案，默认选用 100000
  （可在配置修改；找不到时自动回退到预设里的第一个方案）。
