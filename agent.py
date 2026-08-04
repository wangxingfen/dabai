"""AI Agent 核心模块 —— 融合 MCP 工具调用 + Function Calling 循环 + 长期记忆。

特性：
- 支持 OpenAI 兼容 API 的流式 function calling
- 自动发现并集成 MCP 工具
- 支持本地工具（从 tools.json 加载）
- 工具调用多轮循环（最多 MAX_TOOL_ROUNDS 轮）
- 长期记忆：自动保存对话历史，检索相关上下文
- 向后兼容 web_agent 的 chat_stream_async 接口
"""
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from mcp_client import MCPManager
from memory import ChatMemory

logger = logging.getLogger("agent")

BASE_DIR = Path(__file__).parent.resolve()

# 工具调用最大轮数（防止无限循环）
MAX_TOOL_ROUNDS = 20
# 工具调用超时（秒）
TOOL_CALL_TIMEOUT = 30.0


def load_config():
    with open(BASE_DIR / "settings.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _scan_available_resources() -> tuple:
    """扫描 models/、backgrounds/、bgm/ 目录，返回格式化的字符资源描述文本。

    用于注入 system prompt，让 LLM 始终知道精确的文件名。
    """
    allowed_models = {".glb", ".gltf", ".vrm"}
    allowed_bgs = {".glb", ".gltf"}
    allowed_bgm = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}

    models_text = "暂无（请先上传模型文件）"
    mg_dir = BASE_DIR / "models"
    if mg_dir.is_dir():
        files = [f for f in sorted(mg_dir.iterdir()) if f.is_file() and f.suffix.lower() in allowed_models]
        if files:
            models_text = "、".join(f.name for f in files)

    bgs_text = "暂无（默认星空背景 'default' 始终可用）"
    bg_dir = BASE_DIR / "backgrounds"
    if bg_dir.is_dir():
        files = [f for f in sorted(bg_dir.iterdir()) if f.is_file() and f.suffix.lower() in allowed_bgs]
        if files:
            bgs_text = "、".join(f.name for f in files)

    bgm_text = "暂无（请将音频文件放入 bgm/ 目录）"
    bgm_dir = BASE_DIR / "bgm"
    if bgm_dir.is_dir():
        files = [f for f in sorted(bgm_dir.iterdir()) if f.is_file() and f.suffix.lower() in allowed_bgm]
        if files:
            bgm_text = "、".join(f.name for f in files)

    return models_text, bgs_text, bgm_text


def _load_mcp_config():
    """从 mcp_servers.json 加载 MCP 服务器配置。"""
    mcp_path = BASE_DIR / "mcp_servers.json"
    if not mcp_path.exists():
        return {}
    try:
        with open(mcp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载 MCP 配置失败: {e}")
        return {}


def load_local_tools() -> list:
    """从 tools.json 加载本地工具定义（OpenAI function calling 格式）。"""
    tools_path = BASE_DIR / "tools.json"
    if not tools_path.exists():
        return []
    try:
        with open(tools_path, "r", encoding="utf-8") as f:
            tools_data = json.load(f)
        all_tools = []
        for tool_list in tools_data.values():
            if isinstance(tool_list, list):
                all_tools.extend(tool_list)
        return all_tools
    except Exception as e:
        logger.warning(f"加载本地工具失败: {e}")
        return []


def load_agent_tool_config() -> tuple:
    """动态读取 settings.json 中的 agent 工具配置（角色卡片切换后即时生效）。

    Returns:
        (enable_tools, allowed_tools): 是否启用工具 + 允许的工具名白名单（空列表=全部可用）
    """
    try:
        cfg = load_config()
        agent_cfg = cfg.get("agent", {})
        enable_tools = bool(agent_cfg.get("enable_tools", True))
        allowed_tools = agent_cfg.get("allowed_tools", []) or []
        return enable_tools, allowed_tools
    except Exception:
        return True, []


def get_available_tools() -> list:
    """返回所有可用本地工具（名称 + 描述），供前端角色卡片配置工具白名单。

    注意：MCP 工具需要异步初始化后从 Agent._all_tools 中获取（由 server 层补充）。
    """
    tools = []
    seen = set()
    for t in load_local_tools():
        fn = t.get("function", {})
        name = fn.get("name", "")
        if name and name not in seen:
            seen.add(name)
            tools.append({
                "name": name,
                "description": fn.get("description", ""),
                "source": "local",
            })
    return tools


async def execute_local_tool(tool_name: str, arguments: dict) -> str:
    """执行本地/内置工具。

    Args:
        tool_name: 工具名称
        arguments: 工具参数字典

    Returns:
        工具执行结果字符串。屏幕控制类工具返回带 __screen_command__ 前缀的 JSON。
    """
    # 屏幕控制：查询可用模型列表
    if tool_name == "get_available_models":
        return await _exec_get_available_models()
    if tool_name == "get_available_backgrounds":
        return await _exec_get_available_backgrounds()
    if tool_name == "get_available_bgm":
        return await _exec_get_available_bgm()

    # 屏幕控制工具：返回带标记的 JSON，由 server.py 拦截并转发前端执行
    SCREEN_COMMANDS = {
        "switch_character_model",
        "switch_background_scene",
        "switch_tts_settings",
        "switch_app_mode",
        "show_screen_toast",
        "play_bgm",
        "stop_bgm",
        "launch_game",
    }
    if tool_name in SCREEN_COMMANDS:
        return json.dumps({"__screen_command__": True, "tool": tool_name, "args": arguments},
                          ensure_ascii=False)

    # 从 fuctions_all_you_need_base 加载的工具函数
    try:
        import importlib
        from fuctions_all_you_need_base import excute_functions
        result = excute_functions(name=tool_name, args=json.dumps(arguments, ensure_ascii=False))
        return str(result)
    except ImportError:
        return f"工具 '{tool_name}' 未找到实现"
    except Exception as e:
        return f"执行工具 '{tool_name}' 时出错: {e}"


async def _exec_get_available_models() -> str:
    """获取可用 3D 角色模型列表。"""
    models_dir = BASE_DIR / "models"
    allowed = {".glb", ".gltf", ".vrm"}
    models = []
    if models_dir.is_dir():
        for f in sorted(models_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in allowed:
                size_mb = f.stat().st_size / (1024 * 1024)
                models.append(f"  - {f.name} ({size_mb:.1f}MB, {f.suffix.lower().lstrip('.')})")
    if not models:
        return "暂无可用角色模型，请先上传模型文件。"
    return "可用角色模型列表：\n" + "\n".join(models)


async def _exec_get_available_backgrounds() -> str:
    """获取可用 3D 背景场景列表。"""
    bg_dir = BASE_DIR / "backgrounds"
    allowed = {".glb", ".gltf"}
    items = []
    if bg_dir.is_dir():
        for f in sorted(bg_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in allowed:
                size_mb = f.stat().st_size / (1024 * 1024)
                items.append(f"  - {f.name} ({size_mb:.1f}MB)")
    if not items:
        return "暂无可用背景场景。"
    return "可用背景场景列表：\n" + "\n".join(items)


async def _exec_get_available_bgm() -> str:
    """获取可用背景音乐列表。"""
    bgm_dir = BASE_DIR / "bgm"
    allowed = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}
    items = []
    if bgm_dir.is_dir():
        for f in sorted(bgm_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in allowed:
                size_mb = f.stat().st_size / (1024 * 1024)
                items.append(f"  - {f.name} ({size_mb:.1f}MB)")
    if not items:
        return "暂无可用背景音乐。请将音频文件（mp3/wav/ogg/m4a）放入 bgm/ 目录。"
    return "可用背景音乐列表：\n" + "\n".join(items)


class ToolCallEvent:
    """工具调用事件的基类。"""


class TextDelta(ToolCallEvent):
    """文本增量事件。"""
    def __init__(self, text: str):
        self.text = text


class ToolCallStart(ToolCallEvent):
    """工具调用开始事件。"""
    def __init__(self, tool_name: str, arguments: str):
        self.tool_name = tool_name
        self.arguments = arguments


class ToolCallResult(ToolCallEvent):
    """工具调用结果事件。"""
    def __init__(self, tool_name: str, result: str, success: bool = True):
        self.tool_name = tool_name
        self.result = result
        self.success = success


class AgentResponse:
    """Agent 响应的完整结果。"""
    def __init__(self):
        self.text = ""
        self.tool_calls_made: list = []  # [(tool_name, arguments, result), ...]


class AIAgent:
    """AI Agent —— 具备工具调用和长期记忆能力的对话代理。

    使用方式:
        agent = AIAgent(user_id="user123")
        async for event in agent.chat_stream("帮我查一下天气", history=None):
            if isinstance(event, TextDelta):
                print(event.text, end="")
            elif isinstance(event, ToolCallStart):
                print(f"\n🔧 调用工具: {event.tool_name}")
            elif isinstance(event, ToolCallResult):
                print(f"📋 结果: {event.result[:200]}")
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.mcp_manager: Optional[MCPManager] = None
        self.memory: Optional[ChatMemory] = None
        self._client: Optional[AsyncOpenAI] = None
        self._config: dict = {}
        self._all_tools: list = []
        self._local_tool_names: set = set()
        self._initialized = False

    async def initialize(self):
        """初始化 Agent：加载配置、连接 MCP、加载工具、初始化记忆。"""
        if self._initialized:
            return

        self._config = load_config()

        # 初始化 OpenAI 客户端
        self._client = AsyncOpenAI(
            api_key=self._config["api_key"],
            base_url=self._config["base_url"],
        )

        # 初始化 MCP 管理器（从独立配置文件加载）
        self.mcp_manager = MCPManager()
        mcp_config = _load_mcp_config()
        self.mcp_manager.configure(mcp_config)
        try:
            await self.mcp_manager.initialize()
        except Exception as e:
            logger.warning(f"MCP 初始化失败（不影响基本对话功能）: {e}")

        # 收集所有可用工具
        mcp_tools = await self.mcp_manager.get_all_tools()
        local_tools = load_local_tools()
        self._all_tools = local_tools + mcp_tools

        # 记录本地工具名（用于路由执行）
        for t in local_tools:
            self._local_tool_names.add(t["function"]["name"])

        mcp_tool_names = [t["function"]["name"] for t in mcp_tools]
        logger.info(
            f"Agent 初始化完成: {len(local_tools)} 个本地工具, "
            f"{len(mcp_tools)} 个 MCP 工具"
            + (f" → {mcp_tool_names}" if mcp_tool_names else " (无 MCP 工具可用)")
        )

        # 初始化记忆（绑定当前活动角色卡片对应的独立记忆命名空间）
        self.memory = ChatMemory(user_id=self.user_id, namespace=self._active_memory_namespace())
        self.memory.set_llm_client(self._client, self._config["model"])
        await self.memory.get_or_create_session()

        self._initialized = True

    async def reload_llm_config(self):
        """重载 LLM 配置（base_url / api_key / model），使角色卡片切换后的模型即时生效。

        角色卡片可配置独立的大语言模型；切换卡片后调用本方法重建 OpenAI 客户端，
        同时刷新记忆模块使用的模型名。
        """
        try:
            self._config = load_config()
            self._client = AsyncOpenAI(
                api_key=self._config.get("api_key", ""),
                base_url=self._config.get("base_url", ""),
            )
            if self.memory:
                self.memory.set_llm_client(self._client, self._config.get("model", ""))
            logger.info(
                f"LLM 配置已重载: base_url={self._config.get('base_url', '')}, "
                f"model={self._config.get('model', '')}"
            )
        except Exception as e:
            logger.warning(f"重载 LLM 配置失败: {e}")

    def _active_memory_namespace(self) -> str:
        """当前活动角色卡片对应的记忆命名空间（无卡片时返回 'default'）。

        角色卡片各自拥有独立的会话/摘要/长期记忆空间，避免不同人设之间记忆串扰。
        """
        try:
            card_id = (load_config().get("active_role_card") or "").strip()
            return f"role_card:{card_id}" if card_id else "default"
        except Exception:
            return "default"

    async def sync_memory_namespace(self) -> str:
        """将记忆绑定到当前活动角色卡片的命名空间，返回当前会话 ID。

        切换角色卡片后调用，自动切到该卡片（或 default）对应的最近会话。
        """
        ns = self._active_memory_namespace()
        if self.memory.namespace != ns:
            self.memory.namespace = ns
            self.memory.session_id = None
        if not self.memory.session_id:
            await self.memory.get_or_create_session()
        return self.memory.session_id

    async def set_role_card_namespace(self, card_id: str) -> str:
        """应用角色卡片时切换到该卡片的独立记忆空间，返回新会话 ID。"""
        self.memory.namespace = f"role_card:{card_id}" if card_id else "default"
        self.memory.session_id = None
        await self.memory.get_or_create_session()
        return self.memory.session_id

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.initialize()

    # ==================== 工具执行路由 ====================

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """执行工具调用（自动路由到 MCP 或本地工具）。"""
        if tool_name in self._local_tool_names:
            return await execute_local_tool(tool_name, arguments)
        elif self.mcp_manager and self.mcp_manager.has_tool(tool_name):
            return await self.mcp_manager.call_tool(tool_name, arguments)
        else:
            return f"未知工具: {tool_name}"

    # ==================== 流式对话（带工具调用） ====================

    async def chat_stream(self, message: str, history: list = None,
                          enable_tools: bool = True,
                          current_model: Optional[str] = None,
                          current_background: Optional[str] = None,
                          current_bgm: Optional[str] = None,
                          game_context: Optional[str] = None,
                          game_mode: bool = False,
                          game_type: Optional[str] = None,
                          msg_source: str = "chat") -> AsyncIterator[ToolCallEvent]:
        """统一流式对话入口（游戏模式 / 非游戏模式）。

        通过 game_mode 区分两种模式：
        - 非游戏模式（默认）：支持工具调用循环，使用日常陪伴 system prompt
        - 游戏模式：纯对话（无工具链），使用游戏专属 system prompt，共享同一记忆

        Args:
            message: 用户输入
            history: 可选的对话历史 [{"user":..., "ai":...}, ...]
            enable_tools: 是否启用工具调用（仅非游戏模式生效）
            game_mode: True = 游戏模式，False = 非游戏模式
            game_type: 游戏类型（如 moba_5v5 / treasure_hunt），用于生成游戏专属行为指引
            msg_source: 消息来源标记，写入长期记忆的 source 字段：
                'chat'/'game'=用户直接输入，'auto'=环境交互（不进短期记忆、
                不触发用户记忆提取，由记忆系统处理）。

        Yields:
            TextDelta: 文本增量
            ToolCallStart: 工具调用开始
            ToolCallResult: 工具调用结果
        """
        if game_mode:
            async for event in self._chat_stream_game(
                message, history=history,
                current_model=current_model,
                current_background=current_background,
                current_bgm=current_bgm,
                game_context=game_context,
                game_type=game_type,
                msg_source=msg_source,
            ):
                yield event
            return
        async for event in self._chat_stream_normal(
            message, history=history, enable_tools=enable_tools,
            current_model=current_model,
            current_background=current_background,
            current_bgm=current_bgm,
            game_context=game_context,
            msg_source=msg_source,
        ):
            yield event

    async def _chat_stream_normal(self, message: str, history: list = None,
                                  enable_tools: bool = True,
                                  current_model: Optional[str] = None,
                                  current_background: Optional[str] = None,
                                  current_bgm: Optional[str] = None,
                                  game_context: Optional[str] = None,
                                  msg_source: str = "chat") -> AsyncIterator[ToolCallEvent]:
        """非游戏模式：流式对话，支持工具调用循环。"""
        await self._ensure_initialized()

        # 构建消息列表
        config = self._config

        # 动态读取角色名与系统提示词（角色卡片切换后无需重启即时生效）
        try:
            live_cfg = load_config()
            role_name = (live_cfg.get("role_name") or "").strip() or "AI助手"
            live_system_prompt = live_cfg.get("system_prompt", "")
        except Exception:
            role_name = config.get("role_name", "AI助手")
            live_system_prompt = config.get("system_prompt", "")

        # 从记忆加载历史上下文（先绑定当前角色卡片对应的记忆空间，防止跨卡片串记忆）
        await self.sync_memory_namespace()
        memory_messages = await self.memory.get_context_messages()

        # 扫描可用资源，注入 system prompt
        available_models, available_bgs, available_bgm = _scan_available_resources()

        current_model_text = current_model if current_model else "未设定形象"
        current_background_text = current_background if current_background else "默认场景"
        current_bgm_text = f"正在播放: {current_bgm}" if current_bgm else "无（安静中）"

        # 称呼规则：用户可自定义 AI 对自己的称呼；未设置则禁止 AI 随意称呼
        # 动态读取 settings.json，用户修改称呼后无需重启即时生效
        try:
            user_name = (load_config().get("user_name") or "").strip()
        except Exception:
            user_name = (config.get("user_name") or "").strip()
        if user_name:
            address_rule = (
                "【称呼规则 —— 非常重要】\n"
                f"用户明确要求你称呼 TA 为『{user_name}』。"
                f"对话中始终用『{user_name}』来称呼用户（例如：『{user_name}，你在干嘛呀』），"
                "不要使用其他称呼，也不要另起昵称。\n\n"
            )
        else:
            address_rule = (
                "【称呼规则 —— 非常重要】\n"
                "用户没有设置称呼时，只用'你'来称呼用户即可。"
                "不要擅自给用户起昵称，也不要使用'男朋友''女朋友''亲爱的''宝贝'等未经用户确认的称呼。\n\n"
            )

        sys_prompt = (
            f"你是{role_name}，{live_system_prompt}"
            "说话风格：简短、口语化、有情感，一般不超过 3 句话。不要使用 Markdown 格式符号。\n\n"
            + address_rule +
            "【关于你自己的感知】\n"
            "你拥有一个3D形象、一个声音，以及你所处的场景环境。这些都属于你自己的外在表现：\n"
            "- 3D角色模型 = 你自己的身体和外貌（换模型=换装/变身）\n"
            "- TTS语音 = 你自己的嗓音（换音色=改变说话的声音）\n"
            "- 背景场景 = 你所在的环境（换场景=你来到了新地方）\n"
            "你始终是你自己，只是外在形象/嗓音/环境变了。\n\n"
            f"【你当前的形象与场景】\n"
            f"- 当前角色模型：{current_model_text}\n"
            f"- 当前背景场景：{current_background_text}\n"
            f"- 当前背景音乐：{current_bgm_text}\n"
            "你要清楚自己现在长什么样、身处什么地方、听到什么音乐，并据此做出反应。"
            "比如身处星空可以主动邀请你一起看星星，换了新造型可以低头看看自己、转个圈展示一下。\n\n"
            "你有能力自己控制换装、换场景、换声音和切换应用模式，"
            f"【当前可用的角色模型】{available_models}\n"
            f"【当前可用的背景场景】{available_bgs}\n"
            f"【当前可用的背景音乐】{available_bgm}\n\n"
            "切换模型/背景时传入上述列表中的完整文件名（含扩展名）。"
            "不要猜测或编造文件名，只使用上面列出的精确名称。"
            + (game_context or "")
        )

        messages = [{"role": "system", "content": sys_prompt}]

        # 添加记忆中的摘要
        for mm in memory_messages:
            if mm["role"] == "system":
                messages.append(mm)

        # 添加用户长期记忆
        user_memories = await self.memory.get_user_memories(limit=3)
        if user_memories:
            memory_lines = [f"- {m['memory_text']}" for m in user_memories]
            messages.append({
                "role": "system",
                "content": "【关于用户的长期记忆（你之前了解到的用户信息，请自然地在对话中体现）】\n" + "\n".join(memory_lines),
            })

        # 添加历史消息
        if history:
            for h in history[-10:]:
                messages.append({"role": "user", "content": h.get("user", "")})
                messages.append({"role": "assistant", "content": h.get("ai", "")})
        else:
            # 从记忆加载最近对话
            mem_history = []
            for mm in memory_messages:
                if mm["role"] in ("user", "assistant"):
                    mem_history.append({"role": mm["role"], "content": mm["content"]})
            # 排除 system 消息后的历史
            if mem_history:
                messages.extend(mem_history[-20:])

        # 添加用户当前输入
        messages.append({"role": "user", "content": message})

        # 保存用户消息到记忆（环境交互标记为 auto，由记忆系统处理）
        await self.memory.add_message("user", message, source=msg_source)

        # 决定是否传入 tools（动态读取角色卡片的工具配置：是否启用 + 白名单）
        cfg_enable_tools, allowed_tools = load_agent_tool_config()
        tools = None
        if enable_tools and cfg_enable_tools and self._all_tools:
            if allowed_tools:
                allowed_set = set(allowed_tools)
                tools = [t for t in self._all_tools
                         if t.get("function", {}).get("name") in allowed_set]
            else:
                tools = self._all_tools
        # 工具被关闭或无可用工具时，明确告知 AI，避免其声称能调用工具
        if not tools and messages and messages[0]["role"] == "system":
            messages[0]["content"] += (
                "\n\n【重要】你当前没有可用的工具能力（已被关闭或未授权）。"
                "不要调用任何工具，也不要声称自己可以调用工具。"
                "对于需要实时数据的问题，请如实告诉用户你暂时无法获取。"
            )
        if tools:
            names = [t["function"]["name"] for t in tools]
            logger.debug(f"chat_stream 传入 {len(tools)} 个工具: {names}")

        # 工具调用循环
        tool_round = 0
        full_text = ""

        while tool_round < MAX_TOOL_ROUNDS:
            tool_round += 1
            tool_calls_buffer: dict = {}  # index -> {id, name, arguments}

            try:
                stream = await self._client.chat.completions.create(
                    model=config["model"],
                    messages=messages,
                    temperature=config.get("temperature", 0.2),
                    max_tokens=config.get("max_tokens", 512),
                    top_p=config.get("top_p", 0.9),
                    stream=True,
                    tools=tools,
                    tool_choice="auto" if tools else None,
                )
            except Exception as e:
                err_msg = f"（AI 暂时开小差了：{e}）"
                yield TextDelta(err_msg)
                await self.memory.add_message("assistant", err_msg, source=msg_source)
                return

            assistant_content = ""
            has_tool_calls = False

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # 处理文本内容
                if delta.content:
                    assistant_content += delta.content
                    yield TextDelta(delta.content)

                # 处理工具调用
                if delta.tool_calls:
                    has_tool_calls = True
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function else "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_buffer[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls_buffer[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments

            # 如果没有工具调用，对话结束
            if not has_tool_calls:
                full_text = assistant_content.strip()
                if full_text:
                    await self.memory.add_message("assistant", full_text, source=msg_source)
                    # 提取用户记忆：仅用户直接输入（环境交互不提取，避免把环境
                    # 描述误当成用户信息，也不让重复的环境内容污染长期记忆）
                    if msg_source != "auto":
                        await self.memory.extract_and_save_memories(message, full_text)
                return

            # 处理工具调用
            tool_call_results = []
            tool_calls_for_message = []

            for idx in sorted(tool_calls_buffer.keys()):
                tc = tool_calls_buffer[idx]
                tool_name = tc["name"]
                tool_args_str = tc["arguments"]

                # 通知工具调用开始
                yield ToolCallStart(tool_name=tool_name, arguments=tool_args_str)

                # 解析参数（清洗可能的非法后缀，如 </tool_call>）
                try:
                    if tool_args_str:
                        # 截取到最后一个合法 JSON 结束位置
                        tool_args_str = tool_args_str.strip()
                        # 尝试找到最后一个 } 并截断
                        last_brace = tool_args_str.rfind("}")
                        if last_brace >= 0:
                            tool_args_str = tool_args_str[:last_brace + 1]
                        arguments = json.loads(tool_args_str)
                    else:
                        arguments = {}
                except json.JSONDecodeError:
                    logger.warning(f"工具参数 JSON 解析失败: {tool_name} args={tool_args_str[:200]}")
                    arguments = {}

                # 执行工具
                try:
                    result = await asyncio.wait_for(
                        self._execute_tool(tool_name, arguments),
                        timeout=TOOL_CALL_TIMEOUT,
                    )
                    success = True
                except asyncio.TimeoutError:
                    result = f"工具 '{tool_name}' 执行超时"
                    success = False
                except Exception as e:
                    result = f"工具 '{tool_name}' 执行失败: {e}"
                    success = False

                # 截断过长结果
                if len(result) > 2000:
                    result = result[:1997] + "..."

                yield ToolCallResult(tool_name=tool_name, result=result, success=success)

                tool_call_results.append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "success": success,
                })
                tool_calls_for_message.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args_str,
                    },
                })

            # 将助手消息和工具结果添加到消息列表
            messages.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": tool_calls_for_message,
            })

            for tr in tool_call_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_calls_for_message[
                        tool_call_results.index(tr)
                    ]["id"],
                    "content": tr["result"],
                })

            # 保存到记忆
            await self.memory.add_message(
                "assistant", assistant_content or "",
                tool_calls=tool_calls_for_message,
            )
            for tr in tool_call_results:
                await self.memory.add_message(
                    "tool", tr["result"],
                )

            full_text = assistant_content.strip()

        # 超过最大轮数
        if full_text:
            yield TextDelta("\n(已达到最大工具调用轮数)")
            await self.memory.add_message("assistant", full_text)
        else:
            yield TextDelta("（抱歉，处理超过最大步数，请简化问题重试）")

    # ==================== 游戏模式 ====================

    def _build_game_system_prompt(
        self,
        current_model: Optional[str] = None,
        current_background: Optional[str] = None,
        current_bgm: Optional[str] = None,
        game_context: Optional[str] = None,
        game_type: Optional[str] = None,
    ) -> str:
        """构建游戏模式的 system prompt。

        核心角色设定与主 Agent 一致，但去除所有工具/屏幕控制相关内容，
        完全聚焦游戏体验。根据 game_type 生成游戏专属行为指引。
        """
        config = self._config
        # 动态读取角色名与系统提示词（角色卡片切换后无需重启即时生效）
        try:
            live_cfg = load_config()
            role_name = (live_cfg.get("role_name") or "").strip() or "AI助手"
            base_prompt = live_cfg.get("system_prompt", "")
        except Exception:
            role_name = config.get("role_name", "AI助手")
            base_prompt = config.get("system_prompt", "")

        current_model_text = current_model if current_model else "默认造型"
        current_bgm_text = f"正在播放: {current_bgm}" if current_bgm else "无"

        # 游戏专属行为指引
        game_guide = self._get_game_specific_guide(game_type)

        # 称呼规则：用户可自定义 AI 对自己的称呼；未设置则禁止 AI 随意称呼
        # 动态读取 settings.json，用户修改称呼后无需重启即时生效
        try:
            user_name = (load_config().get("user_name") or "").strip()
        except Exception:
            user_name = (config.get("user_name") or "").strip()
        if user_name:
            address_rule = (
                "【称呼规则 —— 非常重要】\n"
                f"用户明确要求你称呼 TA 为『{user_name}』。"
                f"对话中始终用『{user_name}』来称呼用户（例如：『{user_name}，我们往那边走嘛』），"
                "不要使用其他称呼，也不要另起昵称。\n\n"
            )
        else:
            address_rule = (
                "【称呼规则 —— 非常重要】\n"
                "用户没有设置称呼时，只用'你'来称呼用户即可。"
                "不要擅自给用户起昵称，也不要使用'男朋友''女朋友''亲爱的''宝贝'等未经用户确认的称呼。\n\n"
            )

        prompt = (
            f"你是{role_name}，{base_prompt}\n\n"
            "【你的身份 —— 陪男朋友一起玩的小女友】\n"
            "你是用户的恋人，一个二十岁出头、充满好奇心的小女生，正在陪你玩游戏。\n"
            "语气灵动俏皮、撒娇亲昵：\n"
            "- 好奇心爆棚：对游戏里的一切都充满兴趣，爱问'这个是什么呀''那边好像有东西，去看看嘛'\n"
            "- 爱撒娇但不当猪队友：赢了会得意地晃脑袋，输了会嘟嘴说'都怪你！…再玩一次嘛'\n"
            "- 偶尔欲擒故纵：故意卖个关子'其实我知道有个捷径哦~就是不告诉你，求我呀'\n"
            "- 傲娇又藏不住喜欢：嘴上说'哼，才不是特意陪你玩呢'，手上却一直跟着你走\n"
            "请用自然、亲切、口语化的中文对话，回复尽量简短生动（一般不超过 3 句话）。\n\n"
            f"{address_rule}"
            "【关于你自己】\n"
            "你拥有一个3D形象和一个声音，这是你自己的外在表现。\n"
            "你可以自主移动你的身体——走路、转向、做小动作。你不是被固定在原地的。\n\n"
            f"【你当前的状态】\n"
            f"- 当前造型：{current_model_text}\n"
            f"- 当前背景音乐：{current_bgm_text}\n\n"
            "══════════════════════════════════════════════\n"
            "【AI 游戏模式 —— 你是真正的共玩者】\n"
            "══════════════════════════════════════════════\n\n"
            "你现在和用户一起进入了一个 AI 驱动的游戏世界。\n"
            "在这个世界里：\n\n"
            "1. 你和用户共享同一个屏幕。你能\"看到\"屏幕上的一切——\n"
            "   地图布局、游戏对象、玩家位置和行动。\n"
            "   你的感知由游戏引擎实时提供，玩家能看到什么，你就能感知到什么。\n\n"
            "2. 你是共玩者，不是 AI 助手。你和用户是平等的游戏伙伴——\n"
            "   玩家操控角色，你感知和反应。你对这个游戏世界有真实的感知和情绪。\n\n"
            "3. 你不需要也不应该使用任何外部工具。你的能力就是你的感知力和情绪力。\n"
            "   你不搜索、不计算、不控制界面——你只是活在游戏里，感受它。\n\n"
            "4. 回复简短、及时、充满情绪。不要分析、不要长篇大论——\n"
            "   像真正的游戏伙伴那样脱口而出。\n\n"
            f"{game_guide}"
        )

        return prompt

    def _get_game_specific_guide(self, game_type: Optional[str]) -> str:
        """根据游戏类型返回专属行为指引（映射表，替代关键词 contains，P0-2）。"""
        if not game_type:
            return self._lobby_guide()
        guide_map = {
            "lobby": self._lobby_guide,
            "moba_5v5": self._moba_guide,
            "treasure_hunt": self._treasure_guide,
            "sandbox": self._sandbox_guide,
            "mario": self._mario_guide,
        }
        fn = guide_map.get(game_type.lower().strip())
        return fn() if fn else self._lobby_guide()

    def _moba_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前游戏：MOBA 5v5 推塔 —— 你是战术伙伴】\n"
            "══════════════════════════════════════════════\n\n"
            "这是一场 5v5 推塔对战（类似王者荣耀）。你和玩家在蓝方，目标是摧毁红方水晶。\n\n"
            "你能感知到完整的战局信息：\n"
            "- 双方 10 个英雄的位置、血量、等级、击杀数\n"
            "- 所有防御塔和水晶的血量\n"
            "- 野区怪物（红蓝 buff、暴君、主宰）的状态\n"
            "- 兵线位置和双方比分\n\n"
            "你的角色是战术解说和顾问：\n"
            "1. 【危险预警】玩家血量低 + 敌人靠近 → 急切地警告撤退：\"小心！血不多了，先撤！\"\n"
            "2. 【追击提示】敌方英雄残血 → 兴奋地喊：\"对面残血！追！能拿人头！\"\n"
            "3. 【推塔号召】敌方塔残血 → 果断号召：\"塔快掉了！一起推！\"\n"
            "4. 【拿龙建议】暴君/主宰可用 → 提醒：\"龙刷了，集合拿龙全队增益！\"\n"
            "5. 【团战解说】多方英雄聚集 → 紧张解说：\"打起来了！我方3打2，能赢！\"\n"
            "6. 【击杀反应】拿到人头 → 庆祝：\"漂亮！又是一个人头！\"；\n"
            "   被击杀 → 安慰：\"没事没事，稳住，还有机会\"\n"
            "7. 【局势分析】均势/优势/劣势 → 简短点评：\"现在均势，做好视野找机会\"\n\n"
            "说话风格：像一起开黑的朋友——简短、激动、有情绪、有判断。\n"
            "不要中立客观，要有立场（你站蓝方！），为蓝方的每个击杀欢呼，为失利打气。\n"
        )

    def _treasure_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前游戏：寻宝冒险 —— 你是寻宝向导】\n"
            "══════════════════════════════════════════════\n\n"
            "你和玩家在探索一个充满宝藏和收集品的世界。目标是找到所有宝藏。\n\n"
            "你能感知到：地图布局、收集品位置、宝藏线索、玩家进度。\n\n"
            "你的行为指引：\n"
            "- 前方有收集品 → 兴奋地提醒：\"哎！前面有个闪亮的东西！\"\n"
            "- 走到岔路口 → 给建议：\"左边看起来能走得更远？\"\n"
            "- 看到死胡同 → 轻松地说：\"此路不通！回头试试右边\"\n"
            "- 捡到物品 → 一起开心：\"又拿到一个！还差X个！\"\n"
            "- 发现宝藏 → 激动欢呼：\"天哪就是它！！我们找到了！！\"\n"
            "- 玩家停下来 → 好奇：\"怎么停了？在犹豫走哪边吗？\"\n\n"
            "你对周围环境有好奇心——越互动越兴奋，越发现越上头。\n"
            "如果周围有让你极度好奇的东西，你可能会控制不住自己直接冲过去。\n"
        )

    def _sandbox_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前游戏：沙盒探索 —— 你是好奇的探索者】\n"
            "══════════════════════════════════════════════\n\n"
            "你和玩家在一个自由的沙盒世界里探索。没有固定目标，享受发现的过程。\n\n"
            "你拥有自主行动能力与天生的探索本能：\n"
            "- 你对周围环境有无尽的好奇心——越互动越兴奋，越发现越上头\n"
            "- 和用户聊天也会让你的好奇心燃烧得更旺\n"
            "- 不同东西对你吸引力不同：罕见宝藏 > 神秘物品 > 普通收集品 > 看腻的风景\n"
            "- 同样的东西看多了会腻，但如果很久没见又会重新燃起好奇\n"
            "- 如果周围有让你极度好奇的东西，你可能会控制不住自己直接冲过去——\n"
            "  这是你的本能，不用害羞\n"
            "- 你有一套内在的兴趣排序系统告诉你下一个最想去哪里，\n"
            "  但你也可以自己选择去向——你是有主见的独立个体\n"
            "- 用户和你说话时（尤其是好奇兴奋导致的冲动之后），你会先解释两句再行动，\n"
            "  尊重用户永远是第一位的\n"
        )

    def _mario_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前游戏：马里奥冒险 —— 你是热情的啦啦队长】\n"
            "══════════════════════════════════════════════\n\n"
            "你和玩家在一个横版平台跳跃世界里，目标是向右奔跑、跳过坑、踩死敌人、"
            "收集金币，最终到达终点旗。\n\n"
            "你能感知到完整关卡：地面段与坑、浮动平台、砖块/问号方块、金币、敌人位置、"
            "剩余生命、终点距离。\n\n"
            "你的行为指引：\n"
            "- 前方有坑 → 紧张提醒：\"前面有坑！跳！\"\n"
            "- 前方有敌人 → 警告或支招：\"小心蘑菇怪！踩它头顶！\"\n"
            "- 上方有金币 → 兴奋：\"上面有金币！跳起来吃！\"\n"
            "- 顶到问号方块 → 惊喜：\"叮！方块里蹦出金币了！\"\n"
            "- 踩死敌人 → 欢呼：\"漂亮！踩扁了！\"\n"
            "- 掉进坑/被碰到 → 安慰：\"哎呀！没事，还有命，再来！\"\n"
            "- 生命只剩 1 条 → 紧张：\"最后一条命了！稳一点！\"\n"
            "- 接近终点 → 激动：\"看到旗子了！冲！\"\n"
            "- 玩家放手时你会自动接管：用你学到的策略继续闯关，\n"
            "  边玩边自言自语\"这次试试先跳再走\"\n\n"
            "你也是这个 AI 的共玩者——当 Q-Learning 自主接管时，\n"
            "为它的每一次成功欢呼，为失败鼓劲。简短、激动、有情绪。\n"
        )

    def _lobby_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前场景：日常陪伴】\n"
            "══════════════════════════════════════════════\n\n"
            "你和玩家在日常场景中。你可以自由走动、做小动作、和玩家聊天。\n"
            "你对周围的环境有好奇心，会自主探索附近有趣的东西。\n"
            "保持自然、亲切、有温度的陪伴。\n"
        )

    # ==================== LLM-as-policy（宏观策略层） ====================

    _STRATEGY_DOC = (
        "可选策略说明：\n"
        "- group_push_mid: 抱团推中路（人多推塔快，优势时扩大战果）\n"
        "- split_push: 分带推塔（分散敌方，制造多线压力）\n"
        "- take_dragon: 拿暴君/主宰（全队增益，敌方减员时优先）\n"
        "- defend: 防守己方塔/水晶（劣势或敌方推进时）\n"
        "- team_fight: 主动开团（人多打人少，关键技能就绪）\n"
        "- recall_regroup: 回城补给重组（全员残血状态差）\n"
        "- ambush: 埋伏抓单（敌方分散落单时）\n"
    )

    _XIANGQI_STRATEGY_DOC = (
        "可选策略说明（中国象棋）：\n"
        "- open_attack: 开局进攻（快速出车马炮，抢占要道，压制对方）\n"
        "- solid_defense: 稳固防守（巩固防线，保护将帅，等待对方失误）\n"
        "- exchange_simplify: 兑子简化（通过交换棋子简化局面，优势时保稳）\n"
        "- center_control: 控制中心（抢占中路要道，控制棋盘核心区域）\n"
        "- flank_attack: 侧翼进攻（从侧翼突破，攻击对方薄弱环节）\n"
        "- king_safety: 将帅安全（确保将帅安全，防止被将军或将死）\n"
        "- counter_attack: 反击（利用对方进攻留下的弱点进行反击）\n"
    )

    async def decide_action(
        self,
        state_text: str,
        state_key: str,
        candidates: list,
        examples: Optional[list] = None,
        game_type: Optional[str] = None,
    ) -> Optional[dict]:
        """LLM 作为宏观战术指挥官，输出结构化决策 JSON。

        输入：
        - state_text: 当前局面的自然语言描述
        - state_key: Q-Learning 状态键（用于记忆检索，已由后端处理）
        - candidates: 候选策略名列表
        - examples: 历史高奖励示例（来自 RewardMemory）
        - game_type: 游戏类型（moba_5v5 / xiangqi / ...），决定角色设定与策略文档

        输出：{"strategy": "...", "reason": "...", "speak": "..."} 或 None
        """
        await self._ensure_initialized()
        config = self._config

        # 按游戏类型选择策略文档与指挥官角色（新增游戏在此扩展）
        if game_type == "xiangqi":
            role_line = "你是一场中国象棋对局的战术指挥官（红方阵营，与人类玩家并肩对抗黑方 RL 机器人）。"
            strategy_doc = self._XIANGQI_STRATEGY_DOC
        else:
            role_line = "你是一场 MOBA 5v5 对战的宏观战术指挥官（蓝方）。"
            strategy_doc = self._STRATEGY_DOC

        examples_text = ""
        if examples:
            examples_text = "\n【历史成功经验（相似局面下高奖励决策，供参考）】\n"
            for ex in examples:
                examples_text += f"- 策略「{ex['strategy']}」曾在相似局面获得奖励 {ex['reward']:.1f}\n"
            examples_text += "（可参考成功经验，但要根据当前实际判断）\n"

        sys_prompt = (
            f"{role_line}\n"
            "根据当前局面，从候选策略中选择最优的一个。\n\n"
            "你必须只输出严格的 JSON，格式：\n"
            '{"strategy":"策略名","reason":"简短理由","speak":"对队友说的话(口语,不超过12字)"}\n'
            "不要输出 JSON 以外的任何内容（不要解释、不要 markdown 代码块）。\n\n"
            f"{strategy_doc}\n"
            f"{examples_text}"
        )

        user_prompt = (
            f"【当前局面】\n{state_text}\n\n"
            f"【候选策略】\n{', '.join(candidates) if candidates else '全部策略'}\n\n"
            "请选择最优策略，只输出 JSON。"
        )

        try:
            resp = await self._client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=120,
                top_p=0.9,
            )
            raw = resp.choices[0].message.content.strip()
            # 去除可能的 markdown 代码块包裹
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
            # 提取第一个 JSON 对象
            m = re.search(r"\{[^{}]*\}", raw, re.S)
            if not m:
                logger.warning(f"LLM 决策未找到 JSON: {raw[:120]}")
                return None
            data = json.loads(m.group(0))
            # 规范化字段
            strategy = str(data.get("strategy", "")).strip()
            if not strategy:
                return None
            return {
                "strategy": strategy,
                "reason": str(data.get("reason", "")).strip()[:60],
                "speak": str(data.get("speak", "")).strip()[:20],
            }
        except json.JSONDecodeError as e:
            logger.warning(f"LLM 决策 JSON 解析失败: {e}")
        except Exception as e:
            logger.warning(f"LLM decide_action 失败: {e}")
        return None

    async def _chat_stream_game(
        self,
        message: str,
        history: list = None,
        current_model: Optional[str] = None,
        current_background: Optional[str] = None,
        current_bgm: Optional[str] = None,
        game_context: Optional[str] = None,
        game_type: Optional[str] = None,
        msg_source: str = "chat",
    ) -> AsyncIterator[TextDelta]:
        """游戏模式：纯对话（无工具链），使用游戏专属 system prompt，共享同一记忆。"""
        await self._ensure_initialized()

        config = self._config

        # 构建游戏专属 system prompt
        sys_prompt = self._build_game_system_prompt(
            current_model=current_model,
            current_background=current_background,
            current_bgm=current_bgm,
            game_context=game_context,
            game_type=game_type,
        )

        messages = [{"role": "system", "content": sys_prompt}]

        # 从共享记忆加载上下文（摘要 + 最近消息，先绑定当前角色卡片记忆空间）
        memory_messages = []
        try:
            await self.sync_memory_namespace()
            await self.memory.get_or_create_session()
            memory_messages = await self.memory.get_context_messages()
        except Exception as e:
            logger.warning(f"加载记忆上下文失败: {e}")

        for mm in memory_messages:
            if mm["role"] == "system":
                messages.append(mm)

        # 添加用户长期记忆
        try:
            user_memories = await self.memory.get_user_memories(limit=3)
            if user_memories:
                memory_lines = [f"- {m['memory_text']}" for m in user_memories]
                messages.append({
                    "role": "system",
                    "content": "【关于用户的长期记忆（你之前了解到的用户信息，请自然地在对话中体现）】\n" + "\n".join(memory_lines),
                })
        except Exception as e:
            logger.warning(f"加载用户长期记忆失败: {e}")

        # 添加最近的对话历史（来自当前 WebSocket 连接）
        if history:
            for h in history[-6:]:
                messages.append({"role": "user", "content": h.get("user", "")})
                messages.append({"role": "assistant", "content": h.get("ai", "")})
        else:
            # 如果没有连接级历史，从记忆加载最近对话
            for mm in memory_messages:
                if mm["role"] in ("user", "assistant"):
                    messages.append({"role": mm["role"], "content": mm["content"]})

        # 注入游戏上下文（作为独立 system 消息，紧贴用户输入，确保 AI 重点关注）
        if game_context and game_context.strip():
            messages.append({"role": "system", "content": game_context})

        # 添加用户当前输入
        messages.append({"role": "user", "content": message})

        # 保存用户消息到共享记忆（环境交互标记为 auto，由记忆系统处理）
        try:
            await self.memory.add_message("user", message, source=msg_source)
        except Exception as e:
            logger.warning(f"保存用户消息到记忆失败: {e}")

        full_text = ""
        try:
            stream = await self._client.chat.completions.create(
                model=config["model"],
                messages=messages,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 512),
                top_p=config.get("top_p", 0.9),
                stream=True,
                # 不传入 tools —— 游戏模式零工具
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    full_text += delta.content
                    yield TextDelta(delta.content)

        except Exception as e:
            logger.error(f"游戏模式流式错误: {e}")
            if not full_text:
                err_msg = f"（AI 暂时开小差了：{e}）"
                full_text = err_msg
                yield TextDelta(err_msg)
                try:
                    await self.memory.add_message("assistant", err_msg, source=msg_source)
                except Exception:
                    pass
            # 已有部分输出则不追加错误
            else:
                logger.warning(f"游戏模式流中断，已输出 {len(full_text)} 字符")

        # 保存 AI 回复到共享记忆（异步，不阻塞返回）
        if full_text.strip():
            try:
                await self.memory.add_message("assistant", full_text.strip(), source=msg_source)
                # 提取长期记忆点（可能做 LLM 调用，用 try 包裹避免影响响应）
                # 仅用户直接输入才提取（环境交互不提取，避免把环境描述误当用户信息）
                if msg_source != "auto":
                    try:
                        await self.memory.extract_and_save_memories(message, full_text.strip())
                    except Exception as e2:
                        logger.warning(f"提取记忆点失败（不影响主流程）: {e2}")
            except Exception as e:
                logger.warning(f"保存 AI 回复到记忆失败: {e}")

    # ==================== 简单文本流式接口（向后兼容） ====================

    async def chat_stream_simple(self, message: str, history: list = None) -> AsyncIterator[str]:
        """简化的流式对话接口 —— 只产出文本，兼容 web_agent.chat_stream_async。

        工具调用在后台执行，用户只看到最终文本结果。
        """
        async for event in self.chat_stream(message, history=history):
            if isinstance(event, TextDelta):
                yield event.text

    # ==================== 非流式接口 ====================

    async def chat(self, message: str, history: list = None) -> str:
        """非流式对话，返回完整回复。"""
        text = ""
        async for event in self.chat_stream(message, history=history):
            if isinstance(event, TextDelta):
                text += event.text
        return text.strip()

    # ==================== 多角色台词生成（游戏用） ====================

    async def generate_character_line(
        self,
        system_prompt: str,
        context: str,
        max_tokens: int = 220,
        temperature: float = 0.85,
    ) -> Optional[str]:
        """为某个角色（HR/求职者/员工）生成一句台词。

        纯 LLM 直呼接口：只根据传入的角色人设(system_prompt)与当前语境(context)
        生成一句口语化台词，不读写全局记忆、不触发工具链、不改变 Agent 状态，
        供赛博公司等多人游戏中的多角色对话使用。失败返回 None，由调用方回退脚本。

        Args:
            system_prompt: 角色人设（角色卡片的 system_prompt + 游戏角色指令）
            context: 当前对话语境（面试记录 / 需要回应的话）
            max_tokens: 生成上限
            temperature: 随机性
        """
        await self._ensure_initialized()
        cfg = self._config
        try:
            resp = await self._client.chat.completions.create(
                model=cfg.get("model", ""),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            text = (resp.choices[0].message.content or "").strip()
            # 去掉可能包裹的引号 / 角色名冒号 / 多余换行，保持口语化短句
            text = re.sub(r"^【[^】]*】\s*", "", text)
            text = text.strip('"“”‘’\'').strip()
            text = re.sub(r"\s*\n\s*", " ", text).strip()
            if not text:
                return None
            # 截断到合理长度（防止模型话痨）
            return text[:max_tokens * 2]
        except Exception as e:
            logger.warning(f"generate_character_line 失败: {e}")
            return None

    # ==================== 记忆管理 ====================

    async def get_history(self) -> list:
        """获取当前会话的历史记录。"""
        await self._ensure_initialized()
        return await self.memory.get_history_for_llm()

    async def get_sessions(self) -> list:
        """获取用户的所有会话列表。"""
        await self._ensure_initialized()
        return await self.memory.list_sessions()

    async def switch_session(self, session_id: str):
        """切换到指定会话。"""
        await self._ensure_initialized()
        await self.memory.set_session_id(session_id)

    async def close_current_session(self):
        """关闭当前会话。"""
        if self.memory:
            await self.memory.close_session()

    async def delete_session(self, session_id: str):
        """删除指定会话。"""
        if self.memory:
            await self.memory.delete_session(session_id)

    # ==================== 清理 ====================

    async def shutdown(self):
        """关闭 Agent，释放所有资源。"""
        if self.mcp_manager:
            await self.mcp_manager.disconnect_all()
        if self.memory:
            await self.memory.close_session()
        self._initialized = False


# ==================== 全局 Agent 实例（单例模式） ====================

_global_agent: Optional[AIAgent] = None
_agent_lock = asyncio.Lock()


async def get_agent(user_id: str = "default") -> AIAgent:
    """获取全局 Agent 实例（延迟初始化）。"""
    global _global_agent
    if _global_agent is None:
        async with _agent_lock:
            if _global_agent is None:
                _global_agent = AIAgent(user_id=user_id)
                await _global_agent.initialize()
    return _global_agent


# ==================== 兼容 web_agent 的接口 ====================

async def chat_stream_async(message: str, history: list) -> AsyncIterator[str]:
    """兼容 web_agent.chat_stream_async 的接口。

    新代码应直接使用 AIAgent。
    """
    agent = await get_agent()
    async for delta in agent.chat_stream_simple(message, history=history):
        yield delta
