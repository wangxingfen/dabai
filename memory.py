"""长期记忆模块 —— SQLite 持久化对话历史、会话管理、记忆检索与摘要。

特性：
- SQLite 本地存储，无需额外数据库服务
- 按 user_id 管理多用户会话
- 自动生成对话摘要（长对话压缩）
- 智能记忆检索：最近 N 条 + 相关历史摘要
- 支持异步操作（线程池执行同步 SQLite）
"""
import asyncio
import functools
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory")

BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "chat_memory.db"

# 摘要触发阈值：单会话超过此消息数后自动生成摘要
SUMMARY_THRESHOLD = 20
# 摘要间隔：每隔多少条消息生成一次摘要
SUMMARY_INTERVAL = 10
# 检索时返回的最近消息数
RECENT_MESSAGE_COUNT = 10
# 检索时返回的最大摘要数
MAX_SUMMARIES = 3


_db_lock = threading.Lock()
_db_local = threading.local()

def _get_db() -> sqlite3.Connection:
    """获取数据库连接（线程安全 + 连接复用 + 超时等待 + 自动重连）。

    每个线程复用同一连接（threading.local），避免多连接写冲突。
    写操作通过 _with_db_lock 串行化，彻底消除 'database is locked' 错误。
    自动检测连接是否已关闭并重建。
    """
    conn = getattr(_db_local, 'conn', None)
    # 检查连接是否有效（未关闭）
    if conn is not None:
        try:
            conn.execute("SELECT 1")
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            # 连接已关闭（如 _init_db 遗留的关闭状态），重建
            conn = None
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        _db_local.conn = conn
    return conn

def _with_db_lock(func):
    """装饰器：用全局锁串行化所有数据库写操作。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _db_lock:
            return func(*args, **kwargs)
    return wrapper


def _init_db():
    """初始化数据库表结构。"""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'default',
            namespace TEXT NOT NULL DEFAULT 'default',
            title TEXT DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tool_calls TEXT DEFAULT NULL,
            tool_results TEXT DEFAULT NULL,
            source TEXT NOT NULL DEFAULT 'chat',
            created_at REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_source
            ON messages(session_id, source);
        CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON sessions(user_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            range_start INTEGER NOT NULL,
            range_end INTEGER NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_summaries_session
            ON summaries(session_id, created_at DESC);

        -- 用户记忆：跨会话的长期记忆片段（按命名空间隔离，角色卡片互不串扰）
        CREATE TABLE IF NOT EXISTS user_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            namespace TEXT NOT NULL DEFAULT 'default',
            memory_text TEXT NOT NULL,
            source_session_id TEXT,
            importance REAL DEFAULT 0.5,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_user_memories
            ON user_memories(user_id, importance DESC);
    """)
    # 迁移：为旧数据库添加 source 字段
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN source TEXT NOT NULL DEFAULT 'chat'")
    except sqlite3.OperationalError:
        pass  # 字段已存在
    # 迁移：为旧数据库添加 namespace 字段（角色卡片独立记忆空间）
    for _tbl in ("sessions", "user_memories"):
        try:
            conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'")
        except sqlite3.OperationalError:
            pass  # 字段已存在
    # namespace 相关索引（必须在 ALTER 添加列之后再建，否则旧库会报 no such column）
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_sessions_ns
            ON sessions(user_id, namespace, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_user_memories_ns
            ON user_memories(user_id, namespace, importance DESC);
    """)
    conn.commit()
    # 注意：不关闭连接 — _get_db() 使用 threading.local() 缓存了连接，
    # 关闭会导致后续调用拿到已关闭的连接引发 "Cannot operate on a closed database"
    # 连接在进程退出时自动释放


# 模块加载时自动初始化
_init_db()


class ChatMemory:
    """对话记忆管理器。

    每个实例关联一个用户和一个会话，提供消息存储、检索和摘要功能。
    """

    def __init__(self, user_id: str = "default", session_id: str = None,
                 namespace: str = "default"):
        self.user_id = user_id
        self.session_id = session_id
        self.namespace = namespace  # 记忆命名空间：无卡片='default'，角色卡片='role_card:<card_id>'
        self._message_count = 0  # 当前会话消息计数缓存
        self._llm_client = None  # 可选：用于 LLM 驱动的摘要生成和记忆提取
        self._llm_model = None

    # ==================== 会话管理 ====================

    def set_llm_client(self, client, model_name: str):
        """设置 LLM 客户端，用于生成高质量摘要和提取长期记忆。

        Args:
            client: OpenAI AsyncOpenAI 客户端实例
            model_name: 模型名称
        """
        self._llm_client = client
        self._llm_model = model_name

    async def get_or_create_session(self, title: str = "") -> str:
        """获取或创建当前用户的活动会话。

        Returns:
            session_id
        """
        if self.session_id:
            return self.session_id

        @_with_db_lock
        def _do():
            conn = _get_db()
            now = time.time()
            # 查找近期活动会话（15 分钟内，仅限当前命名空间 —— 角色卡片记忆互不串扰）
            recent = now - 900
            row = conn.execute(
                "SELECT id FROM sessions WHERE user_id=? AND namespace=? AND is_active=1 AND updated_at>? "
                "ORDER BY updated_at DESC LIMIT 1",
                (self.user_id, self.namespace, recent),
            ).fetchone()
            if row:
                self.session_id = row["id"]
                conn.execute(
                    "UPDATE sessions SET updated_at=? WHERE id=?",
                    (now, self.session_id),
                )
                # 加载当前消息计数（环境交互消息不计入，与 add_message 保持一致）
                cnt = conn.execute(
                    "SELECT COUNT(*) as cnt FROM messages "
                    "WHERE session_id=? AND source != 'auto'",
                    (self.session_id,),
                ).fetchone()["cnt"]
                conn.commit()
                self._message_count = cnt
                return self.session_id

            # 创建新会话
            import uuid
            sid = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO sessions (id, user_id, namespace, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sid, self.user_id, self.namespace, title or f"对话 {datetime.now():%m-%d %H:%M}", now, now),
            )
            conn.commit()
            self.session_id = sid
            self._message_count = 0
            return sid

        return await asyncio.to_thread(_do)

    async def set_session_id(self, session_id: str):
        """手动设置会话 ID（用于恢复历史会话）。"""
        self.session_id = session_id
        # 更新会话活动时间并加载当前消息计数
        @_with_db_lock
        def _do():
            conn = _get_db()
            conn.execute(
                "UPDATE sessions SET updated_at=?, is_active=1 WHERE id=?",
                (time.time(), session_id),
            )
            cnt = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages "
                "WHERE session_id=? AND source != 'auto'",
                (session_id,),
            ).fetchone()["cnt"]
            conn.commit()
            return cnt
        self._message_count = await asyncio.to_thread(_do)

    async def session_belongs_to_namespace(self, session_id: str) -> bool:
        """判断会话是否属于当前命名空间（防止跨角色卡片切换/恢复串记忆）。"""
        if not session_id:
            return False
        def _do():
            conn = _get_db()
            row = conn.execute(
                "SELECT namespace FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
            conn.close()
            return bool(row and row["namespace"] == self.namespace)
        return await asyncio.to_thread(_do)

    async def list_sessions(self, limit: int = 20) -> list:
        """列出当前命名空间下用户的所有会话。"""
        def _do():
            conn = _get_db()
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at, is_active "
                "FROM sessions WHERE user_id=? AND namespace=? ORDER BY updated_at DESC LIMIT ?",
                (self.user_id, self.namespace, limit),
            ).fetchall()
            sessions = [dict(r) for r in rows]
            for s in sessions:
                s["created_at"] = datetime.fromtimestamp(s["created_at"]).isoformat()
                s["updated_at"] = datetime.fromtimestamp(s["updated_at"]).isoformat()
            conn.close()
            return sessions
        return await asyncio.to_thread(_do)

    async def update_title(self, title: str):
        """更新当前会话标题。"""
        if not self.session_id:
            return
        def _do():
            conn = _get_db()
            conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, self.session_id))
            conn.commit()
            conn.close()
        await asyncio.to_thread(_do)

    # ==================== 消息存储 ====================

    async def add_message(self, role: str, content: str,
                          tool_calls: list = None, tool_results: list = None,
                          source: str = "chat"):
        """添加一条消息到当前会话。

        Args:
            role: user / assistant / system / tool
            content: 消息内容
            tool_calls: tool_call 信息列表（assistant 消息可能携带）
            tool_results: 工具调用结果列表（tool 消息）
            source: 消息来源：'chat'=用户直接输入（大厅）、'game'=用户直接输入（游戏）、
                'auto'=环境交互（感知触发/自主行为等）。
                环境交互消息只做持久化存储，不计入消息计数、不进 LLM 上下文、
                不参与摘要——短期记忆只与用户直接输入有关，环境交互由记忆系统兜底存储。
        """
        if not self.session_id:
            await self.get_or_create_session()

        @_with_db_lock
        def _do():
            conn = _get_db()
            now = time.time()
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_results, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id, role, content,
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    json.dumps(tool_results, ensure_ascii=False) if tool_results else None,
                    source,
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (now, self.session_id),
            )
            conn.commit()
            # 环境交互消息不计入摘要计数（避免高频快照触发无意义的摘要生成）
            if source != "auto":
                self._message_count += 1

        await asyncio.to_thread(_do)

        # 检查是否需要生成摘要
        if self._message_count > 0 and self._message_count % SUMMARY_INTERVAL == 0:
            await self.maybe_summarize()

    async def maybe_summarize(self):
        """检查并在需要时生成对话摘要。"""
        # 使用缓存的 _message_count，避免 COUNT(*) 查询
        if self._message_count < SUMMARY_THRESHOLD:
            return

        # 检查最新的摘要覆盖范围
        def _get_last_summary_range():
            conn = _get_db()
            row = conn.execute(
                "SELECT range_end FROM summaries WHERE session_id=? ORDER BY range_end DESC LIMIT 1",
                (self.session_id,),
            ).fetchone()
            conn.close()
            return row["range_end"] if row else 0

        last_end = await asyncio.to_thread(_get_last_summary_range)

        def _get_messages_since(start_id):
            conn = _get_db()
            rows = conn.execute(
                "SELECT id, role, content FROM messages "
                "WHERE session_id=? AND id > ? AND source != 'auto' ORDER BY id",
                (self.session_id, start_id),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

        msgs = await asyncio.to_thread(_get_messages_since, last_end)
        if len(msgs) >= SUMMARY_INTERVAL * 2:
            await self._generate_summary(msgs)

    async def _generate_summary(self, messages: list):
        """生成对话摘要并存储。优先使用 LLM，降级为关键词提取。

        每次生成摘要时，会传入前一次摘要作为参考，让 LLM 做增量式补充，
        避免重复描述已经总结过的内容。
        """
        if not messages:
            return

        # 构建简要对话文本
        dialog_text = ""
        for m in messages[-SUMMARY_INTERVAL * 2:]:
            role_tag = "用户" if m["role"] == "user" else "AI"
            # 过滤纯游戏事件消息，避免摘要混入大量系统通知
            content = m["content"]
            if m["role"] == "system" and content.startswith("[游戏]"):
                dialog_text += f"事件: {content}\n"
            else:
                dialog_text += f"{role_tag}: {content[:200]}\n"

        # 获取前一次摘要，传递给 LLM 做增量式补充
        prev_summary = ""
        if self._llm_client and self._llm_model:
            prev_summary = await self._get_latest_summary_text()

        # 优先使用 LLM 生成语义摘要
        if self._llm_client and self._llm_model:
            summary = await self._llm_summarize(dialog_text, prev_summary)
        else:
            summary = self._keyword_summarize(dialog_text)

        # 截断过长摘要
        if len(summary) > 300:
            summary = summary[:297] + "..."

        @_with_db_lock
        def _save():
            conn = _get_db()
            conn.execute(
                "INSERT INTO summaries (session_id, summary_text, range_start, range_end, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    summary,
                    messages[0]["id"],
                    messages[-1]["id"],
                    time.time(),
                ),
            )
            conn.commit()
            logger.info(f"会话 {self.session_id} 生成摘要: {summary[:100]}...")

        await asyncio.to_thread(_save)

    async def _get_latest_summary_text(self) -> str:
        """获取当前会话最新的一条摘要文本。"""
        def _do():
            conn = _get_db()
            row = conn.execute(
                "SELECT summary_text FROM summaries WHERE session_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (self.session_id,),
            ).fetchone()
            conn.close()
            return row["summary_text"] if row else ""
        return await asyncio.to_thread(_do)

    async def _llm_summarize(self, dialog_text: str, prev_summary: str = "") -> str:
        """使用 LLM 生成语义摘要。

        Args:
            dialog_text: 本次需要总结的对话文本
            prev_summary: 前一次摘要，用于增量式补充，避免重复
        """
        system_msg = (
            "你是对话摘要助手。请用1-3句中文提炼以下对话的**新发生**的关键内容："
            "话题变迁、用户透露的新信息、重要决策或事件。"
            "忽略日常寒暄和重复内容。"
        )
        if prev_summary:
            user_msg = (
                f'【前一段对话的摘要】\n{prev_summary}\n\n'
                f'【新对话内容】\n{dialog_text}\n\n'
                f'请只补充前一段摘要中**没有**的新内容。如果新内容和前一段摘要完全重复或没有新信息，请回复"同上"。'
                f'只输出补充内容，不要加任何前缀。'
            )
        else:
            user_msg = f'请总结以下对话：\n{dialog_text}'

        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            if not content or content == "同上":
                return prev_summary  # 无新内容，复用前次摘要
            return content
        except Exception as e:
            logger.warning(f"LLM 摘要生成失败，降级为关键词提取: {e}")
            return self._keyword_summarize(dialog_text)

    @staticmethod
    def _keyword_summarize(dialog_text: str) -> str:
        """降级方案：从对话中提取关键话题和事件作为摘要。

        优先提取用户消息中的实质内容，过滤寒暄和语气词。
        """
        lines = dialog_text.strip().split("\n")
        topics = []
        filler_words = {"嗯", "啊", "哦", "好", "好的", "行", "可以", "对", "是的", "没错",
                        "哈哈", "嘿嘿", "嘻嘻", "呵呵", "谢谢", "不客气", "知道了", "明白了"}

        for line in lines:
            # 提取角色和内容
            if ": " not in line:
                continue
            _, content = line.split(": ", 1)
            content = content.strip()
            # 跳过太短或纯寒暄的
            if len(content) < 6 or content in filler_words:
                continue
            # 跳过游戏事件（[游戏] 前缀）
            if content.startswith("[游戏]"):
                continue
            # 截取有意义的前半部分
            if len(content) > 80:
                content = content[:77] + "..."
            topics.append(content)

        if not topics:
            return "(暂无重要话题)"
        # 最多保留6条，用分号连接
        return "；".join(topics[:6])

    # ==================== 记忆检索 ====================

    async def get_context_messages(self, recent_n: int = None) -> list:
        """获取最近的消息列表，结合摘要提供上下文。

        Returns:
            OpenAI 消息格式的列表: [{"role": ..., "content": ...}, ...]
        """
        if not self.session_id:
            return []

        recent_n = recent_n or RECENT_MESSAGE_COUNT

        def _do():
            conn = _get_db()
            messages = []

            # 仅在消息数超过阈值时才加载摘要（小于阈值的会话不可能有摘要）
            if self._message_count >= SUMMARY_THRESHOLD:
                summary_rows = conn.execute(
                    "SELECT summary_text FROM summaries WHERE session_id=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (self.session_id, MAX_SUMMARIES),
                ).fetchall()
                if summary_rows:
                    summary_texts = [s["summary_text"] for s in summary_rows]
                    # 合并多条摘要，限制总长度避免浪费 token
                    combined = "；".join(summary_texts)
                    if len(combined) > 600:
                        combined = combined[:597] + "..."
                    summary_text = "以下是之前的对话摘要：\n" + combined
                    messages.append({
                        "role": "system",
                        "content": summary_text,
                    })

            # 获取最近消息（排除环境交互消息：短期记忆只与用户直接输入有关，
            # 环境交互内容由记忆系统兜底存储，不进 LLM 上下文避免重复内容僵化）
            msg_rows = conn.execute(
                "SELECT role, content, tool_calls, tool_results FROM messages "
                "WHERE session_id=? AND source != 'auto' "
                "ORDER BY created_at DESC LIMIT ?",
                (self.session_id, recent_n),
            ).fetchall()[::-1]  # 反转为正序

            conn.close()

            for row in msg_rows:
                msg = {"role": row["role"], "content": row["content"]}
                if row["tool_calls"]:
                    try:
                        msg["tool_calls"] = json.loads(row["tool_calls"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                if row["tool_results"]:
                    try:
                        msg["tool_results"] = json.loads(row["tool_results"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                messages.append(msg)

            return messages

        return await asyncio.to_thread(_do)

    async def get_history_for_llm(self, max_messages: int = 10) -> list:
        """获取用于 LLM 上下文的历史消息（简化格式）。

        Returns:
            [{"user": "...", "ai": "..."}, ...]  兼容现有的 history 格式
        """
        if not self.session_id:
            return []

        def _do():
            conn = _get_db()
            rows = conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id=? AND source != 'auto' "
                "ORDER BY created_at DESC LIMIT ?",
                (self.session_id, max_messages * 2),
            ).fetchall()[::-1]
            conn.close()

            history = []
            pending_user = None
            for row in rows:
                if row["role"] == "user":
                    pending_user = row["content"]
                elif row["role"] == "assistant" and pending_user is not None:
                    history.append({"user": pending_user, "ai": row["content"]})
                    pending_user = None
            return history[-max_messages:]  # 只保留最近 N 轮

        return await asyncio.to_thread(_do)

    # ==================== 用户长期记忆 ====================

    async def save_user_memory(self, memory_text: str, importance: float = 0.5):
        """保存用户相关的长期记忆片段。

        若已存在相同内容的记忆，则只更新重要性和访问时间，避免重复入库。

        Args:
            memory_text: 记忆内容
            importance: 重要性评分 (0.0-1.0)
        """
        @_with_db_lock
        def _do():
            conn = _get_db()
            now = time.time()
            existing = conn.execute(
                "SELECT id FROM user_memories WHERE user_id=? AND namespace=? AND memory_text=?",
                (self.user_id, self.namespace, memory_text),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE user_memories SET importance=?, last_accessed=?, source_session_id=? "
                    "WHERE id=?",
                    (max(importance, 0.5), now, self.session_id, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO user_memories (user_id, namespace, memory_text, source_session_id, "
                    "importance, created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (self.user_id, self.namespace, memory_text, self.session_id, importance, now, now),
                )
            conn.commit()

        await asyncio.to_thread(_do)

    async def get_user_memories(self, limit: int = 5) -> list:
        """获取当前命名空间下用户最重要的长期记忆片段。"""
        @_with_db_lock
        def _do():
            conn = _get_db()
            now = time.time()
            rows = conn.execute(
                "SELECT id, memory_text, importance FROM user_memories "
                "WHERE user_id=? AND namespace=? ORDER BY importance DESC, last_accessed DESC LIMIT ?",
                (self.user_id, self.namespace, limit),
            ).fetchall()
            # 更新访问时间
            if rows:
                ids = [r["id"] for r in rows]
                conn.execute(
                    "UPDATE user_memories SET last_accessed=? WHERE id IN ({})".format(
                        ",".join("?" * len(ids))
                    ),
                    [now] + ids,
                )
            conn.commit()
            return [dict(r) for r in rows]

        return await asyncio.to_thread(_do)

    async def extract_and_save_memories(self, user_message: str, ai_response: str):
        """从对话中提取用户偏好/信息并保存为长期记忆。

        优先使用 LLM 智能提取，降级为简单关键词匹配。
        """
        if self._llm_client and self._llm_model:
            await self._llm_extract_memories(user_message, ai_response)
        else:
            self._keyword_extract_memories(user_message, ai_response)

    async def _llm_extract_memories(self, user_message: str, ai_response: str):
        """使用 LLM 智能提取用户记忆点。

        注意：必须只提取「用户本人明确陈述或确认」的真实信息。
        AI 在角色扮演/对话中对用户的描述、推测、虚构设定一律不得入库。
        """
        prompt = (
            f'你是记忆提取助手。从对话中提取**用户本人明确说出或明确确认**的关于自己的真实信息'
            f'（偏好、身份、职业、习惯、经历、真实生活细节）。\n\n'
            f'【严格规则】\n'
            f'1. 只能提取用户自己明确陈述、或明确承认的信息。\n'
            f'2. AI 对用户的描述、外貌/能力设定、推测或想象（例如"你的指尖泛起微光"）一律忽略，'
            f'除非用户亲口承认那是事实。\n'
            f'3. AI 的愿望、感受、行动不是用户的信息（例如"我想和你一起探索"是 AI 的想法，不提取）。\n'
            f'4. 虚构场景、游戏剧情、比喻修辞、AI 编造的环境描写一律不提取。\n'
            f'5. 若用户没有透露任何个人信息，只回复"无"。\n\n'
            f'【对话】\n'
            f'用户: {user_message[:500]}\n'
            f'AI: {ai_response[:300]}\n\n'
            f'请只输出用户明确透露的信息，每行一条、简洁短句，不要编号或加前缀；'
            f'拿不准的信息不要输出。如果没有，只回复"无"。'
        )
        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": "你是严格的记忆提取助手。只提取用户本人明确陈述或确认的真实信息，绝不采信 AI 对用户的描述、推测或虚构设定。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            if content and content != "无":
                for line in content.split("\n"):
                    line = line.strip()
                    if line and len(line) > 2:
                        await self.save_user_memory(line, importance=0.7)
                        logger.info(f"LLM 提取用户记忆: {line}")
        except Exception as e:
            logger.warning(f"LLM 记忆提取失败，降级为关键词提取: {e}")
            self._keyword_extract_memories(user_message, ai_response)

    def _keyword_extract_memories(self, user_message: str, ai_response: str):
        """降级方案：关键词匹配提取用户信息。"""
        patterns = {
            "我叫": "用户自称",
            "我是": "用户身份",
            "我喜欢": "用户偏好-喜欢",
            "我不喜欢": "用户偏好-不喜欢",
            "我的": "用户信息",
            "我住": "用户住址",
            "我工作": "用户工作",
            "我从事": "用户职业",
        }

        async def _save():
            for keyword, category in patterns.items():
                if keyword in user_message:
                    idx = user_message.find(keyword)
                    snippet = user_message[idx:idx + 80].strip()
                    memory = f"[{category}] {snippet}"
                    await self.save_user_memory(memory, importance=0.7)
                    logger.info(f"提取用户记忆: {memory}")

        # 在后台线程执行
        import asyncio as _asyncio
        try:
            _asyncio.create_task(_save())
        except Exception:
            pass

    # ==================== 清理 ====================

    async def close_session(self):
        """关闭当前会话（标记为非活跃）。"""
        if not self.session_id:
            return
        @_with_db_lock
        def _do():
            conn = _get_db()
            conn.execute(
                "UPDATE sessions SET is_active=0, updated_at=? WHERE id=?",
                (time.time(), self.session_id),
            )
            conn.commit()
        await asyncio.to_thread(_do)
        self.session_id = None
        self._message_count = 0

    async def delete_session(self, session_id: str = None):
        """删除指定会话及其所有消息。"""
        sid = session_id or self.session_id
        if not sid:
            return
        @_with_db_lock
        def _do():
            conn = _get_db()
            conn.execute("DELETE FROM summaries WHERE session_id=?", (sid,))
            conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
            conn.commit()
        await asyncio.to_thread(_do)
        if sid == self.session_id:
            self.session_id = None
            self._message_count = 0
