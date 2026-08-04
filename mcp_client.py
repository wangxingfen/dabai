"""MCP (Model Context Protocol) 客户端 —— 连接 MCP 服务器，获取工具列表并执行工具调用。

支持：
- stdio 传输（子进程方式）
- SSE 传输（HTTP Server-Sent Events）
- HTTP 传输（Streamable HTTP，直接 HTTP POST）
- 自动初始化握手
- 工具列表获取与缓存
- 异步工具调用
- 多 MCP 服务器并发管理
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from urllib.parse import urljoin
from typing import Any, Optional, Union

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

logger = logging.getLogger("mcp_client")

# MCP JSON-RPC 版本
MCP_VERSION = "2024-11-05"
CLIENT_NAME = "dabai-mcp-client"
CLIENT_VERSION = "1.0.0"


class MCPError(Exception):
    """MCP 协议错误"""
    pass


class MCPServerConnection:
    """单个 MCP 服务器的连接管理。

    通过子进程 stdio 与 MCP 服务器通信，使用 JSON-RPC 2.0 协议。
    """

    def __init__(self, name: str, command: str, args: list = None,
                 env: dict = None, cwd: str = None, timeout: float = 30.0):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._initialized = False
        self._tools: list = []
        self._server_info: dict = {}
        self._request_id = 0
        self._pending: dict = {}  # request_id -> Future
        self._read_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """启动 MCP 服务器子进程并完成初始化握手。"""
        if self._initialized:
            return

        try:
            env = {**os.environ, **(self.env or {})}
        except Exception:
            env = None  # 继承当前环境
        try:
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.cwd,
                text=False,  # binary mode, we'll decode manually
            )
        except FileNotFoundError:
            raise MCPError(
                f"MCP 服务器 [{self.name}] 启动失败：找不到命令 '{self.command}'，"
                f"请确保已安装对应工具（如 npx、uvx、python）"
            )
        except Exception as e:
            raise MCPError(f"MCP 服务器 [{self.name}] 启动失败: {e}")

        # 启动后台读取任务
        self._read_task = asyncio.create_task(self._read_loop())

        # 发送 initialize 请求
        result = await self._send_request("initialize", {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {},  # 声明客户端支持工具调用
            },
            "clientInfo": {
                "name": CLIENT_NAME,
                "version": CLIENT_VERSION,
            },
        })
        self._server_info = result
        logger.info(f"MCP [{self.name}] 初始化成功: {result}")

        # 发送 initialized 通知
        await self._send_notification("notifications/initialized", {})
        self._initialized = True

        # 获取工具列表
        await self.refresh_tools()

    async def _read_loop(self):
        """后台持续读取 MCP 服务器的 stdout 输出（JSON-RPC 消息，换行分隔）。"""
        try:
            loop = asyncio.get_event_loop()
            while self._process and self._process.poll() is None:
                try:
                    # 以 4KB 块读取，避免逐字节读取的低效
                    chunk = await loop.run_in_executor(
                        None, self._process.stdout.read, 4096
                    )
                except Exception:
                    break

                if not chunk:
                    # EOF：子进程已退出
                    break

                # 按换行符分割，处理多行消息
                for line in chunk.decode("utf-8").split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        asyncio.create_task(self._handle_message(msg))
                    except json.JSONDecodeError:
                        logger.warning(f"MCP [{self.name}] 无法解析响应: {line[:200]}")
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"MCP [{self.name}] 读取循环异常: {e}")
        finally:
            # 子进程已退出或连接断开 → 取消所有等待中的请求
            if self._pending:
                logger.warning(
                    f"MCP [{self.name}] 连接断开，取消 {len(self._pending)} 个等待中的请求"
                )
                for rid, future in list(self._pending.items()):
                    if not future.done():
                        future.set_exception(
                            MCPError(f"MCP [{self.name}] 进程已退出")
                        )

    async def _handle_message(self, msg: dict):
        """处理收到的 JSON-RPC 消息。"""
        rid = msg.get("id")
        if rid is not None and rid in self._pending:
            # 这是对之前请求的响应
            future = self._pending.pop(rid)
            if "error" in msg:
                future.set_exception(MCPError(
                    f"MCP [{self.name}] 错误 (code={msg['error'].get('code')}): "
                    f"{msg['error'].get('message', 'Unknown')}"
                ))
            else:
                future.set_result(msg.get("result"))
        else:
            # 服务器主动推送的通知/请求
            logger.debug(f"MCP [{self.name}] 收到通知: {msg.get('method', 'unknown')}")

    async def _send_request(self, method: str, params: dict = None) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        self._request_id += 1
        rid = self._request_id
        request = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params or {},
        }
        future = asyncio.get_event_loop().create_future()
        self._pending[rid] = future

        await self._send_raw(request)

        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise MCPError(f"MCP [{self.name}] 请求超时: {method}")

    async def _send_notification(self, method: str, params: dict = None):
        """发送 JSON-RPC 通知（无需响应）。"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        await self._send_raw(notification)

    async def _send_raw(self, data: dict):
        """将 JSON-RPC 消息写入子进程 stdin。"""
        if not self._process or not self._process.stdin:
            raise MCPError(f"MCP [{self.name}] 未连接")
        line = json.dumps(data, ensure_ascii=False) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await asyncio.get_event_loop().run_in_executor(None, self._process.stdin.flush)

    async def refresh_tools(self):
        """获取并缓存 MCP 服务器的工具列表。"""
        if not self._initialized:
            await self.connect()

        result = await self._send_request("tools/list", {})
        self._tools = result.get("tools", [])
        logger.info(f"MCP [{self.name}] 获取到 {len(self._tools)} 个工具: "
                    f"{[t.get('name','?') for t in self._tools]}")
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用 MCP 服务器上的指定工具。

        Args:
            tool_name: 工具名称
            arguments: 工具参数（dict）

        Returns:
            工具执行结果
        """
        if not self._initialized:
            await self.connect()

        logger.info(f"MCP [{self.name}] 调用工具: {tool_name}({arguments})")
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        # MCP 返回的结果格式: { content: [{ type: "text", text: "..." }] }
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") == "resource":
                    texts.append(json.dumps(item.get("resource", {}), ensure_ascii=False))
                else:
                    texts.append(str(item))
            return "\n".join(texts) if texts else str(result)
        return str(result)

    def get_tools_as_openai_format(self) -> list:
        """将 MCP 工具转换为 OpenAI function calling 格式。"""
        openai_tools = []
        for tool in self._tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": input_schema.get("type", "object"),
                        "properties": input_schema.get("properties", {}),
                        "required": input_schema.get("required", []),
                    },
                },
            })
        return openai_tools

    async def disconnect(self):
        """断开与 MCP 服务器的连接。"""
        if self._read_task:
            self._read_task.cancel()
            self._read_task = None
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
        self._initialized = False
        logger.info(f"MCP [{self.name}] 已断开")


class MCPServerConnectionSSE:
    """基于 HTTP SSE 的 MCP 服务器连接。

    通过 HTTP GET 建立 SSE 长连接接收消息，HTTP POST 发送 JSON-RPC 请求。
    """

    def __init__(self, name: str, url: str, headers: dict = None,
                 timeout: float = 30.0):
        if not _HAS_AIOHTTP:
            raise MCPError(
                f"MCP SSE [{name}] 需要 aiohttp 库，请执行: pip install aiohttp"
            )
        self.name = name
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self._session: "aiohttp.ClientSession" = None
        self._sse_task: Optional[asyncio.Task] = None
        self._post_url: Optional[str] = None
        self._initialized = False
        self._tools: list = []
        self._server_info: dict = {}
        self._request_id = 0
        self._pending: dict = {}  # request_id -> Future
        self._endpoint_ready = asyncio.Event()
        self._response_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self):
        """建立 SSE 连接并完成初始化握手。"""
        if self._initialized:
            return

        # 清理上一次失败的残留状态
        self._endpoint_ready.clear()
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
            self._sse_task = None
        if self._session:
            await self._session.close()
            self._session = None

        self._session = aiohttp.ClientSession(
            headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                **self.headers,
            },
            timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
        )

        # 启动 SSE 长连接读取循环
        self._sse_task = asyncio.create_task(self._sse_loop())

        # 等待 endpoint 事件（携带 POST 地址）
        try:
            await asyncio.wait_for(self._endpoint_ready.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            await self._cleanup()
            raise MCPError(f"MCP SSE [{self.name}] 等待 endpoint 超时，URL: {self.url}")

        if not self._post_url:
            await self._cleanup()
            raise MCPError(f"MCP SSE [{self.name}] 未收到 endpoint 事件")

        logger.info(f"MCP SSE [{self.name}] endpoint: {self._post_url}")

        # 发送 initialize 请求
        try:
            result = await self._send_request("initialize", {
                "protocolVersion": MCP_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            })
        except MCPError:
            await self._cleanup()
            raise

        self._server_info = result
        logger.info(f"MCP SSE [{self.name}] 初始化成功: {result}")

        # 发送 initialized 通知
        await self._send_notification("notifications/initialized", {})
        self._initialized = True

        # 获取工具列表
        await self.refresh_tools()

    async def _cleanup(self):
        """清理 SSE 连接资源。"""
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
            self._sse_task = None
        if self._session:
            await self._session.close()
            self._session = None
        self._post_url = None
        self._endpoint_ready.clear()
        self._initialized = False

    async def _sse_loop(self):
        """后台 SSE 长连接读取循环。"""
        try:
            async with self._session.get(self.url) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    # 尝试解析服务端错误信息
                    try:
                        err = json.loads(body)
                        code = err.get("Code", "")
                        msg = err.get("Message", body[:500])
                    except json.JSONDecodeError:
                        code = ""
                        msg = body[:500]

                    if code == "CAExited":
                        raise MCPError(
                            f"MCP SSE [{self.name}] 云端服务实例启动失败，"
                            f"请检查 ModelScope 上的 MCP 服务配置是否正确: {msg[:300]}"
                        )
                    raise MCPError(
                        f"MCP SSE [{self.name}] 连接失败: HTTP {resp.status}"
                        + (f", Code={code}, {msg[:300]}" if code else f", {msg[:300]}")
                    )
                logger.info(f"MCP SSE [{self.name}] SSE 流已建立")

                event_type = None
                data_buffer = ""

                async for raw_line in resp.content:
                    try:
                        line = raw_line.decode("utf-8").rstrip("\n").rstrip("\r")
                    except UnicodeDecodeError:
                        continue

                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                    elif line.startswith("data: "):
                        data_buffer += line[6:]
                    elif line == "" and data_buffer:
                        # 空行表示一个事件结束
                        await self._handle_sse_event(event_type, data_buffer.strip())
                        event_type = None
                        data_buffer = ""
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"MCP SSE [{self.name}] SSE 连接异常: {e}")
            # 将所有 pending 请求标记为失败
            for rid, future in list(self._pending.items()):
                if not future.done():
                    future.set_exception(
                        MCPError(f"MCP SSE [{self.name}] 连接断开")
                    )

    async def _handle_sse_event(self, event_type: Optional[str], data: str):
        """处理单个 SSE 事件。"""
        if event_type == "endpoint":
            # 服务器告知 POST 消息的目标地址（可能是相对路径）
            raw = data.strip()
            if raw.startswith(("http://", "https://")):
                self._post_url = raw
            else:
                self._post_url = urljoin(self.url, raw)
            self._endpoint_ready.set()
            return

        # 尝试解析为 JSON-RPC 消息（接受任意 event type，不限于 "message"）
        if data:
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                return  # 非 JSON 数据，忽略

            rid = msg.get("id")
            if rid is not None:
                # 放入响应队列，供 _send_raw 消费
                await self._response_queue.put(msg)
            else:
                logger.debug(
                    f"MCP SSE [{self.name}] 收到通知: {msg.get('method', 'unknown')}"
                )

    async def _send_request(self, method: str, params: dict = None) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        self._request_id += 1
        rid = self._request_id
        request = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params or {},
        }
        future = asyncio.get_event_loop().create_future()
        self._pending[rid] = future

        await self._send_raw(request)

        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise MCPError(f"MCP SSE [{self.name}] 请求超时: {method}")

    async def _send_notification(self, method: str, params: dict = None):
        """发送 JSON-RPC 通知（无需响应）。"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        await self._send_raw(notification)

    async def _send_raw(self, data: dict):
        """通过 HTTP POST 发送 JSON-RPC 消息，SSE 流返回响应。"""
        if not self._post_url:
            raise MCPError(f"MCP SSE [{self.name}] 未获取到 POST endpoint")

        try:
            async with self._session.post(
                self._post_url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status not in (200, 202):
                    body = await resp.text()
                    raise MCPError(
                        f"MCP SSE [{self.name}] POST 失败: HTTP {resp.status}, {body[:200]}"
                    )

                # 有些实现直接在 POST 响应中返回 JSON-RPC 结果
                if resp.status == 200 and data.get("id") is not None:
                    try:
                        resp_data = await resp.json()
                        if "id" in resp_data:
                            await self._response_queue.put(resp_data)
                    except Exception:
                        pass  # 不是 JSON，从 SSE 流获取
        except aiohttp.ClientError as e:
            raise MCPError(f"MCP SSE [{self.name}] 发送失败: {e}")

        # 等待 SSE 流返回响应（如果 POST 响应已放入队列，这里会立即返回）
        if data.get("id") is not None:
            rid = data["id"]
            deadline = asyncio.get_event_loop().time() + self.timeout
            try:
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    msg = await asyncio.wait_for(
                        self._response_queue.get(), timeout=remaining
                    )
                    if msg.get("id") == rid:
                        future = self._pending.pop(rid, None)
                        if future and not future.done():
                            if "error" in msg:
                                future.set_exception(MCPError(
                                    f"MCP SSE [{self.name}] 错误 "
                                    f"(code={msg['error'].get('code')}): "
                                    f"{msg['error'].get('message', 'Unknown')}"
                                ))
                            else:
                                future.set_result(msg.get("result"))
                        return
                    else:
                        # 不是给这个请求的，放回去
                        await self._response_queue.put(msg)
            except asyncio.TimeoutError:
                self._pending.pop(rid, None)
                raise MCPError(f"MCP SSE [{self.name}] 等待响应超时")

    async def refresh_tools(self):
        """获取并缓存 MCP 服务器的工具列表。"""
        if not self._initialized:
            await self.connect()

        result = await self._send_request("tools/list", {})
        self._tools = result.get("tools", [])
        logger.info(f"MCP SSE [{self.name}] 获取到 {len(self._tools)} 个工具: "
                    f"{[t.get('name', '?') for t in self._tools]}")
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用 MCP 服务器上的指定工具。"""
        if not self._initialized:
            await self.connect()

        logger.info(f"MCP SSE [{self.name}] 调用工具: {tool_name}({arguments})")
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        # MCP 返回的结果格式: { content: [{ type: "text", text: "..." }] }
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") == "resource":
                    texts.append(json.dumps(item.get("resource", {}), ensure_ascii=False))
                else:
                    texts.append(str(item))
            return "\n".join(texts) if texts else str(result)
        return str(result)

    def get_tools_as_openai_format(self) -> list:
        """将 MCP 工具转换为 OpenAI function calling 格式。"""
        openai_tools = []
        for tool in self._tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": input_schema.get("type", "object"),
                        "properties": input_schema.get("properties", {}),
                        "required": input_schema.get("required", []),
                    },
                },
            })
        return openai_tools

    async def disconnect(self):
        """断开 SSE 连接。"""
        if self._sse_task:
            self._sse_task.cancel()
            self._sse_task = None
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False
        self._post_url = None
        self._endpoint_ready.clear()
        logger.info(f"MCP SSE [{self.name}] 已断开")


class MCPServerConnectionHTTP:
    """基于 HTTP (Streamable HTTP) 的 MCP 服务器连接。

    通过 HTTP POST 发送 JSON-RPC 请求，响应直接在 HTTP 响应中返回。
    支持可选的 Mcp-Session-Id 会话管理。
    """

    def __init__(self, name: str, url: str, headers: dict = None,
                 timeout: float = 30.0):
        if not _HAS_AIOHTTP:
            raise MCPError(
                f"MCP HTTP [{name}] 需要 aiohttp 库，请执行: pip install aiohttp"
            )
        self.name = name
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self._session: "aiohttp.ClientSession" = None
        self._session_id: Optional[str] = None
        self._initialized = False
        self._tools: list = []
        self._server_info: dict = {}
        self._request_id = 0

    async def connect(self):
        """发送 initialize 请求完成握手。"""
        if self._initialized:
            return

        if self._session:
            await self._session.close()

        self._session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                **self.headers,
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )

        try:
            result = await self._send_request("initialize", {
                "protocolVersion": MCP_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            })
        except MCPError:
            await self._cleanup()
            raise

        self._server_info = result
        logger.info(f"MCP HTTP [{self.name}] 初始化成功: {result}")

        await self._send_notification("notifications/initialized", {})
        self._initialized = True

        await self.refresh_tools()

    async def _cleanup(self):
        """清理 HTTP 连接资源。"""
        if self._session:
            await self._session.close()
            self._session = None
        self._session_id = None
        self._initialized = False

    async def _send_request(self, method: str, params: dict = None) -> dict:
        """发送 JSON-RPC 请求并等待 HTTP 响应。"""
        self._request_id += 1
        rid = self._request_id
        request = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params or {},
        }

        headers = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        try:
            async with self._session.post(
                self.url,
                json=request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                # 保存 session ID
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid

                if resp.status not in (200, 202):
                    body = await resp.text()
                    raise MCPError(
                        f"MCP HTTP [{self.name}] 请求失败: HTTP {resp.status}, {body[:300]}"
                    )

                # 202 Accepted 表示服务器稍后通过 SSE 返回（兼容处理）
                if resp.status == 202:
                    logger.warning(
                        f"MCP HTTP [{self.name}] 收到 202 Accepted，"
                        f"但当前不支持异步响应，跳过请求 {method}"
                    )
                    return {}

                # 直接解析 JSON-RPC 响应
                try:
                    msg = await resp.json()
                except Exception:
                    body = await resp.text()
                    raise MCPError(
                        f"MCP HTTP [{self.name}] 响应不是有效 JSON: {body[:200]}"
                    )

                if "error" in msg:
                    raise MCPError(
                        f"MCP HTTP [{self.name}] 错误 "
                        f"(code={msg['error'].get('code')}): "
                        f"{msg['error'].get('message', 'Unknown')}"
                    )

                return msg.get("result", {})

        except aiohttp.ClientError as e:
            raise MCPError(f"MCP HTTP [{self.name}] 发送失败: {e}")

    async def _send_notification(self, method: str, params: dict = None):
        """发送 JSON-RPC 通知（无需响应）。"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }

        headers = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        try:
            async with self._session.post(
                self.url,
                json=notification,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid
        except aiohttp.ClientError as e:
            logger.warning(f"MCP HTTP [{self.name}] 通知发送失败: {e}")

    async def refresh_tools(self):
        """获取并缓存 MCP 服务器的工具列表。"""
        if not self._initialized:
            await self.connect()

        result = await self._send_request("tools/list", {})
        self._tools = result.get("tools", [])
        logger.info(f"MCP HTTP [{self.name}] 获取到 {len(self._tools)} 个工具: "
                    f"{[t.get('name', '?') for t in self._tools]}")
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用 MCP 服务器上的指定工具。"""
        if not self._initialized:
            await self.connect()

        logger.info(f"MCP HTTP [{self.name}] 调用工具: {tool_name}({arguments})")
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") == "resource":
                    texts.append(json.dumps(item.get("resource", {}), ensure_ascii=False))
                else:
                    texts.append(str(item))
            return "\n".join(texts) if texts else str(result)
        return str(result)

    def get_tools_as_openai_format(self) -> list:
        """将 MCP 工具转换为 OpenAI function calling 格式。"""
        openai_tools = []
        for tool in self._tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": input_schema.get("type", "object"),
                        "properties": input_schema.get("properties", {}),
                        "required": input_schema.get("required", []),
                    },
                },
            })
        return openai_tools

    async def disconnect(self):
        """断开 HTTP 连接。"""
        await self._cleanup()
        logger.info(f"MCP HTTP [{self.name}] 已断开")


# 连接类型联合别名
MCPConnection = Union[MCPServerConnection, MCPServerConnectionSSE, MCPServerConnectionHTTP]


class MCPManager:
    """管理多个 MCP 服务器连接。

    从 settings.json 读取 MCP 服务器配置，统一管理连接、工具发现和调用。
    """

    def __init__(self, config: dict = None):
        self._connections: dict[str, MCPConnection] = {}
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name
        self._config = config or {}
        self._initialized = False

    def configure(self, config: dict):
        """从配置字典加载 MCP 服务器列表。

        配置格式:
        {
            "mcpServers": {
                "filesystem": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
                    "env": {},
                    "cwd": null
                },
                "remote_api": {
                    "type": "sse",
                    "url": "https://example.com/mcp/sse",
                    "headers": {"Authorization": "Bearer xxx"}
                },
                "tavily": {
                    "type": "http",
                    "url": "https://mcp.tavily.com/mcp/",
                    "headers": {}
                }
            }
        }
        """
        self._config = config
        # 不清除已有连接，允许增量添加

    async def initialize(self):
        """初始化所有配置的 MCP 服务器连接。"""
        if self._initialized:
            return

        servers = self._config.get("mcpServers", {})
        if not servers:
            logger.debug("没有配置 MCP 服务器，跳过初始化")
            self._initialized = True
            return

        tasks = []
        for name, cfg in servers.items():
            # 支持 "enabled" (默认 true) 和 "disabled" 两种写法
            if cfg.get("disabled", False) or cfg.get("enabled", True) is False:
                logger.debug(f"MCP [{name}] 已禁用，跳过")
                continue

            srv_type = cfg.get("type", "stdio")

            if srv_type == "sse":
                # --- SSE 传输 ---
                url = cfg.get("url", "")
                if not url:
                    logger.warning(f"MCP [{name}] SSE 类型缺少 'url' 字段，跳过")
                    continue
                headers = cfg.get("headers", {})
                timeout = cfg.get("timeout", 30.0)
                try:
                    conn = MCPServerConnectionSSE(
                        name=name, url=url, headers=headers, timeout=timeout,
                    )
                except MCPError as e:
                    logger.warning(f"MCP [{name}] 跳过: {e}")
                    continue

            elif srv_type == "stdio":
                # --- stdio 传输 ---
                if "command" not in cfg:
                    logger.warning(f"MCP [{name}] stdio 类型缺少 'command' 字段，跳过")
                    continue
                conn = MCPServerConnection(
                    name=name,
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=cfg.get("env"),
                    cwd=cfg.get("cwd"),
                    timeout=cfg.get("timeout", 30.0),
                )

            elif srv_type == "http":
                # --- HTTP (Streamable HTTP) 传输 ---
                url = cfg.get("url", "")
                if not url:
                    logger.warning(f"MCP [{name}] http 类型缺少 'url' 字段，跳过")
                    continue
                headers = cfg.get("headers", {})
                timeout = cfg.get("timeout", 30.0)
                try:
                    conn = MCPServerConnectionHTTP(
                        name=name, url=url, headers=headers, timeout=timeout,
                    )
                except MCPError as e:
                    logger.warning(f"MCP [{name}] 跳过: {e}")
                    continue

            else:
                logger.warning(f"MCP [{name}] 未知传输类型 '{srv_type}'，跳过")
                continue

            self._connections[name] = conn
            tasks.append(self._connect_server(name, conn))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(self._connections.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"MCP [{name}] 初始化失败: {result}")

        self._initialized = True

    async def _connect_server(self, name: str, conn: MCPConnection):
        """连接单个 MCP 服务器并注册其工具。"""
        await conn.connect()
        tools = conn.get_tools_as_openai_format()
        for tool in tools:
            tool_name = tool["function"]["name"]
            self._tool_to_server[tool_name] = name
        logger.info(f"MCP [{name}] 已注册 {len(tools)} 个工具")

    async def get_all_tools(self) -> list:
        """获取所有 MCP 服务器的工具列表（OpenAI 兼容格式）。"""
        if not self._initialized:
            await self.initialize()

        all_tools = []
        for conn in self._connections.values():
            all_tools.extend(conn.get_tools_as_openai_format())
        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用指定工具。根据工具名自动路由到正确的 MCP 服务器。"""
        if not self._initialized:
            await self.initialize()

        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            available = list(self._tool_to_server.keys())
            raise MCPError(
                f"未找到工具 '{tool_name}'，"
                f"已知 MCP 工具: {available[:20]}"
            )

        conn = self._connections.get(server_name)
        if not conn:
            raise MCPError(f"MCP 服务器 '{server_name}' 未连接")

        logger.info(f"MCP 路由: [{tool_name}] → [{server_name}]")
        result = await conn.call_tool(tool_name, arguments)
        logger.info(f"MCP [{tool_name}] 返回 {len(result)} 字符")
        return result

    def has_tool(self, tool_name: str) -> bool:
        """检查指定工具是否已注册。"""
        return tool_name in self._tool_to_server

    def has_mcp_tools(self) -> bool:
        """检查是否有可用的 MCP 工具。"""
        return len(self._tool_to_server) > 0

    async def disconnect_all(self):
        """断开所有 MCP 服务器连接。"""
        for conn in self._connections.values():
            await conn.disconnect()
        self._connections.clear()
        self._tool_to_server.clear()
        self._initialized = False
