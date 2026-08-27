# 工作区切换（workspace_switch）

## 是什么
切换/收藏/浏览**全局工作区**的技能。与前端工作区面板（web/js/audio/32_workspace_ui.ts）**完全同一套 API**，绝不另写一套持久化 —— 切换后 DSH / Codex / OpenCode / shell 全部围绕新工作区执行。

## 触发场景
- 用户说「切换到 XX 项目」「把工作区设为 XX」「去 XX 目录干活」
- 用户说「收藏这个工作区」「看看收藏了哪些工作区」「取消收藏 XX」
- 用户说「当前工作区在哪」「有哪些盘符/目录可选」「浏览一下 XX 目录」

## 后端同源 API（server.py）
| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | /api/workspace | 当前工作区（cwd / codex_work_dir / bridge_cwd） |
| POST | /api/workspace | 设置/切换工作区（写 codex_config.json + harness_bridge.json，热同步执行器） |
| GET | /api/workspace/roots | 可选根目录（盘符 + 桌面/下载/文档/视频/音乐/图片 + 当前） |
| GET | /api/workspace/list?path= | 逐级下钻浏览子目录 |
| GET | /api/workspaces | 已收藏工作区列表 + 当前 |
| POST | /api/workspaces | 收藏一个路径（去重，不切换） |
| DELETE | /api/workspaces | 移出收藏（不删磁盘目录） |
| POST | /api/workspaces/{path}/activate | 激活已收藏工作区（等价于切换） |

> 协议：server 以 **HTTPS** 运行（cert.pem/key.pem 存在即 use_https，server.py:5754），本技能统一走 `https://127.0.0.1:8000`，自签名证书不校验（与前端浏览器访问一致）。

## 工具
| 工具 | 参数 | 作用 |
| --- | --- | --- |
| workspace_get | （无） | 获取当前全局工作区路径 |
| workspace_set | path | 设置/切换全局工作区 |
| workspace_roots | （无） | 列出可选根目录 |
| workspace_list | path? | 列出指定目录下的子目录（下钻浏览） |
| workspaces_list | （无） | 获取已收藏工作区列表 + 当前 |
| workspaces_add | path | 收藏一个路径 |
| workspaces_remove | path | 移出收藏（不删磁盘目录） |
| workspaces_activate | path | 激活已收藏工作区 |

## 使用要点
1. **切换前先确认路径存在**：`workspace_set` 后端会校验目录存在，不存在返回 404。
2. **切换后所有智能体围绕新工作区执行**：DSH / Codex / OpenCode / shell 的 EXECUTOR.cwd 都会同步。
3. **收藏 ≠ 切换**：`workspaces_add` 只加入收藏列表，不改变当前工作区；要切换用 `workspace_set` 或 `workspaces_activate`。
4. **移出收藏不删磁盘**：`workspaces_remove` 只是不再收藏，目录本身不动。
5. **浏览目录**：`workspace_list` 不带 path 返回根目录，带 path 返回其子目录，可逐级下钻。
6. **与前端一致**：所有数据读写都走 server.py 的 /api/workspace* 接口，前端面板和本技能看到的是同一份状态。

## 边界
- 只操作工作区配置（codex_config.json / harness_bridge.json / workspace_saved.json），不读写项目文件。
- 不删除任何磁盘目录。
- 服务未启动时接口调用会失败，如实告知用户即可。
