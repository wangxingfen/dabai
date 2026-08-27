# 在线音乐技能（music）

自包含的在线点播技能：内置酷我 / 网易云免费曲库的聚合搜索与播放直链解析，
**不依赖任何本地/外部服务**。核心逻辑在根目录 `music_lib.py`，与 server.py 前端 API 共用同一实例。

## 工具
- music_search(keyword, limit?) —— 聚合搜歌（双源并行），返回带序号列表
- music_play(index | source+song_id, volume?) —— 解析直链并播放（__screen_command__）
- music_stop() —— 停止播放
- music_control(action, value?) —— 控制正在播放的音乐：pause 暂停 / resume 继续 / toggle 切换 / stop 停止 / volume 调音量（__screen_command__ control_music）
- music_lyric(index | source+song_id) —— 歌词 LRC
- music_billboard() —— 内置榜单（云音乐热歌/新歌/飙升/原创）
- music_playlist(keyword | playlist_id) —— 搜/开在线歌单（网易云）
- music_my_playlists() —— 查看用户自建歌单
- music_play_playlist(playlist_id | playlist_name) —— 顺序播放自建歌单（__screen_command__ play_playlist）
- music_status() —— 能力与用法提示

## 自建歌单
- 存储：根目录 `music_playlists.json`（{id, name, created, songs[]}）
- 前端「在线音乐」页面可创建/改名歌曲加入/移除；AI 通过 music_my_playlists /
  music_play_playlist 播放
- 播放歌单：前端收到 play_playlist 屏幕命令后按队列顺序播放，每首播放前
  通过 GET /api/music/resolve?source=&song_id= 现取直链

## 实现要点
- 搜索：酷我 `search.kuwo.cn/r.s`（失败回退 kw_token API）+ 网易云 web 搜索
- 直链：酷我 antiserver（OGG 优先 → mp3/aac 回退）；网易云 `outer/url` 302 解析
- 缓存：直链 15 分钟、搜索 10 分钟（TTLCache，None 也缓存防打爆失效接口）
- 最近一次搜索/歌单结果缓存在内存 `_last_results`，AI 报序号即可播放/取歌词

## 规则
- 听歌流程：search → 报序号给用户 → play(index)；换歌/安静时 stop。
- VIP / 版权受限歌曲拿不到直链时返回友好提示，主动换一首重试。
- 播放音量默认约 0.8（区别于低音量氛围 BGM 场景）。

## 已知限制
- 酷我单曲详情/榜单官方接口已加签名校验，这些能力由网易云承担。
- 网易云 VIP 歌曲 outer/url 拿不到 302 直链，会明确提示换歌。