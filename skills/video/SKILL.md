# 在线视频技能（video）

自包含的在线视频技能：聚合 B站/AcFun 关键词搜索、点播、连播队列与播放控制，
**能力全部内建在同目录 `video_lib.py`，无任何外部服务依赖**（删掉历史遗留的
video-hub 目录也不影响运行）。**画面投到角色身后的直播大屏**，带声音全屏播放，
播完自动接队列下一部。

## 工具
- video_search(keyword, platform?, sort?, limit?) —— 聚合搜视频（B站/AcFun），返回带序号列表
- video_play(index | query | url, platform?, sort?) —— 大屏立即播放（__screen_command__ play_video）
- video_queue(index | query | url) —— 加入连播队列，当前播完自动接上
- video_control(action, value?) —— pause / resume / seek / volume / mute / stop / next
- video_status() —— 当前播放、进度、音量与队列

## 架构
```
AI 调 video_play ──▶ skills/video/skill.py ──直调──▶ video_lib.py（同进程）
                         │  B站/AcFun 聚合搜索 + yt-dlp 直链解析 + 流注册
                         │  返回 __screen_command__ play_video{url:"/api/video_hub/..."}
                         ▼
server.py 检测标记 ──WebSocket screen_command──▶ 前端 handleScreenCommand
                         ▼
30_task_big_screen.js：大屏切「大白影院」全屏模式，<video> 带声播放
  - 流地址 = 主服务原生端点 /api/video_hub/*（server.py 调 video_lib 输出）：
    相对路径同源可用——手机/局域网不受 127.0.0.1 指向错误与混合内容策略影响
  - direct 流（音视频合一 mp4）：/api/video_hub/proxy?k= Range 透传可拖进度；
    relay 流（B站 DASH 高清 / AcFun HLS）：/api/video_hub/relay/<k> ffmpeg 实时合流，
    失败自动重试 ?t=1 转码兜底（源编码不兼容时）
  - 播完 POST /api/video_hub/api/ended → 取队列下一部继续播（连播）
  - 每 5 秒 POST /api/video_hub/api/report 上报进度 → video_status 可见
  - video_control 的 pause/seek/volume 等直接作用在大屏播放器上
```

skill.py 与 server.py 共享同一个 video_lib 模块实例（流注册表/队列状态互通）。
阻塞调用（yt-dlp/网络）全部经 asyncio.to_thread 执行，不卡服务事件循环。

## 依赖（主 Python 环境）
- requests、yt-dlp（已安装）
- ffmpeg 可选但强烈建议（relay 高清合流与转码兜底依赖；缺了仍能播 direct 流）

## 性能与健壮性
- 关键词直点（play query）：B 站走 flat 快速路径，3-5 秒起播（完整搜索要 15-25 秒）；
  候选逐个尝试，付费课/失效视频自动跳过，全失败回退完整搜索
- 直链/元数据缓存 1 小时；B 站 buvid cookie 预热缓解 412 风控
- **断流自动恢复**（前端看门狗 + 服务端配合，播放中断不再直接躺平）：
  - 播放中卡死 / 取流超时 / 流报错 → 自动进入恢复流程，大屏显示"🔄 正在自动恢复"
  - 卡死看门狗四重判定：缓冲无进展（readyState≤2 + timeupdate 停滞）／解码器停交付帧
    （rVFC 不再回调，专治"有声音但画面冻结"）／绘制停滞（帧还在走但没画上大屏）／
    无 rVFC 老浏览器像素探针兜底（时间前进但画面纹丝不动）
  - **换片/恢复一律新建全新的 <video> 元素**并彻底释放旧元素（停网络/解码器/事件监听）：
    长时间播放后浏览器解码管线可能卡死，复用一个卡死元素换 src 也救不回来
    （表现为音轨正常、画面永冻，换任何视频依旧卡死），新建元素让每次起播都是干净状态
  - 恢复阶梯：direct 同流快重载(Range 续播) → 强制重解析拿全新直链（服务重启 key 失效也能救）
    → 断点续播（direct 前端定位 / relay `?ss=` 服务端 ffmpeg 输入端定位）→ 从头重试 → 放弃回待机
  - 最多自动恢复 3 次，稳定播放 30 秒后计数重置；恢复成功提示"已自动恢复播放"
  - relay 直播流的 duration 是增长估计值，流中途断掉浏览器会误发 ended——
    以服务端解析的真实时长判断，没播完走恢复而不是误跳下一部；真播到片尾附近断流才接队列
- B 站搜索偶发 412 风控（等两三分钟自动恢复），此时可换 acfun

## 规则
- 看片流程：video_search → 报序号给用户 → video_play(index)；或直接 query 点播。
- 教育类/课程内容用 platform=bilibili 最全；热门内容 sort=hot。
- 解析失败/无结果：主动换关键词或换平台重试，别把错误原样丢给用户。
- 播视频前若在放音乐，前端会自动停掉 BGM，避免声音叠加。

## 已知限制
- 平台直链几小时过期（缓存 1 小时自动重解析）。
- relay 流为直播式不可拖进度；direct 流可拖。
- 大屏在游戏模式会隐藏，视频随之暂停、退出游戏后自动续播。
