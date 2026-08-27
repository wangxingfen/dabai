# 系统文件搜索技能（sys_search）

大白的基础能力：全盘找文件，不再只靠猜路径。

## 工具

| 工具 | 用途 |
|------|------|
| sys_find(name?, ext?, dir?, kind?, min_size_mb?, max_size_mb?, newer_than_days?, ...) | 多条件全盘查找文件/目录：名称关键词或通配符、扩展名、目录、大小、最近修改天数；kind 可选 file/dir/both |
| sys_recent(days?, dir?, ext?, ...) | 最近修改的文件（按修改时间倒序） |
| sys_locate(name) | 定位 PATH 里的可执行程序（python、ffmpeg、codex…） |

## 规则

1. 用户要找文件/目录时，**先 sys_find 拿真实完整路径**，绝不猜路径；
2. 找"最近改过什么"用 sys_recent；找程序路径用 sys_locate；
3. 默认跳过 Windows/AppData/缓存/依赖目录等噪音，需要进系统目录才设 include_sys=true；
4. 全部只读；带时间预算和结果上限，超时返回部分结果并提示缩小范围；
5. 找到后要打开/操作文件，仍走 shell_run / find_file 的既有安全规则。
