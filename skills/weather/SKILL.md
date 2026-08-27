# 天气助手技能（weather）

用免费开放接口 wttr.in 查询任意城市的实时天气，无需 API key。

## 工具
- weather_check(city) —— 查询某城市的气温/天气现象/体感/湿度/风速

## 说明
网络不可用时返回降级提示，不影响其它能力。

## 文件
- skill.json —— 清单与工具定义
- skill.py —— 实现（HANDLERS 表）