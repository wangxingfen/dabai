# AI 画图技能（image_gen）

根据画面描述调用 settings.json 里配置的绘图模型生成图片（默认 SiliconFlow Kolors）。

## 工具
- image_gen_create(prompt, size?) —— 生成图片，保存到 web/generated/ 并返回 /generated/<文件名> 链接

## 配置（settings.json）
- images_base_url —— 兼容 OpenAI Images API 的端点（默认 https://api.siliconflow.cn/v1）
- images_model —— 模型名（默认 Kwai-Kolors/Kolors）
- images_api_key —— API Key（必填）

## 文件
- skill.json —— 清单与工具定义
- skill.py —— 实现（HANDLERS 表，asyncio.to_thread 跑同步生成）