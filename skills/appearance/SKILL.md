# 外观形象技能（appearance）

管理大白的 3D 造型与所处场景。

## 工具
- get_available_models() —— 列出可选角色模型（models/ 目录）
- get_available_backgrounds() —— 列出可选背景场景（backgrounds/ 目录）
- switch_character_model(model_name) —— 换装/换形象（精确文件名，含扩展名；"default" 恢复）
- switch_background_scene(bg_name) —— 换环境场景（精确文件名；"default" 恢复星空）

## 规则
1. 只有用户明确要求换装/换形象/变身时才调用，不要擅自更换；
2. 切换前先查询可用列表，文件名必须精确匹配，不要编造；
3. 模型文件名只是外观皮肤，不是身份——回复中不要提文件名。