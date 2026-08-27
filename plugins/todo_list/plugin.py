"""TODO 任务清单插件入口（风格 A —— 继承 Plugin 基类）。

能力总览（均通过 OpenAI function calling 工具暴露给大白）：
- todo_breakdown  ：解析复杂自然语言任务 → 自动分解关键任务与子任务 + 生成执行计划；
- todo_create     ：按参数直接创建任务（含子任务、优先级、截止时间）；
- todo_plan       ：按任务/子任务依赖关系生成执行计划（拓扑排序）；
- todo_list       ：查询任务（按状态/优先级/关键词过滤，附进度概览）；
- todo_get        ：查看单个任务与子任务详情；
- todo_update     ：更新任务（待办/进行中/已完成状态、优先级、截止时间）；
- todo_subtask    ：更新子任务状态（父任务状态自动汇总）；
- todo_remind     ：设置/清除提醒（一次性 + 每天/每周/每隔N天 重复提醒）；
- todo_delete     ：删除任务。

提醒由后台线程（scheduler.py）触发：到期提醒与逾期提醒都会写入服务事件缓冲，
同时投递给已注册的 delivery 回调（推送到 harness 健康事件的示例实现）。
REST API 在 on_load 时挂载到大白 FastAPI app（见 api.py），路径前缀 /api/todo/*。
业务层在 service.py，其余模块可直接 import 复用。
"""
from __future__ import annotations

import logging

from harness import Plugin

logger = logging.getLogger('todo_list.plugin')


class TodoListPlugin(Plugin):
    name = 'todo_list'
    title = 'TODO 任务清单'
    version = '1.0.0'
    description = ('接收复杂任务并自动分解为可执行的子任务：支持优先级（高/中/低）、'
                   '截止时间、任务状态（待办/进行中/已完成）、依赖执行计划，'
                   '提供定时与重复提醒，并对外提供 /api/todo/* REST API（JSON）。')
    author = 'dabai'
    prompt = (
        '【TODO 任务清单插件已启用】当用户描述一项整体性/多步骤任务时，'
        '优先调用 todo_breakdown 把描述自动分解为关键任务与有序子任务并生成执行计划；'
        '用户提到优先级、截止时间、进度或提醒时，用 todo_create / todo_update /'
        ' todo_remind 管理任务；涉及依赖或先后步骤时调用 todo_plan 查看执行顺序。'
        '工具名都带 todo_ 前缀。子任务状态更新后父任务状态会自动汇总。'
    )

    # ---------- 生命周期 ----------

    def on_load(self) -> None:
        from plugins.todo_list.service import TodoService
        from plugins.todo_list.scheduler import ReminderScheduler

        self.service = TodoService()
        self.scheduler = ReminderScheduler(self.service)
        self.scheduler.add_delivery_callback(self._deliver_reminder)
        self.scheduler.start()
        self._register_http_api()
        self.manager.harness.record(
            'plugin', f'todo_list 已加载（任务 {self.service.stats()["total"]} 项，'
                      f'提醒调度器已启动）')

    def on_unload(self) -> None:
        if getattr(self, 'scheduler', None) is not None:
            self.scheduler.stop()
        self.manager.harness.record('plugin', 'todo_list 已卸载（提醒调度器已停止）')

    # ---------- 提醒投递回调（可扩展到语音/前端推送） ----------

    def _deliver_reminder(self, task, event) -> None:
        """提醒触发的默认投递：写入日志 + 记录到 harness 健康事件。

        想推送到 3D 形象语音/前端 Toast，可在子类覆写本方法，
        或继续用 scheduler.add_delivery_callback 追加新的投递回调。
        """
        logger.info('[TODO提醒] %s', event.get('message', ''))
        try:
            self.manager.harness.record('todo_reminder', event.get('message', ''))
        except Exception as e:  # noqa: BLE001
            logger.warning('todo_list 提醒写入 harness 事件失败：%s', e)

    # ---------- REST API 挂载（供其它模块/前端调用，JSON 交互） ----------

    def _register_http_api(self) -> None:
        try:
            from plugins.todo_list.api import build_router
            from server import app
            if not getattr(app.state, 'todo_list_api_registered', False):
                app.include_router(build_router(self.service))
                app.state.todo_list_api_registered = True
                self.manager.harness.record(
                    'plugin', 'todo_list REST API 已挂载：/api/todo/*')
            else:
                logger.info('todo_list REST API 已存在（跳过重复挂载）')
        except Exception as e:  # noqa: BLE001 —— 服务未就绪时 REST 不可用，工具照常工作
            logger.warning('todo_list REST API 挂载失败（仅启用工具能力）：%s', e)

    # ---------- 工具定义 ----------

    def define_tools(self) -> list:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'todo_breakdown',
                    'description': ('解析一句/一段复杂任务描述，自动识别关键任务与子任务，'
                                    '并按先后依赖生成可执行计划；默认直接落库创建。'),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_text': {
                                'type': 'string',
                                'description': ('用户的复杂任务描述，例如：'
                                                '"明天前整理项目资料，先收集需求文档，'
                                                '然后整理成PPT，最后做评审汇报（加急）。"'),
                            },
                            'create': {
                                'type': 'boolean',
                                'description': '是否直接创建成任务，默认 true',
                            },
                        },
                        'required': ['task_text'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'todo_create',
                    'description': '创建一个任务（可带子任务、优先级、截止时间、依赖）。',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'title': {'type': 'string', 'description': '任务标题'},
                            'description': {'type': 'string', 'description': '任务描述'},
                            'priority': {
                                'type': 'string', 'enum': ['high', 'medium', 'low'],
                                'description': '优先级：高/中/低',
                            },
                            'status': {
                                'type': 'string', 'enum': ['todo', 'in_progress', 'done'],
                                'description': '状态：待办/进行中/已完成，默认待办',
                            },
                            'due_date': {
                                'type': 'string',
                                'description': ('截止时间，如 "2026-08-23 18:00"、'
                                                '"明天"、"周五18点"'),
                            },
                            'subtasks': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': '子任务标题列表（按顺序自动建立先后依赖）',
                            },
                            'depends_on': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': '依赖的任务 id 列表',
                            },
                        },
                        'required': ['title'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'todo_plan',
                    'description': '根据任务/子任务依赖关系生成执行计划（建议先做哪个，按顺序列出）。',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_id': {
                                'type': 'string',
                                'description': '只规划某个任务下的子任务时填任务 id；缺省规划全部',
                            },
                        },
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'todo_list',
                    'description': '查询任务清单，可按状态/优先级/关键词过滤，返回进度概览。',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'status': {
                                'type': 'string', 'enum': ['todo', 'in_progress', 'done'],
                                'description': '按状态过滤（待办/进行中/已完成）',
                            },
                            'priority': {
                                'type': 'string', 'enum': ['high', 'medium', 'low'],
                                'description': '按优先级过滤',
                            },
                            'keyword': {'type': 'string', 'description': '按关键词搜索标题/描述'},
                        },
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'todo_get',
                    'description': '查看单个任务的详情（含子任务、优先级、截止时间、提醒）。',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_id': {'type': 'string', 'description': '任务 id'},
                        },
                        'required': ['task_id'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'todo_update',
                    'description': '更新任务：状态（待办/进行中/已完成）、优先级、截止时间等。',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_id': {'type': 'string', 'description': '任务 id'},
                            'status': {
                                'type': 'string', 'enum': ['todo', 'in_progress', 'done'],
                            },
                            'priority': {
                                'type': 'string', 'enum': ['high', 'medium', 'low'],
                            },
                            'due_date': {'type': 'string', 'description': '新截止时间'},
                            'title': {'type': 'string', 'description': '新标题'},
                        },
                        'required': ['task_id'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'todo_subtask',
                    'description': '更新某个子任务的状态/优先级（父任务状态会自动汇总）。',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_id': {'type': 'string', 'description': '父任务 id'},
                            'subtask_id': {'type': 'string', 'description': '子任务 id'},
                            'status': {
                                'type': 'string', 'enum': ['todo', 'in_progress', 'done'],
                            },
                        },
                        'required': ['task_id', 'subtask_id'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'todo_remind',
                    'description': ('设置任务提醒（一次性或每天/每周/每隔N天重复），'
                                    '或清除提醒。'),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_id': {'type': 'string', 'description': '任务 id'},
                            'at': {
                                'type': 'string',
                                'description': ('提醒时刻，如 "18:30"、"明天9点"、'
                                                '"周五下午2点"'),
                            },
                            'repeat': {
                                'type': 'string', 'enum': ['none', 'hourly', 'daily', 'weekly'],
                                'description': '重复周期（none=一次性，默认）',
                            },
                            'repeat_every': {
                                'type': 'integer',
                                'description': '周期倍数，如每隔2天提醒 → daily + 2',
                            },
                            'clear': {
                                'type': 'boolean',
                                'description': 'true 表示清除提醒',
                            },
                        },
                        'required': ['task_id'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'todo_delete',
                    'description': '删除一个任务（连同子任务，其它任务的依赖引用随之清理）。',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_id': {'type': 'string', 'description': '任务 id'},
                        },
                        'required': ['task_id'],
                    },
                },
            },
        ]

    # ---------- 工具执行 ----------

    async def execute_tool(self, name: str, arguments: dict) -> str:
        if name == 'todo_breakdown':
            return self._do_breakdown(arguments)
        if name == 'todo_create':
            return self._do_create(arguments)
        if name == 'todo_plan':
            return self._do_plan(arguments)
        if name == 'todo_list':
            return self._do_list(arguments)
        if name == 'todo_get':
            return self._do_get(arguments)
        if name == 'todo_update':
            return self._do_update(arguments)
        if name == 'todo_subtask':
            return self._do_subtask(arguments)
        if name == 'todo_remind':
            return self._do_remind(arguments)
        if name == 'todo_delete':
            return self._do_delete(arguments)
        from harness import PluginError
        raise PluginError(f'todo_list 未实现工具 {name}')

    # ---------- 工具逻辑（同步方法，execute_tool 中逐个分发） ----------

    def _do_breakdown(self, args) -> str:
        from plugins.todo_list.parser import decompose
        text = str(args.get('task_text') or args.get('text') or '').strip()
        if not text:
            return 'todo_breakdown 缺少 task_text（任务描述）'
        plan = decompose(text)
        if not plan.get('subtasks'):
            return '未能从描述中分解出可执行子任务，请换一种说法：\n' + plan['goal']
        created = ''
        if bool(args.get('create', True)):
            result = self.service.import_plan(plan)
            task = result['task']
            created = f'已创建任务 {task["id"]}（标题：{task["title"]}）\n'
            lines = []
            for i, idx in enumerate(result['execution_order'], 1):
                sub = next(s for s in result['subtasks'] if s['id'] == idx)
                lines.append(f'  {i}. {sub["title"]}（优先级 {sub["priority"]}）')
            return (created + '执行计划（按依赖排序）：\n' + '\n'.join(lines)
                    + '\n使用 todo_list / todo_get 查看，用 todo_update / todo_subtask 更新进度。')
        lines = []
        for i, idx in enumerate(plan.get('execution_order') or [], 1):
            sub = plan['subtasks'][idx]
            lines.append(f'  {i}. {sub["title"]}（优先级 {sub["priority"]}）')
        return ('解析结果（未落库，create=true 可创建）：\n' + '执行计划：\n'
                + '\n'.join(lines))

    def _do_create(self, args) -> str:
        title = str(args.get('title') or '').strip()
        if not title:
            return 'todo_create 缺少 title'
        try:
            task = self.service.create_task(
                title=title,
                description=str(args.get('description') or ''),
                priority=args.get('priority'),
                status=args.get('status'),
                due_date=args.get('due_date'),
                depends_on=args.get('depends_on'),
            )
            subtask_titles = args.get('subtasks') or []
            order = []
            prev = None
            for i, st in enumerate(subtask_titles):
                sub = self.service.add_subtask(task['id'], title=str(st).strip(),
                                               depends_on=[prev] if prev else None)
                order.append(sub['id'])
                prev = sub['id']  # 子任务默认按先后顺序建立依赖
            extra = f'（{len(order)} 个子任务，顺序执行）' if order else ''
            return f'已创建任务 {task["id"]}：{task["title"]}，优先级 {task["priority"]}，' \
                   f'状态 {task["status"]}' + (f'，截止 {task["due_date"]}' if task['due_date'] else '') \
                   + extra
        except KeyError as e:
            return f'todo_create 失败：{e}'

    def _do_plan(self, args) -> str:
        plan = self.service.compute_execution_plan()
        if not plan['sequence']:
            return '当前没有待执行的工作项。'
        lines = []
        for i, item in enumerate(plan['sequence'], 1):
            lines.append(f'  {i}. [{item["priority"]}] {item["title"]}'
                         f'（{"子任务" if item["kind"] == "subtask" else "任务"}）')
        head = ('⚠️ 依赖存在环路，顺序仅供参考' if plan.get('cycle')
                else f'执行计划（{len(plan["sequence"])} 项依赖序）')
        return head + '：\n' + '\n'.join(lines)

    def _do_list(self, args) -> str:
        items = self.service.list_tasks(status=args.get('status'),
                                        priority=args.get('priority'),
                                        keyword=args.get('keyword'))
        if not items:
            return '当前没有符合条件的任务。'
        status_name = {'todo': '待办', 'in_progress': '进行中', 'done': '已完成'}
        pri_name = {'high': '高', 'medium': '中', 'low': '低'}
        lines = [f'任务清单（共 {len(items)} 项）：']
        for t in items:
            done = sum(1 for s in t['subtasks'] if s['status'] == 'done')
            sub_info = f'（子任务 {done}/{len(t["subtasks"])}）' if t['subtasks'] else ''
            lines.append(f'  [{status_name[t["status"]]}/{pri_name[t["priority"]]}] '
                         f'{t["id"]} {t["title"]}'
                         + (f' 截止 {t["due_date"]}' if t['due_date'] else '')
                         + sub_info)
        return '\n'.join(lines)

    def _do_get(self, args) -> str:
        task = self.service.get_task(str(args.get('task_id') or '').strip())
        if task is None:
            return '任务不存在'
        status_name = {'todo': '待办', 'in_progress': '进行中', 'done': '已完成'}
        pri_name = {'high': '高', 'medium': '中', 'low': '低'}
        lines = [f'任务 {task["id"]}：{task["title"]}',
                 f'  状态：{status_name[task["status"]]}  优先级：{pri_name[task["priority"]]}',
                 f'  截止：{task["due_date"] or "未设置"}',
                 f'  描述：{task["description"] or "-"}']
        if task.get('reminder'):
            r = task['reminder']
            repeat_name = {'none': '一次性', 'hourly': '每小时',
                           'daily': '每天', 'weekly': '每周'}
            lines.append(f'  提醒：{r["at"]}（{repeat_name.get(r["repeat"], r["repeat"])}'
                         + (f' ×{r["repeat_every"]}' if r['repeat'] != 'none' else '')
                         + f'，已触发 {r["count"]} 次）')
        if task['subtasks']:
            lines.append('  子任务：')
            for s in task['subtasks']:
                mark = {'todo': '○', 'in_progress': '◐', 'done': '●'}[s['status']]
                lines.append(f'    {mark} {s["id"]} {s["title"]}'
                             f'（{pri_name[s["priority"]]}'
                             + (f'，截止 {s["due_date"]}' if s.get('due_date') else '') + '）')
        return '\n'.join(lines)

    def _do_update(self, args) -> str:
        task_id = str(args.get('task_id') or '').strip()
        changes = {}
        for key in ('title', 'priority', 'status', 'due_date', 'description'):
            if args.get(key) is not None:
                changes[key] = args[key]
        if not changes:
            return 'todo_update 没有需要更新的字段'
        try:
            task = self.service.update_task(task_id, **changes)
        except KeyError as e:
            return f'todo_update 失败：{e}'
        return (f'任务 {task_id} 已更新：{task["title"]}，'
                f'状态 {task["status"]}，优先级 {task["priority"]}'
                + (f'，截止 {task["due_date"]}' if task['due_date'] else ''))

    def _do_subtask(self, args) -> str:
        task_id = str(args.get('task_id') or '').strip()
        sub_id = str(args.get('subtask_id') or '').strip()
        status = str(args.get('status') or args.get('subtask_status') or '').strip()
        try:
            sub = self.service.update_subtask(task_id, sub_id, status=status)
        except KeyError as e:
            return f'todo_subtask 失败：{e}'
        task = self.service.get_task(task_id)
        return (f'子任务 {sub_id}「{sub["title"]}」状态已改为 {sub["status"]}，'
                f'父任务状态自动汇总为 {task["status"]}')

    def _do_remind(self, args) -> str:
        task_id = str(args.get('task_id') or '').strip()
        if bool(args.get('clear')):
            try:
                result = self.service.clear_reminder(task_id)
            except KeyError as e:
                return f'todo_remind 失败：{e}'
            return f'任务 {task_id} 的提醒已清除'
        at = str(args.get('at') or '').strip()
        repeat = str(args.get('repeat') or 'none').strip()
        every = args.get('repeat_every') or 1
        if not at:
            return 'todo_remind 需要 at（提醒时刻），或用 clear=true 清除提醒'
        try:
            result = self.service.set_reminder(task_id, at=at, remind_type=None,
                                               repeat=repeat, repeat_every=every)
        except KeyError as e:
            return f'todo_remind 失败：{e}'
        if not result.get('ok'):
            return f'todo_remind 失败：{result.get("error", "未知错误")}'
        r = result['reminder']
        repeat_name = {'none': '一次性', 'hourly': '每小时',
                       'daily': '每天', 'weekly': '每周'}
        return (f'任务 {task_id} 提醒已设置：{r["at"]}'
                + (f'，{repeat_name.get(r["repeat"], r["repeat"])}'
                   + (f' ×{r["repeat_every"]}' if r['repeat'] != 'none' else '')))

    def _do_delete(self, args) -> str:
        task_id = str(args.get('task_id') or '').strip()
        if not self.service.delete_task(task_id):
            return f'todo_delete 失败：任务 {task_id} 不存在'
        return f'任务 {task_id} 已删除'