"""TODO 插件 REST API —— 基于 FastAPI 的 JSON 数据交互接口。

所有路由统一挂在 /api/todo/*（由 plugin.py 的 on_load 挂载到大白 FastAPI app）：
- 任务：创建 / 查询 / 更新 / 删除（POST/GET/PATCH/DELETE /api/todo/tasks*）；
- 复杂任务分解：POST /api/todo/breakdown（解析自然语言 → 子任务计划 → 可落库）；
- 执行计划：POST /api/todo/plan（按依赖拓扑排序，回执顺序执行清单）；
- 提醒：GET /api/todo/reminders（轮询到期提醒事件，?consume=1 同时消费）。
成功响应统一为 {"code": 200, ...业务字段}；失败为 {"code": 非0, "message": ...}。
本模块通过 build_router(service) 返回 APIRouter，便于复用/扩展（接口即契约）。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from plugins.todo_list.parser import decompose as _decompose


def build_router(service) -> APIRouter:
    """构造 todo_list 的全部 REST 路由（绑定到给定 TodoService 实例）。"""
    router = APIRouter(prefix='/api/todo', tags=['todo_list'])

    # ---------------- 复杂任务分解 ----------------

    @router.post('/breakdown')
    async def todo_breakdown(request: Request):
        """解析自然语言任务描述，返回可执行的子任务计划。

        请求体：{"text": "...", "create": true|false}
        create=true 时计划直接落库（自动生成父任务 + 子任务 + 依赖）。
        """
        body = await _json_body(request)
        text = str((body or {}).get('text') or '').strip()
        if not text:
            return JSONResponse({'code': 400, 'message': '缺少 text（任务描述）'},
                                status_code=400)
        plan = _decompose(text)
        result = {'code': 200, 'plan': plan}
        if (body or {}).get('create'):
            imported = service.import_plan(plan)
            result['created'] = imported
        return result

    # ---------------- 执行计划 ----------------

    @router.post('/plan')
    async def todo_plan():
        """按当前任务的依赖关系生成执行计划（拓扑排序）。"""
        return {'code': 200, 'plan': service.compute_execution_plan()}

    # ---------------- 任务 CRUD ----------------

    @router.post('/tasks')
    async def create_task(request: Request):
        """创建任务：{title, description, priority, status, due_date,
        depends_on, tags, notes, subtasks:[{title,...}, ...], reminder:{...}}"""
        body = await _json_body(request)
        if not str((body or {}).get('title') or '').strip():
            return JSONResponse({'code': 400, 'message': '缺少 title'},
                                status_code=400)
        try:
            task = service.create_task(
                title=body['title'],
                description=body.get('description') or '',
                priority=body.get('priority'),
                status=body.get('status'),
                due_date=body.get('due_date'),
                depends_on=body.get('depends_on'),
                tags=body.get('tags'),
                notes=body.get('notes') or '',
            )
            for sub in (body.get('subtasks') or []):
                service.add_subtask(task['id'], title=sub.get('title') or '',
                                    description=sub.get('description') or '',
                                    priority=sub.get('priority'),
                                    depends_on=sub.get('depends_on'),
                                    due_date=sub.get('due_date'))
            reminder = body.get('reminder')
            if reminder:
                rtype = reminder.get('type')
                repeat = reminder.get('repeat') or ('none' if rtype == 'once' else 'daily')
                service.set_reminder(task['id'], at=reminder.get('at'),
                                     remind_type=rtype, repeat=repeat,
                                     repeat_every=reminder.get('repeat_every') or 1)
            return {'code': 200, 'task': service.get_task(task['id'])}
        except KeyError as e:
            return JSONResponse({'code': 404, 'message': str(e)}, status_code=404)
        except Exception as e:
            return JSONResponse({'code': 500, 'message': str(e)}, status_code=500)

    @router.get('/tasks')
    async def list_tasks(status: str | None = None,
                         priority: str | None = None,
                         keyword: str | None = None,
                         due_before: str | None = None):
        """查询任务列表；可按状态/优先级/关键词/截止时间过滤。"""
        due_ts = None
        if due_before:
            try:
                due_ts = float(due_before)
            except (TypeError, ValueError):
                return JSONResponse(
                    {'code': 400, 'message': 'due_before 需为 Unix 时间戳'},
                    status_code=400)
        items = service.list_tasks(status=status, priority=priority,
                                   keyword=keyword, due_before=due_ts)
        return {'code': 200, 'tasks': items, 'count': len(items)}

    @router.get('/tasks/{task_id}')
    async def get_task(task_id: str):
        """按 id 获取单个任务（含子任务与提醒信息）。"""
        task = service.get_task(task_id)
        if task is None:
            return JSONResponse({'code': 404, 'message': f'任务不存在: {task_id}'},
                                status_code=404)
        return {'code': 200, 'task': task}

    @router.patch('/tasks/{task_id}')
    async def update_task(task_id: str, request: Request):
        """更新任务字段：{title, description, priority, status,
        due_date, tags, notes, depends_on}"""
        body = await _json_body(request)
        try:
            service.update_task(task_id, **{k: v for k, v in body.items()})
        except KeyError as e:
            return JSONResponse({'code': 404, 'message': str(e)}, status_code=404)
        return {'code': 200, 'task': service.get_task(task_id)}

    @router.delete('/tasks/{task_id}')
    async def delete_task(task_id: str):
        """删除任务（并清除其它任务对它的依赖引用）。"""
        if not service.delete_task(task_id):
            return JSONResponse({'code': 404, 'message': f'任务不存在: {task_id}'},
                                status_code=404)
        return {'code': 200, 'message': f'任务 {task_id} 已删除'}

    # ---------------- 子任务 ----------------

    @router.post('/tasks/{task_id}/subtasks')
    async def add_subtask(task_id: str, request: Request):
        """给任务追加子任务：{title, description, priority, due_date, depends_on}"""
        body = await _json_body(request)
        if not str((body or {}).get('title') or '').strip():
            return JSONResponse({'code': 400, 'message': '缺少 title'},
                                status_code=400)
        try:
            sub = service.add_subtask(task_id, title=body['title'],
                                      description=body.get('description') or '',
                                      priority=body.get('priority'),
                                      depends_on=body.get('depends_on'),
                                      due_date=body.get('due_date'))
        except KeyError as e:
            return JSONResponse({'code': 404, 'message': str(e)}, status_code=404)
        return {'code': 200, 'subtask': sub}

    @router.patch('/tasks/{task_id}/subtasks/{sub_id}')
    async def update_subtask(task_id: str, sub_id: str, request: Request):
        """更新子任务状态/优先级等：{status, priority, title, description}"""
        body = await _json_body(request)
        try:
            sub = service.update_subtask(task_id, sub_id,
                                         status=body.get('status'),
                                         priority=body.get('priority'),
                                         title=body.get('title'),
                                         description=body.get('description'))
        except KeyError as e:
            return JSONResponse({'code': 404, 'message': str(e)}, status_code=404)
        return {'code': 200, 'subtask': sub, 'task': service.get_task(task_id)}

    @router.delete('/tasks/{task_id}/subtasks/{sub_id}')
    async def delete_subtask(task_id: str, sub_id: str):
        """删除子任务（并清除其它子任务对它的依赖引用）。"""
        if not service.delete_subtask(task_id, sub_id):
            return JSONResponse({'code': 404, 'message': f'子任务不存在: {sub_id}'},
                                status_code=404)
        return {'code': 200, 'message': f'子任务 {sub_id} 已删除'}

    # ---------------- 提醒 ----------------

    @router.post('/tasks/{task_id}/remind')
    async def set_reminder(task_id: str, request: Request):
        """设置提醒：{at, remind_type: once|repeat, repeat: none|hourly|daily|weekly,
        repeat_every: 周期倍数}"""
        body = await _json_body(request)
        try:
            result = service.set_reminder(
                task_id,
                at=body.get('at'),
                remind_type=body.get('remind_type'),
                repeat=body.get('repeat') or 'none',
                repeat_every=body.get('repeat_every') or 1,
            )
        except KeyError as e:
            return JSONResponse({'code': 404, 'message': str(e)}, status_code=404)
        if not result.get('ok'):
            return JSONResponse({'code': 400, 'message': result.get('error', '设置失败')},
                                status_code=400)
        return {'code': 200, **result}

    @router.delete('/tasks/{task_id}/remind')
    async def clear_reminder(task_id: str):
        """清除任务提醒。"""
        try:
            result = service.clear_reminder(task_id)
        except KeyError:
            return JSONResponse({'code': 404, 'message': f'任务不存在: {task_id}'},
                                status_code=404)
        return {'code': 200, **result}

    @router.get('/reminders')
    async def get_reminders(consume: int = Query(0)):
        """轮询已触发的提醒事件（提醒投递）。consume=1 时取出并清空缓冲。"""
        if consume:
            events = service.pop_reminder_events()
        else:
            events = service.pending_reminder_events()
        return {'code': 200, 'events': events, 'count': len(events),
                'next_at': service.next_due_reminder_ts()}

    # ---------------- 统计 ----------------

    @router.get('/stats')
    async def todo_stats():
        """任务统计：总数 / 各状态数量 / 子任务数 / 提醒数 / 逾期数。"""
        return {'code': 200, 'stats': service.stats()}

    return router


async def _json_body(request: Request) -> dict:
    """兼容 FastAPI 直接把 body 作为参数传入的场景（body 已注入时不读流）。"""
    raw = await request.body()
    try:
        return json.loads(raw or b'{}')
    except (json.JSONDecodeError, TypeError):
        return {}