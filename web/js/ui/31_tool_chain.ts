import type { AppKernel } from '../types/app-kernel.js';

export default (function init(App: AppKernel) {
  /* ============================================================
   *  工作流工具链 —— 工具调用内联进「回合气泡」
   *  一轮回复（一次用户请求 → AI 完成）里的所有工具调用，不再单独
   *  刷屏独立卡片，而是追加进当前 AI 回合气泡的 .turn-tools-body，
   *  与思考段、最终回复文本在同一个气泡里连续呈现：
   *  编号 → 工具图标 + 名称 → 参数/结果（可展开）→ 状态。
   *  同时把链实时投递给角色身后的任务直播大屏（30_task_big_screen）。
   *  若回合气泡机制不可用（模块缺失/历史遗留），退化为旧的独立卡片。
   * ============================================================ */

  const TOOL_ICONS: Record<string, string> = {
    web_search: '🌐', search: '🔍', read: '📄', read_image: '🖼️', write: '✏️', edit: '📝',
    glob: '📂', grep: '🔎', pwsh: '💻', run_code: '⚙️',
    subagent: '🧩', subagent_fork: '🧬', send_message: '💬', list_agents: '👥', interrupt_agent: '✋',
    workflow: '🛠️', ralph: '🌀', skill: '📚', exit_plan_mode: '🧭',
    create_goal: '🎯', update_goal: '🎯', get_goal: '🎯', ask_user_question: '❓', todo_write: '✅',
    job_list: '📋', job_output: '📜', job_kill: '⏹️', fetch: '🛠️', evaluate: '🧮', remove: '✂️',
    codex: '🤖', ai: '🧑‍💻', image_gen: '🎨', video_gen: '🎬', music_play: '🎵', browser: '🌐'
  };
  function toolIcon(name: string) {
    const k = String(name || '');
    return TOOL_ICONS[k] || TOOL_ICONS[k.split('_')[0]] || '🔧';
  }
  function toolLabel(name: string) {
    if (!name) return '工具';
    return String(name).replace(/_/g, ' ');
  }

  function fmtArg(a: any) {
    if (a == null || a === '') return '';
    let s = typeof a === 'string' ? a : JSON.stringify(a);
    s = s.replace(/\s+/g, ' ').trim();
    if (!s || s === '{}' || s === '[]') return '';
    return s.length > 140 ? s.slice(0, 140) + '…' : s;
  }
  function fullArg(a: any) {
    if (a == null || a === '') return '';
    let s = typeof a === 'string' ? a : JSON.stringify(a, null, 2);
    return s.trim() || '';
  }
  function fmtResult(r: any) {
    if (r == null) return '（无返回）';
    let s = String(r).replace(/\s+/g, ' ').trim();
    if (!s) return '（无返回）';
    return s.length > 160 ? s.slice(0, 160) + '…' : s;
  }
  function fullResult(r: any) {
    if (r == null) return '（无返回）';
    return String(r).trim() || '（无返回）';
  }
  // 展开/收起：展开时回填完整文本，收起时恢复一行摘要
  function bindToggle(el: HTMLElement, prefix: string, shortText: string, fullGetter: () => string) {
    el.addEventListener('click', () => {
      const open = el.classList.toggle('open');
      el.textContent = prefix + (open ? fullGetter() : shortText);
    });
  }

  interface ToolChainStep {
    name: string;
    args: any;
    status: string;
    result: any;
    success: boolean | undefined;
    el: HTMLElement;
  }

  let round = 0;                    // 当前轮次（thinking 时 +1）
  let steps: ToolChainStep[] = [];  // 当前轮次步骤数据
  let stepsBox: HTMLElement | null = null;  // 内联模式：回合气泡 .turn-tools-body
  let card: HTMLElement | null = null;      // 降级模式：独立 .msg.tool-chain 卡片
  let collapseHintShown = false;    // 折叠提示行是否已挂到 stepsBox
  let startAt = 0;
  let timer: number | null = null;

  // ---------- 容器 ----------

  /** 头部元素（内联模式读回合气泡工具段；降级模式读卡片） */
  function headEls(): { title: HTMLElement | null; state: HTMLElement | null } {
    if (stepsBox && stepsBox.closest) {
      const sec = stepsBox.closest('.turn-sec.turn-tools') as HTMLElement | null;
      if (sec) {
        return {
          title: sec.querySelector('.turn-tools-title'),
          state: sec.querySelector('.turn-tools-state')
        };
      }
    }
    if (card) {
      return {
        title: card.querySelector('.tc-title'),
        state: card.querySelector('.tc-status')
      };
    }
    return { title: null, state: null };
  }

  /** 确保工具步骤容器存在：优先内联进回合气泡，否则退化独立卡片 */
  function ensureStepsBox(): HTMLElement | null {
    if (stepsBox && document.body.contains(stepsBox)) return stepsBox;
    if (card && document.body.contains(card)) return card.querySelector('.tc-steps');
    // 内联模式：找（或建）当前回合气泡
    let b = App._turnMsgEl;
    if (App.beginTurnBubble && (!b || !document.body.contains(b))) {
      b = App.beginTurnBubble();
    }
    if (b && b.querySelector) {
      const sec = b.querySelector('.turn-tools') as HTMLElement | null;
      if (sec) {
        const box = sec.querySelector('.turn-tools-body') as HTMLElement | null;
        if (box) {
          const head = sec.querySelector('.turn-tools-head') as HTMLElement | null;
          if (head) head.hidden = false;
          box.innerHTML = '';
          stepsBox = box;
          collapseHintShown = false;
          updateHeader();
          return box;
        }
      }
    }
    // 降级：独立工具链卡片
    card = document.createElement('div');
    card.className = 'msg tool-chain';
    card.innerHTML =
      '<div class="tc-head">' +
        '<span class="tc-logo">🔗</span>' +
        '<span class="tc-title">工具链</span>' +
        '<span class="tc-round">' + (round > 0 ? '第 ' + round + ' 轮' : '实时') + '</span>' +
        '<span class="tc-status running">执行中</span>' +
        '<span class="tc-elapsed">0s</span>' +
      '</div>' +
      '<div class="tc-steps"></div>';
    App.messagesEl!.appendChild(card);
    App._trimMessages();
    App.bumpNewMsg(card);
    App.scrollToBottom();
    App.notifyFullscreenChat();
    return card.querySelector('.tc-steps');
  }

  // ---------- 对外 API ----------

  /** 新一轮回复开始（websocket thinking 事件触发）：上一轮收尾 + 轮次 +1 */
  App.toolChainBeginTurn = function toolChainBeginTurn() {
    finalizePrevious();
    round += 1;
    steps = [];
    stepsBox = null;
    card = null;
    collapseHintShown = false;
    startAt = Date.now();
    if (timer) { clearInterval(timer); timer = null; }
  };

  /** 会话切换/历史恢复：清空链条状态，避免旧步骤串到新会话 */
  App.toolChainReset = function toolChainReset() {
    steps = [];
    stepsBox = null;
    card = null;
    collapseHintShown = false;
    round = 0;
    startAt = 0;
    if (timer) { clearInterval(timer); timer = null; }
  };

  /** 工具调用开始（tool_call_start） */
  App.toolChainStart = function toolChainStart(toolName: string, args: any) {
    const box = ensureStepsBox();
    if (!box) return;
    const d = document.createElement('div');
    d.className = 'tc-step running';
    d.innerHTML =
      '<div class="tc-rail">' +
        '<span class="tc-idx"></span>' +
        '<span class="tc-conn"></span>' +
      '</div>' +
      '<div class="tc-body">' +
        '<div class="tc-node">' +
          '<span class="tc-tool-icon"></span>' +
          '<span class="tc-tool-name"></span>' +
          '<span class="tc-state">' +
            '<span class="tc-spinner"></span>' +
            '<span class="tc-step-elapsed"></span>' +
            '<span class="tc-flag tc-ok">✓</span>' +
            '<span class="tc-flag tc-bad">✗</span>' +
          '</span>' +
        '</div>' +
        '<div class="tc-args" style="display:none"></div>' +
        '<div class="tc-result" style="display:none"></div>' +
      '</div>';
    const st: ToolChainStep = { name: toolName, args, status: 'running', result: undefined, success: undefined, el: d };
    steps.push(st);
    d.dataset.step = String(steps.length - 1);
    const idx = d.querySelector('.tc-idx') as HTMLElement;
    idx.textContent = String(steps.length).padStart(2, '0');
    const icon = d.querySelector('.tc-tool-icon') as HTMLElement;
    icon.textContent = toolIcon(toolName);
    const nm = d.querySelector('.tc-tool-name') as HTMLElement;
    nm.textContent = toolLabel(toolName);
    nm.title = toolName;
    const a = fmtArg(args);
    if (a) {
      const argsEl = d.querySelector('.tc-args') as HTMLElement;
      argsEl.style.display = 'block';
      argsEl.textContent = '参数 ' + a;
      argsEl.title = '点击展开/收起';
      bindToggle(argsEl, '参数 ', a, () => fullArg(args));
    }
    box.appendChild(d);
    updateHeader();
    refreshCollapse();
    App._trimMessages();
    App.scrollToBottom();
    pushToBoard();
  };

  /** 工具执行心跳（tool_call_progress）：更新运行中步骤的已运行时长 */
  App.toolChainProgress = function toolChainProgress(toolName: string, elapsed: number, message?: string) {
    let target: ToolChainStep | null = null;
    for (let i = steps.length - 1; i >= 0; i--) {
      const s = steps[i];
      if (s.status === 'running' && (!toolName || s.name === toolName)) { target = s; break; }
    }
    if (!target) return;
    const el = target.el.querySelector('.tc-step-elapsed') as HTMLElement;
    if (el) el.textContent = fmtElapsed((elapsed || 0) * 1000);
    const stateEl = target.el.querySelector('.tc-state') as HTMLElement;
    if (stateEl && message) stateEl.title = message;
    updateHeader();
  };

  /** 工具调用结果（tool_call_result） */
  App.toolChainResult = function toolChainResult(toolName: string, result: any, success: boolean) {
    // 找同名的最后一个还在等待结果的步骤；找不到则取最后一个待定步骤
    let target: ToolChainStep | null = null;
    for (let i = steps.length - 1; i >= 0; i--) {
      const s = steps[i];
      if (s.status === 'running' && (!toolName || s.name === toolName)) { target = s; break; }
    }
    if (!target) {
      // 孤立结果（链条可能已收尾）：兜底补一个只读步骤
      App.toolChainStart!(toolName || '工具', '');
      target = steps[steps.length - 1];
    }
    target.status = success === false ? 'error' : 'done';
    target.result = result;
    target.success = success !== false;
    const d = target.el;
    d.className = 'tc-step ' + target.status;
    const resEl = d.querySelector('.tc-result') as HTMLElement;
    const prefix = success === false ? '返回 ' : '结果 ';
    const short = fmtResult(result);
    resEl.style.display = 'block';
    resEl.textContent = prefix + short;
    resEl.title = '点击展开/收起';
    bindToggle(resEl, prefix, short, () => fullResult(result));
    updateHeader();
    pushToBoard();
    // 结果到达后自动滚动一次，让最新状态可见
    App.scrollToBottom();
  };

  /** 委派链路：codex/opencode 子任务建立后，把下钻入口挂到当前工具步骤上 */
  App.codexLinkTask = function codexLinkTask(toolName: string, taskId: string) {
    if (!taskId) return;
    let target: ToolChainStep | null = null;
    for (let i = steps.length - 1; i >= 0; i--) {
      const s = steps[i];
      if (s.status === 'running') { target = s; break; }
    }
    if (!target) return;
    let chip = target.el.querySelector('.tc-subtask') as HTMLElement | null;
    if (!chip) {
      chip = document.createElement('span');
      chip.className = 'tc-subtask';
      const body = target.el.querySelector('.tc-body');
      if (body) body.appendChild(chip);
    }
    const agentName = toolName === 'codex' ? 'Codex' : toolName === 'ai' ? 'OpenCode' : toolLabel(toolName);
    chip.textContent = `🧩 ${agentName} 子任务 ${String(taskId).slice(-6)} ↗`;
    chip.title = '点击打开任务中心，查看该子任务每一步的详细动作、输入输出与中间结果';
    chip.addEventListener('click', (e) => {
      e.stopPropagation();
      if (App.selectTaskCenter) App.selectTaskCenter(taskId);
    });
    pushToBoard();
  };

  /** 整轮收尾（audio_end：回复完成），并触发思考段收尾 */
  App.toolChainEndTurn = function toolChainEndTurn() {
    const hasError = steps.some(s => s.status === 'error');
    const still = steps.filter(s => s.status === 'running');
    for (const s of still) {
      s.status = 'ended';
      s.el.className = 'tc-step ended';
    }
    const h = headEls();
    if (stepsBox && h.state) h.state.textContent = hasError ? '有失败' : '已完成';
    if (card) {
      const statusEl = card.querySelector('.tc-status') as HTMLElement | null;
      if (statusEl) {
        statusEl.className = 'tc-status ' + (hasError ? 'error' : 'done');
        statusEl.textContent = hasError ? '有失败' : '已完成';
      }
      const elapsed = card.querySelector('.tc-elapsed') as HTMLElement | null;
      if (elapsed) elapsed.textContent = fmtElapsed(Date.now() - startAt);
    }
    if (timer) { clearInterval(timer); timer = null; }
    if (App.finishTurn) App.finishTurn(false);
    pushToBoard();
    App.scrollToBottom();
  };

  /** 被打断 / 出错（interrupted、error 消息） */
  App.toolChainAbort = function toolChainAbort() {
    for (const s of steps) {
      if (s.status === 'running') {
        s.status = 'ended';
        s.el.className = 'tc-step ended';
      }
    }
    const h = headEls();
    if (stepsBox && h.state) h.state.textContent = '已中断';
    if (card) {
      const statusEl = card.querySelector('.tc-status') as HTMLElement | null;
      if (statusEl) {
        statusEl.className = 'tc-status cancelled';
        statusEl.textContent = '已中断';
      }
    }
    if (timer) { clearInterval(timer); timer = null; }
    if (App.finishTurn) App.finishTurn(true);
    pushToBoard();
  };

  // ---------- 内部 ----------

  function finalizePrevious() {
    const running = steps.some(s => s.status === 'running');
    if (!running) return;
    for (const s of steps) {
      if (s.status === 'running') { s.status = 'ended'; s.el.className = 'tc-step ended'; }
    }
    const h = headEls();
    if (stepsBox && h.state) h.state.textContent = '已完成';
    if (card) {
      const statusEl = card.querySelector('.tc-status') as HTMLElement | null;
      if (statusEl) { statusEl.className = 'tc-status done'; statusEl.textContent = '已完成'; }
    }
    if (timer) { clearInterval(timer); timer = null; }
  }

  function updateHeader() {
    if (!stepsBox && !card) return;
    const done = steps.filter(s => s.status === 'done').length;
    const err = steps.filter(s => s.status === 'error').length;
    const running = steps.filter(s => s.status === 'running').length;
    const h = headEls();
    if (h.title) {
      h.title.textContent = '工具调用 · ' + steps.length + ' 步' +
        (done ? ' · ' + done + ' 成功' : '') +
        (err ? ' · ' + err + ' 失败' : '');
    }
    if (!stepsBox && card) {
      const title = card.querySelector('.tc-title') as HTMLElement | null;
      if (title) title.textContent = '工具链 · ' + steps.length + ' 步' +
        (done ? ' · ' + done + ' 成功' : '') +
        (err ? ' · ' + err + ' 失败' : '');
      const elapsed = card.querySelector('.tc-elapsed') as HTMLElement | null;
      if (elapsed && running > 0) elapsed.textContent = fmtElapsed(Date.now() - startAt);
    }
    if (!timer && running > 0) {
      timer = setInterval(tickElapsed, 1000);
    } else if (running === 0 && timer) {
      clearInterval(timer); timer = null;
    }
  }

  function tickElapsed() {
    if (!stepsBox && !card) { if (timer) { clearInterval(timer); timer = null; } return; }
    const running = steps.filter(s => s.status === 'running').length;
    if (running === 0) { if (timer) { clearInterval(timer); timer = null; } return; }
    const elapsedTxt = fmtElapsed(Date.now() - startAt);
    const h = headEls();
    if (stepsBox && h.state) {
      const done = steps.filter(s => s.status === 'done').length;
      h.state.textContent = '执行中 ' + elapsedTxt + (done ? ' · ' + done + ' 成功' : '');
    }
    if (card) {
      const el = card.querySelector('.tc-elapsed') as HTMLElement | null;
      if (el) el.textContent = elapsedTxt;
      const statusEl = card.querySelector('.tc-status') as HTMLElement | null;
      if (statusEl) { statusEl.className = 'tc-status running'; statusEl.textContent = '执行中'; }
    }
  }

  function fmtElapsed(ms: number) {
    const s = Math.max(0, Math.floor(ms / 1000));
    if (s < 60) return s + 's';
    return Math.floor(s / 60) + 'm' + (s % 60) + 's';
  }

  function refreshCollapse() {
    const all = steps.map(s => s.el);
    if (all.length <= 6) {
      all.forEach(n => n.classList.remove('old'));
      if (stepsBox) {
        const hint = stepsBox.querySelector('.tc-collapsed-hint') as HTMLElement | null;
        if (hint) hint.style.display = 'none';
      }
      if (card) card.classList.remove('has-fold');
      return;
    }
    const hidden = all.length - 6;
    all.forEach((n, i) => n.classList.toggle('old', i < hidden));
    if (stepsBox) {
      let hint = stepsBox.querySelector('.tc-collapsed-hint') as HTMLElement | null;
      if (!hint) {
        hint = document.createElement('div');
        hint.className = 'tc-collapsed-hint';
        hint.innerHTML = '· 更早步骤已收起 <span class="tc-hidden-count">0</span> 步，点上方「工具调用」展开查看';
        stepsBox.appendChild(hint);
      }
      hint.style.display = 'block';
      const cnt = hint.querySelector('.tc-hidden-count') as HTMLElement | null;
      if (cnt) cnt.textContent = String(hidden);
    }
    if (card) {
      card.classList.add('has-fold');
      const hint = card.querySelector('.tc-collapsed-hint') as HTMLElement | null;
      if (hint) hint.style.display = 'block';
      const cnt = card.querySelector('.tc-hidden-count') as HTMLElement | null;
      if (cnt) cnt.textContent = String(hidden);
    }
  }

  // ---------- 投递到直播大屏 ----------
  function pushToBoard() {
    if (!App.taskBoardOnToolChain) return;
    const ch = {
      round,
      title: round > 0 ? '对话 · 第 ' + round + ' 轮' : '智能体 · 实时工具链',
      steps: steps.map(s => ({ name: s.name, status: s.status })),
      ts: Date.now()
    };
    App.taskBoardOnToolChain(ch);
  }

  console.log('[ToolChain] 工作流工具链已改为内联回合气泡');
});