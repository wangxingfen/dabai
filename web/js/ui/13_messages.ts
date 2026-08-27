import type { AppKernel, TokenStats } from '../types/app-kernel.js';
import type { UsageMessage } from '../types/ws-protocol.js';

export default (function init(App: AppKernel) {
  /* ============================================================
   *  消息渲染
   * ============================================================ */
  // 聊天面板收起时新消息到达 → 提示切换按钮
  App.notifyFullscreenChat = function notifyFullscreenChat() {
    const panel = document.getElementById('chat-panel');
    if (panel!.classList.contains('collapsed')) {
      App.chatToggle!.classList.add('has-new');
    }
  };
  App.isFullscreen = false;

  /* ============================================================
   *  消息内联媒体：识别文本里的图片/视频链接并渲染成可查看的内容
   *  （智能体画图 / 生成视频时，链接直接变成图片/视频）
   * ============================================================ */
  const MEDIA_EXTS = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'avif', 'bmp', 'mp4', 'webm', 'ogv', 'mov', 'm4v'];
  const MEDIA_URL_RE = /(?:https?:\/\/[^\s<>"'`]+|\/[^\s<>"'`]+)\.(?:png|jpe?g|gif|webp|avif|bmp|mp4|webm|ogv|mov|m4v)(?:[?#][^\s<>"'`]*)?/gi;
  const MEDIA_TRAIL_RE = /[.,;:!?)\]}\u3001\u3002\uff0c\uff1b\uff1a\uff01\uff1f\u201d\u2019'`*_]+$/;

  function escHtml(s: string | null | undefined) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }
  function isVideoUrl(url: string) {
    return /\.(mp4|webm|ogv|mov|m4v)(?:[?#]|$)/i.test(url);
  }

  // 从文本里提取图片/视频链接（供聊天框渲染 & 大屏媒体墙共用）
  App.extractMediaUrls = function extractMediaUrls(text: string) {
    if (!text) return [];
    const out: string[] = [];
    MEDIA_URL_RE.lastIndex = 0;
    let m;
    while ((m = MEDIA_URL_RE.exec(text)) !== null) {
      const url = String(m[0]).replace(MEDIA_TRAIL_RE, '');
      const ext = (url.match(/\.([a-z0-9]+)(?:[?#]|$)/i) || [])[1];
      if (ext && MEDIA_EXTS.includes(ext.toLowerCase()) && !out.includes(url)) out.push(url);
    }
    return out;
  };

  // 普通网址链接化（媒体链接已单独处理）：raw 文本段 → 转义 + 可点击 <a>
  const PLAIN_URL_RE = /https?:\/\/[^\s<>"'`\u3000\uff08\uff09]+/gi;
  function linkifySegment(seg: string) {
    if (!seg) return '';
    let out = '';
    let last = 0;
    PLAIN_URL_RE.lastIndex = 0;
    let m;
    while ((m = PLAIN_URL_RE.exec(seg)) !== null) {
      const url = String(m[0]).replace(MEDIA_TRAIL_RE, '');
      if (!url) continue;
      out += escHtml(seg.slice(last, m.index)) +
        '<a class="msg-link" href="' + escHtml(url) + '" target="_blank" rel="noopener">' + escHtml(url) + '</a>';
      last = m.index + url.length;
    }
    out += escHtml(seg.slice(last));
    return out;
  }

  // 把文本渲染进消息元素：媒体链接内联成 <img>/<video>，普通网址转可点击链接，其余转义
  App.renderMsgMedia = function renderMsgMedia(el: HTMLElement | null, text: string) {
    if (!el) return;
    if (!text) { el.textContent = ''; return; }
    const urls = App.extractMediaUrls(text);
    if (!urls.length && !/(https?:\/\/)/i.test(text)) { el.textContent = text; return; }
    let html = '';
    let last = 0;
    MEDIA_URL_RE.lastIndex = 0;
    let m;
    while ((m = MEDIA_URL_RE.exec(text)) !== null) {
      const url = String(m[0]).replace(MEDIA_TRAIL_RE, '');
      if (!urls.includes(url)) continue;
      const attrs = 'src="' + escHtml(url) + '"';
      const failFallback = 'this.style.display=\'none\';this.parentNode.classList.add(\'media-failed\')';
      const media = isVideoUrl(url)
        ? '<video class="msg-media msg-media-video" ' + attrs + ' controls playsinline preload="metadata" onerror="' + failFallback + '"></video>'
        : '<img class="msg-media msg-media-img" ' + attrs + ' alt="' + escHtml(url) + '" loading="lazy" referrerpolicy="no-referrer" onerror="' + failFallback + '">';
      html += linkifySegment(text.slice(last, m.index)) +
        '<a class="msg-media-link" href="' + escHtml(url) + '" target="_blank" rel="noopener" title="' + escHtml(url) + '">' + media + '</a>';
      last = m.index + m[0].length;
    }
    html += linkifySegment(text.slice(last));
    el.innerHTML = html;
  };

  // 大图查看器：点击消息内图片弹层放大
  App.openMediaViewer = function openMediaViewer(url: string | null) {
    if (!url) return;
    let ov = document.getElementById('media-viewer') as HTMLElement | null;
    if (!ov) {
      ov = document.createElement('div');
      ov.id = 'media-viewer';
      ov.className = 'media-viewer';
      ov.innerHTML = '<div class="media-viewer-backdrop"></div><img class="media-viewer-img" alt=""><button class="media-viewer-close" title="关闭">×</button>';
      ov.addEventListener('click', (e) => {
        const t = e.target as HTMLElement;
        if (t === ov || t.classList.contains('media-viewer-backdrop') || t.classList.contains('media-viewer-close')) {
          ov!.classList.remove('show');
        }
      });
      document.body.appendChild(ov);
    }
    const img = ov.querySelector('.media-viewer-img') as HTMLImageElement;
    img.onload = () => ov!.classList.add('show');
    img.onerror = () => { if (App.showToast) App.showToast('图片加载失败'); ov!.classList.remove('show'); };
    img.src = url;
  };

  // 限制消息 DOM 节点数量，防止长时间聊天导致 DOM 膨胀卡顿
  App._trimMessages = function _trimMessages() {
    const all = App.messagesEl!.querySelectorAll('.msg:not(.streaming):not(.typing)');
    if (all.length > 200) {
      // 移除最旧的一半消息；用户正在上方翻阅时保持视觉位置不跳动
      const el = App.messagesEl!;
      const prevTop = el.scrollTop;
      const prevH = el.scrollHeight;
      const removeCount = Math.floor(all.length / 2);
      for (let i = 0; i < removeCount; i++) {
        all[i].remove();
      }
      const dh = prevH - el.scrollHeight;
      if (dh > 0 && prevTop > 0) el.scrollTop = Math.max(0, prevTop - dh);
    }
  };

  /* ============================================================
   *  智能滚动：用户上滑翻阅历史时暂停自动滚底，
   *  新消息以计数 + 未读高亮提示；滚回底部自动恢复跟随。
   * ============================================================ */
  App._newMsgCount = 0;

  App.isNearBottom = function isNearBottom() {
    const el = App.messagesEl!;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 90;
  };

  function clearUnread() {
    App.messagesEl!.querySelectorAll('.msg.unread').forEach((n) => n.classList.remove('unread'));
  }

  // 登记"来了一条新消息"（仅在用户未在底部时产生提示）；各卡片模块追加内容时调用
  App.bumpNewMsg = function bumpNewMsg(el?: HTMLElement | null) {
    if (App.isNearBottom()) return;
    App._newMsgCount += 1;
    if (el && el.classList && el.classList.contains('msg')) el.classList.add('unread');
    App.updateScrollHint();
  };

  App.scrollToBottom = function scrollToBottom(force?: boolean) {
    if (force === true) {
      // 平滑滚动动画期间不显示"回到底部"提示，避免闪烁
      App._forceScrolling = true;
      if (App._forceScrollTimer) clearTimeout(App._forceScrollTimer);
      App._forceScrollTimer = setTimeout(() => { App._forceScrolling = false; App.updateScrollHint(); }, 650);
      App.messagesEl!.scrollTo({ top: App.messagesEl!.scrollHeight, behavior: 'smooth' });
      App._newMsgCount = 0;
      clearUnread();
    } else if (App.isNearBottom()) {
      // 正在底部跟随：直接贴底（auto 避免流式高频更新时平滑动画抖动）
      App.messagesEl!.scrollTop = App.messagesEl!.scrollHeight;
    }
    App.updateScrollHint();
  };

  App.updateScrollHint = function updateScrollHint() {
    if (App._forceScrolling) return;
    const atBottom = App.isNearBottom();
    if (atBottom && App._newMsgCount > 0) {
      App._newMsgCount = 0;
      clearUnread();
    }
    const hint = App.scrollHint;
    if (!hint) return;
    if (atBottom) {
      hint.classList.remove('show');
      return;
    }
    hint.classList.add('show');
    if (App._newMsgCount > 0) {
      hint.innerHTML = '↓ <span class="scroll-hint-badge">' + App._newMsgCount + '</span> 新消息';
    } else {
      hint.textContent = '↓';
    }
  };
  App.messagesEl!.addEventListener('scroll', App.updateScrollHint, { passive: true });
  App.scrollHint!.addEventListener('click', () => App.scrollToBottom(true));

  App.addUserMsg = function addUserMsg(text: string, fromVoice = false) {
    const el = document.createElement('div');
    el.className = 'msg user';
    if (App.renderMsgMedia) App.renderMsgMedia(el, text);
    else el.textContent = text;
    if (fromVoice) el.title = '来自语音输入';
    App.messagesEl!.appendChild(el);
    App._trimMessages();
    App.scrollToBottom(true); // 用户自己的发言：始终跟到最新
    App.notifyFullscreenChat();
  };
  App.addAIMsg = function addAIMsg(text: string) {
    const el = document.createElement('div');
    el.className = 'msg ai';
    if (App.renderMsgMedia) App.renderMsgMedia(el, text);
    else el.textContent = text;
    App.messagesEl!.appendChild(el);
    App._trimMessages();
    App.bumpNewMsg(el); // 用户在翻历史：标记未读但不打扰
    App.scrollToBottom();
    App.notifyFullscreenChat();
    // 图片/视频链接同步投递到角色身后的任务直播大屏
    if (App.taskBoardAddMedia) {
      const urls = App.extractMediaUrls ? App.extractMediaUrls(text) : [];
      if (urls.length) App.taskBoardAddMedia(urls);
    }
    // 最近的交互信息 → 大屏全屏焦点（每条停留 1 分钟，无新信息回待机）
    if (App.taskBoardOnInteraction) {
      App.taskBoardOnInteraction({
        kind: 'msg',
        title: '💬 大白 · 刚刚回复',
        text,
        accent: '#7c5cff',
        tag: '💬 大白回复'
      });
    }
  };
  App.addSystemMsg = function addSystemMsg(text: string) {
    const el = document.createElement('div');
    el.className = 'msg system';
    el.textContent = text;
    App.messagesEl!.appendChild(el);
    App._trimMessages();
  };
  App.showTyping = function showTyping() {
    App.removeTyping();
    const el = document.createElement('div');
    el.className = 'msg ai typing';
    el.id = 'typing-indicator';
    el.innerHTML = '<span></span><span></span><span></span>';
    App.messagesEl!.appendChild(el);
    App.scrollToBottom();
  };
  App.removeTyping = function removeTyping() {
    const t = App.$('typing-indicator');
    if (t) t.remove();
  };

  /* ============================================================
   *  回合气泡：思考 → 工具调用 → 回复 内联成一条 AI 气泡
   *  一轮回复（thinking 开始 → audio_end 收尾）的全部过程
   *  都连续呈现在同一气泡内，避免独立卡片刷屏。
   *  - 思考段：默认折叠成「🧠 思考中…」摘要，点击展开完整过程；
   *  - 工具段：默认折叠，工具链模块（31_tool_chain）往里追加步骤；
   *  - 文本段：最终回复流式写入 .turn-text，不覆盖思考/工具内容。
   * ============================================================ */
  App._turnMsgEl = null;

  /** 新一轮回复开始：创建（或复用逻辑上最新的）回合气泡 */
  App.beginTurnBubble = function beginTurnBubble(sessionId?: string | null) {
    App.finishTurn(); // 上一轮若未收尾，先静态收尾
    const el = document.createElement('div');
    el.className = 'msg ai turn';
    el.dataset.session = sessionId || '';
    el.innerHTML =
      '<div class="turn-sec turn-think">' +
        '<button type="button" class="turn-head turn-think-head" aria-expanded="false">' +
          '<span class="turn-ic">🧠</span>' +
          '<span class="turn-title turn-think-title">思考中…</span>' +
          '<span class="turn-sub turn-think-sub"><span class="turn-spin">⟳</span>思考中</span>' +
          '<span class="turn-caret">▾</span>' +
        '</button>' +
        '<div class="turn-body turn-think-body" hidden></div>' +
      '</div>' +
      '<div class="turn-sec turn-tools">' +
        '<button type="button" class="turn-head turn-tools-head" aria-expanded="false" hidden>' +
          '<span class="turn-ic">🛠</span>' +
          '<span class="turn-title turn-tools-title">工具调用</span>' +
          '<span class="turn-sub turn-tools-state"></span>' +
          '<span class="turn-caret">▾</span>' +
        '</button>' +
        '<div class="turn-body turn-tools-body" hidden></div>' +
      '</div>' +
      '<div class="turn-text"></div>';
    App.messagesEl!.appendChild(el);
    App._turnMsgEl = el;
    App._trimMessages();
    App.bumpNewMsg(el);
    App.scrollToBottom();
    App.notifyFullscreenChat();
    return el;
  };

  /** 思考内容增量追加（后端 thinking.text 到达时持续更新） */
  App.appendTurnThinking = function appendTurnThinking(text: string) {
    const b = App._turnMsgEl;
    if (!b || !document.body.contains(b)) return;
    const body = b.querySelector('.turn-think-body');
    if (!body) return;
    if (text) {
      body.textContent = (body.textContent || '') + text;
      const sub = b.querySelector('.turn-think-sub');
      if (sub) {
        const brief = String(text).replace(/[\r\n]+/g, ' ').trim();
        const max = 56;
        const shown = brief.length > max ? brief.slice(0, max) + '…' : brief;
        sub.innerHTML = '<span class="turn-spin">⟳</span>' + escHtml(shown);
      }
    }
    App.scrollToBottom();
  };

  /** 整轮收尾：思考段由「思考中…」变「思考过程」，中断时标记气泡 */
  App.finishTurn = function finishTurn(interrupted?: boolean) {
    const b = App._turnMsgEl;
    if (!b || !document.body.contains(b)) return;
    const th = b.querySelector('.turn-think-head');
    if (th) {
      const title = th.querySelector('.turn-think-title');
      const sub = th.querySelector('.turn-think-sub');
      if (title && title.textContent === '思考中…') title.textContent = '思考过程';
      if (sub) {
        sub.classList.remove('spinning');
        sub.textContent = interrupted ? '已中断' : '';
      }
    }
    if (interrupted) {
      b.classList.add('interrupted');
      const ts = b.querySelector('.turn-tools-state');
      if (ts) ts.textContent = '已中断';
    }
  };

  /** 回复文本落点：回合气泡 → .turn-text；普通气泡 → 气泡本身 */
  App.turnTextContainer = function turnTextContainer() {
    const el = App.pendingAIMsgEl && document.body.contains(App.pendingAIMsgEl)
      ? App.pendingAIMsgEl
      : App._turnMsgEl;
    if (!el) return null;
    if (el.classList && el.classList.contains('turn')) {
      return el.querySelector('.turn-text') as HTMLElement | null;
    }
    return el;
  };

  /** 流式过程文本写入（不覆盖思考/工具段） */
  App.setTurnStreamText = function setTurnStreamText(text: string) {
    const el = App.turnTextContainer();
    if (!el) return;
    el.textContent = text;
  };

  /** 最终回复全文渲染（媒体链接内联等，落到 .turn-text） */
  App.renderTurnText = function renderTurnText(text: string) {
    const el = App.turnTextContainer();
    if (!el) return;
    if (App.renderMsgMedia) App.renderMsgMedia(el, text);
    else el.textContent = text;
  };

  /* ============================================================
   *  消息一键复制
   * ============================================================ */
  const COPY_ICON =
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
  const CHECK_ICON =
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';

  function copyTextToClipboard(text: string): Promise<void> {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    // 非安全上下文降级：隐藏 textarea + execCommand
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none;';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    ta.remove();
    if (!ok) return Promise.reject(new Error('copy failed'));
    return Promise.resolve();
  }

  // 给单条消息补复制按钮（打字中/流式中的消息跳过——文本尚未定型）
  App.ensureCopyBtn = function ensureCopyBtn(el: Node) {
    const e = el as HTMLElement;
    if (!e || !e.classList || !e.classList.contains('msg')) return;
    if (e.classList.contains('typing') || e.classList.contains('streaming')) return;
    if (e.querySelector('.msg-copy-btn')) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'msg-copy-btn';
    btn.title = '复制本条消息';
    btn.setAttribute('aria-label', '复制本条消息');
    btn.innerHTML = COPY_ICON;
    e.classList.add('has-copy');
    e.appendChild(btn);
  };

  // 提取消息可见文本（忽略复制按钮本身；任务树卡片去掉工具栏/箭头/图标，保留状态徽章）
  function msgText(el: HTMLElement) {
    const clone = el.cloneNode(true) as HTMLElement;
    clone.querySelectorAll('.msg-copy-btn, .tt-tools, .tt-caret, .tt-status-icon').forEach((b) => b.remove());
    return (clone.innerText || clone.textContent || '').trim();
  }

  // 复制按钮点击（事件委托：清空/裁剪消息也不会泄漏监听器）
  App.messagesEl!.addEventListener('click', (e) => {
    const target = e.target as HTMLElement;
    // 回合气泡折叠段：点头部展开/收起思考 / 工具调用
    const turnHead = target.closest && target.closest('.turn-head') as HTMLElement | null;
    if (turnHead) {
      const sec = turnHead.parentElement;
      if (sec) {
        const body = sec.querySelector('.turn-body') as HTMLElement | null;
        if (body) {
          body.hidden = !body.hidden;
          turnHead.setAttribute('aria-expanded', String(!body.hidden));
          sec.classList.toggle('open', !body.hidden);
        }
      }
      App.scrollToBottom();
      return;
    }
    // 点击消息内图片 → 弹大图查看
    if (target.closest && target.closest('.msg-media-img')) {
      const link = target.closest('.msg-media-link');
      if (link && App.openMediaViewer) App.openMediaViewer(link.getAttribute('href'));
      return;
    }
    const btn = target.closest('.msg-copy-btn') as HTMLElement | null;
    if (!btn) return;
    const msgEl = btn.closest('.msg') as HTMLElement | null;
    if (!msgEl) return;
    // 卡片类消息可挂 data-copy-text：只复制关键结果，而不是整卡文本
    const text = (btn.dataset.copyText && String(btn.dataset.copyText).trim()) || msgText(msgEl);
    if (!text) {
      if (App.showToast) App.showToast('没有可复制的内容');
      return;
    }
    copyTextToClipboard(text).then(() => {
      if (App.showToast) App.showToast('已复制');
      btn.classList.add('copied');
      btn.innerHTML = CHECK_ICON;
      btn.title = '已复制';
      setTimeout(() => {
        btn.classList.remove('copied');
        btn.innerHTML = COPY_ICON;
        btn.title = '复制本条消息';
      }, 1200);
    }).catch(() => {
      if (App.showToast) App.showToast('复制失败，请手动选择文本');
    });
  });

  // 观察消息容器：新气泡出现即补按钮；流式消息写完（移除 streaming 类）后补按钮
  const copyObserver = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === 'childList') {
        m.addedNodes.forEach((n) => {
          if (n.nodeType === 1) App.ensureCopyBtn(n);
        });
      } else if (m.type === 'attributes') {
        App.ensureCopyBtn(m.target);
      }
    }
  });
  copyObserver.observe(App.messagesEl!, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['class']
  });

  /* ============================================================
   *  聊天框头部：消息计数 / 高度档位 / 收起
   * ============================================================ */
  App.updateChatHeadCount = function updateChatHeadCount() {
    const el = document.getElementById('chat-head-count');
    if (!el) return;
    const n = App.messagesEl!.querySelectorAll('.msg:not(.typing)').length;
    el.textContent = String(n);
    el.title = n + ' 条消息';
  };
  // 复用消息观察器：任何消息增删后同步计数（延迟到微任务，避免重复统计）
  const countObserver = new MutationObserver(() => {
    queueMicrotask(App.updateChatHeadCount);
  });
  countObserver.observe(App.messagesEl!, { childList: true, subtree: true });
  App.updateChatHeadCount();

  /* ============================================================
   *  上下文 / Token 用量统计（localStorage 持久化，跨刷新累计）
   * ============================================================ */
  const TOKEN_STATS_KEY = 'dabai.token_stats_v1';
  App.fmtTokens = function fmtTokens(n: number) {
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
    return String(n);
  };
  App._tokenStats = { context: 0, completion: 0, total: 0, rounds: 0, msgs: 0 };
  try {
    const saved = JSON.parse(localStorage.getItem(TOKEN_STATS_KEY) || 'null');
    if (saved && typeof saved === 'object') Object.assign(App._tokenStats, saved);
  } catch (e) {}
  App._lastUsage = null;
  App.handleUsageMessage = function handleUsageMessage(msg: UsageMessage) {
    App._lastUsage = msg;
    App._tokenStats.completion += msg.completion_tokens || 0;
    App._tokenStats.total += msg.total_tokens || 0;
    App._tokenStats.rounds += msg.rounds || 0;
    App._tokenStats.msgs += 1;
    App._tokenStats.context = msg.prompt_tokens || 0; // 覆盖为最近值
    try {
      localStorage.setItem(TOKEN_STATS_KEY, JSON.stringify(App._tokenStats));
    } catch (e) {}
    App.updateTokenMeter();
  };
  App.attachMsgTokenBadge = function attachMsgTokenBadge(el: HTMLElement) {
    if (!el || !el.classList || !el.classList.contains('msg') || !el.classList.contains('ai')) return;
    if (!App._lastUsage) return;
    if (el.querySelector('.msg-token-badge')) return;
    const u = App._lastUsage;
    const badge = document.createElement('span');
    badge.className = 'msg-token-badge';
    badge.textContent = '⸙ ' + App.fmtTokens(u.total_tokens || 0);
    badge.title = '本轮 输入 ' + App.fmtTokens(u.prompt_tokens || 0) + ' + 输出 ' +
      App.fmtTokens(u.completion_tokens || 0) + '，共 ' + App.fmtTokens(u.total_tokens || 0) + ' tokens';
    el.appendChild(badge);
    App._lastUsage = null; // 防止误挂到下一条回复
  };
  App.updateTokenMeter = function updateTokenMeter() {
    const meter = document.getElementById('chat-token-meter');
    if (!meter) return;
    const u = App._lastUsage;
    if (u) {
      const win = u.context_window || 0;
      meter.textContent = '上下文 ' + App.fmtTokens(u.prompt_tokens || 0) +
        (win > 0 ? '/' + App.fmtTokens(win) : '') +
        ' · 本轮 ' + App.fmtTokens(u.total_tokens || 0);
      const pct = win > 0 ? Math.round(((u.prompt_tokens || 0) / win) * 1000) / 10 : 0;
      meter.title = '上下文占用 ' + pct + '%（输入 ' + App.fmtTokens(u.prompt_tokens || 0) +
        ' / 输出 ' + App.fmtTokens(u.completion_tokens || 0) + '）· 累计 ' +
        App.fmtTokens(App._tokenStats.total) + ' tokens · 回复 ' + App._tokenStats.msgs + ' 条';
      return;
    }
    if (App._tokenStats.total > 0) {
      meter.textContent = '累计 ' + App.fmtTokens(App._tokenStats.total) + ' tokens';
      meter.title = '累计 ' + App.fmtTokens(App._tokenStats.total) + ' tokens · 回复 ' +
        App._tokenStats.msgs + ' 条';
      return;
    }
    meter.textContent = '—';
  };
  const tokenMeterEl = document.getElementById('chat-token-meter');
  if (tokenMeterEl) tokenMeterEl.addEventListener('click', () => {
    const u = App._lastUsage;
    const win = u && u.context_window ? u.context_window : 0;
    const ctx = u ? App.fmtTokens(u.prompt_tokens || 0) : App.fmtTokens(App._tokenStats.context);
    App.showToast('累计 tokens：' + App.fmtTokens(App._tokenStats.total) +
      ' · 上下文 ' + ctx + (win > 0 ? '/' + App.fmtTokens(win) : ''));
  });
  App.updateTokenMeter(); // 让持久化数据上屏

  // 聊天框全屏：点击 ⤢ 展开为全屏（占满屏幕），再点还原
  App.chatFullscreen = false;
  App.setChatFullscreen = function setChatFullscreen(on: boolean) {
    const panel = document.getElementById('chat-panel');
    const appEl = document.getElementById('app');
    if (!panel) return;
    App.chatFullscreen = !!on;
    App.chatHeightLevel = App.chatFullscreen ? 1 : 0;
    // 清掉旧的高度档位类，统一走全屏
    panel.classList.remove('chat-h-md', 'chat-h-lg');
    panel.classList.toggle('chat-fs', App.chatFullscreen);
    if (appEl) appEl.classList.toggle('chat-fullscreen', App.chatFullscreen);
    const btn = document.getElementById('chat-height-btn');
    if (btn) {
      btn.classList.toggle('active', App.chatFullscreen);
      btn.textContent = App.chatFullscreen ? '⤡' : '⤢';
      btn.title = App.chatFullscreen ? '退出全屏聊天（点击还原）' : '全屏聊天（点击展开）';
    }
    const toggle = document.getElementById('chat-toggle');
    if (toggle) toggle.classList.toggle('shifted', App.chatFullscreen);
    setTimeout(() => App.scrollToBottom(true), 320);
  };
  App.cycleChatHeight = function cycleChatHeight() {
    App.setChatFullscreen(!App.chatFullscreen);
  };
  // 收起聊天（与右下角切换按钮行为一致）
  App.closeChatPanel = function closeChatPanel() {
    const panel = document.getElementById('chat-panel');
    const controls = document.getElementById('controls');
    if (!panel || panel.classList.contains('collapsed')) return;
    panel.classList.add('collapsed');
    controls!.classList.add('collapsed');
    if (App.chatToggle) App.chatToggle.classList.remove('shifted', 'has-new');
    setTimeout(App.onResize, 350);
  };
  const heightBtn = document.getElementById('chat-height-btn');
  if (heightBtn) heightBtn.addEventListener('click', App.cycleChatHeight);
  const closeBtn = document.getElementById('chat-close-btn');
  if (closeBtn) closeBtn.addEventListener('click', App.closeChatPanel);
  const head = document.getElementById('chat-head');
  if (head) head.addEventListener('touchstart', () => {}, { passive: true }); // 避免头部滚动穿透

  /* ============================================================
   *  Toast
   * ============================================================ */
});
