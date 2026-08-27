import type { AppKernel } from '../types/app-kernel.js';

export default (function init(App: AppKernel) {
  /* ============================================================
   *  唤醒词待机模式（Wake Word Standby）
   *  - 默认/唯一入口：只有呼叫唤醒词才进入聆听状态（说话→识别→对话）；
   *  - 唤醒词可在角色卡片配置（wake_word），默认取角色名；
   *  - 服务端 STT 后做唤醒词匹配（含拼音/字符模糊容错），命中才开启对话；
   *  - 会话空闲超时自动回到待机，再次对话仍需唤醒词。
   * ============================================================ */
  const WAKE_WORDS_KEY = 'dabai.wakeWords';
  App.WAKE_DEFAULT_WORDS = ['大白'];
  // 会话空闲多久（毫秒）自动回到唤醒待机：保证"只有叫唤醒词才进入聆听"
  App.AUTO_STANDBY_MS = 90000;
  App._lastConversationAt = 0; // 最近一次对话活跃时间（用户输入 / AI 回复结束）
  App._enteredViaWake = false; // 当前自动会话是否由唤醒词开启（决定空闲是否回待机）

  /* ---------- 唤醒词列表（本地缓存 + 服务端配置同步） ---------- */
  App.wakeWords = (() => {
    try {
      const saved = JSON.parse(localStorage.getItem(WAKE_WORDS_KEY) || 'null');
      if (Array.isArray(saved) && saved.length) return saved.slice(0, 5);
    } catch (e) { /* 忽略损坏的本地缓存 */ }
    return [...App.WAKE_DEFAULT_WORDS];
  })();

  /**
   * 更新唤醒词列表：合并去重（角色名优先）+ 本地持久化 + 同步服务端匹配。
   * 服务端 wake_config.words 与 whisper 初始提示词都会据此即时更新，
   * 保证「前端显示 / 服务端匹配 / 本地模型偏置」三者一致。
   */
  App.applyWakeWords = function applyWakeWords(words: string[]) {
    const clean = (Array.isArray(words) ? words : [])
      .map(w => String(w).trim()).filter(Boolean);
    if (!clean.length) return;
    const merged: string[] = [];
    for (const w of [...clean, ...App.wakeWords]) {
      if (w && !merged.includes(w)) merged.push(w);
    }
    App.wakeWords = merged.slice(0, 5);
    localStorage.setItem(WAKE_WORDS_KEY, JSON.stringify(App.wakeWords));
    fetch('/api/wake/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ words: App.wakeWords })
    }).catch(() => { /* 离线忽略，下次同步 */ });
  };

  /** 旧版单点保存（兼容保留） */
  App.saveWakeWords = function saveWakeWords(words: string[]) {
    const cleaned = (Array.isArray(words) ? words : [])
      .map(w => String(w).trim()).filter(Boolean).slice(0, 5);
    if (!cleaned.length) return false;
    App.wakeWords = cleaned;
    localStorage.setItem(WAKE_WORDS_KEY, JSON.stringify(App.wakeWords));
    return true;
  };

  /** 从角色卡片解析唤醒词：卡片显式配置 > 角色名 > 卡片名 */
  App.resolveRoleWakeWord = function resolveRoleWakeWord(card: any) {
    if (!card) return '';
    return ((card.wake_word || '').trim())
      || ((card.role_name || '').trim())
      || ((card.name || '').trim())
      || '';
  };

  /**
   * 根据当前角色刷新唤醒词：
   * 卡片显式唤醒词/角色名 → 全局角色名（/api/config/role）→ 保持服务端配置兜底。
   * 在切换角色卡片 / 启动恢复 / 角色名变更时调用。
   */
  App.refreshWakeWordsFromRole = async function refreshWakeWordsFromRole(card: any) {
    let word = App.resolveRoleWakeWord(card);
    if (!word && !card) {
      try {
        const r = await fetch('/api/config/role');
        const cfg = await r.json() as { role_name?: string };
        word = (cfg.role_name || '').trim();
      } catch (e) { /* 离线忽略 */ }
    }
    if (word && word !== 'AI助手') {
      App.applyWakeWords([word]);
    }
  };

  /** 记录一次对话活跃（用户语音识别成功 / 文字发送 / AI 回复结束）。
   *  时间基准必须与 VAD 循环/checkAutoReturnStandby 一致（performance.now()），
   *  否则空闲超时判断会把两个时钟混在一起，永远不会回到唤醒待机。 */
  App.bumpConversation = function bumpConversation() {
    App._lastConversationAt = performance.now();
  };

  /** 从服务端同步唤醒词配置（仅作兜底词源，不覆盖角色派生唤醒词） */
  App.syncWakeConfig = async function syncWakeConfig() {
    try {
      const res = await fetch('/api/wake/config');
      if (!res.ok) return;
      const cfg = await res.json() as { enabled?: boolean; words?: string[] };
      if (cfg.enabled === false) return;
      // 仅当本地还没有角色派生的唤醒词时，用服务端词表兜底
      const roleDerived = App.wakeWords.filter(w => !App.WAKE_DEFAULT_WORDS.includes(w));
      if (!roleDerived.length && Array.isArray(cfg.words) && cfg.words.length) {
        App.wakeWords = cfg.words;
        localStorage.setItem(WAKE_WORDS_KEY, JSON.stringify(App.wakeWords));
      }
    } catch (e) { /* 离线/服务器未启动，忽略 */ }
  };

  /* ---------- 唤醒结果处理（由 WebSocket 消息驱动） ---------- */
  let wakeFailCount = 0;

  /** 唤醒成功：进入自动对话会话（此后说话不再需要唤醒词，直到空闲回待机） */
  App.onWakeOk = function onWakeOk(word?: string, transcript?: string) {
    if (App.voiceMode !== 'wake') return; // 已被用户手动切换，忽略过期消息
    wakeFailCount = 0;
    // 清掉唤醒失败冷却（恢复自动会话后不再需要它，避免残留值影响下次待机）
    App._wakeRetryAt = 0;
    console.log('[Wake] 唤醒成功:', word, '|', transcript || '');
    App.bumpConversation();
    App.setVoiceMode('auto');
    App._enteredViaWake = true; // 标记由唤醒词开启的会话 → 空闲自动回待机
    App.showToast('已唤醒 · 开始对话吧');
  };

  /** 未命中：静默回待机；提示节流（首次提示一次，之后每 3 次提醒一下）。
   *  同时设置冷却，防止环境噪音立刻再次触发录音 → "一直聆听中"。 */
  App.onWakeFail = function onWakeFail(transcript?: string) {
    if (App.voiceMode !== 'wake') return;
    wakeFailCount++;
    // 冷却 2.5s：避免噪音/回声连发反复进入录音。
    // 注意必须与 VAD 循环里比较的时间基准一致（performance.now()），
    // 之前误用 Date.now() 导致 after 第一次唤醒失败后冷却永远不过期、待机再也听不到唤醒词。
    App._wakeRetryAt = performance.now() + 2500;
    // 回到 IDLE 让 VAD 继续监听下一句
    if (App.currentState === App.State.LISTENING || App.currentState === App.State.THINKING) {
      App.setState(App.State.IDLE);
    }
    if (wakeFailCount === 1 || wakeFailCount % 3 === 0) {
      const words = (App.wakeWords && App.wakeWords.length ? App.wakeWords : ['大白']).join(' / ');
      App.showToast(`没听到「${words}」，再叫一次试试~`);
    }
  };

  /** 会话空闲超时 → 自动回到唤醒待机（只有叫唤醒词才进入聆听状态）。
   *  仅对「由唤醒词开启的自动会话」生效；用户手动切到自动对话不自动回退。 */
  App.checkAutoReturnStandby = function checkAutoReturnStandby(now: number) {
    if (App.voiceMode !== 'auto') return false;
    if (!App._enteredViaWake) return false; // 手动选的自动模式不自动回待机
    if (App.currentState !== App.State.IDLE || App.vadState !== 'idle') return false;
    if (App.isRecording || (App.vadRecorder && App.vadRecorder.state === 'recording')) return false;
    if (!App._lastConversationAt) return false;
    if (now - App._lastConversationAt < App.AUTO_STANDBY_MS) return false;
    // 回到待机：清空计时与标记，避免进入 wake 后立刻再触发
    App._lastConversationAt = 0;
    App._enteredViaWake = false;
    App.setVoiceMode('wake');
    const words = (App.wakeWords && App.wakeWords.length ? App.wakeWords : ['大白']).join(' / ');
    App.showToast(`先聊到这里~ 叫「${words}」随时唤醒我`);
    return true;
  };

  /* ---------- 页面可见性恢复（复杂场景容错） ----------
   * 浏览器后台/锁屏后可能挂起 AudioContext 或回收麦克风流，
   * 回前台时立即恢复，避免唤醒待机"假死"。 */
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    if (App.voiceMode !== 'auto' && App.voiceMode !== 'wake') return;
    if (App.audioCtx && App.audioCtx.state === 'suspended') {
      App.audioCtx.resume().then(() => console.log('[Wake] AudioContext 已恢复')).catch(() => {});
    }
    // 流已被回收时由 vadLoop 自身的活性检查负责重建，这里只做加速触发
    if (App.voiceMode === 'wake' && !App.vadRAF) {
      App.vadState = 'idle';
      App.vadLoop();
    }
  });

  // 启动即同步服务端唤醒词配置（异步，不阻塞启动流程）
  App.syncWakeConfig();
});
