export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  TTS 播放 + 口型同步
   * ============================================================ */
  App.ensureAudioCtx = function ensureAudioCtx() {
    if (!App.audioCtx) {
      App.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      App.analyser = App.audioCtx.createAnalyser();
      App.analyser.fftSize = 256;
      App.analyser.smoothingTimeConstant = 0.35; // 适中平滑：既能跟踪音节，又能让字间快速归零
      App.analyserData = new Uint8Array(App.analyser.frequencyBinCount);
      // 分析器只需连接一次 destination
      App.analyser.connect(App.audioCtx.destination);
      console.log('[Audio] AudioContext + Analyser 已就绪');
    }
    if (App.audioCtx.state === 'suspended') {
      App.audioCtx.resume().then(() => console.log('[Audio] AudioContext resumed')).catch(() => {});
    }
  };
  App.playNextAudio = function playNextAudio() {
    if (App.audioQueue.length === 0) {
      // 队列暂时空但未收到 end → 允许后续 chunk 重新触发播放
      App.isPlayingQueue = false;
      return;
    }
    const chunk = App.audioQueue.shift();
    // 哨兵：整个回复结束
    if (chunk.end) {
      App.isPlayingQueue = false;
      if (App.pendingAIMsgEl) {
        App.pendingAIMsgEl.classList.remove('streaming');
        App.pendingAIMsgEl = null;
      }
      App.currentReplyText = '';
      App.currentReplySession = null;
      App.setState(App.State.IDLE);
      App.showSubtitle('');
      // VAD 自动模式：AI 说完后恢复监听
      if (App.voiceMode === 'auto') App.vadResumeAfterSpeak();
      return;
    }
    App.isPlayingQueue = true;

    // 无音频数据（TTS 失败），跳过只保留文本
    if (!chunk.audio_b64) {
      App.playNextAudio();
      return;
    }
    App.ensureAudioCtx();
    App.setState(App.State.SPEAKING);
    App.showSubtitle(App.currentReplyText);

    // 释放上一个 Audio + MediaElementSource + 销毁旧的 blob URL
    if (App.currentAudio) {
      App.currentAudio.onended = null;
      App.currentAudio.onerror = null;
      App.currentAudio.pause();
      // 断开 MediaElementSourceNode，防止 AudioContext 节点泄露
      if (App.currentAudioSource) {
        try { App.currentAudioSource.disconnect(); } catch (e) { /* 已断开 */ }
        App.currentAudioSource = null;
      }
      if (App.currentAudio._blobUrl) {
        URL.revokeObjectURL(App.currentAudio._blobUrl);
        App.currentAudio._blobUrl = null;
      }
      App.currentAudio.src = '';
    }

    // base64 → Blob → blob URL（播放完即时销毁，不积压缓存）
    const binary = atob(chunk.audio_b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: chunk.audio_mime || 'audio/mpeg' });
    const blobUrl = URL.createObjectURL(blob);

    App.currentAudio = new Audio(blobUrl);
    App.currentAudio._blobUrl = blobUrl; // 记录以便后续销毁
    // 每个 Audio 元素只能 createMediaElementSource 一次
    try {
      App.currentAudioSource = App.audioCtx.createMediaElementSource(App.currentAudio);
      App.currentAudioSource.connect(App.analyser);
    } catch (e) {
      console.warn('[Audio] createMediaElementSource 失败:', e.message);
    }
    App.currentAudio.onplay = () => {
      if (App.isPlayingQueue) App.setState(App.State.SPEAKING);
    };
    App.currentAudio.onended = () => {
      // 播完立即销毁 blob URL，不留缓存
      if (App.currentAudio && App.currentAudio._blobUrl) {
        URL.revokeObjectURL(App.currentAudio._blobUrl);
        App.currentAudio._blobUrl = null;
      }
      App.playNextAudio();
    };
    App.currentAudio.onerror = () => {
      console.warn('[Audio] 播放失败，跳过');
      if (App.currentAudio && App.currentAudio._blobUrl) {
        URL.revokeObjectURL(App.currentAudio._blobUrl);
        App.currentAudio._blobUrl = null;
      }
      App.playNextAudio();
    };
    App.currentAudio.play().catch(e => {
      console.warn('[Audio] play() rejected:', e);
      if (App.audioCtx && App.audioCtx.state === 'suspended') {
        App.showToast('点击页面任意位置以启用音频');
      }
      App.playNextAudio();
    });
  };
  /* 流式音频分片到达 */
  App.handleAudioChunk = function handleAudioChunk(msg) {
    // 过滤过期 session 的消息
    if (App.currentReplySession && msg.session_id && msg.session_id !== App.currentReplySession) return;
    App.removeTyping();
    // 首句到达时创建占位消息
    if (!App.pendingAIMsgEl) {
      App.pendingAIMsgEl = document.createElement('div');
      App.pendingAIMsgEl.className = 'msg ai streaming';
      App.messagesEl.appendChild(App.pendingAIMsgEl);
      App.scrollToBottom();
    }
    App.audioQueue.push({
      seq: msg.seq,
      text: msg.text,
      audio_b64: msg.audio_b64,
      audio_mime: msg.audio_mime
    });
    App.currentReplyText += msg.text;
    App.pendingAIMsgEl.textContent = App.currentReplyText;
    App.scrollToBottom();

    // 首句立即播放
    if (!App.isPlayingQueue) App.playNextAudio();
  };
  /* 整个回复结束 */
  App.handleAudioEnd = function handleAudioEnd(msg) {
    if (App.currentReplySession && msg.session_id && msg.session_id !== App.currentReplySession) return;
    // 用完整文本替换（防止流式拼接误差）
    if (msg.full_text && App.pendingAIMsgEl) {
      App.currentReplyText = msg.full_text;
      App.pendingAIMsgEl.textContent = App.currentReplyText;
    }
    // 推入结束哨兵
    App.audioQueue.push({
      seq: Infinity,
      text: '',
      audio_b64: null,
      audio_mime: null,
      end: true
    });
    if (!App.isPlayingQueue) App.playNextAudio();
  };
  /* 被打断 */
  App.handleInterrupted = function handleInterrupted() {
    App.clearAudioQueue();
    // 如果 VAD 正在录音中，不要覆盖录音状态；否则回到 IDLE
    if (App.vadState !== 'recording' && !App.isRecording) {
      App.setState(App.State.IDLE);
    }
    App.showSubtitle('');
    if (App.pendingAIMsgEl) {
      if (App.currentReplyText) App.pendingAIMsgEl.classList.remove('streaming');else App.pendingAIMsgEl.remove();
      App.pendingAIMsgEl = null;
    }
    App.currentReplyText = '';
    App.currentReplySession = null;
  };
  /* 清空播放队列 + 停止当前播放 */
  App.clearAudioQueue = function clearAudioQueue() {
    App.audioQueue = [];
    App.isPlayingQueue = false;
    if (App.currentAudio) {
      App.currentAudio.onended = null;
      App.currentAudio.onerror = null;
      App.currentAudio.pause();
      // 断开 MediaElementSourceNode，防止 AudioContext 节点泄露
      if (App.currentAudioSource) {
        try { App.currentAudioSource.disconnect(); } catch (e) { /* 已断开 */ }
        App.currentAudioSource = null;
      }
      if (App.currentAudio._blobUrl) {
        URL.revokeObjectURL(App.currentAudio._blobUrl);
        App.currentAudio._blobUrl = null;
      }
      App.currentAudio.src = '';
      App.currentAudio = null;
    }
  };
  /* 主动打断：停止本地播放 + 通知服务端取消 */
  App.triggerInterrupt = function triggerInterrupt() {
    App.clearAudioQueue();
    if (App.pendingAIMsgEl) {
      if (App.currentReplyText) App.pendingAIMsgEl.classList.remove('streaming');else App.pendingAIMsgEl.remove();
      App.pendingAIMsgEl = null;
    }
    App.currentReplyText = '';
    App.currentReplySession = null;
    if (App.ws && App.ws.readyState === WebSocket.OPEN) {
      App.ws.send(JSON.stringify({
        type: 'interrupt'
      }));
    }
  };
  /* ============================================================
   *  语音录制
   * ============================================================ */
  /* 录音格式优先级：兼容 iOS Safari（mp4）+ Chrome/Firefox（webm） */
});