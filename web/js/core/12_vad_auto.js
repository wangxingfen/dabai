export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  VAD 自动对话模式（无需按住，说话即录、停顿即发）
   * ============================================================ */
  let vadEmaVol = 0;       // 指数移动平均音量（平滑）
  let vadVoiceEma = 0;     // 语音特性评分 EMA（平滑，防单帧抖动）
  const VAD_EMA_ALPHA = 0.35;  // EMA 平滑系数（越低越平滑，越高越灵敏）
  const VAD_VOICE_EMA_ALPHA = 0.35; // 语音评分平滑系数（比音量稍慢，防噪声单帧误判）

  /**
   * 人声特性检测（稳健版）：基于"语音频段能量占比 + 频谱起伏度"。
   * 人声特性：
   *   1) 能量集中在 200~3500Hz 语音频段（低频环境噪音/高频键盘声占比低）
   *   2) 频谱有明显起伏（元音谐波峰谷交替），而非平坦白噪
   * 用相对量（占比/方差），不依赖绝对量纲，对真人声宽容、对稳态噪音严格。
   * 返回 0~1 语音特性评分。
   */
  App.vadIsVoice = function vadIsVoice() {
    if (!App.vadAnalyser || !App.vadData) return 0;
    App.vadAnalyser.getByteFrequencyData(App.vadData);
    const N = App.vadData.length;
    const sampleRate = App.audioCtx ? App.audioCtx.sampleRate : 48000;
    const binHz = sampleRate / 2 / N;

    // 1) 各频段能量
    const i200 = Math.max(1, Math.floor(200 / binHz));   // 语音频段下限
    const i3500 = Math.min(N, Math.ceil(3500 / binHz));  // 语音频段上限
    const i100 = Math.max(1, Math.floor(100 / binHz));   // 全频段下限（含低频噪音）
    const i8000 = Math.min(N, Math.ceil(8000 / binHz));  // 全频段上限（含高频噪音）

    let voiceEnergy = 0, fullEnergy = 0;
    for (let i = i100; i < i8000; i++) fullEnergy += App.vadData[i];
    for (let i = i200; i < i3500; i++) voiceEnergy += App.vadData[i];
    if (fullEnergy < 800) return 0; // 能量太低，无有效信号

    // 2) 语音频段占比：人声能量高度集中在 200~3500Hz
    const bandRatio = fullEnergy > 0 ? voiceEnergy / fullEnergy : 0;

    // 3) 频谱起伏度：语音频段内相邻 bin 差分的平均绝对值（人声谐波峰谷交替大）
    let diffSum = 0, diffCnt = 0;
    let prev = App.vadData[i200];
    for (let i = i200 + 1; i < i3500; i++) {
      diffSum += Math.abs(App.vadData[i] - prev);
      prev = App.vadData[i];
      diffCnt++;
    }
    const meanBin = fullEnergy / (i8000 - i100); // 平均 bin 能量
    const variance = diffCnt > 0 && meanBin > 0 ? (diffSum / diffCnt) / meanBin : 0;

    // 4) 综合评分：语音频段占比（主）+ 频谱起伏（辅）
    //    人声：bandRatio≈0.8~0.95, variance≈0.8~2.5 → 评分 0.6~0.9
    //    白噪：bandRatio≈0.45, variance≈0.2 → 评分 ≈0.3
    //    空调/风扇（低频隆隆）：bandRatio≈0.3~0.5, variance≈0.2 → 评分 <0.35
    //    键盘/金属（高频尖刺）：bandRatio≈0.3~0.5 → 评分 <0.4
    let score = bandRatio * 0.65 + Math.min(1, variance / 1.5) * 0.35;

    // 5) 惩罚项：低频(100~200Hz)能量异常集中（空调/风扇/引擎的隆隆声）
    let lowEnergy = 0;
    for (let i = i100; i < i200; i++) lowEnergy += App.vadData[i];
    if (lowEnergy > fullEnergy * 0.55) score *= 0.4;

    return Math.max(0, Math.min(1, score));
  };

  /** 综合判断当前是否为"人声"：能量达标 + 语音特性评分达标 */
  App.vadIsHumanVoice = function vadIsHumanVoice() {
    if (!App.VAD_VOICE_ENABLED) return true; // 关闭过滤 = 原逻辑（仅能量）
    const score = App.vadIsVoice();
    // EMA 平滑语音评分，抑制短暂噪声误判
    vadVoiceEma = VAD_VOICE_EMA_ALPHA * score + (1 - VAD_VOICE_EMA_ALPHA) * vadVoiceEma;
    return vadVoiceEma >= App.VAD_VOICE_SCORE_THRESHOLD;
  };
  App.vadResetVoiceEma = function vadResetVoiceEma() { vadVoiceEma = 0; };

  App.startVADMode = async function startVADMode() {
    if (App.vadStream) return true;
    try {
      const { stream } = await App._acquireMicStream();
      // 校验流状态
      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0 || audioTracks.every(t => t.readyState !== 'live')) {
        throw new Error('麦克风未就绪');
      }
      App.vadStream = stream;
      App.micStream = stream;
    } catch (err) {
      console.error('[VAD] 启动失败:', err.name, err.message);
      App.vadStream = null;
      App.micStream = null;
      let msg = '无法访问麦克风：' + (err.message || err.name);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = '麦克风权限被拒绝，请在浏览器地址栏点击锁形图标重新授权';
      } else if (err.name === 'NotReadableError') {
        msg = '麦克风被其他应用占用，请关闭其他使用麦克风的程序';
      } else if (err.name === 'NotFoundError') {
        msg = '未找到麦克风设备';
      } else if (err.message === '麦克风未就绪') {
        msg = '麦克风未就绪，请重新开启自动对话';
      }
      App.showToast(msg);
      return false;
    }
    App.ensureAudioCtx();
    App.vadAnalyser = App.audioCtx.createAnalyser();
    App.vadAnalyser.fftSize = 1024;
    App.vadAnalyser.smoothingTimeConstant = 0.2;
    App.vadData = new Uint8Array(App.vadAnalyser.frequencyBinCount);
    const src = App.audioCtx.createMediaStreamSource(App.vadStream);
    src.connect(App.vadAnalyser);
    vadEmaVol = 0;
    vadVoiceEma = 0; // 重置语音特性评分
    console.log('[VAD] 自动对话模式已启动 (fftSize=1024)');
    return true;
  };
  App.stopVADMode = function stopVADMode() {
    if (App.vadRAF) {
      cancelAnimationFrame(App.vadRAF);
      App.vadRAF = null;
    }
    // 停止当前录音（如果有的话）
    if (App.vadRecorder && App.vadRecorder.state !== 'inactive') {
      try { App.vadRecorder.stop(); } catch (e) {}
    }
    App.vadRecorder = null;
    // 停止克隆的轨（如果有的话）
    if (App._vadClonedTrack) {
      try { App._vadClonedTrack.stop(); } catch (e) {}
      App._vadClonedTrack = null;
    }
    App.vadStream = null;
    App.vadAnalyser = null;
    App.vadData = null;
    App.vadState = 'idle';
    App.vadSilenceStart = 0;
    App.vadInterruptStart = 0;
    App.vadVoiceStart = 0;
    vadEmaVol = 0;
    vadVoiceEma = 0; // 重置语音特性评分

    // 延迟释放麦克风流，给切回按住模式后立刻录音留出复用窗口
    if (App._micStreamReleaseTimer) clearTimeout(App._micStreamReleaseTimer);
    App._micStreamReleaseTimer = setTimeout(() => {
      App._micStreamReleaseTimer = null;
      // 如果当前仍在自动模式或正在录音，不要释放
      if (App.voiceMode === 'auto' || App.isRecording) return;
      if (App.micStream) {
        console.log('[VAD] 延迟释放麦克风流');
        App.micStream.getTracks().forEach(t => t.stop());
        App.micStream = null;
      }
    }, 3000);
  };
  App.setVoiceMode = function setVoiceMode(mode) {
    App.voiceMode = mode;
    localStorage.setItem('dabai.voiceMode', mode);
    const modeBtn = App.$('voice-mode-btn');
    if (mode === 'auto') {
      App.startVADMode().then(ok => {
        if (!ok) {
          App.voiceMode = 'press';
          localStorage.setItem('dabai.voiceMode', 'press');
          App.voiceBtn.classList.remove('auto');
          App.voiceBtn.title = '按住说话';
          if (modeBtn) modeBtn.classList.remove('active');
          return;
        }
        App.vadState = 'idle';
        App.vadLoop();
        App.voiceBtn.classList.add('auto');
        App.voiceBtn.title = '自动对话中（用左侧按钮切回按住说话）';
        if (modeBtn) modeBtn.classList.add('active');
        App.showToast('已切换为自动对话 · 直接说话即可');
        App.sendAIAction('（用户解放了双手，现在你能一直听到Ta的声音了，可以更自然随意地聊天）', true);
      });
    } else {
      App.stopVADMode();
      App.voiceBtn.classList.remove('auto');
      App.voiceBtn.title = '按住说话';
      if (modeBtn) modeBtn.classList.remove('active');
      App.showToast('已切换为按住说话');
      App.sendAIAction('（用户切换了对话方式，现在需要按住按钮才能听到Ta说话，等Ta准备好再说）', true);
    }
  };
  /* 频率域能量检测：比时域 RMS 更稳定，能综合捕捉全频段语音能量 */
  App.vadGetVolume = function vadGetVolume() {
    if (!App.vadAnalyser || !App.vadData) return 0;
    App.vadAnalyser.getByteFrequencyData(App.vadData);
    let sum = 0;
    // 只统计语音频段 (85Hz ~ 3500Hz ≈ 频段 bin 3~110，fftSize=1024 → 512 bins, 采样率48k → bin带宽≈93.75Hz)
    const minBin = 1;   // ~94Hz，跳过直流分量
    const maxBin = Math.min(110, App.vadData.length);
    for (let i = minBin; i < maxBin; i++) {
      sum += App.vadData[i];
    }
    const raw = sum / (maxBin - minBin) / 255;  // 归一化到 0~1
    // EMA 平滑，减少单帧抖动
    vadEmaVol = VAD_EMA_ALPHA * raw + (1 - VAD_EMA_ALPHA) * vadEmaVol;
    return vadEmaVol;
  };
  App.vadLoop = function vadLoop() {
    if (App.voiceMode !== 'auto') return;

    // 检查 AudioContext 是否被浏览器暂停（长时间空闲后浏览器会挂起音频上下文）
    if (App.audioCtx && App.audioCtx.state === 'suspended') {
      App.audioCtx.resume().then(() => console.log('[VAD] AudioContext 已自动恢复'));
      // 等待下一个周期再检测，让 resume 生效
      App.vadRAF = requestAnimationFrame(App.vadLoop);
      return;
    }
    if (!App.vadAnalyser) return;

    // 检查 vadStream 是否还活着（浏览器长时间后台可能回收麦克风流）
    if (App.vadStream && !App.vadStream.active) {
      console.warn('[VAD] 麦克风流已失效，尝试重建…');
      App.stopVADMode();
      App.startVADMode().then(ok => {
        if (ok) {
          App.vadState = 'idle';
          App.vadLoop();
        } else {
          App.showToast('自动对话已断开，请切回按住说话');
          App.voiceMode = 'press';
        }
      });
      return;
    }
    App.vadRAF = requestAnimationFrame(App.vadLoop);

    // 性能分级：降频VAD检测以节省CPU
    if (!App.shouldVADFrame()) return;

    const vol = App.vadGetVolume();
    const now = performance.now();

    // AI 说话中：检测打断。
    // 打断判定用"音量 + 稍长确认窗口"（用户开口说话必然高音量），
    // 不用人声特性评分（评分EMA有爬升延迟，会导致打断迟钝/失效）。
    // 用户 VAD 输入必须能随时打断 AI 输出。
    if (App.currentState === App.State.SPEAKING) {
      if (vol > App.VAD_INTERRUPT_THRESHOLD) {
        if (App.vadInterruptStart === 0) App.vadInterruptStart = now;
        if (now - App.vadInterruptStart > App.VAD_INTERRUPT_MS) {
          console.log('[VAD] 检测到用户输入，打断AI输出 vol=', vol.toFixed(3));
          App.triggerInterrupt();
          App.vadInterruptStart = 0;
          // 立即切到 IDLE，防止本函数下一帧再次进打断分支
          App.currentState = App.State.IDLE;
          if (App.vadState === 'idle') App.startVADRecording();
        }
      } else {
        App.vadInterruptStart = 0;
      }
      return;
    }

    // AI 思考中：不录音（避免录到环境音/提示音）
    if (App.currentState === App.State.THINKING) {
      App.vadInterruptStart = 0;
      return;
    }

    // IDLE：自动检测说话开始（人声特性 + 音量达标，连续确认防误判）
    if (App.vadState === 'idle') {
      const isVoice = App.vadIsHumanVoice();
      const voiceDetected = vol > App.VAD_THRESHOLD && isVoice;
      if (voiceDetected) {
        // 连续多帧确认：防止单帧噪声/偶发误判
        if (App.vadVoiceStart === 0) App.vadVoiceStart = now;
        if (now - App.vadVoiceStart > App.VAD_VOICE_CONFIRM_MS) {
          App.vadVoiceStart = 0;
          App.startVADRecording();
        }
      } else {
        App.vadVoiceStart = 0;
      }
    } else if (App.vadState === 'recording') {
      if (vol < App.VAD_THRESHOLD * 0.6) {
        if (App.vadSilenceStart === 0) App.vadSilenceStart = now;
        if (now - App.vadSilenceStart > App.VAD_SILENCE_MS) {
          App.stopVADRecording();
          App.vadSilenceStart = 0;
        }
      } else {
        App.vadSilenceStart = 0;
      }
    }
  };
  App.startVADRecording = function startVADRecording() {
    if (!App.vadStream) return;
    // 防止上一个 recorder 还没完全停止就创建新的
    if (App.vadRecorder && App.vadRecorder.state === 'recording') {
      try { App.vadRecorder.stop(); } catch (e) {}
    }
    App.vadState = 'recording';
    App.vadChunks = [];
    App.vadSilenceStart = 0;

    // 克隆音频轨创建独立的 MediaStream，Chrome 会为新流写入完整 EBML 头部
    try {
      const originalTrack = App.vadStream.getAudioTracks()[0];
      if (!originalTrack || originalTrack.readyState !== 'live') {
        console.warn('[VAD] 原始音频轨不可用，重建 VAD');
        App.stopVADMode();
        App.startVADMode().then(ok => {
          if (ok) { App.vadState = 'idle'; App.vadLoop(); }
          else { App.showToast('自动对话已断开'); App.voiceMode = 'press'; }
        });
        return;
      }
      // 停止上一个克隆轨
      if (App._vadClonedTrack) {
        try { App._vadClonedTrack.stop(); } catch (e) {}
        App._vadClonedTrack = null;
      }
      App._vadClonedTrack = originalTrack.clone();
      const clonedStream = new MediaStream([App._vadClonedTrack]);

      const mime = App.pickRecorderMime();
      App.vadRecorder = mime
        ? new MediaRecorder(clonedStream, { mimeType: mime, audioBitsPerSecond: 128000 })
        : new MediaRecorder(clonedStream);
    } catch (e) {
      console.warn('[VAD] MediaRecorder 创建失败，尝试重建流:', e);
      App.stopVADMode();
      App.vadState = 'idle';
      App.startVADMode().then(ok => {
        if (ok) App.vadLoop();
        else { App.showToast('自动对话已断开'); App.voiceMode = 'press'; }
      });
      return;
    }
    App.vadRecorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) App.vadChunks.push(e.data);
    };
    App.vadRecorder.onstop = () => {
      const chunks = App.vadChunks;
      App.vadChunks = [];
      const mimeType = App.vadRecorder.mimeType || 'audio/webm';
      const blob = new Blob(chunks, { type: mimeType });
      // 停止并释放克隆轨
      if (App._vadClonedTrack) {
        try { App._vadClonedTrack.stop(); } catch (e) {}
        App._vadClonedTrack = null;
      }
      if (blob.size < 3000) {
        App.vadState = 'idle';
        App.showToast('声音太短，再说一次？');
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => {
        App.sendAudioBase64(reader.result, mimeType);
        App.showTyping();
      };
      reader.readAsDataURL(blob);
    };
    App.vadRecorder.start(200);
    App.setState(App.State.LISTENING);
  };
  App.stopVADRecording = function stopVADRecording() {
    if (!App.vadRecorder || App.vadRecorder.state === 'inactive') {
      App.vadState = 'idle';
      return;
    }
    App.vadState = 'idle';
    try {
      if (App.vadRecorder.state === 'recording') {
        App.vadRecorder.requestData();
      }
      App.vadRecorder.stop();
    } catch (e) {}
  };
  /* AI 说完后恢复监听（VAD 自动模式）：重置计时避免把尾音当用户说话 */
  App.vadResumeAfterSpeak = function vadResumeAfterSpeak() {
    App.vadSilenceStart = 0;
    App.vadInterruptStart = 0;
    App.vadVoiceStart = 0;
    App.vadState = 'idle';
    vadEmaVol = 0;  // 重置 EMA 避免残留
    vadVoiceEma = 0; // 重置语音特性评分，避免把 AI 自己的尾音当用户人声
  };
  /* ============================================================
   *  消息渲染
   * ============================================================ */
  // 聊天面板收起时新消息到达 → 提示切换按钮
});
