import type { AppKernel, VoiceMode } from '../types/app-kernel.js';

export default (function init(App: AppKernel) {
  /* ============================================================
   *  VAD 自动对话模式（无需按住，说话即录、停顿即发）
   * ============================================================ */
  let vadEmaVol = 0;       // 指数移动平均音量（平滑）
  let vadVoiceEma = 0;     // 语音特性评分 EMA（平滑，防单帧抖动）
  let vadNoiseFloor = 0;   // 自适应环境噪声底（缓慢跟踪非语音段音量）
  let vadRecordStart = 0;  // 本次录音开始时间（自适应静音超时用）
  let vadSpeechLevel = 0;  // 录音期间语音音量水平（峰值保持+慢衰减，静音判定相对基准）
  const VAD_EMA_ALPHA = 0.35;  // EMA 平滑系数（越低越平滑，越高越灵敏）
  // 语音评分 EMA：攻击快、释放慢 —— 人声一来立刻确认（提速），噪声走后缓慢回落（防误判）
  const VAD_VOICE_EMA_ATTACK = 0.80;
  const VAD_VOICE_EMA_RELEASE = 0.12;
  const VAD_NOISE_FLOOR_ALPHA = 0.02; // 噪声底跟踪速度（很慢，防把说话当噪声）
  const VAD_NOISE_MARGIN = 1.6;       // 环境吵闹时：有效阈值 = 噪声底 × 此倍数
  const VAD_NOISE_FLOOR_MAX = 0.10;   // 噪声底上限，防止异常累积导致完全失灵
  // 自适应静音超时：说话越久意图越明确，超时越短（短句多等、长句快切）。
  // 阈值取"保守档"：自然语速的句内停顿可达 0.5~1s，超时过短会把长句切在中间，
  // 导致只识别出一两个字。
  const VAD_SILENCE_SHORT_MS = 2.0 * 1000;   // 短句（<2s）保持默认宽松超时
  const VAD_SILENCE_MID_MS = 1600;           // 中等长度（2~6s）：普通话句内停顿可到 1.5s，太短会把长句切在半句
  const VAD_SILENCE_LONG_MS = 1250;          // 长句（>6s）：适度快速切出，但保留足够的句内停顿余量
  const VAD_SILENCE_MID_BOUND_MS = 6.0 * 1000; // 中/长句分界
  const VAD_SPEECH_DECAY = 0.997;            // 语音水平每帧衰减（约 4s 回落到 30%）
  const VAD_SILENCE_RATIO = 0.30;            // 静音判定 = 语音水平 × 此比例（相对阈值）
  const VAD_MAX_RECORD_MS = 30000;           // 录音上限（防噪声环境永远切不断）

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
    if (fullEnergy < (i8000 - i100) * 8) return 0; // 能量太低，无有效信号（阈值随 bin 数缩放）

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
    // 不对称 EMA：人声一来快速上升（缩短确认时间），噪声走后缓慢回落（防误判）
    const alpha = score > vadVoiceEma ? VAD_VOICE_EMA_ATTACK : VAD_VOICE_EMA_RELEASE;
    vadVoiceEma = alpha * score + (1 - alpha) * vadVoiceEma;
    // 唤醒待机：放宽人声门槛（只要求"明显对噪声发声"即可进录音，
    // 是否命中唤醒词由服务端做模糊/拼音匹配裁决，避免过于严格的频谱过滤漏听唤醒词）
    const threshold = App.vadRelaxVoice
      ? Math.max(0.22, App.VAD_VOICE_SCORE_THRESHOLD - 0.18)
      : App.VAD_VOICE_SCORE_THRESHOLD;
    return vadVoiceEma >= threshold;
  };
  App.vadResetVoiceEma = function vadResetVoiceEma() { vadVoiceEma = 0; };

  /** 自适应人声确认窗口：音量越强（清晰大声）确认越快，最小 50ms */
  App.vadGetConfirmMs = function vadGetConfirmMs(vol: number) {
    if (vol > App.VAD_THRESHOLD * 2) return 50;
    return App.VAD_VOICE_CONFIRM_MS;
  };

  /** 自适应静音超时：短句宽容（防切断），长句适度快速切出（提速） */
  App.vadGetSilenceMs = function vadGetSilenceMs() {
    if (!vadRecordStart) return App.VAD_SILENCE_MS;
    const dur = performance.now() - vadRecordStart;
    if (dur < VAD_SILENCE_SHORT_MS) return App.VAD_SILENCE_MS;
    if (dur < VAD_SILENCE_MID_BOUND_MS) return VAD_SILENCE_MID_MS;
    return VAD_SILENCE_LONG_MS;
  };

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
      const e = err as DOMException;
      console.error('[VAD] 启动失败:', e.name, e.message);
      App.vadStream = null;
      App.micStream = null;
      let msg = '无法访问麦克风：' + (e.message || e.name);
      if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
        msg = '麦克风权限被拒绝，请在浏览器地址栏点击锁形图标重新授权';
      } else if (e.name === 'NotReadableError') {
        msg = '麦克风被其他应用占用，请关闭其他使用麦克风的程序';
      } else if (e.name === 'NotFoundError') {
        msg = '未找到麦克风设备';
      } else if (e.message === '麦克风未就绪') {
        msg = '麦克风未就绪，请重新开启自动对话';
      }
      App.showToast(msg);
      return false;
    }
    App.ensureAudioCtx();
    App.vadAnalyser = App.audioCtx!.createAnalyser();
    // 高帧率档用 2048 点 FFT：频率分辨率更高，人声特性判断更准（bin 带宽 ≈47Hz）
    App.vadAnalyser.fftSize = App.perfTier === 'high' ? 2048 : 1024;
    App.vadAnalyser.smoothingTimeConstant = 0.2;
    App.vadData = new Uint8Array(App.vadAnalyser.frequencyBinCount);
    const src = App.audioCtx!.createMediaStreamSource(App.vadStream!);
    src.connect(App.vadAnalyser);
    vadEmaVol = 0;
    vadVoiceEma = 0; // 重置语音特性评分
    vadNoiseFloor = 0; // 重置噪声底
    vadRecordStart = 0;
    vadSpeechLevel = 0;
    console.log('[VAD] 自动对话模式已启动 (fftSize=' + App.vadAnalyser.fftSize + ')');
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
    App.vadRelaxVoice = false; // 恢复严格人声门槛
    vadEmaVol = 0;
    vadVoiceEma = 0; // 重置语音特性评分
    vadNoiseFloor = 0; // 重置噪声底
    vadRecordStart = 0;
    vadSpeechLevel = 0;

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
  App.setVoiceMode = function setVoiceMode(mode: VoiceMode) {
    App.voiceMode = mode;
    localStorage.setItem('dabai.voiceMode', mode);
    const modeBtn = App.$('voice-mode-btn');
    const wakeBtn = App.$('wake-mode-btn');
    if (mode === 'auto') {
      // 默认视为手动开启；由 onWakeOk 在切换完成后标记为「唤醒开启」
      App._enteredViaWake = false;
      if (wakeBtn) wakeBtn.classList.remove('active');
      App.startVADMode().then(ok => {
        if (!ok) {
          App.voiceMode = 'press';
          localStorage.setItem('dabai.voiceMode', 'press');
          App.voiceBtn!.classList.remove('auto');
          App.voiceBtn!.title = '按住说话';
          if (modeBtn) modeBtn.classList.remove('active');
          return;
        }
        App.vadState = 'idle';
        App.vadLoop();
        App.voiceBtn!.classList.add('auto');
        App.voiceBtn!.title = '自动对话中（用左侧按钮切回按住说话）';
        if (modeBtn) modeBtn.classList.add('active');
        App.showToast('已切换为自动对话 · 直接说话即可');
        App.sendAIAction('（用户解放了双手，现在你能一直听到Ta的声音了，可以更自然随意地聊天）', true);
      });
    } else if (mode === 'wake') {
      // 唤醒词待机：VAD 持续聆听但只做唤醒判定，说唤醒词才开启对话
      if (modeBtn) modeBtn.classList.remove('active');
      App.startVADMode().then(ok => {
        if (!ok) {
          App.voiceMode = 'press';
          localStorage.setItem('dabai.voiceMode', 'press');
          App.voiceBtn!.classList.remove('auto');
          App.voiceBtn!.title = '按住说话';
          if (wakeBtn) wakeBtn.classList.remove('active');
          return;
        }
        // 进入待机：清掉上次唤醒失败的冷却与语音评分残留，确保立刻能听到唤醒词
        App._wakeRetryAt = 0;
        App.vadResetVoiceEma();
        App.vadState = 'idle';
        App.vadLoop();
        App.voiceBtn!.classList.add('auto'); // 复用聆听高亮样式，提示正在监听
        App.voiceBtn!.title = '唤醒词待机中（说唤醒词开始对话）';
        if (wakeBtn) wakeBtn.classList.add('active');
        const words = (App.wakeWords && App.wakeWords.length ? App.wakeWords : ['大白']).join(' / ');
        App.showToast(`已进入唤醒词待机 · 说「${words}」开始对话`);
      });
    } else {
      if (wakeBtn) wakeBtn.classList.remove('active');
      App.stopVADMode();
      App.voiceBtn!.classList.remove('auto');
      App.voiceBtn!.title = '按住说话';
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
    // 只统计语音频段 (100Hz ~ 3500Hz)：
    // 低于 100Hz 是空调/风扇隆隆声，高于 3500Hz 是键盘/鼠标/金属噪音，
    // 都不计入，避免环境噪音抬高音量导致误触发
    const sampleRate = App.audioCtx ? App.audioCtx.sampleRate : 48000;
    const binHz = sampleRate / 2 / App.vadData.length;
    const minBin = Math.max(1, Math.floor(100 / binHz));
    const maxBin = Math.min(Math.ceil(3500 / binHz), App.vadData.length);
    for (let i = minBin; i < maxBin; i++) {
      sum += App.vadData[i];
    }
    const raw = sum / (maxBin - minBin) / 255;  // 归一化到 0~1
    // EMA 平滑，减少单帧抖动
    vadEmaVol = VAD_EMA_ALPHA * raw + (1 - VAD_EMA_ALPHA) * vadEmaVol;
    return vadEmaVol;
  };
  App.vadLoop = function vadLoop() {
    // 'auto'（自动对话）与 'wake'（唤醒词待机）都需要 VAD 持续聆听：
    // 待机模式做「录音→唤醒判定」，自动模式做「说话即录、停顿即发」。
    // 原来只放行 'auto'，导致唤醒待机模式整条监听循环从未运行，
    // 麦克风虽已授权但永远不会录音，唤醒词自然无法唤醒。
    if (App.voiceMode !== 'auto' && App.voiceMode !== 'wake') return;

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
    // 自动对话会话空闲超时 → 自动回到唤醒待机（保证「只有叫唤醒词才进入聆听」）
    if (App.checkAutoReturnStandby && App.checkAutoReturnStandby(now)) return;
    // 唤醒词待机模式：AI 说话/思考时同样只做「录音→唤醒判定」，
    // 不走打断分支（是否打断由服务端唤醒匹配结果决定，防止环境音误打断）
    const wakeStandby = App.voiceMode === 'wake';
    // 待机模式放宽人声评分门槛（唤醒判定交给服务端），自动模式保持严格
    App.vadRelaxVoice = wakeStandby;

    // 有效音量阈值：安静环境用固定下限；环境吵闹时抬高到「噪声底 × 倍数」，
    // 保证说话声必须显著高于环境噪声才触发（自适应，不依赖单一固定值）
    const effThreshold = vadNoiseFloor > App.VAD_THRESHOLD
      ? Math.min(vadNoiseFloor * VAD_NOISE_MARGIN, VAD_NOISE_FLOOR_MAX)
      : App.VAD_THRESHOLD;

    // AI 说话中：检测打断。
    // 打断判定用"音量 + 稍长确认窗口"（用户开口说话必然高音量），
    // 不用人声特性评分（评分EMA有爬升延迟，会导致打断迟钝/失效）。
    // 用户 VAD 输入必须能随时打断 AI 输出。
    if (!wakeStandby && App.currentState === App.State.SPEAKING) {
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

    // AI 思考中：不播放声音，可被用户语音打断（说句话把 AI 从思考中拉回聆听）。
    // 思考时 AI 无声 → 不存在自打断风险，用"音量 + 人声特性"双确认：
    // 环境噪音/音乐被语音评分过滤，只有真人声才打断思考。
    if (!wakeStandby && App.currentState === App.State.THINKING) {
      const isVoice = App.vadIsHumanVoice();
      const voiceDetected = vol > App.VAD_THRESHOLD && isVoice;
      if (voiceDetected) {
        if (App.vadInterruptStart === 0) App.vadInterruptStart = now;
        if (now - App.vadInterruptStart > App.VAD_INTERRUPT_MS) {
          console.log('[VAD] 检测到用户输入，打断AI思考，进入聆听 vol=', vol.toFixed(3));
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

    // 唤醒待机 + AI 说话中：不录音（TTS 外放回声会被麦克风拾取，
    // 若允许录音会反复触发唤醒判断 → 一直"聆听中"）；
    // AI 说完后由 vadResumeAfterSpeak 复位，再恢复待机监听唤醒词
    if (wakeStandby && App.currentState === App.State.SPEAKING) {
      App.vadVoiceStart = 0;
      return;
    }

    // IDLE：自动检测说话开始（人声特性 + 音量达标，连续确认防误判）
    if (App.vadState === 'idle') {
      const isVoice = App.vadIsHumanVoice();
      const voiceDetected = vol > effThreshold && isVoice;
      // 唤醒待机：唤醒失败后的冷却期内忽略声音，防止环境噪音连发 → "一直聆听"
      if (voiceDetected && wakeStandby && now < (App._wakeRetryAt || 0)) {
        App.vadVoiceStart = 0;
      } else if (voiceDetected) {
        // 连续多帧确认：防止单帧噪声/偶发误判（音量越大确认越快）
        if (App.vadVoiceStart === 0) App.vadVoiceStart = now;
        if (now - App.vadVoiceStart > App.vadGetConfirmMs(vol)) {
          App.vadVoiceStart = 0;
          App.startVADRecording();
        }
      } else {
        App.vadVoiceStart = 0;
        // 无语音时缓慢跟踪噪声底（只在安静段更新，避免把说话声当噪声）
        if (vadNoiseFloor === 0 || vol < vadNoiseFloor) {
          vadNoiseFloor = vol;
        } else {
          vadNoiseFloor += (vol - vadNoiseFloor) * VAD_NOISE_FLOOR_ALPHA;
        }
        vadNoiseFloor = Math.min(vadNoiseFloor, VAD_NOISE_FLOOR_MAX);
      }
    } else if (App.vadState === 'recording') {
      // 录音上限：防止噪声环境永远切不断；唤醒待机只喊唤醒词（短句），上限更短
      const maxRecord = wakeStandby ? 6000 : VAD_MAX_RECORD_MS;
      if (vadRecordStart && now - vadRecordStart > maxRecord) {
        console.log('[VAD] 录音达上限，强制结束');
        App.stopVADRecording();
        App.vadSilenceStart = 0;
      }
      // 跟踪语音音量水平：峰值保持 + 缓慢衰减（适应麦克风增益/说话音量，
      // 轻声细语时静音阈值也跟着降低，不会把说话中的小停顿误判为静音）
      if (vol > vadSpeechLevel) {
        vadSpeechLevel = vol;
      } else {
        vadSpeechLevel *= VAD_SPEECH_DECAY;
      }
      // 相对静音阈值 = 语音水平的 30%（兜底下限 = 环境阈值的一半）
      const silThr = Math.max(effThreshold * 0.5, vadSpeechLevel * VAD_SILENCE_RATIO);
      // 判停阈值再与"噪声底附近"取大：环境噪声较大时，噪声本身就会把音量顶在
      // silThr 之上导致永远录不完（直到 30s 上限），此时以噪声底为基准判定停顿
      const pauseThr = Math.max(silThr, vadNoiseFloor * 1.25);
      if (vol < pauseThr) {
        if (App.vadSilenceStart === 0) App.vadSilenceStart = now;
        if (now - App.vadSilenceStart > App.vadGetSilenceMs()) {
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
    vadRecordStart = performance.now(); // 自适应静音超时基准
    vadNoiseFloor = 0; // 录音中环境已变化，重置噪声底
    vadSpeechLevel = 0; // 重置语音水平基准

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
      // 语音识别用 96kbps Opus：比 48k 保留更多辅音细节，提升识别准确率
      // （文件仍足够小，上传/转码速度几乎无感）
      App.vadRecorder = mime
        ? new MediaRecorder(clonedStream, { mimeType: mime, audioBitsPerSecond: 96000 })
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
    App.vadRecorder!.ondataavailable = e => {
      if (e.data && e.data.size > 0) App.vadChunks.push(e.data);
    };
    // 录音过程出错（设备被抢占/克隆轨中断等）→ 静默重建 VAD，不惊扰用户
    App.vadRecorder!.onerror = ev => {
      console.error('[VAD] MediaRecorder 出错:', (ev.error && ev.error.name) || 'unknown');
      try { App.vadRecorder!.stop(); } catch (e2) {}
    };
    App.vadRecorder!.onstop = () => {
      const chunks = App.vadChunks;
      App.vadChunks = [];
      const mimeType = App.vadRecorder!.mimeType || 'audio/webm';
      const blob = new Blob(chunks, { type: mimeType });
      // 停止并释放克隆轨
      if (App._vadClonedTrack) {
        try { App._vadClonedTrack.stop(); } catch (e) {}
        App._vadClonedTrack = null;
      }
      const isWakeSend = App.voiceMode === 'wake';
      if (blob.size < 3000) {
        App.vadState = 'idle';
        if (!isWakeSend) App.showToast('声音太短，再说一次？');
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => {
        if (isWakeSend) {
          // 唤醒词待机：音频带 wake_check 标记，由服务端判定是否唤醒
          App.sendAudioBase64(reader.result as string, mimeType, true);
        } else {
          App.sendAudioBase64(reader.result as string, mimeType);
          App.showTyping();
        }
      };
      reader.readAsDataURL(blob);
    };
    App.vadRecorder!.start(200);
    // 唤醒待机：不切「聆听中」，保持待机外观（只有真正唤醒进入对话才显示聆听）；
    // 识别结果由 wake_ok / wake_fail 提示。
    if (App.voiceMode !== 'wake') {
      App.setState(App.State.LISTENING);
    }
  };
  App.stopVADRecording = function stopVADRecording() {
    if (!App.vadRecorder || App.vadRecorder.state === 'inactive') {
      App.vadState = 'idle';
      vadRecordStart = 0;
      return;
    }
    App.vadState = 'idle';
    vadRecordStart = 0;
    vadSpeechLevel = 0;
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
    vadNoiseFloor = 0; // 重置噪声底，重新适应当前环境
    vadSpeechLevel = 0;
  };
});
