export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  状态切换 / 性能分级系统
   * ============================================================ */
  // 性能分级: 'high'=桌面60fps | 'default'=移动30fps | 'low'=低功耗20fps
  App.perfTier = 'high';
  App._renderFrameSkip = 1;       // 每N帧渲染1次 (1=每帧, 2=隔帧30fps, 3=20fps)
  App._renderFrameCount = 0;     // 帧计数器
  App._vadFrameSkip = 1;          // VAD检测每N帧执行1次
  App._vadFrameCount = 0;

  /** 自动检测设备性能级别 */
  App.detectPerfTier = function detectPerfTier() {
    // 优先使用用户手动选择保存的级别
    try {
      const saved = localStorage.getItem('dabai.perfTier');
      if (saved && ['high', 'default', 'low'].includes(saved)) {
        App.setPerfTier(saved);
        return;
      }
    } catch (_) {}
    // 检查是否为移动设备
    const isMobile = /Android|iPhone|iPad|iPod|webOS/i.test(navigator.userAgent) ||
      ('ontouchstart' in window && window.innerWidth < 1024);
    // 检查是否低端设备（内存<4GB粗略判断）
    const lowMemory = navigator.deviceMemory && navigator.deviceMemory < 4;

    if (isMobile && lowMemory) {
      App.setPerfTier('low');
    } else if (isMobile) {
      App.setPerfTier('default');
    } else {
      App.setPerfTier('high');
    }
  };

  /** 设置性能级别并应用对应优化 */
  App.setPerfTier = function setPerfTier(tier) {
    App.perfTier = tier;
    switch (tier) {
      case 'high':
        App._renderFrameSkip = 1;   // 60fps
        App._vadFrameSkip = 1;      // 全速VAD
        App._targetDPR = Math.min(window.devicePixelRatio, 2);
        App._useAA = true;
        App._starCount = 200;
        break;
      case 'default':
        App._renderFrameSkip = 2;   // 30fps
        App._vadFrameSkip = 2;      // 半速VAD (~30fps)
        App._targetDPR = Math.min(window.devicePixelRatio, 1.5);
        App._useAA = false;         // 移动端关抗锯齿
        App._starCount = 80;
        break;
      case 'low':
        App._renderFrameSkip = 3;   // 20fps
        App._vadFrameSkip = 3;      // 1/3速VAD (~20fps)
        App._targetDPR = Math.min(window.devicePixelRatio, 1.0);
        App._useAA = false;
        App._starCount = 0;         // 低功耗下去掉粒子
        break;
    }
    // 运行时更新渲染器
    if (App.renderer) {
      App.renderer.setPixelRatio(App._targetDPR);
      // 低功耗下去掉星空粒子
      if (App.starField) {
        App.starField.visible = App._starCount > 0;
      }
    }
    // 更新帧率按钮显示
    const fpsLabel = document.getElementById('fps-label');
    if (fpsLabel) {
      fpsLabel.textContent = Math.round(60 / App._renderFrameSkip);
    }
    // 持久化
    try { localStorage.setItem('dabai.perfTier', tier); } catch (_) {}
    console.log('[Perf] 性能级别:', tier,
      '渲染:', Math.round(60 / App._renderFrameSkip) + 'fps',
      'VAD:', Math.round(60 / App._vadFrameSkip) + 'fps',
      'DPR:', App._targetDPR,
      'AA:', App._useAA);
  };

  /** 循环切换性能级别：high → default → low → high */
  App.cyclePerfTier = function cyclePerfTier() {
    const tiers = ['high', 'default', 'low'];
    const idx = tiers.indexOf(App.perfTier);
    const next = tiers[(idx + 1) % tiers.length];
    App.setPerfTier(next);
    const fps = Math.round(60 / App._renderFrameSkip);
    App.showToast('帧率: ' + fps + 'fps · ' + (fps <= 20 ? '极致省电' : fps <= 30 ? '流畅省电' : '最佳画质'));
  };

  /** 渲染帧节流：返回true表示本帧应该渲染 */
  App.shouldRenderFrame = function shouldRenderFrame() {
    App._renderFrameCount++;
    if (App._renderFrameCount >= App._renderFrameSkip) {
      App._renderFrameCount = 0;
      return true;
    }
    return false;
  };

  /** VAD帧节流：返回true表示本帧应该执行VAD检测 */
  App.shouldVADFrame = function shouldVADFrame() {
    App._vadFrameCount++;
    if (App._vadFrameCount >= App._vadFrameSkip) {
      App._vadFrameCount = 0;
      return true;
    }
    return false;
  };

  /* ============================================================
   *  动态渲染（自适应分辨率）+ 轻量内存巡检
   * ============================================================ */
  // 依据实测帧率自动升降采样率：帧率低 → 逐级降 DPR，恢复 → 逐级升回档位上限。
  // 只在性能档位允许的区间内浮动，不改变任何场景/游戏逻辑。
  App._adaptiveDPR = true;        // 动态渲染总开关
  App._fpsAccum = 0;
  App._fpsCount = 0;
  App._lastFpsCheck = 0;
  App._dprAdjustAt = 0;

  /** 每帧调用（仅渲染帧）：统计真实帧率，按需动态调整渲染分辨率 */
  App.adaptiveFrame = function adaptiveFrame(dt) {
    if (!App._adaptiveDPR || !App.renderer || App.perfTier === 'low') return;
    App._fpsAccum += dt;
    App._fpsCount++;
    const now = performance.now();
    if (now - App._lastFpsCheck < 2000) return;
    const avg = App._fpsAccum / Math.max(1, App._fpsCount);
    const fps = 1 / Math.max(avg, 0.001);
    const target = App.perfTier === 'high' ? 48 : 27;
    const maxDPR = App.perfTier === 'high' ? Math.min(window.devicePixelRatio || 1, 2) : 1.5;
    const minDPR = App.perfTier === 'high' ? 1.0 : 0.75;
    if (fps < target * 0.8 && App._targetDPR > minDPR && now - App._dprAdjustAt > 2500) {
      App._targetDPR = Math.max(minDPR, App._targetDPR - 0.25);
      App.renderer.setPixelRatio(App._targetDPR);
      App._dprAdjustAt = now;
      console.log('[Perf] 帧率 ' + Math.round(fps) + 'fps 偏低，动态降采样 → DPR ' + App._targetDPR.toFixed(2));
    } else if (fps >= target && App._targetDPR < maxDPR && now - App._dprAdjustAt > 6000) {
      App._targetDPR = Math.min(maxDPR, App._targetDPR + 0.25);
      App.renderer.setPixelRatio(App._targetDPR);
      App._dprAdjustAt = now;
    }
    App._fpsAccum = 0;
    App._fpsCount = 0;
    App._lastFpsCheck = now;
  };

  /** 回到当前性能档位的基准 DPR（进/出游戏、切换档位时调用） */
  App.resetAdaptiveDPR = function resetAdaptiveDPR() {
    const base = {
      high: Math.min(window.devicePixelRatio || 1, 2),
      default: 1.5,
      low: 1.0,
    }[App.perfTier] || 1;
    App._targetDPR = Math.min(window.devicePixelRatio || 1, base);
    if (App.renderer) App.renderer.setPixelRatio(App._targetDPR);
    App._fpsAccum = 0;
    App._fpsCount = 0;
    App._lastFpsCheck = 0;
    App._dprAdjustAt = 0;
  };

  /** 轻量内存巡检：主循环每 30 秒调用一次，释放可安全回收的资源 */
  App.memoryTick = function memoryTick() {
    try {
      // 游戏内部资源巡检（各游戏实现 memoryTick：清已播完音频等）
      if (App.currentGame && typeof App.currentGame.memoryTick === 'function') {
        App.currentGame.memoryTick();
      }
      // 释放已结束/暂停的 TTS blob URL（播放中不打断）
      if (App.currentAudio && App.currentAudio._blobUrl
        && App.currentState !== App.State.SPEAKING && App.currentAudio.paused) {
        try {
          URL.revokeObjectURL(App.currentAudio._blobUrl);
          App.currentAudio._blobUrl = null;
          App.currentAudio.src = '';
        } catch (e) { /* ignore */ }
      }
      // WebGL 统计信息自动归零，避免 info 计数长期累积（不影响渲染）
      if (App.renderer && App.renderer.info) App.renderer.info.autoReset = true;
    } catch (e) { /* 巡检失败不影响主流程 */ }
  };

  /** 进入游戏前调用：释放大厅残留资源 + 复位自适应分辨率 */
  App.prepareForGame = function prepareForGame() {
    App.resetAdaptiveDPR();
    if (App.memoryTick) App.memoryTick();
  };

  /** 退出游戏后调用：复位自适应分辨率，回到大厅最佳档位 */
  App.prepareForLobby = function prepareForLobby() {
    App.resetAdaptiveDPR();
    if (App.memoryTick) App.memoryTick();
  };

  /* ============================================================
   *  低功耗模式
   * ============================================================ */
  App.enterLowPowerMode = function enterLowPowerMode() {
    App.lowPowerMode = true;
    localStorage.setItem(App.LP_KEY, '1');
    const app = document.getElementById('app');
    const stage = App.$('stage');

    // 退出FPV/移动模式
    if (App.fpvMode) {
      App.exitFPV();
    }
    if (App.moveMode) {
      App.setMoveMode(false);
    }

    // 退出 VR 模式（WebXR 会话结束）
    if (App.exitXrMode && App.xrMode !== 'off') {
      App.exitXrMode();
    }

    // 清理VR晃动强度状态（晃动回调因 vs 为 null 直接返回，避免残留误加）
    App.vrShake = null;

    // 保存当前性能级别以便恢复
    App._preLPPerfTier = App.perfTier;
    // 切换到低功耗性能级别 (20fps, DPR=1, 关AA, 无粒子)
    App.setPerfTier('low');

    // 停止3D动画循环 —— 角色冻结固定在最后一帧，保留场景/渲染器/模型不销毁
    if (typeof cancelAnimationFrame !== 'undefined' && window._animFrame) {
      cancelAnimationFrame(window._animFrame);
      window._animFrame = null;
    }
    // 停止VAD循环
    if (App.vadRAF) {
      cancelAnimationFrame(App.vadRAF);
      App.vadRAF = null;
    }

    // 切换为全屏式布局：3D场景占满整屏，聊天框默认折叠
    app.classList.add('low-power');
    stage.classList.add('low-power');

    // 默认折叠聊天面板，聚焦角色；点击对话图标展开
    document.getElementById('chat-panel').classList.add('collapsed');
    document.getElementById('controls').classList.add('collapsed');

    // 隐藏3D专用按钮（保留低功耗/会话/TTS/全屏）
    [App.modelBtn, App.bgBtn, App.moveBtn, App.fpvBtn, App.resetCamBtn, App.camSettingsBtn, App.$('gyro-btn')].forEach(b => {
      if (b) b.style.display = 'none';
    });

    // 渲染一帧确保角色可见
    if (App.renderer && App.scene && App.camera) {
      App.renderer.render(App.scene, App.camera);
    }

    // 等待布局过渡完成后重新渲染，确保画布尺寸正确
    setTimeout(() => {
      if (App.lowPowerMode && App.renderer && App.scene && App.camera) {
        App.onResize();
        App.renderer.render(App.scene, App.camera);
      }
    }, 400);
    App.showToast('已进入低功耗模式 · 角色已固定');
    console.log('[LP] 低功耗模式已激活 —— 动画循环已停止，角色形象保持固定可见');
  };
  App.exitLowPowerMode = function exitLowPowerMode() {
    App.lowPowerMode = false;
    localStorage.removeItem(App.LP_KEY);
    const app = document.getElementById('app');
    const stage = App.$('stage');
    app.classList.remove('low-power');
    stage.classList.remove('low-power');

    // 退出低功耗后默认展示聊天面板
    document.getElementById('chat-panel').classList.remove('collapsed');
    document.getElementById('controls').classList.remove('collapsed');
    App.chatToggle.classList.remove('has-new');

    // 恢复3D专用按钮
    [App.modelBtn, App.bgBtn, App.moveBtn, App.fpvBtn, App.resetCamBtn, App.$('gyro-btn')].forEach(b => {
      if (b) b.style.display = '';
    });

    // 恢复之前的性能级别
    const restoreTier = App._preLPPerfTier || 'default';
    App.setPerfTier(restoreTier);

    // 重新启动动画循环（场景/渲染器/模型一直保留，无需重新初始化）
    if (App.renderer && App.scene && App.camera && !window._animFrame) {
      if (App.clock) App.clock.getDelta(); // 重置时钟，避免恢复后 dt 过大导致跳变
      App.onResize();
      App.animate();
    }
    App.showToast('已退出低功耗模式');
    console.log('[LP] 低功耗模式已关闭 —— 动画循环已恢复');
  };

  /* ============================================================
   *  沉浸模式：隐藏所有UI按钮，防误触，长按屏幕退出
   * ============================================================ */
  App.immerseMode = false;
  App._immersePressTimer = null;

  App.toggleImmerseMode = function toggleImmerseMode() {
    App.immerseMode = !App.immerseMode;
    if (App.immerseMode) {
      document.body.classList.add('immersed');
      App.showToast('长按屏幕任意位置退出沉浸模式');
    } else {
      document.body.classList.remove('immersed');
    }
    console.log('[Immerse]', App.immerseMode ? '沉浸模式开启' : '沉浸模式关闭');
  };

  /** 长按屏幕退出沉浸模式（仅单指，双指手势不触发，微移容差5px） */
  App.initImmerseLongPress = function initImmerseLongPress() {
    let activePointers = 0;
    let startX = 0, startY = 0;
    const MOVE_THRESHOLD = 5; // 像素容差，避免手指微颤取消长按

    const clearTimer = () => {
      if (App._immersePressTimer) {
        clearTimeout(App._immersePressTimer);
        App._immersePressTimer = null;
      }
    };

    document.body.addEventListener('pointerdown', (e) => {
      if (!App.immerseMode) return;
      activePointers++;
      // 只有单指触摸才开始计时，多指按下时取消
      if (activePointers > 1) {
        clearTimer();
        return;
      }
      startX = e.clientX;
      startY = e.clientY;
      App._immersePressTimer = setTimeout(() => {
        App.toggleImmerseMode();
        App.showToast('已退出沉浸模式');
      }, 800);
    });

    document.body.addEventListener('pointerup', () => {
      activePointers = Math.max(0, activePointers - 1);
      clearTimer();
    });
    document.body.addEventListener('pointercancel', () => {
      activePointers = Math.max(0, activePointers - 1);
      clearTimer();
    });
    // 移动超过阈值才取消（允许手指微颤）
    document.body.addEventListener('pointermove', (e) => {
      if (!App._immersePressTimer) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      if (dx * dx + dy * dy > MOVE_THRESHOLD * MOVE_THRESHOLD) {
        clearTimer();
      }
    });
  };
  App.toggleLowPowerMode = function toggleLowPowerMode() {
    if (App.lowPowerMode) {
      App.exitLowPowerMode();
    } else {
      App.enterLowPowerMode();
    }
    const lpBtn = App.$('low-power-btn');
    if (lpBtn) lpBtn.classList.toggle('active', App.lowPowerMode);
  };
  App.setState = function setState(s) {
    // 状态切换时重置口型平滑值，防止从 SPEAKING 切出后嘴部残留
    if (s !== App.State.SPEAKING && App.smoothMouth !== undefined) {
      App.smoothMouth = 0;
    }
    App.currentState = s;
    App.statusBadge.className = 'status-badge';
    const dot = App.statusBadge.querySelector('.status-dot');
    const map = {
      [App.State.IDLE]: {
        t: '在线',
        c: 'online'
      },
      [App.State.THINKING]: {
        t: '思考中',
        c: 'thinking'
      },
      [App.State.LISTENING]: {
        t: '聆听中',
        c: 'listening'
      },
      [App.State.SPEAKING]: {
        t: '说话中',
        c: 'speaking'
      }
    };
    const v = map[s];
    // 保留指示灯，仅更新文字
    if (dot) {
      App.statusBadge.innerHTML = '';
      App.statusBadge.appendChild(dot);
      App.statusBadge.appendChild(document.createTextNode(' ' + v.t));
    } else {
      App.statusBadge.textContent = v.t;
    }
    App.statusBadge.classList.add(v.c);
    // 统一调度纪律：AI 空闲时补发排队中的 AI action（不互相打扰、不打断用户）
    if (s === App.State.IDLE && App._flushPendingAIActions) {
      // 延迟一帧补发，避免与刚结束的回复时序冲突
      setTimeout(App._flushPendingAIActions, 50);
    }
  };
  App.showSubtitle = function showSubtitle(text) {
    if (!text) {
      App.subtitle.classList.remove('show');
      return;
    }
    App.subtitle.textContent = text;
    App.subtitle.classList.add('show');
  };
  /* ============================================================
   *  WebSocket
   * ============================================================ */
  /* ============================================================
   *  场景状态持久化（相机/模型位置/缩放）
   * ============================================================ */
});
