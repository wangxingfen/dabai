import { EngagementRLAgent } from '../game/rl/engagement-rl-agent.js';
import { DatingRLSystem } from '../game/rl/dating-rl-system.js';
import { ExpressionRLAgent } from '../game/rl/expression-rl-agent.js';

export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  启动
   * ============================================================ */
  window.addEventListener('load', async () => {
    // 设备性能自动检测（必须在 initThree 之前，影响渲染器参数）
    App.detectPerfTier();

    // 检查是否上次为低功耗模式
    const savedLP = localStorage.getItem(App.LP_KEY);
    if (savedLP === '1') {
      App.lowPowerMode = true;
      // 覆盖设备检测结果，直接使用低功耗级别
      App.setPerfTier('low');
      const app = document.getElementById('app');
      const stage = App.$('stage');
      // 应用低功耗全屏式布局：3D场景占满整屏，聊天框默认折叠
      app.classList.add('low-power');
      stage.classList.add('low-power');
      document.getElementById('chat-panel').classList.add('collapsed');
      document.getElementById('controls').classList.add('collapsed');
      [App.modelBtn, App.bgBtn, App.moveBtn, App.fpvBtn, App.resetCamBtn, App.camSettingsBtn, App.$('gyro-btn')].forEach(b => {
        if (b) b.style.display = 'none';
      });
      App.$('low-power-btn').classList.add('active');
      console.log('[LP] 恢复低功耗模式（持久化）');
    }

    // 加载相机设置，必须在 initThree 之前
    App.loadCameraSettings();
    App.updateCameraSettingsUI();

    // 始终初始化3D场景（低功耗模式也需要渲染角色形象）
    App.initThree();
    App.restoreSceneState();
    setTimeout(() => {
      App.applySavedPositions();
    }, 500);
    App.bindEvents();
    App.initTTSConfig();
    App.initNameConfig();
    App.initRoleCards();
    App.connectWS();
    App.initSessionUI();
    App.addSystemMsg('正在连接服务器…');

    // 监听任意用户交互，用于 AI 自主触发计时
    ['click', 'touchstart', 'keydown', 'mousemove'].forEach(evt => {
      document.addEventListener(evt, () => {
        App.lastUserActivityTime = Date.now();
      }, {
        passive: true
      });
    });

    // 低功耗按钮事件
    const lpBtn = App.$('low-power-btn');
    if (lpBtn) lpBtn.addEventListener('click', App.toggleLowPowerMode);

    // 自动加载模型：优先按上次使用的角色卡片配置加载，否则恢复上次模型，否则加载列表第一个
    const activeCard = App.roleCardActiveId ? await App.restoreActiveRoleCard() : null;
    let modelLoaded = false;
    const saved = JSON.parse(localStorage.getItem('dabai.currentModel') || 'null');
    const preferModel = activeCard && activeCard.model_url
      ? { url: activeCard.model_url, name: activeCard.model_name || activeCard.model_url.split('/').pop() }
      : null;
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      const models = data.models || [];
      // 1. 角色卡片指定了外形 → 以卡片配置为准
      if (preferModel && models.some(m => m.url === preferModel.url)) {
        await App.loadModelFromUrl(preferModel.url, preferModel.name);
        modelLoaded = true;
      }
      // 2. 否则恢复上次使用的模型
      if (!modelLoaded && saved && saved.url && models.some(m => m.url === saved.url)) {
        await App.loadModelFromUrl(saved.url, saved.name);
        modelLoaded = true;
      } else if (saved && saved.url) {
        localStorage.removeItem('dabai.currentModel');
      }
      // 3. 都没有 → 加载列表第一个
      if (!modelLoaded && models.length > 0) {
        const first = models[0];
        await App.loadModelFromUrl(first.url, first.name);
        modelLoaded = true;
      }
    } catch {
      // 离线/出错则忽略
    }

    // 自动恢复上次使用的背景
    const savedBg = JSON.parse(localStorage.getItem('dabai.currentBackground') || 'null');
    if (savedBg && savedBg.url) {
      try {
        const res = await fetch('/api/backgrounds');
        const data = await res.json();
        const exists = (data.backgrounds || []).some(m => m.url === savedBg.url);
        if (exists) {
          await App.loadBackgroundFromUrl(savedBg.url, savedBg.name);
        } else {
          localStorage.removeItem('dabai.currentBackground');
        }
      } catch {
        // 离线/出错则忽略，使用默认背景
      }
    }

    // 启动完成，之后模型/背景切换可以正常触发 AI 动作
    App._isBooting = false;

    // 沉浸模式：初始化长按退出手势
    App.initImmerseLongPress();

    // 非游戏模式互动强化学习（后台运行）
    App.initEngagementRL();

    // RL 自我表达：视觉层表情动作智能体（随互动/恋爱系统自动开启）
    App.initExpressionRL();
    if (App.engagementRLActive && !App.gameModeActive) {
      App._expressionRL?.start();
    }

    // 恋爱养成系统（默认跟随互动 RL 开启）
    const savedDating = localStorage.getItem('dabai.datingMode');
    if (savedDating === '1' || !savedDating) {
      // 首次使用或之前开启过 → 自动启动
      App.initDatingSystem();
      App.datingSystemActive = true;
      App._datingSystem?.startSession();
      if (!savedDating) localStorage.setItem('dabai.datingMode', '1');
    }

    // 低功耗模式：模型加载完成后冻结角色（停止动画循环，保留最后一帧）
    if (App.lowPowerMode && window._animFrame) {
      cancelAnimationFrame(window._animFrame);
      window._animFrame = null;
      if (App.renderer && App.scene && App.camera) {
        App.onResize();
        App.renderer.render(App.scene, App.camera);
      }
      console.log('[LP] 角色已冻结固定');
    }
    setTimeout(() => {
      // 只有没有历史消息时才显示欢迎语
      if (App.messagesEl.children.length <= 1) {
        App.addAIMsg('嗨~ 自动对话已开启，直接说话就行~ 也可以打字，或点左侧小按钮切换为按住说话模式');
      }
    }, 1500);
  });

  // ==================== 非游戏模式互动 RL ====================
  App.initEngagementRL = function () {
    try {
      // 统一架构：与恋爱养成系统共享同一 UnifiedDatingSystem 实例
      if (App._datingSystem) {
        App._engagementRL = App._datingSystem;
        console.log('[EngagementRL] 复用统一RL系统实例（与恋爱养成共享）');
      } else {
        App._engagementRL = new EngagementRLAgent(App);
      }
      App.engagementRLActive = true;
      console.log('[EngagementRL] 互动智能体已启动');
    } catch (e) {
      console.warn('[EngagementRL] 初始化失败:', e.message);
      App.engagementRLActive = false;
    }
  };

  // ==================== RL 自我表达（视觉层表情动作） ====================
  App.initExpressionRL = function () {
    try {
      if (!App._expressionRL) {
        App._expressionRL = new ExpressionRLAgent(App);
      }
      console.log('[ExpressionRL] 自我表达智能体已初始化');
    } catch (e) {
      console.warn('[ExpressionRL] 初始化失败:', e.message);
    }
  };

  /** 切换 RL 自我表达（开/关） */
  App.toggleExpressionRL = function () {
    if (!App._expressionRL) App.initExpressionRL();
    const on = App._expressionRL.toggle();
    console.log('[ExpressionRL] 自我表达', on ? '已开启' : '已关闭');
    return on;
  };

  /** 进入/离开游戏模式时联动（游戏内动作由游戏系统接管） */
  App.setGameModeExpressionRL = function (inGame) {
    if (!App._expressionRL) return;
    if (inGame) {
      App._expressionRL.stop();
    } else if (App.engagementRLActive) {
      App._expressionRL.start();
    }
  };

  // ==================== 恋爱养成强化学习系统 ====================
  App.initDatingSystem = function () {
    try {
      // 统一架构：与互动RL共享同一 UnifiedDatingSystem 实例
      if (App._engagementRL) {
        App._datingSystem = App._engagementRL;
        console.log('[DatingRL] 复用统一RL系统实例（与互动RL共享）');
      } else {
        App._datingSystem = new DatingRLSystem(App);
      }
      App.datingSystemActive = true;
      console.log('[DatingRL] 恋爱养成系统已启动');
    } catch (e) {
      console.warn('[DatingRL] 初始化失败:', e.message);
      App.datingSystemActive = false;
    }
  };

  /** 切换恋爱养成模式（开/关） */
  App.toggleDatingMode = function () {
    if (!App._datingSystem) {
      App.initDatingSystem();
    }
    App.datingSystemActive = !App.datingSystemActive;
    if (App.datingSystemActive) {
      App._datingSystem.startSession();
      console.log('[DatingRL] 恋爱养成模式已开启');
    } else {
      App._datingSystem.endSession();
      App._datingSystem.flush();
      console.log('[DatingRL] 恋爱养成模式已关闭，数据已保存');
    }
    localStorage.setItem('dabai.datingMode', App.datingSystemActive ? '1' : '0');
    return App.datingSystemActive;
  };

  // DEBUG expose（使用 getter 确保访问到 initThree 后创建的实例）
  Object.defineProperty(window, '_scene', {
    get: () => App.scene
  });
  Object.defineProperty(window, '_camera', {
    get: () => App.camera
  });
  Object.defineProperty(window, '_modelGroup', {
    get: () => App.modelGroup
  });
  Object.defineProperty(window, '_smoothRotY', {
    get: () => App.smoothRotY
  });
  Object.defineProperty(window, '_currentAvatar', {
    get: () => App.currentAvatar
  });
});