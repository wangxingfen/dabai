// ============================================================
// web/js/vr/webxr-vr.js —— WebXR 真实 VR 模式（模块 26）
// ------------------------------------------------------------
// 唯一 VR 模式：immersive-vr WebXR 会话（真立体渲染，由浏览器/头显接管）。
//   - 进入：世界统一平移设定用户站位（角色正前方 1.8m）
//   - 手柄：XR 手柄射线戳角色；蓝牙标准手柄回退通道（摇杆移动/转身/缩放/按键）
//   - 视线引导：注视稳定 2s（视线与角色夹角<10°）角色自动归位到视线落点，可在相机设置中关闭
//   - 视野升降：低头看地面持续1秒→视野缓缓升高（俯视角色）；抬头看天空持续1秒→视野缓缓降低
//   - 晃动反馈：摇杆移动产生晃动强度，角色弹跳/摆动，AI 感知反馈
//
// 注：cardboard 双屏与 gyro 陀螺仪伪 VR 模式已删除，仅保留 WebXR。
// ============================================================
import * as THREE from 'three';

export default function initWebXR(App) {
  // ---------------- 状态 ----------------
  App.xrMode = 'off'; // 'off' | 'webxr'
  App.xrPresenting = false; // WebXR 会话进行中（帧循环由 XR 驱动）
  App._lastXrMode = null;
  App._xrSession = null;
  App._xrControllers = [];
  App._xrSavedCamera = null; // 进入 VR 前保存的相机状态（退出时恢复）
  App._xrWorldObjs = null;
  App._xrWorldShift = null; // 世界统一平移向量（进入 WebXR 时设置，退出时还原）
  App._xrUserOrigin = new THREE.Vector3();
  App._xrHeadPos = new THREE.Vector3();
  App._xrRaycaster = new THREE.Raycaster();
  App._xrTmpQuat = new THREE.Quaternion();
  App._upVec = new THREE.Vector3(0, 1, 0);
  // ---------------- VR 视野升降状态（低头/抬头持续凝视触发） ----------------
  App._xrEyeOffY = 0; // 当前视野高度偏移（米，叠加到左右眼矩阵 Y）
  App._xrHeightState = 'idle'; // 'idle' | 'rising' | 'lowering'
  App._xrLookDownTimer = 0; // 低头看地面持续累计时长（秒）
  App._xrLookUpTimer = 0; // 抬头看天空持续累计时长（秒）
  // ---------------- VR 注视引导状态（Gaze Guide） ----------------
  // 解决"VR 中摇杆无效/摸不到屏幕时无法移动、场景不随目光对齐"：
  // 视线与角色方向水平夹角 <10° 且稳定停留在该方向 2 秒后，角色自动归位到
  // 视线所指位置（用户正前方约 1.8m），并面向用户 —— 无论用户看向哪里，
  // 角色都会"走进视线"。可在相机设置弹窗中关闭此功能（gazeAssistEnabled）。
  // 用户主动操作（摇杆/触摸/拖拽/键盘/扳机）时暂停，停止操作 2.5s 后恢复。
  App._gaze = {
    stable: 0, // 视线方向稳定的累计时长（秒）
    dir: null, // 当前视线方向（水平，{x,z}）
    aligning: false, // 角色归位进行中
    moving: false, // 角色正在向视线落点移动（smoothTeleport 进行中）
    lockUntil: 0, // 用户主动操作暂停截止（performance.now ms）
    cooldownUntil: 0, // 完成一次归位后的冷却截止（ms）
    notified: false // 本会话是否已提示过自动归位
  };

  // ---------------- 能力检测 ----------------
  const isSecure = () =>
    location.protocol === 'https:' || ['localhost', '127.0.0.1'].includes(location.hostname);

  App._xrModeAvailable = function _xrModeAvailable() {
    // 仅 WebXR：immersive-vr 会话需 HTTPS + 浏览器支持
    return !!(
      typeof navigator !== 'undefined' &&
      navigator.xr &&
      navigator.xr.isSessionSupported &&
      isSecure()
    );
  };

  // ---------------- 覆盖层 UI ----------------
  App.showVrOverlay = function showVrOverlay() {
    const ov = App.$('vr-overlay');
    if (!ov) return;
    ov.style.display = 'block';
    const hint = App.$('vr-hint');
    if (hint) {
      hint.textContent = App._xrGameMode
        ? '左摇杆移动 · 右摇杆转身 · 头显自由环视 · 点「退出 VR」返回游戏'
        : '左摇杆移动 · 右摇杆转身/缩放 · 扳机戳一戳 · 低头/抬头1秒升降视野 · 注视2秒归位';
    }
  };
  App.hideVrOverlay = function hideVrOverlay() {
    const ov = App.$('vr-overlay');
    if (ov) ov.style.display = 'none';
  };

  // ---------------- 模式切换 ----------------
  // 移动端判定（触屏 + UA，与性能分级一致）
  App._isMobileDevice = function _isMobileDevice() {
    return /Android|iPhone|iPad|iPod|webOS/i.test(navigator.userAgent) ||
      ('ontouchstart' in window && window.innerWidth < 1024);
  };

  // 统一视角基准：进入 WebXR 前调用，重置相机轨道视角到"角色正前方 + 标准距离"
  App._xrResetView = function _xrResetView() {
    App.gyroYaw = 0;
    App.gyroPitch = 0;
    if (App.dragOrbitYaw !== undefined) App.dragOrbitYaw = 0;
    if (App.dragOrbitPitch !== undefined) App.dragOrbitPitch = 0;
  };

  App.cycleXrMode = async function cycleXrMode() {
    if (App.xrMode !== 'off') {
      App.exitXrMode();
      return;
    }
    // 仅 WebXR：不支持则提示（无其他模式可顺延）
    App._lastXrMode = 'webxr';
    const result = await App.enterXrMode('webxr');
    if (result !== true) App._lastXrMode = null;
  };

  // 晃动衰减常量（头显晃动累积后平滑平息；每秒衰减 2.5，
  // 强度60 平息约需 24 秒，可被新晃动叠加）
  const VR_DECAY_INTERVAL = 200; // 每 200ms 递减一次
  const VR_DECAY_STEP = 0.5; // 每次递减 0.5（每秒约 -2.5，强度60 平息约需 24 秒）

  // 晃动系统初始化：WebXR 进入时统一启用（头显晃动/扳机反馈产生晃动数据），
  // 幂等重建；同时启动衰减定时器（停止晃动后逐渐平息，可被新操作叠加）
  App._ensureVrShake = function _ensureVrShake() {
    if (!App.vrShake) {
      App.vrShake = { upDown: 0, leftRight: 0, lastActive: 0, stopNotified: false, gyro: 0 };
    }
    if (!App.vrDecayTimer) {
      App.vrDecayTimer = setInterval(() => {
        const vs = App.vrShake;
        if (!vs) return;
        if (vs.upDown > 0) vs.upDown = Math.max(0, vs.upDown - VR_DECAY_STEP);
        if (vs.leftRight > 0) vs.leftRight = Math.max(0, vs.leftRight - VR_DECAY_STEP);
      }, VR_DECAY_INTERVAL);
    }
  };

  // 头显晃动累积（WebXR）：晃动强度绑定头显真实运动，而非摇杆——
  // 头部左右转动（yaw 角速度）→ 左右摆动强度；头部俯仰（pitch 角速度）→
  // 上下弹跳强度；头显位置快速移动（线速度，如走动/身体起伏）辅助叠加。
  // 平稳化：速率先一阶低通滤掉帧间毛刺，再用 sqrt 压缩高角速度的增益，
  // 最后对每帧增量做上限钳制 —— 剧烈甩头不会瞬间拉满强度，只平滑爬升；
  // 停止晃动后仍由衰减定时器（24 秒平息）负责回落。
  App._accumHeadShake = function _accumHeadShake(dt) {
    const vs = App.vrShake;
    if (!vs) return;
    const renderer = App.renderer;
    if (!renderer || !renderer.xr || !renderer.xr.isPresenting) return;
    let xrCam;
    try { xrCam = renderer.xr.getCamera(); } catch (_) {}
    if (!xrCam) return;
    // 头显前向向量 → 水平朝向角 yaw / 俯仰角 pitch
    const fwd = new THREE.Vector3();
    try { xrCam.getWorldDirection(fwd); } catch (_) { return; }
    const yaw = Math.atan2(fwd.x, fwd.z);
    const pitch = Math.asin(THREE.MathUtils.clamp(fwd.y, -1, 1));
    // 头显线速度（位移变化率）
    const pos = new THREE.Vector3();
    try { pos.setFromMatrixPosition(xrCam.matrixWorld); } catch (_) {}
    let moving = false;
    if (App._xrHeadYawPrev !== undefined && App._xrHeadPitchPrev !== undefined) {
      // yaw 角度环绕处理（-179°↔+179° 不产生假大角速度）
      let dy = yaw - App._xrHeadYawPrev;
      while (dy > Math.PI) dy -= Math.PI * 2;
      while (dy < -Math.PI) dy += Math.PI * 2;
      const yawRate = Math.abs(dy) / Math.max(dt, 1e-4);
      const pitchRate = Math.abs(pitch - App._xrHeadPitchPrev) / Math.max(dt, 1e-4);
      const prevPos = App._xrHeadPosPrev;
      const linX = prevPos ? Math.abs(pos.x - prevPos.x) / Math.max(dt, 1e-4) : 0;
      const linY = prevPos ? Math.abs(pos.y - prevPos.y) / Math.max(dt, 1e-4) : 0;
      const linZ = prevPos ? Math.abs(pos.z - prevPos.z) / Math.max(dt, 1e-4) : 0;
      const lin = linX + linY + linZ;
      // 一阶低通：平滑速率（约 8Hz 截止），滤掉帧间毛刺；持续晃动时逐渐趋近真实值
      const sm = App._xrShakeSmooth || (App._xrShakeSmooth = { yaw: 0, pitch: 0, lin: 0 });
      const lp = Math.min(1, dt * 8);
      sm.yaw += (yawRate - sm.yaw) * lp;
      sm.pitch += (pitchRate - sm.pitch) * lp;
      sm.lin += (lin - sm.lin) * lp;
      // sqrt 非线性压缩：4rad/s 才接近满增益，剧烈转动不线性放大
      const yawN = Math.sqrt(Math.min(1, sm.yaw / 4));
      const pitchN = Math.sqrt(Math.min(1, sm.pitch / 4));
      const linN = Math.sqrt(Math.min(1, sm.lin / 2));
      const linYN = Math.sqrt(Math.min(1, linY / 1.5));
      // 每帧增量钳制（约每秒 +3.6）：强度累积减半，单次晃动累积更少
      const MAX_FRAME = 0.06;
      if (yawN > 0.05 || linN > 0.05) {
        const g = Math.min(MAX_FRAME, (yawN * 12 + linN * 9) * dt);
        vs.leftRight = Math.min(100, vs.leftRight + g);
        moving = true;
      }
      if (pitchN > 0.05 || linYN > 0.05) {
        const g = Math.min(MAX_FRAME, (pitchN * 12 + linYN * 9) * dt);
        vs.upDown = Math.min(100, vs.upDown + g);
        moving = true;
      }
    }
    App._xrHeadYawPrev = yaw;
    App._xrHeadPitchPrev = pitch;
    App._xrHeadPosPrev = pos.clone();
    if (moving) {
      vs.lastActive = Date.now();
      vs.stopNotified = false;
    }
  };

  App.enterXrMode = async function enterXrMode(mode) {
    if (App.xrMode !== 'off') App.exitXrMode();
    // 独立系统游戏（赛博公司等）可自行声明进入 VR：由游戏置 App._xrGameMode 后复用
    // 同一套 WebXR 会话管线（相机/移动由游戏接管）；其他游戏模式仍禁止进入 VR
    if (App.gameModeActive && !App._xrGameMode) {
      App.showToast('请先退出游戏模式，再进入 VR');
      return false;
    }
    if (App.lowPowerMode) App.exitLowPowerMode();
    if (App.fpvMode) App.exitFPV();
    if (App.moveMode) App.setMoveMode(false);
    // 退出部位聚焦：focusPart 每帧把相机钉在 target 上，若不关闭，
    // 退出 VR 后相机将卡在聚焦目标点无法脱离
    if (App.focusPart) {
      App.focusPart.active = false;
      App.focusPart.time = 0;
    }
    // 暂停角色自主移动（AI 路径注入 + idle 随机走）：VR 中角色若持续走动
    // 会径直走到相机位置造成重叠/穿模（游戏 VR 由游戏自己接管移动，跳过）
    if (!App._xrGameMode) App._xrPauseAutonomy();
    // 保存进入前的相机状态（WebXR 会话中相机矩阵被 XR 接管覆盖）
    App._xrSaveCamera();

    App.xrMode = mode; // 先置位（仅 'webxr'）
    App._ensureVrShake(); // 晃动系统统一初始化（含衰减定时器）
    App._stdPadPressed = false; // 手柄按钮边沿状态重置
    App._xrPrevImmersed = App.immerseMode;
    App.immerseMode = true;
    document.body.classList.add('immersed');
    document.body.classList.add('vr-active');
    App.showVrOverlay();

    const ok = await App._enterWebXR();
    if (ok !== true) {
      App.exitXrMode(true); // 静默回滚（不弹"已关闭"提示）
      return ok;
    }
    const gyroBtn = App.$('gyro-btn');
    if (gyroBtn) gyroBtn.classList.add('active');
    return true;
  };

  App.exitXrMode = function exitXrMode(silent) {
    const mode = App.xrMode;
    if (mode === 'off') return;
    const wasGameXR = !!App._xrGameMode;
    // 游戏实例统一从 GameModeManager 读取（App.currentGame 从未被赋值）
    const game = (App.gameModeManager && App.gameModeManager.currentGame) || App.currentGame;
    App.xrMode = 'off';
    if (mode === 'webxr') App._exitWebXR();
    // 恢复进入 VR 前的相机状态（关键：matrixAutoUpdate 在 WebXR 会话中被
    // three.js 置 false 且退出时不恢复，若不重置为 true，相机矩阵停留在 VR
    // 最后姿态，机位卡死无法脱离）
    App._xrRestoreCamera();
    // 立即对齐到当前轨道视角：VR 期间默认相机分支被跳过，targetCamPos/
    // autoLookTarget 停在进入前旧值，若仅靠恢复状态后 lerp，相机会长时间
    // 漂移到过期目标，视觉上"退出后不恢复"。这里直接重算并硬对齐。
    // （游戏模式 VR：相机由游戏 FPV 接管，不做大厅轨道对齐）
    if (!wasGameXR) App._xrSnapBackCamera();
    // 恢复角色自主移动（AI 路径 + idle 随机走）
    if (!wasGameXR) App._xrResumeAutonomy();
    // 若退出时角色正在向视线落点移动，中止平滑传送（避免普通模式下角色继续"走向视线"）
    if (App._smoothTeleport && App._gaze.moving) App._smoothTeleport = null;
    // 手柄按钮边沿状态重置（防止退出后残留按住状态导致下次进入立即点击）
    App._stdPadPressed = false;
    // 重置 WebXR 视野升降状态（高度偏移与凝视计时清零，下次进入重新开始）
    App._xrEyeOffY = 0;
    App._xrHeightState = 'idle';
    App._xrLookDownTimer = 0;
    App._xrLookUpTimer = 0;
    // 重置头显晃动跟踪（避免下次进入用旧帧差分产生瞬时假晃动）
    App._xrHeadYawPrev = undefined;
    App._xrHeadPitchPrev = undefined;
    App._xrHeadPosPrev = null;
    App._xrShakeSmooth = null;
    // 停止晃动衰减定时器并清空晃动数据
    if (App.vrDecayTimer) { clearInterval(App.vrDecayTimer); App.vrDecayTimer = null; }
    App.vrShake = null;
    // 退出部位聚焦（防止聚焦 target 残留影响普通模式相机）
    if (App.focusPart) {
      App.focusPart.active = false;
      App.focusPart.time = 0;
    }
    // 重置注视引导状态（下次进入重新开始）
    App._gaze = Object.assign(App._gaze, { stable: 0, dir: null, aligning: false, moving: false, lockUntil: 0, cooldownUntil: 0, notified: false });
    const gyroBtn = App.$('gyro-btn');
    if (gyroBtn) gyroBtn.classList.remove('active');
    document.body.classList.remove('vr-active');
    if (!App._xrPrevImmersed) {
      App.immerseMode = false;
      document.body.classList.remove('immersed');
    }
    App._xrGameMode = false;
    App.hideVrOverlay();
    // 游戏模式 VR 退出钩子：由游戏恢复自身 VR 状态（按钮/相机/提示）
    if (wasGameXR && game && typeof game.onExitXR === 'function') {
      try { game.onExitXR(); } catch (e) { console.warn('[WebXR] 游戏 VR 退出钩子异常:', e?.message || e); }
    }
    if (!silent) App.showToast('VR模式已关闭');
  };

  // 保存进入 VR 前的相机状态
  App._xrSaveCamera = function _xrSaveCamera() {
    const c = App.camera;
    if (!c) return;
    App._xrSavedCamera = {
      matrixAutoUpdate: c.matrixAutoUpdate,
      position: c.position.clone(),
      quaternion: c.quaternion.clone(),
      aspect: c.aspect // 进入 VR 时保存，退出需复原
    };
  };
  // 恢复进入 VR 前的相机状态。
  // 注意：WebXR 会话期间 three.js WebXRManager 会把 camera.matrixAutoUpdate
  // 置为 false 并用头显 pose 覆盖矩阵，退出时不会恢复——必须手动重置为 true，
  // 否则轨道相机逻辑的 position.lerp()/lookAt() 不会重建 matrixWorld，
  // 渲染将永远使用 VR 最后姿态的陈旧矩阵，机位卡死无法脱离。
  App._xrRestoreCamera = function _xrRestoreCamera() {
    const c = App.camera;
    if (!c) return;
    c.matrixAutoUpdate = true;
    const s = App._xrSavedCamera;
    if (s && s.position) {
      c.position.copy(s.position);
      c.quaternion.copy(s.quaternion);
      c.rotation.setFromQuaternion(s.quaternion);
      // 复原 aspect（VR 会话期间比例可能被改写，退出需还原，否则画面变形）
      if (s.aspect && Math.abs(c.aspect - s.aspect) > 1e-6) {
        c.aspect = s.aspect;
        c.updateProjectionMatrix();
      }
    }
    // 恢复默认旋转序 'XYZ' 避免影响后续轨道逻辑
    c.rotation.order = 'XYZ';
    App._xrSavedCamera = null;
  };

  // 立即对齐到当前轨道视角（退出 VR 时调用）。
  // VR 期间默认轨道相机分支被跳过，targetCamPos/autoLookTarget 停留在进入前旧值，
  // 若只恢复保存状态后靠 lerp 收敛，相机会长时间漂移到过期目标，视觉上"退出后不恢复"。
  // 这里用最新角色位置重算轨道参数并硬对齐，下一帧即呈现正常环绕视角。
  App._xrSnapBackCamera = function _xrSnapBackCamera() {
    const c = App.camera;
    const mc = App.modelGroup;
    if (!c || !mc) return;
    const mcY = 1.0;
    const zoom = App.camZoom || 1;
    const orbitR = (App.cameraDistance || 4) * zoom;
    const baseY = mcY + ((App.cameraHeight || 1.6) - mcY) * zoom;
    const orbYaw = (App.gyroYaw || 0) + (App.dragOrbitYaw || 0);
    const orbPitch = (App.gyroPitch || 0) + (App.dragOrbitPitch || 0);
    if (!App.autoLookTarget) App.autoLookTarget = new THREE.Vector3();
    App.autoLookTarget.set(mc.position.x, mcY, mc.position.z);
    if (!App.targetCamPos) App.targetCamPos = new THREE.Vector3();
    App.targetCamPos.set(
      (App.camOffsetX || 0) + App.autoLookTarget.x + orbitR * Math.sin(orbYaw) * Math.cos(orbPitch),
      baseY + (App.camOffsetY || 0) + orbitR * Math.sin(orbPitch),
      (App.camOffsetZ || 0) + App.autoLookTarget.z + orbitR * Math.cos(orbYaw) * Math.cos(orbPitch)
    );
    c.position.copy(App.targetCamPos);
    c.rotation.order = 'XYZ';
    c.lookAt(App.autoLookTarget.x, App.autoLookTarget.y, App.autoLookTarget.z);
    c.updateMatrixWorld(true);
  };

  // 暂停角色自主移动（进入 VR 时调用）：清空 AI 路径与 idle 随机行走目标，
  // 防止角色在 VR 中走到相机位置造成重叠/穿模。
  App._xrPauseAutonomy = function _xrPauseAutonomy() {
    const gm = App.gameModeManager;
    const ai = gm && gm.aiAutonomy ? gm.aiAutonomy : null;
    App._xrSavedAutonomy = {
      aiMoving: ai ? ai._aiMoving : false,
      aiDrivenWalk: App._aiDrivenWalk,
      walkPath: (App.walkPath || []).slice(),
      walkSegmentIndex: App.walkSegmentIndex || 0,
      idleWalkTarget: App.idleWalkTarget ? { x: App.idleWalkTarget.x, z: App.idleWalkTarget.z } : null,
      idleWalkProgress: App.idleWalkProgress || 0
    };
    if (ai) ai._aiMoving = false;
    App._aiDrivenWalk = false;
    App.walkPath = [];
    App.walkSegmentIndex = 0;
    App.idleWalkTarget = null;
    App.idleWalkProgress = 0;
  };
  // 恢复角色自主移动（退出 VR 时调用）：还原进入前的移动状态。
  App._xrResumeAutonomy = function _xrResumeAutonomy() {
    const s = App._xrSavedAutonomy;
    if (!s) return;
    const gm = App.gameModeManager;
    if (gm && gm.aiAutonomy && s.aiMoving) gm.aiAutonomy._aiMoving = true;
    App._aiDrivenWalk = !!s.aiDrivenWalk;
    App.walkPath = s.walkPath || [];
    App.walkSegmentIndex = s.walkSegmentIndex || 0;
    App.idleWalkTarget = s.idleWalkTarget;
    App.idleWalkProgress = s.idleWalkProgress || 0;
    App._xrSavedAutonomy = null;
  };

  // ---------------- WebXR 沉浸模式 ----------------
  App._enterWebXR = async function _enterWebXR() {
    const renderer = App.renderer;
    if (!renderer || !renderer.xr || !navigator.xr) {
      App.showToast('当前浏览器不支持 WebXR');
      return false;
    }
    if (!isSecure()) {
      App.showToast('WebXR 需要 HTTPS 访问 · 请用 https:// 打开');
      return false;
    }
    let supported = false;
    try {
      supported = await navigator.xr.isSessionSupported('immersive-vr');
    } catch (_) {
      supported = false;
    }
    if (!supported) {
      App.showToast('未检测到可用的 VR 设备');
      return 'unsupported';
    }

    // 初始化手柄与射线（一次性）
    App._setupXRControllers();

    // 设定用户站位：进入前相机位置即用户期望站位（角色前方保留原方位/距离）。
    // 注意：requestSession 后 WebXRManager 会把相机重置到 reference space 原点，
    // 且 three.js 文档明确「setReferenceSpace() 不能在会话期间调用」，所以不能靠
    // 参考系偏移把相机挪到期望站位。改为「世界统一平移」：进入前把角色/背景/星空
    // 整体平移 -desiredUserPos，使会话开始时相机位于原点即可看到与期望站位完全
    // 一致的画面（等效用户站到 desiredUserPos）。退出时再平移回去。
    if (!App._xrGameMode) {
      // 大厅模式：统一视角固定从角色正前方 1.8m 进入（保证每次进入视角一致）
      const mc = App.modelGroup ? App.modelGroup.position : { x: 0, y: 0, z: 0 };
      App.camera.position.set(mc.x, 1.6, mc.z + 1.8); // 眼高 1.6m，角色正前方 1.8m
      App.camera.lookAt(mc.x, 1.0, mc.z);
      const desiredUserPos = App.camera.position.clone();
      App._xrUserOrigin.copy(desiredUserPos);
      App._xrHeadPos.copy(desiredUserPos);
      App._xrCaptureWorld();
      // 世界统一平移：进入 WebXR 会话前立即生效，时序安全（不依赖 setReferenceSpace）
      if (App._xrWorldObjs && App._xrWorldObjs.length) {
        App._xrWorldShift = new THREE.Vector3(-desiredUserPos.x, 0, -desiredUserPos.z);
        const ws = App._xrWorldShift;
        for (const obj of App._xrWorldObjs) {
          obj.position.x += ws.x;
          obj.position.z += ws.z;
        }
        // 用户锚点/头部锚点同步平移，保证后续移动/转身/缩放以「世界中的用户位置」为基准
        App._xrUserOrigin.add(ws);
        App._xrHeadPos.add(ws);
      }
    } else {
      // 游戏模式（赛博公司等）：世界不平移，相机由游戏 FPV 逻辑接管；
      // 游戏自带角色自主移动/蜂群，不受大厅注视引导影响
      App._xrWorldObjs = [];
    }

    try {
      const sessionInit = { optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking', 'layers'] };
      const session = await navigator.xr.requestSession('immersive-vr', sessionInit);
      App._xrSession = session;
      App.xrPresenting = true;
      try {
        renderer.xr.setReferenceSpaceType('local-floor');
      } catch (_) {}
      renderer.xr.setSession(session);
      // 注意：不再调用 renderer.xr.setReferenceSpace() —— 该 API 文档明确禁止在会话
      // 期间调用，会静默失效导致相机停留在 (0,0,0)「跳原点」。用户站位已通过上面的
      // 世界统一平移实现，会话开始即处于正确相对位置。
      session.addEventListener('end', App._onSessionEnd);
      // 暂停 rAF 自调度，切换为 XR 帧循环
      if (window._animFrame) {
        cancelAnimationFrame(window._animFrame);
        window._animFrame = null;
      }
      renderer.setAnimationLoop(App.animate);
      App.showToast(App._xrGameMode ? '已进入 VR · 摇杆移动/转身 · 沉浸体验中' : '已进入 VR · 扳机戳一戳 · 摇杆移动/转身');
      if (!App._xrGameMode) {
        App.sendAIAction('（你戴上了VR头显，走进了她的世界，她好奇又期待地看着你）', true);
      }
      return true;
    } catch (err) {
      App.xrPresenting = false;
      App._xrSession = null;
      // 会话请求失败（拒绝授权/设备不可用等）：回滚进入前已做的世界统一平移，
      // 避免场景留在偏移位置
      App._xrUndoWorldShift();
      App._xrWorldObjs = null;
      try { renderer.setAnimationLoop(null); } catch (_) {}
      if (!App.lowPowerMode) App.animate();
      App.showToast('无法进入 VR · ' + (err && err.message ? err.message : '设备不支持'));
      return 'unsupported';
    }
  };

  App._onSessionEnd = function _onSessionEnd() {
    App.exitXrMode();
  };

  App._xrUndoWorldShift = function _xrUndoWorldShift() {
    // 还原世界统一平移（进入 WebXR 时整体平移了 -desiredUserPos，这里平移回去，
    // 保证角色/背景/星空回到原始世界坐标，非 VR 相机逻辑恢复正常）
    if (App._xrWorldShift && App._xrWorldObjs) {
      const ws = App._xrWorldShift;
      for (const obj of App._xrWorldObjs) {
        obj.position.x -= ws.x;
        obj.position.z -= ws.z;
      }
      App._xrUserOrigin.sub(ws);
      App._xrHeadPos.sub(ws);
    }
    App._xrWorldShift = null;
  };

  App._exitWebXR = function _exitWebXR() {
    const renderer = App.renderer;
    App.xrPresenting = false;
    App._xrUndoWorldShift();
    App._xrWorldObjs = null;
    if (App.camera) App.camera.matrixAutoUpdate = true; // 兜底：恢复自动矩阵更新
    if (renderer) {
      try { renderer.setAnimationLoop(null); } catch (_) {}
    }
    const s = App._xrSession;
    App._xrSession = null;
    if (s) {
      try { s.end(); } catch (_) {}
    }
    if (!App.lowPowerMode) App.animate();
  };

  // ---------------- 手柄 ----------------
  App._setupXRControllers = function _setupXRControllers() {
    if (App._xrControllers.length) return;
    const renderer = App.renderer;
    if (!renderer || !renderer.xr) return;
    for (let i = 0; i < 2; i++) {
      const controller = renderer.xr.getController(i);
      const rayGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, -1)
      ]);
      const rayMat = new THREE.LineBasicMaterial({
        color: 0x00e5ff,
        transparent: true,
        opacity: 0.35,
        depthWrite: false
      });
      const ray = new THREE.Line(rayGeo, rayMat);
      ray.name = 'xr-ray';
      ray.scale.z = 3;
      controller.add(ray);

      const dotMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.9 });
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.008, 8, 8), dotMat);
      dot.name = 'xr-dot';
      dot.visible = false;
      controller.add(dot);

      controller.userData.index = i;
      controller.userData.ray = ray;
      controller.userData.dot = dot;
      controller.addEventListener('selectstart', App._onControllerSelectStart);
      controller.addEventListener('squeezestart', App._onControllerSqueezeStart);
      controller.addEventListener('connected', App._onControllerConnected);
      controller.addEventListener('disconnected', App._onControllerDisconnected);
      App.scene.add(controller);
      App._xrControllers.push(controller);
    }
  };

  App._onControllerConnected = function _onControllerConnected(e) {
    const c = e.target;
    c.userData.inputSource = e.data;
    if (!c.userData.gripMesh) {
      const grip = new THREE.Mesh(
        new THREE.SphereGeometry(0.022, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0x7c5cff })
      );
      c.add(grip);
      c.userData.gripMesh = grip;
    }
    const h = e.data && e.data.handedness === 'left' ? '左手' : '右手';
    App.showToast('已连接' + h + '手柄');
  };
  App._onControllerDisconnected = function _onControllerDisconnected(e) {
    const c = e.target;
    c.userData.inputSource = null;
  };

  // 扳机：射线戳角色
  App._onControllerSelectStart = function _onControllerSelectStart(e) {
    const c = e.target;
    // 游戏模式 VR：没有大厅角色可戳，跳过（避免误触发隐藏角色的大厅交互）
    if (App._xrGameMode) return;
    if (!App.currentAvatar) return;
    // 扳机为主动操作：暂停注视自动对齐
    App._xrMarkInteraction();
    c.updateMatrixWorld(true);
    const origin = App._xrRaycaster.ray.origin;
    const dir = App._xrRaycaster.ray.direction;
    origin.setFromMatrixPosition(c.matrixWorld);
    dir.set(0, 0, -1).applyQuaternion(c.getWorldQuaternion(App._xrTmpQuat));
    App._xrRaycaster.far = 5;
    const hits = App._xrRaycaster.intersectObjects([App.currentAvatar], true);
    if (hits.length === 0) return;
    const hitPoint = hits[0].point.clone();
    const result = App.identifyModelPart(hitPoint);
    // 射线视觉缩短到命中点
    if (c.userData.ray) {
      c.userData.ray.scale.z = Math.max(0.15, origin.distanceTo(hitPoint));
      c.userData.ray.visible = true;
    }
    if (c.userData.dot) {
      c.userData.dot.position.z = -origin.distanceTo(hitPoint);
      c.userData.dot.visible = true;
      setTimeout(() => { if (c.userData && c.userData.dot) c.userData.dot.visible = false; }, 150);
    }
    if (App.triggerPokeAt) App.triggerPokeAt(hitPoint, result);
    // 触觉反馈
    try {
      const src = c.userData.inputSource;
      if (src && src.gamepad && src.gamepad.hapticActuators && src.gamepad.hapticActuators[0]) {
        src.gamepad.hapticActuators[0].pulse(0.5, 60);
      }
    } catch (_) {}
  };

  // 抓握：张开双臂拥抱/打招呼
  App._onControllerSqueezeStart = function _onControllerSqueezeStart(e) {
    const c = e.target;
    // 游戏模式 VR：跳过大厅拥抱/打招呼交互
    if (App._xrGameMode) return;
    try {
      const src = c.userData.inputSource;
      if (src && src.gamepad && src.gamepad.hapticActuators && src.gamepad.hapticActuators[0]) {
        src.gamepad.hapticActuators[0].pulse(0.2, 120);
      }
    } catch (_) {}
    if (c.userData.gripMesh) {
      const m = c.userData.gripMesh;
      m.scale.set(1.4, 1.4, 1.4);
      setTimeout(() => { if (c.userData && c.userData.gripMesh) c.userData.gripMesh.scale.set(1, 1, 1); }, 120);
    }
    App.sendAIAction('（你向TA张开了双臂，像是在邀请一个拥抱）', true);
  };

  // 每帧：摇杆移动/转身 + 角色面向用户 + 头部追踪
  App.updateXRControllers = function updateXRControllers(dt) {
    const renderer = App.renderer;
    if (!renderer || !renderer.xr) return;
    // 游戏模式 VR：移动/转身由游戏自己接管（幽灵锚点 + 第一人称相机）
    // 游戏实例统一从 GameModeManager 读取（App.currentGame 从未被赋值）
    const game = (App.gameModeManager && App.gameModeManager.currentGame) || App.currentGame;
    if (App._xrGameMode && game && typeof game.updateXR === 'function') {
      try { game.updateXR(dt); } catch (e) { console.warn('[WebXR] 游戏 VR 输入异常:', e?.message || e); }
      return;
    }
    // 双通道判定：WebXRManager.isPresenting 或自有会话标记（避免状态时序异常时摇杆失效）
    if (!renderer.xr.isPresenting && !App.xrPresenting) return;
    // 真实头部位置（供角色转向）
    try {
      const xrCam = renderer.xr.getCamera();
      if (xrCam) App._xrHeadPos.setFromMatrixPosition(xrCam.matrixWorld);
    } catch (_) {}
    // 头显晃动 → 晃动强度（摇头/点头/头部移动累积；摇杆移动不再产生晃动）
    if (App._accumHeadShake) App._accumHeadShake(dt);
    // 摇杆输入：优先四轴手柄（左摇杆移动/右摇杆转身缩放）；
    // 仅两轴手柄时兜底按钮缩放（常见 A/B 键），保证无右摇杆也能缩放。
    // 读取双通道：XRInputSource.gamepad（XR 帧回调内有效）+ renderer.xr.getGamepad(i)
    const readPad = (src, idx) => {
      let gp = null;
      if (src && src.gamepad) gp = src.gamepad;
      if (!gp && renderer.xr && typeof renderer.xr.getGamepad === 'function') {
        try { gp = renderer.xr.getGamepad(idx); } catch (_) {}
      }
      return gp;
    };
    let moveX = 0, moveZ = 0, turnX = 0, zoomY = 0;
    let fourAxis = false;
    for (let i = 0; i < App._xrControllers.length; i++) {
      const c = App._xrControllers[i];
      const gp = readPad(c.userData.inputSource, i);
      if (!gp) continue;
      const axes = gp.axes || [];
      if (axes.length >= 4) {
        moveX = axes[0]; moveZ = axes[1]; turnX = axes[2]; zoomY = axes[3];
        fourAxis = true;
        break;
      }
      if (axes.length >= 2 && !fourAxis && moveX === 0 && moveZ === 0) {
        moveX = axes[0]; moveZ = axes[1];
      }
    }
    // ---- 标准 Gamepad 回退通道（手机 + VR 眼镜：蓝牙手柄不走 XRInputSource）----
    // 无 XR 手柄输入时，回退到 navigator.getGamepads 读取标准手柄，
    // 保证 WebXR 下摇杆移动/转身/缩放/按钮点击与 XR 手柄体验一致。
    if (!fourAxis && moveX === 0 && moveZ === 0 && turnX === 0 && zoomY === 0) {
      const pad = App._readStdGamepad();
      if (pad && pad.connected) {
        if (pad.moveX !== 0 || pad.moveZ !== 0 || pad.turnX !== 0 || pad.zoomY !== 0) {
          moveX = pad.moveX; moveZ = pad.moveZ; turnX = pad.turnX; zoomY = pad.zoomY;
        }
        // 按钮点击（边沿检测，按住不重复触发）
        if (pad.buttonPressed && !App._stdPadPressed) App._xrPadClick();
        App._stdPadPressed = pad.buttonPressed;
      }
    }
    // 两轴手柄按钮缩放兜底（Oculus/常见映射：A=索引4 拉近，B=索引5 推远）
    if (!fourAxis) {
      for (let i = 0; i < App._xrControllers.length; i++) {
        const gp = readPad(App._xrControllers[i].userData.inputSource, i);
        if (!gp) continue;
        const btns = gp.buttons || [];
        if (btns.length >= 6) {
          const press = j => btns[j] && (btns[j].pressed || btns[j].value > 0.5);
          if (press(4)) zoomY = -1;
          else if (press(5)) zoomY = 1;
        }
        break;
      }
    }
    // 抓握键（squeeze）缩放兜底：左右任意一个挤压 = 拉近，同时挤压不动。
    // 优先采用主动挤压：若两指都挤压（左右手）则视为无缩放，避免误触
    {
      let squeezeCount = 0, squeezeVal = 0;
      for (let i = 0; i < App._xrControllers.length; i++) {
        const gp = readPad(App._xrControllers[i].userData.inputSource, i);
        if (!gp || !gp.buttons || gp.buttons.length < 2) continue;
        const b = gp.buttons[1];
        if (b && (b.pressed || b.value > 0.5)) { squeezeCount++; squeezeVal += b.value || 1; }
      }
      if (squeezeCount === 1) zoomY = squeezeVal > 0 ? (zoomY === 0 ? -0.9 : zoomY) : zoomY;
    }
    if (Math.abs(moveX) < 0.15) moveX = 0;
    if (Math.abs(moveZ) < 0.15) moveZ = 0;
    if (Math.abs(turnX) < 0.15) turnX = 0;
    if (Math.abs(zoomY) < 0.15) zoomY = 0;
    // 用户主动操作 → 暂停注视自动对齐（避免自动旋转与手动输入打架）
    if (moveX !== 0 || moveZ !== 0 || turnX !== 0 || zoomY !== 0) App._xrMarkInteraction();
    if (!App._xrWorldObjs) App._xrCaptureWorld();
    if (!App._xrWorldObjs) return;
    // 缩放（右摇杆 Y：上推(Y=-1) 拉近 / 下推(Y=+1) 推远）—— 手机在 VR 眼镜里摸不到
    // 屏幕，手柄缩放是唯一可用的缩放通道
    if (zoomY !== 0 && App.modelGroup) {
      const toChar = new THREE.Vector3().subVectors(App.modelGroup.position, App._xrUserOrigin);
      toChar.y = 0;
      const dist = toChar.length();
      if (dist > 0.01) {
        toChar.normalize();
        const delta = zoomY * 1.5 * dt; // 上推(Y=-1) → delta<0 → 拉近
        const newDist = Math.min(Math.max(dist + delta, 0.8), 8);
        const shift = toChar.multiplyScalar(newDist - dist);
        for (const obj of App._xrWorldObjs) {
          obj.position.x += shift.x;
          obj.position.z += shift.z;
        }
      }
    }
    // 前向/右向（水平面）
    const fwd = new THREE.Vector3();
    try {
      // 优先用 XR 相机真实朝向（部分设备 App.camera 矩阵更新时序较晚）
      const xrCam = renderer.xr.getCamera();
      if (xrCam) xrCam.getWorldDirection(fwd);
      else App.camera.getWorldDirection(fwd);
    } catch (_) {
      try { App.camera.getWorldDirection(fwd); } catch (_2) {}
    }
    fwd.y = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
    fwd.normalize();
    const right = new THREE.Vector3().crossVectors(fwd, App._upVec).normalize();
    // 转身（右摇杆 X：世界绕用户旋转）
    if (Math.abs(turnX) > 0) {
      const origin = App._xrUserOrigin;
      const ang = -turnX * 1.1 * dt;
      const cosA = Math.cos(ang), sinA = Math.sin(ang);
      for (const obj of App._xrWorldObjs) {
        const dx = obj.position.x - origin.x;
        const dz = obj.position.z - origin.z;
        obj.position.x = origin.x + dx * cosA - dz * sinA;
        obj.position.z = origin.z + dx * sinA + dz * cosA;
        obj.rotation.y += ang;
      }
    }
    // 移动（左摇杆：世界反向位移 → 等效用户前进/横移）
    const speed = 1.5 * dt;
    const wx = (fwd.x * moveZ - right.x * moveX) * speed;
    const wz = (fwd.z * moveZ - right.z * moveX) * speed;
    if (wx !== 0 || wz !== 0) {
      for (const obj of App._xrWorldObjs) {
        obj.position.x += wx;
        obj.position.z += wz;
      }
      // 用户锚点同步：移动后再转身应绕当前所在位置旋转
      App._xrUserOrigin.x += wx;
      App._xrUserOrigin.z += wz;
    }
    // 角色面向用户（由 animate 相机分支调用 updateXRFaceUser 处理平滑）
  };

  App._xrCaptureWorld = function _xrCaptureWorld() {
    if (App._xrGameMode) {
      App._xrWorldObjs = [];
      return;
    }
    App._xrWorldObjs = [
      App.modelGroup,
      App.backgroundGroup,
      App.parts && App.parts.glow,
      App.parts && App.parts.contactShadow,
      App.starField
    ].filter(Boolean);
  };

  // ---------------- VR 注视辅助对齐系统（Gaze Assist） ----------------
  // WebXR 沉浸模式通用机制（可在相机设置弹窗中关闭）：
  //   1) 视线与角色方向水平夹角<10° 且稳定停留≥2s 后触发自动对齐；
  //   2) 方向对齐：世界绕用户位置平滑旋转，使角色回到视线正前方；
  //   3) 距离对齐：过近(<1.15m)推远 / 过远(>2.8m)拉近到舒适距离 2.1m；
  //   4) 用户主动操作（摇杆/触摸/拖拽/键盘/扳机）时暂停，停止操作 2.5s 后恢复；
  //   5) 完成一次对齐后冷却 4s，避免反复触发。
  App._xrMarkInteraction = function _xrMarkInteraction() {
    App._gaze.lockUntil = performance.now() + 2500;
    App._gaze.stable = 0;
    App._gaze.aligning = false;
  };
  // 当前水平注视方向（WebXR 用 XR 相机真实头部朝向；无会话时用主相机朝向）
  App._xrLookDir = function _xrLookDir() {
    try {
      if (App.xrPresenting && App.renderer && App.renderer.xr) {
        const xrCam = App.renderer.xr.getCamera();
        if (xrCam) {
          const v = new THREE.Vector3();
          xrCam.getWorldDirection(v);
          return { x: v.x, z: v.z };
        }
      }
      if (App.camera) {
        const v = new THREE.Vector3();
        App.camera.getWorldDirection(v);
        return { x: v.x, z: v.z };
      }
    } catch (_) {}
    return null;
  };
  // 世界绕用户位置（_xrUserOrigin）水平旋转：等效"用户原地转身看向对齐目标"
  App._xrRotateWorld = function _xrRotateWorld(ang) {
    if (Math.abs(ang) < 1e-6 || !App._xrWorldObjs) return;
    const origin = App._xrUserOrigin;
    const cosA = Math.cos(ang), sinA = Math.sin(ang);
    for (const obj of App._xrWorldObjs) {
      const dx = obj.position.x - origin.x;
      const dz = obj.position.z - origin.z;
      obj.position.x = origin.x + dx * cosA - dz * sinA;
      obj.position.z = origin.z + dx * sinA + dz * cosA;
      obj.rotation.y += ang;
    }
  };
  // 世界水平平移（同步用户锚点）：等效"用户在世界中走动"。
  // 注意：锚点=头显物理位置，常规位移/距离调整不应移动锚点（否则相对距离不变），
  // 此工具仅用于需要锚点跟随的整段拖移场景
  App._xrShiftWorld = function _xrShiftWorld(wx, wz) {
    if ((!wx && !wz) || !App._xrWorldObjs) return;
    for (const obj of App._xrWorldObjs) {
      obj.position.x += wx;
      obj.position.z += wz;
    }
    App._xrUserOrigin.x += wx;
    App._xrUserOrigin.z += wz;
  };
  // 标准 Gamepad（蓝牙/USB 手柄）读取：左摇杆移动、右摇杆 X 转身、右摇杆 Y 缩放、
  // A/B/X/Y 任一按钮按下触发点击（WebXR 下 XR 手柄无输入时的回退通道）。
  App._readStdGamepad = function _readStdGamepad() {
    const out = { moveX: 0, moveZ: 0, turnX: 0, zoomY: 0, buttonPressed: false, connected: false };
    try {
      const pads = navigator.getGamepads ? navigator.getGamepads() : [];
      for (const gp of pads) {
        if (!gp || !gp.connected) continue;
        out.connected = true;
        const axes = gp.axes || [];
        if (axes.length >= 2) {
          // 用户设备摇杆映射：交换轴序 + 符号翻转。
          // axes[0]实际是上下、axes[1]实际是左右；上推(axes[0]=+1)→前进(+1)
          out.moveX = Math.abs(axes[1]) > 0.15 ? -axes[1] : 0; // 实际左右（左推+1 → -1 左移）
          out.moveZ = Math.abs(axes[0]) > 0.15 ? axes[0] : 0; // 实际上下（上推+1 → +1 前进）
        }
        if (axes.length >= 4) {
          out.turnX = Math.abs(axes[2]) > 0.15 ? axes[2] : 0; // 右摇杆 X（右推=右转）
          out.zoomY = Math.abs(axes[3]) > 0.15 ? axes[3] : 0; // 右摇杆 Y（上推=-1 → 拉近）
        }
        // 按钮：A(0)/B(1)/X(2)/Y(3) 任一按下 → 触发点击（边沿检测在调用方）
        const btns = gp.buttons || [];
        for (let i = 0; i < Math.min(4, btns.length); i++) {
          if (btns[i] && (btns[i].pressed || btns[i].value > 0.5)) {
            out.buttonPressed = true;
            break;
          }
        }
        break;
      }
    } catch (_) {}
    return out;
  };

  // 手机端手柄按钮点击：视线中心（屏幕正中）射线命中角色 → 戳一戳 + 部位聚焦，
  // 并产生持续摇晃反馈（强度+6，配合平滑衰减约 4.8 秒逐渐平息）。
  // WebXR 模式下用 XR 相机真实朝向/位置；无会话时用主相机（gyroYaw/Pitch 兜底）
  App._xrPadClick = function _xrPadClick() {
    if (!App.currentAvatar) return;
    // 射线起点：WebXR 用 XR 相机位置，否则主相机位置
    let origin;
    let dir;
    if (App.xrPresenting && App.renderer && App.renderer.xr) {
      try {
        const xrCam = App.renderer.xr.getCamera();
        if (xrCam) {
          origin = new THREE.Vector3().setFromMatrixPosition(xrCam.matrixWorld);
          dir = new THREE.Vector3();
          xrCam.getWorldDirection(dir);
        }
      } catch (_) {}
    }
    if (!origin || !dir) {
      // 兜底：与相机 YXZ 旋转一致（yaw 水平、pitch 俯仰）
      const yaw = App.gyroYaw || 0;
      const pitch = App.gyroPitch || 0;
      const cp = Math.cos(pitch);
      origin = App.camera.position.clone();
      dir = new THREE.Vector3(-Math.sin(yaw) * cp, Math.sin(pitch), -Math.cos(yaw) * cp);
    }
    const ray = new THREE.Raycaster(origin, dir, 0.1, 30);
    const hits = ray.intersectObjects([App.currentAvatar], true);
    if (hits.length) {
      const hitPoint = hits[0].point.clone();
      App.triggerPokeAt(hitPoint, App.identifyModelPart(hitPoint));
      // 点击产生晃动反馈：持续一段时间逐渐平息（可叠加）
      const vs = App.vrShake;
      if (vs) {
        vs.upDown = Math.min(100, vs.upDown + 6);
        vs.lastActive = Date.now();
        vs.stopNotified = false;
      }
    } else {
      App.showToast('没戳到~ 对准角色再试');
    }
    App._xrMarkInteraction(); // 点击视为用户操作，暂停注视自动归位
  };

  // 每帧调用（animate 中 updateXRControllers 之后）
  // 统一机制（WebXR 沉浸模式）：
  //   视线与角色方向水平夹角 <10° 且稳定停留 2 秒 → 角色平滑归位到视线所指位置
  //   （用户正前方约 1.8m）—— 无论用户看向哪里，角色都会"走进视线"（用 XR
  //   相机真实头部朝向）。可在相机设置弹窗中关闭（gazeAssistEnabled）。
  //   用户主动操作时暂停 2.5s，完成归位后冷却 4s。
  App.updateXrGazeAssist = function updateXrGazeAssist(dt) {
    if (!App.xrMode || App.xrMode === 'off') return;
    if (!App.modelGroup) return;
    // 设置中关闭注视归位 → 直接跳过（并清空进行中的归位，避免残留）
    if (!App.gazeAssistEnabled) {
      App._gaze.moving = false;
      App._gaze.aligning = false;
      return;
    }
    const now = performance.now();
    const gaze = App._gaze;

    // ---- 角色归位完成检测（smoothTeleport 已结束）----
    if (gaze.moving && !App._smoothTeleport) App._xrFinishGazeAlign();

    // ---- 用户主动操作期间暂停 ----
    if (now < gaze.lockUntil) {
      gaze.stable = 0;
      return;
    }
    if (now < gaze.cooldownUntil) return;

    // ---- 视线方向采集与稳定性检测 ----
    const dir = App._xrLookDir();
    if (!dir) return;
    let angleVel = Infinity;
    if (gaze.dir) {
      // 视线方向角度变化率（rad/s），>10°/s 视为正在转头，重置稳定计时
      const dot = THREE.MathUtils.clamp(dir.x * gaze.dir.x + dir.z * gaze.dir.z, -1, 1);
      angleVel = Math.acos(dot) / Math.max(dt, 1e-4);
    }
    gaze.dir = { x: dir.x, z: dir.z };
    if (angleVel > 10 * Math.PI / 180 || gaze.aligning) {
      gaze.stable = 0;
      return;
    }
    // 触发门槛：视线与角色方向水平夹角必须 <10°，看向别处则重置计时
    const mc = App.modelGroup;
    const ux = App._xrUserOrigin.x;
    const uz = App._xrUserOrigin.z;
    const toCharX = mc.position.x - ux;
    const toCharZ = mc.position.z - uz;
    const toCharLen = Math.hypot(toCharX, toCharZ);
    if (toCharLen > 0.01) {
      const dotC = THREE.MathUtils.clamp((dir.x * toCharX + dir.z * toCharZ) / toCharLen, -1, 1);
      if (Math.acos(dotC) > 10 * Math.PI / 180) {
        gaze.stable = 0;
        return;
      }
    }
    gaze.stable += dt;
    if (gaze.stable < 2.0) return;

    // ---- 触发角色归位：平滑移动到视线所指位置 ----
    // 视线落点 = 用户位置（水平，XR 头显锚点） + 视线方向 * 1.8m
    const tx = ux + dir.x * 1.8;
    const tz = uz + dir.z * 1.8;
    // 用官方平滑传送通道（updateSmoothTeleport 每帧执行），不硬瞬移
    if (App.updateSmoothTeleport) {
      App._smoothTeleport = {
        x0: mc.position.x, y0: mc.position.y, z0: mc.position.z,
        x1: tx, y1: mc.position.y, z1: tz,
        t: 0, dur: 1.2
      };
      gaze.aligning = true;
      gaze.moving = true;
      gaze.stable = 0;
      gaze.lockUntil = now + 2600; // 平滑 1.2s + 余量，防归位途中重复触发
      App.showToast('角色正向你走来…');
    }
    return;
  };

  // 角色归位完成（smoothTeleport 结束）收尾：置冷却并提示
  App._xrFinishGazeAlign = function _xrFinishGazeAlign() {
    const gaze = App._gaze;
    if (!gaze.moving) return;
    gaze.moving = false;
    gaze.aligning = false;
    gaze.stable = 0;
    gaze.cooldownUntil = performance.now() + 4000;
    if (!gaze.notified) {
      gaze.notified = true;
      App.showToast('已归位 · 注视停留2秒可再次引导角色');
    }
  };

  // 角色转向用户真实头部位置（在 animate 相机分支调用）
  App.updateXRFaceUser = function updateXRFaceUser() {
    const avatar = App.currentAvatar || App.modelGroup;
    if (!avatar) return;
    if (App.currentAction && App.currentAction.type === App.ActionType.TURN) return;
    const dx = App._xrHeadPos.x - avatar.position.x;
    const dz = App._xrHeadPos.z - avatar.position.z;
    const targetY = Math.atan2(dx, dz);
    let diff = targetY - App.smoothRotY;
    while (diff > Math.PI) diff -= Math.PI * 2;
    while (diff < -Math.PI) diff += Math.PI * 2;
    App.smoothRotY += diff * 0.08;
    avatar.rotation.y = App.smoothRotY;
  };


  // WebXR 视野升降（低头看地面持续 1 秒 → 缓缓升高俯视角色；抬头看天空持续 1 秒 → 缓缓降低）：
  // 头部俯仰体现在 XR 相机矩阵第三列 Y 分量（e[9] = 相机前向的 Y）：抬头 e[9]>0、低头 e[9]<0。
  // 改为"持续凝视触发 + 匀速缓动"，不再随俯仰角比例直变：
  //   1) 低头（看地面）持续 1 秒 → rising：视野匀速缓缓升高，直到最高 +5 米；
  //   2) 抬头（看天空）持续 1 秒 → lowering：视野匀速缓缓降低，直到最低 -5 米；
  //   3) 低头/抬头均须超过 45°（前向Y 阈值 ±sin45°）才累计计时，平视立即清零并停止当前升降；
  //   4) 到达目标高度自动停止；视线回平随时停止；重复凝视可再次触发（高低两档间往返）。
  // 渲染时 three 逐眼用 cameraL/cameraR 的 matrix 更新 matrixWorld，在 render 前
  // （animate 内）修改左右眼矩阵 Y 平移分量（e[13]）即可生效。
  const XR_EYE_HIGH_Y = 5; // 视野最高高度（米，+5 俯视角色）
  const XR_EYE_LOW_Y = -5; // 视野最低高度（米，-5 低于地面）
  const XR_HEIGHT_SPEED = 0.8; // 升降速度（米/秒，缓缓）
  const XR_LOOK_HOLD = 1.0; // 低头/抬头持续触发阈值（秒）
  const XR_PITCH_DOWN = -0.7071; // 低头判定阈值（前向Y < -sin45°，低头超过 45° 才计）
  const XR_PITCH_UP = 0.7071; // 抬头判定阈值（前向Y > sin45°，抬头超过 45° 才计）
  App.updateXRHeight = function updateXRHeight(dt) {
    if (!App.renderer || !App.renderer.xr || !App.renderer.xr.isPresenting) return;
    const xrCam = App.renderer.xr.getCamera();
    const eyes = xrCam ? xrCam.cameras : null;
    if (!eyes || !eyes.length) return;
    if (App._xrEyeOffY === undefined) App._xrEyeOffY = 0;
    if (!App._xrHeightState) App._xrHeightState = 'idle';
    // 部位聚焦时保持当前高度并暂停凝视计时，避免聚焦结束后误触发
    const focus = App.focusPart && App.focusPart.active;
    if (focus) {
      App._xrLookDownTimer = 0;
      App._xrLookUpTimer = 0;
      // 聚焦期间冻结当前高度（停止升降），聚焦结束后保持
      App._xrHeightState = 'idle';
    } else {
      const pitchY = eyes[0].matrix.elements[9]; // 前向Y：低头负、抬头正
      // 方向保持计时：仅当视线持续看向地面/天空才累计，平视立即清零
      if (pitchY < XR_PITCH_DOWN) {
        App._xrLookUpTimer = 0;
        App._xrLookDownTimer += dt;
      } else if (pitchY > XR_PITCH_UP) {
        App._xrLookDownTimer = 0;
        App._xrLookUpTimer += dt;
      } else {
        App._xrLookDownTimer = 0;
        App._xrLookUpTimer = 0;
        // 视线回平（停止低头/抬头）→ 立即停止当前的升降过程
        App._xrHeightState = 'idle';
      }
      // 低头看地面持续满 1 秒 → 缓缓升高（若尚未到达顶部，且未在升高途中）
      if (App._xrLookDownTimer >= XR_LOOK_HOLD && App._xrHeightState !== 'rising' &&
        App._xrEyeOffY < XR_EYE_HIGH_Y - 0.05) {
        App._xrHeightState = 'rising';
        App._xrLookDownTimer = 0;
        App.showToast('视野缓缓升高 · 俯视角色');
      }
      // 抬头看天空持续满 1 秒 → 缓缓降低（若尚未降到最低，且未在降低途中）
      if (App._xrLookUpTimer >= XR_LOOK_HOLD && App._xrHeightState !== 'lowering' &&
        App._xrEyeOffY > XR_EYE_LOW_Y + 0.05) {
        App._xrHeightState = 'lowering';
        App._xrLookUpTimer = 0;
        App.showToast('视野缓缓降低');
      }
    }
    // 匀速缓动：缓缓升降，到达目标高度自动停止
    if (App._xrHeightState === 'rising') {
      App._xrEyeOffY += XR_HEIGHT_SPEED * dt;
      if (App._xrEyeOffY >= XR_EYE_HIGH_Y) {
        App._xrEyeOffY = XR_EYE_HIGH_Y;
        App._xrHeightState = 'idle';
      }
    } else if (App._xrHeightState === 'lowering') {
      App._xrEyeOffY -= XR_HEIGHT_SPEED * dt;
      if (App._xrEyeOffY <= XR_EYE_LOW_Y) {
        App._xrEyeOffY = XR_EYE_LOW_Y;
        App._xrHeightState = 'idle';
      }
    }
    const off = App._xrEyeOffY;
    if (Math.abs(off) < 0.001) return;
    for (let i = 0; i < eyes.length; i++) {
      eyes[i].matrix.elements[13] += off; // 左右眼矩阵 Y 平移叠加
    }
  };


  // ---------------- 初始化 ----------------
  // 构建覆盖层 DOM（若 index.html 未预置）
  if (!App.$('vr-overlay')) {
    const ov = document.createElement('div');
    ov.id = 'vr-overlay';
    ov.className = 'vr-overlay';
    ov.style.display = 'none';
    ov.innerHTML =
      '<button id="vr-exit-btn" class="vr-btn vr-exit">退出 VR</button>' +
      '<div id="vr-hint" class="vr-hint"></div>';
    document.body.appendChild(ov);
  }
  const exitBtn = App.$('vr-exit-btn');
  if (exitBtn) {
    exitBtn.addEventListener('click', () => {
      if (App._xrSession) {
        try { App._xrSession.end(); } catch (_) {}
      } else {
        App.exitXrMode();
      }
    });
  }
}
