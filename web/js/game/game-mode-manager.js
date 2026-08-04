/* ============================================================
 * 游戏模式管理器 —— 游戏模式总控
 *
 * 职责：
 * - 管理游戏模式的生命周期（进入/退出/切换游戏）
 * - 协调各子系统：场景生成、控制桥接、状态感知
 * - 注册和发现可用游戏
 * - 与主程序通信（WebSocket、UI更新）
 * ============================================================ */

import { GameSceneGenerator } from './game-scene-generator.js';
import { GameControlBridge } from './game-control-bridge.js';
import { GameStateObserver } from './game-state-observer.js';
import { VirtualJoystick } from './virtual-joystick.js';
import { AIAutonomyController } from './ai-autonomy-controller.js';

// 可用游戏注册表
const GAME_REGISTRY = {};

export class GameModeManager {
  constructor(app) {
    this.App = app;
    this.active = false;           // 是否在游戏模式中
    this.currentGame = null;       // 当前运行的游戏实例
    this.sceneGenerator = new GameSceneGenerator(app);
    this.controlBridge = new GameControlBridge(app);
    this.stateObserver = new GameStateObserver(app);
    this.aiAutonomy = new AIAutonomyController(app);

    // 暴露给全局供 WebSocket handler 访问
    app.aiAutonomyController = this.aiAutonomy;

    // 订阅主程序动画循环
    this._origAnimFrame = null;
    this._gameLoopBound = this._gameLoop.bind(this);
    this._rafId = null;
    this._lastGameLoopTime = 0;

    // UI元素（延迟创建）
    this._uiOverlay = null;
    this._uiPanel = null;
    this._uiGameList = null;
    this._uiScoreEl = null;
    this._uiTimerEl = null;
    this._uiHintEl = null;
    this._uiExitBtn = null;
    this._settlementOverlay = null;  // 结算界面
    this._settlementShown = false;   // 是否已显示结算界面

    // AI 自动跳跃 + 瞬移兜底
    this._aiLastStuckPos = null;     // { x, z } 上次记录位置（逐帧微停检测用）
    this._aiJumpAttempts = 0;        // 连续跳跃次数
    this._aiStallOriginPos = null;   // { x, z } 开始卡住时的位置
    this._aiStallStartTime = 0;      // 开始卡住的墙钟时间戳（毫秒）
    // AI 平滑攀爬过渡（替代瞬移）：以正常移动速度向目标位置过渡，避免视觉不适
    this._aiClimbTarget = null;      // { x, z, startY, endY, totalDist } 攀爬目标
    this._aiClimbCurrentPos = null;  // { x, y, z } 本帧攀爬位置（用于覆盖物理层对位置的修改）

    // 移动端控制
    this._isMobile = this._detectMobile();
    this._joystick = null;           // 虚拟摇杆
    this._touchState = null;         // 触控状态
    this._mobileFullscreen = false;  // 是否进入了全屏横屏
  }

  /** 检测是否为移动端 */
  _detectMobile() {
    return /Android|iPhone|iPad|iPod|webOS/i.test(navigator.userAgent)
      || ('ontouchstart' in window && window.innerWidth < 1024);
  }

  /**
   * 注册一个游戏
   * @param {string} key - 游戏唯一标识
   * @param {Function} factory - 工厂函数 (app) => BaseGame 实例
   */
  static registerGame(key, factory) {
    GAME_REGISTRY[key] = factory;
  }

  /** 获取所有已注册游戏 */
  static getAvailableGames() {
    return Object.keys(GAME_REGISTRY).map(key => {
      const inst = GAME_REGISTRY[key](null); // 临时实例获取元信息
      const info = { key, name: inst.displayName, description: inst.description };
      if (inst.cleanup) inst.cleanup();
      return info;
    });
  }

  /**
   * 进入游戏模式
   * @param {string} gameKey - 游戏标识符（可选，不传则显示选择界面）
   */
  enterGameMode(gameKey = null) {
    if (this.active) return;

    if (!gameKey) {
      this._showGameSelection();
      return;
    }

    const factory = GAME_REGISTRY[gameKey];
    if (!factory) {
      this.App.showToast('游戏不存在: ' + gameKey);
      return;
    }

    this._doEnterGame(factory);
  }

  /**
   * 退出游戏模式
   */
  exitGameMode() {
    if (!this.active) return;

    // 先标记退出，防止游戏循环继续（如果下面某步抛异常，游戏循环也会因 !active 而停止）
    this.active = false;

    // 移动端：退出全屏横屏
    this._exitFullscreen();

    // 移除结算界面
    this._removeSettlementUI();
    this._settlementShown = false;

    // 强制停止游戏循环
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }

    // 清理 walkPath / AI 自主移动状态（防止退出后仍残留 walkPath 导致大厅走动）
    this.App._aiDrivenWalk = false;
    this.App.walkPath = [];
    this.App.walkSegmentIndex = 0;
    this.App.idleWalkTarget = null;
    this.App.idleWalkProgress = 0;
    // 清理攀爬过渡状态
    this._aiClimbTarget = null;
    this._aiClimbCurrentPos = null;
    this._aiLastStuckPos = null;
    this._aiJumpAttempts = 0;
    this._aiStallOriginPos = null;
    this._aiStallStartTime = 0;

    // 累计错误日志（不中断流程）
    const errors = [];

    // 清理当前游戏
    if (this.currentGame) {
      try { this.currentGame.cleanup(); } catch (e) { errors.push('currentGame.cleanup: ' + e.message); }
      // 独立系统游戏：恢复被隔离的大厅感知函数（sendAIAction/sendRLSync 等）
      try { this._restoreLobbyIsolation(this.currentGame); } catch (e) { errors.push('restoreLobbyIsolation: ' + e.message); }
      this.currentGame = null;
    }

    // 清理场景生成器
    try { this.sceneGenerator.cleanup(); } catch (e) { errors.push('sceneGenerator.cleanup: ' + e.message); }

    // 清理控制桥接
    try { this.controlBridge.cleanup(); } catch (e) { errors.push('controlBridge.cleanup: ' + e.message); }

    // 清理 AI 自主控制器
    try { this.aiAutonomy.cleanup(); } catch (e) { errors.push('aiAutonomy.cleanup: ' + e.message); }

    // 解绑状态感知器
    try { this.stateObserver.unbind(); } catch (e) { errors.push('stateObserver.unbind: ' + e.message); }

    // 恢复AI自主动作
    try { this.controlBridge.releaseUserControl(); } catch (e) { errors.push('releaseUserControl: ' + e.message); }

    // 重置角色手臂到休息状态
    try { this._resetWalkAnimArms(); } catch (e) { errors.push('_resetWalkAnimArms: ' + e.message); }

    // 移除游戏UI
    try { this._removeGameUI(); } catch (e) { errors.push('_removeGameUI: ' + e.message); }

    // 兜底：强制从 DOM 中移除所有游戏 UI 残留
    this._forceRemoveAllGameUI();

    // 通知服务器
    if (this.App.ws && this.App.ws.readyState === WebSocket.OPEN) {
      try {
        this.App.ws.send(JSON.stringify({ type: 'exit_game_mode' }));
      } catch (e) { errors.push('ws.exit_game_mode: ' + e.message); }
    }

    // ========== 恢复相机状态 ==========
    this.App.gameModeActive = false;
    // 退出游戏：复位自适应分辨率，清理游戏残留音频引用
    try { if (this.App.prepareForLobby) this.App.prepareForLobby(); } catch (e) { /* ignore */ }
    if (this._savedCamera) {
      try {
        this.App.camZoom = this._savedCamera.camZoom;
        this.App.camOffsetX = this._savedCamera.camOffsetX;
        this.App.camOffsetY = this._savedCamera.camOffsetY;
        this.App.camOffsetZ = this._savedCamera.camOffsetZ;
        this.App.gyroYaw = this._savedCamera.gyroYaw;
        this.App.gyroPitch = this._savedCamera.gyroPitch;
        this.App.cameraHeight = this._savedCamera.cameraHeight;
        this.App.cameraTiltDeg = this._savedCamera.cameraTiltDeg;
        this.App.cameraDistance = this._savedCamera.cameraDistance;
        this.App.dragOrbitYaw = this._savedCamera.dragOrbitYaw || 0;
        this.App.dragOrbitPitch = this._savedCamera.dragOrbitPitch || 0;
        this.App.userRotY = this._savedCamera.userRotY;
        this.App.userRotX = this._savedCamera.userRotX;
      } catch (e) { errors.push('restoreCamera: ' + e.message); }
    }
    this._savedCamera = null;

    // ========== 恢复大厅场景对象 ==========
    if (this._savedSceneObjects) {
      try {
        for (const { obj, name } of this._savedSceneObjects) {
          if (obj) obj.visible = true;
        }
      } catch (e) { errors.push('restoreSceneObjects: ' + e.message); }
      this._savedSceneObjects = null;
    }

    // 清理游戏相机参数
    this.App._gameCamAzimuth = undefined;
    this.App._gameCamPitch = undefined;
    this.App._gameCamRadius = undefined;
    this._walkAnimTime = 0;
    this._walkAnimActive = false;

    // 重置角色位置
    try {
      this.App.resetAvatarToOrigin();
    } catch (e) { errors.push('resetAvatarToOrigin: ' + e.message); }

    this.App.showToast('已退出游戏模式');

    // 清理场景生成器引用
    this.App._gameSceneGen = null;

    // 游戏模式可能因全屏/后台切换导致麦克风流失效，退出后检查并清理
    const hadAutoVoice = this.App.voiceMode === 'auto';
    if (this.App.micStream && (!this.App.micStream.active ||
        !this.App.micStream.getAudioTracks().some(t => t.readyState === 'live'))) {
      console.log('[GameMode] 清理已失效的麦克风流');
      try { this.App.micStream.getTracks().forEach(t => t.stop()); } catch (e) {}
      this.App.micStream = null;
      this.App.vadStream = null;
      this.App.vadAnalyser = null;
      this.App.vadData = null;
      if (hadAutoVoice) {
        setTimeout(() => {
          if (this.App.voiceMode === 'auto') this.App.startVADMode();
        }, 300);
      }
    }

    // 触发AI反应（sendAIAction 通过 WebSocket 发送用户消息触发 AI 回复，
    // 此时服务器端 exit_game_mode 已将 agent 切回正常模式，AI 用正常身份回复）
    if (this.App.sendAIAction) {
      this.App.sendAIAction('（你和小伙伴刚刚一起玩完游戏，回到了原来的场景，有点意犹未尽的感觉）', true);
    }

    if (errors.length > 0) {
      console.warn('[GameMode] 退出时遇到错误（不影响退出）:', errors);
    }
  }

  /**
   * 游戏主循环更新
   * 注意：使用独立的 performance.now() 计时，不与主动画循环共享 THREE.Clock
   * （Clock.getDelta() 只能被调用一次，共享会导致 dt=0 卡死）
   */
  _gameLoop() {
    if (!this.active || !this.currentGame) return;

    const now = performance.now() / 1000;
    if (!this._lastGameLoopTime) this._lastGameLoopTime = now;
    const dt = Math.min(now - this._lastGameLoopTime, 0.1);
    this._lastGameLoopTime = now;
    const t = now;

    // 始终先调度下一帧，防止异常导致游戏循环静默死亡
    // （主渲染循环将 requestAnimationFrame 放在顶部，游戏循环放在底部，
    //   若帧内代码抛异常，游戏循环会静默停止，世界"冻结"）
    if (this.currentGame.state !== 'completed' && this.currentGame.state !== 'failed') {
      this._rafId = requestAnimationFrame(this._gameLoopBound);
    }

    try {
      let moveVec = null;
      let isClimbing = false;

      // AI 平滑攀爬过渡：以正常移动速度向攀爬目标移动，替代瞬移避免视觉不适
      if (this._aiClimbTarget) {
        moveVec = this._updateAIClimb(dt);
        isClimbing = !!moveVec;
      }

      if (!isClimbing) {
        // 更新控制桥接（用户移动）
        moveVec = this.controlBridge.updateMovement(dt);

        // 答题弹窗/亮灯期间暂停 AI 自主行走（避免 AI 在玩家答题或谜题触发时乱跑）
        const quizPaused = !!(this.currentGame && (this.currentGame._quizOpen || this.currentGame._lightingUp));

        // AI 自主移动：生成与用户操控一致的 moveVec，完全复用 _applyPlayerMovement
        // 二者仅输入来源不同，位移/旋转/碰撞全部走同一管道
        if (!moveVec && !quizPaused && this.App._aiDrivenWalk && this.App.idleWalkTarget && this.App.idleWalkProgress < 1) {
          moveVec = this._buildAIMoveVec(dt);
        }

        if (moveVec) {
          this._applyPlayerMovement(moveVec, dt);
        }
      }

      // AI 自动跳跃 + 瞬移兜底：卡住两秒没移动两米直接跳跑出去
      // 攀爬过程中跳过卡住检测（攀爬已绕过碰撞检测，无需再判断卡住）
      if (moveVec && !isClimbing) {
        const isAIDriven = !this.controlBridge.userControlling && this.App._aiDrivenWalk && this.App.idleWalkTarget;
        if (isAIDriven) {
          const avatar = this.App.currentAvatar || this.App.modelGroup;
          if (avatar) {
            const STALL_DIST = 0.08;          // 逐帧位移小于此值视为微停顿，启动卡住计时
            const STALL_CHECK_TIME = 2.0;      // 卡住检测周期
            const JUMP_MIN_DIST = 1.0;         // 2秒内移动不足1米 → 跳跃
            const TELEPORT_MIN_DIST = 2.0;     // 2秒内移动不足2米 → 瞬移
            const TELEPORT_JUMP_DIST = 3.0;    // 瞬移向前距离
            const MAX_JUMPS_BEFORE_REPATH = 3; // 连续跳跃上限后重新寻路

            const pos = { x: avatar.position.x, z: avatar.position.z };

            // 逐帧微停顿检测：静止时启动卡住计时，移动时重置
            if (this._aiLastStuckPos) {
              const frameMoved = Math.hypot(pos.x - this._aiLastStuckPos.x, pos.z - this._aiLastStuckPos.z);
              if (frameMoved < STALL_DIST) {
                // 微停顿中，继续累积
              } else {
                // 有实质移动，重置卡住追踪
                this._aiJumpAttempts = 0;
                this._aiStallOriginPos = null;
                this._aiStallStartTime = 0;
              }
            }
            this._aiLastStuckPos = pos;

            // 首次检测到微停顿时，记录卡住起点（位置 + 墙钟时间戳）
            if (!this._aiStallOriginPos) {
              this._aiStallOriginPos = { x: pos.x, z: pos.z };
              this._aiStallStartTime = performance.now();
            }

            // --- 卡住检测：每2秒检查一次总进度 ---
            if (this._aiStallOriginPos && this._aiStallStartTime > 0) {
              const stallElapsed = (performance.now() - this._aiStallStartTime) / 1000;
              if (stallElapsed >= STALL_CHECK_TIME) {
                const totalMoved = Math.hypot(pos.x - this._aiStallOriginPos.x, pos.z - this._aiStallOriginPos.z);
                const target = this.App.idleWalkTarget;

                if (totalMoved < JUMP_MIN_DIST) {
                  // 2秒移动不足1米 → 尝试跳跃翻越（先校验落点安全）
                  const jumpAttempts = this._aiJumpAttempts || 0;
                  if (jumpAttempts < MAX_JUMPS_BEFORE_REPATH && this.currentGame && this.currentGame.requestJump) {
                    const landing = this._findSafeJumpLanding(pos, target, 1.5);
                    const needsJump = landing && !this._isPathWalkable(this.currentGame, pos.x, pos.z, landing.x, landing.z);
                    if (needsJump) {
                      const jumped = this.currentGame.requestJump();
                      if (jumped !== false) {
                        this._aiJumpAttempts = jumpAttempts + 1;
                        console.log(`[游戏模式] AI 卡住${stallElapsed.toFixed(1)}秒仅移动${totalMoved.toFixed(1)}m，跳跃 (${this._aiJumpAttempts}/${MAX_JUMPS_BEFORE_REPATH})`);
                      } else if (this.aiAutonomy) {
                        // 游戏本身不支持跳跃或跳跃失败，直接重新寻路
                        console.log('[游戏模式] 当前游戏不支持跳跃，触发重新寻路');
                        this.aiAutonomy.repathFromCurrent(this.currentGame);
                        this._aiJumpAttempts = 0;
                      }
                    } else {
                      // 落点不安全或前方无障碍（无需跳跃）→ 重新寻路
                      if (this.aiAutonomy) {
                        console.log('[游戏模式] 跳跃落点不安全或无障碍，触发重新寻路');
                        this.aiAutonomy.repathFromCurrent(this.currentGame);
                        this._aiJumpAttempts = 0;
                      }
                    }
                  } else {
                    // 跳跃超限 → 重新寻路
                    if (this.aiAutonomy) {
                      console.log('[游戏模式] AI 连续跳跃超限，触发重新寻路');
                      this.aiAutonomy.repathFromCurrent(this.currentGame);
                      this._aiJumpAttempts = 0;
                    }
                  }
                  // 跳跃后重置卡住起点，重新计时下次检测
                  this._aiStallOriginPos = null;
                  this._aiStallStartTime = 0;

                } else if (totalMoved < TELEPORT_MIN_DIST) {
                  // 2秒移动不足2米 → 尝试沿目标方向短距离移动，但必须先通过地形/碰撞校验
                  // 改为平滑攀爬：以正常移动速度过渡到目标位置，避免瞬移视觉不适
                  const teleportPos = this._findSafeTeleportPos(pos, target, TELEPORT_JUMP_DIST);
                  if (teleportPos) {
                    const endY = (this.currentGame && this.currentGame._getGroundHeight)
                      ? this.currentGame._getGroundHeight(teleportPos.x, teleportPos.z)
                      : avatar.position.y;
                    this._aiClimbTarget = {
                      x: teleportPos.x,
                      z: teleportPos.z,
                      startY: avatar.position.y,
                      endY: endY,
                      totalDist: Math.hypot(teleportPos.x - pos.x, teleportPos.z - pos.z),
                    };
                    console.log(`[游戏模式] AI 卡住${stallElapsed.toFixed(1)}秒仅移动${totalMoved.toFixed(1)}m，启动平滑攀爬 ${this._aiClimbTarget.totalDist.toFixed(1)}m → (${teleportPos.x.toFixed(1)}, ${teleportPos.z.toFixed(1)})`);
                  } else if (this.aiAutonomy) {
                    console.log('[游戏模式] 攀爬落点不安全，触发重新寻路');
                    this.aiAutonomy.repathFromCurrent(this.currentGame);
                  }
                  this._aiJumpAttempts = 0;
                  this._aiStallOriginPos = null;
                  this._aiStallStartTime = 0;

                } else {
                  // 2秒移动已达2米 → 不再卡住，重置
                  this._aiJumpAttempts = 0;
                  this._aiStallOriginPos = null;
                  this._aiStallStartTime = 0;
                }
              }
            }
          }
        }
      } else if (!moveVec) {
        // 未在移动，重置停顿追踪
        this._aiLastStuckPos = null;
        this._aiJumpAttempts = 0;
        this._aiStallOriginPos = null;
        this._aiStallStartTime = 0;
      }

      // 移动端：单指滑动轨道更新
      if (this._isMobile) {
        this._updateMobileOrbit();
      }

      // 更新游戏逻辑（物理/碰撞等必须在行走动画之前，确保位置正确）
      this.currentGame.update(dt);

      // 攀爬过程中，物理层（重力/地面修正）可能修改 avatar 位置，
      // 这里强制恢复攀爬位置，保证平滑过渡
      if (isClimbing && this._aiClimbCurrentPos) {
        this._enforceClimbPosition();
      }

      // 定期刷新地图数据（程序化地形随玩家移动变化）
      this._mapRefreshTimer = (this._mapRefreshTimer || 0) + dt;
      if (this._mapRefreshTimer > 2.0) {
        this._mapRefreshTimer = 0;
        const mapData = this.currentGame._getMapData ? this.currentGame._getMapData() : null;
        if (mapData) {
          this.aiAutonomy.setMapData(mapData, mapData.type || 'heightmap');
        }
      }

      // 行走动画（在物理确定的位置上叠加bob）—— 独立系统游戏（幽灵模式）跳过：
      // 玩家无实体（大厅角色隐藏），不叠加大厅角色骨骼行走动画
      if (!(this.currentGame && this.currentGame.isIsolated)) {
        this._updateWalkAnimation(dt, moveVec);
      }

      // 更新游戏相机：固定在角色头部后方
      this._updateGameCamera(dt);

      // 更新状态感知器（独立系统游戏不接入大厅感知）
      if (!(this.currentGame && this.currentGame.isIsolated)) {
        this.stateObserver.update();
      }

      // 更新 AI 自主控制器（答题弹窗/亮灯期间暂停，避免干扰玩家答题）
      // 独立系统游戏彻底禁用：玩家/员工行为由游戏内部蜂群与 RL 驱动，不经过大厅自主系统
      if (!(this.currentGame && (this.currentGame.isIsolated || this.currentGame._quizOpen || this.currentGame._lightingUp))) {
        this.aiAutonomy.update(dt);
      }

      // 更新场景特效（发光、旋转等）
      this._updateSceneEffects(t);

      // 更新UI
      this._updateGameUI();

      // 检测游戏完成/失败，显示结算界面
      if (this.currentGame.state === 'completed' || this.currentGame.state === 'failed') {
        if (!this._settlementShown) {
          this._settlementShown = true;
          this._showSettlement(this.currentGame);
        }
        // 注意：此时已不在对帧调度 requestAnimationFrame（提前 return 的条件已处理）
      }
    } catch (err) {
      console.error('[GameLoop] 游戏循环异常:', err);
      // 循环继续运行（已在顶部调度下一帧），不会静默死亡
    }
  }

  // ==================== 内部实现 ====================

  /** 请求全屏横屏（移动端专用） */
  async _requestFullscreenLandscape() {
    if (!this._isMobile || this._mobileFullscreen) return;
    try {
      const el = document.documentElement;
      if (el.requestFullscreen) {
        await el.requestFullscreen();
      } else if (el.webkitRequestFullscreen) {
        await el.webkitRequestFullscreen();
      }
      // 锁定横屏方向
      if (screen.orientation && screen.orientation.lock) {
        await screen.orientation.lock('landscape').catch(() => {});
      }
      this._mobileFullscreen = true;
      // 给浏览器一点时间切换
      await new Promise(r => setTimeout(r, 300));
    } catch (e) {
      console.warn('[GameMode] 全屏横屏请求失败:', e.message);
    }
  }

  /** 退出全屏横屏（移动端专用） */
  async _exitFullscreen() {
    if (!this._mobileFullscreen) return;
    try {
      if (screen.orientation && screen.orientation.unlock) {
        screen.orientation.unlock();
      }
      if (document.fullscreenElement && document.exitFullscreen) {
        await document.exitFullscreen();
      } else if (document.webkitFullscreenElement && document.webkitExitFullscreen) {
        await document.webkitExitFullscreen();
      }
    } catch (e) {
      console.warn('[GameMode] 退出全屏失败:', e.message);
    }
    this._mobileFullscreen = false;
  }

  _doEnterGame(factory) {
    // 设置场景生成器引用（供游戏使用）
    this.App._gameSceneGen = this.sceneGenerator;

    // 进入游戏前退出 VR 模式（WebXR 会话与游戏轨道相机冲突）
    if (this.App.exitXrMode && this.App.xrMode !== 'off') {
      this.App.exitXrMode();
    }

    // 进入游戏（用户点击手势栈内）：主动解锁脚步音效播放池，
    // 确保游戏模式中轮换播放不被浏览器自动播放策略拦截
    try { if (this.App.unlockFootstepSFX) this.App.unlockFootstepSFX(); } catch (e) { /* ignore */ }

    // ========== 游戏模式相机锁定 ==========
    this.App.gameModeActive = true;
    // 进入游戏：释放大厅残留资源（已播完音频等）+ 复位自适应分辨率
    try { if (this.App.prepareForGame) this.App.prepareForGame(); } catch (e) { /* ignore */ }
    // 保存进入前的相机状态
    this._savedCamera = {
      camZoom: this.App.camZoom,
      camOffsetX: this.App.camOffsetX,
      camOffsetY: this.App.camOffsetY,
      camOffsetZ: this.App.camOffsetZ,
      gyroYaw: this.App.gyroYaw,
      gyroPitch: this.App.gyroPitch,
      cameraHeight: this.App.cameraHeight,
      cameraTiltDeg: this.App.cameraTiltDeg,
      cameraDistance: this.App.cameraDistance,
      isDragging: this.App.isDragging,
      dragOrbitYaw: this.App.dragOrbitYaw,
      dragOrbitPitch: this.App.dragOrbitPitch,
      userRotY: this.App.userRotY,
      userRotX: this.App.userRotX,
      focusPartActive: this.App.focusPart ? this.App.focusPart.active : false,
    };
    // 强制禁用拖拽旋转（游戏内相机由本管理器接管）
    this.App.isDragging = false;
    this.App.dragOrbitYaw = 0;
    this.App.dragOrbitPitch = 0;
    this.App.userRotY = 0;
    this.App.userRotX = 0;
    this.App.gyroYaw = 0;
    this.App.gyroPitch = 0;
    if (this.App.focusPart) this.App.focusPart.active = false;
    // 游戏轨道摄像机参数
    this._camAzimuth = 0;           // 水平旋转角 (rad)，绕角色左右转
    this._camPitch = 0.35;          // 垂直俯仰角 (rad)，0=水平，正=上方俯视
    this._camRadius = 5.0;          // 距离（滚轮拉近推远），范围 1.5~15
    this._camLookTarget = new this.App.THREE.Vector3();
    this._camCurrent = new this.App.THREE.Vector3();
    this._camSmooth = 0.08;         // 平滑跟随速度
    // 暴露给控制桥接
    this.App._gameCamAzimuth = 0;
    this.App._gameCamPitch = 0.35;
    this.App._gameCamRadius = 5.0;

    // 行走动画
    this._walkAnimTime = 0;
    this._walkAnimActive = false;
    this._walkPhase = 0;           // 行走相位 (rad)
    this._prevAvatarPosY = 0;      // 记录角色初始Y位置
    this._WALK_STEP_LENGTH = 1.5;  // 每步距离（控制步频），值越大步频越慢

    // ========== 隔离大厅场景：隐藏大厅对象，确保游戏独立空间 ==========
    if (!this._savedSceneObjects) {
      this._savedSceneObjects = [];
      const lobbyObjects = [
        { obj: this.App.backgroundGroup, name: 'backgroundGroup' },
        { obj: this.App.starField, name: 'starField' },
        { obj: this.App.parts && this.App.parts.contactShadow, name: 'contactShadow' },
        { obj: this.App.parts && this.App.parts.glow, name: 'glow' },
      ];
      for (const { obj, name } of lobbyObjects) {
        if (obj && obj.visible !== false) {
          obj.visible = false;
          this._savedSceneObjects.push({ obj, name });
        }
      }
    }

    // 创建游戏实例
    const game = factory(this.App);
    this.currentGame = game;

    // 生成游戏场景
    game.generateScene();

    // 独立系统游戏（赛博公司）：彻底切断大厅感知——
    // sendAIAction / sendRLSync / 点击互动等直发 ws 通道绕过 sendText hook，
    // 必须整体禁用，防止玩家操作/角色行为被当成大厅感知触发大厅 AI 聆听与回应
    if (game.isIsolated) {
      this._installLobbyIsolation(game);
    }

    // 设置 AI 自主控制器的地图数据
    const mapData = game._getMapData ? game._getMapData() : null;
    if (mapData) {
      this.aiAutonomy.setMapData(mapData, mapData.type || 'grid');
    }
    this.aiAutonomy.setUserControlling(true);

    // 设置控制桥接
    this.controlBridge.activateUserControl();
    this.controlBridge.setSpeed(game.moveSpeed || 3.5);

    // 绑定状态感知器（独立系统游戏不接入大厅感知：不向大厅后端推送游戏状态）
    if (!game.isIsolated) {
      this.stateObserver.bind(game);
    }

    // 启动游戏
    game.onStart();

    // 设置摄像机初始位置：角色正后方，对准角色，距离由游戏配置（默认 2m）
    // 用户操控时摄像机与角色朝向解绑（仅拖拽改变方位角）；AI 自主时绑定在背后跟随
    const initFacing = (this.App.smoothRotY || 0);
    this._camAzimuth = initFacing + Math.PI;
    this._camRadius = (typeof game.initialCameraRadius === 'number') ? game.initialCameraRadius : 2.0;
    // 由期望高度反算俯仰角：cameraY = avatarY + headH + R*sin(pitch)
    const _camHeadH = 1.2;
    if (typeof game.initialCameraHeight === 'number') {
      const sinP = (game.initialCameraHeight - _camHeadH) / this._camRadius;
      this._camPitch = Math.asin(Math.max(-1, Math.min(1, sinP)));
    }
    this.App._gameCamAzimuth = this._camAzimuth;      // 供首帧外部读取
    this.App._gameCamPitch = this._camPitch;
    this.App._gameCamRadius = this._camRadius;
    if (this._camCurrent) this._camCurrent.set(0, 0, 0);  // 重置缓存，首帧直接到位
    this._updateGameCamera(0);

    // 移动端：请求全屏横屏
    if (this._isMobile) {
      this._requestFullscreenLandscape();
    }

    // 显示游戏UI
    this._createGameUI(game);

    // 通知服务器进入游戏模式
    if (this.App.ws && this.App.ws.readyState === WebSocket.OPEN) {
      this.App.ws.send(JSON.stringify({
        type: 'enter_game_mode',
        game_key: game.name,
        game_name: game.displayName
      }));
    }

    this.active = true;

    // 启动游戏循环
    this._rafId = requestAnimationFrame(this._gameLoopBound);

    // 显示控制提示
    const baseHint = this._isMobile
      ? '🎮 摇杆移动 · 滑动视角 · 双指缩放'
      : '🎮 WASD移动 · 空格/单击跳跃 · 双击二段跳 · 拖拽转向 · 滚轮视角';
    const controlHint = game.requestJump ? baseHint
      : (this._isMobile
        ? `🎮 进入「${game.displayName}」· 摇杆移动 · 滑动视角 · 双指缩放`
        : `🎮 进入「${game.displayName}」· WASD移动 · 拖拽转向 · 滚轮视角`);
    this.App.showToast(controlHint);

    // 触发AI进入游戏的反应
    if (this.App.sendAIAction) {
      this.App.sendAIAction(`（你和小伙伴进入了「${game.displayName}」游戏世界！周围是全新的游戏场景，用户正在操控你的身体探索。请兴奋地感受这个新世界。）`);
    }
  }

  /**
   * 独立系统游戏（赛博公司）专用：安装大厅感知隔离器。
   * 这些通道直接 App.ws.send，绕过 sendText/addUserMsg 的 hook——
   * 不禁用会让玩家在游戏内的操作（点选、换装、设置等 UI 动作）、AI 自主行为、
   * RL 统一心跳、戳一戳互动全部作为"大厅感知"推送给大厅 AI，触发其聆听与回应。
   */
  _installLobbyIsolation(game) {
    const App = this.App;
    game._isolatedFns = {};
    const noops = ['sendAIAction', '_sendAIActionNow', '_flushPendingAIActions', 'sendRLSync'];
    for (const k of noops) {
      if (typeof App[k] === 'function') {
        game._isolatedFns[k] = App[k];
        App[k] = function isolatedNoop() {};
      }
    }
    if (App._pendingAIActions) App._pendingAIActions = [];
  }

  /** 退出独立系统游戏：恢复大厅感知函数（与大厅彻底隔离互不残留） */
  _restoreLobbyIsolation(game) {
    if (game && game._isolatedFns) {
      for (const k of Object.keys(game._isolatedFns)) {
        this.App[k] = game._isolatedFns[k];
      }
      game._isolatedFns = null;
    }
  }

  /**
   * 角度插值（最短路径，处理 ±π 环绕）
   * AI 自主移动与玩家操控共用：避免跨越 ±π 时角色走长弧（整圈旋转）。
   * 替代裸 App.lerp 在朝向插值上的使用。
   */
  _lerpAngle(a, b, t) {
    let diff = b - a;
    while (diff > Math.PI) diff -= Math.PI * 2;
    while (diff < -Math.PI) diff += Math.PI * 2;
    return a + diff * t;
  }

  _applyPlayerMovement(moveVec, dt) {
    // 幽灵锚点优先：游戏可声明 playerAnchor（无实体 Object3D），移动逻辑完全脱离大厅角色模型
    const avatar = (this.currentGame && this.currentGame.playerAnchor) || this.App.currentAvatar;
    if (!avatar || !moveVec || !moveVec.isMoving) {
      if (this.currentGame && this.currentGame.setPlayerSpeed) {
        this.currentGame.setPlayerSpeed(0);
      }
      return false;
    }

    let dx = moveVec.x || 0;
    let dz = moveVec.z || 0;

    // 渐强因子：使实际位移与行走动画同步加速
    const rampFactor = this._walkRampFactor !== undefined ? this._walkRampFactor : 1.0;
    dx *= rampFactor;
    dz *= rampFactor;

    // 计算实际世界空间速度
    const actualSpeed = Math.sqrt(dx * dx + dz * dz) / Math.max(dt, 0.001);
    if (this.currentGame && this.currentGame.setPlayerSpeed) {
      this.currentGame.setPlayerSpeed(actualSpeed);
    }

    let newX = avatar.position.x + dx;
    let newZ = avatar.position.z + dz;
    const savedY = avatar.position.y;

    // 碰撞检测
    if (this.currentGame && this.currentGame.checkCollision) {
      // 腾空时跳过地形碰撞（让 AI 起跳后能在空中平移翻越矮墙），但仍保留实体碰撞（不穿树/岩石）
      // 落地由游戏物理层 _updatePlayerPhysics 自动修正 Y
      const isAirborne = this.currentGame._isGrounded === false;
      const collisionOpts = isAirborne ? { ignoreTerrain: true } : {};

      const blocked = this.currentGame.checkCollision(newX, newZ, collisionOpts);
      if (blocked) {
        // 尝试抬脚跨过（自动迈步）：临时抬高 avatar Y 再检测
        // 例如前方有 1 格台阶（1m），checkCollision 在脚底 Y 处判定阻挡，
        // 抬脚后台阶变得"可跨过去"，移动获批后由物理层 _updatePlayerPhysics 自动修正 Y
        const STEP_UP_HEIGHT = 1.05; // 略大于 1 格方块高度
        avatar.position.y = savedY + STEP_UP_HEIGHT;
        const blockedAfterStepUp = this.currentGame.checkCollision(newX, newZ, collisionOpts);
        avatar.position.y = savedY;

        if (!blockedAfterStepUp) {
          // 抬脚后可通过 —— 允许移动，物理层会处理 Y 定位
          newX = avatar.position.x + dx;
          newZ = avatar.position.z + dz;
        } else {
          // 抬脚也过不去（≥2格悬崖/实体阻挡）→ 尝试贴墙滑动
          const bxOnly = this.currentGame.checkCollision(newX, avatar.position.z, collisionOpts);
          const bzOnly = this.currentGame.checkCollision(avatar.position.x, newZ, collisionOpts);
          if (!bxOnly) { newX = avatar.position.x + dx; newZ = avatar.position.z; }
          else if (!bzOnly) { newX = avatar.position.x; newZ = avatar.position.z + dz; }
          else {
            // 完全被阻挡（贴墙也过不去）
            return false;
          }
        }
      }
    }

    avatar.position.set(newX, avatar.position.y, newZ);

    // 角色面向移动方向（AI 自主移动与玩家操控共用同一套旋转逻辑：最短角度路径插值）
    // 输入来源不同（玩家=摄像机相对方向，AI=寻路目标方向），但朝向追随效果完全一致
    if (Math.abs(dx) > 0.0001 || Math.abs(dz) > 0.0001) {
      const moveAngle = Math.atan2(dx, dz);
      const lerpSpeed = Math.min(1, 8.0 * dt);
      this.App.smoothRotY = this._lerpAngle(this.App.smoothRotY || 0, moveAngle, lerpSpeed);
    }

    // 边界限制
    if (this.currentGame && this.currentGame.boundarySize) {
      const halfBoundary = this.currentGame.boundarySize / 2;
      avatar.position.x = Math.max(-halfBoundary, Math.min(halfBoundary, avatar.position.x));
      avatar.position.z = Math.max(-halfBoundary, Math.min(halfBoundary, avatar.position.z));
    }

    return true;
  }

  /**
   * AI 自主移动：生成与用户操控完全一致的 moveVec。
   * 方向 = 当前位置 → 目标点，速度 = 游戏移动速度，带 ramp 加速，
   * 旋转由 _applyPlayerMovement 统一处理（face direction = walk direction）。
   */
  _buildAIMoveVec(dt) {
    const App = this.App;
    if (!App.idleWalkTarget) return null;

    const avatar = App.currentAvatar || App.modelGroup;
    if (!avatar) return null;

    // 方向：从当前位置指向目标点（世界空间）
    const dx = App.idleWalkTarget.x - avatar.position.x;
    const dz = App.idleWalkTarget.z - avatar.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    if (dist < 0.01) return null;

    // 归一化 × 游戏移动速度 × dt（格式与 controlBridge.updateMovement 完全一致）
    const speed = this.controlBridge._moveSpeed * dt;
    return {
      x: (dx / dist) * speed,
      z: (dz / dist) * speed,
      isMoving: true,
    };
  }

  /**
   * AI 平滑攀爬过渡：以正常移动速度向攀爬目标移动，替代瞬移避免视觉不适
   * 每帧调用，直接修改 avatar 位置（绕过碰撞检测），Y 坐标按移动进度插值。
   * 返回 moveVec 供行走动画使用；攀爬完成时返回 null。
   * @param {number} dt - 帧时间增量（秒）
   * @returns {Object|null} moveVec，null 表示攀爬已完成
   */
  _updateAIClimb(dt) {
    const avatar = this.App.currentAvatar || this.App.modelGroup;
    const target = this._aiClimbTarget;
    if (!avatar || !target) {
      this._aiClimbTarget = null;
      this._aiClimbCurrentPos = null;
      return null;
    }

    const startX = avatar.position.x;
    const startZ = avatar.position.z;
    const dx = target.x - startX;
    const dz = target.z - startZ;
    const dist = Math.hypot(dx, dz);
    const speed = (this.controlBridge._moveSpeed || 3.5) * dt;

    let newX, newZ, newY, moveX, moveZ;
    let completed = false;

    if (dist <= speed || dist < 0.01) {
      // 到达目标
      newX = target.x;
      newZ = target.z;
      newY = target.endY;
      moveX = dx;
      moveZ = dz;
      completed = true;
    } else {
      // 沿目标方向以正常移动速度移动
      const nx = dx / dist;
      const nz = dz / dist;
      moveX = nx * speed;
      moveZ = nz * speed;
      newX = startX + moveX;
      newZ = startZ + moveZ;
      // Y 坐标按移动进度平滑过渡（从起点高度爬升/下降到终点高度）
      const movedDist = target.totalDist - dist;
      const progress = target.totalDist > 0.01
        ? Math.max(0, Math.min(1, movedDist / target.totalDist))
        : 1;
      newY = target.startY + (target.endY - target.startY) * progress;
    }

    // 边界限制
    if (this.currentGame && this.currentGame.boundarySize) {
      const halfBoundary = this.currentGame.boundarySize / 2;
      newX = Math.max(-halfBoundary, Math.min(halfBoundary, newX));
      newZ = Math.max(-halfBoundary, Math.min(halfBoundary, newZ));
    }

    // 应用到 avatar
    avatar.position.set(newX, newY, newZ);

    // 记录本帧位置，供 _enforceClimbPosition 使用（物理层可能修改 avatar 位置）
    this._aiClimbCurrentPos = { x: newX, y: newY, z: newZ };

    // 角色面向攀爬方向（与移动管道共用最短角度路径插值）
    if (Math.abs(moveX) > 0.0001 || Math.abs(moveZ) > 0.0001) {
      const moveAngle = Math.atan2(moveX, moveZ);
      const lerpSpeed = Math.min(1, 8.0 * dt);
      this.App.smoothRotY = this._lerpAngle(this.App.smoothRotY || 0, moveAngle, lerpSpeed);
    }

    if (completed) {
      this._aiClimbTarget = null;
      this._aiClimbCurrentPos = null;
      console.log('[游戏模式] AI 平滑攀爬完成');
      return null;
    }

    // 返回 moveVec 用于行走动画（保持攀爬过程中有行走动作）
    return {
      x: moveX,
      z: moveZ,
      isMoving: true,
      stepLength: this._WALK_STEP_LENGTH || 3.0,
    };
  }

  /**
   * 强制恢复攀爬位置：在 currentGame.update(dt) 之后调用，
   * 覆盖物理层（重力/地面修正）对 avatar 位置的修改，保证攀爬过渡平滑。
   */
  _enforceClimbPosition() {
    if (!this._aiClimbCurrentPos) return;
    const avatar = this.App.currentAvatar || this.App.modelGroup;
    if (avatar) {
      avatar.position.set(
        this._aiClimbCurrentPos.x,
        this._aiClimbCurrentPos.y,
        this._aiClimbCurrentPos.z
      );
    }
  }

  /**
   * 计算 AI 跳跃的安全落点
   * @param {{x:number, z:number}} pos - 当前位置
   * @param {{x:number, z:number}} target - 目标位置
   * @param {number} distance - 期望落点距离
   * @returns {{x:number, z:number}|null}
   */
  _findSafeJumpLanding(pos, target, distance) {
    const dirX = target.x - pos.x;
    const dirZ = target.z - pos.z;
    const dirLen = Math.hypot(dirX, dirZ);
    if (dirLen < 0.01) return null;

    const landing = {
      x: pos.x + (dirX / dirLen) * distance,
      z: pos.z + (dirZ / dirLen) * distance,
    };
    if (this._isLandingSafe(this.currentGame, landing.x, landing.z, pos.x, pos.z)) {
      return landing;
    }
    return null;
  }

  /**
   * 计算 AI 瞬移的安全位置：沿目标方向从近到远寻找最远的安全落点
   * @param {{x:number, z:number}} pos - 当前位置
   * @param {{x:number, z:number}} target - 目标位置
   * @param {number} maxDistance - 最大瞬移距离
   * @returns {{x:number, z:number}|null}
   */
  _findSafeTeleportPos(pos, target, maxDistance) {
    const dirX = target.x - pos.x;
    const dirZ = target.z - pos.z;
    const dirLen = Math.hypot(dirX, dirZ);
    if (dirLen < 0.01) return null;

    const nx = dirX / dirLen;
    const nz = dirZ / dirLen;
    let best = null;
    // 从 0.5 米开始逐步探测，步长 0.5 米
    for (let d = 0.5; d <= maxDistance + 0.001; d += 0.5) {
      const tx = pos.x + nx * d;
      const tz = pos.z + nz * d;
      if (this._isLandingSafe(this.currentGame, tx, tz, pos.x, pos.z)) {
        best = { x: tx, z: tz };
      } else {
        // 遇到不安全点停止，不再往前探
        break;
      }
    }
    return best;
  }

  /**
   * 检查指定世界坐标是否可以作为 AI 跳跃/瞬移的安全落点
   * @param {Object} game - 当前游戏实例
   * @param {number} x - 落点世界坐标 X
   * @param {number} z - 落点世界坐标 Z
   * @param {number} [fromX] - 当前位置 X（用于高度差校验）
   * @param {number} [fromZ] - 当前位置 Z（用于高度差校验）
   * @returns {boolean}
   */
  _isLandingSafe(game, x, z, fromX, fromZ) {
    if (!game) return false;

    // 沙盒/体素模式：检查区块已加载、不是水体、高度差在可接受范围
    if (game._getGroundHeight) {
      const groundY = game._getGroundHeight(x, z);
      // 高度异常也视为不安全
      if (groundY === undefined || groundY === null || Number.isNaN(groundY)) return false;

      // 与当前位置的高度差校验，防止跳到悬崖下方或过高的位置
      if (fromX !== undefined && fromZ !== undefined) {
        const fromGroundY = game._getGroundHeight(fromX, fromZ);
        const MAX_LANDING_HEIGHT_DIFF = 2.0; // 米
        if (Math.abs(groundY - fromGroundY) > MAX_LANDING_HEIGHT_DIFF) return false;
      }

      // 实体/地形碰撞校验（沙盒中的树木、岩石等）
      if (game.checkCollision && game.checkCollision(x, z)) return false;

      // 未加载区块或 fallback 到默认高度时，尝试读取列数据二次确认
      if (game._getColumn) {
        const blockSize = game.blockSize || 1;
        const col = game._getColumn(Math.round(x / blockSize), Math.round(z / blockSize));
        if (!col) return false;
        if (col.type === 'water') return false;
      }
      return true;
    }

    // 迷宫模式：检查是否与墙壁碰撞
    if (game.checkCollision) {
      return !game.checkCollision(x, z);
    }

    return false;
  }

  /**
   * 检查从当前位置到目标落点之间是否可步行通过
   * @param {Object} game - 当前游戏实例
   * @param {number} fromX, fromZ - 起点世界坐标
   * @param {number} toX, toZ - 终点世界坐标
   * @returns {boolean}
   */
  _isPathWalkable(game, fromX, fromZ, toX, toZ) {
    if (!game) return false;

    if (game._isWalkableBetween) {
      return game._isWalkableBetween(fromX, fromZ, toX, toZ);
    }

    if (game.checkCollision) {
      const steps = Math.max(3, Math.floor(Math.hypot(toX - fromX, toZ - fromZ) / 0.5));
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const cx = fromX + (toX - fromX) * t;
        const cz = fromZ + (toZ - fromZ) * t;
        if (game.checkCollision(cx, cz)) return false;
      }
      return true;
    }

    return true;
  }

  /**
   * 更新游戏摄像机（观察者机位）：
   * - 摄像机始终是角色的"观察者"：方位角独立于角色朝向，仅由拖拽改变
   * - 无论用户操控还是 AI 自主，角色转身/移动都不会带动摄像机旋转
   * - 摄像机始终跟随角色位置（绕角色轨道），刚性吸附、无滞后，始终看向角色头部
   */
  _updateGameCamera(dt) {
    // 幽灵锚点优先：游戏可声明 playerAnchor（无实体 Object3D），相机完全跟随幽灵玩家位置
    const avatar = (this.currentGame && this.currentGame.playerAnchor) || this.App.currentAvatar;
    const camera = this.App.camera;
    if (!avatar || !camera) return;

    const THREE = this.App.THREE;
    // 第一人称模式（游戏声明 fpvCamera=true）：相机 = 玩家"幽灵"眼睛位置，
    // 朝向由 _gameCamAzimuth（水平）/_gameCamPitch（俯仰）控制，拖拽即视角转动
    if (this.currentGame && this.currentGame.fpvCamera) {
      const eyeH = typeof this.currentGame.fpvEyeHeight === 'number' ? this.currentGame.fpvEyeHeight : 1.5;
      const A = this.App._gameCamAzimuth || 0;
      const P = this.App._gameCamPitch || 0;
      // 站立高度：优先取游戏地面高度 + 眼睛高度（幽灵漂浮）
      let groundY = avatar.position.y;
      if (this.currentGame._getGroundHeight) {
        try {
          const gy = this.currentGame._getGroundHeight(avatar.position.x, avatar.position.z);
          if (gy !== undefined && gy !== null && !Number.isNaN(gy)) groundY = gy;
        } catch (e) {}
      }
      camera.position.set(avatar.position.x, groundY + eyeH, avatar.position.z);
      const cosP = Math.cos(P);
      const dir = new THREE.Vector3(-Math.sin(A) * cosP, Math.sin(P), -Math.cos(A) * cosP);
      this._camLookTarget.copy(camera.position).add(dir);
      camera.lookAt(this._camLookTarget);
      return;
    }

    // 方位角完全独立于角色朝向（观察者模式）：仅由拖拽改变，角色转身不影响摄像机
    let azimuth = this.App._gameCamAzimuth;
    if (azimuth === undefined || azimuth === null) {
      azimuth = (this.App.smoothRotY || 0) + Math.PI;
      this.App._gameCamAzimuth = azimuth;
    }

    // 空值合并：pitch 为 0（水平视角）时保持 0，而非被 || 强转为 0.35
    // （否则用户把视角拖到水平会被弹回俯视，无法精确仰视/俯视）
    const pitch = this.App._gameCamPitch ?? 0.35;
    const R = this.App._gameCamRadius || 5.0;

    // 球心 = 角色头部
    const headH = 1.2;
    const cx = avatar.position.x;
    const cy = avatar.position.y + headH;
    const cz = avatar.position.z;

    // 球形坐标 → 世界坐标（轨道摄像机）
    // pitch: 0=水平, >0=上方俯视
    const cosP = Math.cos(pitch);
    const sinP = Math.sin(pitch);
    const cosA = Math.cos(azimuth);
    const sinA = Math.sin(azimuth);

    const targetX = cx + R * cosP * sinA;
    const targetY = cy + R * sinP;
    const targetZ = cz + R * cosP * cosA;

    // 刚性绑定：直接吸附到目标位置（与角色转向/位移同步，无滞后）
    this._camCurrent.set(targetX, targetY, targetZ);

    camera.position.copy(this._camCurrent);
    this._camLookTarget.set(cx, cy, cz);
    camera.lookAt(this._camLookTarget);

    // 游戏自定义相机修正钩子（如幽灵玩家的防穿模：镜头不穿进角色模型）
    if (this.currentGame && typeof this.currentGame.afterCameraUpdate === 'function') {
      try { this.currentGame.afterCameraUpdate(camera, this._camLookTarget); } catch (e) {}
    }
  }

  /**
   * 行走动画（统一使用 App.applyFullBodyWalkAnimation）
   */
  _updateWalkAnimation(dt, moveVec) {
    // 传入游戏模式步长
    if (moveVec) moveVec.stepLength = this._WALK_STEP_LENGTH || 3.0;
    this.App.applyFullBodyWalkAnimation(dt, moveVec);
  }

  /** 重置行走动画状态和手臂到休息姿势 */
  _resetWalkAnimArms() {
    const App = this.App;
    // 重置行走动画状态
    App._fullWalkAnimActive = false;
    App._fullWalkPhase = 0;
    App._fullWalkRampT = 0;
    App._fullWalkRampFactor = 1.0;
    // 将手臂骨骼恢复到放松站姿（ARM_REST_Z）
    try {
      const B = App.vrmBones;
      if (B) {
        if (B.leftUpperArm)  B.leftUpperArm.rotation.set(0, 0,  App.ARM_REST_Z || 1.35);
        if (B.rightUpperArm) B.rightUpperArm.rotation.set(0, 0, -(App.ARM_REST_Z || 1.35));
        if (B.leftLowerArm)  B.leftLowerArm.rotation.set(0, 0, 0);
        if (B.rightLowerArm) B.rightLowerArm.rotation.set(0, 0, 0);
      }
    } catch (e) {
      // VRM 骨骼不可用时忽略
    }
  }

  _updateSceneEffects(t) {
    // 更新生成的发光物体
    const objects = this.sceneGenerator.generatedObjects;
    for (const obj of objects) {
      if (obj.userData && obj.userData.ring) {
        obj.userData.ring.rotation.z += 0.02;
        obj.userData.particles.rotation.y += 0.01;
      }
      if (obj.userData && obj.userData.sprite) {
        obj.userData.sprite.material.opacity = 0.5 + Math.sin(t * 3) * 0.3;
      }
      if (obj.userData && obj.userData.isCollectible) {
        obj.rotation.y += 0.03;
        obj.position.y += Math.sin(t * 4 + obj.position.x) * 0.003;
      }
    }

    // 更新游戏自身的特效
    if (this.currentGame && this.currentGame.updateSceneEffects) {
      this.currentGame.updateSceneEffects(t);
    }
  }

  // ==================== UI ====================

  _createGameUI(game) {
    // 游戏模式UI覆盖层
    const overlay = document.createElement('div');
    overlay.id = 'game-mode-overlay';
    const controlHintText = this._isMobile
      ? '摇杆移动 · 滑动视角 · 双指缩放'
      : 'WASD 移动 · 拖拽转向 · 滚轮视角';
    overlay.innerHTML = `
      <div class="game-hud">
        <div class="game-hud-top">
          <div class="game-title">${game.displayName}</div>
          <div class="game-stats">
            <span class="game-score">⭐ <span id="game-score-val">0</span></span>
            <span class="game-timer">⏱ <span id="game-timer-val">0</span>s</span>
          </div>
          <button id="game-exit-btn" class="game-exit-btn">✕ 退出游戏</button>
        </div>
        <div class="game-hint" id="game-hint">${controlHintText}</div>
      </div>
    `;
    document.body.appendChild(overlay);

    this._uiOverlay = overlay;
    this._uiScoreEl = overlay.querySelector('#game-score-val');
    this._uiTimerEl = overlay.querySelector('#game-timer-val');
    this._uiHintEl = overlay.querySelector('#game-hint');

    // 退出按钮
    const exitBtn = overlay.querySelector('#game-exit-btn');
    exitBtn.addEventListener('click', () => this.exitGameMode());

    // 键盘事件 - 绑定到document
    this._onKeyDown = (e) => this.controlBridge.handleKeyDown(e.key.toLowerCase());
    this._onKeyUp = (e) => this.controlBridge.handleKeyUp(e.key.toLowerCase());
    document.addEventListener('keydown', this._onKeyDown);
    document.addEventListener('keyup', this._onKeyUp);

    const canvas = this.App.canvas;

    // ========== PC端控制：鼠标拖拽 + 滚轮 ==========
    if (!this._isMobile) {
      this._gameDragInfo = { dragging: false, lastX: 0, lastY: 0, startX: 0, startY: 0, startTime: 0 };
      this._lastClickTime = 0;        // 上一次点击时间，用于双击检测
      this._clickTimeout = null;      // 单击延迟定时器（用于区分单击/双击）

      this._onGamePointerDown = (e) => {
        this._gameDragInfo.dragging = true;
        this._gameDragInfo.lastX = e.clientX;
        this._gameDragInfo.lastY = e.clientY;
        this._gameDragInfo.startX = e.clientX;
        this._gameDragInfo.startY = e.clientY;
        this._gameDragInfo.startTime = performance.now();
        canvas.setPointerCapture(e.pointerId);
      };
      this._onGamePointerMove = (e) => {
        if (!this._gameDragInfo.dragging) return;
        const dx = e.clientX - this._gameDragInfo.lastX;
        const dy = e.clientY - this._gameDragInfo.lastY;
        this.controlBridge.addOrbitDrag(-dx, dy);
        this._gameDragInfo.lastX = e.clientX;
        this._gameDragInfo.lastY = e.clientY;
      };
      this._onGamePointerUp = (e) => {
        const dragInfo = this._gameDragInfo;
        const moved = Math.abs(e.clientX - dragInfo.startX) + Math.abs(e.clientY - dragInfo.startY);
        const elapsed = performance.now() - dragInfo.startTime;
        dragInfo.dragging = false;

        // 判断是否为快速点击（非拖拽）：移动距离小且时间短
        if (moved < 6 && elapsed < 400) {
          this._handleClickJump();
        }
      };
      this._onGameWheel = (e) => {
        e.preventDefault();
        this.controlBridge.addZoom(e.deltaY);
      };

      canvas.addEventListener('pointerdown', this._onGamePointerDown);
      canvas.addEventListener('pointermove', this._onGamePointerMove);
      canvas.addEventListener('pointerup', this._onGamePointerUp);
      canvas.addEventListener('pointerleave', this._onGamePointerUp);
      canvas.addEventListener('wheel', this._onGameWheel, { passive: false });
    }

    // ========== 移动端控制：摇杆 + 滑动视角 + 双指缩放 ==========
    if (this._isMobile) {
      this._initMobileControls(canvas, overlay);
    }

    // 防止浏览器默认行为（如空格滚动）
    document.addEventListener('keydown', this._preventDefaultKeys, { passive: false });
  }

  /** 初始化移动端触控 */
  _initMobileControls(canvas, overlay) {
    // --- 隐藏侧边栏和聊天面板，释放全屏视野 ---
    this._hiddenElements = [];
    const hideSelectors = ['.stage-tools', '#chat-panel', '#controls', '#chat-toggle'];
    for (const sel of hideSelectors) {
      const el = document.querySelector(sel);
      if (el && el.style.display !== 'none') {
        el.style.display = 'none';
        this._hiddenElements.push(el);
      }
    }
    this._joystick = new VirtualJoystick({
      container: overlay,
      size: 130,
      thumbSize: 56,
      margin: 20,
      onMove: (val) => {
        // 摇杆输出映射到 ControlBridge 的键盘状态
        // val.x: -1(left) ~ 1(right) → 左右移动
        // val.z: -1(down=前) ~ 1(up=后) → 前后移动
        this.controlBridge.setVirtualInput(val);
      },
      onEnd: () => {
        this.controlBridge.setVirtualInput({ x: 0, z: 0, isMoving: false });
      }
    });

    // --- Canvas触控：滑动视角 + 双指缩放 ---
    this._touchState = {
      touches: {},        // touchId → { startX, startY, lastX, lastY }
      orbitId: null,      // 用于单指轨道控制
      pinchActive: false,
      pinchStartDist: 0,
      pinchStartRadius: 0,
    };

    this._onTouchStart = (e) => {
      // 如果触摸在摇杆区域，跳过（摇杆自己处理）
      for (let i = 0; i < e.changedTouches.length; i++) {
        const t = e.changedTouches[i];
        const target = document.elementFromPoint(t.clientX, t.clientY);
        if (target && target.closest('.virtual-joystick-outer')) {
          continue; // 摇杆处理
        }
        this._touchState.touches[t.identifier] = {
          startX: t.clientX, startY: t.clientY,
          lastX: t.clientX, lastY: t.clientY,
          startTime: performance.now(), // 记录触摸开始时间，用于单击/双击检测
        };
      }

      const count = Object.keys(this._touchState.touches).length;

      if (count === 1 && !this._touchState.orbitId) {
        // 单指：开始轨道控制
        const ids = Object.keys(this._touchState.touches);
        this._touchState.orbitId = ids[0];
      } else if (count >= 2) {
        // 双指：开始缩放
        this._touchState.pinchActive = true;
        this._touchState.orbitId = null; // 停止轨道
        this._updatePinch();
      }
      e.preventDefault();
    };

    this._onTouchMove = (e) => {
      e.preventDefault();
      const state = this._touchState;

      // 更新touch位置
      for (let i = 0; i < e.changedTouches.length; i++) {
        const t = e.changedTouches[i];
        if (state.touches[t.identifier]) {
          const ti = state.touches[t.identifier];
          ti.lastX = t.clientX;
          ti.lastY = t.clientY;
        }
      }

      const count = Object.keys(state.touches).length;

      // 双指缩放
      if (count >= 2 && state.pinchActive) {
        this._updatePinch();
      }
    };

    this._onTouchEnd = (e) => {
      for (let i = 0; i < e.changedTouches.length; i++) {
        const id = e.changedTouches[i].identifier;
        const touchData = this._touchState.touches[id];

        // 检测快速点击（非拖拽）：移动距离小且时间短 → 触发跳跃
        if (touchData) {
          const dx = touchData.lastX - touchData.startX;
          const dy = touchData.lastY - touchData.startY;
          const moved = Math.abs(dx) + Math.abs(dy);
          const elapsed = performance.now() - touchData.startTime;
          if (moved < 10 && elapsed < 300) {
            this._handleClickJump();
          }
        }

        delete this._touchState.touches[id];
        if (this._touchState.orbitId === String(id)) {
          this._touchState.orbitId = null;
        }
      }

      const count = Object.keys(this._touchState.touches).length;
      if (count < 2) {
        this._touchState.pinchActive = false;
        this._touchState.pinchStartDist = 0;  // 重置，下次双指重新记录起点
      }
      if (count === 1 && !this._touchState.pinchActive && !this._touchState.orbitId) {
        // 恢复到单指轨道
        const ids = Object.keys(this._touchState.touches);
        this._touchState.orbitId = ids[0];
      }
    };

    this._onTouchCancel = (e) => {
      this._onTouchEnd(e);
    };

    canvas.addEventListener('touchstart', this._onTouchStart, { passive: false });
    canvas.addEventListener('touchmove', this._onTouchMove, { passive: false });
    canvas.addEventListener('touchend', this._onTouchEnd);
    canvas.addEventListener('touchcancel', this._onTouchCancel);
  }

  /** 每帧调用的移动端轨道更新（由_gameLoop触发） */
  _updateMobileOrbit() {
    const state = this._touchState;
    if (!state) return false;

    // 单指轨道控制
    if (state.orbitId !== null && state.touches[state.orbitId]) {
      const ti = state.touches[state.orbitId];
      const dx = ti.lastX - ti.startX;
      const dy = ti.lastY - ti.startY;
      if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
        // 消费增量（重置起点，使滑动连续）
        this.controlBridge.addOrbitDrag(-dx, dy);
        ti.startX = ti.lastX;
        ti.startY = ti.lastY;
        return true;
      }
    }
    return false;
  }

  /** 更新双指缩放 */
  _updatePinch() {
    const state = this._touchState;
    const ids = Object.keys(state.touches);
    if (ids.length < 2) return;

    const t0 = state.touches[ids[0]];
    const t1 = state.touches[ids[1]];
    const dx = t0.lastX - t1.lastX;
    const dy = t0.lastY - t1.lastY;
    const currentDist = Math.sqrt(dx * dx + dy * dy);

    if (state.pinchStartDist === 0) {
      state.pinchStartDist = currentDist;
      state.pinchStartRadius = this.App._gameCamRadius || 5.0;
      return;
    }

    // 手指靠拢 → scale<1 → 拉近(半径减小)；手指张开 → scale>1 → 拉远(半径增大)
    // 降低灵敏度：0.7 系数防止缩放过快
    const scale = 1 + (currentDist / state.pinchStartDist - 1) * 0.7;
    const newRadius = Math.max(1.5, Math.min(15.0, state.pinchStartRadius / scale));
    this.App._gameCamRadius = newRadius;
  }

  _preventDefaultKeys = (e) => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ', 'Space'].includes(e.key)) {
      e.preventDefault();
    }
    // 空格键触发跳跃
    if (e.key === ' ' || e.key === 'Spacebar' || e.code === 'Space') {
      e.preventDefault();
      this._handleClickJump();
    }
  }

  /**
   * 处理单击/双击跳跃
   * - 每次点击/空格立即触发跳跃，不再延迟
   * - 300ms内的连续点击会依次消耗剩余跳跃次数（实现二段跳）
   */
  _handleClickJump() {
    const game = this.currentGame;
    if (!game || !game.requestJump) return;

    // 立即触发跳跃，不再等待
    game.requestJump();
  }

  _removeGameUI() {
    // 先移除结算界面
    this._removeSettlementUI();

    // 清理移动端控制
    if (this._joystick) {
      this._joystick.destroy();
      this._joystick = null;
    }
    this._touchState = null;

    // 恢复被隐藏的侧边栏
    if (this._hiddenElements) {
      for (const el of this._hiddenElements) {
        el.style.display = '';
      }
      this._hiddenElements = null;
    }

    if (this._uiOverlay && this._uiOverlay.parentNode) {
      this._uiOverlay.parentNode.removeChild(this._uiOverlay);
    }
    this._uiOverlay = null;
    this._uiScoreEl = null;
    this._uiTimerEl = null;
    this._uiHintEl = null;

    // 移除键盘事件
    if (this._onKeyDown) {
      document.removeEventListener('keydown', this._onKeyDown);
      document.removeEventListener('keyup', this._onKeyUp);
      this._onKeyDown = null;
      this._onKeyUp = null;
    }
    document.removeEventListener('keydown', this._preventDefaultKeys);

    // 移除游戏鼠标/滚轮事件
    const canvas = this.App.canvas;
    if (canvas) {
      if (this._onGamePointerDown) canvas.removeEventListener('pointerdown', this._onGamePointerDown);
      if (this._onGamePointerMove) canvas.removeEventListener('pointermove', this._onGamePointerMove);
      if (this._onGamePointerUp) canvas.removeEventListener('pointerup', this._onGamePointerUp);
      if (this._onGamePointerUp) canvas.removeEventListener('pointerleave', this._onGamePointerUp);
      if (this._onGameWheel) canvas.removeEventListener('wheel', this._onGameWheel);
      // 移除移动端触控事件
      if (this._onTouchStart) canvas.removeEventListener('touchstart', this._onTouchStart);
      if (this._onTouchMove) canvas.removeEventListener('touchmove', this._onTouchMove);
      if (this._onTouchEnd) canvas.removeEventListener('touchend', this._onTouchEnd);
      if (this._onTouchCancel) canvas.removeEventListener('touchcancel', this._onTouchCancel);
    }
    this._onGamePointerDown = null;
    this._onGamePointerMove = null;
    this._onGamePointerUp = null;
    this._onGameWheel = null;
    this._onTouchStart = null;
    this._onTouchMove = null;
    this._onTouchEnd = null;
    this._onTouchCancel = null;
    this._gameDragInfo = null;

    // 清理单击/双击定时器
    if (this._clickTimeout) {
      clearTimeout(this._clickTimeout);
      this._clickTimeout = null;
    }
    this._lastClickTime = 0;
  }

  /** 兜底：强制从 DOM 中移除所有游戏相关的 UI（即使 _removeGameUI 异常也能清理） */
  _forceRemoveAllGameUI() {
    // 移除所有可能残留的游戏 UI 元素
    const selectors = [
      '#game-exit-btn', '.game-exit-btn',
      '.game-ui-overlay', '#game-ui-overlay',
      '.game-hud', '#game-hud',
      '.settlement-overlay',
      '.game-joystick-container',
    ];
    for (const sel of selectors) {
      try {
        document.querySelectorAll(sel).forEach(el => {
          if (el && el.parentNode) el.parentNode.removeChild(el);
        });
      } catch (e) { /* ignore */ }
    }
    this._uiOverlay = null;
    this._settlementOverlay = null;
  }

  _updateGameUI() {
    if (!this.currentGame) return;
    if (this._uiScoreEl) this._uiScoreEl.textContent = this.currentGame.score;
    if (this._uiTimerEl) this._uiTimerEl.textContent = Math.floor(this.currentGame.elapsedTime);
    if (this._uiHintEl) {
      // 模式自带 uiHint 时显示，否则隐藏（避免残留通用 WASD/摇杆提示）
      if (this.currentGame.uiHint) {
        this._uiHintEl.textContent = this.currentGame.uiHint;
        this._uiHintEl.style.display = '';
      } else {
        this._uiHintEl.style.display = 'none';
      }
    }
  }

  /**
   * 显示游戏结算界面
   * @param {BaseGame} game - 游戏实例
   */
  _showSettlement(game) {
    if (this._settlementOverlay) return;

    const isCompleted = game.state === 'completed';
    const elapsed = Math.floor(game.elapsedTime);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    const timeStr = minutes > 0 ? `${minutes}分${seconds}秒` : `${seconds}秒`;

    const resultText = isCompleted ? '游戏完成!' : '游戏结束';
    const resultClass = isCompleted ? 'settlement-success' : 'settlement-fail';
    const resultIcon = isCompleted ? '' : '';

    // 构建额外结果信息
    let extraHTML = '';
    const extra = game.getExtraState ? game.getExtraState() : {};
    const resultData = game.events
      .filter(e => e.type === 'game_completed' || e.type === 'game_failed')
      .pop()?.data || {};

    // 收集所有可展示的额外结果
    const displayItems = [];
    for (const [key, val] of Object.entries({ ...extra, ...resultData })) {
      if (key === 'score' || key === 'elapsed' || key === 'reason') continue;
      if (typeof val === 'boolean' && val) displayItems.push({ label: key, value: '✓' });
      else if (typeof val === 'number') displayItems.push({ label: key, value: val });
      else if (typeof val === 'string' && val) displayItems.push({ label: key, value: val });
    }

    if (displayItems.length > 0) {
      extraHTML = `
        <div class="settlement-extras">
          ${displayItems.map(item => `
            <div class="settlement-extra-item">
              <span class="settlement-extra-label">${item.label}</span>
              <span class="settlement-extra-value">${item.value}</span>
            </div>
          `).join('')}
        </div>`;
    }

    const overlay = document.createElement('div');
    overlay.className = 'settlement-overlay';
    overlay.innerHTML = `
      <div class="settlement-panel">
        <div class="settlement-icon ${resultClass}">${resultIcon}</div>
        <h2 class="settlement-title ${resultClass}">${resultText}</h2>
        <div class="settlement-game-name">${game.displayName}</div>
        <div class="settlement-stats">
          <div class="settlement-stat">
            <span class="settlement-stat-label">得分</span>
            <span class="settlement-stat-value score">${game.score}</span>
          </div>
          <div class="settlement-stat">
            <span class="settlement-stat-label">用时</span>
            <span class="settlement-stat-value time">${timeStr}</span>
          </div>
        </div>
        ${extraHTML}
        <div class="settlement-actions">
          <button class="settlement-btn replay-btn">🔄 再来一次</button>
          <button class="settlement-btn exit-btn">🚪 退出游戏</button>
        </div>
        <div class="settlement-auto-return" id="settlement-auto-return">20 秒后自动返回大厅</div>
      </div>
    `;
    document.body.appendChild(overlay);
    this._settlementOverlay = overlay;

    // 绑定事件
    const replayBtn = overlay.querySelector('.replay-btn');
    const exitBtn = overlay.querySelector('.exit-btn');

    replayBtn.addEventListener('click', () => {
      const gameKey = game.name;
      this._removeSettlementUI();
      this._restartGame(gameKey);
    });

    exitBtn.addEventListener('click', () => {
      this.exitGameMode();
    });

    // 20 秒无操作自动返回大厅（用户点击「再来一次」或「退出游戏」时定时器会被清理）
    let _autoReturnSec = 20;
    const autoReturnEl = overlay.querySelector('#settlement-auto-return');
    this._settlementAutoExitTimer = setInterval(() => {
      _autoReturnSec -= 1;
      if (_autoReturnSec <= 0) {
        clearInterval(this._settlementAutoExitTimer);
        this._settlementAutoExitTimer = null;
        this.exitGameMode();
        return;
      }
      if (autoReturnEl) autoReturnEl.textContent = `${_autoReturnSec} 秒后自动返回大厅`;
    }, 1000);
  }

  /** 移除结算界面 */
  _removeSettlementUI() {
    if (this._settlementAutoExitTimer) {
      clearInterval(this._settlementAutoExitTimer);
      this._settlementAutoExitTimer = null;
    }
    if (this._settlementOverlay && this._settlementOverlay.parentNode) {
      this._settlementOverlay.parentNode.removeChild(this._settlementOverlay);
    }
    this._settlementOverlay = null;
  }

  /** 重新开始当前游戏 */
  _restartGame(gameKey) {
    // 先退出当前游戏（但不恢复相机等完整退出流程）
    const errors = [];
    if (this.currentGame) {
      try { this.currentGame.cleanup(); } catch (e) { errors.push('currentGame.cleanup: ' + e.message); }
      this.currentGame = null;
    }
    try { this.sceneGenerator.cleanup(); } catch (e) { errors.push('sceneGenerator.cleanup: ' + e.message); }
    try { this.controlBridge.cleanup(); } catch (e) { errors.push('controlBridge.cleanup: ' + e.message); }
    try { this.stateObserver.unbind(); } catch (e) { errors.push('stateObserver.unbind: ' + e.message); }
    try { this.controlBridge.releaseUserControl(); } catch (e) { errors.push('releaseUserControl: ' + e.message); }
    try { this._resetWalkAnimArms(); } catch (e) { errors.push('_resetWalkAnimArms: ' + e.message); }

    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }

    this._settlementShown = false;

    // 先清理旧的HUD UI（避免叠加）
    this._removeGameUI();

    // 通知服务器退出当前游戏（重置引擎状态）
    if (this.App.ws && this.App.ws.readyState === WebSocket.OPEN) {
      try {
        this.App.ws.send(JSON.stringify({ type: 'exit_game_mode' }));
      } catch (e) { /* ignore */ }
    }

    // 保存大厅初始状态（_doEnterGame 会覆盖 _savedCamera/_savedSceneObjects，
    // 必须在调用前后保存并恢复，否则后续 exitGameMode 无法正确恢复大厅场景）
    const savedCamera = this._savedCamera;
    const savedSceneObjects = this._savedSceneObjects;

    // 以相同gameKey重新进入
    const factory = GAME_REGISTRY[gameKey];
    if (factory) {
      this._doEnterGame(factory);
    }

    // 恢复大厅状态（_doEnterGame 保存的是游戏模式下的相机和场景状态）
    this._savedCamera = savedCamera;
    this._savedSceneObjects = savedSceneObjects;

    if (errors.length > 0) {
      console.warn('[GameMode] 重新开始游戏时遇到错误（不影响重启）:', errors);
    }
  }

  _showGameSelection() {
    const games = GameModeManager.getAvailableGames();
    if (games.length === 0) {
      this.App.showToast('还没有可用的游戏');
      return;
    }

    // 选择弹窗
    const overlay = document.createElement('div');
    overlay.className = 'game-select-overlay';
    overlay.innerHTML = `
      <div class="game-select-panel">
        <h3 class="game-select-title">🎮 选择小游戏</h3>
        <div class="game-select-list">
          ${games.map(g => `
            <div class="game-select-card" data-key="${g.key}">
              <div class="game-select-card-name">${g.name}</div>
              <div class="game-select-card-desc">${g.description}</div>
            </div>
          `).join('')}
        </div>
        <button class="game-select-cancel">取消</button>
      </div>
    `;
    document.body.appendChild(overlay);

    const self = this;
    overlay.querySelector('.game-select-cancel').addEventListener('click', () => {
      document.body.removeChild(overlay);
    });

    overlay.querySelectorAll('.game-select-card').forEach(card => {
      card.addEventListener('click', () => {
        document.body.removeChild(overlay);
        self.enterGameMode(card.dataset.key);
      });
    });
  }
}

export { GAME_REGISTRY };
export default GameModeManager;
