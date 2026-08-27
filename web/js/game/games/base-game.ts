/* ============================================================
 * 游戏基类 —— 所有小游戏必须实现的接口
 *
 * 设计理念：
 * - 每个游戏是一个独立的"mod"，有完整的生命周期
 * - 场景由系统自动生成（generateScene）
 * - 用户操控角色(body)，AI感知游戏状态(soul)
 * - 两者共享游戏过程和结果
 * ============================================================ */

export class BaseGame {
  constructor(app: any) {
    this.App = app;
    this.THREE = app ? app.THREE : null;
    this.name = 'base';
    this.displayName = '未命名游戏';
    this.description = '';
    this.state = 'idle'; // idle | playing | paused | completed | failed
    this.sceneObjects = [];   // 游戏专属3D对象（退出时清理）
    this.gameData = {};       // 游戏特定数据
    this.startTime = 0;
    this.elapsedTime = 0;
    this.score = 0;
    this.events = [];         // 游戏事件队列（用于发送给AI）
    this._lastEventIndex = 0;
    this.userControlling = false; // 用户是否正在操控

    // 通用 UI 提示（各游戏可覆盖）
    this.uiHint = '';

    // 兼容性：默认不限制活动边界（0 表示无边界）
    this.boundarySize = 0;

    // 默认移动速度（最快速度 7.0 的一半）
    this.moveSpeed = 3.5;

    // 游戏开始时摄像机距角色距离（米）—— 摄像机位于角色正后方并对准角色
    this.initialCameraRadius = 2.0;
  }

  // ==================== 生命周期 ====================

  /** 场景自动生成 —— 子类必须实现 */
  generateScene() {
    throw new Error('子类必须实现 generateScene()');
  }

  /** 游戏开始 */
  onStart() {
    this.state = 'playing';
    this.startTime = performance.now();
    this.elapsedTime = 0;
    this.score = 0;
    this.events = [];
    this._lastEventIndex = 0;
    this.userControlling = true;
    this._pushEvent('game_start', { game: this.name });
  }

  /** 每帧更新 —— 子类必须实现 */
  update(dt) {
    if (this.state !== 'playing') return;
    this.elapsedTime += dt;
  }

  /** 处理用户输入 */
  onUserInput(type, data) {
    // 子类可重写
  }

  /** 游戏暂停 */
  onPause() {
    this.state = 'paused';
    this._pushEvent('game_paused', {});
  }

  /** 游戏恢复 */
  onResume() {
    this.state = 'playing';
    this._pushEvent('game_resumed', {});
  }

  /** 游戏完成 */
  onComplete(result = {}) {
    this.state = 'completed';
    this._pushEvent('game_completed', {
      score: this.score,
      elapsed: this.elapsedTime,
      ...result
    });
  }

  /** 游戏失败 */
  onFail(reason = '') {
    this.state = 'failed';
    this._pushEvent('game_failed', {
      score: this.score,
      reason
    });
  }

  /** 清理资源 */
  cleanup() {
    const scene = this.App ? this.App.scene : null;
    for (const obj of this.sceneObjects) {
      if (obj.parent) obj.parent.remove(obj);
      this._disposeObject(obj);
    }
    this.sceneObjects = [];
    this.gameData = {};
    this.events = [];
    this.state = 'idle';
  }

  // ==================== 游戏状态快照（发送给AI感知） ====================

  /** 获取当前游戏状态快照 */
  getStateSnapshot() {
    return {
      game_type: this.name,
      game_name: this.displayName,
      state: this.state,
      score: this.score,
      elapsed_sec: Math.floor(this.elapsedTime),
      player_position: this._getPlayerPosition(),
      extra: this.getExtraState(),
    };
  }

  /** 获取本轮新产生的事件（消费后清空） */
  consumeNewEvents() {
    const newEvents = this.events.slice(this._lastEventIndex);
    this._lastEventIndex = this.events.length;
    return newEvents;
  }

  /** 获取额外状态 —— 子类可重写 */
  getExtraState() {
    return {};
  }

  /** 获取 AI 感知数据（富快照） —— 子类必须重写
   *
   *  返回 AI 共玩者所需的完整游戏世界视图。
   *  玩家能看到什么，AI 就能感知到什么。
   *
   *  数据结构:
   *  {
   *    game_type, game_name, state, score, elapsed_sec,
   *    player: { x, y, z, facing, speed },
   *    map: { type, rows, cols, cell_size, cells, ... },
   *    objects: { collectible: [...], treasure: [...], clue: [...], ... },
   *    nearby: [ { type, id, x, z, distance, direction, ... }, ... ],
   *    progress: { collected, total_collectibles, treasure_found, ... },
   *    recent_events: [ { type, data, importance }, ... ]
   *  }
   */
  getPerceptionData() {
    const pos = this._getPlayerPosition();
    return {
      game_type: this.name,
      game_name: this.displayName,
      state: this.state,
      score: this.score,
      elapsed_sec: Math.floor(this.elapsedTime),
      player: {
        x: pos ? pos.x : 0, y: pos ? pos.y : 0, z: pos ? pos.z : 0,
        facing: this._getPlayerFacing(),
        speed: this._getPlayerSpeed(),
      },
      map: this._getMapData(),
      objects: this._getObjectsData(),
      nearby: this._getNearbyObjects(),
      progress: this.getExtraState(),
      recent_events: this._getRecentEvents(),
    };
  }

  // ---- 子类可重写的感知数据方法 ----

  _getMapData() { return null; }
  _getObjectsData() { return {}; }
  _getNearbyObjects() { return []; }
  _getRecentEvents(max = 10) {
    return this.events.slice(-max).map(e => ({
      type: e.type, data: e.data, time: e.time,
      importance: this._eventImportance(e.type),
    }));
  }
  _eventImportance(type) {
    const map = { game_start: 2, game_completed: 3, game_failed: 3,
                  treasure_found: 3, item_collected: 2, clue_discovered: 2,
                  level_up: 2, enemy_defeated: 2, player_hurt: 2,
                  quiz_question: 2, quiz_correct: 2, quiz_wrong: 2,
                  treasure_unlocked: 2, treasure_locked: 2 };
    return map[type] || 1;
  }
  _getPlayerFacing() {
    // 子类可重写：返回角色朝向角 (rad)
    const avatar = this.App ? this.App.currentAvatar : null;
    if (!avatar) return 0;
    return this.App.smoothRotY || 0;
  }
  _getPlayerSpeed() {
    // 子类可重写：返回当前移动速度 (m/s)
    return 0;
  }

  // ==================== 工具方法 ====================

  /** 记录游戏事件 */
  _pushEvent(type, data) {
    this.events.push({ type, data, time: this.elapsedTime });
  }

  /** 获取玩家（角色）位置 */
  _getPlayerPosition() {
    const avatar = this.App ? this.App.currentAvatar : null;
    if (!avatar) return null;
    return { x: +avatar.position.x.toFixed(2), y: +avatar.position.y.toFixed(2), z: +avatar.position.z.toFixed(2) };
  }

  /** 递归销毁3D对象 */
  _disposeObject(obj) {
    if (!obj) return;
    obj.traverse(child => {
      if (child.geometry && child.geometry !== obj.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach(m => m.dispose());
        } else {
          child.material.dispose();
        }
      }
    });
  }

  /** 将对象加入scene并追踪 */
  addToScene(obj) {
    this.App.scene.add(obj);
    this.sceneObjects.push(obj);
    return obj;
  }

  /** 发送AI动作提示（游戏中的AI反应） */
  sendAIAction(text, userDriven) {
    if (this.App.sendAIAction) {
      this.App.sendAIAction(text, userDriven);
    }
  }

  // ==================== 引擎兼容钩子（子类可选重写） ====================

  /**
   * 碰撞检测钩子 —— 游戏管理器在移动前调用
   * @param {number} x - 目标 X 坐标
   * @param {number} z - 目标 Z 坐标
   * @returns {boolean} 返回 true 表示目标位置被阻挡
   */
  checkCollision(x, z) {
    return false;
  }

  /**
   * 设置玩家移动速度 —— 由 GameModeManager 每帧调用
   * @param {number} speed - 当前移动速度 (m/s)
   */
  setPlayerSpeed(speed) {}

  /**
   * 场景特效更新钩子 —— 由 GameModeManager 每帧调用
   * @param {number} t - 累计时间 (s)
   */
  updateSceneEffects(t) {}

  // ==================== RL 环境契约（可选实现，默认无 RL） ====================
  //
  // 统一 RL 适配框架的核心接口（对应方案报告 P0-1）：
  // 任何游戏只要实现下面 5 个方法，即可被 UnifiedRLAgent 统一训练，
  // 无需在游戏内部私建编码器/动作表/奖励表。
  // 默认实现均返回"无 RL"语义，不破坏现有游戏。

  /** RL 环境规格版本号（观察/动作规格变更时递增，用于经验与权重兼容校验） */
  rlSpecVersion() { return null; }

  /**
   * 声明观察空间规格（声明式，驱动通用归一化编码器）
   * @returns {Array|null} [{ name, kind:'scalar'|'grid', shape, scale, offset }]；
   *                       返回 null 表示该游戏不支持 RL
   */
  getObservationSpec() { return null; }

  /**
   * 声明动作空间（动作表，替代各游戏 switch-case 硬编码分发）
   * @returns {Array} [{ id, name, semantics:'semantic'|'primitive', executable }]
   */
  getActionSpec() { return []; }

  /**
   * 获取当前观察向量（必须与 getObservationSpec() 对齐）
   * @returns {Float64Array|Array|null} 归一化前的原始观察值序列
   */
  getObservation() { return null; }

  /**
   * 执行一个 RL 动作（由游戏自身落到物理/状态机，可含安全护栏）
   * @param {number} actionId - getActionSpec() 中的动作索引
   * @returns {boolean} 是否成功执行
   */
  applyAction(actionId) { return false; }

  /**
   * 推进一个 RL 决策步（配合 applyAction 使用）
   * @param {number} actionId - 要执行的动作索引
   * @returns {Object|null} { obs, reward, done, info }；不实现 RL 返回 null
   */
  rlStep(actionId) { return null; }

  /**
   * 获取当前可用动作索引（掩码；null 表示全部可用）
   * @returns {Array<number>|null}
   */
  getValidActions() { return null; }

  /** 当前 RL 回合是否结束 */
  rlDone() { return false; }

  /** 重置 RL 回合（新对局开始时调用，返回初始观察） */
  rlReset() { return null; }

  /**
   * 获取 RL 训练超参（默认由 games-config 注册表提供，游戏可覆盖）
   * @returns {Object|null}
   */
  getRLHyperparams() { return null; }

  declare App: any;
  declare THREE: any;
  declare _lastEventIndex: any;
  declare boundarySize: any;
  declare description: any;
  declare displayName: any;
  declare elapsedTime: any;
  declare events: any;
  declare gameData: any;
  declare initialCameraRadius: any;
  declare moveSpeed: any;
  declare name: any;
  declare sceneObjects: any;
  declare score: any;
  declare startTime: any;
  declare state: any;
  declare uiHint: any;
  declare userControlling: any;
}

export default BaseGame;