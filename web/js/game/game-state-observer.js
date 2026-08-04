/* ============================================================
 * 游戏状态感知器 —— 将游戏状态实时发送给AI
 *
 * AI通过这个模块"感受"游戏：
 * - 每N秒发送一次完整状态快照（包含地图、对象、视野内物品、玩家状态）
 * - 关键事件立即推送
 * - AI可以基于状态做出反应（庆祝、鼓励、提示、导航等）
 *
 * 核心理念：玩家能看到什么，AI 就能感知到什么
 * ============================================================ */

export class GameStateObserver {
  constructor(app) {
    this.App = app;
    this.activeGame = null;              // 当前活跃的游戏实例
    this.snapshotInterval = 10000;       // 快照间隔(ms) - 每10秒发送完整状态（提升AI感知响应速度）
    this._lastSnapshotTime = 0;
    this._processedEventCount = 0;
    this._sentGameContext = false;       // 是否已发送游戏上下文给AI

    // 非游戏模式快照
    this._lobbyEnabled = false;          // 是否启用大厅模式快照
    this._lobbyInterval = 30000;         // 大厅快照间隔 (ms)（低频，配合服务端主动说话闸门）
    this._lastLobbySnapshotTime = 0;

    // 用户（摄像机）位置追踪：用于计算摄像机的移动速度
    this._lastUserPos = null;            // { x, z, t }
  }

  /**
   * 采集用户（摄像机）的位置/朝向/速度。
   *
   * 摄像机即用户在场景中的"第一人称位置"：FPV 模式下用户可脱离角色自由走动，
   * 普通模式下摄像机跟随并环绕角色。让 AI 掌握用户的实际空间位置，
   * 才能真正做到"知道你在哪、离你多远、看着哪里"。
   *
   * 朝向统一为 body 约定（forward = (sinθ, 0, cosθ)，与后端 ai_facing 一致）：
   * - FPV：视线方向 = (-sinY, 0, -cosY) → θ = atan2(-sinY, -cosY)
   * - 普通：摄像机看向角色 → θ = atan2(角色x-相机x, 角色z-相机z)
   */
  _getUserState() {
    const cam = this.App ? this.App.camera : null;
    if (!cam) return null;

    let facing = 0;
    if (this.App.fpvMode) {
      const Y = this.App.fpvYaw || 0;
      facing = Math.atan2(-Math.sin(Y), -Math.cos(Y));
    } else if (this.App.currentAvatar) {
      facing = Math.atan2(
        this.App.currentAvatar.position.x - cam.position.x,
        this.App.currentAvatar.position.z - cam.position.z
      );
    } else {
      facing = cam.rotation.y || 0;
    }

    // 摄像机移动速度：最近两次快照之间的平均速度
    const now = performance.now();
    let speed = 0;
    if (this._lastUserPos) {
      const dt = (now - this._lastUserPos.t) / 1000;
      if (dt > 0.05) {
        speed = Math.hypot(
          cam.position.x - this._lastUserPos.x,
          cam.position.z - this._lastUserPos.z
        ) / dt;
      }
    }
    this._lastUserPos = { x: cam.position.x, z: cam.position.z, t: now };

    return {
      x: +cam.position.x.toFixed(2),
      y: +cam.position.y.toFixed(2),
      z: +cam.position.z.toFixed(2),
      facing: +facing.toFixed(3),
      speed: +speed.toFixed(2),
    };
  }

  /**
   * 绑定游戏实例
   */
  bind(game) {
    this.activeGame = game;
    this._lastSnapshotTime = -this.snapshotInterval; // 强制第一帧立即发送快照
    this._processedEventCount = 0;
    this._sentGameContext = false;

    // 尝试立即发送，若失败则在 update 中持续重试
    this._sendGameContext();
  }

  /**
   * 每帧调用，自动处理快照和事件推送
   */
  update() {
    if (!this.activeGame || this.activeGame.state !== 'playing') return;

    // 若首次 game_context 未发送成功（如 WebSocket 尚未就绪），每帧重试
    if (!this._sentGameContext) {
      this._sendGameContext();
    }

    const now = performance.now();

    // 定期发送完整状态快照（包含 AI 能"看到"的一切）
    if (now - this._lastSnapshotTime > this.snapshotInterval) {
      this._lastSnapshotTime = now;
      this._sendRichSnapshot();
    }

    // 消费新事件
    this._processNewEvents();
  }

  /** 手动推送给AI（例如游戏完成/失败） */
  notifyAI(eventType, data = {}) {
    if (!this.App.ws || this.App.ws.readyState !== WebSocket.OPEN) return;

    const snap = this.activeGame ? this.activeGame.getStateSnapshot() : {};
    this.App.ws.send(JSON.stringify({
      type: 'game_state',
      event: eventType,
      data: { ...snap, ...data, user: this._getUserState() }
    }));
  }

  /** 解绑并清理 */
  unbind() {
    this.activeGame = null;
    this._sentGameContext = false;
  }

  // ==================== 内部方法 ====================

  _sendGameContext() {
    if (!this.activeGame || !this.App.ws || this.App.ws.readyState !== WebSocket.OPEN) return;
    if (this._sentGameContext) return;

    this._sentGameContext = true;

    // 使用富感知数据作为初始游戏上下文
    const perception = this.activeGame.getPerceptionData();
    if (perception) {
      this.App.ws.send(JSON.stringify({
        type: 'game_context',
        data: {
          ...perception,
          user: this._getUserState(),
          // 额外描述 AI 角色
          description: `用户和你正在玩「${perception.game_name}」。这是一个 AI 游戏——你能看到游戏中的一切，和用户共享同一个屏幕。请像一起玩游戏的朋友那样自然地反应。`
        }
      }));
    }
  }

  _sendRichSnapshot() {
    if (!this.activeGame || !this.App.ws || this.App.ws.readyState !== WebSocket.OPEN) return;

    const perception = this.activeGame.getPerceptionData();
    if (!perception) return;

    // 发送完整感知快照（新协议）
    this.App.ws.send(JSON.stringify({
      type: 'game_update',
      data: { ...perception, user: this._getUserState() },
    }));
  }

  _processNewEvents() {
    if (!this.activeGame) return;
    const newEvents = this.activeGame.consumeNewEvents();
    if (newEvents.length === 0) return;

    for (const evt of newEvents) {
      // 关键事件推送给AI
      switch (evt.type) {
        case 'game_completed':
          this._sendGameResult('completed', evt.data);
          break;
        case 'game_failed':
          this._sendGameResult('failed', evt.data);
          break;
        case 'treasure_found':
        case 'item_collected':
        case 'clue_discovered':
        case 'level_up':
        case 'resource_collected':
        case 'biome_changed':
        case 'chunk_changed':
        case 'quiz_question':
        case 'quiz_correct':
        case 'quiz_wrong':
        case 'treasure_unlocked':
        case 'treasure_locked':
          this._sendGameEvent(evt.type, evt.data);
          break;
        default:
          // 兼容未来游戏：重要性 >= 2 的自定义事件也自动转发
          if (this.activeGame && this.activeGame._eventImportance) {
            const importance = this.activeGame._eventImportance(evt.type);
            if (importance >= 2) {
              this._sendGameEvent(evt.type, evt.data);
            }
          }
          break;
      }
    }
  }

  _sendGameEvent(type, data) {
    if (!this.App.ws || this.App.ws.readyState !== WebSocket.OPEN) return;
    this.App.ws.send(JSON.stringify({
      type: 'game_event',
      event: type,
      data: data
    }));
  }

  _sendGameResult(result, data) {
    if (!this.App.ws || this.App.ws.readyState !== WebSocket.OPEN) return;

    // 合并 getExtraState() 数据（收集数、背包、资源等），让AI获得完整游戏结果
    const extraState = this.activeGame && this.activeGame.getExtraState
      ? this.activeGame.getExtraState()
      : {};
    const fullData = { ...data, ...extraState };

    // 生成包含详细结果的自然语言描述
    const elapsed = Math.floor(fullData.elapsed || 0);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    const timeStr = minutes > 0 ? `${minutes}分${seconds}秒` : `${seconds}秒`;

    let text = '';
    if (result === 'completed') {
      const parts = [`游戏完成！得分：${fullData.score || 0}，用时：${timeStr}`];
      // 收集类数据
      if (fullData.collected !== undefined && fullData.total_collectibles !== undefined) {
        parts.push(`收集了 ${fullData.collected}/${fullData.total_collectibles} 个物品`);
      }
      if (fullData.treasure_found) {
        parts.push('找到了宝藏');
      }
      if (fullData.resources_collected !== undefined) {
        parts.push(`采集了 ${fullData.resources_collected} 个资源`);
      }
      // 背包物品
      if (fullData.inventory && fullData.inventory.length > 0) {
        parts.push(`背包里有：${fullData.inventory.join('、')}`);
      }
      // 生物群系
      if (fullData.biome_name) {
        parts.push(`当前在${fullData.biome_name}区域`);
      }
      text = parts.join('，') + '。';
    } else {
      text = `游戏结束。${fullData.reason || ''}。得分：${fullData.score || 0}`;
    }

    this.App.ws.send(JSON.stringify({
      type: 'game_result',
      result: result,
      data: fullData,
      text: text
    }));
  }

  // ==================== 非游戏模式（大厅）快照 ====================

  /**
   * 启用在非游戏模式下发送环境快照
   */
  enableLobbySnapshots() {
    this._lobbyEnabled = true;
    this._lastLobbySnapshotTime = 0; // 立即发送第一次
  }

  /**
   * 禁用大厅快照
   */
  disableLobbySnapshots() {
    this._lobbyEnabled = false;
  }

  /**
   * 应用 RL 决策的大厅快照间隔（秒）—— 由后端 SnapshotIntervalController 学习得出。
   * 带上下界保护：10 秒 ~ 5 分钟，防止异常值导致过密或过稀。
   * @param {number} sec
   */
  applyRlLobbyInterval(sec) {
    if (typeof sec !== 'number' || !isFinite(sec) || sec <= 0) return;
    const clamped = Math.min(300, Math.max(10, sec));
    this._lobbyInterval = clamped * 1000;
    // 触发后立即重置计时，让新间隔立刻生效
    this._lastLobbySnapshotTime = performance.now();
  }

  /**
   * 应用 RL 决策的游戏快照间隔（秒）—— 由后端 SnapshotIntervalController 学习得出。
   * 游戏档位：5/10/30/60s。带上下界保护：3 秒 ~ 2 分钟。
   * @param {number} sec
   */
  applyRlGameInterval(sec) {
    if (typeof sec !== 'number' || !isFinite(sec) || sec <= 0) return;
    const clamped = Math.min(120, Math.max(3, sec));
    this.snapshotInterval = clamped * 1000;
    // 触发后立即重置计时，让新间隔立刻生效
    this._lastSnapshotTime = performance.now();
  }

  /**
   * 在非游戏模式下每帧调用
   */
  updateLobby() {
    if (!this._lobbyEnabled) return;
    if (!this.App.ws || this.App.ws.readyState !== WebSocket.OPEN) return;

    const now = performance.now();
    if (now - this._lastLobbySnapshotTime > this._lobbyInterval) {
      this._lastLobbySnapshotTime = now;
      this._sendLobbySnapshot();
    }
  }

  _sendLobbySnapshot() {
    if (!this.App.ws || this.App.ws.readyState !== WebSocket.OPEN) return;

    const avatar = this.App.currentAvatar;
    const data = {
      player: {
        x: avatar ? +avatar.position.x.toFixed(2) : 0,
        y: avatar ? +avatar.position.y.toFixed(2) : 0,
        z: avatar ? +avatar.position.z.toFixed(2) : 0,
        facing: this.App.smoothRotY || 0,
        speed: 0,
      },
      user: this._getUserState(),   // 用户（摄像机）位置 —— AI 对用户实际位置的参考
      user_engaged: false,
      scene_type: "lobby",
      nearby: [],  // 大厅模式没有结构化 nearby 数据
      objects: {},
    };

    // 如果有背景模型信息，包含进去
    if (this.App.backgroundGroup) {
      data.objects.background = [{
        id: 'lobby_background',
        x: this.App.backgroundGroup.position.x,
        z: this.App.backgroundGroup.position.z,
      }];
    }

    this.App.ws.send(JSON.stringify({
      type: 'environment_snapshot',
      data: data,
    }));
  }
}

export default GameStateObserver;
