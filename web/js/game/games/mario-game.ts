/* ============================================================
 * 马里奥无限跑酷 (Mario Endless Runner) —— DQN 泛化能力验证
 *
 * 玩法：
 * - 横版 2.5D 无限跑酷：向右奔跑、跳跃、踩敌人、吃金币
 * - 地形随机程序化生成：每局地面段长度、坑宽、平台位置都不同
 * - 玩家操控角色(body)：A/D 左右移动，空格跳跃（含二段跳）
 * - 玩家空闲时（无操作 4 秒），DQN 强化学习系统接管角色自主闯关
 * - RL 智能体必须在随机地形中学会通用跳跃策略，而非记忆固定路线
 *
 * 决策逻辑：Double Dueling DQN + Experience Replay（unified-rl-agent.js + nn-advanced.js）
 * ============================================================ */

import { BaseGame } from "./base-game.ts";
import { RLAgentManager } from "./../rl/rl-agent-manager.ts";

// ==================== 动作空间（8 动作：含二段跳） ====================
const MARIO_ACTIONS = [
  'idle',            // 0
  'move_left',       // 1
  'move_right',      // 2
  'jump',            // 3
  'jump_left',       // 4
  'jump_right',      // 5
  'double_jump',     // 6  空中二段跳（原地）
  'double_jump_right', // 7  空中二段跳（向右）
];

// ==================== 物理常量 ====================
const GRAVITY = 34;
const JUMP_VELOCITY = 15.5;
const DOUBLE_JUMP_VELOCITY = 12.0;
const MAX_JUMPS = 2;
const STOMP_BOUNCE = 9;
const MOVE_SPEED = 5.5;
const MAX_FALL_SPEED = 32;
const PIT_DEATH_Y = -7;
const PLAYER_HALF_WIDTH = 0.32;
const PLAYER_HEIGHT = 1.6;
const PLAY_Z = 0;

// ==================== RL 常量 ====================
// 决策节奏由 HumanInterfaceController 接管（P2-3）：人化反应延迟 + ≤20Hz 硬约束
const DECISION_INTERVAL = 0.18;  // 兼容参考值（已不再驱动决策循环）
const IDLE_TAKEOVER_DELAY = 4.0;
const RL_REWARD = {
  // ===== 谨慎策略：大幅强化死亡惩罚 =====
  coin:            +6,      // 金币奖励（降低，不让贪婪压过安全）
  coin_near:       +0.08,   // 靠近金币正反馈（降低）
  stomp:           +15,     // 踩敌奖励（降低，不鼓励冒险攻击）
  stomp_chain:     +3,      // 连续踩敌额外奖励
  jump_over_enemy: +4,      // 成功跳过敌人（不碰撞）
  hurt:            -80,     // 被敌人碰到：大幅惩罚（-45→-80）
  enemy_near:      -0.35,   // 敌人附近未行动的警惕惩罚（-0.15→-0.35）
  pit_death:       -150,    // 掉坑死亡（-80→-150）—— 最严厉惩罚
  life_lost:       -80,     // 失去生命（-50→-80）
  win:             +80,     // 里程碑奖励（+100→+80）
  lose:            -180,    // 局结束失败（-120→-180）
  progress:        +0.1,    // 前进正反馈（+0.3→+0.1）—— 降低冒进激励
  regress:         -0.08,   // 后退惩罚（-0.05→-0.08）
  step:            +0.02,   // 存活基础奖励（+0.01→+0.02）—— 鼓励活着
  milestone:       +6,      // 里程碑奖励
  // ===== 谨慎策略：强化安全行为奖励 =====
  jump_pit:        +0.3,    // 坑边起跳奖励（+0.5→+0.3）
  jump_pit_edge:   +1.0,    // 从坑边缘起跳额外奖励（+1.5→+1.0）
  pit_edge:        -2.5,    // 坑边危险惩罚（-1.0→-2.5）—— 大幅加强
  safe_land:       +3.0,    // 安全着陆奖励（+1.5→+3.0）—— 大幅加强
  grounded_safe:   +0.04,   // 【新增】着地安全奖励：每步在安全地面+0.04
  airborne_pit:    -0.5,    // 【新增】空中在坑上方惩罚：鼓励尽快落地
  enemy_caution:   -0.2,    // 【新增】敌人2m内未跳跃的谨慎惩罚
  pit_move_away:   +0.15,   // 【新增】远离坑边缘奖励
  reckless_jump:   -0.3,    // 【新增】坑边2m内无意义跳跃惩罚
  // ===== 停留惩罚：允许更长的观察时间 =====
  idle_penalty:    -0.08,   // 原地停留扣分（-0.15→-0.08）—— 允许谨慎停顿
  idle_threshold:  5.0,     // 停留超过5秒才开始扣分（3.0→5.0）
  high_jump:       -0.5,    // 高台无意义跳跃惩罚（保留作为安全提示）
  early_jump:      -0.4,    // 离坑太远就起跳惩罚（保留）
  enemy_land_near: -0.3,    // 跳跃落点在敌人附近惩罚
};
const TRAIN_SPEEDS = [1, 2, 4, 10, 30, 60];

// ==================== RL 屏幕视觉传感器常量 ====================
const VISION_SCAN_MIN   = -2;    // 扫描起点（玩家后方2m）
const VISION_SCAN_MAX   = 12;    // 扫描终点（玩家前方12m）
const VISION_SCAN_STEP  = 1;     // 采样间隔1m
const VISION_POINTS     = 15;    // 采样点数: -2,-1,0,1,...,12
const VISION_CHANNELS   = 6;     // 每点6通道: 地形高度/坑/敌人/金币/石块/石块高度
const RL_PLAYER_STATE_SIZE = 22; // 玩家全局状态维度
const RL_STATE_SIZE     = VISION_POINTS * VISION_CHANNELS + RL_PLAYER_STATE_SIZE; // 112
const REFERENCE_DIST = 200;  // 状态归一化参考距离（非终点，仅用于状态编码）
const SPAWN_X = 2;
const START_LIVES = 1;       // 无限跑酷模式：1 条命，死亡即结束
const GENERATE_AHEAD = 40;   // 玩家前方多远生成新地形
const RECYCLE_BEHIND = 25;   // 玩家后方多远回收旧地形

// ==================== 程序化地形参数 ====================
const TERRAIN = {
  minGroundLen: 6,       // 地面段最短长度
  maxGroundLen: 16,      // 地面段最长长度
  minPitWidth: 2.0,      // 坑最窄
  maxPitWidth: 4.5,      // 坑最宽（一段跳极限约5m，需二段跳）
  platformChance: 0.35,  // 生成浮动平台概率
  blockChance: 0.25,     // 生成实心方块概率
  enemyChance: 0.45,     // 生成敌人概率
  coinChance: 0.6,       // 生成金币概率
  minEnemySpeed: 1.0,
  maxEnemySpeed: 1.8,
};

// ==================== 主类 ====================
export class MarioGame extends BaseGame {
  constructor(app) {
    super(app);
    this.name = 'mario';
    this.displayName = '马里奥无限跑酷';
    this.description = '随机地形无限跑酷：DQN 必须学会通用跳跃策略才能存活。玩家放手时 AI 自主闯关。';
    this.moveSpeed = MOVE_SPEED;
    this.initialCameraRadius = 9;
    this.initialCameraHeight = 1.4;
    this.uiHint = 'A/D 左右移动 · 空格跳跃 · 停手 4 秒后 DQN AI 接管自主跑酷';

    // 运行时
    this._vy = 0;
    this._isGrounded = false;
    this._jumpsRemaining = MAX_JUMPS;
    this._prevFeetY = 0;
    this._invincible = 0;
    this._rlClimbing = false;  // 爬石块标志
    this._lastInputTime = performance.now() / 1000;
    this.rlDriving = false;
    this._rl = null;
    this._rlMoveDir = 0;
    this._rlWantsJump = false;
    this._rlFaceDir = 1;
    this._rlEpisodeReward = 0;
    this._rlRestartTimer = 0;
    this._rlSettingSpeed = false;
    this._rlEnded = false;
    this._trainSpeed = 1;
    this._maxX = SPAWN_X;
    this._lastCheckpointX = SPAWN_X;
    this._lastGroundSegIdx = 0;
    this._idleTimer = 0;        // 连续停留计时器
    this._lastIdleX = SPAWN_X;  // 上次记录的 X 位置（用于判断是否移动）
    this._lastStompTime = 0;    // 上次踩敌时间（连击判定）
    this._stompChain = 0;       // 当前踩敌连击数
    this._lastMilestone = 0;    // 上次里程碑距离

    // 程序化地形状态
    this._terrainSeed = 0;
    this._generatedUntil = 0;    // 已生成到的 X 坐标
    this._segmentCount = 0;       // 已生成的地面段数
    this._enemyIdCounter = 0;

    // UI
    this._rlPanel = null;
    this._rlBtn = null;
  }

  // ==================== 生命周期 ====================

  generateScene() {
    const avatar = this.App.currentAvatar;
    const THREE = this.THREE;

    this._buildEnvironment();
    this._initProceduralLevel();

    if (avatar) {
      avatar.position.set(SPAWN_X, 3, PLAY_Z);
      avatar.rotation.set(0, Math.PI / 2, 0);
      if (this.App) this.App.smoothRotY = Math.PI / 2;
    }
    if (THREE) this._dummy = new THREE.Object3D();
  }

  onStart() {
    super.onStart();
    this._vy = 0;
    this._isGrounded = false;
    this._jumpsRemaining = MAX_JUMPS;
    this._invincible = 0;
    this._maxX = SPAWN_X;
    this._lastInputTime = performance.now() / 1000;
    this.rlDriving = false;
    this._rlRestartTimer = 0;
    this._resetEpisode();
    this._ensureRLPanel();
    this.uiHint = 'A/D 左右移动 · 空格跳跃 · 停手 4 秒后 DQN AI 接管自主跑酷';
    if (this.App.sendAIAction) {
      this.App.sendAIAction('（进入无限跑酷世界！地形每次都不同，让我学会在各种地形上跳跃生存！）');
    }
  }

  update(dt) {
    super.update(dt);
    if (this.state !== 'playing') return;

    if (this.App) {
      if (this.App._aiDrivenWalk) this.App._aiDrivenWalk = false;
      if (this.App.walkPath && this.App.walkPath.length) this.App.walkPath = [];
    }

    const steps = this.rlDriving ? this._trainSpeed : 1;
    for (let i = 0; i < steps; i++) this._subUpdate(dt);

    this._lockSideCamera();
    this._updateRLPanel();
  }

  _subUpdate(dt) {
    const avatar = this.App.currentAvatar;
    if (!avatar) return;

    // 动态生成前方地形
    this._generateTerrainAhead(avatar.position.x);
    // 回收后方旧地形
    this._recycleTerrainBehind(avatar.position.x);

    // 物理
    this._updatePhysics(avatar, dt);

    // 实体
    this._updateEnemies(dt);
    this._checkCoins(avatar);
    this._checkEnemies(avatar);
    this._checkMilestone(avatar);

    if (this._invincible > 0) this._invincible = Math.max(0, this._invincible - dt);

    const now = performance.now() / 1000;
    if (!this.rlDriving && (now - this._lastInputTime) > IDLE_TAKEOVER_DELAY && this.state === 'playing') {
      this._startRLDriving();
    }

    if (this.rlDriving) {
      // P2-3 接口节奏真实化：人化反应延迟触发决策（硬约束 ≤20Hz）
      if (this._rl.interfaceController.shouldAct(performance.now())) {
        this._rlStep();
      }
      this._rlApplyContinuous(dt);

      if (this._rlRestartTimer > 0) {
        this._rlRestartTimer -= dt;
        if (this._rlRestartTimer <= 0) {
          this._resetEpisode();
          this.rlDriving = true;
          this._lastInputTime = performance.now() / 1000 + 9999;
        }
      }
    }
  }

  cleanup() {
    this._onExitPersist();
    this._removeRLPanel();
    super.cleanup();
  }

  // ==================== 程序化地形生成 ====================

  _initProceduralLevel() {
    const THREE = this.THREE;
    this._terrainSeed = Math.floor(Math.random() * 1000000);
    this._generatedUntil = 0;
    this._segmentCount = 0;
    this._enemyIdCounter = 0;

    const gd = {
      blocks: [],
      surfaces: [],
      coinList: [],
      enemyList: [],
      groundSegments: [],   // 程序化生成的地面段 [{start, end}]
      coinsCollected: 0,
      enemiesStomped: 0,
      questionMats: [],
      lives: 0,
      score: 0,
      // 用于动态 mesh 管理
      _meshRegistry: [],    // 所有生成的 mesh，用于回收
    };

    this.gameData = gd;
    gd.lives = START_LIVES;
    gd.score = 0;

    // 生成第一段安全地面（出生区域，无坑）
    this._generateGroundSegment(0, 12, 0);
    // 预生成前方地形
    this._generateTerrainAhead(SPAWN_X + GENERATE_AHEAD);
  }

  /** 生成一段地面 + 上面的实体 */
  _generateGroundSegment(startX, endX, pitWidthBefore) {
    const THREE = this.THREE;
    const gd = this.gameData;
    const len = endX - startX;
    const mid = (startX + endX) / 2;

    // 记录地面段
    gd.groundSegments.push({ start: startX, end: endX });
    gd.surfaces.push({ xMin: startX, xMax: endX, topY: 0 });

    if (THREE) {
      const groundMat = this._getMaterial('ground');
      const dirtMat = this._getMaterial('dirt');
      const top = new THREE.Mesh(new THREE.BoxGeometry(len, 0.4, 4), groundMat);
      top.position.set(mid, -0.2, PLAY_Z);
      this.addToScene(top);
      gd._meshRegistry.push(top);
      const dirt = new THREE.Mesh(new THREE.BoxGeometry(len, 2.0, 4), dirtMat);
      dirt.position.set(mid, -1.4, PLAY_Z);
      this.addToScene(dirt);
      gd._meshRegistry.push(dirt);
    }

    this._segmentCount++;

    // 随机生成浮动平台
    if (THREE && Math.random() < TERRAIN.platformChance && len > 5) {
      const platLen = 3 + Math.random() * 2;
      const platStart = startX + 1 + Math.random() * (len - platLen - 2);
      const platY = 2.8 + Math.random() * 0.8;
      this._generatePlatform(platStart, platStart + platLen, platY);
    }

    // 随机生成实心方块
    if (THREE && Math.random() < TERRAIN.blockChance && len > 4) {
      const bx = startX + 2 + Math.random() * (len - 4);
      const isQ = Math.random() < 0.5;
      this._generateBlock(bx, 3, isQ);
    }

    // 随机生成金币
    if (THREE && Math.random() < TERRAIN.coinChance) {
      const nCoins = 1 + Math.floor(Math.random() * 3);
      for (let i = 0; i < nCoins; i++) {
        const cx = startX + 1 + Math.random() * (len - 2);
        const cy = 1.2 + Math.random() * 1.5;
        this._generateCoin(cx, cy);
      }
    }

    // 随机生成敌人（不在出生区域）
    if (THREE && this._segmentCount > 1 && Math.random() < TERRAIN.enemyChance) {
      const ex = startX + 2 + Math.random() * (len - 4);
      const speed = TERRAIN.minEnemySpeed + Math.random() * (TERRAIN.maxEnemySpeed - TERRAIN.minEnemySpeed);
      const patrolRange = Math.min(4, len / 2);
      this._generateEnemy(ex, ex - patrolRange, ex + patrolRange, speed);
    }
  }

  _generatePlatform(x1, x2, y) {
    const THREE = this.THREE;
    const gd = this.gameData;
    const len = x2 - x1;
    const mid = (x1 + x2) / 2;
    gd.surfaces.push({ xMin: x1, xMax: x2, topY: y });
    if (THREE) {
      const platMat = this._getMaterial('platform');
      const m = new THREE.Mesh(new THREE.BoxGeometry(len, 0.3, 2.4), platMat);
      m.position.set(mid, y, PLAY_Z);
      this.addToScene(m);
      gd._meshRegistry.push(m);
    }
  }

  _generateBlock(cx, cy, isQuestion) {
    const THREE = this.THREE;
    const gd = this.gameData;
    const w = 1, h = 1;
    gd.surfaces.push({ xMin: cx - w / 2, xMax: cx + w / 2, topY: cy + h });
    if (THREE) {
      const mat = isQuestion
        ? new THREE.MeshStandardMaterial({ color: 0xf2c200, roughness: 0.5, emissive: 0xf2c200, emissiveIntensity: 0.35 })
        : this._getMaterial('brick');
      if (isQuestion) gd.questionMats.push(mat);
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, 1.2), mat);
      m.position.set(cx, cy + h / 2, PLAY_Z);
      this.addToScene(m);
      gd._meshRegistry.push(m);
      gd.blocks.push({ cx, cy, w, h, type: isQuestion ? 'question' : 'brick', mesh: m, mat, _usedBrickMat: isQuestion ? this._getMaterial('brick') : null });
    } else {
      gd.blocks.push({ cx, cy, w, h, type: isQuestion ? 'question' : 'brick' });
    }
  }

  _generateCoin(x, y) {
    const THREE = this.THREE;
    const gd = this.gameData;
    if (THREE) {
      const coinMat = this._getMaterial('coin');
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.08, 18), coinMat);
      m.rotation.x = Math.PI / 2;
      m.position.set(x, y, PLAY_Z);
      this.addToScene(m);
      gd._meshRegistry.push(m);
      gd.coinList.push({ x, y, collected: false, mesh: m });
    } else {
      gd.coinList.push({ x, y, collected: false, mesh: null });
    }
  }

  _generateEnemy(x, minX, maxX, speed) {
    const THREE = this.THREE;
    const gd = this.gameData;
    const id = this._enemyIdCounter++;
    if (THREE) {
      const bodyMat = this._getMaterial('enemyBody');
      const footMat = this._getMaterial('enemyFoot');
      const eyeMat = this._getMaterial('enemyEye');
      const pupilMat = this._getMaterial('enemyPupil');
      const g = new THREE.Group();
      const body = new THREE.Mesh(new THREE.SphereGeometry(0.4, 14, 12), bodyMat);
      body.scale.set(1, 0.7, 1);
      body.position.y = 0.35;
      g.add(body);
      const foot = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.2, 0.6), footMat);
      foot.position.y = 0.1;
      g.add(foot);
      for (const sx of [-1, 1]) {
        const eye = new THREE.Mesh(new THREE.SphereGeometry(0.09, 8, 8), eyeMat);
        eye.position.set(sx * 0.14, 0.42, 0.32);
        g.add(eye);
        const pup = new THREE.Mesh(new THREE.SphereGeometry(0.045, 6, 6), pupilMat);
        pup.position.set(sx * 0.14, 0.42, 0.4);
        g.add(pup);
      }
      g.position.set(x, 0, PLAY_Z);
      this.addToScene(g);
      gd._meshRegistry.push(g);
      gd.enemyList.push({ id, x, y: 0, dir: 1, alive: true, minX, maxX, speed, mesh: g });
    } else {
      gd.enemyList.push({ id, x, y: 0, dir: 1, alive: true, minX, maxX, speed, mesh: null });
    }
  }

  /** 在玩家前方持续生成新地形 */
  _generateTerrainAhead(playerX) {
    while (this._generatedUntil < playerX + GENERATE_AHEAD) {
      // 下一段地面
      const prevEnd = this._generatedUntil;
      // 坑宽（第一段之后才有坑）
      let pitWidth = 0;
      if (this._segmentCount > 0) {
        pitWidth = TERRAIN.minPitWidth + Math.random() * (TERRAIN.maxPitWidth - TERRAIN.minPitWidth);
      }
      const groundStart = prevEnd + pitWidth;
      const groundLen = TERRAIN.minGroundLen + Math.random() * (TERRAIN.maxGroundLen - TERRAIN.minGroundLen);
      const groundEnd = groundStart + groundLen;

      this._generateGroundSegment(groundStart, groundEnd, pitWidth);
      this._generatedUntil = groundEnd;
    }
  }

  /** 回收玩家后方的旧地形 mesh */
  _recycleTerrainBehind(playerX) {
    const THREE = this.THREE;
    if (!THREE) return;
    const gd = this.gameData;
    const recycleX = playerX - RECYCLE_BEHIND;

    // 回收 surfaces（保留逻辑数据用于碰撞，只回收 mesh）
    // 实际上 surfaces 是纯数据，不占显存。主要回收 mesh。
    // 但 surfaces 数组会无限增长，需要定期裁剪
    if (gd.surfaces.length > 60) {
      // 只保留玩家附近的 surfaces
      gd.surfaces = gd.surfaces.filter(s => s.xMax > recycleX - 5);
    }
    if (gd.groundSegments.length > 40) {
      gd.groundSegments = gd.groundSegments.filter(s => s.end > recycleX - 5);
    }

    // 回收 coinList 中已经过去的金币（含 mesh）
    for (let i = gd.coinList.length - 1; i >= 0; i--) {
      const c = gd.coinList[i];
      if (c.x < recycleX) {
        if (c.mesh) this._safeRemoveMesh(c.mesh);
        gd.coinList.splice(i, 1);
      }
    }

    // 回收 enemyList
    for (let i = gd.enemyList.length - 1; i >= 0; i--) {
      const e = gd.enemyList[i];
      if (e.x < recycleX || !e.alive) {
        if (e.mesh) this._safeRemoveMesh(e.mesh);
        gd.enemyList.splice(i, 1);
      }
    }

    // 回收 blocks
    for (let i = gd.blocks.length - 1; i >= 0; i--) {
      const b = gd.blocks[i];
      if (b.cx < recycleX) {
        if (b.mesh) this._safeRemoveMesh(b.mesh);
        gd.blocks.splice(i, 1);
      }
    }

    // 回收 _meshRegistry 中的地面/平台 mesh
    for (let i = gd._meshRegistry.length - 1; i >= 0; i--) {
      const m = gd._meshRegistry[i];
      if (m.position.x < recycleX - 5) {
        this._safeRemoveMesh(m);
        gd._meshRegistry.splice(i, 1);
      }
    }

    // 定期清理 sceneObjects 中已移除的引用（每帧回收时检查）
    if (this.sceneObjects && this.sceneObjects.length > 80) {
      this.sceneObjects = this.sceneObjects.filter(obj => obj.parent !== null);
    }
  }

  _safeRemoveMesh(mesh) {
    if (!mesh) return;
    try {
      if (mesh.parent) mesh.parent.remove(mesh);
      if (mesh.geometry) mesh.geometry.dispose();
      // 材质可能是共享的，不 dispose（由 _getMaterial 管理）
    } catch (e) { /* 忽略 */ }
  }

  // ==================== 材质缓存 ====================
  _materialCache = null;

  _getMaterial(name) {
    const THREE = this.THREE;
    if (!THREE) return null;
    if (!this._materialCache) this._materialCache = {};
    if (!this._materialCache[name]) {
      const opts = {
        ground: { color: 0x6bbf5a, roughness: 0.95 },
        dirt: { color: 0x8b5a2b, roughness: 1 },
        platform: { color: 0xc77b3a, roughness: 0.8, emissive: 0x3a1a00, emissiveIntensity: 0.1 },
        brick: { color: 0xc77b3a, roughness: 0.7 },
        coin: { color: 0xffd23a, roughness: 0.2, metalness: 0.6, emissive: 0xffaa00, emissiveIntensity: 0.6 },
        enemyBody: { color: 0x8b4513, roughness: 0.8 },
        enemyFoot: { color: 0x3a1a00, roughness: 0.9 },
        enemyEye: { color: 0xffffff, emissive: 0x444444, emissiveIntensity: 0.2 },
        enemyPupil: { color: 0x000000 },
      };
      this._materialCache[name] = new THREE.MeshStandardMaterial(opts[name] || { color: 0x888888 });
    }
    return this._materialCache[name];
  }

  // ==================== 引擎兼容钩子 ====================

  checkCollision(newX, newZ, options = {}) {
    const avatar = this.App.currentAvatar;
    if (!avatar) return false;
    if (Math.abs(newZ - PLAY_Z) > 0.15) return true;
    if (newX < 0) return true;

    const feetY = avatar.position.y;
    const headY = feetY + PLAYER_HEIGHT;
    const blocks = this.gameData.blocks;
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      const bxMin = b.cx - b.w / 2, bxMax = b.cx + b.w / 2;
      if (newX + PLAYER_HALF_WIDTH * 0.85 > bxMin && newX - PLAYER_HALF_WIDTH * 0.85 < bxMax) {
        if (headY > b.cy + 0.02 && feetY < b.cy + b.h - 0.02) {
          // RL 模式：自动爬上石块（抬升 Y 到顶部，放行移动）
          if (this.rlDriving) {
            const blockTopY = b.cy + b.h;
            avatar.position.y = feetY + (blockTopY - feetY) * 0.25;
            this._rlClimbing = true;
            return false;  // 放行
          }
          return true;
        }
      }
    }
    return false;
  }

  setPlayerSpeed(speed) {
    this._lastMoveSpeed = speed;
    if (this._rlSettingSpeed) return;
    if (speed > 0.5) {
      this._lastInputTime = performance.now() / 1000;
      if (this.rlDriving) this._stopRLDriving();
    }
  }

  updateSceneEffects(t) {
    const gd = this.gameData;
    if (!gd) return;
    for (const c of gd.coinList) {
      if (c.collected) continue;
      if (c.mesh) {
        c.mesh.rotation.y = t * 2.5;
        c.mesh.position.y = c.y + Math.sin(t * 3 + c.x) * 0.12;
      }
    }
    if (gd.questionMats) {
      for (const m of gd.questionMats) m.emissiveIntensity = 0.3 + Math.sin(t * 4) * 0.15;
    }
  }

  // ==================== 物理 ====================

  _updatePhysics(avatar, dt) {
    const px = avatar.position.x;
    this._prevFeetY = avatar.position.y;

    this._vy -= GRAVITY * dt;
    if (this._vy < -MAX_FALL_SPEED) this._vy = -MAX_FALL_SPEED;
    avatar.position.y += this._vy * dt;
    const feet = avatar.position.y;

    // 头顶撞实心方块
    if (this._vy > 0) {
      const headPrev = this._prevFeetY + PLAYER_HEIGHT;
      const headNow = feet + PLAYER_HEIGHT;
      for (const b of this.gameData.blocks) {
        const bxMin = b.cx - b.w / 2, bxMax = b.cx + b.w / 2;
        if (px >= bxMin - PLAYER_HALF_WIDTH && px <= bxMax + PLAYER_HALF_WIDTH) {
          if (headPrev <= b.cy + 0.08 && headNow >= b.cy) {
            avatar.position.y = b.cy - PLAYER_HEIGHT - 0.01;
            this._vy = 0;
            this._onBlockBump(b);
            break;
          }
        }
      }
    }

    // 着地检测
    if (this._vy <= 0) {
      let landY = null;
      const surfaces = this.gameData.surfaces;
      for (let i = 0; i < surfaces.length; i++) {
        const s = surfaces[i];
        if (px < s.xMin - PLAYER_HALF_WIDTH || px > s.xMax + PLAYER_HALF_WIDTH) continue;
        if (this._prevFeetY >= s.topY - 0.12 && feet <= s.topY) {
          if (landY === null || s.topY > landY) landY = s.topY;
        }
      }
      if (landY !== null) {
        avatar.position.y = landY;
        this._vy = 0;
        if (!this._isGrounded) this._onLand(landY);
        this._isGrounded = true;
        this._updateCheckpoint(px);
      } else {
        this._isGrounded = false;
      }
    } else {
      this._isGrounded = false;
    }

    if (avatar.position.y < PIT_DEATH_Y && !this._rlEnded) {
      this._onPitDeath();
    }

    if (this.rlDriving && px > this._maxX) {
      const gain = px - this._maxX;
      this._rl.accumReward += gain * RL_REWARD.progress;
      this._maxX = px;
    } else if (this.rlDriving && px < this._maxX - 0.3) {
      this._rl.accumReward += RL_REWARD.regress * Math.min(1, (this._maxX - px) * 0.5);
    }
  }

  _onLand(landY) {
    this._jumpsRemaining = MAX_JUMPS;
    this._pushEvent('player_landed', { y: +landY.toFixed(2) });

    // 成功跳过敌人奖励：着陆时身后有活着的敌人
    if (this.rlDriving) {
      const avatar = this.App.currentAvatar;
      if (avatar) {
        const px = avatar.position.x;
        for (const e of this.gameData.enemyList) {
          if (!e.alive) continue;
          // 敌人在身后 0~3m 范围内 = 刚刚跳过了它
          const d = px - e.x;
          if (d > 0 && d < 3.0) {
            this._rlAward(RL_REWARD.jump_over_enemy);
            break;
          }
        }
      }
    }
  }

  _updateCheckpoint(px) {
    const segs = this.gameData.groundSegments;
    for (let i = 0; i < segs.length; i++) {
      const seg = segs[i];
      if (px >= seg.start && px <= seg.end) {
        this._lastCheckpointX = Math.max(seg.start + 1, px - 2);
        if (this.rlDriving && i > this._lastGroundSegIdx) {
          this._rlAward(RL_REWARD.milestone * (i - this._lastGroundSegIdx));
          this._rlAward(RL_REWARD.safe_land);
          this._lastGroundSegIdx = i;
        }
        return;
      }
    }
  }

  // ==================== 跳跃 ====================

  requestJump() {
    if (this.state !== 'playing') return false;
    this._lastInputTime = performance.now() / 1000;
    if (this.rlDriving) this._stopRLDriving();
    return this._doJump();
  }

  _doJump() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return false;
    if (this._isGrounded) {
      this._vy = JUMP_VELOCITY;
      this._isGrounded = false;
      this._jumpsRemaining = MAX_JUMPS - 1;
      avatar.position.y += 0.05;
      this._pushEvent('player_jump', { type: 'ground' });
      return true;
    }
    if (this._jumpsRemaining > 0) {
      this._vy = DOUBLE_JUMP_VELOCITY;
      this._jumpsRemaining = 0;
      this._pushEvent('player_jump', { type: 'double' });
      return true;
    }
    return false;
  }

  _predictLandingX() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return 0;
    const py = avatar.position.y;
    const vy = this._vy;
    const disc = vy * vy + 2 * GRAVITY * py;
    if (disc < 0) return avatar.position.x;
    const t = (vy + Math.sqrt(disc)) / GRAVITY;
    // 坑上方无方向时假设向右（安全方向）
    let moveDir = this._rlMoveDir;
    if (moveDir === 0 && !this._isOnGroundSegment(avatar.position.x)) {
      moveDir = 1;
    }
    return avatar.position.x + moveDir * MOVE_SPEED * t;
  }

  // ==================== 环境构建 ====================

  _buildEnvironment() {
    const THREE = this.THREE;
    const scene = this.App.scene;
    if (!THREE) return;

    scene.background = new THREE.Color(0x87ceeb);
    if (scene.fog) scene.fog = new THREE.Fog(0x9fd3f5, 30, 90);

    const hemi = new THREE.HemisphereLight(0xbfe3ff, 0x6b8e3a, 0.9);
    this.addToScene(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(20, 40, 25);
    this.addToScene(dir);

    // 远景云朵
    const cloudMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, emissive: 0x335577, emissiveIntensity: 0.05 });
    for (let i = 0; i < 8; i++) {
      const cx = -10 + i * 12;
      const cloud = new THREE.Group();
      for (let j = 0; j < 3; j++) {
        const s = new THREE.Mesh(new THREE.SphereGeometry(1.6 + (j === 1 ? 0.8 : 0), 12, 10), cloudMat);
        s.position.set(j * 1.6 - 1.6, (j === 1 ? 0.4 : 0), 0);
        cloud.add(s);
      }
      cloud.position.set(cx, 14 + (i % 4) * 2, -22 - (i % 3) * 4);
      this.addToScene(cloud);
    }

    // 深渊底面（大尺寸，覆盖无限区域）
    const abyss = new THREE.Mesh(
      new THREE.PlaneGeometry(2000, 60),
      new THREE.MeshStandardMaterial({ color: 0x1a1530, roughness: 1, emissive: 0x0a0820, emissiveIntensity: 0.2 })
    );
    abyss.rotation.x = -Math.PI / 2;
    abyss.position.set(500, PIT_DEATH_Y - 6, PLAY_Z);  // 中心偏右覆盖大范围
    this.addToScene(abyss);
  }

  // ==================== 实体更新 ====================

  _updateEnemies(dt) {
    const gd = this.gameData;
    for (const e of gd.enemyList) {
      if (!e.alive) continue;
      e.x += e.dir * e.speed * dt;
      if (e.x > e.maxX) { e.x = e.maxX; e.dir = -1; }
      else if (e.x < e.minX) { e.x = e.minX; e.dir = 1; }
      if (e.mesh) {
        e.mesh.position.x = e.x;
        e.mesh.rotation.y = e.dir > 0 ? -Math.PI / 2 : Math.PI / 2;
        e.mesh.position.y = Math.abs(Math.sin(performance.now() / 1000 * 8 + e.x)) * 0.05;
      }
    }
  }

  _checkCoins(avatar) {
    const gd = this.gameData;
    const px = avatar.position.x;
    const cy = avatar.position.y + 0.8;
    for (const c of gd.coinList) {
      if (c.collected) continue;
      if (Math.abs(c.x - px) < 0.85 && Math.abs(c.y - cy) < 1.2) {
        c.collected = true;
        if (c.mesh) c.mesh.visible = false;
        gd.coinsCollected++;
        gd.score += 100;
        this.score = gd.score;
        this._pushEvent('item_collected', { item: 'coin', total: gd.coinsCollected });
        if (this.rlDriving) this._rlAward(RL_REWARD.coin);
      }
    }
  }

  _checkEnemies(avatar) {
    if (this._invincible > 0 || this._rlEnded) return;
    const gd = this.gameData;
    const px = avatar.position.x;
    const feet = avatar.position.y;
    for (const e of gd.enemyList) {
      if (!e.alive) continue;
      if (Math.abs(e.x - px) < 0.55 && Math.abs(e.y - feet) < 1.1) {
        if (feet > e.y + 0.45 && this._vy <= 0) {
          e.alive = false;
          if (e.mesh) { e.mesh.visible = false; }
          gd.enemiesStomped++;
          gd.score += 200;
          this.score = gd.score;
          this._vy = STOMP_BOUNCE;
          this._pushEvent('enemy_defeated', { id: e.id });
          if (this.rlDriving) {
            this._rlAward(RL_REWARD.stomp);
            // 连续踩敌奖励：短时间内连续踩敌额外加分
            const now = performance.now() / 1000;
            if (this._lastStompTime && now - this._lastStompTime < 2.0) {
              this._stompChain = (this._stompChain || 0) + 1;
              this._rlAward(RL_REWARD.stomp_chain * this._stompChain);
            } else {
              this._stompChain = 0;
            }
            this._lastStompTime = now;
          }
        } else {
          this._onPlayerHurt();
          return;
        }
      }
    }
  }

  /** 里程碑奖励：每跑 50m 给一次奖励，不结束游戏 */
  _checkMilestone(avatar) {
    if (this._rlEnded) return;
    const dist = avatar.position.x - SPAWN_X;
    const milestone = Math.floor(dist / 50) * 50;
    if (milestone > 0 && milestone > (this._lastMilestone || 0)) {
      this._lastMilestone = milestone;
      if (this.rlDriving) {
        this._rlAward(RL_REWARD.milestone);
        if (this.App.sendAIAction) {
          this.App.sendAIAction(`（你已经跑了 ${milestone}m！继续保持！）`);
        }
      }
      this._pushEvent('milestone', { distance: milestone });
    }
  }

  // ==================== 事件处理 ====================

  _onBlockBump(b) {
    if (b.type === 'question') {
      b.type = 'brick';
      if (b.mesh && b.mat && b._usedBrickMat) {
        b.mat.color.set(0xc77b3a);
        b.mat.roughness = 0.7;
        b.mat.emissive.set(0x000000);
        b.mat.emissiveIntensity = 0;
        const gd = this.gameData;
        const idx = gd.questionMats.indexOf(b.mat);
        if (idx >= 0) gd.questionMats.splice(idx, 1);
        const origY = b.mesh.position.y;
        b.mesh.position.y = origY + 0.25;
        setTimeout(() => { if (b.mesh) b.mesh.position.y = origY; }, 120);
      }
      const gd = this.gameData;
      gd.coinsCollected++;
      gd.score += 100;
      this.score = gd.score;
      this._pushEvent('item_collected', { item: 'coin_from_block' });
      if (this.rlDriving) this._rlAward(RL_REWARD.coin);
    }
  }

  _onPlayerHurt() {
    const gd = this.gameData;
    this._pushEvent('player_hurt', { lives: gd.lives });
    if (this.rlDriving) this._rlAward(RL_REWARD.hurt);
    this._loseLife();
  }

  _onPitDeath() {
    const gd = this.gameData;
    this._pushEvent('player_hurt', { reason: 'pit', lives: gd.lives });
    if (this.rlDriving) this._rlAward(RL_REWARD.pit_death);
    this._loseLife();
  }

  _loseLife() {
    const gd = this.gameData;
    gd.lives--;
    if (this.rlDriving) this._rlAward(RL_REWARD.life_lost);
    // 无限跑酷模式：1条命，死亡即结束
    if (gd.lives <= 0) {
      if (this.rlDriving) {
        this._endEpisode(false);
      } else {
        this.onFail('坠崖/被敌人击败');
      }
    } else {
      // 终局经验存储
      if (this.rlDriving && this._rl && this._rl.lastStateVec !== null) {
        const agent = this._rl.agent;
        const r = this._rl.accumReward + RL_REWARD.step;
        const zeroState = new Array(RL_STATE_SIZE).fill(0);
        agent.store(this._rl.lastStateVec, this._rl.lastAction, r, zeroState, true);
        agent.train();
        this._rlEpisodeReward += r;
      }
      this._respawn();
      if (this.rlDriving && this._rl) {
        this._rl.lastStateVec = null;
        this._rl.lastAction = null;
        this._rl.accumReward = 0;
      }
    }
  }

  _respawn() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return;
    avatar.position.set(this._lastCheckpointX, 4, PLAY_Z);
    this._vy = 0;
    this._isGrounded = false;
    this._jumpsRemaining = MAX_JUMPS;
    this._invincible = 1.5;
  }

  // ==================== RL 接管 ====================

  _startRLDriving() {
    this.rlDriving = true;
    this._rlMoveDir = 0;
    this._rlWantsJump = false;
    if (this.App.sendAIAction) {
      this.App.sendAIAction('（玩家放手了，我来接管！这次地形是全新的，让我用学到的策略来挑战！）');
    }
    this._pushEvent('ai_takeover', {});
  }

  _stopRLDriving() {
    this.rlDriving = false;
    this._rlMoveDir = 0;
    this._rlWantsJump = false;
    if (this._rl) {
      this._rl.lastStateVec = null;
      this._rl.lastAction = null;
    }
  }

  _rlEnsureAgent() {
    // P1-1 收敛：智能体由 RLAgentManager 按注册表统一创建与缓存
    if (!this._rl) {
      this._rl = {
        agent: null, // 惰性获取（见下方）
        lastStateVec: null,
        lastAction: null,
        lastWasRandom: false,
        accumReward: 0,
        decisionTimer: 0,
        // P2-3 接口节奏真实化：人化反应延迟（替代固定 0.18s 周期）
        interfaceController: RLAgentManager.get().getInterfaceController(),
        // P2-2/3 实际决策间隔（人化延迟的真实经过时间）
        lastStepTs: performance.now(),
        episodeStartTs: performance.now(),
      };
      this._rl.agent = RLAgentManager.get().getAgent('mario', this);
      // P2-1b 从已采集的人类轨迹挂接行为克隆先验（异步、静默跳过）
      RLAgentManager.get().enableBehaviorCloning('mario', this).catch(() => {});
      // P3-1 世界模型增强训练（观察 112 维较大，用保守参数）
      RLAgentManager.get().enableWorldModel('mario', 0.3, 4);
    }
    return this._rl.agent;
  }

  _rlStep() {
    const agent = this._rlEnsureAgent();
    const newStateVec = this._rlEncodeState();
    // P2-3：真实经过的人化决策间隔（供停留惩罚等计时使用）
    const stepSec = Math.min(0.5, Math.max(0.05, (performance.now() - this._rl.lastStepTs) / 1000));
    this._rl.lastStepTs = performance.now();

    // 空中：谨慎感知——坑上方惩罚，鼓励尽快找安全落点
    if (!this._isGrounded) {
      const avatar0 = this.App.currentAvatar;
      if (avatar0) {
        const px0 = avatar0.position.x;
        // 空中在坑上方：额外惩罚，鼓励尽快落地
        if (!this._isOnGroundSegment(px0)) {
          this._rlAward(RL_REWARD.airborne_pit);
        }
      }

      if (this._rl.lastStateVec !== null && this._rl.lastAction !== null) {
        const r = this._rl.accumReward + RL_REWARD.step;
        agent.store(this._rl.lastStateVec, this._rl.lastAction, r, newStateVec, false);
        agent.train();
        this._rlEpisodeReward += r;
      }
      this._rl.accumReward = 0;

      if (this._jumpsRemaining > 0) {
        const valid = this._rlGetValidActions();
        const { action: actionIdx, wasRandom } = agent.chooseAction(newStateVec, valid);
        this._rlApplyAction(actionIdx);
        this._rl.lastStateVec = newStateVec;
        this._rl.lastAction = actionIdx;
        this._rl.lastWasRandom = wasRandom;
      } else {
        this._rl.lastStateVec = newStateVec;
      }
      return;
    }

    // 地面：谨慎策略——强化危险感知 + 安全奖励 + 远离危险奖励
    const avatar = this.App.currentAvatar;
    if (avatar) {
      const px = avatar.position.x;
      const py = avatar.position.y;

      // 【谨慎】着地安全奖励：在安全地面段上每步获得正反馈
      if (this._isOnGroundSegment(px)) {
        this._rlAward(RL_REWARD.grounded_safe);
      }

      // 【谨慎】坑边危险惩罚：扩展感知范围至2.5m，惩罚更强
      const pit = this._getNextPit(px);
      if (pit) {
        const distToPit = pit.start - px;
        if (distToPit > 0 && distToPit < 2.5) {
          const dangerLevel = 1.0 - (distToPit / 2.5);
          this._rlAward(RL_REWARD.pit_edge * dangerLevel);
        }
        // 【新增】远离坑边缘奖励：如果正在远离坑（向左移动），给予正反馈
        if (distToPit > 0 && distToPit < 2.0 && this._rlMoveDir < 0) {
          this._rlAward(RL_REWARD.pit_move_away);
        }
        // 【新增】坑边鲁莽跳跃惩罚：坑边2m内非过坑跳跃
        if (distToPit > 0.6 && distToPit < 2.0 && this._rlWantsJump) {
          this._rlAward(RL_REWARD.reckless_jump);
        }
      }

      // 【谨慎】敌人警惕惩罚：扩展感知范围至5m，分两级危险
      let nearestEnemyD = Infinity;
      let nearestEnemy = null;
      for (const e of this.gameData.enemyList) {
        if (!e.alive) continue;
        const d = e.x - px;
        if (d > -0.3 && d < nearestEnemyD) {
          nearestEnemyD = d;
          nearestEnemy = e;
        }
      }
      if (nearestEnemyD < 5.0) {
        // 5m内：渐增警惕惩罚
        const dangerLevel = 1.0 - (nearestEnemyD / 5.0);
        this._rlAward(RL_REWARD.enemy_near * dangerLevel);
        // 2m内：额外谨慎惩罚
        if (nearestEnemyD < 2.0) {
          this._rlAward(RL_REWARD.enemy_caution * (1.0 - nearestEnemyD / 2.0));
        }
      }

      // 金币引导：前方有金币且朝金币方向移动 → 正反馈（降低，不鼓励冒险）
      let nearestCoin = null;
      let nearestCoinD = Infinity;
      for (const c of this.gameData.coinList) {
        if (c.collected) continue;
        const d = c.x - px;
        if (d > 0 && d < 5 && d < nearestCoinD) {
          nearestCoinD = d;
          nearestCoin = c;
        }
      }
      if (nearestCoin) {
        const proximityReward = RL_REWARD.coin_near * (1.0 - nearestCoinD / 5.0);
        this._rlAward(proximityReward);
        // 仅在金币在上方且距离近时奖励跳跃
        if (nearestCoin.y - py > 1.0 && this._rlWantsJump && nearestCoinD < 2.0) {
          this._rlAward(RL_REWARD.coin_near * 0.5);
        }
      }

      // 停留惩罚：允许更长的观察时间（5秒）
      const movedDist = Math.abs(px - this._lastIdleX);
      if (movedDist > 0.3) {
        this._idleTimer = 0;
        this._lastIdleX = px;
      } else {
        this._idleTimer += stepSec;
        if (this._idleTimer > RL_REWARD.idle_threshold) {
          this._rlAward(RL_REWARD.idle_penalty);
          const extraPenalty = (this._idleTimer - RL_REWARD.idle_threshold) * 0.01;
          this._rlAward(-extraPenalty);
        }
      }
    }

    if (this._rl.lastStateVec !== null && this._rl.lastAction !== null) {
      const r = this._rl.accumReward + RL_REWARD.step;
      agent.store(this._rl.lastStateVec, this._rl.lastAction, r, newStateVec, false);
      agent.train();
      this._rlEpisodeReward += r;
    }
    this._rl.accumReward = 0;
    const valid = this._rlGetValidActions();
    const { action: actionIdx, wasRandom } = agent.chooseAction(newStateVec, valid);
    this._rlApplyAction(actionIdx);

    // ===== 跳跃评估：移除主观惩罚，让AI通过环境反馈自主学习 =====
    // 不再惩罚高台跳跃、过早起跳、无意义跳跃——AI拥有完整屏幕视觉，
    // 可以通过掉坑死亡惩罚(-80)和踩敌/吃金币奖励自行学习最优跳跃时机

    this._rl.lastStateVec = newStateVec;
    this._rl.lastAction = actionIdx;
    this._rl.lastWasRandom = wasRandom;
  }

  /** 状态编码器：16 维，全部使用相对坐标（泛化核心）
   *  [0]  py_norm       玩家 Y 归一化
   *  [1]  grounded      是否着地
   *  [2]  vy_norm       垂直速度归一化
   *  [3]  dist_to_pit   到前方坑的距离归一化
   *  [4]  pit_width     坑宽归一化
   *  [5]  on_safe_ground 是否在安全地面段上
   *  [6]  enemy_dist    最近敌人距离归一化
   *  [7]  enemy_ahead   敌人是否在前方
   *  [8]  coin_dist     最近金币距离归一化
   *  [9]  coin_y_diff   金币高度差归一化
   *  [10] progress_norm 本局前进距离归一化（相对 REFERENCE_DIST）
   *  [11] lives_norm    剩余生命归一化
   *  [12] move_dir      当前水平移动方向
   *  [13] jumps_remaining 剩余跳跃次数归一化
   *  [14] will_land_in_pit 预测落点是否在坑中
   *  [15] speed_norm    当前移动速度归一化
   */
  /**
   * 完整屏幕视觉传感器：在玩家前后扫描15个采样点（-2m ~ +12m），
   * 每点检测6个通道，构建90维空间感知 + 22维全局状态 = 112维。
   *
   * 通道说明（每采样点6维）：
   *   1. terrainHeight  地形高度（相对玩家Y，坑=-1）
   *   2. isPit          是否为坑（1=坑/无地面，0=有地面）
   *   3. enemySignal    敌人信号（0=无敌人，0.5=同高度，>0.5=上方，<0.5=下方）
   *   4. coinSignal     金币信号（同敌人编码）
   *   5. blockSignal    石块阻挡信号（1=有阻挡石块，0=无）
   *   6. blockHeight    石块顶部高度（相对玩家脚部，0=无石块，归一化0~1）
   */
  _rlScanScreen(px, py) {
    const gd = this.gameData;
    const result = new Array(VISION_POINTS * VISION_CHANNELS);
    const feetY = py;
    const headY = py + PLAYER_HEIGHT;

    for (let i = 0; i < VISION_POINTS; i++) {
      const sampleX = px + VISION_SCAN_MIN + i * VISION_SCAN_STEP;
      const base = i * VISION_CHANNELS;

      // --- 通道1+2: 地形高度 & 坑检测（取最高表面） ---
      let terrainY = null;
      for (const s of gd.surfaces) {
        if (sampleX >= s.xMin - 0.1 && sampleX <= s.xMax + 0.1) {
          if (terrainY === null || s.topY > terrainY) terrainY = s.topY;
        }
      }
      if (terrainY === null) {
        result[base + 0] = -1.0;
        result[base + 1] = 1.0;
      } else {
        result[base + 0] = Math.max(-1, Math.min(1, (terrainY - py) / 5.0));
        result[base + 1] = 0.0;
      }

      // --- 通道3: 敌人信号（取最强信号，不break） ---
      let enemySignal = 0.0;
      for (const e of gd.enemyList) {
        if (!e.alive) continue;
        if (Math.abs(e.x - sampleX) < 1.0) {
          const yOff = Math.max(-0.49, Math.min(0.49, (e.y - py) / 10.0));
          const sig = 0.5 + yOff;
          if (sig > enemySignal) enemySignal = sig;  // 取最强信号
        }
      }
      result[base + 2] = enemySignal;

      // --- 通道4: 金币信号（取最强信号，不break） ---
      let coinSignal = 0.0;
      for (const c of gd.coinList) {
        if (c.collected) continue;
        if (Math.abs(c.x - sampleX) < 1.0) {
          const yOff = Math.max(-0.49, Math.min(0.49, (c.y - py) / 10.0));
          const sig = 0.5 + yOff;
          if (sig > coinSignal) coinSignal = sig;
        }
      }
      result[base + 3] = coinSignal;

      // --- 通道5+6: 石块阻挡信号 + 石块高度 ---
      let blockSignal = 0.0;
      let blockHeight = 0.0;
      for (const b of gd.blocks) {
        const bxMin = b.cx - b.w / 2, bxMax = b.cx + b.w / 2;
        if (sampleX >= bxMin - 0.3 && sampleX <= bxMax + 0.3) {
          if (headY > b.cy + 0.02 && feetY < b.cy + b.h - 0.02) {
            blockSignal = 1.0;
            const topRel = (b.cy + b.h - feetY) / 3.0;
            blockHeight = Math.max(blockHeight, Math.max(0, Math.min(1, topRel)));
          }
        }
      }
      result[base + 4] = blockSignal;
      result[base + 5] = blockHeight;
    }

    return result;
  }

  _rlEncodeState() {
    const avatar = this.App.currentAvatar;
    const px = avatar.position.x;
    const py = avatar.position.y;
    const gd = this.gameData;

    // ===== 1. 屏幕视觉传感器阵列（90维） =====
    const vision = this._rlScanScreen(px, py);

    // ===== 2. 全局最近敌人：距离/Y偏移/接近方向/碰撞时间 =====
    let nearestEnemyDist = 1.0;
    let nearestEnemyYOff = 0;
    let nearestEnemyApproaching = 0.5;  // 0.5=无敌人, 1.0=接近, 0.0=远离
    let nearestEnemyTTC = 0.0;          // time-to-collision, 0=无, 1=即将碰撞
    let nearestED = Infinity;
    let nearestEnemy = null;
    let enemiesVisible = 0;
    for (const e of gd.enemyList) {
      if (!e.alive) continue;
      const d = e.x - px;
      if (d > -0.3 && d < 15) enemiesVisible++;
      if (d > -0.3 && d < nearestED) {
        nearestED = d;
        nearestEnemy = e;
      }
    }
    if (nearestED < 15 && nearestEnemy) {
      nearestEnemyDist = Math.max(0, nearestED / 15.0);
      nearestEnemyYOff = Math.max(-1, Math.min(1, (nearestEnemy.y - py) / 5.0));
      const approaching = nearestEnemy.dir < 0;  // 敌人向左走=朝玩家方向
      nearestEnemyApproaching = approaching ? 1.0 : 0.0;
      const relSpeed = (approaching ? nearestEnemy.speed : 0) + MOVE_SPEED;
      const ttc = nearestED / relSpeed;
      nearestEnemyTTC = Math.max(0, Math.min(1, 1.0 - ttc / 3.0));  // 3s内碰撞→1, 远→0
    }

    // ===== 3. 全局最近金币 + 金币数量 =====
    let nearestCoinDist = 1.0;
    let nearestCD = Infinity;
    let coinsVisible = 0;
    for (const c of gd.coinList) {
      if (c.collected) continue;
      const d = c.x - px;
      if (d > -0.5 && d < 15) coinsVisible++;
      if (d > -0.5 && d < 15 && d < nearestCD) nearestCD = d;
    }
    if (nearestCD < 15) {
      nearestCoinDist = Math.max(0, nearestCD / 15.0);
    }

    // ===== 4. 最近的坑：距离 + 宽度 =====
    let nearestPitDist = 1.0;
    let nearestPitWidth = 0;
    const pit = this._getNextPit(px);
    if (pit) {
      const d = pit.start - px;
      if (d > -1 && d < 12) {
        nearestPitDist = Math.max(0, d / 12.0);
        nearestPitWidth = Math.max(0, Math.min(1, (pit.end - pit.start) / 5.0));
      }
    }

    // ===== 5. 地形坡度：前方3m vs 当前高度 =====
    let terrainGradient = 0;
    {
      let curH = null, aheadH = null;
      for (const s of gd.surfaces) {
        if (px >= s.xMin - 0.1 && px <= s.xMax + 0.1) {
          if (curH === null || s.topY > curH) curH = s.topY;
        }
        const aheadX = px + 3;
        if (aheadX >= s.xMin - 0.1 && aheadX <= s.xMax + 0.1) {
          if (aheadH === null || s.topY > aheadH) aheadH = s.topY;
        }
      }
      if (curH !== null && aheadH !== null) {
        terrainGradient = Math.max(-1, Math.min(1, (aheadH - curH) / 3.0));
      }
    }

    // ===== 6. 落点安全预测 =====
    let willLandInPit = 0;
    if (!this._isGrounded) {
      const landX = this._predictLandingX();
      willLandInPit = this._isOnGroundSegment(landX) ? 0 : 1;
    } else if (py > 1.5) {
      const edge = this._getCurrentSurfaceEdge(px, py);
      if (edge) {
        const distToEdge = edge.edgeX - px;
        if (distToEdge > 0 && distToEdge < 2.0) {
          willLandInPit = this._isSafeBelow(edge.edgeX + 0.5) ? 0 : 1;
        }
      }
    }

    // ===== 7. 前方安全落地区域检测（3~8m） =====
    let safeLandingAhead = 0;
    for (let sx = px + 3; sx <= px + 8; sx += 1) {
      if (this._isOnGroundSegment(sx)) { safeLandingAhead = 1; break; }
    }

    // ===== 8. 前方阻挡石块：距离 + 高度 =====
    let blockAheadDist = 1.0;
    let blockAheadHeight = 0;
    const blockAhead = this._getBlockAhead(px, py);
    if (blockAhead) {
      blockAheadDist = Math.max(0, Math.min(1, blockAhead.dist / 3.5));
      blockAheadHeight = Math.max(0, Math.min(1, blockAhead.h / 2.0));
    }

    // ===== 9. 当前表面右边缘距离 =====
    let distToSurfaceEdge = 1.0;
    if (py > 0.5) {
      const edge = this._getCurrentSurfaceEdge(px, py);
      if (edge) {
        const d = edge.edgeX - px;
        if (d > 0 && d < 5) distToSurfaceEdge = Math.max(0, d / 5.0);
      }
    }

    const progress = Math.min(1, (px - SPAWN_X) / REFERENCE_DIST);

    // ===== 10. 玩家全局状态（22维） =====
    const playerState = [
      py / 5.0,                                                      // 0  当前高度
      this._isGrounded ? 1 : 0,                                      // 1  是否着地
      Math.max(-2, Math.min(1, this._vy / JUMP_VELOCITY)),           // 2  垂直速度
      progress,                                                      // 3  关卡进度
      gd.lives / START_LIVES,                                        // 4  剩余生命
      this._rlMoveDir,                                               // 5  移动方向
      this._jumpsRemaining / MAX_JUMPS,                              // 6  剩余跳跃次数
      willLandInPit,                                                 // 7  预测落点是否为坑
      (this._lastMoveSpeed || 0) / MOVE_SPEED,                       // 8  最近移动速度
      nearestEnemyDist,                                              // 9  全局最近敌人距离
      nearestEnemyYOff,                                              // 10 全局最近敌人Y偏移
      nearestEnemyApproaching,                                       // 11 敌人接近方向(1=接近,0=远离,0.5=无)
      nearestEnemyTTC,                                               // 12 碰撞时间紧迫度
      nearestCoinDist,                                               // 13 全局最近金币距离
      nearestPitDist,                                                // 14 最近坑距离
      nearestPitWidth,                                               // 15 坑宽度
      terrainGradient,                                               // 16 地形坡度
      Math.min(1, enemiesVisible / 5.0),                             // 17 可见敌人数量
      Math.min(1, coinsVisible / 5.0),                               // 18 可见金币数量
      safeLandingAhead,                                              // 19 前方安全落地区域
      blockAheadDist,                                                // 20 前方石块距离
      blockAheadHeight,                                              // 21 前方石块高度
    ];

    return [...vision, ...playerState];
  }

  /**
   * 动作有效性：仅保留物理约束，移除所有行为限制。
   * AI 通过完整屏幕视觉传感器自主判断何时跳跃、何时停下、何时走。
   * 物理约束：
   *   - 空中且无跳跃次数 → 只能等待落地
   *   - 空中且有跳跃次数 → 可二段跳或等待
   *   - 不在地面段上（坑边缘） → 不能原地不动
   *   - 地面 → 所有地面动作均可
   */
  _rlGetValidActions() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return [0, 2, 3, 5];

    // 空中：只有二段跳或等待落地（物理约束）
    if (!this._isGrounded) {
      if (this._jumpsRemaining > 0) {
        return [0, 6, 7];  // 等待 / 二段跳原地 / 二段跳向右
      }
      return [0];  // 无可跳：等待落地
    }

    // 不在安全地面上（已在坑边缘）：不能原地不动（物理约束）
    const px = avatar.position.x;
    if (!this._isOnGroundSegment(px)) {
      return [2, 5];  // 只能向右走或向右跳
    }

    // 地面：所有地面动作均可，让AI自主决策
    return [0, 1, 2, 3, 4, 5];
  }

  /** 获取 px 前方最近的坑 {start, end} */
  _getNextPit(px) {
    const segs = this.gameData.groundSegments;
    for (let i = 0; i < segs.length - 1; i++) {
      if (segs[i].end > px - 0.5) {
        return { start: segs[i].end, end: segs[i + 1].start };
      }
    }
    return null;
  }

  /** 获取玩家当前所站表面的右边缘
   *  返回 {edgeX, topY} 或 null */
  _getCurrentSurfaceEdge(px, py) {
    const surfaces = this.gameData.surfaces;
    let bestEdge = null;
    for (const s of surfaces) {
      // 玩家在这个表面上（X 范围内，Y 接近 topY）
      if (px >= s.xMin - 0.1 && px <= s.xMax + 0.1 && Math.abs(py - s.topY) < 0.3) {
        // 找最右边缘
        if (!bestEdge || s.xMax > bestEdge.edgeX) {
          bestEdge = { edgeX: s.xMax, topY: s.topY };
        }
      }
    }
    return bestEdge;
  }

  /** 检查从高台边缘走下去是否安全（下方有地面可落） */
  _isSafeBelow(x) {
    const surfaces = this.gameData.surfaces;
    for (const s of surfaces) {
      if (x >= s.xMin - 0.1 && x <= s.xMax + 0.1 && s.topY < 0.5) {
        return true;  // 下方有地面层
      }
    }
    return false;
  }

  /** 判断 px 是否在某个可落点表面上方 */
  _isOnGroundSegment(px) {
    const surfaces = this.gameData.surfaces;
    for (let i = 0; i < surfaces.length; i++) {
      const s = surfaces[i];
      if (px >= s.xMin - 0.1 && px <= s.xMax + 0.1) return true;
    }
    return false;
  }

  /** 预测向右跳跃的落点是否安全
   *  检查：1.落点有地面  2.落点附近没有敌人
   *  返回 true=安全可跳，false=危险不要跳 */
  _isJumpRightSafe(px, py) {
    // 模拟向右跳跃轨迹：初速 JUMP_VELOCITY，水平速度 MOVE_SPEED
    const vy = JUMP_VELOCITY;
    let y = py;
    let x = px;
    const dt = 0.05;
    let vyCur = vy;
    let landed = false;
    let landX = px;

    for (let step = 0; step < 60; step++) {
      vyCur -= GRAVITY * dt;
      if (vyCur < -MAX_FALL_SPEED) vyCur = -MAX_FALL_SPEED;
      y += vyCur * dt;
      x += MOVE_SPEED * dt;

      // 检测着地
      if (vyCur <= 0) {
        const surfaces = this.gameData.surfaces;
        for (const s of surfaces) {
          if (x >= s.xMin - 0.1 && x <= s.xMax + 0.1 && Math.abs(y - s.topY) < 0.3) {
            landX = x;
            landed = true;
            break;
          }
        }
        if (landed) break;
      }
    }

    if (!landed) return false;  // 没找到落点=不安全

    // 落点附近 1.5m 内有活着的敌人 → 不安全
    for (const e of this.gameData.enemyList) {
      if (!e.alive) continue;
      if (Math.abs(e.x - landX) < 1.5) return false;
    }

    return true;
  }

  /** 检测前方是否有阻挡的石块（玩家身体高度范围内）
   *  返回 {cx, cy, w, h, dist} 或 null */
  _getBlockAhead(px, py) {
    const feetY = py;
    const headY = py + PLAYER_HEIGHT;
    for (const b of this.gameData.blocks) {
      const bxMin = b.cx - b.w / 2, bxMax = b.cx + b.w / 2;
      const d = bxMin - px;
      // 前方 0~3.5m 内的石块
      if (d > -0.3 && d < 3.5) {
        // 检查身体是否与石块高度重叠
        if (headY > b.cy + 0.02 && feetY < b.cy + b.h - 0.02) {
          return { cx: b.cx, cy: b.cy, w: b.w, h: b.h, dist: d };
        }
      }
    }
    return null;
  }

  _rlApplyAction(actionIdx) {
    const name = MARIO_ACTIONS[actionIdx];
    switch (name) {
      case 'idle':              this._rlMoveDir = 0;  this._rlWantsJump = false; break;
      case 'move_left':         this._rlMoveDir = -1; this._rlWantsJump = false; this._rlFaceDir = -1; break;
      case 'move_right':        this._rlMoveDir = 1;  this._rlWantsJump = false; this._rlFaceDir = 1; break;
      case 'jump':              this._rlMoveDir = 0;  this._rlWantsJump = true; break;
      case 'jump_left':         this._rlMoveDir = -1; this._rlWantsJump = true; this._rlFaceDir = -1; break;
      case 'jump_right':        this._rlMoveDir = 1;  this._rlWantsJump = true; this._rlFaceDir = 1; break;
      case 'double_jump':       this._rlWantsJump = true; break;
      case 'double_jump_right': this._rlMoveDir = 1;  this._rlWantsJump = true; this._rlFaceDir = 1; break;
    }
  }

  // ==================== RL Env 契约（P0-1/P0-3 示范接入） ====================
  // 通过 BaseGame 声明的统一 RL 接口暴露本游戏的观察/动作空间，
  // 使 UnifiedRLAgent 及未来服务端训练器无需感知游戏内部实现。

  /** RL 环境规格版本 */
  rlSpecVersion() { return '1.0.0'; }

  /** 声明动作空间：由 MARIO_ACTIONS 生成 */
  getActionSpec() {
    if (!this._actionSpecCache) {
      this._actionSpecCache = MARIO_ACTIONS.map((name, id) => ({
        id,
        name,
        semantics: (name === 'jump' || name === 'jump_left' || name === 'jump_right'
          || name === 'double_jump' || name === 'double_jump_right') ? 'semantic' : 'primitive',
        executable: true,
      }));
    }
    return this._actionSpecCache;
  }

  /** 声明观察空间：90 维视觉网格 + 22 维玩家状态 */
  getObservationSpec() {
    if (!this._obsSpecCache) {
      this._obsSpecCache = [
        { name: 'vision', kind: 'grid', shape: [VISION_POINTS, VISION_CHANNELS], scale: 1, offset: 0 },
        { name: 'player', kind: 'vector', dim: RL_PLAYER_STATE_SIZE, scale: 1, offset: 0 },
      ];
    }
    return this._obsSpecCache;
  }

  /** 当前观察（复用已有 _rlEncodeState，已归一化到 0-1） */
  getObservation() {
    return this._rlEncodeState();
  }

  /** 执行 RL 动作：委托给 _rlApplyAction */
  applyAction(actionId) {
    if (actionId < 0 || actionId >= MARIO_ACTIONS.length) return false;
    this._rlApplyAction(actionId);
    return true;
  }

  /** 当前可用动作（RL 驾驶中全部可用，否则返回 null 由上层处理） */
  getValidActions() {
    if (!this.rlDriving) return null;
    return MARIO_ACTIONS.map((_, i) => i);
  }

  /** RL 超参：返回 null，由注册表 games-config.js 统一提供（P1-1 单一事实源） */
  getRLHyperparams() { return null; }

  _rlApplyContinuous(dt) {
    const avatar = this.App.currentAvatar;
    if (!avatar) return;

    // ===== 物理必要反射（仅保留不可违反的物理约束） =====

    // 空中在坑上方时强制向右移动（物理必要：不能往回掉入坑）
    if (this.rlDriving && !this._isGrounded) {
      const px = avatar.position.x;
      if (!this._isOnGroundSegment(px)) {
        this._rlMoveDir = 1;
      }
    }

    // DQN 选择的空中二段跳
    if (this.rlDriving && !this._isGrounded && this._rlWantsJump && this._jumpsRemaining > 0) {
      this._doJump();
      this._rlWantsJump = false;
    }

    // ===== 执行移动 =====
    if (this._rlMoveDir !== 0) {
      const nx = avatar.position.x + this._rlMoveDir * MOVE_SPEED * dt;
      // checkCollision 内部 RL 模式遇到石块会自动抬升 Y 并放行
      if (!this.checkCollision(nx, PLAY_Z)) {
        avatar.position.x = nx;
      }
      this._setFacing(this._rlMoveDir);
      this._rlSettingSpeed = true; this.setPlayerSpeed(MOVE_SPEED); this._rlSettingSpeed = false;
    } else {
      this._rlSettingSpeed = true; this.setPlayerSpeed(0); this._rlSettingSpeed = false;
    }
    if (this._rlWantsJump && this._isGrounded) {
      this._doJump();
      this._rlWantsJump = false;
    }
  }

  _setFacing(dir) {
    if (this.App) this.App.smoothRotY = dir > 0 ? Math.PI / 2 : -Math.PI / 2;
  }

  _rlAward(amount) {
    if (!this.rlDriving || !this._rl) return;
    this._rl.accumReward += amount;
  }

  _endEpisode(win) {
    this._rlEnded = true;
    const agent = this._rlEnsureAgent();
    if (this._rl.lastStateVec !== null && this._rl.lastAction !== null) {
      const r = this._rl.accumReward + RL_REWARD.step + (win ? RL_REWARD.win : RL_REWARD.lose);
      const zeroState = new Array(RL_STATE_SIZE).fill(0);
      agent.store(this._rl.lastStateVec, this._rl.lastAction, r, zeroState, true);
      agent.train();
      this._rlEpisodeReward += r;
    }
    agent.endEpisode(this._rlEpisodeReward, {
      win,
      coins: this.gameData.coinsCollected,
      stomps: this.gameData.enemiesStomped,
      progress: this._maxX - SPAWN_X,
    });
    // P2-2 人类限时评估基准：记录本局耗时与胜负
    if (this._rl) {
      RLAgentManager.get().getBaselineEvaluator().recordEpisode('mario', {
        durationMs: performance.now() - this._rl.episodeStartTs,
        win: !!win,
      });
      this._rl.episodeStartTs = performance.now();
      this._rl.interfaceController.reset();
    }
    this._rlEpisodeReward = 0;
    this._rl.lastStateVec = null;
    this._rl.lastAction = null;
    this._rl.accumReward = 0;
    this._rlRestartTimer = 1.2;
    if (this.App.sendAIAction) {
      this.App.sendAIAction(win ? '（你存活了足够远的距离！泛化能力不错！）' : '（又失败了……每次失败都让你更接近通关。）');
    }
  }

  _resetEpisode() {
    const avatar = this.App.currentAvatar;

    // 清除旧的程序化地形 mesh
    const THREE = this.THREE;
    const gd = this.gameData;
    if (gd && gd._meshRegistry && THREE) {
      for (const m of gd._meshRegistry) {
        this._safeRemoveMesh(m);
      }
    }
    // 同步清理 sceneObjects 中已无效的引用，防止长时间训练数组膨胀
    if (this.sceneObjects && this.sceneObjects.length > 0) {
      this.sceneObjects = this.sceneObjects.filter(obj => obj.parent !== null);
    }

    // 重新初始化程序化关卡（每局新地形！）
    this._initProceduralLevel();

    if (avatar) avatar.position.set(SPAWN_X, 3, PLAY_Z);
    this._vy = 0;
    this._isGrounded = false;
    this._jumpsRemaining = MAX_JUMPS;
    this._invincible = 0;
    this._maxX = SPAWN_X;
    this._lastCheckpointX = SPAWN_X;
    this._lastGroundSegIdx = 0;
    this._idleTimer = 0;
    this._lastIdleX = SPAWN_X;
    this._lastStompTime = 0;
    this._stompChain = 0;
    this._lastMilestone = 0;
    this.gameData.lives = START_LIVES;
    this.gameData.score = 0;
    this.gameData.coinsCollected = 0;
    this.gameData.enemiesStomped = 0;
    this.score = 0;
    if (this._rl) {
      this._rl.lastStateVec = null;
      this._rl.lastAction = null;
      this._rl.accumReward = 0;
      this._rl.decisionTimer = 0;
    }
    this._rlEpisodeReward = 0;
    this._rlRestartTimer = 0;
    this._rlEnded = false;
  }

  // ==================== 摄像机 ====================

  _lockSideCamera() {
    if (!this.App) return;
    this.App._gameCamAzimuth = 0;
    this.App._gameCamPitch = 0.12;
    this.App._gameCamRadius = 9;
  }

  // ==================== AI 感知 ====================

  getExtraState() {
    const gd = this.gameData || {};
    const avatar = this.App ? this.App.currentAvatar : null;
    const distance = avatar ? Math.max(0, +(avatar.position.x - SPAWN_X).toFixed(1)) : 0;
    return {
      lives: gd.lives ?? 0,
      coins: gd.coinsCollected ?? 0,
      enemies_stomped: gd.enemiesStomped ?? 0,
      enemies_left: (gd.enemyList || []).filter(e => e.alive).length,
      distance,
      progress_pct: 0,  // 无限跑酷无终点，不再计算百分比
      rl_driving: this.rlDriving,
    };
  }

  _getMapData() {
    const gd = this.gameData || {};
    return {
      type: 'endless_runner',
      win_distance: 0,  // 0 表示无限，无终点
      ground_segments: (gd.groundSegments || []).map(s => [s.start, s.end]),
      pits: this._getPits(),
    };
  }

  _getPits() {
    const pits = [];
    const segs = (this.gameData || {}).groundSegments || [];
    for (let i = 0; i < segs.length - 1; i++) {
      pits.push([segs[i].end, segs[i + 1].start]);
    }
    return pits;
  }

  _getObjectsData() {
    const gd = this.gameData || {};
    return {
      coin: (gd.coinList || []).filter(c => !c.collected).map(c => ({ id: `coin_${c.x.toFixed(1)}`, x: c.x, y: c.y, z: PLAY_Z })),
      enemy: (gd.enemyList || []).filter(e => e.alive).map(e => ({ id: `enemy_${e.id}`, x: e.x, y: e.y, z: PLAY_Z })),
    };
  }

  _getNearbyObjects() {
    const avatar = this.App ? this.App.currentAvatar : null;
    if (!avatar) return [];
    const px = avatar.position.x;
    const result = [];
    const gd = this.gameData;
    if (gd) {
      for (const c of gd.coinList) {
        if (c.collected) continue;
        const d = c.x - px;
        if (Math.abs(d) < 12) result.push({ type: 'coin', x: c.x, y: c.y, distance: +Math.abs(d).toFixed(1), direction: d > 0 ? 'right' : 'left' });
      }
      for (const e of gd.enemyList) {
        if (!e.alive) continue;
        const d = e.x - px;
        if (Math.abs(d) < 14) result.push({ type: 'enemy', id: e.id, x: e.x, y: e.y, distance: +Math.abs(d).toFixed(1), direction: d > 0 ? 'right' : 'left' });
      }
    }
    return result.sort((a, b) => a.distance - b.distance);
  }

  _getPlayerFacing() {
    return this.App ? (this.App.smoothRotY || 0) : 0;
  }

  _getPlayerSpeed() {
    return this._lastMoveSpeed || 0;
  }

  // ==================== RL 训练面板（UI） ====================

  _ensureRLPanel() {
    if (this._rlBtn) return;
    const btn = document.createElement('button');
    btn.textContent = '🧠 RL';
    btn.style.cssText = 'position:fixed;right:14px;top:80px;z-index:9998;padding:6px 12px;' +
      'background:#2a2a3a;color:#7be38b;border:1px solid #3a3a55;border-radius:6px;cursor:pointer;font-size:13px;';
    btn.title = '强化学习训练面板';
    btn.addEventListener('click', () => this._toggleRLPanel());
    document.body.appendChild(btn);
    this._rlBtn = btn;
  }

  _toggleRLPanel() {
    if (this._rlPanel) { this._removeRLPanel(); return; }
    const agent = this._rlEnsureAgent();
    const panel = document.createElement('div');
    panel.style.cssText = 'position:fixed;right:14px;top:116px;z-index:9998;width:280px;' +
      'background:rgba(20,22,35,0.94);color:#e0e0e8;border:1px solid #3a3a55;border-radius:8px;' +
      'padding:12px;font-size:12px;font-family:monospace;box-shadow:0 6px 24px rgba(0,0,0,0.5);';
    panel.innerHTML = `
      <div style="font-weight:bold;color:#7be38b;margin-bottom:8px;">马里奥 DQN v7 · 112维感知 · 谨慎策略</div>
      <div id="mrl-stats"></div>
      <div style="margin-top:10px;border-top:1px solid #3a3a55;padding-top:8px;">
        <div style="margin-bottom:6px;">训练加速：
          <span id="mrl-speed-btns"></span>
        </div>
        <button id="mrl-save" style="width:100%;margin-top:4px;padding:4px;background:#2e4a3a;color:#aef0c0;border:1px solid #3a5a4a;border-radius:4px;cursor:pointer;">保存网络</button>
        <button id="mrl-best" style="width:100%;margin-top:4px;padding:4px;background:#2a3a4a;color:#a0c0f0;border:1px solid #3a4a5a;border-radius:4px;cursor:pointer;">恢复最佳策略</button>
        <button id="mrl-reset" style="width:100%;margin-top:4px;padding:4px;background:#4a2a2a;color:#f0aeae;border:1px solid #5a3a3a;border-radius:4px;cursor:pointer;">重置网络</button>
      </div>`;
    document.body.appendChild(panel);
    this._rlPanel = panel;

    const speedWrap = panel.querySelector('#mrl-speed-btns');
    for (const s of TRAIN_SPEEDS) {
      const b = document.createElement('button');
      b.textContent = s + 'x';
      b.dataset.speed = String(s);
      b.style.cssText = 'margin:0 2px;padding:2px 6px;background:#2a2a3a;color:#cfcfe0;border:1px solid #3a3a55;border-radius:3px;cursor:pointer;font-size:11px;';
      b.addEventListener('click', () => { this._trainSpeed = s; this._updateRLPanel(); });
      speedWrap.appendChild(b);
    }
    panel.querySelector('#mrl-save').addEventListener('click', () => { agent.flush(); this._flashPanel('已保存'); });
    panel.querySelector('#mrl-best').addEventListener('click', () => {
      if (agent.restoreBest()) { this._flashPanel('已恢复最佳策略'); }
      else { this._flashPanel('无最佳快照'); }
    });
    panel.querySelector('#mrl-reset').addEventListener('click', () => {
      if (confirm('确定重置 DQN 网络？所有学习成果将清空。')) { agent.reset(); this._updateRLPanel(); }
    });
    this._updateRLPanel();
  }

  _flashPanel(msg) {
    if (!this._rlPanel) return;
    const stats = this._rlPanel.querySelector('#mrl-stats');
    stats.innerHTML = `<div style="color:#7be38b;">${msg}</div>`;
    setTimeout(() => this._updateRLPanel(), 800);
  }

  _updateRLPanel() {
    if (!this._rlPanel || !this._rl) return;
    const a = this._rl.agent;
    const s = a.stats;
    const stats = this._rlPanel.querySelector('#mrl-stats');
    const winRate = s.episodes > 0 ? ((s.wins / s.episodes) * 100).toFixed(1) : '0.0';
    const recentWR = s.recentGames > 0 ? ((s.recentWins / s.recentGames) * 100).toFixed(1) : '0.0';
    const replaySize = a._replay ? a._replay.size : 0;
    const replayCap = a._replay ? a._replay.states.length : 0;
    const replayPct = replayCap > 0 ? ((replaySize / replayCap) * 100).toFixed(0) : '0';
    const avatar = this.App ? this.App.currentAvatar : null;
    const curDist = avatar ? Math.max(0, avatar.position.x - SPAWN_X).toFixed(0) : '0';
    stats.innerHTML = `
      <div>局数: <b>${s.episodes}</b> · 胜率: <b style="color:#7be38b">${winRate}%</b> · 近期: <b style="color:#e3c87b">${recentWR}%</b></div>
      <div>最佳胜率: <b style="color:#a0c0f0">${(s.bestWinRate * 100).toFixed(1)}%</b> · 最远: <b style="color:#e3c87b">${(s.bestProgress || 0).toFixed(1)}m</b></div>
      <div>金币: ${s.coinsCollected} · 踩敌: ${s.enemiesStomped} · 里程碑: ${s.wins}</div>
      <div>当前距离: <b style="color:#e3c87b">${curDist}m</b> ∞</div>
      <div>视觉: <span style="color:#8ba8e3">15点×6通道</span> (±2~+12m) + 全局22维</div>
      <div>ε: ${(a.epsilon * 100).toFixed(1)}% · α:${a.alpha} · γ:${a.gamma}</div>
      <div>训练步: ${a.trainSteps} · 经验池: ${replaySize}/${replayCap} (${replayPct}%) · loss: ${s.avgLoss.toFixed(4)}</div>
      <div>状态: ${this.rlDriving ? '<span style="color:#7be38b">🤖 AI 接管中</span>' : '<span style="color:#8ba8e3">🎮 玩家操控</span>'}</div>`;
    const btns = this._rlPanel.querySelectorAll('#mrl-speed-btns button');
    btns.forEach(b => {
      b.style.background = (+b.dataset.speed === this._trainSpeed) ? '#2e4a3a' : '#2a2a3a';
      b.style.color = (+b.dataset.speed === this._trainSpeed) ? '#aef0c0' : '#cfcfe0';
    });
  }

  _removeRLPanel() {
    if (this._rlPanel) { this._rlPanel.remove(); this._rlPanel = null; }
    if (this._rlBtn) { this._rlBtn.remove(); this._rlBtn = null; }
  }

  _onExitPersist() {
    if (this._rl) this._rl.agent.flush();
  }

  declare App: any;
  declare THREE: any;
  declare _actionSpecCache: any;
  declare _dummy: any;
  declare _enemyIdCounter: any;
  declare _generatedUntil: any;
  declare _idleTimer: any;
  declare _invincible: any;
  declare _isGrounded: any;
  declare _jumpsRemaining: any;
  declare _lastCheckpointX: any;
  declare _lastGroundSegIdx: any;
  declare _lastIdleX: any;
  declare _lastInputTime: any;
  declare _lastMilestone: any;
  declare _lastMoveSpeed: any;
  declare _lastStompTime: any;
  declare _maxX: any;
  declare _obsSpecCache: any;
  declare _prevFeetY: any;
  declare _pushEvent: any;
  declare _rl: any;
  declare _rlBtn: any;
  declare _rlClimbing: any;
  declare _rlEnded: any;
  declare _rlEpisodeReward: any;
  declare _rlFaceDir: any;
  declare _rlMoveDir: any;
  declare _rlPanel: any;
  declare _rlRestartTimer: any;
  declare _rlSettingSpeed: any;
  declare _rlWantsJump: any;
  declare _segmentCount: any;
  declare _stompChain: any;
  declare _terrainSeed: any;
  declare _trainSpeed: any;
  declare _vy: any;
  declare addToScene: any;
  declare description: any;
  declare displayName: any;
  declare gameData: any;
  declare initialCameraHeight: any;
  declare initialCameraRadius: any;
  declare moveSpeed: any;
  declare name: any;
  declare onFail: any;
  declare rlDriving: any;
  declare sceneObjects: any;
  declare score: any;
  declare state: any;
  declare uiHint: any;
}

export default MarioGame;