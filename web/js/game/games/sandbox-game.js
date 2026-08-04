/* ============================================================
 * 沙盒大世界 (Sandbox World)
 *
 * 玩法：
 * - 无限随机生成的方块大世界，类似简化版 Minecraft
 * - 玩家操控角色在草原上流浪、探索不同生态
 * - 靠近树木/矿石/花草即可自动采集，积累背包资源
 * - AI 感知周围环境：地形、生态、附近资源、背包变化
 * - 作为通用 Mod，完全基于 BaseGame 接口实现
 * ============================================================ */

import { BaseGame } from './base-game.js';
import { RLAgentManager } from '../rl/rl-agent-manager.js';
import { encodeObservation } from '../rl/observation-spec.js';

// ==================== RL 配置（P1-4：统一契约接入） ====================
// 决策节奏由 HumanInterfaceController 接管（P2-3）：人化反应延迟 + ≤20Hz 硬约束
const RL_MOVE_SPEED = 3.5;          // RL 移动速度 (m/s)
const RL_TAKEOVER_DELAY_MS = 5000;  // 用户停止操作 5s 后 RL 接管
const RL_WORLD_BOUND = 85;          // 世界边界（|x|/|z| 超过视为出界）
// 奖励表（统一符号策略）
const RL_REWARD = {
  STEP: -0.01,        // 时间代价（负小）
  HIT_OBSTACLE: -0.2, // 碰撞阻挡（负中）
  COLLECT: 5,         // 采集资源（正大）
  APPROACH: 0.1,      // 靠近资源（正小）
  BOUNDARY: -0.1,     // 靠近世界边界（负小）
  EVAL_WIN_COLLECT: 10, // P2-2 评估：达到该采集数视为"完成目标"
};

// ==================== 世界常量 ====================
const BLOCK_SIZE = 1;
const CHUNK_SIZE = 16;
const RENDER_DISTANCE_CHUNKS = 4;   // 玩家周围加载的区块半径（从4降到3，81→49区块）
const WATER_LEVEL = 0;
const MAX_STEP_HEIGHT = 0.9;        // 可攀爬的最大高度差（主角身高~1.8的一半）
const PICKUP_RANGE = 1.2;           // 自动采集距离
const PERCEPTION_RANGE = 200;       // AI 感知附近资源的距离（视野200）

// ==================== 物理常量 ====================
const GRAVITY = 20;                 // 重力加速度 (m/s^2) —— 稍大于地球重力
const JUMP_VELOCITY = 10;           // 单跳初速度 (约1.5个方块高)
const DOUBLE_JUMP_VELOCITY = 7;     // 二段跳初速度
const WATER_GRAVITY = 4;            // 水中下落速度
const WATER_BUOYANCY = 3;           // 水中上浮力
const MAX_FALL_SPEED = 25;          // 最大下落速度
const COLLISION_RADIUS = 0.3;       // 玩家碰撞半径
const RESOURCE_COLLISION_RADIUS = {
  tree: 0.45, rock: 0.35, cactus: 0.3,
  coal_ore: 0.35, iron_ore: 0.35,
};

// ==================== 动物类型定义 ====================
const ANIMAL_DEFS = {
  rabbit:  { name: '兔子', biomes: ['plains', 'forest'],         color: 0xd4a574, size: 0.25, speed: 1.5, fly: false },
  deer:    { name: '鹿',   biomes: ['plains', 'forest'],         color: 0xc4945a, size: 0.55, speed: 2.0, fly: false },
  bird:    { name: '小鸟', biomes: ['plains', 'forest', 'desert', 'snow'], color: 0x4488cc, size: 0.18, speed: 3.0, fly: true },
  fox:     { name: '狐狸', biomes: ['forest', 'snow'],           color: 0xdd6633, size: 0.3,  speed: 2.5, fly: false },
  goat:    { name: '山羊', biomes: ['mountain'],                 color: 0xcccccc, size: 0.4,  speed: 2.2, fly: false },
  turtle:  { name: '乌龟', biomes: ['plains', 'desert'],         color: 0x669944, size: 0.22, speed: 0.6, fly: false },
  bear:    { name: '棕熊', biomes: ['forest', 'mountain'],       color: 0x8B4513, size: 0.65, speed: 1.3, fly: false },
  wolf:    { name: '狼',   biomes: ['forest', 'snow', 'mountain'], color: 0x888888, size: 0.45, speed: 3.2, fly: false },
  eagle:   { name: '鹰',   biomes: ['mountain', 'forest'],       color: 0x4a3728, size: 0.35, speed: 4.5, fly: true },
};
const ANIMAL_SPAWN_CHANCE = 0.0045;  // 每个地块生成动物的概率（0.45%）
const MAX_ANIMALS = 50;              // 全局最大动物数量

// ==================== 方块颜色 ====================
const BLOCK_COLORS = {
  grass: 0x6bbf5a,
  dirt: 0x8b5a2b,
  stone: 0x888888,
  sand: 0xe8c99a,
  water: 0x4fa4b8,
  snow: 0xffffff,
  wood: 0x8b5a2b,
  leaves: 0x52a840,
};

// ==================== 生态与物品名称 ====================
const BIOME_NAMES = {
  plains: '草原',
  forest: '森林',
  desert: '沙漠',
  mountain: '山地',
  snow: '雪原',
};

const ITEM_NAMES = {
  wood: '木材',
  stone: '石头',
  cactus: '仙人掌',
  flower: '花朵',
  berry: '浆果',
  coal: '煤矿',
  iron: '铁矿',
};

// ==================== 资源类型定义 ====================
const RESOURCE_TYPES = {
  tree:      { name: '树木',   item: 'wood',   amount: 2, color: 0x8b5a2b, solid: true },
  pine_tree: { name: '松树',   item: 'wood',   amount: 2, color: 0x2d5a1e, solid: true },
  rock:      { name: '岩石',   item: 'stone',  amount: 1, color: 0x999999, solid: true },
  cactus:    { name: '仙人掌', item: 'cactus', amount: 1, color: 0x66aa44, solid: true },
  flower:    { name: '野花',   item: 'flower', amount: 1, color: 0xff66aa, solid: false },
  berry_bush:{ name: '浆果丛', item: 'berry',  amount: 2, color: 0xcc3333, solid: false },
  mushroom:  { name: '蘑菇',   item: 'berry',  amount: 1, color: 0xff4444, solid: false },
  tall_grass:{ name: '草丛',   item: 'flower', amount: 1, color: 0x669944, solid: false },
  coal_ore:  { name: '煤矿',   item: 'coal',   amount: 1, color: 0x333333, solid: true },
  iron_ore:  { name: '铁矿',   item: 'iron',   amount: 1, color: 0xb87333, solid: true },
};

// ==================== 伪随机噪声（确定性 + 连续） ====================
function hash2d(x, z) {
  let h = Math.sin(x * 12.9898 + z * 78.233) * 43758.5453;
  h = h - Math.floor(h);
  // 二次混合，减少格子感
  const h2 = Math.sin((x + 31.415) * 53.123 + (z + 27.182) * 91.321) * 12345.6789;
  return (h + (h2 - Math.floor(h2))) % 1;
}

function valueNoise(x, z) {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const u = fx * fx * (3 - 2 * fx);
  const v = fz * fz * (3 - 2 * fz);

  const a = hash2d(ix, iz);
  const b = hash2d(ix + 1, iz);
  const c = hash2d(ix, iz + 1);
  const d = hash2d(ix + 1, iz + 1);

  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

function fbm(x, z, octaves = 6, frequency = 0.02, persistence = 0.5) {
  let total = 0;
  let amp = 1;
  let freq = frequency;
  let maxValue = 0;
  for (let i = 0; i < octaves; i++) {
    total += valueNoise(x * freq, z * freq) * amp;
    maxValue += amp;
    amp *= persistence;
    freq *= 2;
  }
  return total / maxValue; // 归一化到约 0~1
}

// 将 0~1 的噪声映射到 -1~1
function signedNoise(x, z, octaves, frequency, persistence) {
  return fbm(x, z, octaves, frequency, persistence) * 2 - 1;
}

// ==================== 生态判断 ====================
function getBiome(wx, wz, height, seed) {
  const temp = fbm(wx + seed, wz + seed * 0.7, 2, 0.0035, 0.5);
  const moist = fbm(wx - seed, wz + seed * 1.3, 2, 0.0045, 0.5);

  // 高山和雪原不受温湿度影响，优先判断
  if (height >= 6) return 'mountain';
  if (height <= -1 || temp < 0.22) return 'snow';
  if (temp > 0.68 && moist < 0.45) return 'desert';
  if (moist > 0.62 && temp > 0.3) return 'forest';
  return 'plains';
}

export class SandboxGame extends BaseGame {
  constructor(app) {
    super(app);

    this.name = 'sandbox';
    this.displayName = '沙盒世界';
    this.description = '无限随机生成的方块大世界，流浪探索、采集资源，和AI伙伴一起冒险。';
    this.moveSpeed = 3.5;
    this.initialCameraRadius = 12; // 相机初始化距离主角12米
    this.initialCameraHeight = 10;   // 相机初始化高度10米（相对角色脚底）
    this.uiHint = 'WASD 移动 · 空格/单击跳跃 · 双击二段跳 · 靠近资源自动采集';

    // 世界参数
    this.blockSize = BLOCK_SIZE;
    this.chunkSize = CHUNK_SIZE;
    this.renderDistanceChunks = RENDER_DISTANCE_CHUNKS;
    this.seed = Math.floor(Math.random() * 1000000);

    // 运行时数据
    this.chunks = new Map();          // key: "cx,cz" -> chunk
    this.resources = [];              // 全局资源列表
    this.animals = [];                // 全局动物列表
    this.inventory = { wood: 0, stone: 0, cactus: 0, flower: 0, berry: 0, coal: 0, iron: 0 };
    this.currentBiome = 'plains';
    this.currentChunk = { x: 0, z: 0 };

    this._lastMoveSpeed = 0;
    this._resourcesCollected = 0;
    this._originalBackground = undefined;
    this._lastKnownGroundY = 0;

    // RL 会话状态（P1-4）
    this._rl = null;

    // 物理状态
    this._playerVelocityY = 0;        // 垂直速度
    this._isGrounded = false;         // 是否在地面
    this._jumpsLeft = 2;              // 剩余跳跃次数（最大2次=二段跳）
    this._maxJumps = 2;
    this._lastJumpTime = 0;

    // 共享材质池（避免每个mesh创建重复材质）
    const THREE = this.THREE;
    this._dummy = THREE ? new THREE.Object3D() : null;
    this._sharedMaterials = THREE ? {} : null;
    if (THREE) {
      // 地形共享材质：平面着色关闭，让光照在方块面间过渡更自然
      this._sharedMaterials.terrainSolid = new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0.0, flatShading: false });
      // 水面：半透明、低粗糙度、微微自发光，更接近真实水面
      this._sharedMaterials.terrainWater = new THREE.MeshStandardMaterial({
        color: 0x4fa4b8, roughness: 0.05, metalness: 0.1,
        transparent: true, opacity: 0.72,
        emissive: 0x0a3344, emissiveIntensity: 0.15,
      });
      // 资源共享材质（按颜色缓存）
      this._sharedMaterials.resource = {};
    }
  }

  // ==================== 生命周期 ====================

  generateScene() {
    const avatar = this.App.currentAvatar;

    // 重置旧世界
    this._resetWorld();

    // 光照与天空
    this._createEnvironment();

    // 先生成出生点周围区块，确保能找到地面
    this._ensureChunk(0, 0);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dz = -1; dz <= 1; dz++) {
        if (dx !== 0 || dz !== 0) this._ensureChunk(dx, dz);
      }
    }

    // 放置角色到出生点（略高于地面，开局自然掉落）
    const spawn = this._findSpawnPoint();
    spawn.y += 10; // 高空出生，自由落体开局（掉落10米）
    if (avatar) {
      avatar.position.set(spawn.x, spawn.y, spawn.z);
    }

    this._updateChunks(true);
  }

  onStart() {
    super.onStart();
    // 重置物理（开局悬空，会自然下落）
    this._playerVelocityY = 0;
    this._isGrounded = false; // 开局悬空
    this._jumpsLeft = this._maxJumps;
    this.uiHint = 'WASD 移动 · 空格/单击跳跃 · 双击二段跳 · 靠近资源自动采集';
    if (this.App.sendAIAction) {
      this.App.sendAIAction('（欢迎来到无限沙盒世界！这是一片随机生成的大地，有森林、沙漠、雪原和山地。我们去流浪、探险、收集资源吧！）');
    }
  }

  update(dt) {
    super.update(dt);
    if (this.state !== 'playing') return;

    const avatar = this.App.currentAvatar;
    if (!avatar) return;

    // 动态加载 / 卸载区块
    this._updateChunks();

    // 重力、地形碰撞、跳跃
    this._updatePlayerPhysics(avatar, dt);

    // 动物漫游
    this._updateAnimals(dt);

    // 自动采集资源
    this._checkResourceCollection(avatar);

    // 生态/位置变化事件
    this._checkLocationEvents(avatar);

    // RL 决策驱动（用户停止操作后接管）
    this._rlUpdate(dt);

    // 更新 UI 提示
    this._updateHint();
  }

  // ==================== RL Env 契约（P1-4） ====================

  /** RL 环境规格版本 */
  rlSpecVersion() { return '1.0.0'; }

  /** 声明动作空间：8 方向世界移动 */
  getActionSpec() {
    if (!this._rlActionSpec) {
      // N/NE/E/SE/S/SW/W/NW（世界方向）
      const dirs = [
        [0, -1], [0.707, -0.707], [1, 0], [0.707, 0.707],
        [0, 1], [-0.707, 0.707], [-1, 0], [-0.707, -0.707],
      ];
      this._rlActionSpec = dirs.map((d, id) => ({
        id, name: 'move_' + id, semantics: 'semantic', executable: true,
        dir: d,
      }));
    }
    return this._rlActionSpec;
  }

  /** 声明观察空间：11 维 */
  getObservationSpec() {
    if (!this._rlObsSpec) {
      this._rlObsSpec = [
        { name: 'nearest', kind: 'vector', dim: 3, scale: 1, offset: 0 }, // 最近资源 dx/dz/dist
        { name: 'resValue', kind: 'scalar', scale: 1, offset: 0 },        // 资源价值
        { name: 'nearCount', kind: 'scalar', scale: 5, offset: 0 },       // 5m 内资源数
        { name: 'yaw', kind: 'scalar', scale: 1, offset: 0 },             // 朝向
        { name: 'speed', kind: 'scalar', scale: 5, offset: 0 },           // 速度
        { name: 'progress', kind: 'scalar', scale: 20, offset: 0 },       // 采集进度
        { name: 'frontBlocked', kind: 'scalar', scale: 1, offset: 0 },    // 前方 2m 阻挡
        { name: 'boundary', kind: 'scalar', scale: 1, offset: 0 },        // 靠近边界
        { name: 'grounded', kind: 'scalar', scale: 1, offset: 0 },        // 是否接地
      ];
    }
    return this._rlObsSpec;
  }

  /** 获取当前观察（各特征已归一化 0-1，经规格模块编码为 Float64Array） */
  getObservation() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return new Float64Array(11);
    const px = avatar.position.x, pz = avatar.position.z;

    // 最近资源
    let nearest = [0, 0, 1], nearestDist = Infinity, resValue = 0;
    let nearCount = 0;
    for (const r of this.resources) {
      if (!r.mesh || !r.mesh.parent) continue;
      const dx = r.mesh.position.x - px, dz = r.mesh.position.z - pz;
      const d = Math.hypot(dx, dz);
      if (d < nearestDist) {
        nearestDist = d;
        nearest = [dx / 10, dz / 10, Math.min(1, d / 10)];
        resValue = this._resourceValue(r.type);
      }
      if (d < 5) nearCount++;
    }

    // 前方 2m 阻挡检测（沿角色朝向）
    const yaw = avatar.rotation ? avatar.rotation.y : 0;
    const fx = px + Math.sin(yaw) * 2, fz = pz + Math.cos(yaw) * 2;
    const frontBlocked = this.checkCollision(fx, fz, { ignoreTerrain: false }) ? 1 : 0;

    return encodeObservation(this.getObservationSpec(), {
      nearest,
      resValue,
      nearCount,
      yaw: (yaw + Math.PI) / (2 * Math.PI),
      speed: Math.abs(this._lastMoveSpeed || 0),
      progress: this._resourcesCollected,
      frontBlocked,
      boundary: (Math.abs(px) > RL_WORLD_BOUND || Math.abs(pz) > RL_WORLD_BOUND) ? 1 : 0,
      grounded: this._isGrounded ? 1 : 0,
    });
  }

  /** 资源价值映射（稀有资源价值更高） */
  _resourceValue(type) {
    if (type === 'iron_ore' || type === 'coal_ore') return 1;
    if (type === 'berry' || type === 'flower') return 0.6;
    return 0.3;
  }

  /** 执行 RL 动作：设置移动方向向量 */
  applyAction(actionId) {
    const spec = this.getActionSpec();
    if (actionId < 0 || actionId >= spec.length || !this._rl) return false;
    this._rl.moveDir = spec[actionId].dir;
    return true;
  }

  /** 当前可用动作（全部可用） */
  getValidActions() { return this.getActionSpec().map((_, i) => i); }

  /** RL 超参：注册表统一提供 */
  getRLHyperparams() { return null; }

  /** 回合是否结束（RL 会话持续运行，由外部重置） */
  rlDone() { return false; }

  /** 惰性初始化 RL 智能体（注册表统一配置） */
  _rlEnsureAgent() {
    if (!this._rl) {
      this._rl = {
        agent: RLAgentManager.get().getAgent('sandbox', this),
        active: false,
        lastUserTime: performance.now(),
        hitObstacle: 0,
        moveDir: [0, 0],
        // P2-3 接口节奏真实化：人化反应延迟控制器
        interfaceController: RLAgentManager.get().getInterfaceController(),
        // P2-2 评估：本局起始时间与最近决策时刻
        episodeStartTs: performance.now(),
        lastStepTs: performance.now(),
      };
      // P2-1b 从已采集的人类轨迹挂接行为克隆先验（异步、静默跳过）
      RLAgentManager.get().enableBehaviorCloning('sandbox', this).catch(() => {});
      // P3-1 世界模型增强训练（想象回放，提升样本效率）
      RLAgentManager.get().enableWorldModel('sandbox', 0.5, 8);
    }
    return this._rl.agent;
  }

  /** 每帧 RL 驱动：接管判定 + 决策循环（P2-3 人化节奏） */
  _rlUpdate(dt) {
    const agent = this._rlEnsureAgent();
    if (!agent) return;

    const gm = this.App.gameModeManager;
    const bridge = (gm && gm.controlBridge) || this.App._gameControlBridge || this.App.controlBridge;
    const userActive = !!(bridge && bridge.userControlling);

    if (userActive) {
      this._rl.lastUserTime = performance.now();
      this._rl.active = false;
      // P2-1a 人类轨迹采集（用户真实操控期间记录）
      const rec = RLAgentManager.get().getTrajectoryRecorder();
      if (!rec.isRecording()) rec.startRecording('sandbox');
      rec.recordFrame(this, bridge);
      return;
    }
    // 用户停止操控 → 结束并保存一段人类轨迹
    const rec = RLAgentManager.get().getTrajectoryRecorder();
    if (rec.isRecording()) rec.stopRecording();

    if (!this._rl.active && performance.now() - this._rl.lastUserTime < RL_TAKEOVER_DELAY_MS) return;
    this._rl.active = true;

    // P2-3：人化反应延迟触发决策（频率硬约束 ≤20Hz）
    if (this._rl.interfaceController.shouldAct(performance.now())) {
      this._rlStep();
    }
  }

  /** 单个决策步：感知 → 决策 → 执行 → 结算 → 存储/训练 */
  _rlStep() {
    const agent = this._rlEnsureAgent();
    const avatar = this.App.currentAvatar;
    if (!agent || !avatar) return;

    const stateVec = this.getObservation();
    const { action } = agent.chooseAction(stateVec, this.getValidActions());
    const prevCount = this._resourcesCollected;
    const prevDist = this._nearestResourceDist();

    // 执行移动：碰撞预检（阻挡则原地并惩罚）
    let reward = RL_REWARD.STEP;
    this.applyAction(action);
    const [dx, dz] = this._rl.moveDir;
    const now = performance.now();
    const stepSec = Math.min(0.5, Math.max(0.05, (now - this._rl.lastStepTs) / 1000));
    this._rl.lastStepTs = now;
    const newX = avatar.position.x + dx * RL_MOVE_SPEED * stepSec;
    const newZ = avatar.position.z + dz * RL_MOVE_SPEED * stepSec;
    if (this.checkCollision(newX, newZ)) {
      this._rl.hitObstacle = 1;
      reward += RL_REWARD.HIT_OBSTACLE;
    } else {
      this._rl.hitObstacle = 0;
      avatar.position.x = newX;
      avatar.position.z = newZ;
    }
    // 边界惩罚
    if (Math.abs(avatar.position.x) > RL_WORLD_BOUND || Math.abs(avatar.position.z) > RL_WORLD_BOUND) {
      reward += RL_REWARD.BOUNDARY;
    }

    // 采集奖励（本决策步内 _checkResourceCollection 已执行）
    const gained = this._resourcesCollected - prevCount;
    if (gained > 0) reward += RL_REWARD.COLLECT * gained;
    const nowDist = this._nearestResourceDist();
    if (prevDist !== Infinity && nowDist < prevDist - 0.15) reward += RL_REWARD.APPROACH;

    const nextState = this.getObservation();
    agent.store(stateVec, action, reward, nextState, false);
    agent.train();
  }

  /** 最近资源距离（无则 Infinity） */
  _nearestResourceDist() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return Infinity;
    let min = Infinity;
    for (const r of this.resources) {
      if (!r.mesh || !r.mesh.parent) continue;
      const d = Math.hypot(r.mesh.position.x - avatar.position.x, r.mesh.position.z - avatar.position.z);
      if (d < min) min = d;
    }
    return min;
  }

  cleanup() {
    // P2-2 人类限时评估基准：场景结束记录一局（win = 达到采集目标）
    if (this._rl) {
      RLAgentManager.get().getBaselineEvaluator().recordEpisode('sandbox', {
        durationMs: performance.now() - this._rl.episodeStartTs,
        win: this._resourcesCollected >= RL_REWARD.EVAL_WIN_COLLECT,
      });
      this._rl.interfaceController.reset();
    }
    this._resetWorld();
    super.cleanup();
  }

  // ==================== 引擎兼容钩子 ====================

  checkCollision(newX, newZ, options = {}) {
    const avatar = this.App.currentAvatar;
    if (!avatar) return false;

    const playerY = avatar.position.y;  // 脚底高度
    const STEP_HEIGHT = 0.9;            // 可跨过的最大步高（主角身高~1.8的一半）

    // 1) 地形碰撞：目标地面高于脚底+步高 → 阻挡
    //    腾空时（ignoreTerrain）跳过地形判定，让 AI 起跳后能在空中平移翻越矮墙，
    //    落地由 _updatePlayerPhysics 修正 Y。
    if (!options.ignoreTerrain) {
      const samples = [
        [newX, newZ],
        [newX + COLLISION_RADIUS, newZ],
        [newX - COLLISION_RADIUS, newZ],
        [newX, newZ + COLLISION_RADIUS],
        [newX, newZ - COLLISION_RADIUS],
      ];

      for (const [sx, sz] of samples) {
        const targetGround = this._getGroundHeight(sx, sz);
        if (targetGround > playerY + STEP_HEIGHT) return true;
      }
    }

    // 2) 实体碰撞：XZ平面圆对圆，Y轴也检查（脚底到头顶全量程）
    const playerTop = playerY + 1.8; // 全长碰撞，避免只有头部碰撞

    for (const r of this.resources) {
      if (!r.solid || !r.mesh.parent) continue;
      const rx = r.mesh.position.x;
      const rz = r.mesh.position.z;

      // XZ距离
      const dx = newX - rx;
      const dz = newZ - rz;
      const minDist = COLLISION_RADIUS + (RESOURCE_COLLISION_RADIUS[r.type] || 0.35);
      const newDist2 = dx * dx + dz * dz;
      if (newDist2 >= minDist * minDist) continue;

      // 允许从碰撞区移出
      const curDx = avatar.position.x - rx;
      const curDz = avatar.position.z - rz;
      const curDist2 = curDx * curDx + curDz * curDz;
      if (curDist2 <= newDist2) continue;
      if (Math.abs(curDist2 - newDist2) < 0.0001) continue;

      // Y轴：实体近似高度
      const ry = r.mesh.position.y;
      let entityTop = ry + 1.8; // 默认全高
      if (r.type === 'tree') entityTop = ry + 2.5;
      else if (r.type === 'pine_tree') entityTop = ry + 2.3;
      else if (r.type === 'rock') entityTop = ry + 0.6;
      else if (r.type === 'coal_ore' || r.type === 'iron_ore') entityTop = ry + 0.5;
      else if (r.type === 'cactus') entityTop = ry + 1.0;
      else if (r.type === 'flower' || r.type === 'berry_bush') entityTop = ry + 0.4;

      // 玩家全身范围与实体范围有交集 → 碰撞
      if (playerTop > ry && playerY < entityTop) return true;
    }

    return false;
  }

  setPlayerSpeed(speed) {
    this._lastMoveSpeed = speed;
  }

  _getPlayerSpeed() {
    return this._lastMoveSpeed || 0;
  }

  // ==================== AI 感知数据 ====================

  getExtraState() {
    return {
      inventory: { ...this.inventory },
      biome: this.currentBiome,
      biome_name: BIOME_NAMES[this.currentBiome] || this.currentBiome,
      chunk: { ...this.currentChunk },
      resources_collected: this._resourcesCollected,
    };
  }

  _getMapData() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return null;

    const size = 11; // 11x11 小地图
    const half = Math.floor(size / 2);
    const ox = Math.round(avatar.position.x / BLOCK_SIZE);
    const oz = Math.round(avatar.position.z / BLOCK_SIZE);
    const cells = [];

    for (let z = oz - half; z <= oz + half; z++) {
      const row = [];
      for (let x = ox - half; x <= ox + half; x++) {
        const col = this._getColumn(x, z);
        row.push(col
          ? { type: col.type, height: col.height, biome: col.biome }
          : { type: 'void', height: 0, biome: 'unknown' });
      }
      cells.push(row);
    }

    return {
      type: 'heightmap',
      block_size: BLOCK_SIZE,
      chunk_size: CHUNK_SIZE,
      render_distance_chunks: this.renderDistanceChunks,
      player_cell: { x: ox, z: oz },
      cells,
    };
  }

  _getObjectsData() {
    const data = {};
    for (const r of this.resources) {
      if (!r.mesh.parent) continue;
      const list = data[r.type] || (data[r.type] = []);
      list.push({
        id: `${r.displayName}_${list.length}`,
        type: r.type,
        name: r.displayName,
        x: +r.mesh.position.x.toFixed(1),
        z: +r.mesh.position.z.toFixed(1),
        amount: r.amount,
      });
    }
    return data;
  }

  _getNearbyObjects() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return [];

    const px = avatar.position.x;
    const pz = avatar.position.z;
    const facing = this.App.smoothRotY || 0;
    const nearby = [];

    // 附近资源
    for (const r of this.resources) {
      if (!r.mesh.parent) continue;
      const dx = r.mesh.position.x - px;
      const dz = r.mesh.position.z - pz;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < PERCEPTION_RANGE) {
        nearby.push({
          type: 'resource',
          subtype: r.type,
          name: r.displayName,
          item: ITEM_NAMES[r.itemType] || r.itemType,
          x: +r.mesh.position.x.toFixed(1),
          z: +r.mesh.position.z.toFixed(1),
          distance: +dist.toFixed(1),
          direction: this._relativeDir(dx, dz, facing),
          amount: r.amount,
        });
      }
    }

    // 前方地形提示
    const frontDist = 6;
    const fx = px + Math.sin(facing) * frontDist;
    const fz = pz + Math.cos(facing) * frontDist;
    const frontCol = this._getColumn(Math.round(fx), Math.round(fz));
    if (frontCol) {
      nearby.push({
        type: 'terrain_ahead',
        biome: frontCol.biome,
        biome_name: BIOME_NAMES[frontCol.biome] || frontCol.biome,
        height: frontCol.height,
        distance: frontDist,
        direction: '正前方',
      });
    }

    // 附近的动物
    for (const animal of this.animals) {
      if (!animal.mesh.parent) continue;
      const dx = animal.mesh.position.x - px;
      const dz = animal.mesh.position.z - pz;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < PERCEPTION_RANGE) {
        nearby.push({
          type: 'animal',
          animal: animal.type,
          name: animal.name,
          x: +animal.mesh.position.x.toFixed(1),
          z: +animal.mesh.position.z.toFixed(1),
          distance: +dist.toFixed(1),
          direction: this._relativeDir(dx, dz, facing),
        });
      }
    }

    nearby.sort((a, b) => a.distance - b.distance);
    return nearby.slice(0, 12);
  }

  _eventImportance(type) {
    const map = {
      game_start: 2,
      game_completed: 3,
      game_failed: 3,
      resource_collected: 2,
      biome_changed: 2,
      chunk_changed: 1,
    };
    return map[type] || 1;
  }

  // ==================== 内部：世界生成 ====================

  _ensureChunk(cx, cz) {
    const key = `${cx},${cz}`;
    if (this.chunks.has(key)) return this.chunks.get(key);

    const chunk = {
      cx,
      cz,
      columns: new Map(),
      resources: [],
      solidMesh: null,
      waterMesh: null,
    };

    const THREE = this.THREE;

    for (let lx = 0; lx < CHUNK_SIZE; lx++) {
      for (let lz = 0; lz < CHUNK_SIZE; lz++) {
        const wx = cx * CHUNK_SIZE + lx;
        const wz = cz * CHUNK_SIZE + lz;
        const column = this._generateColumn(wx, wz);
        chunk.columns.set(`${lx},${lz}`, column);
        this._trySpawnResource(chunk, wx, wz, column);
        this._trySpawnAnimal(chunk, wx, wz, column);
      }
    }

    this._buildChunkMesh(chunk);
    this.chunks.set(key, chunk);
    return chunk;
  }

  _generateColumn(wx, wz) {
    // 主噪声：决定宏观地形高度（8~10倍范围，6层FBM增加细节）
    let height = Math.round(2 + signedNoise(wx, wz, 6, 0.02, 0.5) * 5);

    // 微地形噪声：每格额外 ±1 的高度波动，让地表不再平坦
    const micro = Math.round(signedNoise(wx * 3.7 + 100, wz * 3.7 + 200, 2, 0.15, 0.6) * 1.2);
    height += micro;

    const biome = getBiome(wx, wz, height, this.seed);

    // 生态对高度的微调
    if (biome === 'mountain') height += Math.round(Math.max(0, signedNoise(wx + 999, wz + 999, 3, 0.03, 0.5)) * 5);
    if (biome === 'desert') height = Math.max(-1, height - 1);

    // 水面处理
    if (height < WATER_LEVEL) {
      return { type: 'water', height: WATER_LEVEL, biome, obstacle: null };
    }

    // 地表类型
    let type = 'grass';
    if (biome === 'desert') type = 'sand';
    else if (biome === 'snow') type = 'snow';
    else if (biome === 'mountain') type = height >= 6 ? 'snow' : 'stone';
    else if (biome === 'forest') type = 'grass';

    return { type, height, biome, obstacle: null };
  }

  _trySpawnResource(chunk, wx, wz, column) {
    if (column.type === 'water') return;

    const rand = hash2d(wx + this.seed * 2, wz + this.seed * 3);
    let type = null;

    switch (column.biome) {
      case 'forest':
        if (column.type === 'grass') {
          if (rand < 0.10) type = 'tree';
          else if (rand < 0.13) type = 'mushroom';
          else if (rand < 0.16) type = 'berry_bush';
          else if (rand < 0.20) type = 'flower';
          else if (rand < 0.26) type = 'tall_grass';
        } else if (column.type === 'stone' && rand < 0.07) {
          type = 'pine_tree';
        }
        break;
      case 'desert':
        if (column.type === 'sand') {
          if (rand < 0.05) type = 'cactus';
          else if (rand < 0.08) type = 'tall_grass';
        }
        break;
      case 'mountain':
      case 'snow':
        if (column.type === 'stone' || column.type === 'snow') {
          if (rand < 0.04) type = 'pine_tree';
          else if (rand < 0.09) type = 'rock';
          else if (rand < 0.12) type = 'coal_ore';
          else if (rand < 0.14) type = 'iron_ore';
        }
        if (column.type === 'grass' && rand < 0.06) type = 'tall_grass';
        break;
      case 'plains':
        if (rand < 0.04) type = 'flower';
        else if (rand < 0.08) type = 'tall_grass';
        else if (rand < 0.11) type = 'berry_bush';
        else if (rand < 0.13) type = 'rock';
        else if (rand < 0.15) type = 'mushroom';
        break;
      default:
        break;
    }

    if (!type || column.obstacle) return;

    const info = RESOURCE_TYPES[type];
    const pos = new this.THREE.Vector3(
      wx * BLOCK_SIZE,
      column.height * BLOCK_SIZE,
      wz * BLOCK_SIZE
    );
    const mesh = this._createResourceMesh(type, pos, info);
    const resource = {
      mesh,
      chunk,
      type,
      itemType: info.item,
      amount: info.amount,
      displayName: info.name,
      solid: info.solid,
    };
    mesh.userData.resource = resource;

    this.addToScene(mesh);
    chunk.resources.push(resource);
    this.resources.push(resource);

    if (info.solid) column.obstacle = type;
  }

  _buildChunkMesh(chunk) {
    const THREE = this.THREE;
    const solidInstances = [];
    const waterInstances = [];

    for (let lx = 0; lx < CHUNK_SIZE; lx++) {
      for (let lz = 0; lz < CHUNK_SIZE; lz++) {
        const column = chunk.columns.get(`${lx},${lz}`);
        if (!column) continue;

        const wx = chunk.cx * CHUNK_SIZE + lx;
        const wz = chunk.cz * CHUNK_SIZE + lz;

        if (column.type === 'water') {
          waterInstances.push({
            x: wx * BLOCK_SIZE,
            y: WATER_LEVEL * BLOCK_SIZE + BLOCK_SIZE * 0.5,
            z: wz * BLOCK_SIZE,
            color: BLOCK_COLORS.water,
          });
        } else {
          solidInstances.push({
            x: wx * BLOCK_SIZE,
            y: column.height * BLOCK_SIZE + BLOCK_SIZE * 0.5,
            z: wz * BLOCK_SIZE,
            color: BLOCK_COLORS[column.type] || 0x888888,
          });
        }
      }
    }

    chunk.solidMesh = this._createInstancedMesh(solidInstances, {
      roughness: 0.9,
      metalness: 0.0,
    }, false);

    chunk.waterMesh = this._createInstancedMesh(waterInstances, {
      color: BLOCK_COLORS.water,
      roughness: 0.1,
      metalness: 0.0,
      transparent: true,
      opacity: 0.7,
    }, true);
  }

  _createInstancedMesh(instances, materialOptions, isWater) {
    if (instances.length === 0) return null;
    const THREE = this.THREE;

    // 共享几何体（所有方块用同一个BoxGeometry）
    if (!this._sharedBlockGeo) {
      this._sharedBlockGeo = new THREE.BoxGeometry(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    }
    const geometry = this._sharedBlockGeo;
    // 使用共享材质池
    let material;
    if (isWater) {
      material = this._sharedMaterials.terrainWater;
    } else {
      material = this._sharedMaterials.terrainSolid;
    }
    const mesh = new THREE.InstancedMesh(geometry, material, instances.length);

    // 方块不投射阴影（数量太多，性能消耗极大）
    mesh.castShadow = false;
    mesh.receiveShadow = true;

    for (let i = 0; i < instances.length; i++) {
      const inst = instances[i];
      this._dummy.position.set(inst.x, inst.y, inst.z);
      this._dummy.rotation.set(0, 0, 0);
      this._dummy.scale.set(1, 1, 1);
      this._dummy.updateMatrix();
      mesh.setMatrixAt(i, this._dummy.matrix);
      if (!isWater) {
        mesh.setColorAt(i, new THREE.Color(inst.color));
      }
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (!isWater) mesh.instanceColor.needsUpdate = true;

    this.addToScene(mesh);
    return mesh;
  }

  _createResourceMesh(type, pos, info) {
    const THREE = this.THREE;
    const group = new THREE.Group();
    group.position.copy(pos);

    // 获取或创建共享材质（支持自发光、金属感、平面着色等选项）
    const getMat = (color, roughness = 0.7, opts = {}) => {
      const key = `${color}_${roughness}_${JSON.stringify(opts)}`;
      if (!this._sharedMaterials.resource[key]) {
        this._sharedMaterials.resource[key] = new THREE.MeshStandardMaterial({
          color, roughness, metalness: opts.metalness || 0.0,
          emissive: opts.emissive || 0x000000, emissiveIntensity: opts.emissiveIntensity || 0,
          flatShading: opts.flatShading !== undefined ? opts.flatShading : true,
        });
      }
      return this._sharedMaterials.resource[key];
    };

    switch (type) {
      case 'tree': {
        // 树干：上细下粗圆柱，更接近真实树干
        const trunk = new THREE.Mesh(
          new THREE.CylinderGeometry(0.18, 0.25, 1.3, 8),
          getMat(BLOCK_COLORS.wood, 0.9)
        );
        trunk.position.y = 0.65;
        trunk.castShadow = true;
        group.add(trunk);

        // 三层树叶：低面数多面体平面着色，更有自然不规则感
        const leafMat = getMat(BLOCK_COLORS.leaves, 0.75, { flatShading: true });
        const bottomLeaves = new THREE.Mesh(new THREE.IcosahedronGeometry(0.75, 0), leafMat);
        bottomLeaves.scale.set(1, 0.8, 1);
        bottomLeaves.position.y = 1.3;
        bottomLeaves.castShadow = true;
        group.add(bottomLeaves);

        const midLeaves = new THREE.Mesh(new THREE.IcosahedronGeometry(0.55, 0), leafMat);
        midLeaves.position.y = 1.8;
        midLeaves.castShadow = true;
        group.add(midLeaves);

        const topLeaves = new THREE.Mesh(new THREE.IcosahedronGeometry(0.35, 0), leafMat);
        topLeaves.position.y = 2.2;
        topLeaves.castShadow = true;
        group.add(topLeaves);
        break;
      }
      case 'pine_tree': {
        // 松树树干：圆柱
        const trunk = new THREE.Mesh(
          new THREE.CylinderGeometry(0.15, 0.2, 1.0, 6),
          getMat(0x8B6914, 0.85)
        );
        trunk.position.y = 0.5;
        trunk.castShadow = true;
        group.add(trunk);

        // 三层锥形松针，从大到小
        const pineMat = getMat(0x2d5a1e, 0.7, { flatShading: true });
        const sizes = [
          { r: 0.7, h: 0.8, y: 1.1 },
          { r: 0.5, h: 0.7, y: 1.6 },
          { r: 0.3, h: 0.6, y: 2.0 },
        ];
        for (const s of sizes) {
          const layer = new THREE.Mesh(new THREE.ConeGeometry(s.r, s.h, 7), pineMat);
          layer.position.y = s.y;
          layer.castShadow = true;
          group.add(layer);
        }
        break;
      }
      case 'cactus': {
        // 仙人掌主体：圆柱
        const cactusMat = getMat(info.color, 0.6);
        const body = new THREE.Mesh(
          new THREE.CylinderGeometry(0.18, 0.2, 1.3, 8),
          cactusMat
        );
        body.position.y = 0.65;
        body.castShadow = true;
        group.add(body);
        // 顶部圆帽
        const cap = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 6), cactusMat);
        cap.position.y = 1.3;
        group.add(cap);
        // 左右小枝：圆柱水平伸出后向上弯起
        for (let side = -1; side <= 1; side += 2) {
          const arm = new THREE.Mesh(
            new THREE.CylinderGeometry(0.1, 0.12, 0.45, 6),
            cactusMat
          );
          arm.position.set(side * 0.27, 0.75, 0);
          arm.rotation.z = side * Math.PI / 2.5;
          arm.castShadow = true;
          group.add(arm);
        }
        break;
      }
      case 'rock':
      case 'coal_ore':
      case 'iron_ore': {
        // 岩石：圆滑多面体（细分1次更圆滑），矿石保留主体方块
        const geo = type === 'rock'
          ? new THREE.DodecahedronGeometry(0.35, 1)
          : new THREE.BoxGeometry(0.5, 0.4, 0.5);
        const rockMat = getMat(info.color, 0.85, { flatShading: true });
        const mesh = new THREE.Mesh(geo, rockMat);
        mesh.position.y = 0.25;
        mesh.castShadow = true;
        group.add(mesh);
        // 矿石结晶：小球凸起
        if (type === 'coal_ore' || type === 'iron_ore') {
          const crystalColor = type === 'coal_ore' ? 0x222222 : 0xb87333;
          const crystalMat = getMat(crystalColor, 0.5,
            type === 'iron_ore' ? { metalness: 0.4, flatShading: true } : { flatShading: true });
          const crystalSpots = [[0.15, 0.32, 0.1], [-0.1, 0.28, -0.12], [0.05, 0.4, -0.15]];
          for (const [cx, cy, cz] of crystalSpots) {
            const crystal = new THREE.Mesh(new THREE.IcosahedronGeometry(0.08, 0), crystalMat);
            crystal.position.set(cx, cy, cz);
            group.add(crystal);
          }
        }
        // 周围小碎石：圆滑多面体
        const pebbleMat = getMat(
          type === 'coal_ore' ? 0x444444 : type === 'iron_ore' ? 0xc09060 : 0xaaaaaa,
          0.8, { flatShading: true }
        );
        for (let i = 0; i < 3; i++) {
          const angle = (i / 3) * Math.PI * 2;
          const pebble = new THREE.Mesh(
            new THREE.DodecahedronGeometry(0.06, 0),
            pebbleMat
          );
          pebble.position.set(Math.cos(angle) * 0.3, 0.06, Math.sin(angle) * 0.3);
          group.add(pebble);
        }
        break;
      }
      case 'flower': {
        // 细茎：圆柱
        const stem = new THREE.Mesh(
          new THREE.CylinderGeometry(0.025, 0.03, 0.45, 5),
          getMat(0x55aa44, 0.9)
        );
        stem.position.y = 0.22;
        group.add(stem);

        const petalMat = new THREE.MeshStandardMaterial({
          color: info.color, emissive: info.color, emissiveIntensity: 0.35, roughness: 0.5
        });
        // 五片花瓣：小球围一圈，略微压扁
        const petalR = 0.07;
        for (let i = 0; i < 5; i++) {
          const angle = (i / 5) * Math.PI * 2;
          const petal = new THREE.Mesh(new THREE.SphereGeometry(0.06, 6, 4), petalMat);
          petal.scale.y = 0.5;
          petal.position.set(Math.cos(angle) * petalR, 0.44, Math.sin(angle) * petalR);
          group.add(petal);
        }
        // 花心：黄色发光小球
        const center = new THREE.Mesh(
          new THREE.SphereGeometry(0.05, 6, 4),
          new THREE.MeshStandardMaterial({ color: 0xffff44, emissive: 0xffff44, emissiveIntensity: 0.4 })
        );
        center.position.y = 0.44;
        group.add(center);
        break;
      }
      case 'berry_bush': {
        // 浆果丛：低面数多面体叶片球簇拥
        const bushMat = getMat(0x338833, 0.7, { flatShading: true });
        const berryMat = new THREE.MeshStandardMaterial({
          color: info.color, emissive: info.color, emissiveIntensity: 0.3, roughness: 0.5
        });
        // 叶片球
        const leafPositions = [
          [0, 0.25, 0], [0.22, 0.2, 0.1], [-0.2, 0.18, -0.1],
          [0.1, 0.15, -0.22], [-0.15, 0.22, 0.18], [0, 0.32, 0],
        ];
        for (const [lx, ly, lz] of leafPositions) {
          const leaf = new THREE.Mesh(new THREE.IcosahedronGeometry(0.22, 0), bushMat);
          leaf.position.set(lx, ly, lz);
          leaf.castShadow = true;
          group.add(leaf);
        }
        // 浆果：发光小球
        for (let i = 0; i < 4; i++) {
          const angle = (i / 4) * Math.PI * 2 + 0.3;
          const berry = new THREE.Mesh(new THREE.SphereGeometry(0.06, 6, 4), berryMat);
          berry.position.set(Math.cos(angle) * 0.18, 0.28, Math.sin(angle) * 0.18);
          group.add(berry);
        }
        break;
      }
      case 'mushroom': {
        // 蘑菇：圆柱菌柄
        const stem = new THREE.Mesh(
          new THREE.CylinderGeometry(0.04, 0.05, 0.3, 6),
          getMat(0xeeddbb, 0.8)
        );
        stem.position.y = 0.15;
        group.add(stem);

        // 伞盖：半球
        const cap = new THREE.Mesh(
          new THREE.SphereGeometry(0.18, 8, 6, 0, Math.PI * 2, 0, Math.PI / 2),
          new THREE.MeshStandardMaterial({ color: info.color, roughness: 0.5, emissive: info.color, emissiveIntensity: 0.2 })
        );
        cap.position.y = 0.3;
        cap.castShadow = true;
        group.add(cap);

        // 伞盖上小白点：小球
        const dotMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 });
        for (let i = 0; i < 3; i++) {
          const dot = new THREE.Mesh(new THREE.SphereGeometry(0.025, 5, 4), dotMat);
          dot.position.set((Math.random() - 0.5) * 0.2, 0.42, (Math.random() - 0.5) * 0.2);
          group.add(dot);
        }
        break;
      }
      case 'tall_grass': {
        // 草丛：5片扁平草叶，双面材质，随机倾斜更自然
        const grassMatKey = `grass_${info.color}`;
        if (!this._sharedMaterials.resource[grassMatKey]) {
          this._sharedMaterials.resource[grassMatKey] = new THREE.MeshStandardMaterial({
            color: info.color, roughness: 0.75, side: THREE.DoubleSide, flatShading: true,
          });
        }
        const grassMat = this._sharedMaterials.resource[grassMatKey];
        for (let i = 0; i < 5; i++) {
          const blade = new THREE.Mesh(new THREE.PlaneGeometry(0.08, 0.5), grassMat);
          blade.position.set((Math.random() - 0.5) * 0.2, 0.25, (Math.random() - 0.5) * 0.2);
          blade.rotation.y = Math.random() * Math.PI;
          blade.rotation.z = (Math.random() - 0.5) * 0.3;
          group.add(blade);
        }
        break;
      }
      default:
        break;
    }

    // 资源实体投射阴影（小花/小草等太小，跳过以节省性能）
    if (type !== 'flower' && type !== 'tall_grass') {
      for (const child of group.children) child.castShadow = true;
    }
    return group;
  }

  _updateChunks(force = false) {
    const avatar = this.App.currentAvatar;
    if (!avatar) return;

    const cx = Math.floor(avatar.position.x / (CHUNK_SIZE * BLOCK_SIZE));
    const cz = Math.floor(avatar.position.z / (CHUNK_SIZE * BLOCK_SIZE));

    if (!force && cx === this.currentChunk.x && cz === this.currentChunk.z) return;
    this.currentChunk = { x: cx, z: cz };

    const desired = new Set();
    const R = this.renderDistanceChunks;
    for (let dx = -R; dx <= R; dx++) {
      for (let dz = -R; dz <= R; dz++) {
        desired.add(`${cx + dx},${cz + dz}`);
      }
    }

    // 卸载远离的区块
    for (const key of this.chunks.keys()) {
      if (!desired.has(key)) this._unloadChunk(key);
    }

    // 加载新区块
    for (const key of desired) {
      if (!this.chunks.has(key)) {
        const [ccx, ccz] = key.split(',').map(Number);
        this._ensureChunk(ccx, ccz);
      }
    }
  }

  _unloadChunk(key) {
    const chunk = this.chunks.get(key);
    if (!chunk) return;

    // 清理区块内资源（含动物标记）
    for (const r of [...chunk.resources]) {
      if (r.type === '__animal__') {
        // 找到对应的动物实例并移除
        const animal = this.animals.find(a => a.mesh === r.mesh);
        if (animal) this._removeAnimal(animal);
      } else {
        this._disposeResource(r);
      }
    }

    this._removeSceneObject(chunk.solidMesh);
    this._removeSceneObject(chunk.waterMesh);

    this.chunks.delete(key);
  }

  _findSpawnPoint() {
    // 在初始区块中寻找一块非水的草地/沙地
    for (let r = 0; r <= 8; r++) {
      for (let x = -r; x <= r; x++) {
        for (let z = -r; z <= r; z++) {
          if (Math.max(Math.abs(x), Math.abs(z)) !== r) continue;
          const col = this._getColumn(x, z);
          if (col && col.type !== 'water' && !col.obstacle) {
            return { x: x * BLOCK_SIZE, y: col.height * BLOCK_SIZE, z: z * BLOCK_SIZE };
          }
        }
      }
    }
    return { x: 0, y: 1, z: 0 };
  }

  _createEnvironment() {
    const THREE = this.THREE;
    const scene = this.App ? this.App.scene : null;
    if (!scene) return;

    // 保存原始背景色，退出时恢复
    this._originalBackground = scene.background;
    scene.background = new THREE.Color(0x87ceeb);

    // 半球光：天空色与地面色融合，让自然光更真实，替代部分环境光
    const hemi = new THREE.HemisphereLight(0x87ceeb, 0x8b7355, 0.45);
    this.addToScene(hemi);

    // 环境光：配合半球光，强度降低避免画面过平
    const ambient = new THREE.AmbientLight(0xffffff, 0.35);
    this.addToScene(ambient);

    // 主方向光（模拟太阳）：暖阳光色，强度更高
    const sun = new THREE.DirectionalLight(0xfff2dd, 1.3);
    sun.position.set(60, 100, 40);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    // 扩大阴影覆盖范围，让远处也能产生阴影
    sun.shadow.camera.left = -60;
    sun.shadow.camera.right = 60;
    sun.shadow.camera.top = 60;
    sun.shadow.camera.bottom = -60;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 200;
    sun.shadow.camera.updateProjectionMatrix();
    this.addToScene(sun);

    // 补光，暖紫色，让阴影面不过暗
    const fill = new THREE.DirectionalLight(0xddccff, 0.3);
    fill.position.set(-40, 60, -40);
    this.addToScene(fill);
  }

  _resetWorld() {
    // 清理所有动物
    for (const animal of [...this.animals]) {
      this._removeAnimal(animal);
    }
    this.animals = [];

    // 卸载所有区块
    for (const key of [...this.chunks.keys()]) {
      this._unloadChunk(key);
    }
    this.chunks.clear();
    this.resources = [];

    // 重置状态
    this.inventory = { wood: 0, stone: 0, cactus: 0, flower: 0, berry: 0, coal: 0, iron: 0 };
    this.currentBiome = 'plains';
    this.currentChunk = { x: 0, z: 0 };
    this._resourcesCollected = 0;

    // 恢复天空背景
    if (this.App && this.App.scene && this._originalBackground !== undefined) {
      this.App.scene.background = this._originalBackground;
      this._originalBackground = undefined;
    }
  }

  // ==================== 内部：地形查询 ====================

  _getColumn(wx, wz) {
    const cx = Math.floor(wx / CHUNK_SIZE);
    const cz = Math.floor(wz / CHUNK_SIZE);
    const lx = wx - cx * CHUNK_SIZE;
    const lz = wz - cz * CHUNK_SIZE;
    const chunk = this.chunks.get(`${cx},${cz}`);
    if (!chunk) return null;
    return chunk.columns.get(`${lx},${lz}`) || null;
  }

  _getGroundHeight(x, z) {
    const col = this._getColumn(Math.round(x / BLOCK_SIZE), Math.round(z / BLOCK_SIZE));
    // 区块未加载时，用相邻区块高度回退
    if (!col) {
      for (const [dx, dz] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const nearby = this._getColumn(Math.round(x / BLOCK_SIZE) + dx, Math.round(z / BLOCK_SIZE) + dz);
        if (nearby) return (Math.max(nearby.height, WATER_LEVEL) + 1) * BLOCK_SIZE;
      }
      return this._lastKnownGroundY || ((WATER_LEVEL + 1) * BLOCK_SIZE);
    }
    // 方块占据 [height, height+1]，地面 = 方块顶部
    const h = (Math.max(col.height, WATER_LEVEL) + 1) * BLOCK_SIZE;
    this._lastKnownGroundY = h;
    return h;
  }

  /**
   * 检查两个世界坐标之间是否可通行（用于AI自主寻路时的动态障碍检测）
   * @param {number} fromX, fromZ - 起点世界坐标
   * @param {number} toX, toZ - 终点世界坐标
   * @returns {boolean} 是否可通行
   */
  _isWalkableBetween(fromX, fromZ, toX, toZ) {
    const STEP_HEIGHT = 1.05;  // 与游戏碰撞检测的抬脚高度一致
    const fromCol = this._getColumn(Math.round(fromX / BLOCK_SIZE), Math.round(fromZ / BLOCK_SIZE));
    const toCol = this._getColumn(Math.round(toX / BLOCK_SIZE), Math.round(toZ / BLOCK_SIZE));

    // 任一格子不存在（未加载区块）→ 暂不可达
    if (!fromCol || !toCol) return false;

    // 水体不可通行
    if (fromCol.type === 'water' || toCol.type === 'water') return false;

    // 高度差检查：过高台阶无法跨越
    const fromGround = (Math.max(fromCol.height, WATER_LEVEL) + 1) * BLOCK_SIZE;
    const toGround = (Math.max(toCol.height, WATER_LEVEL) + 1) * BLOCK_SIZE;
    if (Math.abs(toGround - fromGround) > STEP_HEIGHT) return false;

    return true;
  }

  /**
   * 角色跳跃请求 —— 由 GameModeManager 在检测到点击/空格时调用
   * 支持二段跳：在地面时重置为2次，空中可再跳1次
   */
  requestJump() {
    if (this._jumpsLeft <= 0) return false;
    if (this.state !== 'playing') return false;

    const avatar = this.App.currentAvatar;
    if (!avatar) return false;

    const isFirstJump = this._isGrounded;
    const velocity = isFirstJump ? JUMP_VELOCITY : DOUBLE_JUMP_VELOCITY;

    this._playerVelocityY = velocity;
    this._isGrounded = false;
    this._jumpsLeft--;
    this._lastJumpTime = performance.now() / 1000;

    avatar.position.y += 0.05;

    this._pushEvent('player_jump', {
      jump_num: this._maxJumps - this._jumpsLeft,
      is_double: !isFirstJump,
    });
    return true;
  }

  // ==================== 内部：物理 ====================

  /**
   * 物理更新：重力 + 地形碰撞 + 着地检测
   * 玩家不会穿入地面以下，从高处会自然下落
   */
  _updatePlayerPhysics(avatar, dt) {
    if (!avatar) return;

    const px = avatar.position.x;
    const pz = avatar.position.z;
    const groundY = this._getGroundHeight(px, pz);

    // 判断是否在水中
    const col = this._getColumn(Math.round(px / BLOCK_SIZE), Math.round(pz / BLOCK_SIZE));
    const inWater = col && col.type === 'water';

    // 重力
    const grav = inWater ? WATER_GRAVITY : GRAVITY;
    this._playerVelocityY -= grav * dt;
    this._playerVelocityY = Math.max(this._playerVelocityY, -MAX_FALL_SPEED);

    // 水中浮力（避免沉底）
    if (inWater && this._playerVelocityY < 0) {
      this._playerVelocityY += WATER_BUOYANCY * dt;
    }

    // 更新Y位置
    avatar.position.y += this._playerVelocityY * dt;

    // 着地检测：落到地面或以下
    if (avatar.position.y <= groundY) {
      avatar.position.y = groundY;
      if (!this._isGrounded) {
        this._pushEvent('player_landed', { fall_speed: Math.abs(this._playerVelocityY).toFixed(1) });
      }
      this._isGrounded = true;
      this._playerVelocityY = 0;
      this._jumpsLeft = this._maxJumps;
    } else {
      this._isGrounded = false;
    }

    // 水中不下沉
    if (inWater && avatar.position.y < groundY) {
      avatar.position.y += (groundY - avatar.position.y) * Math.min(1, 4 * dt);
    }

    // 同步行走动画的基准高度
    if (!avatar.userData) avatar.userData = {};
    avatar.userData._baseY = this._isGrounded ? groundY : avatar.position.y;
  }

  // ==================== 内部：动物系统 ====================

  _trySpawnAnimal(chunk, wx, wz, column) {
    if (column.type === 'water') return;
    if (column.obstacle) return;
    if (this.animals.length >= MAX_ANIMALS) return;
    if (hash2d(wx + this.seed * 7, wz + this.seed * 11) > ANIMAL_SPAWN_CHANCE) return;

    // 筛选当前生物群系可用的动物
    const candidates = [];
    for (const [key, def] of Object.entries(ANIMAL_DEFS)) {
      if (def.biomes.includes(column.biome)) candidates.push(key);
    }
    if (candidates.length === 0) return;

    const type = candidates[Math.floor(hash2d(wx - this.seed, wz + this.seed * 5) * candidates.length)];
    const def = ANIMAL_DEFS[type];
    const y = column.height * BLOCK_SIZE + (def.fly ? 1.5 + hash2d(wx, wz) * 2 : 0.1);

    const mesh = this._createAnimalMesh(type, new this.THREE.Vector3(wx * BLOCK_SIZE, y, wz * BLOCK_SIZE), def);
    const animal = {
      mesh,
      chunk,
      type,
      name: def.name,
      speed: def.speed,
      fly: def.fly,
      targetX: wx * BLOCK_SIZE + (Math.random() - 0.5) * 8,
      targetZ: wz * BLOCK_SIZE + (Math.random() - 0.5) * 8,
      targetY: y,
      changeTimer: 2 + Math.random() * 4,
    };
    this.addToScene(mesh);
    chunk.resources.push({ mesh, chunk, type: '__animal__', solid: false, itemType: null, amount: 0, displayName: def.name });
    this.animals.push(animal);
  }

  _createAnimalMesh(type, pos, def) {
    const THREE = this.THREE;
    const group = new THREE.Group();
    group.position.copy(pos);

    const s = def.size;

    // 共享材质（按颜色缓存）：动物表面平滑，粗糙度略降
    const matKey = `animal_${def.color}`;
    if (!this._sharedMaterials.resource[matKey]) {
      this._sharedMaterials.resource[matKey] = new THREE.MeshStandardMaterial({ color: def.color, roughness: 0.5, flatShading: false });
    }
    const bodyMat = this._sharedMaterials.resource[matKey];

    // --- 身体 ---：球体拉伸代替方块，更圆润
    const bodyGeo = new THREE.SphereGeometry(s * 0.5, 10, 8);
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.scale.set(1, 0.7, 1.3);
    body.position.y = s * 0.45;
    body.castShadow = true;
    group.add(body);

    // --- 头 ---：球体代替方块
    const headSize = s * 0.7;
    const headGeo = new THREE.SphereGeometry(headSize * 0.55, 10, 8);
    const head = new THREE.Mesh(headGeo, bodyMat);
    head.position.set(0, s * 0.85, s * 0.5);
    head.castShadow = true;
    group.add(head);

    // --- 四条腿 ---：细圆柱代替方块
    const legGeo = new THREE.CylinderGeometry(s * 0.08, s * 0.1, s * 0.4, 5);
    const legPositions = [
      [ s * 0.25, s * 0.15,  s * 0.35],
      [-s * 0.25, s * 0.15,  s * 0.35],
      [ s * 0.25, s * 0.15, -s * 0.35],
      [-s * 0.25, s * 0.15, -s * 0.35],
    ];
    for (const [lx, ly, lz] of legPositions) {
      const leg = new THREE.Mesh(legGeo, bodyMat);
      leg.position.set(lx, ly, lz);
      leg.castShadow = true;
      group.add(leg);
    }

    // --- 耳朵（兔子、狐狸、鹿、狼、熊）：小锥形 ---
    const earedTypes = ['rabbit', 'fox', 'deer', 'wolf', 'bear'];
    if (earedTypes.includes(type)) {
      const earH = type === 'rabbit' ? s * 0.55 : s * 0.35;
      const earGeo = new THREE.ConeGeometry(s * 0.08, earH, 4);
      for (let side = -1; side <= 1; side += 2) {
        const ear = new THREE.Mesh(earGeo, bodyMat);
        ear.position.set(side * headSize * 0.3, s * 0.85 + earH * 0.5, s * 0.55);
        group.add(ear);
      }
    }

    // --- 尾巴（狐狸大尾巴）：锥形 ---
    if (type === 'fox') {
      const tailMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 });
      const tailGeo = new THREE.ConeGeometry(s * 0.18, s * 0.7, 6);
      const tail = new THREE.Mesh(tailGeo, tailMat);
      tail.position.set(0, s * 0.55, -s * 0.8);
      tail.rotation.x = 0.5;
      group.add(tail);
    }

    // --- 鹿角 ---：细圆柱分叉
    if (type === 'deer') {
      const antlerMat = new THREE.MeshStandardMaterial({ color: 0x8B6914, roughness: 0.6 });
      for (let side = -1; side <= 1; side += 2) {
        const antlerBase = new THREE.Mesh(
          new THREE.CylinderGeometry(0.02, 0.03, s * 0.35, 4),
          antlerMat
        );
        antlerBase.position.set(side * headSize * 0.25, s * 0.85 + s * 0.18, s * 0.5);
        antlerBase.rotation.z = side * 0.4;
        group.add(antlerBase);
        const antlerTip = new THREE.Mesh(
          new THREE.CylinderGeometry(0.02, 0.03, s * 0.2, 4),
          antlerMat
        );
        antlerTip.position.set(side * headSize * 0.4, s * 0.85 + s * 0.4, s * 0.55);
        antlerTip.rotation.z = side * 0.7;
        group.add(antlerTip);
      }
    }

    // --- 鸟/鹰翅膀 ---：压扁锥形
    if (def.fly) {
      const wingColor = type === 'eagle' ? 0x4a3728 : 0xffffff;
      if (!this._sharedMaterials.resource[`wing_${wingColor}`]) {
        this._sharedMaterials.resource[`wing_${wingColor}`] = new THREE.MeshStandardMaterial({ color: wingColor, roughness: 0.35 });
      }
      const wingMat = this._sharedMaterials.resource[`wing_${wingColor}`];
      const wingW = type === 'eagle' ? s * 0.7 : s * 0.5;
      const wingGeo = new THREE.ConeGeometry(wingW * 0.5, s * 0.7, 4);
      for (let side = -1; side <= 1; side += 2) {
        const wing = new THREE.Mesh(wingGeo, wingMat);
        wing.scale.set(1, 0.15, 1);
        wing.position.set(side * s * 0.45, s * 0.45, 0);
        wing.rotation.z = side * Math.PI / 2;
        group.add(wing);
      }
    }

    // --- 乌龟壳 ---：半球壳
    if (type === 'turtle') {
      const shellMat = new THREE.MeshStandardMaterial({ color: 0x556B2F, roughness: 0.6 });
      const shell = new THREE.Mesh(new THREE.SphereGeometry(s * 0.5, 8, 6, 0, Math.PI * 2, 0, Math.PI / 2), shellMat);
      shell.scale.set(1.6, 0.8, 2.0);
      shell.position.y = s * 0.45;
      shell.castShadow = true;
      group.add(shell);
    }

    // 动物身体与头部投射阴影（已在主要部件上设置，这里补全其余部件）
    for (const child of group.children) child.castShadow = true;

    return group;
  }

  _updateAnimals(dt) {
    const THREE = this.THREE;
    for (const animal of this.animals) {
      if (!animal.mesh.parent) continue;

      const px = animal.mesh.position.x;
      const pz = animal.mesh.position.z;
      const dx = animal.targetX - px;
      const dz = animal.targetZ - pz;
      const dist = Math.sqrt(dx * dx + dz * dz);

      // 到达目标或超时 → 选新目标
      animal.changeTimer -= dt;
      if (dist < 0.5 || animal.changeTimer <= 0) {
        animal.targetX = px + (Math.random() - 0.5) * 12;
        animal.targetZ = pz + (Math.random() - 0.5) * 12;
        animal.targetY = animal.fly
          ? 1.5 + Math.random() * 2.5
          : this._getGroundHeight(px, pz);
        animal.changeTimer = 3 + Math.random() * 5;
      }

      // 移动
      if (dist > 0.1) {
        const speed = animal.speed * dt;
        const nx = dx / dist;
        const nz = dz / dist;
        animal.mesh.position.x += nx * speed;
        animal.mesh.position.z += nz * speed;

        // 飞行高度平滑
        if (animal.fly) {
          animal.mesh.position.y += (animal.targetY - animal.mesh.position.y) * Math.min(1, 2 * dt);
        } else {
          animal.mesh.position.y = this._getGroundHeight(animal.mesh.position.x, animal.mesh.position.z) + 0.1;
        }

        // 面向移动方向
        animal.mesh.rotation.y = Math.atan2(nx, nz);
      }

      // 翅膀拍动
      if (animal.fly) {
        animal.mesh.children[0].position.y += Math.sin(performance.now() * 0.015 + px) * 0.003;
      }
    }
  }

  _removeAnimal(animal) {
    if (!animal) return;
    if (animal.mesh && animal.mesh.parent) animal.mesh.parent.remove(animal.mesh);
    this._disposeObject(animal.mesh);
    this.animals = this.animals.filter(a => a !== animal);
    this.sceneObjects = this.sceneObjects.filter(o => o !== animal.mesh);
  }

  // ==================== 内部：采集与事件 ====================

  _checkResourceCollection(avatar) {
    for (let i = this.resources.length - 1; i >= 0; i--) {
      const r = this.resources[i];
      if (!r.mesh.parent) continue;

      const dx = avatar.position.x - r.mesh.position.x;
      const dz = avatar.position.z - r.mesh.position.z;
      const dy = avatar.position.y - r.mesh.position.y;
      const dist = Math.sqrt(dx * dx + dz * dz);

      if (dist < PICKUP_RANGE && Math.abs(dy) < 1.5) {
        this._collectResource(r, i);
      }
    }
  }

  _collectResource(r, index) {
    // 从地面移除障碍物标记
    const wx = Math.round(r.mesh.position.x / BLOCK_SIZE);
    const wz = Math.round(r.mesh.position.z / BLOCK_SIZE);
    const col = this._getColumn(wx, wz);
    if (col && col.obstacle === r.type) col.obstacle = null;

    // 清理场景对象
    this._disposeResource(r);

    // 增加背包
    this.inventory[r.itemType] = (this.inventory[r.itemType] || 0) + r.amount;
    this._resourcesCollected++;
    this.score += 5;

    this._pushEvent('resource_collected', {
      type: r.type,
      item: r.itemType,
      item_name: ITEM_NAMES[r.itemType] || r.itemType,
      amount: r.amount,
      inventory: { ...this.inventory },
    });

    // AI 语音反馈（避免过于频繁）
    if (this._resourcesCollected <= 3 || r.type === 'iron_ore' || r.type === 'coal_ore') {
      const texts = {
        wood: '（你砍到了一些木材，可以用来生火或者做东西呢）',
        stone: '（你捡到一块石头，沉甸甸的）',
        cactus: '（小心！是仙人掌，刺有点扎手）',
        flower: '（这朵花好漂亮，你把它收藏起来了！）',
        berry: '（你摘到了一些浆果，看起来很好吃）',
        coal: '（你发现煤矿了！这是很重要的燃料）',
        iron: '（是铁矿！你打算用来打造工具呢）',
      };
      this.sendAIAction(texts[r.itemType] || `（你采集到了 ${ITEM_NAMES[r.itemType]}）`);
    }
  }

  _checkLocationEvents(avatar) {
    const wx = Math.round(avatar.position.x / BLOCK_SIZE);
    const wz = Math.round(avatar.position.z / BLOCK_SIZE);
    const col = this._getColumn(wx, wz);
    if (!col) return;

    const biome = col.biome;
    if (biome && biome !== this.currentBiome) {
      this.currentBiome = biome;
      this._pushEvent('biome_changed', {
        biome,
        biome_name: BIOME_NAMES[biome] || biome,
      });
      const texts = {
        forest: '（你面前出现了一片森林，树木好茂密！）',
        desert: '（你走进沙漠了，这里好热，沙子闪闪发光）',
        mountain: '（地势变高了，你进入了山地，要小心攀爬）',
        snow: '（你到雪原了！地面白茫茫一片，有点冷呢）',
        plains: '（你回到了开阔的草原，视野很好）',
      };
      this.sendAIAction(texts[biome] || `（你进入了${BIOME_NAMES[biome]}）`);
    }
  }

  _updateHint() {
    const invStr = Object.entries(this.inventory)
      .filter(([, count]) => count > 0)
      .map(([key, count]) => `${ITEM_NAMES[key]}:${count}`)
      .join(' ');
    this.uiHint = `🌍 ${BIOME_NAMES[this.currentBiome]} | ${invStr || '背包空空'} · 靠近资源自动采集`;
  }

  updateSceneEffects(t) {
    // 资源上下浮动、缓慢旋转，增加活力
    for (const r of this.resources) {
      if (!r.mesh.parent) continue;
      r.mesh.rotation.y += 0.02;
      r.mesh.position.y += Math.sin(t * 3 + r.mesh.position.x * 0.5) * 0.002;
    }
  }

  // ==================== 工具方法 ====================

  _disposeResource(r) {
    if (!r) return;
    if (r.mesh && r.mesh.parent) r.mesh.parent.remove(r.mesh);
    this._disposeObject(r.mesh);
    this.resources = this.resources.filter(x => x !== r);
    if (r.chunk) {
      r.chunk.resources = r.chunk.resources.filter(x => x !== r);
    }
    this.sceneObjects = this.sceneObjects.filter(o => o !== r.mesh);
  }

  _removeSceneObject(obj) {
    if (!obj) return;
    if (obj.parent) obj.parent.remove(obj);
    this._disposeObject(obj);
    this.sceneObjects = this.sceneObjects.filter(o => o !== obj);
  }

  _relativeDir(dx, dz, facing) {
    const cosA = Math.cos(-facing);
    const sinA = Math.sin(-facing);
    const rx = dx * cosA - dz * sinA;
    const rz = dx * sinA + dz * cosA;
    const deg = (Math.atan2(rx, rz) * 180 / Math.PI + 360) % 360;

    if (deg < 22.5 || deg >= 337.5) return '正前方';
    if (deg < 67.5) return '右前方';
    if (deg < 112.5) return '右方';
    if (deg < 157.5) return '右后方';
    if (deg < 202.5) return '正后方';
    if (deg < 247.5) return '左后方';
    if (deg < 292.5) return '左方';
    return '左前方';
  }
}

export default SandboxGame;
