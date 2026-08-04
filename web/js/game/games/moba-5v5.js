/* ============================================================
 * 王者峡谷 5v5 推塔 (MOBA 5v5)
 *
 * 1:1 复刻王者荣耀核心玩法：
 * - 三路兵线 + 野区 + 河道，每路 3 塔 + 2 高地塔 + 主水晶
 * - 5v5 英雄（战士/法师/刺客/射手/辅助），等级/经济/装备/技能
 * - 兵线 30 秒/波，近战/远程/炮车
 * - 防御塔攻击优先级与递增伤害
 * - 野怪：红蓝 buff、暴君、主宰
 * - bot 英雄状态机：对线/补刀/游走/gank/团战/推塔/撤退/回城
 * - 玩家操控一个英雄（avatar），其余 9 个由 bot 驱动，可切换/观战
 * - 一体双魂：玩家操控英雄，AI 伙伴感知全局战局并语音陪伴
 * ============================================================ */

import { BaseGame } from './base-game.js';
import { RLAgentManager } from '../rl/rl-agent-manager.js';
const MOBA_ACTIONS = [
  'lane_fight',    // 0 对线
  'last_hit',      // 1 补刀
  'push',          // 2 推塔
  'retreat',       // 3 撤退
  'recall',        // 4 回城
  'jungle',        // 5 打野
  'gank',          // 6 gank
  'team_fight',    // 7 团战
  'return_to_lane',// 8 回线上
];

// ==================== 地图常量 ====================
const MAP_HALF = 30;                 // 地图半边长（X/Z ∈ [-30, 30]）
const BLUE_BASE = { x: -26, z: -26 }; // 蓝方基地（左下）
const RED_BASE  = { x: 26,  z: 26  }; // 红方基地（右上）
const RIVER_Z  = 0;                  // 河道 Z 坐标
const LANE_WIDTH = 5.0;              // 路面宽度

// 三路 waypoint（蓝方→红方），bot/小兵沿此路径行进
// L 形分路：上路沿下边(Z=-25)→右边(X=25)，下路沿左边(X=-25)→上边(Z=25)，中路对角线
// 蓝方在左下时：上路在画布下方+右侧，下路在画布左侧+上方，符合玩家直觉
const LANES = {
  top: [                  // 上路：蓝方基地→沿下边 Z=-25 东进→转弯沿右边 X=25→红方基地
    { x: -22, z: -26 }, { x: -15, z: -25 }, { x: -8, z: -25 }, { x: 0, z: -25 },
    { x: 8, z: -25 }, { x: 15, z: -25 }, { x: 22, z: -25 },
    { x: 25, z: -22 }, { x: 25, z: -15 }, { x: 25, z: -8 }, { x: 25, z: 0 }, { x: 25, z: 8 }, { x: 25, z: 15 }, { x: 25, z: 22 }, { x: 25, z: 25 },
  ],
  mid: [                  // 中路：对角线
    { x: -22, z: -22 }, { x: -15, z: -15 }, { x: -8, z: -8 }, { x: 0, z: 0 },
    { x: 8, z: 8 }, { x: 15, z: 15 }, { x: 22, z: 22 },
  ],
  bottom: [               // 下路：蓝方基地→沿左边 X=-25 北上→转弯沿上边 Z=25→红方基地
    { x: -26, z: -22 }, { x: -25, z: -15 }, { x: -25, z: -8 }, { x: -25, z: 0 },
    { x: -25, z: 8 }, { x: -25, z: 15 }, { x: -25, z: 22 },
    { x: -18, z: 25 }, { x: -10, z: 25 }, { x: 0, z: 25 }, { x: 10, z: 25 }, { x: 22, z: 25 }, { x: 26, z: 25 },
  ],
};
const LANE_KEYS = ['top', 'mid', 'bottom'];

// 野怪营地按"野区方阵"布局（每个营地有明确地形分区）
// 蓝方野区（左下方块 [-25,-25]~[0,0]）：4 营地呈 2x2 方阵
// 红方野区（右上方块 [0,0]~[25,25]）：4 营地对称
// 河道：2 boss + 2 中立（河蟹、妖花）
const JUNGLE_LAYOUT = {
  // 蓝方野区 2x2 方阵（每个营地半径 ~3，距兵线 >8 避免被路过误伤）
  blue: [
    { id: 'blue_redbuff',  type: 'redbuff',  x: -19, z: -19 },  // 左下：红 buff（挨着下路）
    { id: 'blue_bluebuff', type: 'bluebuff', x: -19, z: -7  },  // 左上：蓝 buff（挨着上路）
    { id: 'blue_wolf',     type: 'wolf',     x: -8,  z: -19 },  // 右下：狼群（挨着下路后方）
    { id: 'blue_golem',    type: 'golem',    x: -8,  z: -7  },  // 右上：石巨像（靠近河道，中路打野）
  ],
  // 红方野区 2x2 方阵（与蓝方中心对称：x,z → -x,-z）
  red: [
    { id: 'red_redbuff',   type: 'redbuff',  x: 19, z: 19 },
    { id: 'red_bluebuff',  type: 'bluebuff', x: 19, z: 7 },
    { id: 'red_wolf',      type: 'wolf',     x: 8,  z: 19 },
    { id: 'red_golem',     type: 'golem',    x: 8,  z: 7 },
  ],
  // 河道中立（沿对角线分布，与河道垂直方向一致）
  river: [
    { id: 'crab',     type: 'crab',    x: 8,  z: -8 },   // 河道右下：河蟹
    { id: 'plant',    type: 'plant',   x: -8, z: 8 },    // 河道左上：妖花
    { id: 'tyrant',   type: 'tyrant',  x: 0,  z: -12 }, // 下河偏蓝方：暴君
    { id: 'overlord', type: 'overlord',x: 0,  z: 12 },  // 上河偏红方：主宰
  ],
};

// ==================== 防御塔/水晶配置 ====================
// tier: 1=外塔 2=中塔 3=高地塔(水晶) ; main=主水晶
function buildStructures() {
  const list = [];
  // 每路每方：外塔、中塔、高地塔（按沿路径距基地的累计距离放置）
  for (const lane of LANE_KEYS) {
    const path = LANES[lane];
    for (const team of ['blue', 'red']) {
      // 外塔、中塔、高地塔
      const towerPositions = towerPosAlongLane(lane, team);
      for (let i = 0; i < towerPositions.length; i++) {
        list.push({
          id: `${team}_${lane}_t${i + 1}`,
          kind: 'tower',
          team, lane, tier: i + 1,
          x: towerPositions[i].x, z: towerPositions[i].z,
          maxHp: i === 2 ? 5500 : 4500,   // 高地塔更硬，拖慢节奏
          hp: i === 2 ? 5500 : 4500,
          atk: i === 2 ? 260 : 220,
          armor: i === 2 ? 80 : 60,        // 护甲减伤
          range: 7.0, attackSpeed: 0.8,    // 略慢攻速
          attackCd: 0, target: null, consecutiveHits: 0,
        });
      }
    }
  }
  // 主水晶（基地，非常硬）
  list.push({ id: 'blue_main', kind: 'main', team: 'blue', x: BLUE_BASE.x, z: BLUE_BASE.z,
              maxHp: 8000, hp: 8000, atk: 0, armor: 100, range: 0, attackCd: 0, target: null, lane: 'base' });
  list.push({ id: 'red_main',  kind: 'main', team: 'red',  x: RED_BASE.x,  z: RED_BASE.z,
              maxHp: 8000, hp: 8000, atk: 0, armor: 100, range: 0, attackCd: 0, target: null, lane: 'base' });
  return list;
}

/** 沿路放置每方 3 座塔（高地塔贴近基地、中塔居中、外塔偏前） */
function towerPosAlongLane(lane, team) {
  const path = LANES[lane];
  // 蓝方从 path[0] 出发走向 path[last]；红方反向
  // 高地塔：距基地约 4 米（紧贴基地）
  // 中塔：距基地约 12 米
  // 外塔：距基地约 20 米（路径中段偏己方）
  const offsets = [4, 12, 20]; // 高地、中、外塔沿路径累计距离
  const positions = [];
  if (team === 'blue') {
    // 从 path[0]（蓝方基地端）沿路径累计距离取点
    for (const off of offsets) {
      positions.push(_pointAlongPath(path, off));
    }
  } else {
    // 红方从 path[last] 反向取点
    const revPath = [...path].reverse();
    for (const off of offsets) {
      positions.push(_pointAlongPath(revPath, off));
    }
  }
  return positions;
}

/** 在路径上按累计距离取点（超出路径长度时返回末端） */
function _pointAlongPath(path, dist) {
  let acc = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i + 1];
    const seg = Math.hypot(b.x - a.x, b.z - a.z);
    if (acc + seg >= dist) {
      const t = (dist - acc) / seg;
      return { x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t };
    }
    acc += seg;
  }
  return { ...path[path.length - 1] };
}

// ==================== 野怪配置 ====================
// 基于 JUNGLE_LAYOUT 生成，每个营地数值按 type 配置
function buildJungle() {
  // 类型 → 属性模板
  const typeStats = {
    redbuff:  { maxHp: 1800, atk: 80,  range: 1.8, buff: 'red' },
    bluebuff: { maxHp: 1800, atk: 80,  range: 1.8, buff: 'blue' },
    wolf:     { maxHp: 900,  atk: 55,  range: 1.6, buff: null },
    golem:    { maxHp: 2400, atk: 95,  range: 1.8, buff: null },
    raptor:   { maxHp: 700,  atk: 50,  range: 1.5, buff: null },
    crab:     { maxHp: 1200, atk: 50,  range: 1.5, buff: null },
    plant:    { maxHp: 800,  atk: 40,  range: 2.0, buff: null },
    tyrant:   { maxHp: 4500, atk: 120, range: 2.5, buff: 'tyrant' },
    overlord: { maxHp: 6500, atk: 150, range: 2.5, buff: 'overlord' },
  };
  const list = [];
  // 蓝方 + 红方野区
  for (const team of ['blue', 'red']) {
    for (const camp of JUNGLE_LAYOUT[team]) {
      const s = typeStats[camp.type];
      list.push({
        id: camp.id, type: camp.type, kind: 'jungle', team,
        x: camp.x, z: camp.z,
        maxHp: s.maxHp, hp: s.maxHp, atk: s.atk, range: s.range,
        attackCd: 0, respawn: 0, buff: s.buff,
      });
    }
  }
  // 河道中立
  for (const camp of JUNGLE_LAYOUT.river) {
    const s = typeStats[camp.type];
    list.push({
      id: camp.id, type: camp.type, kind: 'jungle', team: 'neutral',
      x: camp.x, z: camp.z,
      maxHp: s.maxHp, hp: s.maxHp, atk: s.atk, range: s.range,
      attackCd: 0, respawn: 0, buff: s.buff,
    });
  }
  return list;
}

// ==================== 英雄定义 ====================
const HERO_DEFS = {
  warrior:  { name: '重装战士', role: 'fighter',   hp: 2800, mp: 100, atk: 150, armor: 80,  mr: 55, speed: 3.8, range: 2.5, color: 0xcc4444, skill1Name: '重击', skill2Name: '冲撞', ultName: '狂战之怒' },
  mage:     { name: '秘术法师', role: 'mage',      hp: 2100, mp: 240, atk: 140, armor: 45,  mr: 65, speed: 3.6, range: 5.0, color: 0x9966ff, skill1Name: '火球术', skill2Name: '冰霜新星', ultName: '陨石坠落' },
  assassin: { name: '影刃刺客', role: 'assassin',  hp: 2300, mp: 120, atk: 160, armor: 55,  mr: 55, speed: 4.3, range: 2.0, color: 0x44cccc, skill1Name: '影袭', skill2Name: '瞬步', ultName: '死亡印记' },
  marksman: { name: '神射手',   role: 'marksman',  hp: 2000, mp: 100, atk: 155, armor: 38,  mr: 50, speed: 3.6, range: 6.0, color: 0xffaa33, skill1Name: '穿甲箭', skill2Name: '翻滚', ultName: '箭雨风暴' },
  support:  { name: '守护辅助', role: 'support',   hp: 2500, mp: 200, atk: 130, armor: 65,  mr: 70, speed: 3.7, range: 4.5, color: 0x44dd66, skill1Name: '治疗术', skill2Name: '护盾', ultName: '圣光庇护' },
};
const HERO_KEYS = ['warrior', 'mage', 'assassin', 'marksman', 'support'];

// 双方阵容（5个职业）
const COMPOSITION = [...HERO_KEYS];

// ==================== 装备（简化为自动升级属性） ====================
// 金币达到阈值自动购买，每件提供属性加成
const EQUIPMENT_TIERS = [
  { cost: 800,  bonus: { atk: 25, hp: 200 } },
  { cost: 1800, bonus: { atk: 30, hp: 300, armor: 15 } },
  { cost: 3200, bonus: { atk: 45, hp: 500, armor: 25, mr: 15 } },
  { cost: 5000, bonus: { atk: 60, hp: 700, armor: 35, mr: 25 } },
];

// ==================== 数值常量 ====================
const MINION_INTERVAL = 30;      // 兵线间隔（秒）
const MINION_SPAWN_DELAY = 1.5;  // 出兵延迟（开局）
const MINION_SPEED = 3.2;
const RESPAWN_BASE = 8;          // 复活基础时间
const RESPAWN_PER_LEVEL = 2;     // 每级增加
const MAX_LEVEL = 15;
const RECALL_TIME = 4;           // 回城读条
const BOT_DECISION_INTERVAL = 0.5;
const SKILL_RANGE_BONUS = 1.0;
const TOWER_DAMAGE_RAMP = 0.25;  // 塔连续命中递增
const BUFF_DURATION = 90;        // buff 持续
const JUNGLE_RESPAWN = 90;

// 经验/金币奖励
const REWARD = {
  minion_kill: { gold: 40, exp: 12 },
  jungle_kill: { gold: 50, exp: 20 },
  hero_kill:   { gold: 200, exp: 80 },
  tower_kill:  { gold: 150, exp: 50 },
  assist:      { gold: 120, exp: 50 },
};
// 被动经济/经验（每秒）
const PASSIVE_GOLD_PER_SEC = 5;   // 每秒 5 金币
const PASSIVE_EXP_PER_SEC = 8;    // 每秒 8 经验

// ==================== 强化学习奖励配置（Q-Learning） ====================
// 每个事件给予执行者/受影响者累积奖励，决策步结算时用于 TD 更新
const RL_REWARD = {
  hero_kill:    +10,   // 击杀敌方英雄（执行者）
  assist:       +5,    // 助攻（附近友方）
  death:        -8,    // 自己阵亡
  tower_kill:   +15,   // 推掉敌方塔（参与击杀者）
  tower_lost:   -12,   // 己方塔被推（全队分摊）
  jungle_kill:  +3,    // 击杀野怪
  minion_kill:  +1,    // 补刀小兵
  main_dmg:     +0.02, // 对敌方主水晶造成伤害（按比例）
  main_hurt:    -0.02, // 己方主水晶被伤（按比例）
  win:          +50,   // 局终胜利
  lose:         -50,   // 局终失败
  step:         -0.05, // 时间惩罚（鼓励主动结束）
};
// 加速训练档位（每帧逻辑步数倍率）
const RL_SPEED_PRESETS = [1, 3, 10, 25, 60];

// ==================== 主类 ====================
export class MobaGame extends BaseGame {
  constructor(app) {
    super(app);
    this.name = 'moba_5v5';
    this.displayName = '王者峡谷 5v5';
    this.description = '5v5 推塔对战：三路兵线、野区、防御塔，复刻王者荣耀核心玩法。AI 英雄具备完整对线/gank/团战智能。';
    this.moveSpeed = 3.7;
    this.initialCameraRadius = 15;
    this.initialCameraHeight = 15;
    this.boundarySize = MAP_HALF * 2;

    // 游戏数据
    this.structures = [];      // 塔/水晶
    this.jungle = [];          // 野怪
    this.heroes = [];          // 10 个英雄
    this.minions = [];         // 小兵
    this.projectiles = [];     // 投射物
    this.effects = [];         // 特效（技能命中、死亡爆炸等）
    this.killFeed = [];        // 击杀提示
    this.teamGold = { blue: 0, red: 0 };
    this.teamKills = { blue: 0, red: 0 };

    // 玩家
    this.playerHeroId = 'blue_0';   // 玩家操控的英雄 id
    this.spectating = false;        // 观战模式
    this._skillInputQueue = [];     // 玩家技能输入队列

    // 计时
    this._gameTime = 0;
    this._nextMinionWave = MINION_SPAWN_DELAY;
    this._botDecisionTimer = 0;

    // 共享资源
    this._sharedMaterials = null;
    this._sharedGeometries = null;
    this._healthBarSprites = [];

    // 玩家移动速度记录
    this._lastMoveSpeed = 0;

    // 小地图观察模式：点击/拖拽小地图临时查看对应区域
    this._minimapFocus = null;     // {x, z} 世界坐标，null 表示跟随 avatar
    this._minimapFocusTimer = 0;   // 观察剩余时间（秒），>0 时锁定相机，归零后自动返回
    this._minimapDragging = false; // 是否正在拖拽小地图
    // 物理（MOBA 是俯视平面，不需要跳跃）
    this._groundY = 0;

    this.uiHint = '';

    // ==================== 强化学习状态 ====================
    this._rl = null;                 // 旧版 MobaRLAgent（不再使用，保留为 null）
    this._rlEnabled = false;         // RL 是否接管 bot 决策
    this._rlSpeedIdx = 0;            // 加速档位索引（RL_SPEED_PRESETS）
    this._rlAutoRestart = false;     // 加速自我对弈：局终自动重开
    this._rlAutoRestartDelay = 0;    // 自动重开倒计时
    this._rlEpisodeReward = 0;       // 本局累计奖励（所有bot求和，用于统计）
    this._rlLastMainHp = { blue: 0, red: 0 }; // 主水晶 HP 快照（计算伤害增量奖励）
    this._rlPrevMainHpSet = false;

    // ==================== LLM 宏观策略层（LLM-as-policy）====================
    this._llmPolicyEnabled = false;   // LLM 策略层是否启用
    this._llmStrategy = null;         // 当前宏观策略名
    this._llmRewardBias = null;       // {actionIdx: bonus} 策略→动作偏置
    this._llmPolicyTimer = 0;         // 决策计时器（真实秒）
    this._llmPolicyInterval = 6;      // 决策间隔(秒)
    this._llmAccumReward = 0;         // 上次决策后累积奖励（回传给记忆库）
    this._llmWaitingResponse = false; // 是否等待 LLM 响应
    this._llmStats = { requests: 0, responses: 0 };
  }

  // ==================== 生命周期 ====================

  generateScene() {
    const THREE = this.THREE;
    const scene = this.App.scene;

    // 共享资源
    this._sharedMaterials = {
      // 科技风材质：深色金属地面 + 发光能量道路 + 能量流河道
      ground:   new THREE.MeshStandardMaterial({ color: 0x1a1f2e, roughness: 0.4, metalness: 0.7 }),
      lane:     new THREE.MeshStandardMaterial({ color: 0x2a3a5a, roughness: 0.3, metalness: 0.5, emissive: 0x113355, emissiveIntensity: 0.5 }),
      river:    new THREE.MeshStandardMaterial({ color: 0x00ccff, roughness: 0.15, metalness: 0.3, transparent: true, opacity: 0.75, emissive: 0x00aaff, emissiveIntensity: 0.6 }),
      blue:     new THREE.MeshStandardMaterial({ color: 0x3a6bff, roughness: 0.3, metalness: 0.6, emissive: 0x1a3aff, emissiveIntensity: 0.5 }),
      red:      new THREE.MeshStandardMaterial({ color: 0xff3a3a, roughness: 0.3, metalness: 0.6, emissive: 0xff2a2a, emissiveIntensity: 0.5 }),
      neutral:  new THREE.MeshStandardMaterial({ color: 0x99aa44, roughness: 0.4, metalness: 0.5, emissive: 0x556622, emissiveIntensity: 0.3 }),
      towerBody:new THREE.MeshStandardMaterial({ color: 0x3a4258, roughness: 0.3, metalness: 0.85 }),
      crystal:  new THREE.MeshStandardMaterial({ color: 0x66ddff, roughness: 0.05, metalness: 0.9, emissive: 0x22aacc, emissiveIntensity: 0.8, transparent: true, opacity: 0.9 }),
      mainCrystal: new THREE.MeshStandardMaterial({ color: 0xffee88, roughness: 0.05, metalness: 0.9, emissive: 0xffaa22, emissiveIntensity: 0.9, transparent: true, opacity: 0.9 }),
      // 科技网格线（地面装饰）
      techGrid: new THREE.MeshBasicMaterial({ color: 0x2a4a8a, transparent: true, opacity: 0.15 }),
      // 能量核心（塔顶/野怪核心）
      energyCore: new THREE.MeshStandardMaterial({ color: 0x88ddff, emissive: 0x44aaff, emissiveIntensity: 0.9, roughness: 0.1, metalness: 0.5, transparent: true, opacity: 0.85 }),
    };
    this._sharedGeometries = {
      // 英雄/小兵几何体提高段数，新增科技风部件
      heroBody: new THREE.CapsuleGeometry(0.4, 0.9, 8, 16),
      heroHead: new THREE.SphereGeometry(0.3, 16, 12),
      heroChest: new THREE.BoxGeometry(0.6, 0.5, 0.35),
      // 职业差异化部件
      heroShield: new THREE.BoxGeometry(0.5, 0.7, 0.08),       // 战士盾牌
      heroStaff:  new THREE.CylinderGeometry(0.04, 0.04, 1.6, 6), // 法师法杖
      heroCape:   new THREE.PlaneGeometry(0.7, 0.9),            // 刺客披风
      heroBow:    new THREE.TorusGeometry(0.35, 0.04, 6, 12, Math.PI), // 射手弓
      heroHalo:   new THREE.TorusGeometry(0.45, 0.03, 6, 20),   // 辅助光环
      // 小兵差异化部件
      minionMelee: new THREE.BoxGeometry(0.5, 0.6, 0.5),
      minionRanged:new THREE.BoxGeometry(0.4, 0.5, 0.4),
      minionCannon: new THREE.BoxGeometry(0.7, 0.7, 0.7),
      minionShield: new THREE.BoxGeometry(0.45, 0.5, 0.06),    // 近战盾
      minionCannonBarrel: new THREE.CylinderGeometry(0.1, 0.1, 0.6, 8), // 炮车炮管
      minionBook: new THREE.BoxGeometry(0.18, 0.22, 0.04),      // 远程法术书
      tower: new THREE.CylinderGeometry(0.7, 1.0, 5.5, 8),
      towerTop: new THREE.CylinderGeometry(1.2, 0.8, 0.6, 8),
      // 科技塔六边形底座
      towerHex: new THREE.CylinderGeometry(1.4, 1.6, 0.4, 6),
      // 能量核心多面体
      energyCoreGeo: new THREE.IcosahedronGeometry(0.5, 0),
      // 装饰环
      ringThin: new THREE.TorusGeometry(0.9, 0.04, 8, 24),
      crystal: new THREE.OctahedronGeometry(0.7, 0),
      mainCrystal: new THREE.OctahedronGeometry(1.5, 0),
      jungle: new THREE.DodecahedronGeometry(0.6, 0),
      // 野怪类型差异化几何体
      jungleWolf:  new THREE.ConeGeometry(0.45, 0.9, 5),        // 狼：锥形
      jungleGolem: new THREE.DodecahedronGeometry(0.75, 0),    // 巨像：大型十二面体
      jungleRaptor:new THREE.TetrahedronGeometry(0.55, 0),     // 食肉鸟：四面体
      jungleCrab:  new THREE.BoxGeometry(0.7, 0.3, 0.6),        // 河蟹：扁平箱
      jungleDragon:new THREE.ConeGeometry(0.85, 1.4, 6),       // 龙：高锥
      junglePlant: new THREE.IcosahedronGeometry(0.5, 0),      // 妖花：二十面体
      // 能量弹提高段数
      projectile: new THREE.SphereGeometry(0.15, 12, 8),
    };

    // 光照与天空
    this._createEnvironment();

    // 地形
    this._buildTerrain();

    // 塔/水晶
    this.structures = buildStructures();
    this._buildStructureMeshes();

    // 野怪
    this.jungle = buildJungle();
    this._buildJungleMeshes();

    // 英雄
    this._spawnHeroes();

    // 出生玩家
    this._spawnPlayer();
  }

  onStart() {
    super.onStart();
    this._gameTime = 0;
    this._nextMinionWave = MINION_SPAWN_DELAY;
    this.uiHint = '';
    this._rlPrevMainHpSet = false;   // 重置主水晶快照
    this._rlResetEpisode();          // 重置每局 RL 跟踪
    this._pushEvent('match_start', { composition: COMPOSITION });
    this._createMobaUI();
    this._bindKeyboard();
    // 同步玩家英雄移动速度
    this._syncPlayerMoveSpeed();
    if (this.App.sendAIAction) {
      this.App.sendAIAction('（欢迎来到王者峡谷！5v5 推塔对战开始了，我方是蓝方。我们一起推塔拿人头，目标是摧毁敌方水晶！加油！）');
    }
  }

  /** 同步玩家英雄移动速度到 controlBridge */
  _syncPlayerMoveSpeed() {
    const h = this.heroes.find(h => h.id === this.playerHeroId);
    if (!h) return;
    const mgr = this.App.gameModeManager;
    if (mgr && mgr.controlBridge && mgr.controlBridge.setSpeed) {
      mgr.controlBridge.setSpeed(h.moveSpeed);
    }
  }

  update(dt) {
    super.update(dt);
    if (this.state !== 'playing') {
      this._stepAutoRestart(dt);
      return;
    }
    // LLM 宏观策略层：按真实时间定时请求决策（不受加速影响）
    this._llmStepPolicy(dt);
    // 加速自我对弈：单帧多步逻辑，仅最后一帧更新 UI
    const speed = RL_SPEED_PRESETS[this._rlSpeedIdx] || 1;
    for (let i = 0; i < speed; i++) {
      if (this.state !== 'playing') break;
      this._stepLogic(dt);
    }
    if (this.state === 'playing') this._updateMobaUI();
    // 保存本帧 dt 供 updateSceneEffects 使用（小地图观察相机递减计时）
    this._lastDt = dt;
    // 玩家移动时取消小地图观察，立即返回跟随 avatar
    if (this._minimapFocus && !this._minimapDragging && this._lastMoveSpeed > 0.5) {
      this._cancelMinimapFocusOnMove();
    }
  }

  /** 单步逻辑（与渲染解耦，加速训练时多次调用） */
  _stepLogic(dt) {
    this._gameTime += dt;
    this._updateMinionSpawning(dt);
    this._updateMinions(dt);
    this._updateJungle(dt);
    this._updateTowers(dt);
    this._updateHeroes(dt);
    this._updateProjectiles(dt);
    this._updateEffects(dt);
    this._updateKillFeed(dt);
    this._botDecisionTimer += dt;
    if (this._botDecisionTimer >= BOT_DECISION_INTERVAL) {
      this._botDecisionTimer = 0;
      this._runBotDecisions();
    }
    this._processPlayerSkillInput();
    // RL：主水晶伤害增量奖励
    this._rlStepMainHpReward();
    this._checkVictory();
  }

  /** 自动重开倒计时（加速训练） */
  _stepAutoRestart(dt) {
    if (!this._rlAutoRestart || this.state !== 'completed') return;
    this._rlAutoRestartDelay -= dt;
    if (this._rlAutoRestartDelay <= 0) this._rlRestartMatch();
  }

  _isFastMode() {
    return (RL_SPEED_PRESETS[this._rlSpeedIdx] || 1) > 1;
  }

  cleanup() {
    this._unbindKeyboard();
    // 退出前持久化 Q 表
    if (this._rlShared) this._rlShared.agent.flush();
    // 移除弹窗拖动的 document 级监听器
    if (this._rlDragCleanup) { this._rlDragCleanup(); this._rlDragCleanup = null; }
    this.structures = [];
    this.jungle = [];
    this.heroes = [];
    this.minions = [];
    this.projectiles = [];
    this.effects = [];
    this.killFeed = [];
    this._healthBarSprites = [];
    if (this._sharedMaterials) {
      for (const k in this._sharedMaterials) this._sharedMaterials[k].dispose();
      this._sharedMaterials = null;
    }
    if (this._sharedGeometries) {
      for (const k in this._sharedGeometries) this._sharedGeometries[k].dispose();
      this._sharedGeometries = null;
    }
    if (this._mobaUI && this._mobaUI.parentNode) {
      this._mobaUI.parentNode.removeChild(this._mobaUI);
    }
    this._mobaUI = null;
    // 恢复天空与雾
    if (this.App && this.App.scene && this._originalBackground !== undefined) {
      this.App.scene.background = this._originalBackground;
      this._originalBackground = undefined;
    }
    if (this.App && this.App.scene && this._originalFog !== undefined) {
      this.App.scene.fog = this._originalFog;
      this._originalFog = undefined;
    }
    super.cleanup();
  }

  // ==================== 场景构建 ====================

  _createEnvironment() {
    const THREE = this.THREE;
    const scene = this.App.scene;
    this._originalBackground = scene.background;
    this._originalFog = scene.fog;
    // 深邃科技夜空色 + 指数雾营造深度感
    scene.background = new THREE.Color(0x0a0e1a);
    scene.fog = new THREE.FogExp2(0x0a0e1a, 0.012);

    // 偏冷环境光
    const ambient = new THREE.AmbientLight(0x445588, 0.5);
    this.addToScene(ambient);
    // 冷白月光主光（保留阴影）
    const sun = new THREE.DirectionalLight(0xddeeff, 0.8);
    sun.position.set(40, 80, 30);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -40; sun.shadow.camera.right = 40;
    sun.shadow.camera.top = 40; sun.shadow.camera.bottom = -40;
    sun.shadow.camera.far = 200;
    this.addToScene(sun);
    // 半球光：冷色天空 + 深色地面
    const hemi = new THREE.HemisphereLight(0x335599, 0x0a0e1a, 0.4);
    this.addToScene(hemi);
    // 蓝方基地蓝色点光源
    const blueLight = new THREE.PointLight(0x3a6bff, 0.6, 30);
    blueLight.position.set(BLUE_BASE.x, 6, BLUE_BASE.z);
    this.addToScene(blueLight);
    // 红方基地红色点光源
    const redLight = new THREE.PointLight(0xff3a3a, 0.6, 30);
    redLight.position.set(RED_BASE.x, 6, RED_BASE.z);
    this.addToScene(redLight);
  }

  _buildTerrain() {
    const THREE = this.THREE;
    // 地面基底（深色金属）
    const groundGeo = new THREE.PlaneGeometry(MAP_HALF * 2, MAP_HALF * 2);
    const ground = new THREE.Mesh(groundGeo, this._sharedMaterials.ground);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.addToScene(ground);

    // 科技网格线（叠在地面上方，透明度低）
    const grid = new THREE.GridHelper(MAP_HALF * 2, 30, 0x2a4a8a, 0x1a2a4a);
    grid.position.y = 0.01;
    if (grid.material) {
      // GridHelper 可能是单一材质或材质数组
      const setOpacity = (m) => { m.transparent = true; m.opacity = 0.18; };
      if (Array.isArray(grid.material)) grid.material.forEach(setOpacity);
      else setOpacity(grid.material);
    }
    this.addToScene(grid);

    // 三条路（用平面条带表示，按 lane 不同颜色）
    for (const lane of LANE_KEYS) {
      this._buildLaneMesh(LANES[lane], lane);
    }

    // 野区地形分区：蓝方/红方各一块半透明色块，让野怪营地有明确区域感
    this._buildJungleTerrain();

    // 河道（与中路对角线垂直，沿 (1,-1) 方向延伸）
    const riverGeo = new THREE.PlaneGeometry(MAP_HALF * 2, 4);
    const river = new THREE.Mesh(riverGeo, this._sharedMaterials.river);
    river.rotation.x = -Math.PI / 2;   // 平铺到地面
    river.rotation.y = Math.PI / 4;    // 绕 Y 轴 45°，长边从 X 轴转到 (1,-1) 方向，垂直于中路 (1,1)
    river.position.set(0, 0.02, RIVER_Z);
    this.addToScene(river);

    // 基地区域标记：发光六边形平台
    for (const base of [BLUE_BASE, RED_BASE]) {
      const isBlue = base === BLUE_BASE;
      const baseGeo = new THREE.CylinderGeometry(4, 4, 0.3, 6);
      const baseMat = new THREE.MeshStandardMaterial({
        color: isBlue ? 0x224488 : 0x882222, roughness: 0.3, metalness: 0.8,
        emissive: isBlue ? 0x1a3aff : 0xff2a2a, emissiveIntensity: 0.6,
      });
      const baseMesh = new THREE.Mesh(baseGeo, baseMat);
      baseMesh.position.set(base.x, 0.15, base.z);
      baseMesh.receiveShadow = true;
      this.addToScene(baseMesh);
    }

    // 野区装饰（科技发光圆盘）
    const bushMat = new THREE.MeshStandardMaterial({ color: 0x1a3a3a, roughness: 0.5, metalness: 0.5, emissive: 0x0a6666, emissiveIntensity: 0.3 });
    const bushGeo = new THREE.CircleGeometry(2, 12);
    const bushPositions = [
      { x: -16, z: -6 }, { x: -6, z: -16 }, { x: 16, z: 6 }, { x: 6, z: 16 },
      { x: -10, z: 3 }, { x: 3, z: -10 }, { x: 10, z: -3 }, { x: -3, z: 10 },
    ];
    for (const p of bushPositions) {
      const bush = new THREE.Mesh(bushGeo, bushMat);
      bush.rotation.x = -Math.PI / 2;
      bush.position.set(p.x, 0.05, p.z);
      bush.receiveShadow = true;
      this.addToScene(bush);
    }
  }

  _buildLaneMesh(path, laneKey) {
    const THREE = this.THREE;
    // 每条路用不同的发光色，便于在小地图和 3D 场景中区分
    const laneColors = { top: 0x2a5a8a, mid: 0x4a8a4a, bottom: 0x8a6a2a };
    const laneEmissive = { top: 0x1a3a6a, mid: 0x2a5a2a, bottom: 0x5a3a1a };
    const baseColor = laneColors[laneKey] || 0x2a3a5a;
    const emissive = laneEmissive[laneKey] || 0x113355;
    for (let i = 0; i < path.length - 1; i++) {
      const a = path[i], b = path[i + 1];
      const dx = b.x - a.x, dz = b.z - a.z;
      const len = Math.hypot(dx, dz);
      if (len < 0.01) continue;
      const segGeo = new THREE.PlaneGeometry(LANE_WIDTH, len);
      const segMat = new THREE.MeshStandardMaterial({
        color: baseColor, roughness: 0.4, metalness: 0.4,
        emissive, emissiveIntensity: 0.45, transparent: true, opacity: 0.85,
      });
      const seg = new THREE.Mesh(segGeo, segMat);
      seg.position.set((a.x + b.x) / 2, 0.01, (a.z + b.z) / 2);
      seg.rotation.set(-Math.PI / 2, Math.atan2(dx, dz), 0);
      seg.receiveShadow = true;
      this.addToScene(seg);
      // 路侧发光边线（两条细条带，让分路边界更清晰）
      const sideOffset = LANE_WIDTH * 0.5;
      const nx = -dz / len, nz = dx / len; // 法线
      for (const sign of [-1, 1]) {
        const edgeGeo = new THREE.PlaneGeometry(0.2, len);
        const edgeMat = new THREE.MeshBasicMaterial({ color: baseColor, transparent: true, opacity: 0.8 });
        const edge = new THREE.Mesh(edgeGeo, edgeMat);
        edge.position.set((a.x + b.x) / 2 + nx * sideOffset * sign, 0.02, (a.z + b.z) / 2 + nz * sideOffset * sign);
        edge.rotation.set(-Math.PI / 2, Math.atan2(dx, dz), 0);
        this.addToScene(edge);
      }
    }
  }

  /** 野区地形分区：蓝方/红方各一块半透明色块，河道区域用深色 */
  _buildJungleTerrain() {
    const THREE = this.THREE;
    // 蓝方野区：左下方块 (-25,-25)~(0,0)
    const blueJungleGeo = new THREE.PlaneGeometry(25, 25);
    const blueJungleMat = new THREE.MeshStandardMaterial({
      color: 0x1a3a5a, roughness: 0.6, metalness: 0.2,
      emissive: 0x0a1a3a, emissiveIntensity: 0.25, transparent: true, opacity: 0.5,
    });
    const blueJungle = new THREE.Mesh(blueJungleGeo, blueJungleMat);
    blueJungle.rotation.x = -Math.PI / 2;
    blueJungle.position.set(-12.5, 0.005, -12.5);
    blueJungle.receiveShadow = true;
    this.addToScene(blueJungle);
    // 红方野区：右上方块 (0,0)~(25,25)
    const redJungleMat = new THREE.MeshStandardMaterial({
      color: 0x5a1a1a, roughness: 0.6, metalness: 0.2,
      emissive: 0x3a0a0a, emissiveIntensity: 0.25, transparent: true, opacity: 0.5,
    });
    const redJungle = new THREE.Mesh(blueJungleGeo, redJungleMat);
    redJungle.rotation.x = -Math.PI / 2;
    redJungle.position.set(12.5, 0.005, 12.5);
    redJungle.receiveShadow = true;
    this.addToScene(redJungle);
    // 河道区域：沿对角线方向的条带（与河道 mesh 重合但更宽、更暗）
    const riverAreaGeo = new THREE.PlaneGeometry(MAP_HALF * 2 * 1.414, 8);
    const riverAreaMat = new THREE.MeshStandardMaterial({
      color: 0x0a2030, roughness: 0.3, metalness: 0.5,
      emissive: 0x001020, emissiveIntensity: 0.3, transparent: true, opacity: 0.4,
    });
    const riverArea = new THREE.Mesh(riverAreaGeo, riverAreaMat);
    riverArea.rotation.x = -Math.PI / 2;
    riverArea.rotation.y = Math.PI / 4;
    riverArea.position.set(0, 0.003, 0);
    this.addToScene(riverArea);
    // 营地小圆盘（每个野怪脚下加一个发光营地标记）
    for (const j of this.jungle) {
      const campGeo = new THREE.CircleGeometry(2.2, 24);
      const isBoss = j.type === 'tyrant' || j.type === 'overlord';
      const campColor = isBoss ? 0xffaa22 : (j.team === 'blue' ? 0x1a4a8a : (j.team === 'red' ? 0x8a1a1a : 0x444444));
      const campMat = new THREE.MeshBasicMaterial({
        color: campColor, transparent: true, opacity: 0.35, side: THREE.DoubleSide,
      });
      const camp = new THREE.Mesh(campGeo, campMat);
      camp.rotation.x = -Math.PI / 2;
      camp.position.set(j.x, 0.015, j.z);
      this.addToScene(camp);
    }
  }

  _buildStructureMeshes() {
    const THREE = this.THREE;
    for (const s of this.structures) {
      const group = new THREE.Group();
      group.position.set(s.x, 0, s.z);

      if (s.kind === 'tower') {
        // 科技塔：六边形金属底座 + 中央立柱 + 能量核心 + 装饰环 + 队伍色顶盖
        // 按 tier 差异化：tier1 外塔最矮；tier2 中塔居中；tier3 高地塔最高且带水晶
        const tierScale = s.tier === 1 ? 0.85 : s.tier === 2 ? 1.0 : 1.2;
        const hex = new THREE.Mesh(this._sharedGeometries.towerHex, this._sharedMaterials.towerBody);
        hex.scale.set(tierScale, 1, tierScale);
        hex.position.y = 0.2;
        hex.castShadow = true;
        hex.receiveShadow = true;
        group.add(hex);
        const body = new THREE.Mesh(this._sharedGeometries.tower, this._sharedMaterials.towerBody);
        body.scale.y = tierScale;
        body.position.y = 2.75 * tierScale;
        body.castShadow = true;
        body.receiveShadow = true;
        group.add(body);
        const topY = 5.8 * tierScale;
        const top = new THREE.Mesh(this._sharedGeometries.towerTop, s.team === 'blue' ? this._sharedMaterials.blue : this._sharedMaterials.red);
        top.position.y = topY;
        group.add(top);
        // 塔顶能量核心（供动画旋转）
        const core = new THREE.Mesh(this._sharedGeometries.energyCoreGeo, this._sharedMaterials.energyCore);
        core.scale.setScalar(tierScale);
        core.position.y = topY;
        group.add(core);
        // 塔顶装饰环（水平放置）
        const ringDeco = new THREE.Mesh(this._sharedGeometries.ringThin, s.team === 'blue' ? this._sharedMaterials.blue : this._sharedMaterials.red);
        ringDeco.position.y = topY;
        ringDeco.rotation.x = -Math.PI / 2;
        group.add(ringDeco);
        // tier2/3 在塔身中部加装饰带（中段能量环）
        if (s.tier >= 2) {
          const midRing = new THREE.Mesh(this._sharedGeometries.ringThin, this._sharedMaterials.energyCore);
          midRing.position.y = 2.75 * tierScale;
          midRing.scale.setScalar(tierScale);
          midRing.rotation.x = -Math.PI / 2;
          group.add(midRing);
        }
        // tier3 高地塔：塔顶加小水晶（呼应基地水晶，作为高地标志）
        if (s.tier === 3) {
          const topCrystal = new THREE.Mesh(this._sharedGeometries.crystal, this._sharedMaterials.crystal);
          topCrystal.position.y = topY + 0.8;
          group.add(topCrystal);
        }
        // tier1 外塔：塔顶加细长尖刺（雷达天线感）
        if (s.tier === 1) {
          const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.8, 6), this._sharedMaterials.towerBody);
          antenna.position.y = topY + 0.4;
          group.add(antenna);
        }
        // 攻击范围指示环（发光科技环，提高不透明度）
        const ringGeo = new THREE.RingGeometry(s.range - 0.1, s.range, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: s.team === 'blue' ? 0x3a6bff : 0xff3a3a, transparent: true, opacity: 0.25, side: THREE.DoubleSide });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.05;
        group.add(ring);
        s.mesh = group;
        s.bodyMesh = body;
        s.topMesh = top;
        s.coreMesh = core;
      } else if (s.kind === 'main') {
        const crystal = new THREE.Mesh(this._sharedGeometries.mainCrystal, this._sharedMaterials.mainCrystal);
        crystal.position.y = 2;
        crystal.castShadow = true;
        group.add(crystal);
        // 六边形金属基座
        const baseGeo = new THREE.CylinderGeometry(2, 2.5, 0.6, 6);
        const base = new THREE.Mesh(baseGeo, this._sharedMaterials.towerBody);
        base.position.y = 0.3;
        base.receiveShadow = true;
        group.add(base);
        // 水晶下方旋转装饰环
        const decoRing = new THREE.Mesh(this._sharedGeometries.ringThin, this._sharedMaterials.crystal);
        decoRing.position.y = 1.1;
        decoRing.rotation.x = -Math.PI / 2;
        group.add(decoRing);
        // 水晶位置点光源增强发光感
        const crystalLight = new THREE.PointLight(s.team === 'blue' ? 0x3a6bff : 0xff3a3a, 0.5, 8);
        crystalLight.position.set(0, 2, 0);
        group.add(crystalLight);
        s.mesh = group;
        s.bodyMesh = crystal;
        s.coreMesh = decoRing;
      }
      this.addToScene(group);
      // 血条
      s.healthBar = this._createHealthBar(s.x, 6.5, s.team, s.maxHp);
    }
  }

  _buildJungleMeshes() {
    const THREE = this.THREE;
    for (const j of this.jungle) {
      this._buildJungleMeshFor(j);
    }
  }

  /** 按 type 差异化构建野怪 mesh，供首次生成与复活重建共用 */
  _buildJungleMeshFor(j) {
    const THREE = this.THREE;
    const group = new THREE.Group();
    group.position.set(j.x, 0, j.z);
    // 类型→颜色映射，每种野怪独有色彩
    const colorMap = {
      redbuff: 0xff5544, bluebuff: 0x4488ff,
      wolf: 0x8899aa, golem: 0x8a7a55, raptor: 0xddaa44,
      crab: 0x44aaaa, plant: 0x66cc55,
      tyrant: 0xcc44cc, overlord: 0xffaa22,
    };
    const baseColor = colorMap[j.type] || 0xaaaa44;
    const mat = new THREE.MeshStandardMaterial({ color: baseColor, roughness: 0.5, metalness: 0.3, emissive: baseColor, emissiveIntensity: 0.55 });

    // 营地标记（发光科技环，boss 用更大更亮的环）
    const isBoss = j.type === 'tyrant' || j.type === 'overlord';
    const campColor = isBoss ? baseColor : 0x00aaff;
    const campGeo = new THREE.RingGeometry(isBoss ? 2.4 : 1.8, isBoss ? 2.7 : 2, 24);
    const campMat = new THREE.MeshBasicMaterial({ color: campColor, transparent: true, opacity: 0.35, side: THREE.DoubleSide });
    const camp = new THREE.Mesh(campGeo, campMat);
    camp.rotation.x = -Math.PI / 2;
    camp.position.y = 0.04;
    group.add(camp);

    // 按类型选择主体几何体与高度
    let bodyGeo, bodyY = 0.8;
    switch (j.type) {
      case 'wolf':    bodyGeo = this._sharedGeometries.jungleWolf; bodyY = 0.5; break;   // 锥形低伏
      case 'golem':   bodyGeo = this._sharedGeometries.jungleGolem; bodyY = 0.85; break; // 大型十二面体
      case 'raptor':  bodyGeo = this._sharedGeometries.jungleRaptor; bodyY = 0.7; break;// 四面体
      case 'crab':    bodyGeo = this._sharedGeometries.jungleCrab; bodyY = 0.25; break;  // 扁平箱贴地
      case 'plant':   bodyGeo = this._sharedGeometries.junglePlant; bodyY = 0.6; break;  // 二十面体
      case 'tyrant':
      case 'overlord':bodyGeo = this._sharedGeometries.jungleDragon; bodyY = 1.0; break;// 高锥龙形
      default:        bodyGeo = this._sharedGeometries.jungle; break;                    // redbuff/bluebuff 十二面体
    }
    const body = new THREE.Mesh(bodyGeo, mat);
    body.position.y = bodyY;
    body.castShadow = true;
    group.add(body);
    j.bodyY = bodyY; // 缓存基准高度，供浮动动画使用

    // 外层能量场（半透明球壳，boss 更大，供动画旋转）
    const shellRadius = isBoss ? 1.2 : 0.85;
    const shellMat = new THREE.MeshStandardMaterial({ color: baseColor, transparent: true, opacity: 0.2, emissive: baseColor, emissiveIntensity: 0.4 });
    const shell = new THREE.Mesh(new THREE.IcosahedronGeometry(shellRadius, 0), shellMat);
    shell.position.y = bodyY;
    group.add(shell);

    // 类型差异化装饰
    if (j.type === 'wolf' || j.type === 'raptor') {
      // 小型生物额外加两个"眼"
      const eyeMat = new THREE.MeshStandardMaterial({ color: 0xffee44, emissive: 0xffee44, emissiveIntensity: 1.0 });
      for (const dx of [-0.1, 0.1]) {
        const eye = new THREE.Mesh(new THREE.SphereGeometry(0.05, 6, 4), eyeMat);
        eye.position.set(dx, bodyY + 0.15, 0.25);
        group.add(eye);
      }
    } else if (j.type === 'golem' || j.type === 'crab') {
      // 巨像/河蟹：左右两个"钳/臂"
      const armMat = new THREE.MeshStandardMaterial({ color: baseColor, roughness: 0.4, metalness: 0.5, emissive: baseColor, emissiveIntensity: 0.3 });
      for (const sign of [-1, 1]) {
        const arm = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.25, 0.45), armMat);
        arm.position.set(sign * 0.4, bodyY, 0.2);
        arm.rotation.z = sign * 0.3;
        group.add(arm);
      }
    } else if (j.type === 'plant') {
      // 妖花顶部花瓣（多个小三角）
      const petalMat = new THREE.MeshStandardMaterial({ color: 0xff66aa, emissive: 0xff3388, emissiveIntensity: 0.6 });
      for (let i = 0; i < 4; i++) {
        const petal = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.3, 4), petalMat);
        const a = (i / 4) * Math.PI * 2;
        petal.position.set(Math.cos(a) * 0.2, bodyY + 0.35, Math.sin(a) * 0.2);
        petal.rotation.x = Math.PI;
        group.add(petal);
      }
    } else if (isBoss) {
      // 龙/暴君：皇冠 + 双翼小三角
      const crownMat = new THREE.MeshStandardMaterial({ color: 0xffdd44, emissive: 0xffaa22, emissiveIntensity: 0.8, metalness: 0.8, roughness: 0.2 });
      const crown = new THREE.Mesh(new THREE.ConeGeometry(0.3, 0.4, 5), crownMat);
      crown.position.set(0, bodyY + 0.7, 0);
      group.add(crown);
      const wingMat = new THREE.MeshStandardMaterial({ color: baseColor, emissive: baseColor, emissiveIntensity: 0.4, transparent: true, opacity: 0.7, side: THREE.DoubleSide });
      for (const sign of [-1, 1]) {
        const wing = new THREE.Mesh(new THREE.PlaneGeometry(0.6, 0.4), wingMat);
        wing.position.set(sign * 0.5, bodyY + 0.2, 0);
        wing.rotation.y = sign * 0.4;
        group.add(wing);
      }
    }

    j.mesh = group;
    j.bodyMesh = body;
    j.shellMesh = shell;
    j.mat = mat;
    j.visible = true;
    this.addToScene(group);
    j.healthBar = this._createHealthBar(j.x, isBoss ? 2.4 : 1.8, 'neutral', j.maxHp);
  }

  _spawnHeroes() {
    const THREE = this.THREE;
    this.heroes = [];
    for (const team of ['blue', 'red']) {
      for (let i = 0; i < 5; i++) {
        const heroKey = COMPOSITION[i];
        const def = HERO_DEFS[heroKey];
        const lane = ['top', 'mid', 'bottom', 'bottom', 'top'][i]; // 上2 中1 下2
        const hero = {
          id: `${team}_${i}`,
          team, heroKey, role: def.role, lane,
          name: def.name,
          level: 1, exp: 0, gold: 500,
          hp: def.hp, maxHp: def.hp, mp: def.mp, maxMp: def.mp,
          atk: def.atk, armor: def.armor, mr: def.mr,
          moveSpeed: def.speed, attackRange: def.range,
          x: 0, z: 0, facing: 0,
          alive: true, respawnTimer: 0,
          // 状态
          state: 'lane_fight',   // lane_fight | last_hit | push | retreat | recall | jungle | gank | team_fight | dead | return_to_lane
          stateTimer: 0,
          target: null,           // 当前攻击目标 id
          moveTarget: null,       // 移动目标 {x,z}
          waypointIdx: 0,         // 沿路 waypoint 索引
          recallTimer: 0,         // 回城读条
          // 技能
          skill1Cd: 0, skill2Cd: 0, ultCd: 0,
          skill1Level: 1, skill2Level: 1, ultLevel: 0,
          // 装备
          equipLevel: 0,
          // 经济转数值：累计金币 → 战力（每 100 金币 +2atk +20hp +2armor）
          totalGoldEarned: 0,
          appliedGoldPower: 0,
          // buff
          buffs: [],              // {type, remain}
          // 攻击
          attackCd: 0,
          // 临时
          lastDamager: null,      // 最后伤害来源（用于助攻）
          lastDamagedTime: 0,     // 最后受击时间（用于脱战回血判定）
          kills: 0, deaths: 0, assists: 0,
          // mesh
          kind: 'hero',
          mesh: null, bodyMesh: null, headMesh: null, healthBar: null,
          isPlayer: false,
          // 强化学习跟踪（每英雄独立）：上次状态/动作/累积奖励
          _rl: { lastState: null, lastAction: null, accumReward: 0 },
        };
        this._applyEquipBonus(hero);
        this.heroes.push(hero);

        // 创建 mesh
        const group = new THREE.Group();
        const teamMat = team === 'blue' ? this._sharedMaterials.blue : this._sharedMaterials.red;
        const body = new THREE.Mesh(this._sharedGeometries.heroBody, teamMat);
        body.position.y = 1.1;
        body.castShadow = true;
        group.add(body);
        // 科技胸甲（队伍色高金属）
        const chest = new THREE.Mesh(this._sharedGeometries.heroChest, teamMat);
        chest.position.y = 1.2;
        chest.castShadow = true;
        group.add(chest);
        // 职业色头部（发光）
        const headMat = new THREE.MeshStandardMaterial({ color: def.color, roughness: 0.5, emissive: def.color, emissiveIntensity: 0.3 });
        const head = new THREE.Mesh(this._sharedGeometries.heroHead, headMat);
        head.position.y = 1.85;
        group.add(head);
        // 发光护目镜（队伍色强发光，贴在脸部前方）
        const visorMat = new THREE.MeshStandardMaterial({ color: team === 'blue' ? 0x3a6bff : 0xff3a3a, emissive: team === 'blue' ? 0x3a6bff : 0xff3a3a, emissiveIntensity: 0.8, roughness: 0.3, metalness: 0.6 });
        const visor = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.08, 0.05), visorMat);
        visor.position.set(0, 1.86, 0.28);
        group.add(visor);
        // 职业标识改为发光背光环（水平放在脚下）
        const mark = new THREE.Mesh(this._sharedGeometries.ringThin, headMat);
        mark.position.set(0, 0.05, 0);
        mark.rotation.x = -Math.PI / 2;
        group.add(mark);

        // ====== 职业差异化部件 ======
        // 战士：左手持盾（高大金属盾牌）
        // 法师：右手持法杖（杖顶发光球）
        // 刺客：背后披风（半透明）
        // 射手：左手持弓（半圆弧）
        // 辅助：头顶光环（旋转）
        if (heroKey === 'warrior') {
          const shield = new THREE.Mesh(this._sharedGeometries.heroShield, teamMat);
          shield.position.set(-0.45, 1.25, 0.15);
          shield.rotation.y = Math.PI / 2;
          group.add(shield);
          // 盾面发光纹路
          const shieldGlow = new THREE.Mesh(new THREE.PlaneGeometry(0.4, 0.6),
            new THREE.MeshBasicMaterial({ color: def.color, transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
          shieldGlow.position.set(-0.5, 1.25, 0.15);
          shieldGlow.rotation.y = Math.PI / 2;
          group.add(shieldGlow);
        } else if (heroKey === 'mage') {
          const staff = new THREE.Mesh(this._sharedGeometries.heroStaff, this._sharedMaterials.towerBody);
          staff.position.set(0.35, 1.5, 0.1);
          staff.rotation.z = -0.15;
          group.add(staff);
          // 杖顶发光球
          const orb = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 8),
            new THREE.MeshStandardMaterial({ color: def.color, emissive: def.color, emissiveIntensity: 1.0 }));
          orb.position.set(0.45, 2.3, 0.15);
          group.add(orb);
        } else if (heroKey === 'assassin') {
          // 背后披风（半透明）
          const cape = new THREE.Mesh(this._sharedGeometries.heroCape,
            new THREE.MeshStandardMaterial({ color: def.color, emissive: def.color, emissiveIntensity: 0.4, transparent: true, opacity: 0.7, side: THREE.DoubleSide }));
          cape.position.set(0, 1.3, -0.3);
          group.add(cape);
          // 双匕首（腰侧）
          for (const sign of [-1, 1]) {
            const dagger = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.4, 0.04),
              new THREE.MeshStandardMaterial({ color: 0xeeeeee, metalness: 0.9, roughness: 0.2, emissive: def.color, emissiveIntensity: 0.5 }));
            dagger.position.set(sign * 0.35, 1.0, 0.1);
            dagger.rotation.z = sign * 0.2;
            group.add(dagger);
          }
        } else if (heroKey === 'marksman') {
          // 弓（左前方半圆）
          const bow = new THREE.Mesh(this._sharedGeometries.heroBow, teamMat);
          bow.position.set(-0.4, 1.3, 0.2);
          bow.rotation.z = -Math.PI / 2;
          group.add(bow);
          // 弓弦（直线）
          const string = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.7, 4),
            new THREE.MeshBasicMaterial({ color: 0xffffff }));
          string.position.set(-0.4, 1.3, 0.2);
          string.rotation.z = Math.PI / 2;
          group.add(string);
        } else if (heroKey === 'support') {
          // 头顶光环（旋转）
          const halo = new THREE.Mesh(this._sharedGeometries.heroHalo,
            new THREE.MeshStandardMaterial({ color: 0xffee88, emissive: 0xffee88, emissiveIntensity: 0.9, transparent: true, opacity: 0.85 }));
          halo.position.set(0, 2.3, 0);
          halo.rotation.x = Math.PI / 2;
          group.add(halo);
          hero.haloMesh = halo;
        }
        hero.mesh = group;
        hero.bodyMesh = body;
        hero.chestMesh = chest;
        hero.headMesh = head;
        hero.visorMesh = visor;
        hero.headMat = headMat;

        // 出生位置（基地附近）
        const base = team === 'blue' ? BLUE_BASE : RED_BASE;
        const offsetAngle = (i / 5) * Math.PI * 2;
        hero.x = base.x + Math.cos(offsetAngle) * 3;
        hero.z = base.z + Math.sin(offsetAngle) * 3;
        hero.facing = team === 'blue' ? Math.PI / 4 : -Math.PI * 3 / 4;
        group.position.set(hero.x, 0, hero.z);
        group.rotation.y = hero.facing;
        this.addToScene(group);
        hero.healthBar = this._createHealthBar(hero.x, 2.6, team, hero.maxHp, true);
      }
    }
  }

  _spawnPlayer() {
    // 玩家操控蓝方 0 号英雄
    const hero = this.heroes.find(h => h.id === this.playerHeroId);
    if (!hero) return;
    hero.isPlayer = true;
    // 玩家英雄用 avatar 显示，隐藏内部 mesh 避免重叠
    if (hero.mesh) hero.mesh.visible = false;
    const avatar = this.App.currentAvatar;
    if (avatar) {
      avatar.position.set(hero.x, 0, hero.z);
      this.App.smoothRotY = hero.facing;
    }
  }

  /** 创建血条 sprite */
  _createHealthBar(x, y, team, maxHp, withMp = false) {
    const THREE = this.THREE;
    const canvas = document.createElement('canvas');
    canvas.width = 128; canvas.height = withMp ? 20 : 16;
    const ctx = canvas.getContext('2d');
    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
    sprite.position.set(x, y, 0);
    sprite.scale.set(2, withMp ? 0.32 : 0.25, 1);
    this.addToScene(sprite);
    return { sprite, canvas, ctx, texture, team, maxHp, x, y, withMp };
  }

  // ==================== 兵线系统 ====================

  _updateMinionSpawning(dt) {
    if (this._gameTime >= this._nextMinionWave) {
      this._nextMinionWave += MINION_INTERVAL;
      this._spawnMinionWave();
    }
  }

  _spawnMinionWave() {
    const wave = Math.floor(this._gameTime / MINION_INTERVAL);
    for (const lane of LANE_KEYS) {
      for (const team of ['blue', 'red']) {
        const path = LANES[lane];
        const startIdx = team === 'blue' ? 0 : path.length - 1;
        const start = path[startIdx];
        // 3 近战 + 3 远程，第 3 波起带炮车
        const types = ['melee', 'melee', 'melee', 'ranged', 'ranged', 'ranged'];
        if (wave >= 2 && wave % 2 === 0) types.push('cannon');
        for (let i = 0; i < types.length; i++) {
          const type = types[i];
          const offset = i * 1.2;
          const sx = start.x + (team === 'blue' ? -offset : offset);
          const sz = start.z + (team === 'blue' ? -offset : offset);
          this._spawnMinion(team, lane, type, sx, sz, wave);
        }
      }
    }
    this._pushEvent('minion_wave', { wave, lane: 'all' });
  }

  _spawnMinion(team, lane, type, x, z, wave = 0) {
    const THREE = this.THREE;
    // 兵线随波次增强：每波 +8% HP / +6% ATK（越后期越强）
    const scale = 1 + wave * 0.08;
    const atkScale = 1 + wave * 0.06;
    const stats = {
      melee:  { hp: Math.round(480 * scale),  atk: Math.round(35 * atkScale),  range: 1.5, speed: MINION_SPEED },
      ranged: { hp: Math.round(320 * scale),  atk: Math.round(45 * atkScale),  range: 3.5, speed: MINION_SPEED },
      cannon: { hp: Math.round(900 * scale),  atk: Math.round(75 * atkScale),  range: 4.5, speed: MINION_SPEED * 0.85 },
    }[type];
    const minion = {
      id: `minion_${team}_${lane}_${Math.random().toString(36).slice(2, 7)}`,
      kind: 'minion', team, lane, type,
      x, z, facing: team === 'blue' ? 0 : Math.PI,
      hp: stats.hp, maxHp: stats.hp, atk: stats.atk, range: stats.range, speed: stats.speed,
      attackRange: stats.range,
      attackCd: 0, target: null,
      waypointIdx: team === 'blue' ? 0 : LANES[lane].length - 1,
      moveTarget: null,
      mesh: null, healthBar: null,
    };
    const group = new THREE.Group();
    group.position.set(x, 0, z);
    const geo = type === 'melee' ? this._sharedGeometries.minionMelee
              : type === 'ranged' ? this._sharedGeometries.minionRanged
              : this._sharedGeometries.minionCannon;
    // 机器人化金属队伍色材质
    const mat = new THREE.MeshStandardMaterial({ color: team === 'blue' ? 0x3a5aff : 0xff3a3a, roughness: 0.35, metalness: 0.7, emissive: team === 'blue' ? 0x0a1a55 : 0x55110a, emissiveIntensity: 0.4 });
    const body = new THREE.Mesh(geo, mat);
    const bodyY = type === 'cannon' ? 0.7 : 0.5;
    body.position.y = bodyY;
    body.castShadow = true;
    group.add(body);
    // 发光"眼睛"（强 emissive 队伍色）
    const eyeMat = new THREE.MeshStandardMaterial({ color: team === 'blue' ? 0x3a6bff : 0xff3a3a, emissive: team === 'blue' ? 0x3a6bff : 0xff3a3a, emissiveIntensity: 1.0, roughness: 0.3 });
    const eye = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.06, 0.05), eyeMat);
    eye.position.set(0, bodyY + 0.15, 0.2);
    group.add(eye);

    // ====== 类型差异化部件 ======
    if (type === 'melee') {
      // 近战兵：左手持盾（更厚实、有防御感）
      const shield = new THREE.Mesh(this._sharedGeometries.minionShield, mat);
      shield.position.set(-0.32, bodyY, 0.1);
      shield.rotation.y = Math.PI / 2;
      group.add(shield);
      // 盾面发光纹
      const glowMat = new THREE.MeshBasicMaterial({ color: team === 'blue' ? 0x66aaff : 0xff8888, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
      const glow = new THREE.Mesh(new THREE.PlaneGeometry(0.35, 0.45), glowMat);
      glow.position.set(-0.37, bodyY, 0.1);
      glow.rotation.y = Math.PI / 2;
      group.add(glow);
    } else if (type === 'ranged') {
      // 远程兵：左手持法术书（竖立小方块，发出魔法光）
      const bookMat = new THREE.MeshStandardMaterial({ color: 0xddccaa, roughness: 0.6, emissive: 0xffaa44, emissiveIntensity: 0.4 });
      const book = new THREE.Mesh(this._sharedGeometries.minionBook, bookMat);
      book.position.set(-0.3, bodyY + 0.1, 0.1);
      book.rotation.y = -0.4;
      group.add(book);
      // 书上方发光球
      const orbMat = new THREE.MeshStandardMaterial({ color: 0xffee88, emissive: 0xffee88, emissiveIntensity: 0.8 });
      const orb = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 6), orbMat);
      orb.position.set(-0.3, bodyY + 0.3, 0.15);
      group.add(orb);
    } else if (type === 'cannon') {
      // 炮车：前方伸炮管
      const barrel = new THREE.Mesh(this._sharedGeometries.minionCannonBarrel, mat);
      barrel.position.set(0, bodyY + 0.1, 0.45);
      barrel.rotation.x = Math.PI / 2;
      group.add(barrel);
      // 炮口能量环
      const muzzleRing = new THREE.Mesh(this._sharedGeometries.ringThin, this._sharedMaterials.energyCore);
      muzzleRing.position.set(0, bodyY + 0.1, 0.75);
      muzzleRing.scale.setScalar(0.6);
      group.add(muzzleRing);
      // 顶部旋转装饰环
      const cannonRing = new THREE.Mesh(this._sharedGeometries.ringThin, team === 'blue' ? this._sharedMaterials.blue : this._sharedMaterials.red);
      cannonRing.position.y = bodyY + 0.4;
      cannonRing.rotation.x = -Math.PI / 2;
      group.add(cannonRing);
    }
    minion.mesh = group;
    minion.bodyMesh = body;
    minion.mat = mat;
    this.addToScene(group);
    minion.healthBar = this._createHealthBar(x, type === 'cannon' ? 1.6 : 1.2, team, minion.maxHp);
    this.minions.push(minion);
  }

  _updateMinions(dt) {
    for (let i = this.minions.length - 1; i >= 0; i--) {
      const m = this.minions[i];
      if (!m.mesh) { this.minions.splice(i, 1); continue; }
      // 死亡
      if (m.hp <= 0) {
        this._removeMinion(m, i);
        continue;
      }
      // 攻击 CD
      if (m.attackCd > 0) m.attackCd -= dt;

      // 寻找目标（敌方小兵/英雄/塔，按优先级）
      if (!m.target || !this._isTargetAlive(m.target)) {
        m.target = this._findMinionTarget(m);
      }
      const target = m.target ? this._getUnitById(m.target) : null;
      if (target) {
        const dist = Math.hypot(target.x - m.x, target.z - m.z);
        if (dist <= m.attackRange + 0.5) {
          // 攻击
          m.facing = Math.atan2(target.x - m.x, target.z - m.z);
          if (m.attackCd <= 0) {
            m.attackCd = 1.0;
            this._dealDamage(m, target, m.atk, false);
            if (m.type === 'ranged' || m.type === 'cannon') {
              this._spawnProjectile(m, target, 0xffaa44);
            }
          }
        } else {
          // 移动向目标
          this._moveUnit(m, target.x, target.z, m.speed * dt);
        }
      } else {
        // 沿路线前进
        this._advanceAlongLane(m, dt);
      }

      // 更新 mesh
      if (m.mesh) {
        m.mesh.position.set(m.x, 0, m.z);
        m.mesh.rotation.y = m.facing;
      }
      this._updateHealthBar(m.healthBar, m.x, m.z, m.hp, m.maxHp);
    }
  }

  _findMinionTarget(m) {
    let best = null, bestDist = 8; // 小兵视野范围
    // 优先：敌方小兵 > 英雄 > 塔
    const enemies = this.minions.filter(o => o.team !== m.team && o.hp > 0);
    for (const e of enemies) {
      const d = Math.hypot(e.x - m.x, e.z - m.z);
      if (d < bestDist) { best = e.id; bestDist = d; }
    }
    if (best) return best;
    // 英雄
    for (const h of this.heroes) {
      if (h.team === m.team || !h.alive) continue;
      const d = Math.hypot(h.x - m.x, h.z - m.z);
      if (d < bestDist) { best = h.id; bestDist = d; }
    }
    if (best) return best;
    // 塔
    for (const s of this.structures) {
      if (s.team === m.team || s.hp <= 0 || s.kind === 'main') continue;
      const d = Math.hypot(s.x - m.x, s.z - m.z);
      if (d < bestDist) { best = s.id; bestDist = d; }
    }
    return best;
  }

  _advanceAlongLane(unit, dt) {
    const path = LANES[unit.lane];
    if (!path) return;
    let target = unit.moveTarget;
    if (!target) {
      const idx = unit.waypointIdx;
      if (unit.team === 'blue') {
        if (idx < path.length - 1) {
          unit.waypointIdx = idx + 1;
          target = path[unit.waypointIdx];
        } else { target = path[path.length - 1]; }
      } else {
        if (idx > 0) {
          unit.waypointIdx = idx - 1;
          target = path[unit.waypointIdx];
        } else { target = path[0]; }
      }
      unit.moveTarget = target;
    }
    this._moveUnit(unit, target.x, target.z, unit.speed * dt);
    const d = Math.hypot(target.x - unit.x, target.z - unit.z);
    if (d < 1.0) unit.moveTarget = null;
  }

  _removeMinion(m, idx) {
    if (m.mesh && m.mesh.parent) m.mesh.parent.remove(m.mesh);
    this._disposeObject(m.mesh);
    this._removeHealthBar(m.healthBar);
    this.minions.splice(idx, 1);
  }

  // ==================== 防御塔系统 ====================

  _updateTowers(dt) {
    for (let i = this.structures.length - 1; i >= 0; i--) {
      const s = this.structures[i];
      if (s.hp <= 0) {
        // 已被摧毁
        if (s.mesh) {
          // 爆炸特效
          this._spawnExplosion(s.x, 3, s.z, s.team === 'blue' ? 0x3a6bff : 0xff3a3a);
          if (s.mesh.parent) s.mesh.parent.remove(s.mesh);
          this._disposeObject(s.mesh);
          this._removeHealthBar(s.healthBar);
          s.mesh = null;
        }
        continue;
      }
      if (s.kind === 'main') {
        // 主水晶不主动攻击
      } else if (s.kind === 'tower') {
        if (s.attackCd > 0) s.attackCd -= dt;
        // 选择目标：打塔英雄 > 打我方英雄的英雄 > 小兵
        if (s.attackCd <= 0) {
          const target = this._findTowerTarget(s);
          if (target) {
            s.attackCd = 1 / s.attackSpeed;
            // 递增伤害
            const damage = s.atk * (1 + s.consecutiveHits * TOWER_DAMAGE_RAMP);
            this._dealDamage(s, target, damage, false);
            this._spawnProjectile(s, target, s.team === 'blue' ? 0x66aaff : 0xff6666);
            s.consecutiveHits++;
          } else {
            s.consecutiveHits = 0;
          }
        }
      }
      if (s.healthBar) this._updateHealthBar(s.healthBar, s.x, s.z, s.hp, s.maxHp);
      // 主水晶旋转
      if (s.kind === 'main' && s.bodyMesh) s.bodyMesh.rotation.y += dt * 0.5;
      // 科技能量核心 / 装饰环旋转动画
      if (s.coreMesh) {
        s.coreMesh.rotation.y += dt * 1.5;
        s.coreMesh.rotation.x += dt * 0.8;
      }
    }
  }

  _findTowerTarget(tower) {
    const range = tower.range;
    let bestTarget = null;
    let bestHero = null;
    // 1) 在攻击我方英雄的敌方英雄
    for (const h of this.heroes) {
      if (h.team === tower.team || !h.alive) continue;
      const d = Math.hypot(h.x - tower.x, h.z - tower.z);
      if (d > range) continue;
      // 检查是否在攻击我方英雄
      if (h.target) {
        const t = this._getUnitById(h.target);
        if (t && t.team === tower.team && t.kind === 'hero' && t.alive) {
          if (!bestHero || d < Math.hypot(bestHero.x - tower.x, bestHero.z - tower.z)) bestHero = h;
        }
      }
    }
    if (bestHero) return bestHero;
    // 2) 在范围内的敌方小兵
    let bestMinion = null, bestDist = range;
    for (const m of this.minions) {
      if (m.team === tower.team || m.hp <= 0) continue;
      const d = Math.hypot(m.x - tower.x, m.z - tower.z);
      if (d < bestDist) { bestMinion = m; bestDist = d; }
    }
    if (bestMinion) return bestMinion;
    // 3) 在范围内的敌方英雄
    if (!bestHero) {
      for (const h of this.heroes) {
        if (h.team === tower.team || !h.alive) continue;
        const d = Math.hypot(h.x - tower.x, h.z - tower.z);
        if (d <= range) {
          if (!bestHero || d < Math.hypot(bestHero.x - tower.x, bestHero.z - tower.z)) bestHero = h;
        }
      }
    }
    return bestHero;
  }

  // ==================== 野怪系统 ====================

  _updateJungle(dt) {
    for (const j of this.jungle) {
      if (j.hp <= 0) {
        j.respawn -= dt;
        if (j.mesh) {
          if (j.mesh.parent) j.mesh.parent.remove(j.mesh);
          this._disposeObject(j.mesh);
          j.mesh = null;
          this._removeHealthBar(j.healthBar);
          j.healthBar = null;
        }
        if (j.respawn <= 0 && j.team !== 'neutral') {
          // 普通 buff 怪复活
          j.hp = j.maxHp;
          this._rebuildJungleMesh(j);
        } else if (j.respawn <= 0 && j.team === 'neutral') {
          // 中立生物复活（更长时间）
          j.hp = j.maxHp;
          this._rebuildJungleMesh(j);
        }
        continue;
      }
      if (j.attackCd > 0) j.attackCd -= dt;
      // 野怪不主动追击，只在被攻击时反击（_dealDamage 中处理）
      if (j.mesh) {
        j.mesh.position.set(j.x, 0, j.z);
        j.bodyMesh.rotation.y += dt * 0.3;
        // 外层能量场旋转动画
        if (j.shellMesh) {
          j.shellMesh.rotation.y += dt * 1.2;
          j.shellMesh.rotation.x += dt * 0.6;
        }
      }
      if (j.healthBar) this._updateHealthBar(j.healthBar, j.x, j.z, j.hp, j.maxHp);
    }
  }

  _rebuildJungleMesh(j) {
    // 复用按 type 差异化建模逻辑
    this._buildJungleMeshFor(j);
  }

  // ==================== 英雄系统与 bot AI ====================

  _updateHeroes(dt) {
    for (const h of this.heroes) {
      if (!h.alive) {
        h.respawnTimer -= dt;
        // 玩家死亡期间 avatar 锁定在基地，避免乱跑
        if (h.isPlayer && !this.spectating) {
          const avatar = this.App.currentAvatar;
          if (avatar) {
            const base = h.team === 'blue' ? BLUE_BASE : RED_BASE;
            avatar.position.set(base.x, 0, base.z);
          }
        }
        if (h.respawnTimer <= 0) {
          this._respawnHero(h);
        }
        continue;
      }
      // CD
      if (h.skill1Cd > 0) h.skill1Cd -= dt;
      if (h.skill2Cd > 0) h.skill2Cd -= dt;
      if (h.ultCd > 0) h.ultCd -= dt;
      if (h.attackCd > 0) h.attackCd -= dt;
      // 被动经济/经验（自动成长）
      this._grantGold(h, PASSIVE_GOLD_PER_SEC * dt);
      this._addExp(h, PASSIVE_EXP_PER_SEC * dt);
      // 生命/蓝量自动恢复（脱战更快）
      const inCombat = (h.lastDamagedTime && (this._gameTime - h.lastDamagedTime < 4)) || h.attackCd > 0.3;
      const hpRegen = (inCombat ? 0.5 : 1.5) * dt;   // 每秒 0.5%（战斗）/ 1.5%（脱战）
      const mpRegen = (inCombat ? 1.0 : 2.0) * dt;   // 每秒 1.0%（战斗）/ 2.0%（脱战）
      if (h.hp < h.maxHp) h.hp = Math.min(h.maxHp, h.hp + h.maxHp * hpRegen / 100);
      if (h.mp < h.maxMp) h.mp = Math.min(h.maxMp, h.mp + h.maxMp * mpRegen / 100);
      // buff
      for (let i = h.buffs.length - 1; i >= 0; i--) {
        h.buffs[i].remain -= dt;
        if (h.buffs[i].remain <= 0) h.buffs.splice(i, 1);
      }
      // 装备升级（金币足够自动购买）
      this._tryBuyEquipment(h);
      // 经济转战力（累计金币 → 属性加成）
      this._applyGoldPower(h);
      // 回城
      if (h.state === 'recall') {
        h.recallTimer -= dt;
        if (h.recallTimer <= 0) {
          // 回城完成
          const base = h.team === 'blue' ? BLUE_BASE : RED_BASE;
          h.x = base.x + (Math.random() - 0.5) * 2;
          h.z = base.z + (Math.random() - 0.5) * 2;
          h.hp = h.maxHp;
          h.mp = h.maxMp;
          h.state = 'return_to_lane';
          h.stateTimer = 0;
          this._pushEvent('hero_recall', { hero: h.id, team: h.team });
        }
        continue;
      }
      // 玩家英雄由玩家操控（位置由 GameModeManager 写入），但技能与攻击仍在此处理
      // 加速自我对弈模式下，玩家英雄也交给 AI 控制
      if (h.isPlayer && !this.spectating && !this._isFastMode()) {
        // 同步位置
        const avatar = this.App.currentAvatar;
        if (avatar) {
          h.x = avatar.position.x;
          h.z = avatar.position.z;
          h.facing = this.App.smoothRotY || 0;
        }
        // 自动普攻最近敌人
        this._playerAutoAttack(h, dt);
      } else {
        // bot 英雄移动与攻击（决策由 _runBotDecisions 设置 state/target）
        this._updateBotHero(h, dt);
      }
      // 同步 mesh
      if (h.mesh) {
        h.mesh.position.set(h.x, 0, h.z);
        h.mesh.rotation.y = h.facing;
      }
      // 血条（英雄显示 HP + MP + 等级）
      if (h.healthBar) this._updateHealthBar(h.healthBar, h.x, h.z, h.hp, h.maxHp, h.mp, h.maxMp, h.level);
    }
  }

  _updateBotHero(h, dt) {
    // 根据状态执行行为
    switch (h.state) {
      case 'dead': break;
      case 'recall': break; // 已在上方处理
      case 'retreat': {
        // 撤退到己方最近塔
        const safe = this._nearestSafeTower(h);
        if (safe) {
          this._moveUnit(h, safe.x, safe.z, h.moveSpeed * dt);
          if (Math.hypot(safe.x - h.x, safe.z - h.z) < 3) {
            h.state = 'lane_fight';
            h.stateTimer = 0;
          }
        }
        break;
      }
      case 'jungle': {
        // 前往野怪并攻击
        const j = this.jungle.find(j => j.id === h.target && j.hp > 0);
        if (!j) { h.target = null; h.state = 'lane_fight'; break; }
        const d = Math.hypot(j.x - h.x, j.z - h.z);
        if (d <= h.attackRange) {
          h.facing = Math.atan2(j.x - h.x, j.z - h.z);
          if (h.attackCd <= 0) {
            h.attackCd = 0.8;
            if (h.heroKey !== 'marksman') {
              this._playerAttackAoe(h, j.x, j.z, j);
            } else {
              this._dealDamage(h, j, h.atk, false);
              this._spawnProjectile(h, j, 0xffff44);
            }
          }
        } else {
          this._moveUnit(h, j.x, j.z, h.moveSpeed * dt);
        }
        break;
      }
      case 'gank':
      case 'team_fight':
      case 'lane_fight':
      case 'last_hit':
      case 'push':
      default: {
        // 通用战斗逻辑：寻找目标，攻击或移动
        let target = h.target ? this._getUnitById(h.target) : null;
        if (!target || !this._isTargetAlive(h.target)) {
          target = this._findHeroTarget(h);
          h.target = target ? target.id : null;
        }
        if (target) {
          const d = Math.hypot(target.x - h.x, target.z - h.z);
          // 检查目标是否在敌方塔下（避免越塔）
          const enemyTowerNear = this._isUnderEnemyTower(target.x, target.z, h.team);
          if (enemyTowerNear && h.hp < h.maxHp * 0.6) {
            // 不越塔，撤退
            h.state = 'retreat';
            break;
          }
          if (d <= h.attackRange + SKILL_RANGE_BONUS) {
            h.facing = Math.atan2(target.x - h.x, target.z - h.z);
            // 普攻（非射手为范围伤害）
            if (h.attackCd <= 0) {
              h.attackCd = 0.7;
              if (h.heroKey !== 'marksman') {
                this._playerAttackAoe(h, target.x, target.z, target);
              } else {
                this._dealDamage(h, target, h.atk, false);
                this._spawnProjectile(h, target, 0xffee44);
              }
            }
            // 尝试释放技能
            this._botTryCastSkill(h, target, d);
          } else {
            this._moveUnit(h, target.x, target.z, h.moveSpeed * dt);
          }
        } else {
          // 无目标：前往当前路线前线
          this._advanceHeroToLaneFront(h, dt);
        }
        break;
      }
      case 'return_to_lane': {
        // 回到线上
        const front = this._laneFrontPoint(h.lane, h.team);
        if (front) {
          this._moveUnit(h, front.x, front.z, h.moveSpeed * dt);
          if (Math.hypot(front.x - h.x, front.z - h.z) < 3) {
            h.state = 'lane_fight';
          }
        }
        break;
      }
    }
    h.stateTimer += dt;
  }

  _findHeroTarget(h) {
    let best = null, bestDist = h.attackRange + 6;
    // 优先敌方英雄（在攻击范围+视野内）
    for (const e of this.heroes) {
      if (e.team === h.team || !e.alive) continue;
      const d = Math.hypot(e.x - h.x, e.z - h.z);
      if (d < bestDist) { best = e; bestDist = d; }
    }
    if (best) return best;
    // 敌方小兵（扩大搜索范围，覆盖一整条兵线）
    bestDist = h.attackRange + 10;
    for (const m of this.minions) {
      if (m.team === h.team || m.hp <= 0) continue;
      const d = Math.hypot(m.x - h.x, m.z - h.z);
      if (d < bestDist) { best = m; bestDist = d; }
    }
    if (best) return best;
    // 野怪（玩家可打野，搜索范围覆盖全图野区）
    bestDist = 30;
    for (const j of this.jungle) {
      if (j.hp <= 0) continue;
      const d = Math.hypot(j.x - h.x, j.z - h.z);
      if (d < bestDist) { best = j; bestDist = d; }
    }
    if (best) return best;
    // 敌方塔（可随时攻击，不再限定 push 状态）
    for (const s of this.structures) {
      if (s.team === h.team || s.hp <= 0 || s.kind === 'main') continue;
      const d = Math.hypot(s.x - h.x, s.z - h.z);
      if (d < h.attackRange + 8) return s;
    }
    // 敌方主水晶（最后）
    for (const s of this.structures) {
      if (s.kind === 'main' && s.team !== h.team && s.hp > 0) {
        const d = Math.hypot(s.x - h.x, s.z - h.z);
        if (d < h.attackRange + 4) return s;
      }
    }
    return null;
  }

  _advanceHeroToLaneFront(h, dt) {
    const path = LANES[h.lane];
    if (!path) return;
    // 找到线上最前线（己方最靠前的单位位置）
    const front = this._laneFrontPoint(h.lane, h.team);
    if (front) {
      this._moveUnit(h, front.x, front.z, h.moveSpeed * dt * 0.9);
    }
  }

  _laneFrontPoint(lane, team) {
    // 己方在该路最靠前的单位
    let bestPoint = null;
    let bestProgress = team === 'blue' ? -1 : 999;
    const path = LANES[lane];
    const units = [...this.minions.filter(m => m.lane === lane && m.team === team && m.hp > 0),
                   ...this.heroes.filter(h => h.lane === lane && h.team === team && h.alive)];
    for (const u of units) {
      // 进度：沿路径距离起点的远近
      const prog = this._pathProgress(path, u.x, u.z);
      if (team === 'blue' && prog > bestProgress) { bestProgress = prog; bestPoint = { x: u.x, z: u.z }; }
      if (team === 'red' && prog < bestProgress) { bestProgress = prog; bestPoint = { x: u.x, z: u.z }; }
    }
    if (!bestPoint) {
      // 没有单位，回基地附近的路点
      bestPoint = team === 'blue' ? path[1] : path[path.length - 2];
    }
    return bestPoint;
  }

  _pathProgress(path, x, z) {
    // 返回 (x,z) 在路径上的近似进度（0=起点）
    let bestT = 0, bestDist = Infinity;
    for (let i = 0; i < path.length - 1; i++) {
      const a = path[i], b = path[i + 1];
      const dx = b.x - a.x, dz = b.z - a.z;
      const len2 = dx * dx + dz * dz;
      if (len2 < 0.01) continue;
      const t = Math.max(0, Math.min(1, ((x - a.x) * dx + (z - a.z) * dz) / len2));
      const px = a.x + dx * t, pz = a.z + dz * t;
      const d = Math.hypot(px - x, pz - z);
      if (d < bestDist) { bestDist = d; bestT = i + t; }
    }
    return bestT;
  }

  _nearestSafeTower(h) {
    let best = null, bestDist = Infinity;
    for (const s of this.structures) {
      if (s.team !== h.team || s.hp <= 0 || s.kind === 'main') continue;
      const d = Math.hypot(s.x - h.x, s.z - h.z);
      if (d < bestDist) { best = s; bestDist = d; }
    }
    return best;
  }

  _isUnderEnemyTower(x, z, team) {
    for (const s of this.structures) {
      if (s.team === team || s.hp <= 0 || s.kind === 'main') continue;
      if (Math.hypot(s.x - x, s.z - z) < s.range) return true;
    }
    return false;
  }

  // ==================== bot 决策 ====================

  _runBotDecisions() {
    const fast = this._isFastMode();
    for (const h of this.heroes) {
      if (!h.alive) continue;
      // 加速自我对弈模式：玩家英雄也由 AI 控制；正常模式跳过玩家英雄
      if (h.isPlayer && !this.spectating && !fast) continue;
      if (this._rlEnabled) {
        this._rlDecideBotState(h);
      } else {
        this._decideBotState(h);
      }
    }
  }

  _decideBotState(h) {
    // 紧急情况：血量低或蓝量低 → 回城
    if (h.hp < h.maxHp * 0.25 || (h.mp < h.maxMp * 0.1 && h.role !== 'marksman')) {
      if (h.state !== 'recall' && !this._isInBase(h)) {
        h.state = 'recall';
        h.recallTimer = RECALL_TIME;
        h.target = null;
        return;
      }
    }
    // 被敌方塔攻击 → 撤退
    if (this._isUnderEnemyTower(h.x, h.z, h.team) && h.hp < h.maxHp * 0.5) {
      h.state = 'retreat';
      h.target = null;
      return;
    }
    // 附近有敌方英雄且我方血量劣势 → 撤退
    const enemyHero = this._nearestEnemyHero(h);
    if (enemyHero && h.hp < h.maxHp * 0.35) {
      h.state = 'retreat';
      h.target = enemyHero.id;
      return;
    }

    // 团战检测：附近 3+ 友方英雄且附近有敌方英雄
    const allies = this.heroes.filter(a => a.team === h.team && a.alive && a.id !== h.id && Math.hypot(a.x - h.x, a.z - h.z) < 12);
    const enemies = this.heroes.filter(e => e.team !== h.team && e.alive && Math.hypot(e.x - h.x, e.z - h.z) < 14);
    if (allies.length >= 2 && enemies.length >= 2) {
      h.state = 'team_fight';
      h.target = enemies[0].id;
      return;
    }

    // 中后期 / 主宰刷新 → 打野去
    const jungleAlive = this.jungle.find(j => j.hp > 0 && Math.hypot(j.x - h.x, j.z - h.z) < 15);
    const shouldJungle = h.role === 'warrior' || h.role === 'assassin';
    if (shouldJungle && jungleAlive && (h.hp > h.maxHp * 0.7) && h.state !== 'team_fight') {
      // 偶尔打野
      if (Math.random() < 0.25) {
        h.state = 'jungle';
        h.target = jungleAlive.id;
        return;
      }
    }

    // 推塔：附近敌方塔血量低或线上无敌方英雄
    const enemyTower = this._nearestEnemyTower(h);
    if (enemyTower && Math.hypot(enemyTower.x - h.x, enemyTower.z - h.z) < 8 && !this._isUnderEnemyTower(h.x, h.z, h.team)) {
      h.state = 'push';
      h.target = enemyTower.id;
      return;
    }

    // gank：刺客/战士在中路且附近有敌方英雄血量低
    if ((h.role === 'assassin' || h.role === 'warrior') && enemyHero && enemyHero.hp < enemyHero.maxHp * 0.5) {
      h.state = 'gank';
      h.target = enemyHero.id;
      return;
    }

    // 默认对线
    if (h.state === 'jungle' || h.state === 'gank' || h.state === 'team_fight' || h.state === 'push' || h.state === 'retreat' || h.state === 'return_to_lane') {
      // 检查是否需要回到线上
      const front = this._laneFrontPoint(h.lane, h.team);
      if (front && Math.hypot(front.x - h.x, front.z - h.z) > 15) {
        h.state = 'return_to_lane';
      } else {
        h.state = 'lane_fight';
      }
    }
    h.state = h.state || 'lane_fight';
  }

  _nearestEnemyHero(h) {
    let best = null, bestDist = 12;
    for (const e of this.heroes) {
      if (e.team === h.team || !e.alive) continue;
      const d = Math.hypot(e.x - h.x, e.z - h.z);
      if (d < bestDist) { best = e; bestDist = d; }
    }
    return best;
  }

  _nearestEnemyTower(h) {
    let best = null, bestDist = Infinity;
    for (const s of this.structures) {
      if (s.team === h.team || s.hp <= 0 || s.kind === 'main') continue;
      const d = Math.hypot(s.x - h.x, s.z - h.z);
      if (d < bestDist) { best = s; bestDist = d; }
    }
    return best;
  }

  _isInBase(h) {
    const base = h.team === 'blue' ? BLUE_BASE : RED_BASE;
    return Math.hypot(h.x - base.x, h.z - base.z) < 5;
  }

  // ==================== 强化学习（Q-Learning） ====================

  /** 惰性初始化 UnifiedRL Agent（P1-1：由 RLAgentManager 按注册表统一创建） */
  _rlEnsureAgent() {
    if (!this._rlShared) {
      this._rlShared = {
        agent: RLAgentManager.get().getAgent('moba_5v5', this),
        trainCount: 0,
        // P2-2 评估：本局起始时间
        episodeStartTs: performance.now(),
      };
      // P3-1 世界模型增强训练（想象回放，提升样本效率）
      RLAgentManager.get().enableWorldModel('moba_5v5', 0.5, 6);
    }
    return this._rlShared.agent;
  }

  /**
   * 提取状态键（以英雄视角归一化，蓝红共享同一张 Q 表）
   * 特征：hp/mp/最近敌方英雄距离/威胁/是否在敌方塔下/附近友敌数/
   *       兵线压力/经济差/等级差/附近野怪/游戏阶段
   */
  _rlExtractState(h) {
    const hpRatio = h.hp / h.maxHp;
    const mpRatio = h.mp / h.maxMp;
    // 最近敌方英雄
    let nearestEnemy = null, nearestEnemyDist = Infinity;
    for (const e of this.heroes) {
      if (e.team === h.team || !e.alive) continue;
      const d = Math.hypot(e.x - h.x, e.z - h.z);
      if (d < nearestEnemyDist) { nearestEnemy = e; nearestEnemyDist = d; }
    }
    // 敌方英雄距离桶
    let enh;
    if (!nearestEnemy) enh = 0;
    else if (nearestEnemyDist < 6) enh = 1;
    else if (nearestEnemyDist < 12) enh = 2;
    else if (nearestEnemyDist < 20) enh = 3;
    else enh = 0;
    // 敌方威胁（敌方血量比我低=可击杀，比我高=危险）
    let eth;
    if (!nearestEnemy) eth = 0;
    else {
      const eRatio = nearestEnemy.hp / nearestEnemy.maxHp;
      const myRatio = hpRatio;
      if (eRatio < myRatio - 0.2) eth = 1;       // 敌方更弱（可击杀）
      else if (eRatio > myRatio + 0.2) eth = 3;  // 敌方更强（危险）
      else eth = 2;                                // 势均力敌
    }
    // 是否在敌方塔下
    const uet = this._isUnderEnemyTower(h.x, h.z, h.team) ? 1 : 0;
    // 附近友方/敌方英雄数（12m内）
    let ally = 0, enemy = 0;
    for (const e of this.heroes) {
      if (e.id === h.id || !e.alive) continue;
      const d = Math.hypot(e.x - h.x, e.z - h.z);
      if (d > 12) continue;
      if (e.team === h.team) ally++; else enemy++;
    }
    const allyB = ally >= 2 ? 2 : ally;
    const enemyB = enemy >= 2 ? 2 : enemy;
    // 兵线压力：我方该路最前单位 vs 敌方该路最前单位
    const lane = this._lanePressure(h.lane, h.team);
    // 经济差：我方队伍经济 vs 敌方队伍经济
    const gold = this._teamGoldDiff(h.team);
    // 等级差：我方平均等级 vs 敌方平均等级
    const lvl = this._teamLevelDiff(h.team);
    // 附近野怪
    let jng = 0;
    for (const j of this.jungle) {
      if (j.hp <= 0) continue;
      if (Math.hypot(j.x - h.x, j.z - h.z) < 15) { jng = 1; break; }
    }
    // 游戏阶段
    const t = this._gameTime;
    const phase = t < 300 ? 0 : (t < 720 ? 1 : 2);
    return `${hpRatio < 0.3 ? 0 : hpRatio < 0.7 ? 1 : 2}|${mpRatio < 0.2 ? 0 : mpRatio < 0.6 ? 1 : 2}|${enh}|${eth}|${uet}|${allyB}|${enemyB}|${lane}|${gold}|${lvl}|${jng}|${phase}`;
  }

  /** 数值状态编码（12维 Float64Array，用于 UnifiedRLAgent 神经网络输入） */
  _rlEncodeState(h) {
    const hpRatio = h.hp / h.maxHp;
    const mpRatio = h.mp / h.maxMp;
    let nearestEnemy = null, nearestEnemyDist = Infinity;
    for (const e of this.heroes) {
      if (e.team === h.team || !e.alive) continue;
      const d = Math.hypot(e.x - h.x, e.z - h.z);
      if (d < nearestEnemyDist) { nearestEnemy = e; nearestEnemyDist = d; }
    }
    let enemyThreat = 0;
    if (nearestEnemy) {
      enemyThreat = nearestEnemy.hp / nearestEnemy.maxHp;
    }
    const uet = this._isUnderEnemyTower(h.x, h.z, h.team);
    let allyCount = 0, enemyCount = 0;
    for (const e of this.heroes) {
      if (e.id === h.id || !e.alive) continue;
      const d = Math.hypot(e.x - h.x, e.z - h.z);
      if (d > 12) continue;
      if (e.team === h.team) allyCount++; else enemyCount++;
    }
    const lane = this._lanePressure(h.lane, h.team);
    const gold = this._teamGoldDiff(h.team);
    const lvl = this._teamLevelDiff(h.team);
    let jng = false;
    for (const j of this.jungle) {
      if (j.hp <= 0) continue;
      if (Math.hypot(j.x - h.x, j.z - h.z) < 15) { jng = true; break; }
    }
    const t = this._gameTime;
    const phase = t < 300 ? 0 : (t < 720 ? 1 : 2);
    return new Float64Array([
      hpRatio,
      mpRatio,
      Math.min(nearestEnemyDist, 25) / 25,
      enemyThreat,
      uet ? 1 : 0,
      allyCount / 5,
      enemyCount / 5,
      lane / 2,
      gold / 2,
      lvl / 2,
      jng ? 1 : 0,
      phase / 2,
    ]);
  }

  /** 兵线压力：我方该路前线进度 vs 敌方 → 0落后 1均势 2领先 */
  _lanePressure(lane, team) {
    const path = LANES[lane];
    if (!path) return 1;
    const myFront = this._laneFrontPoint(lane, team);
    const enemyTeam = team === 'blue' ? 'red' : 'blue';
    const enemyFront = this._laneFrontPoint(lane, enemyTeam);
    if (!myFront || !enemyFront) return 1;
    const myProg = this._pathProgress(path, myFront.x, myFront.z);
    const enemyProg = this._pathProgress(path, enemyFront.x, enemyFront.z);
    const diff = team === 'blue' ? myProg - enemyProg : enemyProg - myProg;
    if (diff > 1.5) return 2;
    if (diff < -1.5) return 0;
    return 1;
  }

  /** 队伍经济差 → 0落后 1均势 2领先 */
  _teamGoldDiff(team) {
    const myGold = this.teamGold[team] || 0;
    const enemyGold = this.teamGold[team === 'blue' ? 'red' : 'blue'] || 0;
    const diff = myGold - enemyGold;
    if (diff > 1500) return 2;
    if (diff < -1500) return 0;
    return 1;
  }

  /** 队伍等级差 → 0落后 1均势 2领先 */
  _teamLevelDiff(team) {
    let myLvl = 0, enemyLvl = 0, myN = 0, enemyN = 0;
    for (const h of this.heroes) {
      if (h.team === team) { myLvl += h.level; myN++; }
      else { enemyLvl += h.level; enemyN++; }
    }
    const diff = (myN ? myLvl / myN : 1) - (enemyN ? enemyLvl / enemyN : 1);
    if (diff > 1.5) return 2;
    if (diff < -1.5) return 0;
    return 1;
  }

  /** 当前可行动作索引（根据局面过滤无意义动作，加速学习） */
  _rlGetValidActions(h) {
    const valid = [0]; // lane_fight 始终可选
    // retreat 始终可选
    valid.push(3);
    // return_to_lane：离线上远时可选
    const front = this._laneFrontPoint(h.lane, h.team);
    if (front && Math.hypot(front.x - h.x, front.z - h.z) > 12) valid.push(8);
    // recall：不在基地时可选
    if (!this._isInBase(h)) valid.push(4);
    // jungle：附近有活野怪
    if (this.jungle.some(j => j.hp > 0 && Math.hypot(j.x - h.x, j.z - h.z) < 15)) valid.push(5);
    // last_hit：附近有残血小兵
    if (this.minions.some(m => m.team !== h.team && m.hp > 0 && m.hp < m.maxHp * 0.4 && Math.hypot(m.x - h.x, m.z - h.z) < 8)) valid.push(1);
    // push：附近有敌方塔
    if (this.structures.some(s => s.team !== h.team && s.hp > 0 && s.kind === 'tower' && Math.hypot(s.x - h.x, s.z - h.z) < 12)) valid.push(2);
    // gank：附近有残血敌方英雄
    if (this.heroes.some(e => e.team !== h.team && e.alive && e.hp < e.maxHp * 0.5 && Math.hypot(e.x - h.x, e.z - h.z) < 12)) valid.push(6);
    // team_fight：附近 2+ 友方且 2+ 敌方
    const ally = this.heroes.filter(a => a.team === h.team && a.alive && a.id !== h.id && Math.hypot(a.x - h.x, a.z - h.z) < 12).length;
    const enemy = this.heroes.filter(e => e.team !== h.team && e.alive && Math.hypot(e.x - h.x, e.z - h.z) < 14).length;
    if (ally >= 2 && enemy >= 2) valid.push(7);
    return valid;
  }

  /**
   * RL 决策（替代规则 _decideBotState）
   * 流程：① 用累积奖励更新上一步 ② 提取新状态 ③ ε-greedy 选动作 ④ 执行
   */
  _rlDecideBotState(h) {
    const agent = this._rlEnsureAgent();
    const rl = h._rl;
    // ① 结算上一步：用累积奖励做 TD 更新
    const newState = this._rlEncodeState(h);
    if (rl.lastState !== null && rl.lastAction !== null) {
      const r = rl.accumReward + RL_REWARD.step;
      agent.store(rl.lastState, rl.lastAction, r, newState, false);
      agent.train();
      this._rlEpisodeReward += r;
    }
    rl.accumReward = 0;
    // ② 安全兜底：紧急情况强制 recall/retreat（不参与学习探索）
    let forced = null;
    if (h.hp < h.maxHp * 0.2 && !this._isInBase(h)) {
      forced = 4; // recall
    } else if (this._isUnderEnemyTower(h.x, h.z, h.team) && h.hp < h.maxHp * 0.5) {
      forced = 3; // retreat
    }
    // ③ 选动作
    let actionIdx;
    if (forced !== null) {
      actionIdx = forced;
    } else {
      const valid = this._rlGetValidActions(h);
      const result = agent.chooseAction(newState, valid);
      actionIdx = result.action;
      // LLM 宏观策略软引导：35% 概率按 LLM 偏置覆盖（不破坏 Q-Learning 探索）
      if (this._llmPolicyEnabled && this._llmRewardBias && Math.random() < 0.35) {
        const biased = this._llmPickBiased(valid);
        if (biased !== null) actionIdx = biased;
      }
    }
    // ④ 执行
    this._rlApplyAction(h, actionIdx);
    // 记录本步
    rl.lastState = newState;
    rl.lastAction = actionIdx;
  }

  /** 将动作索引映射到英雄行为（设置 h.state + h.target） */
  _rlApplyAction(h, actionIdx) {
    const name = MOBA_ACTIONS[actionIdx];
    switch (name) {
      case 'recall': {
        if (!this._isInBase(h) && h.state !== 'recall') {
          h.state = 'recall';
          h.recallTimer = RECALL_TIME;
          h.target = null;
        }
        break;
      }
      case 'retreat': {
        h.state = 'retreat';
        h.target = null;
        break;
      }
      case 'jungle': {
        let best = null, bestD = Infinity;
        for (const j of this.jungle) {
          if (j.hp <= 0) continue;
          const d = Math.hypot(j.x - h.x, j.z - h.z);
          if (d < bestD) { best = j; bestD = d; }
        }
        if (best) { h.state = 'jungle'; h.target = best.id; }
        else { h.state = 'lane_fight'; h.target = null; }
        break;
      }
      case 'last_hit': {
        // 找残血小兵
        let best = null, bestD = Infinity;
        for (const m of this.minions) {
          if (m.team === h.team || m.hp <= 0 || m.hp >= m.maxHp * 0.4) continue;
          const d = Math.hypot(m.x - h.x, m.z - h.z);
          if (d < bestD) { best = m; bestD = d; }
        }
        h.state = best ? 'last_hit' : 'lane_fight';
        h.target = best ? best.id : null;
        break;
      }
      case 'push': {
        let best = null, bestD = Infinity;
        for (const s of this.structures) {
          if (s.team === h.team || s.hp <= 0 || s.kind === 'main') continue;
          const d = Math.hypot(s.x - h.x, s.z - h.z);
          if (d < bestD) { best = s; bestD = d; }
        }
        h.state = best ? 'push' : 'lane_fight';
        h.target = best ? best.id : null;
        break;
      }
      case 'gank': {
        let best = null, bestD = Infinity;
        for (const e of this.heroes) {
          if (e.team === h.team || !e.alive || e.hp >= e.maxHp * 0.5) continue;
          const d = Math.hypot(e.x - h.x, e.z - h.z);
          if (d < bestD) { best = e; bestD = d; }
        }
        h.state = best ? 'gank' : 'lane_fight';
        h.target = best ? best.id : null;
        break;
      }
      case 'team_fight': {
        const e = this._nearestEnemyHero(h);
        h.state = 'team_fight';
        h.target = e ? e.id : null;
        break;
      }
      case 'return_to_lane': {
        h.state = 'return_to_lane';
        h.target = null;
        break;
      }
      case 'lane_fight':
      default: {
        h.state = 'lane_fight';
        h.target = null;
        break;
      }
    }
    h.stateTimer = 0;
  }

  /** 累积奖励到英雄的 RL 跟踪器 */
  _rlAward(h, amount) {
    if (!this._rlEnabled || !h || !h._rl) return;
    h._rl.accumReward += amount;
    // LLM 策略层：累积蓝方奖励，供下次决策请求回传给记忆库
    if (this._llmPolicyEnabled && h.team === 'blue') {
      this._llmAccumReward += amount;
    }
  }

  /** 主水晶伤害增量奖励（每决策步结算） */
  _rlStepMainHpReward() {
    if (!this._rlEnabled) return;
    const blueMain = this.structures.find(s => s.id === 'blue_main');
    const redMain = this.structures.find(s => s.id === 'red_main');
    if (!blueMain || !redMain) return;
    if (!this._rlPrevMainHpSet) {
      this._rlLastMainHp.blue = blueMain.hp;
      this._rlLastMainHp.red = redMain.hp;
      this._rlPrevMainHpSet = true;
      return;
    }
    const blueDelta = blueMain.hp - this._rlLastMainHp.blue; // 蓝方水晶掉的血
    const redDelta = redMain.hp - this._rlLastMainHp.red;    // 红方水晶掉的血
    if (blueDelta < 0) {
      // 蓝方水晶被伤 → 红方奖励 / 蓝方惩罚
      const r = RL_REWARD.main_hurt * (-blueDelta);
      for (const h of this.heroes) if (h.team === 'blue' && h.alive) this._rlAward(h, r);
      for (const h of this.heroes) if (h.team === 'red' && h.alive) this._rlAward(h, -r);
    }
    if (redDelta < 0) {
      const r = RL_REWARD.main_dmg * (-redDelta);
      for (const h of this.heroes) if (h.team === 'red' && h.alive) this._rlAward(h, r);
      for (const h of this.heroes) if (h.team === 'blue' && h.alive) this._rlAward(h, -r);
    }
    this._rlLastMainHp.blue = blueMain.hp;
    this._rlLastMainHp.red = redMain.hp;
  }

  /** 局终：发放终局奖励 + 收尾上一步 TD + agent.endEpisode */
  _rlEndMatch(winner) {
    if (!this._rlEnabled) return;
    const agent = this._rlEnsureAgent();
    // 终局奖励 + 收尾上一步（不跳过死亡英雄，确保 TD 序列完整）
    for (const h of this.heroes) {
      const win = (h.team === winner);
      this._rlAward(h, win ? RL_REWARD.win : RL_REWARD.lose);
      const rl = h._rl;
      if (rl.lastState !== null && rl.lastAction !== null) {
        const r = rl.accumReward + RL_REWARD.step;
        agent.store(rl.lastState, rl.lastAction, r, '__terminal__', true);
        agent.train();
        this._rlEpisodeReward += r;
      }
      rl.lastState = null;
      rl.lastAction = null;
      rl.accumReward = 0;
    }
    agent.endEpisode(this._rlEpisodeReward, { win: winner === 'blue', winner });
    // P2-2 人类限时评估基准：记录本局耗时与胜负
    RLAgentManager.get().getBaselineEvaluator().recordEpisode('moba_5v5', {
      durationMs: performance.now() - this._rlShared.episodeStartTs,
      win: winner === 'blue',
    });
    this._rlShared.episodeStartTs = performance.now();
    this._rlEpisodeReward = 0;
    this._rlPrevMainHpSet = false;
  }

  /** 重置每局 RL 跟踪（onStart 时调用） */
  _rlResetEpisode() {
    if (!this._rlEnabled) return;
    for (const h of this.heroes) {
      if (h._rl) { h._rl.lastState = null; h._rl.lastAction = null; h._rl.accumReward = 0; }
    }
    this._rlEpisodeReward = 0;
    this._rlPrevMainHpSet = false;
  }

  // ==================== LLM 宏观策略层（LLM-as-policy）====================
  // LLM 作为战术指挥官，每 N 秒输出一个宏观策略，通过奖励偏置软引导 Q-Learning 的微观决策。
  // 流程：定时构建状态描述 → 发 game_action_request → 后端 LLM 决策 → 回 game_action_response
  //       → 解析策略 → 设置动作偏置 → Q-Learning 在偏置下选动作 → 累积奖励 → 下次请求回传

  /** LLM 策略定时器（真实时间，不受加速影响） */
  _llmStepPolicy(dt) {
    if (!this._llmPolicyEnabled) return;
    this._llmPolicyTimer += dt;
    if (this._llmPolicyTimer >= this._llmPolicyInterval && !this._llmWaitingResponse) {
      this._llmPolicyTimer = 0;
      this._llmRequestAction();
    }
  }

  /** 构建自然语言状态描述（供 LLM 理解） */
  _llmBuildStateText() {
    const t = Math.floor(this._gameTime);
    const phase = t < 180 ? '前期(对线期)' : t < 600 ? '中期(游走期)' : '后期(团战期)';
    const blueAlive = this.heroes.filter(h => h.team === 'blue' && h.alive).length;
    const redAlive = this.heroes.filter(h => h.team === 'red' && h.alive).length;
    const goldDiff = this.teamGold.blue - this.teamGold.red;
    const goldDesc = goldDiff > 1000 ? `蓝方经济领先${goldDiff}` : goldDiff < -1000 ? `蓝方经济落后${-goldDiff}` : '经济均势';
    // 暴君/主宰状态
    const dragon = this.jungle.find(j => j.type === '暴君' || j.type === '主宰');
    const dragonDesc = dragon && dragon.hp > 0 ? '暴君存活(可拿)' : '暴君已被击杀';
    // 各路塔状态
    const blueTowers = this.structures.filter(s => s.team === 'blue' && s.kind === 'tower' && s.hp > 0).length;
    const redTowers = this.structures.filter(s => s.team === 'red' && s.kind === 'tower' && s.hp > 0).length;
    return `${phase}，游戏${Math.floor(t / 60)}分${t % 60}秒。` +
      `击杀比 蓝方${this.teamKills.blue}:${this.teamKills.red}红方。` +
      `存活 蓝方${blueAlive}人 红方${redAlive}人。${goldDesc}。` +
      `防御塔 蓝方${blueTowers}座 红方${redTowers}座。${dragonDesc}。`;
  }

  /** 构建全局状态键（12维，与 RL 状态键同格式，用于记忆检索相似度匹配） */
  _llmBuildStateKey() {
    const blueHeroes = this.heroes.filter(h => h.team === 'blue' && h.alive);
    const redHeroes = this.heroes.filter(h => h.team === 'red' && h.alive);
    const avgHp = blueHeroes.length ? blueHeroes.reduce((s, h) => s + h.hp / h.maxHp, 0) / blueHeroes.length : 0;
    const avgMp = blueHeroes.length ? blueHeroes.reduce((s, h) => s + h.mp / h.maxMp, 0) / blueHeroes.length : 0;
    const hpB = avgHp < 0.3 ? 0 : avgHp < 0.7 ? 1 : 2;
    const mpB = avgMp < 0.2 ? 0 : avgMp < 0.6 ? 1 : 2;
    const enh = 0;                          // 全局视角无单兵距离概念
    const eth = this._teamGoldDiff('blue'); // 用经济差代理威胁
    const uet = 0;
    const ally = Math.min(2, blueHeroes.length);
    const enemy = Math.min(2, redHeroes.length);
    const lane = this._lanePressure('mid', 'blue');
    const gold = this._teamGoldDiff('blue');
    const lvl = this._teamLevelDiff('blue');
    const dragon = this.jungle.find(j => (j.type === '暴君' || j.type === '主宰') && j.hp > 0);
    const jng = dragon ? 1 : 0;
    const t = this._gameTime;
    const phase = t < 300 ? 0 : (t < 720 ? 1 : 2);
    return `${hpB}|${mpB}|${enh}|${eth}|${uet}|${ally}|${enemy}|${lane}|${gold}|${lvl}|${jng}|${phase}`;
  }

  /** 发送 LLM 决策请求 */
  _llmRequestAction() {
    if (!this.App.ws || this.App.ws.readyState !== WebSocket.OPEN) return;
    const stateText = this._llmBuildStateText();
    const stateKey = this._llmBuildStateKey();
    const candidates = ['group_push_mid', 'split_push', 'take_dragon', 'defend', 'team_fight', 'recall_regroup', 'ambush'];
    const lastReward = this._llmAccumReward;
    this._llmAccumReward = 0;
    this._llmWaitingResponse = true;
    this._llmStats.requests++;
    this.App.ws.send(JSON.stringify({
      type: 'game_action_request',
      data: { state_text: stateText, state_key: stateKey, candidates, last_reward: lastReward },
    }));
  }

  /** 接收 LLM 决策响应（由前端 WS 消息分发调用） */
  onLLMActionResponse(data) {
    this._llmWaitingResponse = false;
    this._llmStats.responses++;
    if (!data || !data.strategy) return;
    this._llmApplyStrategy(data.strategy, data.speak || '', data.reason || '');
  }

  /** 应用宏观策略：设置动作偏置 + 显示指挥语音 */
  _llmApplyStrategy(strategy, speak, reason) {
    this._llmStrategy = strategy;
    this._llmRewardBias = this._strategyToBias(strategy);
    // 显示策略到 UI（不触发 sendAIAction，避免对话流打断玩家操控）
    this._llmUpdateStrategyUI(strategy, speak);
  }

  /** 宏观策略 → 微观动作偏置映射 {actionIdx: bonus} */
  _strategyToBias(strategy) {
    switch (strategy) {
      case 'group_push_mid':  return { 7: 1.5, 2: 1.0 };           // team_fight, push
      case 'split_push':      return { 2: 1.5, 8: 0.3 };           // push, return_to_lane
      case 'take_dragon':     return { 5: 2.0 };                   // jungle
      case 'defend':          return { 3: 1.5 };                   // retreat
      case 'team_fight':      return { 7: 2.0, 6: 0.5 };           // team_fight, gank
      case 'recall_regroup':  return { 4: 2.0 };                   // recall
      case 'ambush':          return { 6: 2.0 };                   // gank
      default:                return null;
    }
  }

  /** 从可行动作中选 LLM 偏置最高的（软引导：35% 概率覆盖 Q-Learning 选择） */
  _llmPickBiased(valid) {
    if (!this._llmRewardBias) return null;
    let bestA = null, bestBias = 0;
    for (const a of valid) {
      const b = this._llmRewardBias[a] || 0;
      if (b > bestBias) { bestBias = b; bestA = a; }
    }
    return bestBias > 0 ? bestA : null;
  }

  /** 更新策略 UI 显示 */
  _llmUpdateStrategyUI(strategy, speak) {
    const el = document.getElementById('moba-llm-strategy');
    if (el) {
      const labels = {
        group_push_mid: '🛡️ 抱团推中', split_push: '⚔️ 分带推塔',
        take_dragon: '🐉 拿龙', defend: '🛡️ 防守',
        team_fight: '💥 开团', recall_regroup: '🏰 回城重组',
        ambush: '🎭 埋伏',
      };
      el.textContent = labels[strategy] || strategy;
      if (speak) el.textContent += ` "${speak}"`;
    }
  }

  /** 加速自我对弈：重开一局（不重建场景，仅重置游戏数据，复用现有3D对象） */
  _rlRestartMatch() {
    // 重置游戏数据：复用现有 structures/jungle 对象（保留血条等附加属性），仅回满血
    for (const s of this.structures) s.hp = s.maxHp;
    for (const j of this.jungle) {
      j.hp = j.maxHp;
      if (j.respawn) j.respawn = 0;
    }
    this.minions = [];
    this.projectiles = [];
    this.effects = [];
    this.killFeed = [];
    this.teamGold = { blue: 0, red: 0 };
    this.teamKills = { blue: 0, red: 0 };
    this._gameTime = 0;
    this._nextMinionWave = MINION_SPAWN_DELAY;
    this._botDecisionTimer = 0;
    this.state = 'playing';
    this.startTime = performance.now();
    this.elapsedTime = 0;
    this.events = [];
    this._lastEventIndex = 0;
    // 重置英雄
    for (const h of this.heroes) {
      // 先重置基础属性，再回满血
      const def = HERO_DEFS[h.heroKey];
      h.maxHp = def.hp; h.maxMp = def.mp; h.atk = def.atk;
      h.armor = def.armor; h.mr = def.mr;
      h.hp = h.maxHp; h.mp = h.maxMp;
      h.alive = true; h.state = 'lane_fight'; h.stateTimer = 0;
      h.target = null; h.recallTimer = 0; h.respawnTimer = 0;
      h.kills = 0; h.deaths = 0; h.assists = 0;
      h.skill1Cd = 0; h.skill2Cd = 0; h.ultCd = 0;
      h.level = 1; h.exp = 0; h.gold = 500; h.equipLevel = 0;
      h.totalGoldEarned = 0; h.appliedGoldPower = 0;
      h.buffs = []; h.attackCd = 0; h.lastDamager = null;
      const base = h.team === 'blue' ? BLUE_BASE : RED_BASE;
      const idx = parseInt(h.id.split('_')[1], 10) || 0;
      const off = (idx / 5) * Math.PI * 2;
      h.x = base.x + Math.cos(off) * 3;
      h.z = base.z + Math.sin(off) * 3;
      h.facing = h.team === 'blue' ? Math.PI / 4 : -Math.PI * 3 / 4;
      this._applyEquipBonus(h);
      if (h.mesh) { h.mesh.visible = !h.isPlayer; h.mesh.position.set(h.x, 0, h.z); h.mesh.rotation.y = h.facing; }
    }
    this._rlResetEpisode();
    this._pushEvent('match_start', { composition: COMPOSITION, auto_restart: true });
  }

  // ==================== 强化学习 END ====================

  // ==================== 技能系统 ====================

  _botTryCastSkill(h, target, dist) {
    // 大招
    if (h.ultCd <= 0 && h.ultLevel >= 1 && dist < h.attackRange + 4 && Math.random() < 0.3) {
      this._castSkill(h, 3, target);
      return;
    }
    if (h.skill1Cd <= 0 && dist < h.attackRange + 3 && Math.random() < 0.4) {
      this._castSkill(h, 1, target);
      return;
    }
    if (h.skill2Cd <= 0 && dist < h.attackRange + 3 && Math.random() < 0.4) {
      this._castSkill(h, 2, target);
      return;
    }
  }

  _castSkill(hero, slot, target) {
    const def = HERO_DEFS[hero.heroKey];
    if (slot === 1 && hero.skill1Cd > 0) return false;
    if (slot === 2 && hero.skill2Cd > 0) return false;
    if (slot === 3 && (hero.ultCd > 0 || hero.ultLevel < 1)) return false;
    if (hero.mp < 30) return false;

    const lvl = slot === 1 ? hero.skill1Level : slot === 2 ? hero.skill2Level : hero.ultLevel;
    const baseDmg = hero.atk * (slot === 3 ? 2.5 : 1.2) + lvl * 60;
    const range = hero.attackRange + (slot === 3 ? 5 : 3);

    // 目标命中判定位置（无目标时朝朝向方向释放）
    const aimX = target ? target.x : hero.x + Math.sin(hero.facing) * range;
    const aimZ = target ? target.z : hero.z + Math.cos(hero.facing) * range;
    const inRange = target ? Math.hypot(target.x - hero.x, target.z - hero.z) <= range : true;

    // 根据职业释放不同效果
    switch (hero.heroKey) {
      case 'warrior':
        // 重击/冲撞/狂战之怒
        if (slot === 1) {
          if (target && inRange) {
            this._dealDamage(hero, target, baseDmg, true);
          }
          // 无目标也产生小范围伤害
          for (const e of this._enemiesInRadiusAt(hero, aimX, aimZ, 2)) this._dealDamage(hero, e, baseDmg * 0.6, true);
          this._spawnSkillEffect(aimX, aimZ, 0xff4444, 1.5);
        }
        if (slot === 2) { /* 冲撞：向目标/朝向方向突进 */
          this._dashToward(hero, target, 4);
        }
        if (slot === 3) { /* 群体伤害 */
          for (const e of this._enemiesInRadius(hero, 4)) this._dealDamage(hero, e, baseDmg, true);
          this._spawnSkillEffect(hero.x, hero.z, 0xff6666, 4);
        }
        break;
      case 'mage':
        // 火球/冰霜/陨石
        if (slot === 1) {
          if (target && inRange) {
            this._dealDamage(hero, target, baseDmg, true);
            this._spawnProjectile(hero, target, 0xff6600);
          } else {
            // 无目标：朝朝向方向发射火球（范围伤害）
            for (const e of this._enemiesInRadiusAt(hero, aimX, aimZ, 2.5)) this._dealDamage(hero, e, baseDmg, true);
            this._spawnSkillEffect(aimX, aimZ, 0xff6600, 2);
          }
        }
        if (slot === 2) { /* 冰霜：减速 + 范围伤害 */
          if (target && inRange) { target._slowTimer = 2; this._dealDamage(hero, target, baseDmg * 0.5, true); }
          for (const e of this._enemiesInRadiusAt(hero, aimX, aimZ, 2.5)) { e._slowTimer = 2; this._dealDamage(hero, e, baseDmg * 0.5, true); }
          this._spawnSkillEffect(aimX, aimZ, 0x66ccff, 2.5);
        }
        if (slot === 3) { /* 陨石：范围伤害 */
          for (const e of this._enemiesInRadiusAt(hero, aimX, aimZ, 4)) this._dealDamage(hero, e, baseDmg * 1.5, true);
          this._spawnSkillEffect(aimX, aimZ, 0xff4400, 4);
        }
        break;
      case 'assassin':
        // 影袭/瞬步/死亡印记
        if (slot === 2) { this._dashToward(hero, target, 5); }
        if (slot === 1) {
          if (target && inRange) {
            this._dealDamage(hero, target, baseDmg, true);
          }
          for (const e of this._enemiesInRadiusAt(hero, aimX, aimZ, 1.5)) this._dealDamage(hero, e, baseDmg * 0.5, true);
          this._spawnSkillEffect(aimX, aimZ, 0x44cccc, 1.2);
        }
        if (slot === 3) {
          if (target && inRange) {
            this._dealDamage(hero, target, baseDmg * 2, true);
            this._spawnSkillEffect(target.x, target.z, 0x44cccc, 1.2);
          } else {
            for (const e of this._enemiesInRadiusAt(hero, aimX, aimZ, 3)) this._dealDamage(hero, e, baseDmg, true);
            this._spawnSkillEffect(aimX, aimZ, 0x44cccc, 3);
          }
        }
        break;
      case 'marksman':
        // 穿甲箭/翻滚/箭雨
        if (slot === 2) { /* 翻滚：向前方短距位移 */
          const dirx = Math.sin(hero.facing), dirz = Math.cos(hero.facing);
          hero.x += dirx * 3; hero.z += dirz * 3;
          this._clampToMap(hero);
        }
        if (slot === 1) {
          if (target && inRange) {
            this._dealDamage(hero, target, baseDmg, true);
            this._spawnProjectile(hero, target, 0xffaa33);
          } else {
            // 无目标：朝朝向方向穿透伤害
            for (const e of this._enemiesInRadiusAt(hero, aimX, aimZ, 2)) this._dealDamage(hero, e, baseDmg, true);
            this._spawnSkillEffect(aimX, aimZ, 0xffaa33, 2);
          }
        }
        if (slot === 3) { /* 箭雨 */
          for (const e of this._enemiesInRadiusAt(hero, aimX, aimZ, 4)) this._dealDamage(hero, e, baseDmg, true);
          this._spawnSkillEffect(aimX, aimZ, 0xffcc44, 4);
        }
        break;
      case 'support':
        // 治疗/护盾/圣光
        if (slot === 1) { /* 治疗：附近友方 */
          for (const a of this._alliesInRadius(hero, 6)) {
            a.hp = Math.min(a.maxHp, a.hp + baseDmg * 0.8);
            this._spawnSkillEffect(a.x, a.z, 0x44ff66, 1.5);
          }
          this._spawnSkillEffect(hero.x, hero.z, 0x44ff66, 2);
        }
        if (slot === 2) { /* 护盾：自身或最低血量友方 */
          const low = this._lowestAlly(hero) || hero;
          low._shield = (low._shield || 0) + baseDmg;
          this._spawnSkillEffect(low.x, low.z, 0x88ff88, 1.5);
        }
        if (slot === 3) { /* 圣光：伤害+治疗 */
          if (target && inRange) {
            this._dealDamage(hero, target, baseDmg, true);
          }
          for (const e of this._enemiesInRadiusAt(hero, aimX, aimZ, 3)) this._dealDamage(hero, e, baseDmg * 0.8, true);
          for (const a of this._alliesInRadius(hero, 8)) a.hp = Math.min(a.maxHp, a.hp + baseDmg * 0.5);
          this._spawnSkillEffect(aimX, aimZ, 0xffff88, 2.5);
        }
        break;
    }

    // CD 与蓝耗
    if (slot === 1) { hero.skill1Cd = 6; hero.mp -= 30; }
    if (slot === 2) { hero.skill2Cd = 8; hero.mp -= 40; }
    if (slot === 3) { hero.ultCd = 60; hero.mp -= 80; }

    if (hero.isPlayer) {
      this._pushEvent('skill_cast', { hero: hero.id, slot, target: target ? target.id : null });
    }
    return true;
  }

  _dashToward(hero, target, dist) {
    let dx, dz;
    if (target) {
      dx = target.x - hero.x; dz = target.z - hero.z;
      const d = Math.hypot(dx, dz);
      if (d < 0.01) return;
      hero.x += (dx / d) * dist;
      hero.z += (dz / d) * dist;
    } else {
      // 无目标：朝朝向方向位移
      hero.x += Math.sin(hero.facing) * dist;
      hero.z += Math.cos(hero.facing) * dist;
    }
    this._clampToMap(hero);
  }

  _enemiesInRadius(hero, r) {
    return this._enemiesInRadiusAt(hero, hero.x, hero.z, r);
  }
  _enemiesInRadiusAt(hero, x, z, r) {
    const list = [];
    for (const e of this.heroes) {
      if (e.team === hero.team || !e.alive) continue;
      if (Math.hypot(e.x - x, e.z - z) <= r) list.push(e);
    }
    for (const m of this.minions) {
      if (m.team === hero.team || m.hp <= 0) continue;
      if (Math.hypot(m.x - x, m.z - z) <= r) list.push(m);
    }
    // 包含敌方建筑（塔/水晶）
    for (const s of this.structures) {
      if (s.team === hero.team || s.hp <= 0) continue;
      if (Math.hypot(s.x - x, s.z - z) <= r) list.push(s);
    }
    // 野怪（中立，可被技能攻击）
    for (const j of this.jungle) {
      if (j.hp <= 0) continue;
      if (Math.hypot(j.x - x, j.z - z) <= r) list.push(j);
    }
    return list;
  }
  _alliesInRadius(hero, r) {
    const list = [];
    for (const a of this.heroes) {
      if (a.team !== hero.team || !a.alive || a.id === hero.id) continue;
      if (Math.hypot(a.x - hero.x, a.z - hero.z) <= r) list.push(a);
    }
    list.push(hero);
    return list;
  }
  _lowestAlly(hero) {
    let best = null, bestRatio = 1;
    for (const a of this.heroes) {
      if (a.team !== hero.team || !a.alive) continue;
      const ratio = a.hp / a.maxHp;
      if (ratio < bestRatio) { best = a; bestRatio = ratio; }
    }
    return best;
  }

  // ==================== 玩家操控 ====================

  /** 是否为范围攻击型英雄（非射手/非法师的近战 & 法师范围普攻） */
  _isAoeAttacker(h) {
    // 射手单点远程；其余职业普攻为范围伤害
    return h.heroKey !== 'marksman';
  }

  /** 范围普攻：在目标点附近造成 AoE 伤害（非射手/法师） */
  _playerAttackAoe(h, aimX, aimZ, mainTarget) {
    const aoeRadius = h.heroKey === 'mage' ? 2.0 : 1.8;
    const dmgRatio = h.heroKey === 'mage' ? 0.75 : 0.85; // 范围伤害略低
    this._spawnSkillEffect(aimX, aimZ, h.heroKey === 'mage' ? 0x9966ff : 0xffee44, aoeRadius);
    let hitAny = false;
    for (const e of [...this.heroes, ...this.minions, ...this.structures, ...this.jungle]) {
      // 野怪（kind:'jungle'）不分阵营，玩家可攻击己方野区；其余单位需过滤友军
      if (e.kind !== 'jungle' && e.team === h.team) continue;
      if (e.hp <= 0 || e.alive === false) continue;
      const d = Math.hypot(e.x - aimX, e.z - aimZ);
      if (d <= aoeRadius) {
        // 主目标全额，次目标减伤
        const dmg = (e === mainTarget) ? h.atk : h.atk * dmgRatio;
        this._dealDamage(h, e, dmg, false);
        hitAny = true;
      }
    }
    return hitAny;
  }

  _playerAutoAttack(h, dt) {
    if (h.attackCd > 0) return;
    // 自动寻找最近敌人普攻
    const target = this._findHeroTarget(h);
    if (!target) return;
    const d = Math.hypot(target.x - h.x, target.z - h.z);
    // 目标超出攻击范围时不自动追击（避免与 WASD 操控冲突），玩家需手动靠近
    if (d > h.attackRange + 0.5) return;
    h.facing = Math.atan2(target.x - h.x, target.z - h.z);
    h.attackCd = 0.5;
    if (this._isAoeAttacker(h)) {
      // 范围普攻
      this._playerAttackAoe(h, target.x, target.z, target);
    } else {
      // 射手单点远程
      this._dealDamage(h, target, h.atk, false);
      this._spawnProjectile(h, target, 0xffee44);
    }
  }

  /** 玩家手动普攻：立即朝最近敌人/朝向方向攻击 */
  _playerManualAttack() {
    const h = this.heroes.find(h => h.id === this.playerHeroId);
    if (!h || !h.alive) return;
    if (h.attackCd > 0) {
      this.App.showToast('普攻冷却中');
      return;
    }
    const isAoe = this._isAoeAttacker(h);
    // 优先最近敌人（扩大搜索范围）
    const target = this._findHeroTarget(h) || this._facingEnemy(h, 12);
    if (target) {
      const d = Math.hypot(target.x - h.x, target.z - h.z);
      h.facing = Math.atan2(target.x - h.x, target.z - h.z);
      if (d <= h.attackRange + 1.5) {
        h.attackCd = 0.5;
        if (isAoe) {
          this._playerAttackAoe(h, target.x, target.z, target);
        } else {
          this._dealDamage(h, target, h.atk, false);
          this._spawnProjectile(h, target, 0xffee44);
        }
      } else {
        // 目标稍远：远程发射投射物 / 近战突进后范围攻击
        h.attackCd = 0.5;
        if (h.attackRange > 3) {
          this._spawnProjectile(h, target, 0xffee44);
        } else {
          this._dashToward(h, target, Math.min(2, d - h.attackRange));
          if (isAoe) {
            this._playerAttackAoe(h, target.x, target.z, target);
          } else {
            this._dealDamage(h, target, h.atk, false);
          }
        }
      }
    } else {
      // 无目标：朝朝向方向攻击
      h.attackCd = 0.5;
      const fx = h.x + Math.sin(h.facing) * (h.attackRange + 1);
      const fz = h.z + Math.cos(h.facing) * (h.attackRange + 1);
      if (isAoe) {
        this._playerAttackAoe(h, fx, fz, null);
      } else {
        this._spawnSkillEffect(fx, fz, 0xffee44, 1.2);
        for (const e of [...this.heroes, ...this.minions, ...this.structures, ...this.jungle]) {
          if (e.kind !== 'jungle' && e.team === h.team) continue;
          if (e.hp <= 0 || e.alive === false) continue;
          const d = Math.hypot(e.x - fx, e.z - fz);
          if (d < 1.5) this._dealDamage(h, e, h.atk * 0.8, false);
        }
      }
    }
  }

  _processPlayerSkillInput() {
    while (this._skillInputQueue.length > 0) {
      const slot = this._skillInputQueue.shift();
      const h = this.heroes.find(h => h.id === this.playerHeroId);
      if (!h || !h.alive) continue;
      // 扩大目标搜索范围：最近敌人 + 朝向方向敌人
      const target = this._findHeroTarget(h) || this._facingEnemy(h, 14);
      const ok = this._castSkill(h, slot, target);
      if (!ok) {
        // 技能未释放：给出反馈
        if (h.mp < 30) this.App.showToast('蓝量不足');
        else if (slot === 3 && h.ultLevel < 1) this.App.showToast('大招未解锁(需4级)');
        else if (slot === 1 && h.skill1Cd > 0) this.App.showToast('技能冷却中');
        else if (slot === 2 && h.skill2Cd > 0) this.App.showToast('技能冷却中');
        else if (slot === 3 && h.ultCd > 0) this.App.showToast('大招冷却中');
        else if (!target) this.App.showToast('技能范围内无目标');
      }
    }
  }

  _facingEnemy(hero, maxDist) {
    // 朝向方向最近的敌人（放宽角度到 1.2 弧度，更易命中）
    // 野怪不分阵营，玩家可攻击己方野区
    let best = null, bestD = maxDist;
    for (const e of [...this.heroes, ...this.minions, ...this.structures, ...this.jungle]) {
      if (e.kind !== 'jungle' && e.team === hero.team) continue;
      if (e.hp <= 0) continue;
      if (e.alive === false) continue;
      const dx = e.x - hero.x, dz = e.z - hero.z;
      const d = Math.hypot(dx, dz);
      if (d > maxDist) continue;
      const ang = Math.atan2(dx, dz);
      let diff = Math.abs(((ang - hero.facing + Math.PI) % (Math.PI * 2)) - Math.PI);
      if (diff < 1.2 && d < bestD) { best = e; bestD = d; }
    }
    return best;
  }

  onUserInput(type, data) {
    if (type === 'skill') {
      this._skillInputQueue.push(data.slot);
    } else if (type === 'recall') {
      const h = this.heroes.find(h => h.id === this.playerHeroId);
      if (h && h.alive && h.state !== 'recall') {
        h.state = 'recall';
        h.recallTimer = RECALL_TIME;
        this.App.showToast('回城中...');
      }
    } else if (type === 'switch_hero') {
      this._switchPlayerHero(data.index);
    } else if (type === 'toggle_spectate') {
      this._toggleSpectate();
    }
  }

  _switchPlayerHero(index) {
    const blueHeroes = this.heroes.filter(h => h.team === 'blue');
    const newHero = blueHeroes[index];
    if (!newHero) return;
    const oldHero = this.heroes.find(h => h.id === this.playerHeroId);
    if (oldHero) oldHero.isPlayer = false;
    this.playerHeroId = newHero.id;
    newHero.isPlayer = true;
    this.spectating = false;
    const avatar = this.App.currentAvatar;
    if (avatar) {
      avatar.position.set(newHero.x, 0, newHero.z);
      this.App.smoothRotY = newHero.facing;
    }
    this.App.showToast(`切换到 ${newHero.name}`);
    this._refreshSkillNames();
    this._syncPlayerMoveSpeed();
  }

  _toggleSpectate() {
    this.spectating = !this.spectating;
    if (!this.spectating) {
      const h = this.heroes.find(h => h.id === this.playerHeroId);
      if (h && this.App.currentAvatar) {
        this.App.currentAvatar.position.set(h.x, 0, h.z);
        this.App.smoothRotY = h.facing;
      }
    }
    this.App.showToast(this.spectating ? '观战模式' : '操控模式');
  }

  // ==================== 战斗与伤害 ====================

  _dealDamage(attacker, target, amount, isSkill) {
    if (!target || (target.hp !== undefined && target.hp <= 0)) return;
    let dmg = amount;
    // 防御减伤
    if (target.armor !== undefined) {
      const reduction = target.armor / (target.armor + 100);
      dmg *= (1 - reduction);
    }
    // 护盾
    if (target._shield && target._shield > 0) {
      const absorbed = Math.min(target._shield, dmg);
      target._shield -= absorbed;
      dmg -= absorbed;
    }
    dmg = Math.max(1, Math.round(dmg));
    target.hp -= dmg;
    if (target.lastDamager !== undefined) target.lastDamager = attacker.id || attacker;
    // 记录英雄受击时间（用于脱战回血判定）
    if (target.kind === 'hero' && target.lastDamagedTime !== undefined) {
      target.lastDamagedTime = this._gameTime;
    }

    // 野怪被攻击时反击（kind:'jungle' 统一判断，兼容 buff 为 null 的新类型）
    if (target.kind === 'jungle' && target.attackCd <= 0) {
      // 野怪反击攻击者
      const hero = attacker.id && this.heroes.find(h => h.id === attacker.id);
      if (hero && hero.alive) {
        target.attackCd = 1.2;
        hero.hp -= target.atk * (1 - hero.armor / (hero.armor + 100));
        hero.lastDamager = target.id;
        if (hero.hp <= 0) this._handleHeroDeath(hero, target);
      }
    }

    // 目标死亡处理
    if (target.hp <= 0) {
      if (target.kind === 'minion') {
        this._onMinionKilled(target, attacker);
      } else if (target.kind === 'jungle') {
        this._onJungleKilled(target, attacker);
      } else if (target.kind === 'tower' || target.kind === 'main') {
        this._onStructureKilled(target, attacker);
      } else if (target.id && this.heroes.includes(target)) {
        this._handleHeroDeath(target, attacker);
      }
    }
  }

  _onMinionKilled(minion, killer) {
    const hero = (killer && killer.id) ? this.heroes.find(h => h.id === killer.id) : null;
    if (hero && hero.alive) {
      this._grantGold(hero, REWARD.minion_kill.gold);
      this._addExp(hero, REWARD.minion_kill.exp);
      this._rlAward(hero, RL_REWARD.minion_kill);
    }
  }

  /** 发放金币并累计 totalGoldEarned（用于经济转战力） */
  _grantGold(hero, amount) {
    hero.gold += amount;
    hero.totalGoldEarned += amount;
    this.teamGold[hero.team] += amount;
  }

  /** 经济转战力：每 100 累计金币提供 +2atk +20hp +2armor */
  _applyGoldPower(h) {
    const power = Math.floor(h.totalGoldEarned / 100);
    if (power === h.appliedGoldPower) return;
    const delta = power - h.appliedGoldPower;
    h.atk += delta * 2;
    h.maxHp += delta * 20;
    h.armor += delta * 2;
    h.hp += delta * 20;
    h.appliedGoldPower = power;
  }

  _onJungleKilled(jungle, killer) {
    const hero = (killer && killer.id) ? this.heroes.find(h => h.id === killer.id) : null;
    if (hero && hero.alive) {
      this._grantGold(hero, REWARD.jungle_kill.gold);
      this._addExp(hero, REWARD.jungle_kill.exp);
      this._rlAward(hero, RL_REWARD.jungle_kill);
      // buff
      if (jungle.buff === 'red') {
        hero.buffs.push({ type: 'red', remain: BUFF_DURATION });
      } else if (jungle.buff === 'blue') {
        hero.buffs.push({ type: 'blue', remain: BUFF_DURATION });
      } else if (jungle.buff === 'tyrant') {
        // 全队增益
        for (const a of this.heroes) if (a.team === hero.team && a.alive) a.buffs.push({ type: 'tyrant', remain: BUFF_DURATION });
      } else if (jungle.buff === 'overlord') {
        for (const a of this.heroes) if (a.team === hero.team && a.alive) a.buffs.push({ type: 'overlord', remain: BUFF_DURATION });
      }
      this._pushEvent('jungle_kill', { hero: hero.id, type: jungle.type, team: hero.team });
    }
    jungle.respawn = jungle.team === 'neutral' ? JUNGLE_RESPAWN * 1.5 : JUNGLE_RESPAWN;
  }

  _onStructureKilled(structure, killer) {
    const hero = (killer && killer.id) ? this.heroes.find(h => h.id === killer.id) : null;
    if (hero) {
      this._grantGold(hero, REWARD.tower_kill.gold);
      this._addExp(hero, REWARD.tower_kill.exp);
      this._rlAward(hero, RL_REWARD.tower_kill);
    }
    // 己方塔被推：全队分摊惩罚
    if (structure.kind !== 'main') {
      for (const h of this.heroes) {
        if (h.team === structure.team && h.alive) this._rlAward(h, RL_REWARD.tower_lost);
      }
    }
    this._pushEvent('tower_destroyed', { structure: structure.id, team: structure.team, killer: hero ? hero.id : null });
    if (this.App.showToast) {
      this.App.showToast(`${structure.team === 'blue' ? '蓝方' : '红方'}${structure.kind === 'main' ? '主水晶' : '防御塔'}被摧毁！`);
    }
  }

  _handleHeroDeath(victim, killer) {
    if (!victim.alive) return;
    victim.alive = false;
    victim.hp = 0;
    victim.deaths++;
    victim.state = 'dead';
    victim.respawnTimer = RESPAWN_BASE + victim.level * RESPAWN_PER_LEVEL;
    victim.target = null;
    // 隐藏 mesh
    if (victim.mesh) victim.mesh.visible = false;
    // 爆炸特效
    this._spawnExplosion(victim.x, 1.5, victim.z, victim.team === 'blue' ? 0x3a6bff : 0xff3a3a);

    // 击杀奖励
    const killerHero = (killer && killer.id) ? this.heroes.find(h => h.id === killer.id) : null;
    if (killerHero && killerHero.team !== victim.team) {
      killerHero.kills++;
      this._grantGold(killerHero, REWARD.hero_kill.gold);
      this._addExp(killerHero, REWARD.hero_kill.exp);
      this.teamKills[killerHero.team]++;
      this._rlAward(killerHero, RL_REWARD.hero_kill);
      // 助攻：附近友方获得助攻奖励
      for (const a of this.heroes) {
        if (a.team === killerHero.team && a.id !== killerHero.id && a.alive) {
          if (Math.hypot(a.x - victim.x, a.z - victim.z) < 15) {
            a.assists++;
            this._grantGold(a, REWARD.assist.gold);
            this._addExp(a, REWARD.assist.exp);
            this._rlAward(a, RL_REWARD.assist);
          }
        }
      }
      this._pushEvent('hero_kill', { killer: killerHero.id, victim: victim.id, killer_team: killerHero.team });
      this._addKillFeed(`${killerHero.name} 击杀了 ${victim.name}`);
    } else {
      this._pushEvent('hero_death', { victim: victim.id, cause: killer ? (killer.id || 'unknown') : 'unknown' });
      this._addKillFeed(`${victim.name} 阵亡`);
    }
    // 阵亡惩罚
    this._rlAward(victim, RL_REWARD.death);
  }

  _respawnHero(h) {
    h.alive = true;
    h.hp = h.maxHp;
    h.mp = h.maxMp;
    h.state = 'return_to_lane';
    const base = h.team === 'blue' ? BLUE_BASE : RED_BASE;
    h.x = base.x + (Math.random() - 0.5) * 2;
    h.z = base.z + (Math.random() - 0.5) * 2;
    if (h.mesh) h.mesh.visible = !h.isPlayer;
    if (h.isPlayer && !this.spectating) {
      const avatar = this.App.currentAvatar;
      if (avatar) {
        avatar.position.set(h.x, 0, h.z);
        this.App.smoothRotY = h.facing;
      }
    }
    this._pushEvent('hero_respawn', { hero: h.id });
  }

  _addExp(hero, exp) {
    hero.exp += exp;
    while (hero.level < MAX_LEVEL && hero.exp >= this._expToNext(hero.level)) {
      hero.exp -= this._expToNext(hero.level);
      hero.level++;
      // 升级属性
      const def = HERO_DEFS[hero.heroKey];
      hero.maxHp += Math.round(def.hp * 0.08);
      hero.maxMp += Math.round(def.mp * 0.08);
      hero.atk += Math.round(def.atk * 0.08);
      hero.armor += 4;
      hero.mr += 2;
      hero.hp = hero.maxHp;
      hero.mp = hero.maxMp;
      // 技能升级
      if (hero.level === 4) hero.skill1Level = 2;
      if (hero.level === 7) hero.skill1Level = 3;
      if (hero.level === 5) hero.skill2Level = 2;
      if (hero.level === 9) hero.skill2Level = 3;
      if (hero.level === 4) hero.ultLevel = 1;
      if (hero.level === 8) hero.ultLevel = 2;
      if (hero.level === 12) hero.ultLevel = 3;
      this._pushEvent('hero_levelup', { hero: hero.id, level: hero.level });
    }
  }

  _expToNext(level) {
    return 100 + level * 60;
  }

  _tryBuyEquipment(h) {
    while (h.equipLevel < EQUIPMENT_TIERS.length && h.gold >= EQUIPMENT_TIERS[h.equipLevel].cost) {
      const tier = EQUIPMENT_TIERS[h.equipLevel];
      h.gold -= tier.cost;
      h.equipLevel++;
      const b = tier.bonus;
      h.atk += b.atk || 0;
      h.maxHp += b.hp || 0;
      h.armor += b.armor || 0;
      h.mr += b.mr || 0;
      h.hp += b.hp || 0;
      this._pushEvent('hero_buy_item', { hero: h.id, tier: h.equipLevel });
    }
  }

  _applyEquipBonus(h) {
    // 初始无装备
  }

  // ==================== 投射物与特效 ====================

  _spawnProjectile(from, target, color) {
    const THREE = this.THREE;
    // 能量弹：带 emissive 的发光材质
    const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.2 });
    const mesh = new THREE.Mesh(this._sharedGeometries.projectile, mat);
    const fx = from.x !== undefined ? from.x : 0;
    const fz = from.z !== undefined ? from.z : 0;
    const fy = from.kind === 'tower' ? 5 : (from.kind === 'main' ? 2 : 1.2);
    mesh.position.set(fx, fy, fz);
    this.addToScene(mesh);
    this.projectiles.push({
      mesh, mat,
      fromX: fx, fromY: fy, fromZ: fz,
      target, color,
      targetId: target.id,
      speed: 18,
      lifetime: 2,
    });
  }

  _updateProjectiles(dt) {
    for (let i = this.projectiles.length - 1; i >= 0; i--) {
      const p = this.projectiles[i];
      p.lifetime -= dt;
      const target = this._getUnitById(p.targetId);
      if (!target || p.lifetime <= 0) {
        if (p.mesh && p.mesh.parent) p.mesh.parent.remove(p.mesh);
        p.mat.dispose();
        this.projectiles.splice(i, 1);
        continue;
      }
      const tx = target.x, ty = target.kind === 'tower' ? 5 : (target.kind === 'main' ? 2 : 1.2), tz = target.z;
      const dx = tx - p.mesh.position.x, dy = ty - p.mesh.position.y, dz = tz - p.mesh.position.z;
      const d = Math.hypot(dx, dy, dz);
      if (d < 0.4) {
        // 命中（伤害已在发射时结算，这里仅移除）
        if (p.mesh && p.mesh.parent) p.mesh.parent.remove(p.mesh);
        p.mat.dispose();
        this.projectiles.splice(i, 1);
        continue;
      }
      const step = p.speed * dt;
      p.mesh.position.x += (dx / d) * step;
      p.mesh.position.y += (dy / d) * step;
      p.mesh.position.z += (dz / d) * step;
    }
  }

  _spawnSkillEffect(x, z, color, radius) {
    const THREE = this.THREE;
    const ringGeo = new THREE.RingGeometry(radius * 0.7, radius, 24);
    const ringMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(x, 0.2, z);
    this.addToScene(ring);
    this.effects.push({ mesh: ring, mat: ringMat, geo: ringGeo, life: 0.6, maxLife: 0.6, type: 'ring' });
    // 中心能量光柱
    const pillarGeo = new THREE.CylinderGeometry(radius * 0.15, radius * 0.3, 3, 12);
    const pillarMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 });
    const pillar = new THREE.Mesh(pillarGeo, pillarMat);
    pillar.position.set(x, 1.5, z);
    this.addToScene(pillar);
    this.effects.push({ mesh: pillar, mat: pillarMat, geo: pillarGeo, life: 0.6, maxLife: 0.6, type: 'pillar' });
  }

  _spawnExplosion(x, y, z, color) {
    const THREE = this.THREE;
    const geo = new THREE.SphereGeometry(0.5, 8, 6);
    const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    this.addToScene(mesh);
    this.effects.push({ mesh, mat, geo, life: 0.8, maxLife: 0.8, type: 'explosion' });
    // 扩散冲击波环（水平放置）
    const shockGeo = new THREE.RingGeometry(0.3, 0.6, 16);
    const shockMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, side: THREE.DoubleSide });
    const shock = new THREE.Mesh(shockGeo, shockMat);
    shock.position.set(x, y, z);
    shock.rotation.x = -Math.PI / 2;
    this.addToScene(shock);
    this.effects.push({ mesh: shock, mat: shockMat, geo: shockGeo, life: 0.8, maxLife: 0.8, type: 'shockwave' });
  }

  _updateEffects(dt) {
    for (let i = this.effects.length - 1; i >= 0; i--) {
      const e = this.effects[i];
      e.life -= dt;
      const t = 1 - e.life / e.maxLife;
      if (e.type === 'explosion') {
        const s = 1 + t * 4;
        e.mesh.scale.set(s, s, s);
        e.mat.opacity = 0.9 * (1 - t);
      } else if (e.type === 'ring') {
        e.mat.opacity = 0.85 * (1 - t);
      } else if (e.type === 'pillar') {
        // 中心光柱：随时间上升 + 淡出
        e.mesh.position.y += dt * 3;
        e.mat.opacity = 0.8 * (1 - t);
      } else if (e.type === 'shockwave') {
        // 冲击波环：随时间扩大 + 淡出
        const sw = 1 + t * 5;
        e.mesh.scale.set(sw, sw, sw);
        e.mat.opacity = 0.8 * (1 - t);
      }
      if (e.life <= 0) {
        if (e.mesh && e.mesh.parent) e.mesh.parent.remove(e.mesh);
        e.mat.dispose();
        e.geo.dispose();
        this.effects.splice(i, 1);
      }
    }
  }

  // ==================== 击杀提示 ====================

  _addKillFeed(text) {
    this.killFeed.push({ text, life: 4 });
    if (this.killFeed.length > 5) this.killFeed.shift();
  }
  _updateKillFeed(dt) {
    for (let i = this.killFeed.length - 1; i >= 0; i--) {
      this.killFeed[i].life -= dt;
      if (this.killFeed[i].life <= 0) this.killFeed.splice(i, 1);
    }
  }

  // ==================== 胜负判定 ====================

  _checkVictory() {
    const blueMain = this.structures.find(s => s.id === 'blue_main');
    const redMain = this.structures.find(s => s.id === 'red_main');
    if (blueMain && blueMain.hp <= 0) {
      this._endMatch('red');
    } else if (redMain && redMain.hp <= 0) {
      this._endMatch('blue');
    }
  }

  _endMatch(winner) {
    if (this.state === 'completed') return;
    const isWin = winner === 'blue';
    this._pushEvent('match_end', { winner, blue_kills: this.teamKills.blue, red_kills: this.teamKills.red });
    // RL：终局奖励 + 结算本局
    this._rlEndMatch(winner);

    // 加速自我对弈模式：直接重开新一局，保持游戏循环持续运行
    // （若调用 onComplete 会让 GameModeManager 停止 update 循环，导致无法连续训练）
    if (this._rlAutoRestart) {
      this._rlRestartMatch();
      return;
    }

    if (this.App.sendAIAction && !this._isFastMode()) {
      this.App.sendAIAction(isWin
        ? '（胜利了！我们摧毁了敌方水晶！这一局打得太精彩了，队友们配合得太好了！）'
        : '（输了...敌方水晶太坚固了。下一局我们会更努力，总结经验再来！）');
    }
    this.onComplete({
      winner,
      result: isWin ? 'victory' : 'defeat',
      blue_kills: this.teamKills.blue,
      red_kills: this.teamKills.red,
      duration: Math.floor(this.elapsedTime),
    });
  }

  // ==================== 工具方法 ====================

  _moveUnit(unit, tx, tz, step) {
    const dx = tx - unit.x, dz = tz - unit.z;
    const d = Math.hypot(dx, dz);
    if (d < 0.01) return;
    const s = Math.min(step, d);
    unit.x += (dx / d) * s;
    unit.z += (dz / d) * s;
    unit.facing = Math.atan2(dx, dz);
    this._clampToMap(unit);
  }

  _clampToMap(unit) {
    const lim = MAP_HALF - 1;
    unit.x = Math.max(-lim, Math.min(lim, unit.x));
    unit.z = Math.max(-lim, Math.min(lim, unit.z));
  }

  _getUnitById(id) {
    if (!id) return null;
    if (id.startsWith('minion_')) return this.minions.find(m => m.id === id);
    if (id.includes('_main') || id.includes('_t')) return this.structures.find(s => s.id === id);
    if (id.startsWith('blue_') || id.startsWith('red_')) {
      if (id.startsWith('blue_0') || id.startsWith('blue_1') || id.startsWith('blue_2') || id.startsWith('blue_3') || id.startsWith('blue_4')
       || id.startsWith('red_0') || id.startsWith('red_1') || id.startsWith('red_2') || id.startsWith('red_3') || id.startsWith('red_4')) {
        return this.heroes.find(h => h.id === id);
      }
    }
    // 野怪
    const j = this.jungle.find(j => j.id === id);
    if (j) return j;
    return null;
  }

  _isTargetAlive(id) {
    const t = this._getUnitById(id);
    if (!t) return false;
    if (t.hp !== undefined && t.hp <= 0) return false;
    if (t.alive === false) return false;
    return true;
  }

  /** 碰撞检测 —— 阻止越界与进入河道外的不可通行区域（简化：仅边界） */
  checkCollision(x, z, options = {}) {
    const lim = MAP_HALF - 1;
    if (Math.abs(x) > lim || Math.abs(z) > lim) return true;
    return false;
  }

  setPlayerSpeed(speed) {
    this._lastMoveSpeed = speed;
    // 同步玩家英雄移动速度到 moveSpeed（影响 AI 感知）
    const h = this.heroes.find(h => h.id === this.playerHeroId);
    if (h && h.alive) {
      // moveSpeed 不变，但记录当前速度
    }
  }

  _getPlayerSpeed() {
    return this._lastMoveSpeed || 0;
  }

  _getGroundHeight(x, z) {
    return 0; // 平面地图
  }

  // ==================== 血条更新 ====================

  _updateHealthBar(bar, x, z, hp, maxHp, mp, maxMp, level) {
    if (!bar) return;
    const ratio = Math.max(0, Math.min(1, hp / maxHp));
    const ctx = bar.ctx;
    const hasMp = mp !== undefined && maxMp !== undefined && maxMp > 0;
    // 有蓝条：画布分上下两段；无蓝条：仅血量
    if (hasMp) {
      ctx.clearRect(0, 0, 128, 20);
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(0, 0, 128, 20);
      const teamColor = bar.team === 'blue' ? '#4488ff' : bar.team === 'red' ? '#ff4444' : '#aaaa44';
      ctx.fillStyle = teamColor;
      ctx.fillRect(1, 1, 126 * ratio, 10);
      const mpRatio = Math.max(0, Math.min(1, mp / maxMp));
      ctx.fillStyle = '#44aaff';
      ctx.fillRect(1, 11, 126 * mpRatio, 8);
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.strokeRect(0, 0, 128, 20);
      // 等级（右上角小字）
      if (level !== undefined) {
        ctx.fillStyle = '#ffdd66';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(`Lv${level}`, 125, 5.5);
      }
    } else {
      ctx.clearRect(0, 0, 128, 16);
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(0, 0, 128, 16);
      const teamColor = bar.team === 'blue' ? '#4488ff' : bar.team === 'red' ? '#ff4444' : '#aaaa44';
      ctx.fillStyle = teamColor;
      ctx.fillRect(1, 1, 126 * ratio, 14);
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.strokeRect(0, 0, 128, 16);
    }
    bar.texture.needsUpdate = true;
    bar.sprite.position.set(x, bar.y, z);
  }

  _removeHealthBar(bar) {
    if (!bar) return;
    if (bar.sprite && bar.sprite.parent) bar.sprite.parent.remove(bar.sprite);
    if (bar.texture) bar.texture.dispose();
    if (bar.sprite && bar.sprite.material) bar.sprite.material.dispose();
  }

  // ==================== AI 感知数据 ====================

  getExtraState() {
    const player = this.heroes.find(h => h.id === this.playerHeroId);
    return {
      player_hero: player ? player.name : '',
      player_level: player ? player.level : 1,
      player_kda: player ? `${player.kills}/${player.deaths}/${player.assists}` : '0/0/0',
      player_gold: player ? player.gold : 0,
      blue_kills: this.teamKills.blue,
      red_kills: this.teamKills.red,
      blue_towers: this.structures.filter(s => s.team === 'blue' && s.kind === 'tower' && s.hp > 0).length,
      red_towers: this.structures.filter(s => s.team === 'red' && s.kind === 'tower' && s.hp > 0).length,
      blue_main_hp: (this.structures.find(s => s.id === 'blue_main') || {}).hp || 0,
      red_main_hp: (this.structures.find(s => s.id === 'red_main') || {}).hp || 0,
      match_time: Math.floor(this._gameTime),
      spectating: this.spectating,
    };
  }

  getPerceptionData() {
    const player = this.heroes.find(h => h.id === this.playerHeroId);
    const pos = this._getPlayerPosition();
    return {
      game_type: this.name,
      game_name: this.displayName,
      state: this.state,
      score: this.score,
      elapsed_sec: Math.floor(this.elapsedTime),
      player: {
        x: pos ? pos.x : 0, y: 0, z: pos ? pos.z : 0,
        facing: this._getPlayerFacing(),
        speed: this._getPlayerSpeed(),
        hero: player ? player.name : '',
        level: player ? player.level : 1,
        hp: player ? Math.round(player.hp) : 0,
        max_hp: player ? player.maxHp : 0,
        mp: player ? Math.round(player.mp) : 0,
        gold: player ? player.gold : 0,
      },
      map: this._getMapData(),
      objects: this._getObjectsData(),
      nearby: this._getNearbyObjects(),
      progress: this.getExtraState(),
      recent_events: this._getRecentEvents(8),
    };
  }

  _getMapData() {
    return {
      type: 'moba',
      half_size: MAP_HALF,
      lanes: LANE_KEYS,
      blue_base: BLUE_BASE,
      red_base: RED_BASE,
      river_z: RIVER_Z,
    };
  }

  _getObjectsData() {
    const data = {};
    data.structures = this.structures.filter(s => s.hp > 0).map(s => ({
      id: s.id, kind: s.kind, team: s.team, lane: s.lane, tier: s.tier,
      x: +s.x.toFixed(1), z: +s.z.toFixed(1),
      hp: Math.round(s.hp), max_hp: s.maxHp,
    }));
    data.heroes = this.heroes.map(h => ({
      id: h.id, team: h.team, name: h.name, role: h.role, lane: h.lane,
      level: h.level, hp: Math.round(h.hp), max_hp: h.maxHp, mp: Math.round(h.mp),
      x: +h.x.toFixed(1), z: +h.z.toFixed(1),
      alive: h.alive, kills: h.kills, deaths: h.deaths, assists: h.assists,
      is_player: h.isPlayer,
      state: h.state,
    }));
    data.jungle = this.jungle.filter(j => j.hp > 0).map(j => ({
      id: j.id, type: j.type, team: j.team,
      x: +j.x.toFixed(1), z: +j.z.toFixed(1),
      hp: Math.round(j.hp), max_hp: j.maxHp,
    }));
    data.minions = this.minions.filter(m => m.hp > 0).map(m => ({
      id: m.id, team: m.team, lane: m.lane, type: m.type,
      x: +m.x.toFixed(1), z: +m.z.toFixed(1),
      hp: Math.round(m.hp), max_hp: m.maxHp,
    }));
    return data;
  }

  _getNearbyObjects() {
    const player = this.heroes.find(h => h.id === this.playerHeroId);
    if (!player) return [];
    const px = player.x, pz = player.z;
    const facing = this._getPlayerFacing();
    const range = 20;
    const nearby = [];
    for (const h of this.heroes) {
      if (h.id === player.id || !h.alive) continue;
      const dx = h.x - px, dz = h.z - pz;
      const d = Math.hypot(dx, dz);
      if (d < range) nearby.push({
        type: 'hero', id: h.id, team: h.team, name: h.name,
        x: +h.x.toFixed(1), z: +h.z.toFixed(1), distance: +d.toFixed(1),
        direction: this._relativeDir(dx, dz, facing),
        level: h.level, hp_ratio: +(h.hp / h.maxHp).toFixed(2),
      });
    }
    for (const s of this.structures) {
      if (s.hp <= 0) continue;
      const dx = s.x - px, dz = s.z - pz;
      const d = Math.hypot(dx, dz);
      if (d < range) nearby.push({
        type: 'structure', id: s.id, kind: s.kind, team: s.team,
        x: +s.x.toFixed(1), z: +s.z.toFixed(1), distance: +d.toFixed(1),
        direction: this._relativeDir(dx, dz, facing),
        hp_ratio: +(s.hp / s.maxHp).toFixed(2),
      });
    }
    for (const j of this.jungle) {
      if (j.hp <= 0) continue;
      const dx = j.x - px, dz = j.z - pz;
      const d = Math.hypot(dx, dz);
      if (d < range) nearby.push({
        type: 'jungle', id: j.id, subtype: j.type,
        x: +j.x.toFixed(1), z: +j.z.toFixed(1), distance: +d.toFixed(1),
        direction: this._relativeDir(dx, dz, facing),
      });
    }
    nearby.sort((a, b) => a.distance - b.distance);
    return nearby.slice(0, 15);
  }

  _relativeDir(dx, dz, facing) {
    const cosA = Math.cos(-facing), sinA = Math.sin(-facing);
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

  _eventImportance(type) {
    const map = {
      match_start: 3, match_end: 3,
      hero_kill: 3, hero_death: 2, hero_respawn: 1,
      tower_destroyed: 3, jungle_kill: 2,
      hero_levelup: 1, hero_recall: 1, hero_buy_item: 1,
      minion_wave: 1, skill_cast: 1,
    };
    return map[type] || 1;
  }

  // ==================== UI ====================

  _createMobaUI() {
    if (this._mobaUI) return;
    const overlay = document.createElement('div');
    overlay.id = 'moba-ui-overlay';
    overlay.innerHTML = `
      <style>
        #moba-ui-overlay { position: fixed; inset: 0; pointer-events: none; z-index: 50; font-family: 'Microsoft YaHei', sans-serif; }
        #moba-ui-overlay > * { pointer-events: auto; }
        #moba-ui-overlay button { font-family: inherit; }

        /* 顶栏：比分 + 时间（居中、紧凑） */
        .moba-topbar { position: absolute; top: 6px; left: 50%; transform: translateX(-50%); display: flex; align-items: center; gap: 12px; background: rgba(10,18,30,0.88); padding: 4px 14px; border-radius: 16px; color: #fff; font-size: 11px; border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 2px 10px rgba(0,0,0,0.45); }
        .moba-score { font-size: 14px; font-weight: bold; display: flex; align-items: center; gap: 5px; }
        .moba-score .blue { color: #66aaff; }
        .moba-score .red { color: #ff7777; }
        .moba-score .sep { color: #888; font-size: 12px; }
        #moba-time { color: #ffdd66; font-weight: bold; font-size: 11px; min-width: 36px; text-align: center; }

        /* 左上：小地图（紧凑） */
        .moba-minimap { position: absolute; top: 6px; left: 6px; width: 120px; height: 120px; background: rgba(10,18,30,0.9); border: 1px solid rgba(255,221,102,0.3); border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.4); overflow: hidden; }
        .moba-minimap canvas { display: block; width: 100%; height: 100%; }

        /* 右上：经济+战绩合并面板（紧凑） */
        .moba-panel { position: absolute; top: 6px; right: 6px; background: rgba(10,18,30,0.92); padding: 5px 8px; border-radius: 6px; color: #fff; font-size: 10px; width: 150px; border: 1px solid rgba(255,221,102,0.3); box-shadow: 0 2px 8px rgba(0,0,0,0.4); line-height: 1.3; }
        .moba-panel .pn-title { font-size: 10px; color: #ffdd66; font-weight: bold; margin-bottom: 3px; padding-bottom: 2px; border-bottom: 1px solid rgba(255,221,102,0.2); }
        .moba-panel .pn-section { font-size: 9px; color: #88aacc; font-weight: bold; margin: 3px 0 1px; }
        .moba-panel .pn-row { display: flex; justify-content: space-between; margin: 1px 0; }
        .moba-panel .pn-row .label { color: #aab; }
        .moba-panel .pn-row .val { color: #ffdd66; font-weight: bold; }
        .moba-panel .pn-kda { color: #ffdd66; font-weight: bold; font-size: 12px; text-align: center; margin: 2px 0; }
        .moba-panel .pn-team { display: flex; justify-content: space-between; margin-top: 3px; padding-top: 2px; border-top: 1px solid rgba(255,255,255,0.15); font-size: 9px; }
        .moba-panel .pn-team .blue { color: #66aaff; }
        .moba-panel .pn-team .red { color: #ff7777; }

        #moba-killfeed { position: absolute; top: 44px; left: 50%; transform: translateX(-50%); display: flex; flex-direction: column; gap: 3px; align-items: center; max-width: 260px; }
        .killfeed-item { background: rgba(10,18,30,0.85); color: #ffdd66; padding: 2px 8px; border-radius: 3px; font-size: 10px; border-left: 2px solid #ff6666; }

        /* 左下角：切换英雄 / 观战（紧凑） */
        .moba-actions-left { position: absolute; bottom: 10px; left: 8px; display: flex; flex-direction: column; gap: 4px; }
        .moba-action-btn { padding: 4px 10px; border-radius: 6px; background: rgba(20,30,50,0.92); border: 1px solid rgba(255,255,255,0.3); color: #fff; cursor: pointer; font-size: 10px; white-space: nowrap; transition: background 0.15s, transform 0.08s; }
        .moba-action-btn:hover { background: rgba(40,60,90,0.95); }
        .moba-action-btn:active { transform: scale(0.95); }

        /* 右下：普攻 + 3 技能 */
        .moba-actions { position: absolute; bottom: 12px; right: 12px; display: flex; flex-direction: column; align-items: flex-end; gap: 7px; }
        .moba-skill-row { display: flex; gap: 7px; }
        .moba-attack { width: 54px; height: 54px; border-radius: 50%; background: radial-gradient(circle at 35% 35%, #ff8855, #cc3322); border: 2px solid rgba(255,200,150,0.65); color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; font-size: 11px; font-weight: bold; user-select: none; box-shadow: 0 3px 12px rgba(200,50,0,0.5); transition: transform 0.08s; position: relative; }
        .moba-attack:hover { border-color: #ffdd99; }
        .moba-attack:active { transform: scale(0.9); }
        .moba-attack .attack-key { position: absolute; top: 3px; right: 7px; font-size: 9px; color: #ffeebb; font-weight: bold; }
        .moba-attack .attack-label { line-height: 1.1; text-align: center; text-shadow: 0 1px 2px #000; }

        .moba-skill { width: 50px; height: 50px; border-radius: 9px; background: rgba(20,30,50,0.92); border: 2px solid rgba(255,255,255,0.3); color: #fff; position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; font-size: 10px; user-select: none; transition: transform 0.08s, border-color 0.12s; }
        .moba-skill:hover { border-color: #ffdd66; }
        .moba-skill:active { transform: scale(0.92); }
        .moba-skill .skill-key { position: absolute; top: 2px; left: 4px; font-size: 9px; color: #ffdd66; font-weight: bold; }
        .moba-skill .skill-name { font-size: 9px; text-align: center; padding: 0 2px; line-height: 1.1; }
        .moba-skill .skill-cd { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: bold; color: #fff; text-shadow: 0 0 6px #000; background: rgba(0,0,0,0.65); border-radius: 8px; }
        .moba-skill.on-cooldown .skill-name { opacity: 0.3; }
        .moba-skill.ult { border-color: #ffaa22; background: rgba(50,30,18,0.92); }
        .moba-skill.ult .skill-key { color: #ffaa22; }

        /* 底部中央：回城按钮（紧凑） */
        .moba-recall-bar { position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); display: flex; gap: 6px; }
        .moba-recall-btn { padding: 6px 16px; border-radius: 8px; background: linear-gradient(135deg, #4488dd, #335599); border: 1px solid rgba(150,200,255,0.6); color: #fff; cursor: pointer; font-size: 11px; font-weight: bold; white-space: nowrap; transition: background 0.15s, transform 0.08s; box-shadow: 0 2px 8px rgba(50,100,200,0.5); }
        .moba-recall-btn:hover { background: linear-gradient(135deg, #5599ee, #4466aa); }
        .moba-recall-btn:active { transform: scale(0.95); }

        /* 左上：RL 触发按钮（紧凑，始终可见） */
        .moba-rl-trigger { position: absolute; top: 132px; left: 6px; display: flex; align-items: center; gap: 5px; background: rgba(10,18,30,0.92); padding: 4px 9px; border-radius: 14px; color: #cfe; font-size: 10px; cursor: pointer; border: 1px solid rgba(102,255,170,0.35); box-shadow: 0 2px 8px rgba(0,0,0,0.4); transition: background 0.15s; user-select: none; }
        .moba-rl-trigger:hover { background: rgba(20,40,32,0.95); }
        .moba-rl-trigger .dot { width: 7px; height: 7px; border-radius: 50%; background: #555; display: inline-block; }
        .moba-rl-trigger .dot.on { background: #66ff66; box-shadow: 0 0 6px #66ff66; }
        .moba-rl-trigger .rl-ep-mini { color: #66ffaa; font-weight: bold; min-width: 22px; text-align: center; }

        /* RL 弹窗浮层（默认隐藏，居中显示，可拖动，z-index 置顶不被遮挡） */
        .moba-rl-modal { position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 260px; background: rgba(10,18,30,0.97); padding: 10px 12px; border-radius: 8px; color: #fff; font-size: 11px; border: 1px solid rgba(102,255,170,0.4); box-shadow: 0 8px 32px rgba(0,0,0,0.65); line-height: 1.5; z-index: 9999; }
        .moba-rl-modal .rl-title { font-size: 12px; color: #66ffaa; font-weight: bold; margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid rgba(102,255,170,0.25); display: flex; justify-content: space-between; align-items: center; cursor: move; user-select: none; }
        .moba-rl-modal .rl-close { background: none; border: none; color: #888; cursor: pointer; font-size: 14px; padding: 0 4px; line-height: 1; }
        .moba-rl-modal .rl-close:hover { color: #ff6666; }
        .moba-rl-modal .rl-row { display: flex; justify-content: space-between; margin: 2px 0; }
        .moba-rl-modal .rl-row .label { color: #aab; }
        .moba-rl-modal .rl-row .val { color: #66ffaa; font-weight: bold; }
        .moba-rl-modal .rl-btns { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
        .moba-rl-modal .rl-btn { flex: 1 1 auto; min-width: 52px; padding: 5px 6px; border-radius: 5px; background: rgba(30,50,40,0.92); border: 1px solid rgba(102,255,170,0.35); color: #cfe; cursor: pointer; font-size: 10px; transition: background 0.15s, transform 0.08s; }
        .moba-rl-modal .rl-btn:hover { background: rgba(50,80,60,0.95); }
        .moba-rl-modal .rl-btn:active { transform: scale(0.94); }
        .moba-rl-modal .rl-btn.active { background: linear-gradient(135deg,#2a7a4a,#1f5a36); color: #fff; border-color: #66ffaa; }
        .moba-rl-modal .rl-hint { font-size: 9px; color: #889; margin-top: 6px; line-height: 1.4; }
      </style>
      <div class="moba-topbar">
        <span class="moba-score"><span class="blue" id="moba-blue-score">0</span><span class="sep">:</span><span class="red" id="moba-red-score">0</span></span>
        <span id="moba-time">00:00</span>
      </div>
      <div class="moba-minimap" id="moba-minimap"><canvas id="moba-minimap-canvas" width="120" height="120"></canvas></div>
      <div class="moba-rl-trigger" id="rl-trigger" title="强化学习训练面板">
        <span class="dot" id="rl-dot"></span>
        <span>🧠 RL</span>
        <span class="rl-ep-mini" id="rl-episodes-mini">0局</span>
      </div>
      <div class="moba-rl-modal" id="moba-rl-modal" style="display:none;">
        <div class="rl-title"><span>🧠 Q-Learning 训练</span><button class="rl-close" id="rl-close" title="关闭">✕</button></div>
        <div class="rl-row"><span class="label">RL 决策</span><span class="val" id="rl-status">关闭</span></div>
        <div class="rl-row"><span class="label">对局数</span><span class="val" id="rl-episodes">0</span></div>
        <div class="rl-row"><span class="label">Q表条目</span><span class="val" id="rl-qsize">0</span></div>
        <div class="rl-row"><span class="label">探索率 ε</span><span class="val" id="rl-epsilon">0.95</span></div>
        <div class="rl-row"><span class="label">蓝胜/红胜</span><span class="val" id="rl-winrate">0/0</span></div>
        <div class="rl-row"><span class="label">倍速</span><span class="val" id="rl-speed">1x</span></div>
        <div class="rl-row"><span class="label">自动对弈</span><span class="val" id="rl-auto">关</span></div>
        <div class="rl-row"><span class="label">LLM 指挥</span><span class="val" id="rl-llm">关</span></div>
        <div class="rl-row"><span class="label">当前策略</span><span class="val" id="moba-llm-strategy" style="color:#ffdd66">—</span></div>
        <div class="rl-btns">
          <button class="rl-btn" id="rl-btn-toggle">开启RL</button>
          <button class="rl-btn" id="rl-btn-speed">1x</button>
          <button class="rl-btn" id="rl-btn-auto">自动对弈</button>
          <button class="rl-btn" id="rl-btn-llm">LLM指挥</button>
          <button class="rl-btn" id="rl-btn-reset">重置Q表</button>
          <button class="rl-btn" id="rl-btn-save">保存</button>
        </div>
        <div class="rl-hint" id="rl-hint">开启RL后9个bot由Q-Learning驱动；LLM指挥让AI做战术指挥官软引导决策；倍速+自动对弈加速训练。</div>
      </div>
      <div class="moba-panel" id="moba-panel">
        <div class="pn-title">📊 战绩 & 经济</div>
        <div class="pn-kda" id="pn-kda">0/0/0</div>
        <div class="pn-row"><span class="label">击杀</span><span class="val" id="pn-kills" style="color:#66ff66">0</span></div>
        <div class="pn-row"><span class="label">死亡</span><span class="val" id="pn-deaths" style="color:#ff6666">0</span></div>
        <div class="pn-row"><span class="label">助攻</span><span class="val" id="pn-assists" style="color:#66aaff">0</span></div>
        <div class="pn-row"><span class="label">等级</span><span class="val" id="pn-level" style="color:#ffdd66">Lv.1</span></div>
        <div class="pn-section">💰 经济</div>
        <div class="pn-row"><span class="label">当前金币</span><span class="val" id="pn-gold">0</span></div>
        <div class="pn-row"><span class="label">累计经济</span><span class="val" id="pn-total">0</span></div>
        <div class="pn-row"><span class="label">战力加成</span><span class="val" id="pn-power">+0%</span></div>
        <div class="pn-row"><span class="label">装备等级</span><span class="val" id="pn-equip">0/4</span></div>
        <div class="pn-team">
          <span>蓝方 <span class="blue" id="pn-blue">0</span></span>
          <span><span class="red" id="pn-red">0</span> 红方</span>
        </div>
      </div>
      <div id="moba-killfeed"></div>
      <div class="moba-actions-left" id="moba-actions-left" style="display:none;">
        <button class="moba-action-btn" id="moba-btn-switch">🔄 切换英雄(Tab)</button>
        <button class="moba-action-btn" id="moba-btn-spectate">👁 观战(V)</button>
      </div>
      <div class="moba-actions">
        <div class="moba-skill-row">
          <div class="moba-skill" id="moba-skill1" data-slot="1"><span class="skill-key">1</span><span class="skill-name">技能一</span><div class="skill-cd" style="display:none"></div></div>
          <div class="moba-skill" id="moba-skill2" data-slot="2"><span class="skill-key">2</span><span class="skill-name">技能二</span><div class="skill-cd" style="display:none"></div></div>
          <div class="moba-skill ult" id="moba-skill3" data-slot="3"><span class="skill-key">3</span><span class="skill-name">大招</span><div class="skill-cd" style="display:none"></div></div>
        </div>
        <div class="moba-attack" id="moba-attack"><span class="attack-key">空格</span><span class="attack-label">普攻</span></div>
      </div>
      <div class="moba-recall-bar">
        <button class="moba-recall-btn" id="moba-btn-recall">� 回城 (Q)</button>
      </div>
    `;
    document.body.appendChild(overlay);
    this._mobaUI = overlay;

    // 技能按钮点击
    overlay.querySelectorAll('.moba-skill').forEach(el => {
      el.addEventListener('click', () => {
        const slot = parseInt(el.dataset.slot, 10);
        this._skillInputQueue.push(slot);
      });
    });
    // 普攻按钮点击
    const attackBtn = overlay.querySelector('#moba-attack');
    if (attackBtn) {
      attackBtn.addEventListener('click', () => this._playerManualAttack());
    }
    overlay.querySelector('#moba-btn-recall').addEventListener('click', () => this.onUserInput('recall', {}));
    overlay.querySelector('#moba-btn-switch').addEventListener('click', () => this._cycleSwitchHero());
    overlay.querySelector('#moba-btn-spectate').addEventListener('click', () => this.onUserInput('toggle_spectate', {}));

    // RL 训练面板：触发按钮 + 弹窗关闭
    overlay.querySelector('#rl-trigger').addEventListener('click', () => this._rlUIShowModal(true));
    overlay.querySelector('#rl-close').addEventListener('click', () => this._rlUIShowModal(false));
    // RL 控制按钮
    overlay.querySelector('#rl-btn-toggle').addEventListener('click', () => this._rlUIToggle());
    overlay.querySelector('#rl-btn-speed').addEventListener('click', () => this._rlUICycleSpeed());
    overlay.querySelector('#rl-btn-auto').addEventListener('click', () => this._rlUIToggleAuto());
    overlay.querySelector('#rl-btn-llm').addEventListener('click', () => this._llmUITogglePolicy());
    overlay.querySelector('#rl-btn-reset').addEventListener('click', () => this._rlUIReset());
    overlay.querySelector('#rl-btn-save').addEventListener('click', () => this._rlUISave());

    // 弹窗拖动
    this._rlInitDrag();

    // 小地图点击/拖拽：临时观察对应区域
    this._initMinimapInteraction();

    // 初始化技能名
    this._refreshSkillNames();
  }

  /** 小地图交互：点击或拖拽 → 相机平移到对应区域观察 */
  _initMinimapInteraction() {
    const canvas = this._mobaUI && this._mobaUI.querySelector('#moba-minimap-canvas');
    if (!canvas) return;
    // 画布像素 → 世界坐标
    const canvasToWorld = (clientX, clientY) => {
      const rect = canvas.getBoundingClientRect();
      const px = (clientX - rect.left) / rect.width;   // 0..1
      const py = (clientY - rect.top) / rect.height;    // 0..1
      // 与 _renderMinimap 的 toX/toY 反向（Y 轴已反转：画布上=世界+Z）
      const x = (px * 2 - 1) * MAP_HALF;
      const z = (1 - py * 2) * MAP_HALF;
      return { x, z };
    };
    const startFocus = (clientX, clientY) => {
      const p = canvasToWorld(clientX, clientY);
      this._minimapFocus = p;
      this._minimapFocusTimer = 4;   // 4 秒观察窗口
      this._minimapDragging = true;
    };
    const moveFocus = (clientX, clientY) => {
      if (!this._minimapDragging) return;
      const p = canvasToWorld(clientX, clientY);
      this._minimapFocus = p;
      this._minimapFocusTimer = 4;
    };
    const endFocus = () => {
      this._minimapDragging = false;
      // 释放后保持 1.5 秒再返回
      this._minimapFocusTimer = Math.max(this._minimapFocusTimer, 1.5);
    };
    // 鼠标
    canvas.addEventListener('mousedown', (e) => { e.preventDefault(); startFocus(e.clientX, e.clientY); });
    canvas.addEventListener('mousemove', (e) => { if (this._minimapDragging) { e.preventDefault(); moveFocus(e.clientX, e.clientY); } });
    window.addEventListener('mouseup', endFocus);
    // 触摸
    canvas.addEventListener('touchstart', (e) => { if (e.touches[0]) { e.preventDefault(); startFocus(e.touches[0].clientX, e.touches[0].clientY); } }, { passive: false });
    canvas.addEventListener('touchmove', (e) => { if (e.touches[0] && this._minimapDragging) { e.preventDefault(); moveFocus(e.touches[0].clientX, e.touches[0].clientY); } }, { passive: false });
    canvas.addEventListener('touchend', endFocus);
    canvas.addEventListener('touchcancel', endFocus);
  }

  /** 玩家移动时清除小地图观察，立即返回跟随 avatar */
  _cancelMinimapFocusOnMove() {
    if (this._minimapFocus) {
      this._minimapFocus = null;
      this._minimapFocusTimer = 0;
      this._minimapDragging = false;
    }
  }

  // ==================== RL UI 控制 ====================

  _rlUIShowModal(show) {
    const modal = this._mobaUI && this._mobaUI.querySelector('#moba-rl-modal');
    if (modal) modal.style.display = show ? 'block' : 'none';
  }

  /** 弹窗拖动（按住标题栏拖动；支持鼠标 + 触摸） */
  _rlInitDrag() {
    const modal = this._mobaUI && this._mobaUI.querySelector('#moba-rl-modal');
    const title = this._mobaUI && this._mobaUI.querySelector('#moba-rl-modal .rl-title');
    if (!modal || !title) return;

    let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;

    const startDrag = (clientX, clientY, target) => {
      // 点关闭按钮不拖动
      if (target && target.closest && target.closest('.rl-close')) return;
      dragging = true;
      const rect = modal.getBoundingClientRect();
      // 切换为像素定位，移除 transform 居中
      modal.style.transform = 'none';
      modal.style.left = rect.left + 'px';
      modal.style.top = rect.top + 'px';
      sx = clientX; sy = clientY;
      ox = rect.left; oy = rect.top;
    };
    const moveDrag = (clientX, clientY) => {
      if (!dragging) return;
      // 限制不拖出视口
      const w = modal.offsetWidth, h = modal.offsetHeight;
      let nx = ox + clientX - sx;
      let ny = oy + clientY - sy;
      nx = Math.max(-w / 2 + 20, Math.min(window.innerWidth - 20, nx));
      ny = Math.max(0, Math.min(window.innerHeight - 30, ny));
      modal.style.left = nx + 'px';
      modal.style.top = ny + 'px';
    };
    const endDrag = () => { dragging = false; };

    const onMouseDown = (e) => { startDrag(e.clientX, e.clientY, e.target); if (dragging) e.preventDefault(); };
    const onMouseMove = (e) => moveDrag(e.clientX, e.clientY);
    const onMouseUp = () => endDrag();
    const onTouchStart = (e) => {
      const t = e.touches[0]; if (!t) return;
      startDrag(t.clientX, t.clientY, e.target);
    };
    const onTouchMove = (e) => {
      const t = e.touches[0]; if (!t) return;
      moveDrag(t.clientX, t.clientY);
    };
    const onTouchEnd = () => endDrag();

    title.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    title.addEventListener('touchstart', onTouchStart, { passive: true });
    document.addEventListener('touchmove', onTouchMove, { passive: true });
    document.addEventListener('touchend', onTouchEnd);

    // 存清理函数，cleanup 时移除 document 级监听器
    this._rlDragCleanup = () => {
      title.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      title.removeEventListener('touchstart', onTouchStart);
      document.removeEventListener('touchmove', onTouchMove);
      document.removeEventListener('touchend', onTouchEnd);
    };
  }

  _rlUIToggle() {
    this._rlEnabled = !this._rlEnabled;
    if (this._rlEnabled) this._rlEnsureAgent();
    this._rlResetEpisode();
    if (this._mobaUI) {
      const btn = this._mobaUI.querySelector('#rl-btn-toggle');
      if (btn) { btn.textContent = this._rlEnabled ? '关闭RL' : '开启RL'; btn.classList.toggle('active', this._rlEnabled); }
      const status = this._mobaUI.querySelector('#rl-status');
      if (status) status.textContent = this._rlEnabled ? '开启' : '关闭';
      const dot = this._mobaUI.querySelector('#rl-dot');
      if (dot) dot.classList.toggle('on', this._rlEnabled);
    }
  }

  _rlUICycleSpeed() {
    this._rlSpeedIdx = (this._rlSpeedIdx + 1) % RL_SPEED_PRESETS.length;
    if (this._mobaUI) {
      const btn = this._mobaUI.querySelector('#rl-btn-speed');
      const val = this._mobaUI.querySelector('#rl-speed');
      const txt = `${RL_SPEED_PRESETS[this._rlSpeedIdx]}x`;
      if (btn) { btn.textContent = txt; btn.classList.toggle('active', this._isFastMode()); }
      if (val) val.textContent = txt;
      if (this._isFastMode() && !this.spectating) {
        const hint = this._mobaUI.querySelector('#rl-hint');
        if (hint) hint.textContent = '加速中：玩家英雄已交由AI，建议观战(V)或直接训练。';
      }
    }
    // 退出加速（1x）时自动恢复玩家操控：重置观战状态，避免玩家英雄继续被AI接管
    if (!this._isFastMode() && this.spectating) {
      this.spectating = false;
      if (this._mobaUI) {
        const hint = this._mobaUI.querySelector('#rl-hint');
        if (hint) hint.textContent = '已恢复玩家操控（1x）。';
      }
    }
  }

  _rlUIToggleAuto() {
    this._rlAutoRestart = !this._rlAutoRestart;
    // 不再强制加速：1x 也能自动对弈（对局较慢，用户可自行调高倍速）
    if (this._mobaUI) {
      const btn = this._mobaUI.querySelector('#rl-btn-auto');
      const val = this._mobaUI.querySelector('#rl-auto');
      if (btn) { btn.textContent = this._rlAutoRestart ? '停止对弈' : '自动对弈'; btn.classList.toggle('active', this._rlAutoRestart); }
      if (val) val.textContent = this._rlAutoRestart ? '开' : '关';
      const hint = this._mobaUI.querySelector('#rl-hint');
      if (hint && this._rlAutoRestart && !this._isFastMode()) {
        hint.textContent = '自动对弈已开启（1x）：对局约5-8分钟，可调高倍速加速训练。';
      }
    }
  }

  /** 切换 LLM 宏观指挥层 */
  _llmUITogglePolicy() {
    this._llmPolicyEnabled = !this._llmPolicyEnabled;
    if (this._llmPolicyEnabled) {
      // LLM 指挥依赖 RL 决策循环（偏置作用于 Q-Learning 选动作），未开 RL 时自动开启
      if (!this._rlEnabled) this._rlUIToggle();
      this._llmPolicyTimer = this._llmPolicyInterval; // 立即触发首次决策
      this._llmAccumReward = 0;
      this._llmWaitingResponse = false;
    } else {
      this._llmStrategy = null;
      this._llmRewardBias = null;
    }
    if (this._mobaUI) {
      const btn = this._mobaUI.querySelector('#rl-btn-llm');
      const val = this._mobaUI.querySelector('#rl-llm');
      if (btn) { btn.textContent = this._llmPolicyEnabled ? '停止指挥' : 'LLM指挥'; btn.classList.toggle('active', this._llmPolicyEnabled); }
      if (val) val.textContent = this._llmPolicyEnabled ? '开' : '关';
      const stratEl = this._mobaUI.querySelector('#moba-llm-strategy');
      if (stratEl && !this._llmPolicyEnabled) stratEl.textContent = '—';
    }
  }

  _rlUIReset() {
    if (!confirm('确定重置 Q 表与统计？此操作不可撤销。')) return;
    if (this._rl) this._rl.reset();
    else { this._rlEnsureAgent().reset(); }
    this._rlUpdateUI();
  }

  _rlUISave() {
    this._rlEnsureAgent().flush();
    const hint = this._mobaUI && this._mobaUI.querySelector('#rl-hint');
    if (hint) {
      const old = hint.textContent;
      hint.textContent = '✓ Q 表已保存到 localStorage';
      setTimeout(() => { if (hint) hint.textContent = old; }, 1500);
    }
  }

  _rlUpdateUI() {
    if (!this._mobaUI) return;
    const agent = this._rlEnsureAgent();
    const s = agent.stats;
    const set = (id, v) => { const el = this._mobaUI.querySelector('#' + id); if (el) el.textContent = v; };
    set('rl-episodes', s.episodes);
    set('rl-episodes-mini', `${s.episodes}局`);
    set('rl-qsize', s.traceSize || 0);
    set('rl-epsilon', '—');
    set('rl-winrate', `${s.wins}/${s.deaths}`);
  }

  _refreshSkillNames() {
    const h = this.heroes.find(h => h.id === this.playerHeroId);
    if (!h || !this._mobaUI) return;
    const def = HERO_DEFS[h.heroKey];
    const names = [def.skill1Name, def.skill2Name, def.ultName];
    const els = this._mobaUI.querySelectorAll('.moba-skill .skill-name');
    els.forEach((el, i) => { if (names[i]) el.textContent = names[i]; });
  }

  _cycleSwitchHero() {
    const blueHeroes = this.heroes.filter(h => h.team === 'blue');
    const idx = blueHeroes.findIndex(h => h.id === this.playerHeroId);
    const next = (idx + 1) % blueHeroes.length;
    this._switchPlayerHero(next);
    this._refreshSkillNames();
  }

  _bindKeyboard() {
    this._onMobaKey = (e) => {
      if (e.repeat) return;
      const key = e.key.toLowerCase();
      if (key === '1') { e.preventDefault(); this._skillInputQueue.push(1); }
      else if (key === '2') { e.preventDefault(); this._skillInputQueue.push(2); }
      else if (key === '3') { e.preventDefault(); this._skillInputQueue.push(3); }
      else if (key === ' ') { e.preventDefault(); this._playerManualAttack(); }
      else if (key === 'q') this.onUserInput('recall', {});
      else if (key === 'tab') { e.preventDefault(); this._cycleSwitchHero(); }
      else if (key === 'v') this.onUserInput('toggle_spectate', {});
    };
    document.addEventListener('keydown', this._onMobaKey);
  }

  _unbindKeyboard() {
    if (this._onMobaKey) {
      document.removeEventListener('keydown', this._onMobaKey);
      this._onMobaKey = null;
    }
  }

  _updateMobaUI() {
    if (!this._mobaUI) return;
    const player = this.heroes.find(h => h.id === this.playerHeroId);
    if (!player) return;

    // RL 面板刷新
    this._rlUpdateUI();

    // 死亡时才显示 切换英雄 / 观战 按钮
    const actionsLeft = this._mobaUI.querySelector('#moba-actions-left');
    if (actionsLeft) {
      actionsLeft.style.display = player.alive ? 'none' : 'flex';
    }

    // 右上面板（战绩 + 经济）
    const setPn = (id, v) => { const el = this._mobaUI.querySelector(id); if (el) el.textContent = v; };
    setPn('#pn-kda', `${player.kills}/${player.deaths}/${player.assists}`);
    setPn('#pn-kills', player.kills);
    setPn('#pn-deaths', player.deaths);
    setPn('#pn-assists', player.assists);
    setPn('#pn-level', `Lv.${player.level}`);
    setPn('#pn-gold', Math.floor(player.gold));
    setPn('#pn-total', Math.floor(player.totalGoldEarned));
    setPn('#pn-power', `+${(player.appliedGoldPower * 2)}%`);
    setPn('#pn-equip', `${player.equipLevel}/${EQUIPMENT_TIERS.length}`);
    setPn('#pn-blue', Math.floor(this.teamGold.blue));
    setPn('#pn-red', Math.floor(this.teamGold.red));

    // 技能 CD（血量/蓝量已在角色头顶血条显示）
    const skills = [
      { sel: '#moba-skill1', cd: player.skill1Cd },
      { sel: '#moba-skill2', cd: player.skill2Cd },
      { sel: '#moba-skill3', cd: player.ultCd },
    ];
    for (const s of skills) {
      const el = this._mobaUI.querySelector(s.sel);
      if (!el) continue;
      const cdEl = el.querySelector('.skill-cd');
      if (cdEl) {
        if (s.cd > 0) {
          cdEl.style.display = 'flex';
          cdEl.textContent = Math.ceil(s.cd);
          el.classList.add('on-cooldown');
        } else {
          cdEl.style.display = 'none';
          cdEl.textContent = '';
          el.classList.remove('on-cooldown');
        }
      }
    }

    // 大招解锁状态（4级解锁）
    const ultEl = this._mobaUI.querySelector('#moba-skill3');
    if (ultEl) {
      if (player.ultLevel < 1) {
        ultEl.style.opacity = '0.4';
        ultEl.style.filter = 'grayscale(0.8)';
      } else {
        ultEl.style.opacity = '';
        ultEl.style.filter = '';
      }
    }

    // 比分（蓝方 : 红方）
    const blueScoreEl = this._mobaUI.querySelector('#moba-blue-score');
    const redScoreEl = this._mobaUI.querySelector('#moba-red-score');
    if (blueScoreEl) blueScoreEl.textContent = this.teamKills.blue;
    if (redScoreEl) redScoreEl.textContent = this.teamKills.red;

    // 击杀提示
    const feedEl = this._mobaUI.querySelector('#moba-killfeed');
    if (feedEl) {
      feedEl.innerHTML = this.killFeed.map(k => `<div class="killfeed-item">${k.text}</div>`).join('');
    }

    // 小地图渲染
    this._renderMinimap();

    // 时间
    const timeEl = this._mobaUI.querySelector('#moba-time');
    if (timeEl) {
      const m = Math.floor(this._gameTime / 60);
      const s = Math.floor(this._gameTime % 60);
      timeEl.textContent = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
  }

  /** 渲染小地图（战场全局） */
  _renderMinimap() {
    const canvas = this._mobaUI?.querySelector('#moba-minimap-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    // 世界坐标 [-MAP_HALF, MAP_HALF] → 画布 [0, W]
    // X 轴：世界 +X → 画布右；Y 轴反转：世界 +Z（敌方）→ 画布上，使「我方蓝左下、敌方红右上」便于观察
    const toX = (x) => ((x + MAP_HALF) / (MAP_HALF * 2)) * W;
    const toY = (z) => H - ((z + MAP_HALF) / (MAP_HALF * 2)) * H;
    // 背景
    ctx.fillStyle = '#0a1220';
    ctx.fillRect(0, 0, W, H);
    // 河道（与中路垂直，画布左上↔右下对角线条带，过中心 (W/2,H/2)）
    ctx.save();
    ctx.fillStyle = 'rgba(60,100,160,0.35)';
    ctx.translate(W / 2, H / 2);
    ctx.rotate(Math.PI / 4);           // 顺时针 45°，长边沿右下方向
    ctx.fillRect(-W * 0.7, -2, W * 1.4, 4);
    ctx.restore();
    // 基地区域
    ctx.fillStyle = 'rgba(60,100,200,0.18)';
    ctx.beginPath(); ctx.arc(toX(BLUE_BASE.x), toY(BLUE_BASE.z), 8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(200,60,60,0.18)';
    ctx.beginPath(); ctx.arc(toX(RED_BASE.x), toY(RED_BASE.z), 8, 0, Math.PI * 2); ctx.fill();
    // 路线（按 lane 不同颜色，更醒目）
    ctx.lineWidth = 1.5;
    const laneStroke = { top: 'rgba(80,140,220,0.7)', mid: 'rgba(120,200,120,0.7)', bottom: 'rgba(220,170,80,0.7)' };
    for (const lane of LANE_KEYS) {
      const path = LANES[lane];
      ctx.strokeStyle = laneStroke[lane] || 'rgba(255,255,255,0.4)';
      ctx.beginPath();
      for (let i = 0; i < path.length; i++) {
        const px = toX(path[i].x), py = toY(path[i].z);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    // 建筑物
    for (const s of this.structures) {
      if (s.hp <= 0) continue;
      const x = toX(s.x), y = toY(s.z);
      if (s.kind === 'main') {
        ctx.fillStyle = s.team === 'blue' ? '#3a6bff' : '#ff3a3a';
        ctx.fillRect(x - 3, y - 3, 6, 6);
      } else {
        ctx.fillStyle = s.team === 'blue' ? '#66aaff' : '#ff7777';
        ctx.fillRect(x - 2, y - 2, 4, 4);
      }
    }
    // 野怪（按类型用不同颜色，boss 更大）
    for (const j of this.jungle) {
      if (j.hp <= 0) continue;
      const x = toX(j.x), y = toY(j.z);
      const isBoss = j.type === 'tyrant' || j.type === 'overlord';
      const colorMap = { redbuff: '#ff5544', bluebuff: '#4488ff', wolf: '#aaa', golem: '#aa8', raptor: '#da4', crab: '#4aa', plant: '#6c5', tyrant: '#c4c', overlord: '#fa2' };
      ctx.fillStyle = colorMap[j.type] || '#888';
      const s = isBoss ? 2 : 1.5;
      ctx.fillRect(x - s/2, y - s/2, s, s);
    }
    // 小兵（按阵营颜色区分，炮车更大更亮）
    for (const m of this.minions) {
      if (m.hp <= 0) continue;
      const x = toX(m.x), y = toY(m.z);
      if (m.type === 'cannon') {
        // 炮车：更大、更亮，带阵营色
        ctx.fillStyle = m.team === 'blue' ? '#88bbff' : '#ff8888';
        ctx.fillRect(x - 1, y - 1, 2, 2);
      } else {
        // 普通兵：阵营色细点
        ctx.fillStyle = m.team === 'blue' ? 'rgba(100,160,255,0.85)' : 'rgba(255,100,100,0.85)';
        ctx.fillRect(x - 0.5, y - 0.5, 1, 1);
      }
    }
    // 英雄（最后绘制，最上层）
    for (const h of this.heroes) {
      if (!h.alive) continue;
      const x = toX(h.x), y = toY(h.z);
      const isPlayer = h.id === this.playerHeroId;
      ctx.fillStyle = h.team === 'blue' ? '#4488ff' : '#ff4444';
      ctx.beginPath();
      ctx.arc(x, y, isPlayer ? 4 : 3, 0, Math.PI * 2);
      ctx.fill();
      if (isPlayer) {
        // 玩家高亮环
        ctx.strokeStyle = '#ffdd66';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        // 视野朝向（Y 轴已反转，方向线 Y 分量取负）
        ctx.strokeStyle = '#ffdd66';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.sin(h.facing) * 6, y - Math.cos(h.facing) * 6);
        ctx.stroke();
      }
    }
    // 小地图观察点标记（点击/拖拽时显示十字准星）
    if (this._minimapFocus && this._minimapFocusTimer > 0) {
      const fx = toX(this._minimapFocus.x), fy = toY(this._minimapFocus.z);
      ctx.strokeStyle = '#ffdd66';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(fx - 4, fy); ctx.lineTo(fx + 4, fy);
      ctx.moveTo(fx, fy - 4); ctx.lineTo(fx, fy + 4);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(255,221,102,0.5)';
      ctx.strokeRect(fx - 4, fy - 4, 8, 8);
    }
  }

  updateSceneEffects(t) {
    // 野怪上下浮动（按各类型 bodyY 浮动，避免硬编码）
    for (const j of this.jungle) {
      if (j.mesh && j.hp > 0 && j.bodyMesh) {
        const baseY = j.bodyY != null ? j.bodyY : 0.8;
        j.bodyMesh.position.y = baseY + Math.sin(t * 2 + j.x) * 0.1;
      }
    }
    // 辅助英雄头顶光环旋转
    for (const h of this.heroes) {
      if (h.alive && h.haloMesh) {
        h.haloMesh.rotation.z += 0.03;
      }
    }
    // 主水晶旋转已在 _updateTowers 处理

    // 小地图观察：覆盖相机位置看向目标点（在 _updateGameCamera 之后执行）
    if (this._minimapFocus && this._minimapFocusTimer > 0) {
      const camera = this.App.camera;
      const THREE = this.THREE;
      if (camera) {
        const dt = this._lastDt || 0.016;
        this._minimapFocusTimer -= dt;
        if (this._minimapFocusTimer <= 0) {
          this._minimapFocus = null;
          this._minimapFocusTimer = 0;
        } else {
          // 相机斜俯视目标点：高度 12，水平偏移 8（保持俯视角度）
          const fx = this._minimapFocus.x, fz = this._minimapFocus.z;
          const camHeight = 14;
          const camOffset = 6;
          // 复用当前相机的方位角，保持视角方向一致
          const azimuth = this.App._gameCamAzimuth || 0;
          const targetCamX = fx + camOffset * Math.sin(azimuth);
          const targetCamZ = fz + camOffset * Math.cos(azimuth);
          const targetCamY = camHeight;
          // 平滑过渡到目标位置
          if (!this._minimapCamPos) this._minimapCamPos = new THREE.Vector3();
          this._minimapCamPos.set(targetCamX, targetCamY, targetCamZ);
          camera.position.lerp(this._minimapCamPos, 0.12);
          camera.lookAt(fx, 1.2, fz);
        }
      }
    }
  }

  /** MOBA 不需要跳跃 */
  requestJump() {
    return false;
  }
}

export default MobaGame;
