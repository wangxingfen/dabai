/* ============================================================
 * ZeroShotEval — 零样本可玩性评测脚本（P3-2，Genie 式）
 *
 * 目标（对应方案报告 P3-2：生成环境评测）：
 * - 用程序化生成器产出"未见关卡"（训练中未出现的 seed）
 * - 加载已训练权重（不训练）在未见关卡上直接游玩
 * - 统计可玩性指标：完成率 / 平均步数 / 平均进度 / 撞墙率
 * - 与人类基线对比，通过门禁（successRate ≥ 人类基线 × 阈值）
 *
 * 设计：
 * - 内置 HeadlessTreasureHunt：复刻 treasure-hunt 的 RL 契约
 *   （观察 13 维 / 动作 4 方向 / 碰撞 / 收集 / 胜利），无 THREE 依赖
 * - 评测循环：reset(level) → while !done: chooseAction → applyAction → step
 * - 结果含 PASS/FAIL 门禁判定
 * ============================================================ */

import { NeuralNetV2 } from "./nn-advanced.ts";
import { encodeObservation, observationDim } from "./observation-spec.ts";
import { generateLevel, generateSeeds, describeLevel } from "./level-procedural.ts";



// ==================== 类型 ====================

/** ZeroShotEvaluator 构造选项（全部可选，与原 JSDoc 一致） */
interface ZeroShotEvalOptions {
  episodes?: number;
  seedCount?: number;
  baseSeed?: number;
  humanBaseline?: { successRate: number; avgTimeSec?: number };
  passRatio?: number;
}

/** 单关评测结果（successRate 在评测后补写） */
interface LevelResult {
  seed: number;
  note: string;
  wins: number;
  episodes: number;
  steps: number[];
  progresses: number[];
  hitWallRates: number[];
  successRate?: number;
}

// ==================== 常量 ====================

const CELL_SIZE = 2;
const MOVE_SPEED = 4;
const PLAYER_RADIUS = 0.4;
const MAX_STEPS = 400;            // 单局最大决策步（防死循环）
const PASS_RATIO = 0.6;           // 门禁：AI 完成率 ≥ 人类基线 × 0.6
const REWARD = { STEP: -0.05, HIT_WALL: -0.3, COLLECT: 3, WIN: 10, APPROACH: 0.15 };
const HUMAN_BASELINE = { successRate: 0.8, avgTimeSec: 90 };  // 与 humanBaseline.json 一致

// ==================== Headless 迷宫环境 ====================

export class HeadlessTreasureHunt {
  /**
   * @param {Object} level - generateLevel('treasure_hunt', seed) 的返回
   */
  constructor(level: any) {
    this.level = level;
    this._mazeRows = level.rows;
    this._mazeCols = level.cols;
    this._mazeData = level.maze;
    this._cellSize = CELL_SIZE;

    // 收集品 / 宝藏世界坐标
    this._collectibles = level.collectibles.map(({ r, c }) => {
      const p = this._cellWorldPos(r, c);
      return { x: p.x, z: p.z, alive: true };
    });
    const tp = this._cellWorldPos(level.treasure.r, level.treasure.c);
    this._treasure = { x: tp.x, z: tp.z, alive: true };
    this._totalCollectibles = this._collectibles.length;

    // 玩家
    const sp = this._cellWorldPos(level.spawn.r, level.spawn.c);
    this._px = sp.x;
    this._pz = sp.z;
    this._collectedCount = 0;
    this._hitWall = 0;
    this._steps = 0;

    // 碰撞盒（墙体）
    this._wallBoxes = [];
    const halfCell = this._cellSize * 0.45;
    for (let r = 0; r < this._mazeRows; r++) {
      for (let c = 0; c < this._mazeCols; c++) {
        if (this._mazeData[r][c] === 1) {
          const p = this._cellWorldPos(r, c);
          this._wallBoxes.push({ minX: p.x - halfCell, maxX: p.x + halfCell, minZ: p.z - halfCell, maxZ: p.z + halfCell });
        }
      }
    }
  }

  _cellWorldPos(r, c) {
    const offsetX = -this._mazeCols * this._cellSize / 2 + this._cellSize / 2;
    const offsetZ = -this._mazeRows * this._cellSize / 2 + this._cellSize / 2;
    return { x: offsetX + c * this._cellSize, z: offsetZ + r * this._cellSize };
  }

  _worldToCell(px, pz) {
    const offsetX = -this._mazeCols * this._cellSize / 2 + this._cellSize / 2;
    const offsetZ = -this._mazeRows * this._cellSize / 2 + this._cellSize / 2;
    const c = Math.round((px - offsetX) / this._cellSize);
    const r = Math.round((pz - offsetZ) / this._cellSize);
    if (r < 0 || r >= this._mazeRows || c < 0 || c >= this._mazeCols) return null;
    return { r, c };
  }

  getActionSpec() {
    return ['up', 'down', 'left', 'right'].map((name, id) => ({ id, name, semantics: 'semantic', executable: true }));
  }

  getObservationSpec() {
    return [
      { name: 'walls', kind: 'vector', dim: 4, scale: 1, offset: 0 },
      { name: 'nearest', kind: 'vector', dim: 3, scale: 1, offset: 0 },
      { name: 'treasure', kind: 'vector', dim: 3, scale: 1, offset: 0 },
      { name: 'progress', kind: 'scalar', scale: 1, offset: 0 },
      { name: 'hitWall', kind: 'scalar', scale: 1, offset: 0 },
      { name: 'speed', kind: 'scalar', scale: 5, offset: 0 },
    ];
  }

  getObservation() {
    const px = this._px, pz = this._pz;
    const walls = this._getSurroundingWalls(px, pz);

    let nearest = [0, 0, 1], nearestDist = Infinity;
    for (const o of this._collectibles) {
      if (!o.alive) continue;
      const d = Math.hypot(o.x - px, o.z - pz);
      if (d < nearestDist) { nearestDist = d; nearest = [o.x - px, o.z - pz, d]; }
    }
    if (nearestDist < Infinity) {
      nearest = [nearest[0] / 8, nearest[1] / 8, Math.min(1, nearest[2] / 12)];
    }

    let tre = [0, 0, 1];
    if (this._treasure.alive) {
      const dx = this._treasure.x - px, dz = this._treasure.z - pz;
      tre = [dx / 12, dz / 12, Math.min(1, Math.hypot(dx, dz) / 16)];
    }

    return encodeObservation(this.getObservationSpec(), {
      walls,
      nearest,
      treasure: tre,
      progress: this._totalCollectibles > 0 ? this._collectedCount / this._totalCollectibles : 0,
      hitWall: this._hitWall,
      speed: 0,
    });
  }

  _getSurroundingWalls(px, pz) {
    const cell = this._worldToCell(px, pz);
    if (!cell) return [1, 1, 1, 1];
    const { r, c } = cell;
    const isWall = (rr, cc) => {
      if (rr < 0 || rr >= this._mazeRows || cc < 0 || cc >= this._mazeCols) return 1;
      return this._mazeData[rr][cc] === 1 ? 1 : 0;
    };
    return [isWall(r - 1, c), isWall(r + 1, c), isWall(r, c - 1), isWall(r, c + 1)];
  }

  /** 执行动作并推进环境，返回 {reward, done, win, hitWall} */
  step(actionId) {
    const prevPos = { x: this._px, z: this._pz };
    const prevCount = this._collectedCount;
    const prevTreasure = this._treasure.alive;
    const prevDist = this._nearestCollectibleDist();

    const stepSec = 0.18;   // 固定类人决策间隔（与 P2-3 默认区间中值一致）
    switch (actionId) {
      case 0: this._pz -= MOVE_SPEED * stepSec; break;
      case 1: this._pz += MOVE_SPEED * stepSec; break;
      case 2: this._px -= MOVE_SPEED * stepSec; break;
      case 3: this._px += MOVE_SPEED * stepSec; break;
      default: break;
    }
    this._checkWallCollision();

    let reward = REWARD.STEP;
    const moved = Math.hypot(this._px - prevPos.x, this._pz - prevPos.z);
    if (moved < 0.05) { reward += REWARD.HIT_WALL; this._hitWall = 1; }
    else this._hitWall = 0;

    const gained = this._collectedCount - prevCount;
    if (gained > 0) reward += REWARD.COLLECT * gained;
    if (prevTreasure && !this._treasure.alive) reward += REWARD.WIN;
    const nowDist = this._nearestCollectibleDist();
    if (prevDist !== Infinity && nowDist < prevDist - 0.15) reward += REWARD.APPROACH;

    this._steps++;
    const done = this.rlDone() || this._steps >= MAX_STEPS;
    return { reward, done, win: this._collectedCount >= this._totalCollectibles, hitWall: this._hitWall };
  }

  _checkWallCollision() {
    const r = PLAYER_RADIUS;
    for (const w of this._wallBoxes) {
      const closestX = Math.max(w.minX - r, Math.min(this._px, w.maxX + r));
      const closestZ = Math.max(w.minZ - r, Math.min(this._pz, w.maxZ + r));
      const dx = this._px - closestX;
      const dz = this._pz - closestZ;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < r && dist > 0.001) {
        const overlap = r - dist;
        const nx = dx / dist;
        const nz = dz / dist;
        this._px += nx * overlap;
        this._pz += nz * overlap;
      }
    }
    // 收集品拾取
    for (const o of this._collectibles) {
      if (!o.alive) continue;
      if (Math.hypot(o.x - this._px, o.z - this._pz) < 0.9) {
        o.alive = false;
        this._collectedCount++;
      }
    }
    // 宝藏拾取
    if (this._treasure.alive && Math.hypot(this._treasure.x - this._px, this._treasure.z - this._pz) < 0.9) {
      this._treasure.alive = false;
    }
  }

  _nearestCollectibleDist() {
    let min = Infinity;
    for (const o of this._collectibles) {
      if (!o.alive) continue;
      const d = Math.hypot(o.x - this._px, o.z - this._pz);
      if (d < min) min = d;
    }
    return min;
  }

  rlDone() {
    return this._collectedCount >= this._totalCollectibles && !this._treasure.alive;
  }

  getStats() {
    return {
      steps: this._steps,
      collected: this._collectedCount,
      total: this._totalCollectibles,
      progress: this._totalCollectibles > 0 ? this._collectedCount / this._totalCollectibles : 0,
      win: this.rlDone(),
      hitWallRate: this._steps > 0 ? (this._hitWallAccum ?? 0) / this._steps : 0,
    };
  }

  // ===== 类型声明（仅类型注解，无运行时副作用；_hitWallAccum 原 JS 中从未赋值） =====
  declare level: any;
  declare _mazeRows: number;
  declare _mazeCols: number;
  declare _mazeData: number[][];
  declare _cellSize: number;
  declare _collectibles: Array<{ x: number; z: number; alive: boolean }>;
  declare _treasure: { x: number; z: number; alive: boolean };
  declare _totalCollectibles: number;
  declare _px: number;
  declare _pz: number;
  declare _collectedCount: number;
  declare _hitWall: number;
  declare _steps: number;
  declare _wallBoxes: Array<{ minX: number; maxX: number; minZ: number; maxZ: number }>;
  declare _hitWallAccum: number | undefined;
}

// ==================== 零样本评测器 ====================

export class ZeroShotEvaluator {
  /**
   * @param {Object} opts
   * @param {number} [opts.episodes=12] 每关评测局数
   * @param {number} [opts.seedCount=6]  未见关卡数量
   * @param {number} [opts.baseSeed=2026]
   * @param {Object} [opts.humanBaseline=HUMAN_BASELINE]
   * @param {number} [opts.passRatio=PASS_RATIO]
   */
  constructor(opts: ZeroShotEvalOptions = {}) {
    this.episodes = opts.episodes ?? 12;
    this.seedCount = opts.seedCount ?? 6;
    this.baseSeed = opts.baseSeed ?? 2026;
    this.humanBaseline = opts.humanBaseline ?? HUMAN_BASELINE;
    this.passRatio = opts.passRatio ?? PASS_RATIO;
    this.results = [];
  }

  /**
   * 加载权重并评测
   * @param {Object} weights - agent.onlineNet 序列化权重 {layers, weights, ...}
   * @param {Object} cfg - {stateSize, nActions, hiddenLayers}
   * @param {Function} [policyFn] 可选：外部策略覆盖（(obs)=>actionId），
   *   用于冒烟测试演示"可玩性通过"门禁路径；缺省用网络贪心。
   * @returns {{pass, summary, perLevel}}
   */
  async evaluate(weights, cfg, policyFn = null) {
    // 重建网络（只推理，不训练 → 零样本）
    const net = (weights && !policyFn)
      ? (() => { const n = new NeuralNetV2([cfg.stateSize, ...(cfg.hiddenLayers || [16, 16]), cfg.nActions], { lr: 0, noisy: false, seed: 1 } as any); n.fromJSON(weights); return n; })()
      : null;

    const seeds = generateSeeds(this.seedCount, this.baseSeed);
    const perLevel = [];
    let totalWin = 0, totalEpisodes = 0;

    for (const seed of seeds) {
      const level = generateLevel('treasure_hunt', seed, 0.5);
      const levelResult: LevelResult = { seed, note: describeLevel(level), wins: 0, episodes: 0, steps: [], progresses: [], hitWallRates: [] };
      for (let e = 0; e < this.episodes; e++) {
        const env = new HeadlessTreasureHunt(level);
        let done = false;
        let hitWallAccum = 0;
        let wins = 0, steps = 0;
        let guard = 0;
        while (!done && guard++ < 500) {
          const obs = env.getObservation();
          // 外部策略覆盖（演示用）或网络贪心（零样本）
          let a;
          if (policyFn) {
            a = policyFn(obs, env);
          } else {
            const out = net.predict(obs);
            a = 0;
            for (let j = 1; j < out.length; j++) if (out[j] > out[a]) a = j;
          }
          const r = env.step(a);
          hitWallAccum += r.hitWall;
          done = r.done;
          steps = env._steps;
          if (r.win) wins = 1;
        }
        totalWin += wins;
        totalEpisodes++;
        levelResult.wins += wins;
        levelResult.episodes++;
        levelResult.steps.push(steps);
        levelResult.progresses.push(env.getStats().progress);
        levelResult.hitWallRates.push(steps > 0 ? hitWallAccum / steps : 0);
      }
      levelResult.successRate = levelResult.wins / levelResult.episodes;
      perLevel.push(levelResult);
    }

    const successRate = totalEpisodes > 0 ? totalWin / totalEpisodes : 0;
    const allSteps = perLevel.flatMap(l => l.steps);
    const allProg = perLevel.flatMap(l => l.progresses);
    const allHw = perLevel.flatMap(l => l.hitWallRates);
    const avgSteps = allSteps.length ? allSteps.reduce((a, b) => a + b, 0) / allSteps.length : 0;
    const avgProgress = allProg.length ? allProg.reduce((a, b) => a + b, 0) / allProg.length : 0;
    const avgHitWall = allHw.length ? allHw.reduce((a, b) => a + b, 0) / allHw.length : 0;

    const humanRate = this.humanBaseline.successRate;
    const ratio = humanRate > 0 ? successRate / humanRate : 0;
    const pass = ratio >= this.passRatio;

    const summary = {
      pass,
      successRate, ratio,
      humanRate, passRatio: this.passRatio,
      avgSteps, avgProgress, avgHitWall,
      episodes: totalEpisodes, levels: perLevel.length,
    };
    this.results.push(summary);
    return { pass, summary, perLevel };
  }

  /** 汇总历史评测 */
  report() {
    return this.results;
  }

  // ===== 类型声明（仅类型注解，无运行时副作用） =====
  declare episodes: number;
  declare seedCount: number;
  declare baseSeed: number;
  declare humanBaseline: { successRate: number; avgTimeSec?: number };
  declare passRatio: number;
  declare results: any[];

  /**
   * 内置 BFS 寻路基准策略（可作为可玩性基准 / 评测框架自检）。
   * 通过环境内部迷宫网格做 BFS 到目标，返回第一步方向动作。
   * 策略要点：目标锁定（选定后不因途中更近目标而切换，避免折返抖动）；
   * 目标被拾取失效后重新选最近目标；到达目标格后朝中心收敛触发拾取。
   * @returns {(obs, env) => number}
   */
  static createBFSHeuristic() {
    // 跨调用维护：当前目标 + 待走路径（避免每步重算与抖动）
    let path = [];      // [{r,c,d}] d=动作方向
    let targetKey = null;

    // 指定 cell 是否仍有存活目标（收集品或宝藏）
    const isCellAlive = (env, tr, tc) => {
      for (const o of env._collectibles) {
        if (!o.alive) continue;
        const oc = env._worldToCell(o.x, o.z);
        if (oc && oc.r === tr && oc.c === tc) return true;
      }
      if (env._treasure.alive) {
        const tc2 = env._worldToCell(env._treasure.x, env._treasure.z);
        if (tc2 && tc2.r === tr && tc2.c === tc) return true;
      }
      return false;
    };

    // 最近存活目标（曼哈顿距离，收集品优先于宝藏兜底）
    const pickTarget = (env, r, c) => {
      let target = null, minDist = Infinity;
      for (const o of env._collectibles) {
        if (!o.alive) continue;
        const oc = env._worldToCell(o.x, o.z);
        if (!oc) continue;
        const d = Math.abs(oc.r - r) + Math.abs(oc.c - c);
        if (d < minDist) { minDist = d; target = oc; }
      }
      if (env._treasure.alive) {
        const tc = env._worldToCell(env._treasure.x, env._treasure.z);
        if (tc) {
          const d = Math.abs(tc.r - r) + Math.abs(tc.c - c);
          if (d < minDist) { minDist = d; target = tc; }
        }
      }
      return target;
    };

    // BFS 网格寻路，回溯完整路径 [{r,c,d}]
    const bfsPath = (env, r, c, target) => {
      const rows = env._mazeRows, cols = env._mazeCols, maze = env._mazeData;
      const q = [[r, c]];
      const prev = new Map([[`${r},${c}`, null]]);
      const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];  // 上 下 左 右
      let found = null;
      while (q.length && !found) {
        const [cr, cc] = q.shift();
        for (let d = 0; d < 4; d++) {
          const nr = cr + dirs[d][0], nc = cc + dirs[d][1];
          if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
          if (maze[nr][nc] === 1) continue;
          const key = `${nr},${nc}`;
          if (prev.has(key)) continue;
          prev.set(key, { pr: cr, pc: cc, d });
          if (nr === target.r && nc === target.c) { found = [nr, nc]; break; }
          q.push([nr, nc]);
        }
      }
      if (!found) return null;
      const rev = [];
      let cur = found;
      while (true) {
        const node = prev.get(`${cur[0]},${cur[1]}`);
        if (!node) break;
        rev.push({ r: cur[0], c: cur[1], d: node.d });
        if (node.pr === r && node.pc === c) break;
        cur = [node.pr, node.pc];
      }
      return rev.reverse();
    };

    return (obs, env) => {
      const cell = env._worldToCell(env._px, env._pz);
      if (!cell) return 0;
      const { r, c } = cell;

      // 1) 锁定目标是否仍然有效（未被拾取）
      let curTarget = null;
      if (targetKey) {
        const [tr, tc] = targetKey.split(',').map(Number);
        if (!isCellAlive(env, tr, tc)) {
          path = [];            // 目标被拾取（或顺路拿掉）→ 失效
          targetKey = null;
        } else {
          curTarget = { r: tr, c: tc };
        }
      }

      // 2) 已到目标 cell：朝中心收敛触发拾取（不切换目标）
      if (curTarget && curTarget.r === r && curTarget.c === c) {
        path = [];
        targetKey = null;
        const wx = env._cellWorldPos(curTarget.r, curTarget.c).x - env._px;
        const wz = env._cellWorldPos(curTarget.r, curTarget.c).z - env._pz;
        if (Math.abs(wx) >= Math.abs(wz)) return wx > 0 ? 3 : 2;
        return wz > 0 ? 1 : 0;
      }

      // 3) 路径仍有效（目标锁定）：沿路径走，途中不因更近目标而切换
      if (path.length && curTarget) {
        const next = path[0];
        if (next.r === r && next.c === c) {
          path.shift();         // 当前格已到达
          if (path.length) return path[0].d;
        } else {
          for (let i = 0; i < path.length; i++) {
            if (path[i].r === r && path[i].c === c) {
              return path[i + 1] ? path[i + 1].d : path[path.length - 1].d;
            }
          }
          path = [];            // 玩家不在路径上（异常）→ 重算
        }
      }

      // 4) 无有效路径：重新选最近目标并 BFS
      const target = pickTarget(env, r, c);
      if (!target) return 0;
      const newPath = bfsPath(env, r, c, target);
      if (!newPath) return 0;
      path = newPath;
      targetKey = `${target.r},${target.c}`;
      return path.length ? path[0].d : 0;
    };
  }
}

export default ZeroShotEvaluator;
