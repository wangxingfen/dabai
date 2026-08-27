/* ============================================================
 * LevelProcedural — 程序化关卡生成器（P3-2，Genie 式）
 *
 * 目标（对应方案报告 P3-2：生成环境评测）：
 * - 用 seed 确定性生成"未见关卡"（训练时未出现的关卡布局）
 * - 评测脚本在未见关卡上做零样本可玩性门禁
 *   （加载已训练权重 → 不训练 → 直接跑 → 统计完成率/耗时）
 *
 * 支持的关卡类型：
 * - treasure_hunt：迷宫矩阵（DFS 生成，seed 驱动）+ 收集品/宝藏位置
 * - sandbox       ：地形种子（方块布局的确定性随机）
 * - mario         ：障碍序列（跳跃时机分布）
 * - moba_5v5      ：兵线相位序列（对战节奏）
 *
 * 设计：
 * - mulberry32 确定性随机：相同 seed → 相同关卡（可复现、可对比）
 * - 难度参数 difficulty ∈ [0,1] 调整关卡规模/密度
 * ============================================================ */

import { mulberry32 } from './world-model.ts';

// ==================== 关卡生成 ====================

/**
 * 生成指定游戏的程序化关卡
 * @param {string} gameKey 游戏 key
 * @param {number} [seed]  随机种子（缺省用当前时间）
 * @param {number} [difficulty=0.5] 难度 [0,1]
 * @returns {{gameKey, seed, difficulty, layout, spawn, collectibles, treasure, note}}
 */
export function generateLevel(gameKey: string, seed?: number, difficulty: number = 0.5): any {
  const s = seed !== undefined && seed !== null ? (seed >>> 0) : ((Date.now() & 0x7fffffff) >>> 0);
  const d = Math.max(0, Math.min(1, difficulty));
  switch (gameKey) {
    case 'treasure_hunt': return generateTreasureLevel(s, d);
    case 'sandbox':       return generateSandboxLevel(s, d);
    case 'mario':         return generateMarioLevel(s, d);
    case 'moba_5v5':      return generateMobaLevel(s, d);
    default:
      return { gameKey, seed: s, difficulty: d, layout: null, spawn: null,
               collectibles: [], treasure: null, note: `未注册的关卡类型: ${gameKey}` };
  }
}

/** 迷宫寻宝：seed 驱动 DFS 迷宫 + 收集品 + 宝藏 */
function generateTreasureLevel(seed, difficulty) {
  const rng = mulberry32(seed ^ 0x9E3779B9);
  // 规模随难度：11/13/15 奇数网格
  const base = 11 + 2 * Math.floor(difficulty * 3);   // 11,13,15,17
  const rows = base, cols = base;

  // 确定性 DFS 迷宫（随机化使用 rng）
  const maze = [];
  for (let r = 0; r < rows; r++) {
    maze[r] = [];
    for (let c = 0; c < cols; c++) maze[r][c] = 1;
  }
  const visited = new Set();
  const dfs = (r, c) => {
    const key = `${r},${c}`;
    if (visited.has(key)) return;
    visited.add(key);
    maze[r][c] = 0;
    const dirs = [[-2, 0], [2, 0], [0, -2], [0, 2]];
    for (let i = dirs.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [dirs[i], dirs[j]] = [dirs[j], dirs[i]];
    }
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited.has(`${nr},${nc}`)) {
        maze[r + dr / 2][c + dc / 2] = 0;
        dfs(nr, nc);
      }
    }
  };
  dfs(1, 1);

  // 空通路口袋
  const empty = [];
  for (let r = 1; r < rows - 1; r += 2) {
    for (let c = 1; c < cols - 1; c += 2) {
      if (maze[r][c] === 0) empty.push([r, c]);
    }
  }

  // 收集品数量随难度（4~10）
  const nCollect = 4 + Math.floor(difficulty * 6);
  const used = new Set(['1,1']);
  const collectibles = [];
  let guard = 0;
  while (collectibles.length < nCollect && guard++ < 400 && empty.length) {
    const cell = empty[(rng() * empty.length) | 0];
    const key = `${cell[0]},${cell[1]}`;
    if (used.has(key)) continue;
    used.add(key);
    collectibles.push({ r: cell[0], c: cell[1] });
  }

  // 宝藏：迷宫最远通路
  let best = { r: rows - 2, c: cols - 2, dist: 0 };
  for (const [r, c] of empty) {
    const dist = Math.abs(r - 1) + Math.abs(c - 1);
    if (dist > best.dist) best = { r, c, dist };
  }

  return {
    gameKey: 'treasure_hunt', seed, difficulty,
    rows, cols, maze,
    spawn: { r: 1, c: 1 },
    collectibles,
    treasure: { r: best.r, c: best.c },
    note: `迷宫 ${rows}x${cols}，收集 ${collectibles.length}，难度 ${difficulty.toFixed(2)}`,
  };
}

/** 沙盒：方块布局 seed（确定性资源分布） */
function generateSandboxLevel(seed, difficulty) {
  const rng = mulberry32(seed ^ 0x85EBCA6B);
  const size = 16 + Math.floor(difficulty * 16);       // 16~32
  const cells = [];
  const nCells = 40 + Math.floor(difficulty * 60);
  for (let i = 0; i < nCells; i++) {
    cells.push({
      x: Math.floor(rng() * size) - Math.floor(size / 2),
      z: Math.floor(rng() * size) - Math.floor(size / 2),
      type: rng() < 0.55 ? 'block' : (rng() < 0.5 ? 'resource' : 'decor'),
      resource: rng() < 0.3 ? 'gold' : 'stone',
    });
  }
  return {
    gameKey: 'sandbox', seed, difficulty,
    layout: { size, cells },
    spawn: { x: 0, z: 0 },
    collectibles: cells.filter(c => c.type === 'resource'),
    treasure: null,
    note: `沙盒 ${size}x${size}，${cells.length} 个方块`,
  };
}

/** 马里奥：障碍序列（跳跃时机） */
function generateMarioLevel(seed, difficulty) {
  const rng = mulberry32(seed ^ 0x2545F491);
  const len = 200 + Math.floor(difficulty * 300);
  const obstacles = [];
  let x = 10;
  while (x < len) {
    const gap = 4 + rng() * 8 * (1 + difficulty);   // 障碍间距随难度
    const h = 1 + rng() * 2 * (1 + difficulty);     // 障碍高度
    obstacles.push({ x: Math.floor(x), h: Math.min(5, Math.round(h)) });
    x += gap;
  }
  return {
    gameKey: 'mario', seed, difficulty,
    layout: { length: len, obstacles },
    spawn: { x: 0, y: 0 },
    collectibles: [],
    treasure: null,
    note: `跑酷 ${len}m，${obstacles.length} 个障碍`,
  };
}

/** MOBA：兵线相位序列（节奏分布） */
function generateMobaLevel(seed, difficulty) {
  const rng = mulberry32(seed ^ 0x3C6EF372);
  const phases = [];
  const n = 24 + Math.floor(difficulty * 24);
  for (let i = 0; i < n; i++) {
    phases.push({
      t: i * (2 + rng() * 2),
      lane: ['mid', 'top', 'bot'][(rng() * 3) | 0],
      enemyPressure: 0.3 + rng() * 0.7 * difficulty,
    });
  }
  return {
    gameKey: 'moba_5v5', seed, difficulty,
    layout: { phases },
    spawn: null,
    collectibles: [],
    treasure: null,
    note: `兵线相位 ${n} 段`,
  };
}

// ==================== 工具 ====================

/** 生成 N 个互不相同的关卡种子（供评测批量使用） */
export function generateSeeds(n: number, baseSeed: number = 2026): number[] {
  const seeds = [];
  const seen = new Set();
  let s = baseSeed >>> 0;
  while (seeds.length < n) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;  // LCG
    if (seen.has(s)) continue;
    seen.add(s);
    seeds.push(s);
  }
  return seeds;
}

/** 关卡布局 → 文本描述（供日志/评测输出） */
export function describeLevel(level: any): string {
  if (!level) return 'null';
  return `${level.gameKey}#${level.seed} ${level.note || ''}`;
}

export default { generateLevel, generateSeeds, describeLevel };