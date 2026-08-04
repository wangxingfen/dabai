/* ============================================================
 * 象棋 AI 引擎 —— 负极大搜索 + 阿尔法贝塔剪枝 + 静态评估
 *
 * 特性：
 * - 快速将军检测（isSquareAttacked 逐棋型布尔判定，避免全量走法生成）
 * - 迭代加深 + 时间控制
 * - 静态搜索（吃子序列）缓解水平线效应
 * - 局面评估：子力 + 位置表(马/炮/兵/士/象/帅) + 机动性
 * - findBestMoves() 供"棋盘快照/局势分析"面板输出候选着法
 * ============================================================ */
import { PIECE, BOARD_COLS, BOARD_ROWS, pieceType, pieceSide } from './engine.js';

// ---------- 常量 ----------
export const MATERIAL = { 1: 10000, 2: 2000, 3: 2000, 4: 4000, 5: 9000, 6: 4500, 7: 1000 };
const MATE = 100000;
const INF = 2000000;

// 位置表：行=rank(0黑底..9红底)，列=file(0..8)，红方视角；黑方使用时上下翻转
const PST_HORSE = [
  [-60, -30, -10,   0,   0,   0, -10, -30, -60],
  [-30,   0,  10,  20,  20,  20,  10,   0, -30],
  [-10,  10,  30,  40,  40,  40,  30,  10, -10],
  [  0,  20,  40,  60,  60,  60,  40,  20,   0],
  [  0,  20,  40,  60,  70,  60,  40,  20,   0],
  [  0,  20,  40,  60,  70,  60,  40,  20,   0],
  [-10,  10,  30,  40,  40,  40,  30,  10, -10],
  [-30,   0,  10,  20,  20,  20,  10,   0, -30],
  [-60, -30, -10,   0,   0,   0, -10, -30, -60],
  [-80, -50, -20,   0,   0,   0, -20, -50, -80],
];
const PST_CANNON = [
  [ 10,  10,  10,  10,  10,  10,  10,  10,  10],
  [ 10,  30,  10,  20,  20,  20,  10,  30,  10],
  [ 10,  20,  10,  20,  20,  20,  10,  20,  10],
  [ 10,  10,  20,  30,  30,  30,  20,  10,  10],
  [  0,  10,  20,  30,  40,  30,  20,  10,   0],
  [  0,  10,  20,  30,  40,  30,  20,  10,   0],
  [  0,   0,  10,  20,  20,  20,  10,   0,   0],
  [  0,   0,   0,  10,  10,  10,   0,   0,   0],
  [  0,   0,   0,   0,   0,   0,   0,   0,   0],
  [  0,   0,   0,   0,   0,   0,   0,   0,   0],
];
const PST_PAWN = [
  [ 30,  40,  60,  80, 100,  80,  60,  40,  30],
  [ 20,  30,  40,  60,  80,  60,  40,  30,  20],
  [ 10,  20,  30,  40,  60,  40,  30,  20,  10],
  [ 10,  10,  20,  30,  40,  30,  20,  10,  10],
  [  0,   0,  10,  20,  30,  20,  10,   0,   0],
  [  0,   0,  10,  20,  30,  20,  10,   0,   0],
  [  0,   0,   0,   0,   0,   0,   0,   0,   0],
  [  0,   0,   0,   0,   0,   0,   0,   0,   0],
  [  0,   0,   0,   0,   0,   0,   0,   0,   0],
  [  0,   0,   0,   0,   0,   0,   0,   0,   0],
];
const PST_ADVISOR = [
  [0,0,0,  0, 0, 0, 0,0,0],
  [0,0,0,  0,10, 0, 0,0,0],
  [0,0,0, 10, 0,10, 0,0,0],
  [0,0,0,  0, 0, 0, 0,0,0],
  [0,0,0,  0, 0, 0, 0,0,0],
  [0,0,0,  0, 0, 0, 0,0,0],
  [0,0,0,  0, 0, 0, 0,0,0],
  [0,0,0, 10, 0,10, 0,0,0],
  [0,0,0,  0,10, 0, 0,0,0],
  [0,0,0,  0, 0, 0, 0,0,0],
];
const PST_ELEPHANT = [
  [0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0],
  [0,10,0,10,0,10,0,10,0],
  [0,10,0,10,0,10,0,10,0],
  [0, 0,0, 0,0, 0,0, 0,0],
  [0,10,0,10,0,10,0,10,0],
  [0, 0,0, 0,0, 0,0, 0,0],
];
const PST_KING = [
  [0,0,0, 0, 0, 0,0,0,0],
  [0,0,0, 0, 0, 0,0,0,0],
  [0,0,0, 0,10, 0,0,0,0],
  [0,0,0, 0, 0, 0,0,0,0],
  [0,0,0, 0, 0, 0,0,0,0],
  [0,0,0, 0, 0, 0,0,0,0],
  [0,0,0, 0, 0, 0,0,0,0],
  [0,0,0, 0, 0, 0,0,0,0],
  [0,0,0, 0,10, 0,0,0,0],
  [0,0,0, 0,20, 0,0,0,0],
];
const PST_BY_TYPE = {
  1: PST_KING, 2: PST_ADVISOR, 3: PST_ELEPHANT,
  4: PST_HORSE, 5: null, 6: PST_CANNON, 7: PST_PAWN,
};

// 机动性权重（每多一步合法走子的加分，厘兵）
const MOBILITY_W = { 4: 2.2, 5: 3.5, 6: 2.8 };

// ---------- 快速将军/攻击检测 ----------
const HORSE_STEPS = [
  [1, 2, 0, 1], [-1, 2, 0, 1], [1, -2, 0, -1], [-1, -2, 0, -1],
  [2, 1, 1, 0], [2, -1, 1, 0], [-2, 1, -1, 0], [-2, -1, -1, 0],
];
const ORTHO = [[1, 0], [-1, 0], [0, 1], [0, -1]];

/** 目标格 (c,r) 是否被 bySide 攻击（bySide: 'red'|'black'） */
export function isSquareAttacked(eng, c, r, bySide) {
  // 车 / 炮 / 帅（相邻 & 对脸）
  for (const [dc, dr] of ORTHO) {
    let nc = c + dc, nr = r + dr;
    let screen = 0;
    while (nc >= 0 && nc < BOARD_COLS && nr >= 0 && nr < BOARD_ROWS) {
      const code = eng.get(nc, nr);
      if (code !== 0) {
        screen++;
        if (screen === 1) {
          if (pieceSide(code) === bySide && pieceType(code) === 5) return true;      // 车
          if (Math.abs(nc - c) + Math.abs(nr - r) === 1 && pieceSide(code) === bySide && pieceType(code) === 1) return true; // 帅一步
          if (nc === c && pieceSide(code) === bySide && pieceType(code) === 1) return true; // 对脸飞将
        } else if (screen === 2) {
          if (pieceSide(code) === bySide && pieceType(code) === 6) return true;      // 炮
          break;
        } else break;
      }
      nc += dc; nr += dr;
    }
  }
  // 马
  for (const [dc, dr, lc, lr] of HORSE_STEPS) {
    const nc = c + dc, nr = r + dr;
    if (nc < 0 || nc >= BOARD_COLS || nr < 0 || nr >= BOARD_ROWS) continue;
    const code = eng.get(nc, nr);
    if (code !== 0 && pieceSide(code) === bySide && pieceType(code) === 4 && eng.get(c + lc, r + lr) === 0) return true;
  }
  // 兵卒
  const fwd = bySide === 'red' ? 1 : -1; // 攻击者相对目标的正前方偏移
  const across = bySide === 'red' ? 4 : 5; // 已过河判定阈值（红: rank<=4，黑: rank>=5）
  const p1 = eng.get(c, r + fwd);
  if (p1 !== 0 && pieceSide(p1) === bySide && pieceType(p1) === 7) return true;
  for (const sdc of [-1, 1]) {
    const nc = c + sdc;
    if (nc < 0 || nc >= BOARD_COLS) continue;
    const p = eng.get(nc, r);
    if (p !== 0 && pieceSide(p) === bySide && pieceType(p) === 7) {
      // 横向攻击仅当过河后成立（按攻击者自身位置判断）
      if (bySide === 'red' ? r <= across : r >= across) return true;
    }
  }
  return false;
}

/** 快速判断某方是否被将军 */
export function isInCheckFast(eng, side) {
  const king = eng.findKing(side);
  if (!king) return false;
  return isSquareAttacked(eng, king[0], king[1], side === 'red' ? 'black' : 'red');
}

/** 生成合法走法（快速将军过滤；capturesOnly 时仅吃子走法） */
export function legalMoves(eng, side, opts = {}) {
  const all = [];
  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const code = eng.get(c, r);
      if (code === 0 || pieceSide(code) !== side) continue;
      const ms = eng.generatePieceMoves(c, r);
      for (const m of ms) {
        if (opts.capturesOnly && m.captured === 0) continue;
        all.push(m);
      }
    }
  }
  const enemy = side === 'red' ? 'black' : 'red';
  const res = [];
  for (const m of all) {
    eng.makeMove(m);
    const k = eng.findKing(side);
    const ok = !k || !isSquareAttacked(eng, k[0], k[1], enemy);
    eng.undoMove();
    if (ok) res.push(m);
  }
  return res;
}

/** 走法排序：吃子优先（MVV-LVA），其次按目标格位置分 */
function orderMoves(eng, moves) {
  for (const m of moves) {
    let s = 0;
    if (m.captured !== 0) s = MATERIAL[pieceType(m.captured)] * 12 - MATERIAL[pieceType(m.piece)];
    m._order = s;
  }
  moves.sort((a, b) => b._order - a._order);
}

// ---------- 静态评估 ----------
function evaluate(eng) {
  let score = 0; // 红方为正
  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const code = eng.get(c, r);
      if (code === 0) continue;
      const side = pieceSide(code);
      const type = pieceType(code);
      const sign = side === 'red' ? 1 : -1;
      let v = MATERIAL[type] || 0;
      const pst = PST_BY_TYPE[type];
      if (pst) {
        const rr = side === 'red' ? r : BOARD_ROWS - 1 - r;
        v += pst[rr][c];
      }
      // 机动性
      if (MOBILITY_W[type]) {
        v += eng.generatePieceMoves(c, r).length * MOBILITY_W[type];
      }
      score += sign * v;
    }
  }
  return score;
}

// ---------- 搜索 ----------
let nodes = 0;
let deadline = 0;
let maxExtend = 0;

function inQuiescence(eng, alpha, beta, color, ply, qd) {
  nodes++;
  const side = color === 1 ? 'red' : 'black';
  const stand = evaluate(eng) * color;
  if (qd >= 6) return stand;
  if (stand >= beta) return stand;
  if (stand > alpha) alpha = stand;
  const moves = legalMoves(eng, side, { capturesOnly: true });
  orderMoves(eng, moves);
  for (const m of moves) {
    eng.makeMove(m);
    const opp = color === 1 ? 'black' : 'red';
    let s;
    if (eng.findKing(opp) === null) s = MATE - ply;
    else s = -inQuiescence(eng, -beta, -alpha, -color, ply + 1, qd + 1);
    eng.undoMove();
    if (s >= beta) return s;
    if (s > alpha) alpha = s;
  }
  return alpha;
}

function negamax(eng, depth, alpha, beta, color, ply) {
  nodes++;
  const side = color === 1 ? 'red' : 'black';
  const moves = legalMoves(eng, side);
  if (moves.length === 0) return -(MATE - ply); // 将死/困毙（象棋中均为负）
  if (ply > 0 && depth <= 0) return inQuiescence(eng, alpha, beta, color, ply, 0);
  if (depth <= 0) return evaluate(eng) * color;
  // 软限时：深度较大时若超时直接返回静态评估
  if (performance.now() > deadline && depth >= 2) return evaluate(eng) * color;
  // 将军延伸
  let d = depth;
  if (ply < maxExtend && isInCheckFast(eng, side)) d = depth + 1;
  // 和棋判定（重复局面/无进展/子力不足）
  const draw = eng._checkDrawRules();
  if (draw) return 0;
  orderMoves(eng, moves);
  let best = -INF;
  for (const m of moves) {
    eng.makeMove(m);
    const opp = color === 1 ? 'black' : 'red';
    let s;
    if (eng.findKing(opp) === null) s = MATE - ply - 1; // 飞将擒王
    else s = -negamax(eng, d - 1, -beta, -alpha, -color, ply + 1);
    eng.undoMove();
    if (s > best) best = s;
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }
  return best;
}

/**
 * 迭代加深找最佳走法
 * @returns {{move, score, depth, nodes}} score 为行棋方视角厘兵值
 */
export function findBestMove(eng, opts = {}) {
  const { timeMs = 500, maxDepth = 4, variety = true } = opts;
  deadline = performance.now() + timeMs;
  maxExtend = 6;
  const side = eng.turn;
  const color = side === 'red' ? 1 : -1;
  const moves = legalMoves(eng, side);
  if (moves.length === 0) return { move: null, score: -(MATE), depth: 0, nodes: 0 };
  orderMoves(eng, moves);
  let bestMove = moves[0];
  let bestScore = -INF;
  let bestDepth = 0;
  let totalNodes = 0;
  let rootScores = null;
  for (let d = 1; d <= maxDepth; d++) {
    nodes = 0;
    let alpha = -INF, beta = INF;
    let iterBest = null, iterScore = -INF, completed = true;
    const scores = [];
    for (const m of moves) {
      eng.makeMove(m);
      const opp = side === 'red' ? 'black' : 'red';
      let s;
      if (eng.findKing(opp) === null) s = MATE - d;
      else s = -negamax(eng, d - 1, -beta, -alpha, -color, 1);
      eng.undoMove();
      scores.push({ m, s });
      if (s > iterScore) { iterScore = s; iterBest = m; }
      if (s > alpha) alpha = s;
      if (performance.now() > deadline) { completed = false; break; }
    }
    totalNodes += nodes;
    if (completed) {
      bestMove = iterBest; bestScore = iterScore; bestDepth = d;
      rootScores = scores;
      // 按本轮分数重排根走法，加速下轮剪枝
      moves.sort((a, b) => {
        const sa = scores.find(x => x.m === a)?.s || -INF;
        const sb = scores.find(x => x.m === b)?.s || -INF;
        return sb - sa;
      });
    } else break;
  }
  // 微扰：在最优分数附近的走法中随机挑选，避免自对弈无限重复
  if (variety && rootScores && rootScores.length > 1) {
    const window = Math.max(15, Math.abs(bestScore) * 0.004);
    const candidates = rootScores.filter(x => x.s >= bestScore - window);
    const pick = candidates[Math.floor(Math.random() * candidates.length)];
    bestMove = pick.m;
  }
  return { move: bestMove, score: bestScore, depth: bestDepth, nodes: totalNodes };
}

/**
 * 根节点候选走法（供"局势分析"面板使用，浅层搜索）
 * @returns [{move, score}] score 为行棋方视角厘兵值，按优劣排序
 */
export function findBestMoves(eng, n = 5) {
  const side = eng.turn;
  const color = side === 'red' ? 1 : -1;
  const moves = legalMoves(eng, side);
  orderMoves(eng, moves);
  const results = [];
  const d = Math.min(2, Math.max(1, moves.length > 30 ? 1 : 2));
  deadline = performance.now() + 400;
  maxExtend = 4;
  for (const m of moves) {
    eng.makeMove(m);
    const opp = side === 'red' ? 'black' : 'red';
    let s;
    if (eng.findKing(opp) === null) s = MATE;
    else s = -negamax(eng, d - 1, -INF, INF, -color, 1);
    eng.undoMove();
    results.push({ move: m, score: s });
  }
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, n);
}

/** 局面评估报告（红方为正，厘兵） */
export function positionReport(eng) {
  const report = { material: { red: 0, black: 0 }, pst: { red: 0, black: 0 }, mobility: { red: 0, black: 0 }, score: 0, phase: '' };
  let heavy = 0, totalPieces = 0;
  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const code = eng.get(c, r);
      if (code === 0) continue;
      totalPieces++;
      const side = pieceSide(code);
      const type = pieceType(code);
      const v = MATERIAL[type] || 0;
      report.material[side] += v;
      const pst = PST_BY_TYPE[type];
      if (pst) {
        const rr = side === 'red' ? r : BOARD_ROWS - 1 - r;
        report.pst[side] += pst[rr][c];
      }
      if (MOBILITY_W[type]) {
        const mv = eng.generatePieceMoves(c, r).length * MOBILITY_W[type];
        report.mobility[side] += mv;
      }
      if (type === 5 || type === 6) heavy++;
    }
  }
  report.score = (report.material.red - report.material.black) +
                 (report.pst.red - report.pst.black) +
                 (report.mobility.red - report.mobility.black);
  // 阶段判定
  if (heavy >= 6) report.phase = '开局';
  else if (totalPieces <= 12 && heavy <= 2) report.phase = '残局';
  else report.phase = '中局';
  return report;
}

/** 统计某方被攻击的棋子（用于"威胁"分析） */
export function piecesUnderAttack(eng, side) {
  const enemy = side === 'red' ? 'black' : 'red';
  const list = [];
  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const code = eng.get(c, r);
      if (code === 0 || pieceSide(code) !== side) continue;
      if (isSquareAttacked(eng, c, r, enemy)) {
        list.push({
          code, c, r,
          type: pieceType(code),
          defended: isSquareAttacked(eng, c, r, side),
          value: MATERIAL[pieceType(code)] || 0,
        });
      }
    }
  }
  list.sort((a, b) => b.value - a.value);
  return list;
}

export const AI = { findBestMove, findBestMoves, positionReport, piecesUnderAttack, isInCheckFast, isSquareAttacked, evaluate, MATERIAL };
export default AI;