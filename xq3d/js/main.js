/* ============================================================
 * 主流程 —— 3D 象棋探索
 * 玩家附身于随机红/黑一方的 AI 角色，第一人称自由探索；
 * 两位 AI 自动对弈；按 F 可捕获棋盘快照并获取局势分析。
 * ============================================================ */
import { XiangqiEngine } from './engine.js';
import { GameScene } from './scene.js';
import { UI } from './ui.js';
import { analyzePosition } from './analysis.js';
import { findBestMove, isInCheckFast } from './ai.js';
import { moveToNotation } from './notation.js';

const engine = new XiangqiEngine();
const scene = new GameScene(document.getElementById('game-canvas'), engine);
const ui = new UI();

// ---------- 对局状态机 ----------
// 'waiting' 开赛倒计时 | 'thinking' AI 思考前奏 | 'searching' 正在搜索 | 'animating' 走子动画 | 'over' 终局
let state = 'waiting';
let stateTimer = 0;
let lastNotation = '';
let overTimer = 0;

function plyCount() { return engine.moveHistory.length; }

function refreshHud(thinking) {
  ui.updateGameStatus({ turn: engine.turn, ply: plyCount(), lastMove: lastNotation, thinking });
  ui.updateScoreBar(engine.materialValue('red') / 1000, engine.materialValue('black') / 1000);
}

function beginTurn() {
  if (engine.checkGameEnd()) { endGame(); return; }
  state = 'thinking';
  stateTimer = 0.45 + Math.random() * 0.45; // 思考前奏（模拟 AI 感知）
  refreshHud(true);
}

function doSearch() {
  state = 'searching';
  ui.setHint('正在计算…');
  // 异步让"思考中"先渲染一帧
  setTimeout(() => {
    const res = findBestMove(engine, { timeMs: 480, maxDepth: 4 });
    ui.setHint('WASD 移动 · 鼠标观察 · Shift 疾走 · <b>F</b> 棋盘快照分析 · 点击画面锁定鼠标');
    if (!res.move) { if (!engine.checkGameEnd()) beginTurn(); else endGame(); return; }
    const notation = moveToNotation(engine, res.move);
    const captured = res.move.captured;
    engine.makeMove(res.move);
    lastNotation = notation;
    // 将军检测
    const mover = engine.turn === 'red' ? 'black' : 'red';
    const inCheck = isInCheckFast(engine, mover);
    scene.setCheckSide(inCheck ? mover : null);
    scene.animateMove(res.move);
    // 音效
    if (inCheck) ui.sfx.check();
    else if (captured !== 0) ui.sfx.capture();
    else ui.sfx.move();
    ui.toast(`第 ${plyCount()} 手 · ${notation}`, captured !== 0 ? '#ff9b3d' : '#ffd75e');
    if (ui.snapshotOpen) refreshSnapshot();
    state = 'animating';
  }, 50);
}

function endGame() {
  state = 'over';
  overTimer = 26;
  const winner = engine.winner;
  const reasonMap = {
    checkmate: '将死',
    stalemate: '困毙',
    draw_repetition: '重复局面',
    draw_50move: '无进展回合',
    draw_material: '子力不足',
  };
  const reason = reasonMap[engine.endReason] || engine.endReason;
  if (winner) {
    ui.showResult({ winner, reason });
    if (winner === scene.mySide) ui.sfx.win(); else ui.sfx.lose();
  } else {
    ui.showResult({ winner: 'draw', reason });
  }
  refreshHud(false);
}

// ---------- 快照 ----------
function refreshSnapshot() {
  const a = analyzePosition(engine, scene.mySide);
  ui.openSnapshot(a, engine);
}
function toggleSnapshot() {
  if (ui.snapshotOpen) { ui.closeSnapshot(); return; }
  ui.sfx.capture();
  ui.toast('已捕获棋盘快照，正在分析局势…', '#6fd08c');
  setTimeout(refreshSnapshot, 80);
}

// ---------- 重新附身 / 重开 ----------
function rePossess() {
  scene.spawn(Math.random() < 0.5 ? 'red' : 'black');
  ui.setAffiliation(scene.mySide);
  ui.toast(`你已附身于${scene.mySide === 'red' ? '红方' : '黑方'} AI`);
}
function newGame() {
  engine.reset();
  scene.lastMove = null;
  scene.setCheckSide(null);
  scene._syncAllPieces();
  ui.hideResult();
  ui.closeSnapshot();
  lastNotation = '';
  beginTurn();
}

// ---------- 事件绑定 ----------
scene.callbacks.onMoveDone = () => {
  scene._updateCheckMarker();
  refreshHud(false);
  beginTurn();
};
scene.callbacks.onLookChange = locked => {
  if (!locked && ui.snapshotOpen) { /* 保持面板 */ }
};
scene.onCanvasClick = () => ui.sfx._ensure();
ui.onRestart = () => newGame();
document.getElementById('btn-possess') && (document.getElementById('btn-possess').onclick = () => rePossess());
document.getElementById('btn-restart') && (document.getElementById('btn-restart').onclick = () => newGame());
document.getElementById('btn-snap') && (document.getElementById('btn-snap').onclick = () => toggleSnapshot());
document.addEventListener('keydown', e => {
  if (e.code === 'KeyF') toggleSnapshot();
  if (e.code === 'Escape' && ui.snapshotOpen) ui.closeSnapshot();
});

// ---------- 主循环 ----------
let last = performance.now();

function loop(now) {
  requestAnimationFrame(loop);
  let dt = Math.min(0.05, (now - last) / 1000);
  last = now;

  // 状态机推进
  if (state === 'waiting') {
    stateTimer -= dt;
    if (stateTimer <= 0) beginTurn();
  } else if (state === 'thinking') {
    stateTimer -= dt;
    if (stateTimer <= 0) doSearch();
  } else if (state === 'over') {
    overTimer -= dt;
    if (overTimer <= 0) newGame();
  }

  scene.update(dt);
}

// ---------- 启动 ----------
scene.init();
scene.spawn();            // 随机立于红黑一方
ui.setAffiliation(scene.mySide);
ui.setHint('你已附身于 AI 角色，对局即将开始');
state = 'waiting';
stateTimer = 1.0;
refreshHud(false);
requestAnimationFrame(loop);
