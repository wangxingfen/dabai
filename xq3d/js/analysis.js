/* ============================================================
 * 棋盘快照 · 局势分析
 * 模拟"附身 AI 角色"的感知能力：扫描棋盘 → 量化评估 → 生成自然语言解读
 * ============================================================ */
import { PIECE_NAMES, pieceType } from './engine.js';
import { moveToNotation, fileNumber, sideName } from './notation.js';
import { findBestMoves, positionReport, piecesUnderAttack, isInCheckFast } from './ai.js';

const CN = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

function pawns(cp) { return (cp / 1000).toFixed(1); }

/** 生成一条自然语言着法解读 */
function describeMove(eng, move) {
  const parts = [];
  const char = PIECE_NAMES[move.piece];
  if (move.captured !== 0) parts.push(`吃掉对方${PIECE_NAMES[move.captured]}`);
  else parts.push(`出动${char}`);
  // 是否将军
  eng.makeMove(move);
  const opp = eng.turn === 'red' ? 'black' : 'red';
  const givesCheck = isInCheckFast(eng, opp);
  eng.undoMove();
  if (givesCheck) parts.push('直接将军');
  if (move.captured !== 0) parts.push('兑子得利');
  return parts.join('，') + '，巩固子力位置';
}

/**
 * 生成完整局势分析
 * @param eng 引擎
 * @param mySide 附身 AI 所在方 ('red'|'black')
 */
export function analyzePosition(eng, mySide) {
  const turn = eng.turn;
  const report = positionReport(eng);
  const checkSide = isInCheckFast(eng, 'red') ? 'red' : (isInCheckFast(eng, 'black') ? 'black' : null);

  const top = findBestMoves(eng, 3);
  const bestMove = top[0] || null;
  const candidates = top.map(m => ({
    notation: moveToNotation(eng, m.move),
    score: m.score,
    rel: m.score - (bestMove ? bestMove.score : 0),
    desc: describeMove(eng, m.move),
  }));

  // 局势领先
  let advantage = null;
  const s = report.score;
  if (Math.abs(s) < 60) advantage = { side: null, amount: 0 };
  else advantage = { side: s > 0 ? 'red' : 'black', amount: Math.abs(s) / 1000 };

  // 威胁
  const threats = [];
  const tu = piecesUnderAttack(eng, turn);
  const tm = piecesUnderAttack(eng, mySide);
  for (const t of tu.slice(0, 3)) {
    threats.push({
      side: turn,
      text: `${sideName(turn)}${PIECE_NAMES[t.code]}（${CN[fileNumber(turn, t.c)]}线）正受攻击${t.defended ? '，但有子保护' : '，且无子保护'}，价值约 ${(t.value / 1000).toFixed(1)} 子`,
    });
  }
  for (const t of tm.slice(0, 2)) {
    threats.push({
      side: mySide,
      text: `【你方】${sideName(mySide)}${PIECE_NAMES[t.code]}（${CN[fileNumber(mySide, t.c)]}线）受威胁${t.defended ? '，有子保护' : '，无子保护'}`,
    });
  }

  // 我的视角总结
  let myView;
  if (advantage.side === null) {
    myView = `你附身于${sideName(mySide)} AI，当前局势接近均势，胜负取决于接下来的关键着法。`;
  } else if (advantage.side === mySide) {
    myView = `你附身于${sideName(mySide)} AI，己方占优约 ${pawns(advantage.amount * 1000)} 子，保持压制，注意别给对手反击机会。`;
  } else {
    myView = `你附身于${sideName(mySide)} AI，己方暂时落后约 ${pawns(advantage.amount * 1000)} 子，宜稳守待变、寻求牵制。`;
  }

  // 策略建议
  let strategy;
  if (checkSide) {
    strategy = checkSide === mySide ? '你方正被将军！优先应将，再图反击。' : '对方正被将军！把握时机扩大战果。';
  } else if (report.phase === '开局') {
    strategy = '开局阶段：尽快出动车马炮等大子，抢占河界与中心要点。';
  } else if (report.phase === '残局') {
    strategy = '残局阶段：注意将帅安全与兵卒的过河推进，先手往往决定胜负。';
  } else {
    strategy = advantage.side === mySide
      ? '中局占优：以多打少，先兑换弱子、简化局面。'
      : '中局落后：寻找牵制与反击点，避免无谓兑子。';
  }

  return {
    ply: eng.moveCount !== undefined ? eng.moveCount : (eng.moveHistory ? eng.moveHistory.length : 0),
    turn,
    checkSide,
    phase: report.phase,
    score: s,
    advantage,
    material: {
      red: report.material.red / 1000,
      black: report.material.black / 1000,
    },
    bestMove: bestMove ? { notation: moveToNotation(eng, bestMove.move), score: bestMove.score, desc: describeMove(eng, bestMove.move) } : null,
    candidates,
    threats,
    myView,
    strategy,
    mySide,
  };
}
