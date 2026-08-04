/* ============================================================
 * 象棋中文记谱 —— 生成 "炮二平五" / "马八进七" / "前车进一" 风格走法
 * 记谱以行棋方视角：红方从右往左编号 1-9，黑方从左往右编号 1-9
 * ============================================================ */
import { PIECE_NAMES, pieceType, pieceSide } from './engine.js';

const CN_NUM = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
const SIDE_NAMES = { red: '红方', black: '黑方' };

/** 该方视角下的纵线编号 */
export function fileNumber(side, col) {
  return side === 'red' ? 9 - col : col + 1;
}

function cn(n) {
  return CN_NUM[n] || String(n);
}

/** 相同类型的其它棋子（用于消歧义） */
function sameTypePieces(eng, type, side) {
  const list = [];
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 9; c++) {
      const code = eng.get(c, r);
      if (code !== 0 && pieceSide(code) === side && pieceType(code) === type) list.push({ c, r });
    }
  }
  return list;
}

/** 将完整走法转为中文记谱；需要传入走子前局面 */
export function moveToNotation(eng, move) {
  const side = pieceSide(move.piece);
  if (!side) return '';
  const type = pieceType(move.piece);
  const name = PIECE_NAMES[move.piece];
  const [fc, fr] = move.from;
  const [tc, tr] = move.to;

  const same = sameTypePieces(eng, type, side);
  let prefix = ''; // 前/后/中
  let fromLabel = cn(fileNumber(side, fc));
  if (same.length >= 2) {
    const onSameFile = same.filter(p => p.c === fc);
    if (onSameFile.length >= 2) {
      // 同线多子：按行进方向标 前/后（三兵同线用 前/中/后）
      const ordered = [...onSameFile].sort((a, b) => side === 'red' ? a.r - b.r : b.r - a.r);
      const idx = ordered.findIndex(p => p.c === fc && p.r === fr);
      prefix = idx === 0 ? '前' : (idx === ordered.length - 1 ? '后' : '中');
    }
  }

  const forward = side === 'red' ? -1 : 1;
  let dir, target;
  if (tr === fr) {
    dir = '平';
    target = cn(fileNumber(side, tc));
  } else {
    const advancing = (tr - fr) * forward > 0;
    dir = advancing ? '进' : '退';
    if (type === 4 || type === 2 || type === 3) {
      target = cn(fileNumber(side, tc)); // 马/相/仕 用落点纵线
    } else {
      target = cn(Math.abs(tr - fr));    // 车/炮/兵/帅 用步数
    }
  }
  const namePart = prefix ? prefix + name : name + fromLabel;
  const suffix = move.captured !== 0 ? ' 吃' + PIECE_NAMES[move.captured] : '';
  return `${namePart}${dir}${target}${suffix}`;
}

/** 局面文字化：每行 9 格，用于快照面板 */
export function boardToText(eng) {
  const lines = [];
  for (let r = 0; r < 10; r++) {
    const row = [];
    for (let c = 0; c < 9; c++) {
      const code = eng.get(c, r);
      row.push(code === 0 ? '·' : PIECE_NAMES[code]);
    }
    lines.push(row.join(' '));
  }
  return lines.join('\n');
}

/** 轻量 FEN 风格编码（用于快照面板展示） */
export function boardToFen(eng) {
  const rows = [];
  for (let r = 0; r < 10; r++) {
    let s = '', empty = 0;
    for (let c = 0; c < 9; c++) {
      const code = eng.get(c, r);
      if (code === 0) { empty++; continue; }
      if (empty) { s += empty; empty = 0; }
      s += PIECE_NAMES[code] === '马' ? (code > 7 ? 'h' : 'H') : PIECE_NAMES[code];
    }
    if (empty) s += empty;
    rows.push(s);
  }
  return `${rows.join('/')} ${eng.turn === 'red' ? 'r' : 'b'}`;
}

export function sideName(side) { return SIDE_NAMES[side] || side; }

export const NOTATION = { moveToNotation, boardToText, boardToFen, fileNumber, sideName };
export default NOTATION;
