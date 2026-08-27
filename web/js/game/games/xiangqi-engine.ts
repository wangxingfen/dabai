/* ============================================================
 * 象棋引擎 —— 纯逻辑层（无渲染依赖）
 *
 * 棋盘坐标系：board[row][col]，row 0=黑方底线（顶部），row 9=红方底线（底部）
 * 棋子编码：0=空 | 1-7=红方(帅仕相马车炮兵) | 8-14=黑方(将士象马车炮卒)
 * 走法表示：{ from:[c,r], to:[c,r], piece, captured }
 * ============================================================ */

// ---------- 棋子常量 ----------
export const PIECE = {
  EMPTY: 0,
  R_KING: 1, R_ADVISOR: 2, R_ELEPHANT: 3, R_KNIGHT: 4, R_ROOK: 5, R_CANNON: 6, R_PAWN: 7,
  B_KING: 8, B_ADVISOR: 9, B_ELEPHANT: 10, B_KNIGHT: 11, B_ROOK: 12, B_CANNON: 13, B_PAWN: 14,
};

export const PIECE_NAMES = {
  1: '帅', 2: '仕', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
  8: '将', 9: '士', 10: '象', 11: '马', 12: '车', 13: '炮', 14: '卒',
};

export const PIECE_VALUES = {
  1: 100, 2: 2, 3: 2, 4: 4.5, 5: 9, 6: 4.5, 7: 1,
  8: 100, 9: 2, 10: 2, 11: 4.5, 12: 9, 13: 4.5, 14: 1,
};

export const BOARD_COLS = 9;
export const BOARD_ROWS = 10;

/** 获取棋子类型（1-7），红黑统一 */
export function pieceType(code) {
  if (code === 0) return 0;
  return code > 7 ? code - 7 : code;
}

/** 获取棋子阵营：'red' | 'black' | null */
export function pieceSide(code) {
  if (code === 0) return null;
  return code > 7 ? 'black' : 'red';
}

/** 是否红方棋子 */
export function isRed(code) { return code >= 1 && code <= 7; }
/** 是否黑方棋子 */
export function isBlack(code) { return code >= 8 && code <= 14; }

// ---------- 引擎类 ----------
export class XiangqiEngine {
  constructor() {
    this.reset();
  }

  /** 重置到初始局面 */
  reset() {
    // 10行×9列，row 0=黑方底线
    this.board = [
      [12, 11, 10, 9, 8, 9, 10, 11, 12],   // row 0: 黑方底线 车马象士将士象马车
      [0, 0, 0, 0, 0, 0, 0, 0, 0],          // row 1
      [0, 13, 0, 0, 0, 0, 0, 13, 0],        // row 2: 黑炮
      [14, 0, 14, 0, 14, 0, 14, 0, 14],     // row 3: 黑卒
      [0, 0, 0, 0, 0, 0, 0, 0, 0],          // row 4: 河界边
      [0, 0, 0, 0, 0, 0, 0, 0, 0],          // row 5: 河界边
      [7, 0, 7, 0, 7, 0, 7, 0, 7],          // row 6: 红兵
      [0, 6, 0, 0, 0, 0, 0, 6, 0],          // row 7: 红炮
      [0, 0, 0, 0, 0, 0, 0, 0, 0],          // row 8
      [5, 4, 3, 2, 1, 2, 3, 4, 5],          // row 9: 红方底线 车马相仕帅仕相马车
    ];
    this.turn = 'red';       // 红方先行
    this.moveHistory = [];
    this.gameOver = false;
    this.winner = null;
    this.endReason = '';
    // 和棋追踪：总手数 / 无吃子·无兵卒移动的手数 / 局面重复计数
    this.plyCount = 0;
    this.noProgressPlies = 0;
    this._positionCounts = new Map();
  }

  /** 当前局面的唯一键（棋盘 + 行棋方），用于重复局面检测 */
  _positionKey() {
    let key = this.turn === 'red' ? 'r' : 'b';
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        key += this.board[r][c].toString(16);
      }
    }
    return key;
  }

  /** 获取棋盘上的棋子 */
  get(c, r) {
    if (r < 0 || r >= BOARD_ROWS || c < 0 || c >= BOARD_COLS) return 0;
    return this.board[r][c];
  }

  /** 设置棋盘位置 */
  set(c, r, code) {
    if (r >= 0 && r < BOARD_ROWS && c >= 0 && c < BOARD_COLS) {
      this.board[r][c] = code;
    }
  }

  /** 克隆当前引擎状态 */
  clone() {
    const e = new XiangqiEngine();
    e.board = this.board.map(row => [...row]);
    e.turn = this.turn;
    e.moveHistory = [...this.moveHistory];
    e.gameOver = this.gameOver;
    e.winner = this.winner;
    e.endReason = this.endReason;
    return e;
  }

  // ==================== 区域判定 ====================

  /** 是否在九宫格内 */
  inPalace(c, r, side) {
    if (c < 3 || c > 5) return false;
    if (side === 'red') return r >= 7 && r <= 9;
    return r >= 0 && r <= 2;
  }

  /** 是否在本方半场（未过河） */
  ownSide(c, r, side) {
    if (side === 'red') return r >= 5;
    return r <= 4;
  }

  /** 是否已过河 */
  crossedRiver(c, r, side) {
    if (side === 'red') return r <= 4;
    return r >= 5;
  }

  // ==================== 走法生成 ====================

  /** 生成某方所有合法走法 */
  generateAllMoves(side) {
    const moves = [];
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        const code = this.board[r][c];
        if (code === 0) continue;
        if (pieceSide(code) !== side) continue;
        moves.push(...this.generatePieceMoves(c, r));
      }
    }
    // 过滤掉走完后己方被将军的步
    return moves.filter(m => !this._movesIntoCheck(m, side));
  }

  /** 生成单个棋子的原始走法（未过滤将军） */
  generatePieceMoves(c, r) {
    const code = this.board[r][c];
    if (code === 0) return [];
    const type = pieceType(code);
    const side = pieceSide(code);
    switch (type) {
      case 1: return this._kingMoves(c, r, side);
      case 2: return this._advisorMoves(c, r, side);
      case 3: return this._elephantMoves(c, r, side);
      case 4: return this._knightMoves(c, r, side);
      case 5: return this._rookMoves(c, r, side);
      case 6: return this._cannonMoves(c, r, side);
      case 7: return this._pawnMoves(c, r, side);
      default: return [];
    }
  }

  /** 帅/将走法：九宫内一步直行 + 飞将吃帅 */
  _kingMoves(c, r, side) {
    const moves = [];
    const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    for (const [dc, dr] of dirs) {
      const nc = c + dc, nr = r + dr;
      if (!this.inPalace(nc, nr, side)) continue;
      const target = this.get(nc, nr);
      if (target === 0 || pieceSide(target) !== side) {
        moves.push(this._mk(c, r, nc, nr));
      }
    }
    // 飞将：同列对面无棋子时可吃对方将/帅
    const enemyKingType = side === 'red' ? 8 : 1;
    const dir = side === 'red' ? -1 : 1; // 红将向上找黑将，黑将向下找红将
    for (let nr = r + dir; nr >= 0 && nr < BOARD_ROWS; nr += dir) {
      const t = this.get(c, nr);
      if (t !== 0) {
        if (t === enemyKingType) {
          moves.push(this._mk(c, r, c, nr));
        }
        break;
      }
    }
    return moves;
  }

  /** 士/仕走法：九宫内一步斜行 */
  _advisorMoves(c, r, side) {
    const moves = [];
    const dirs = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
    for (const [dc, dr] of dirs) {
      const nc = c + dc, nr = r + dr;
      if (!this.inPalace(nc, nr, side)) continue;
      const target = this.get(nc, nr);
      if (target === 0 || pieceSide(target) !== side) {
        moves.push(this._mk(c, r, nc, nr));
      }
    }
    return moves;
  }

  /** 象/相走法：田字格，不过河，塞象眼不可走 */
  _elephantMoves(c, r, side) {
    const moves = [];
    const dirs = [[2, 2], [2, -2], [-2, 2], [-2, -2]];
    for (const [dc, dr] of dirs) {
      const nc = c + dc, nr = r + dr;
      if (nc < 0 || nc >= BOARD_COLS || nr < 0 || nr >= BOARD_ROWS) continue;
      if (!this.ownSide(nc, nr, side)) continue; // 不可过河
      const eyeC = c + dc / 2, eyeR = r + dr / 2;
      if (this.get(eyeC, eyeR) !== 0) continue; // 塞象眼
      const target = this.get(nc, nr);
      if (target === 0 || pieceSide(target) !== side) {
        moves.push(this._mk(c, r, nc, nr));
      }
    }
    return moves;
  }

  /** 马走法：日字格，蹩马腿不可走 */
  _knightMoves(c, r, side) {
    const moves = [];
    // [dc, dr, legC, legR] - 腿的位置
    const steps = [
      [1, 2, 0, 1], [-1, 2, 0, 1],    // 向下跳
      [1, -2, 0, -1], [-1, -2, 0, -1], // 向上跳
      [2, 1, 1, 0], [2, -1, 1, 0],     // 向右跳
      [-2, 1, -1, 0], [-2, -1, -1, 0], // 向左跳
    ];
    for (const [dc, dr, legC, legR] of steps) {
      const nc = c + dc, nr = r + dr;
      if (nc < 0 || nc >= BOARD_COLS || nr < 0 || nr >= BOARD_ROWS) continue;
      if (this.get(c + legC, r + legR) !== 0) continue; // 蹩马腿
      const target = this.get(nc, nr);
      if (target === 0 || pieceSide(target) !== side) {
        moves.push(this._mk(c, r, nc, nr));
      }
    }
    return moves;
  }

  /** 车走法：直线无阻挡 */
  _rookMoves(c, r, side) {
    const moves = [];
    const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    for (const [dc, dr] of dirs) {
      let nc = c + dc, nr = r + dr;
      while (nc >= 0 && nc < BOARD_COLS && nr >= 0 && nr < BOARD_ROWS) {
        const target = this.get(nc, nr);
        if (target === 0) {
          moves.push(this._mk(c, r, nc, nr));
        } else {
          if (pieceSide(target) !== side) {
            moves.push(this._mk(c, r, nc, nr));
          }
          break;
        }
        nc += dc; nr += dr;
      }
    }
    return moves;
  }

  /** 炮走法：移动同车，吃子需翻一个炮架 */
  _cannonMoves(c, r, side) {
    const moves = [];
    const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    for (const [dc, dr] of dirs) {
      let nc = c + dc, nr = r + dr;
      let jumped = false; // 是否已翻过炮架
      while (nc >= 0 && nc < BOARD_COLS && nr >= 0 && nr < BOARD_ROWS) {
        const target = this.get(nc, nr);
        if (!jumped) {
          if (target === 0) {
            moves.push(this._mk(c, r, nc, nr)); // 空格可移动
          } else {
            jumped = true; // 遇到炮架
          }
        } else {
          if (target !== 0) {
            if (pieceSide(target) !== side) {
              moves.push(this._mk(c, r, nc, nr)); // 翻架吃子
            }
            break;
          }
        }
        nc += dc; nr += dr;
      }
    }
    return moves;
  }

  /** 兵/卒走法：过河前只能前进，过河后可左右 */
  _pawnMoves(c, r, side) {
    const moves = [];
    const forward = side === 'red' ? -1 : 1; // 红方向上(row减)，黑方向下(row增)
    // 前进
    const nr = r + forward;
    if (nr >= 0 && nr < BOARD_ROWS) {
      const target = this.get(c, nr);
      if (target === 0 || pieceSide(target) !== side) {
        moves.push(this._mk(c, r, c, nr));
      }
    }
    // 过河后可横走
    if (this.crossedRiver(c, r, side)) {
      for (const dc of [-1, 1]) {
        const nc = c + dc;
        if (nc < 0 || nc >= BOARD_COLS) continue;
        const target = this.get(nc, r);
        if (target === 0 || pieceSide(target) !== side) {
          moves.push(this._mk(c, r, nc, r));
        }
      }
    }
    return moves;
  }

  // ==================== 走法执行与判定 ====================

  /** 构造走法对象 */
  _mk(fc, fr, tc, tr) {
    return {
      from: [fc, fr], to: [tc, tr],
      piece: this.get(fc, fr),
      captured: this.get(tc, tr),
    };
  }

  /** 执行走法（不检查合法性，调用方负责） */
  makeMove(move) {
    const [fc, fr] = move.from;
    const [tc, tr] = move.to;
    move.piece = this.board[fr][fc];
    move.captured = this.board[tr][tc];
    // 记录撤销前快照（供 undoMove 恢复和棋追踪）
    move._ply = this.plyCount;
    move._noProgress = this.noProgressPlies;
    this.board[tr][tc] = this.board[fr][fc];
    this.board[fr][fc] = 0;
    this.moveHistory.push(move);
    this.turn = this.turn === 'red' ? 'black' : 'red';
    // 和棋追踪
    this.plyCount++;
    const key = this._positionKey();
    move._posKey = key;
    this._positionCounts.set(key, (this._positionCounts.get(key) || 0) + 1);
    if (move.captured || pieceType(move.piece) === 7) this.noProgressPlies = 0;
    else this.noProgressPlies++;
  }

  /** 撤销最后一步 */
  undoMove() {
    const move = this.moveHistory.pop();
    if (!move) return null;
    const [fc, fr] = move.from;
    const [tc, tr] = move.to;
    this.board[fr][fc] = move.piece;
    this.board[tr][tc] = move.captured;
    this.turn = this.turn === 'red' ? 'black' : 'red';
    // 恢复和棋追踪
    this.plyCount = move._ply;
    this.noProgressPlies = move._noProgress;
    if (move._posKey) {
      const n = (this._positionCounts.get(move._posKey) || 0) - 1;
      if (n <= 0) this._positionCounts.delete(move._posKey);
      else this._positionCounts.set(move._posKey, n);
    }
    return move;
  }

  /** 查找将/帅位置 */
  findKing(side) {
    const kingCode = side === 'red' ? PIECE.R_KING : PIECE.B_KING;
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        if (this.board[r][c] === kingCode) return [c, r];
      }
    }
    return null;
  }

  /** 判断某方是否被将军 */
  isInCheck(side) {
    const kingPos = this.findKing(side);
    if (!kingPos) return true; // 将不存在视为被将
    const enemy = side === 'red' ? 'black' : 'red';
    // 检查所有敌方棋子能否吃到将
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        const code = this.board[r][c];
        if (code === 0 || pieceSide(code) !== enemy) continue;
        const moves = this.generatePieceMoves(c, r);
        for (const m of moves) {
          if (m.to[0] === kingPos[0] && m.to[1] === kingPos[1]) return true;
        }
      }
    }
    return false;
  }

  /** 判断走完后己方是否被将军 */
  _movesIntoCheck(move, side) {
    this.makeMove(move);
    const checked = this.isInCheck(side);
    this.undoMove();
    return checked;
  }

  /** 判断某方是否被将死 */
  isCheckmate(side) {
    const moves = this.generateAllMoves(side);
    return moves.length === 0;
  }

  /** 判断是否困毙（无棋可走但未被将军，在象棋中判负） */
  isStalemate(side) {
    if (this.isInCheck(side)) return false;
    return this.generateAllMoves(side).length === 0;
  }

  /** 检查游戏是否结束，设置 winner 和 endReason */
  checkGameEnd() {
    if (this.gameOver) return true;
    const side = this.turn;
    const moves = this.generateAllMoves(side);
    if (moves.length === 0) {
      this.gameOver = true;
      if (this.isInCheck(side)) {
        this.winner = side === 'red' ? 'black' : 'red';
        this.endReason = 'checkmate';
      } else {
        this.winner = side === 'red' ? 'black' : 'red';
        this.endReason = 'stalemate';
      }
      return true;
    }
    // 和棋判定（winner=null 表示平局）
    const draw = this._checkDrawRules();
    if (draw) {
      this.gameOver = true;
      this.winner = null;
      this.endReason = draw;
      return true;
    }
    return false;
  }

  /** 和棋规则检测：返回和棋类型或 null */
  _checkDrawRules() {
    // 1. 无进展回合：双方合计 120 手（60 回合）无吃子且无兵卒移动 → 和棋
    if (this.noProgressPlies >= 120) return 'draw_50move';
    // 2. 重复局面：同一局面（含行棋方）出现 3 次 → 和棋
    const key = this._positionKey();
    if ((this._positionCounts.get(key) || 0) >= 3) return 'draw_repetition';
    // 3. 子力不足：双方均无车炮卒，且马总数 ≤ 1 → 无法形成杀棋 → 和棋
    if (this._insufficientMaterial()) return 'draw_material';
    return null;
  }

  /** 子力不足判定（保守版，仅判定绝对无杀棋的情况） */
  _insufficientMaterial() {
    let horseTotal = 0;
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        const code = this.board[r][c];
        if (!code) continue;
        const t = pieceType(code);
        if (t === 5 || t === 6 || t === 7) return false; // 车/炮/兵卒 → 仍有杀棋可能
        if (t === 4) horseTotal++; // 马
      }
    }
    // 无车炮卒时，仅当双方合计至多一马 → 必和（单马/士象残局无法逼杀）
    return horseTotal <= 1;
  }

  // ==================== 状态编码 ====================

  /** 棋盘序列化为一维数组 */
  boardFlat() {
    const arr = new Array(BOARD_COLS * BOARD_ROWS);
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        arr[r * BOARD_COLS + c] = this.board[r][c];
      }
    }
    return arr;
  }

  /** 计算材料价值 */
  materialValue(side) {
    let total = 0;
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        const code = this.board[r][c];
        if (code !== 0 && pieceSide(code) === side) {
          total += PIECE_VALUES[code] || 0;
        }
      }
    }
    return total;
  }

  /** 统计某方棋子数 */
  pieceCount(side) {
    let count = 0;
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        const code = this.board[r][c];
        if (code !== 0 && pieceSide(code) === side) count++;
      }
    }
    return count;
  }

  /** 获取所有棋子位置列表 */
  allPieces() {
    const list = [];
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        const code = this.board[r][c];
        if (code !== 0) list.push({ c, r, code, side: pieceSide(code), type: pieceType(code) });
      }
    }
    return list;
  }

  /** 走法转字符串记号（如 "炮二平五" 风格简化版） */
  moveToString(move) {
    const [fc, fr] = move.from;
    const [tc, tr] = move.to;
    const piece = PIECE_NAMES[move.piece] || '?';
    const action = move.captured ? '吃' : '走';
    return `${piece}(${fc},${fr})${action}(${tc},${tr})`;
  }

  declare _positionCounts: any;
  declare board: any;
  declare endReason: any;
  declare gameOver: any;
  declare moveHistory: any;
  declare noProgressPlies: any;
  declare plyCount: any;
  declare turn: any;
  declare winner: any;
}

export default XiangqiEngine;
