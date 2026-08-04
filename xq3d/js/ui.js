/* ============================================================
 * UI 模块 —— HUD / 快照分析面板 / 迷你棋盘 / 结果弹层 / 音效
 * ============================================================ */
import { PIECE_NAMES, isRed } from './engine.js';

const SIDE_LABEL = { red: '红方', black: '黑方' };
const PIECE_CN = ['', '帅', '仕', '相', '马', '车', '炮', '兵', '将', '士', '象', '马', '车', '炮', '卒'];

export class UI {
  constructor() {
    this._build();
    this.sfx = new Sfx();
    this.snapshotOpen = false;
  }

  _el(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  _build() {
    const body = document.body;

    // ---- HUD 顶栏 ----
    this.hud = this._el('div', 'hud');
    // 左侧：对局状态
    this.statusBox = this._el('div', 'hud-status');
    this.turnEl = this._el('div', 'hud-turn', '红方行棋');
    this.moveEl = this._el('div', 'hud-move', '对局即将开始');
    this.thinkingEl = this._el('div', 'hud-thinking', 'AI 思考中…');
    this.thinkingEl.style.display = 'none';
    this.statusBox.append(this.turnEl, this.moveEl, this.thinkingEl);
    // 中间：子力对比条
    this.barBox = this._el('div', 'hud-bar');
    this.barLabel = this._el('div', 'hud-bar-label', '子力对比');
    this.barTrack = this._el('div', 'hud-bar-track');
    this.barRed = this._el('div', 'hud-bar-red', '红 9.0');
    this.barBlack = this._el('div', 'hud-bar-black', '9.0 黑');
    this.barTrack.append(this.barRed, this.barBlack);
    this.barBox.append(this.barLabel, this.barTrack);
    // 右侧：附身 + 按钮
    this.rightBox = this._el('div', 'hud-right');
    this.affilEl = this._el('div', 'hud-affil', '你附身于红方 AI');
    this.btnPossess = this._el('button', 'hud-btn', '重新附身');
    this.btnPossess.id = 'btn-possess';
    this.btnRestart = this._el('button', 'hud-btn', '重开对局');
    this.btnRestart.id = 'btn-restart';
    this.btnSnap = this._el('button', 'hud-btn primary', '快照分析 [F]');
    this.btnSnap.id = 'btn-snap';
    this.rightBox.append(this.affilEl, this.btnPossess, this.btnSnap, this.btnRestart);
    this.hud.append(this.statusBox, this.barBox, this.rightBox);
    body.appendChild(this.hud);

    // ---- 操作提示 ----
    this.hint = this._el('div', 'hint');
    this.hint.innerHTML = 'WASD 移动 · 鼠标观察 · Shift 疾走 · <b>F</b> 棋盘快照分析 · 点击画面锁定鼠标';
    body.appendChild(this.hint);

    // ---- 附身光晕（红/黑氛围） ----
    this.vignette = this._el('div', 'vignette');
    body.appendChild(this.vignette);

    // ---- Toast ----
    this.toastEl = this._el('div', 'toast');
    body.appendChild(this.toastEl);

    // ---- 快照面板 ----
    this.snap = this._el('div', 'snap');
    this.snap.innerHTML = `
      <div class="snap-head">
        <div class="snap-title">棋盘快照 · 局势分析</div>
        <button class="snap-close" id="snap-close">✕</button>
      </div>
      <canvas id="snap-board" width="288" height="324"></canvas>
      <div class="snap-meta" id="snap-meta"></div>
      <div class="snap-sec">
        <div class="snap-sec-title">最佳着法</div>
        <div class="snap-best" id="snap-best"></div>
      </div>
      <div class="snap-sec">
        <div class="snap-sec-title">候选着法</div>
        <div class="snap-cands" id="snap-cands"></div>
      </div>
      <div class="snap-sec">
        <div class="snap-sec-title">子力对比</div>
        <div class="snap-mat" id="snap-mat"></div>
      </div>
      <div class="snap-sec">
        <div class="snap-sec-title">威胁提示</div>
        <div class="snap-threats" id="snap-threats"></div>
      </div>
      <div class="snap-view" id="snap-view"></div>
      <div class="snap-foot">数据由附身 AI 感知引擎生成 · 按 F 重新扫描</div>
    `;
    body.appendChild(this.snap);
    this.snapBoard = this.snap.querySelector('#snap-board');
    this.snap.querySelector('#snap-close').onclick = () => this.closeSnapshot();

    // ---- 结果弹层 ----
    this.result = this._el('div', 'result');
    this.result.innerHTML = `
      <div class="result-card">
        <div class="result-title" id="result-title">红方胜</div>
        <div class="result-sub" id="result-sub">将死</div>
        <div class="result-btns">
          <button id="btn-again">再来一局</button>
        </div>
      </div>
    `;
    body.appendChild(this.result);
    this.result.querySelector('#btn-again').onclick = () => this.onRestart && this.onRestart();
  }

  // ==================== HUD ====================
  updateGameStatus({ turn, ply, lastMove, thinking }) {
    this.turnEl.textContent = `${SIDE_LABEL[turn]}行棋`;
    this.turnEl.style.color = turn === 'red' ? '#ff6b5e' : '#9fd0ff';
    if (lastMove) this.moveEl.textContent = `第 ${ply} 手 · ${lastMove}`;
    else this.moveEl.textContent = '对局即将开始';
    this.thinkingEl.style.display = thinking ? 'block' : 'none';
  }

  updateScoreBar(red, black) {
    const total = red + black || 1;
    this.barRed.style.width = (red / total * 100) + '%';
    this.barBlack.style.width = (black / total * 100) + '%';
    this.barRed.textContent = `红 ${red.toFixed(1)}`;
    this.barBlack.textContent = `${black.toFixed(1)} 黑`;
  }

  setAffiliation(side) {
    this.affilEl.textContent = `你附身于${SIDE_LABEL[side]} AI`;
    this.affilEl.style.color = side === 'red' ? '#ff6b5e' : '#9fd0ff';
    this.vignette.className = 'vignette ' + (side === 'red' ? 'v-red' : 'v-black');
  }

  toast(text, color) {
    this.toastEl.textContent = text;
    this.toastEl.style.borderLeftColor = color || '#ffd75e';
    this.toastEl.classList.add('show');
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => this.toastEl.classList.remove('show'), 1800);
  }

  // ==================== 快照面板 ====================
  openSnapshot(analysis, eng) {
    this.snapshotOpen = true;
    this.snap.classList.add('open');
    this._renderSnapshot(analysis, eng);
  }

  closeSnapshot() {
    this.snapshotOpen = false;
    this.snap.classList.remove('open');
  }

  _renderSnapshot(a, eng) {
    this.renderMiniBoard(this.snapBoard, eng);

    const cn = (n) => ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'][n] || n;
    const checkTxt = a.checkSide ? ` <b style="color:#ff4b3e">（${SIDE_LABEL[a.checkSide]}被将军！）</b>` : '';
    this.snap.querySelector('#snap-meta').innerHTML =
      `第 ${a.ply} 手 · ${SIDE_LABEL[a.turn]}行棋 · ${a.phase} · 附身 ${SIDE_LABEL[a.mySide]}${checkTxt}`;

    // 最佳着法
    const bestBox = this.snap.querySelector('#snap-best');
    if (a.bestMove) {
      const rel = (a.bestMove.score / 1000).toFixed(1);
      bestBox.innerHTML = `
        <div class="best-notation">${a.bestMove.notation}</div>
        <div class="best-desc">${a.bestMove.desc}（评估 ${rel} 子）</div>
        <div class="best-strategy">${a.strategy}</div>`;
    } else {
      bestBox.innerHTML = '<div class="best-desc">已无合法着法</div>';
    }

    // 候选着法
    const cands = this.snap.querySelector('#snap-cands');
    cands.innerHTML = '';
    a.candidates.forEach((c, i) => {
      const rel = i === 0 ? '最佳' : (c.rel / 1000).toFixed(1) + ' 子';
      const row = this._el('div', 'cand-row');
      row.innerHTML = `<span class="cand-idx">${i + 1}</span><span class="cand-note">${c.notation}</span><span class="cand-tag">${rel}</span>`;
      cands.appendChild(row);
    });

    // 子力对比
    const mat = this.snap.querySelector('#snap-mat');
    const total = a.material.red + a.material.black || 1;
    const advTxt = !a.advantage.side ? '双方接近均势' : `${SIDE_LABEL[a.advantage.side]}约领先 ${a.advantage.amount.toFixed(1)} 子`;
    mat.innerHTML = `
      <div class="mat-bar">
        <div class="mat-red" style="width:${(a.material.red / total * 100).toFixed(1)}%"></div>
        <div class="mat-black" style="width:${(a.material.black / total * 100).toFixed(1)}%"></div>
      </div>
      <div class="mat-label">红 ${a.material.red.toFixed(1)} · 黑 ${a.material.black.toFixed(1)} · ${advTxt}</div>`;

    // 威胁
    const thr = this.snap.querySelector('#snap-threats');
    thr.innerHTML = '';
    if (a.threats.length === 0) thr.innerHTML = '<div class="thr-none">当前无显著受攻子力</div>';
    a.threats.forEach(t => {
      thr.appendChild(this._el('div', 'thr-row', t.text));
    });

    // 附身视角
    this.snap.querySelector('#snap-view').innerHTML = `<div class="view-title">${SIDE_LABEL[a.mySide]} AI 视角</div><div>${a.myView}</div>`;
  }

  /** 迷你棋盘（Canvas 2D） */
  renderMiniBoard(canvas, eng) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const M = 22, dx = (W - M * 2) / 8, dy = (H - M * 2) / 9;
    const X = c => M + c * dx, Y = r => M + r * dy;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#e9c988';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#5a3a16';
    ctx.lineWidth = 2;
    for (let r = 0; r < 10; r++) { ctx.beginPath(); ctx.moveTo(X(0), Y(r)); ctx.lineTo(X(8), Y(r)); ctx.stroke(); }
    for (let c = 0; c < 9; c++) {
      if (c === 0 || c === 8) { ctx.beginPath(); ctx.moveTo(X(c), Y(0)); ctx.lineTo(X(c), Y(9)); ctx.stroke(); }
      else { ctx.beginPath(); ctx.moveTo(X(c), Y(0)); ctx.lineTo(X(c), Y(4)); ctx.stroke(); ctx.beginPath(); ctx.moveTo(X(c), Y(5)); ctx.lineTo(X(c), Y(9)); ctx.stroke(); }
    }
    ctx.beginPath(); ctx.moveTo(X(3), Y(0)); ctx.lineTo(X(5), Y(2)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(5), Y(0)); ctx.lineTo(X(3), Y(2)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(3), Y(9)); ctx.lineTo(X(5), Y(7)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(5), Y(9)); ctx.lineTo(X(3), Y(7)); ctx.stroke();
    ctx.font = `12px "KaiTi","STKaiti",serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = '#5a3a16';
    ctx.fillText('楚 河', X(2), Y(4.5));
    ctx.fillText('漢 界', X(6), Y(4.5));
    // 棋子
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 9; c++) {
        const code = eng.get(c, r);
        if (code === 0) continue;
        const red = isRed(code);
        ctx.beginPath();
        ctx.arc(X(c), Y(r), dx * 0.42, 0, Math.PI * 2);
        ctx.fillStyle = red ? '#f3e2bd' : '#efe6d2';
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = red ? '#c22f2f' : '#232323';
        ctx.stroke();
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = `bold ${Math.round(dx * 0.42)}px "Microsoft YaHei",sans-serif`;
        ctx.fillText(PIECE_CN[code] || PIECE_NAMES[code], X(c), Y(r) + 1);
      }
    }
  }

  // ==================== 结果弹层 ====================
  showResult({ winner, reason }) {
    const title = this.result.querySelector('#result-title');
    const sub = this.result.querySelector('#result-sub');
    if (winner === 'draw') {
      title.textContent = '和棋';
      title.style.color = '#ffd75e';
      sub.textContent = reason || '重复局面 / 子力不足';
    } else {
      title.textContent = `${SIDE_LABEL[winner]}胜`;
      title.style.color = winner === 'red' ? '#ff6b5e' : '#9fd0ff';
      sub.textContent = reason || '将死';
    }
    this.result.classList.add('show');
  }

  hideResult() { this.result.classList.remove('show'); }

  setHint(text) { this.hint.innerHTML = text; }
}

/* ==================== 简单 WebAudio 音效 ==================== */
class Sfx {
  constructor() { this.ctx = null; }
  _ensure() {
    if (!this.ctx) {
      try { this.ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { return null; }
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  }
  _tone(freq, dur, type, gain, when = 0) {
    const ctx = this._ensure();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type || 'sine';
    osc.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, ctx.currentTime + when);
    g.gain.exponentialRampToValueAtTime(gain || 0.08, ctx.currentTime + when + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + when + dur);
    osc.connect(g).connect(ctx.destination);
    osc.start(ctx.currentTime + when);
    osc.stop(ctx.currentTime + when + dur + 0.05);
  }
  move() { this._tone(520, 0.08, 'triangle', 0.05); }
  capture() { this._tone(180, 0.16, 'square', 0.07); this._tone(90, 0.2, 'sine', 0.08, 0.02); }
  check() { this._tone(880, 0.1, 'square', 0.06); this._tone(660, 0.12, 'square', 0.06, 0.11); }
  win() { [523, 659, 784, 1047].forEach((f, i) => this._tone(f, 0.22, 'triangle', 0.07, i * 0.13)); }
  lose() { [392, 330, 262, 196].forEach((f, i) => this._tone(f, 0.25, 'triangle', 0.06, i * 0.15)); }
}
