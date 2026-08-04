/* ============================================================
 * 3D 象棋对战 —— RL 驱动的中国象棋
 *
 * 玩家附身 AI 角色，随机立于红方或黑方一侧，自由在棋盘附近走动，
 * 拥有与其他 3D 探索游戏一致的控制权（WASD移动/拖拽视角/滚轮缩放）。
 * AI 角色可通过棋盘快照感知局势，并按所在方回合进行 LLM 战术分析。
 *
 * 玩家(红方) + AI助手(LLM) vs RL机器人(黑方)
 * RL机器人通过 DQN 不断学习进化
 *
 * 架构：
 * - xiangqi-engine.js → 纯棋盘逻辑（走法/将军/将死）
 * - xiangqi-game.js  → 3D渲染 + 探索操控 + RL集成 + AI辅助 + UI面板
 * - games-config.js  → RL超参注册
 * - agent.py         → LLM策略决策（后端）
 * ============================================================ */

import { BaseGame } from './base-game.js';
import {
  XiangqiEngine, PIECE, PIECE_NAMES, PIECE_VALUES,
  BOARD_COLS, BOARD_ROWS, pieceType, pieceSide, isRed, isBlack,
} from './xiangqi-engine.js';
import { RLAgentManager } from '../rl/rl-agent-manager.js';

// ==================== 常量 ====================
const CELL = 1.0;
const PIECE_R = 0.38;
const PIECE_H = 0.16;
const MAX_ACTIONS = 48;
const STATE_SIZE = 105;

// 观战角色：玩家附身 AI 角色随机立于红方或黑方一侧，自由走动观战
// 红方位于棋盘下沿(row 9 → z=+4.5)，黑方位于上沿(row 0 → z=-4.5)
const OBSERVER_POS = {
  red:   { x: 3.2, z: 6.0 },  // 红方一侧（棋盘下沿外）
  black: { x: 3.2, z: -6.0 }, // 黑方一侧（棋盘上沿外）
};

const SPEED_PRESETS = [1, 3, 10, 25, 60];
const SPEED_DELAYS = { 1: 600, 3: 250, 10: 80, 25: 30, 60: 10 };

const RL_REWARD = {
  capture_mult: 1.0,
  check: 3,
  checkmate: 100,
  stalemate: -15,
  lose: -100,
  draw: -10, // 平局：小负，鼓励 RL 争取胜利而非磨和
  step: -0.02,
  develop: 0.1,
  center_control: 0.03,
  king_safety: 0.01,
  threat: 0.15,      // 威胁对方车/马/炮等高分棋子的奖励系数
  lose_piece: 0.5,   // 送死惩罚系数（按被吃子价值）
  in_check: -2,      // 被将军惩罚
};

const PIECE_CHARS = {
  1: '帅', 2: '仕', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
  8: '将', 9: '士', 10: '象', 11: '马', 12: '车', 13: '炮', 14: '卒',
};

const STRATEGY_LABELS = {
  open_attack: '⚔️ 开局进攻',
  solid_defense: '🛡️ 稳固防守',
  exchange_simplify: '🔄 兑子简化',
  center_control: '🎯 控制中心',
  flank_attack: '↔️ 侧翼进攻',
  king_safety: '👑 将帅安全',
  counter_attack: '↩️ 反击',
};

const STRATEGY_CANDIDATES = Object.keys(STRATEGY_LABELS);

// ==================== 游戏类 ====================
export class XiangqiGame extends BaseGame {
  constructor(app) {
    super(app);
    this.name = 'xiangqi';
    this.displayName = '象棋对战';
    this.description = '玩家附身AI角色随机立于红黑一方，自由在棋盘附近走动观战，AI实时分析棋局并提供战术建议。';
    this.uiHint = 'WASD自由走动 · 拖拽视角 · 点击红方棋子走子 · AI角色分析局势';

    this.initialCameraRadius = 8;    // 相机与角色距离（探索视角，可拖拽/滚轮调整）
    this.initialCameraHeight = 6.5;  // 相机初始高度（俯瞰角色与棋盘）
    this.boundarySize = 14;          // 活动边界：棋盘附近自由走动（半宽 7m）
    this.moveSpeed = 3.5;            // 探索移动速度（与其他3D游戏一致）

    this._engine = new XiangqiEngine();
    this._selectedPiece = null;
    this._validMoves = [];
    this._isAnimating = false;
    this._lastMove = null;
    this._moveCount = 0;
    this._turnTimer = 0;
    this._waitingForRobot = false;

    // 玩家附身的 AI 角色所在一方（随机：红/黑），决定观战位置与 LLM 分析视角
    this._observerSide = 'red';
    this._playerSpeed = 0;           // 当前移动速度（感知数据用）
    this._pointerDownInfo = null;    // 指针按下位置（区分点击走子与拖拽视角）

    this._boardGroup = null;
    this._pieceMeshes = new Map();
    this._pieceTextures = new Map();
    this._boardPlane = null;
    this._selectedRing = null;
    this._moveIndicators = [];
    this._lastMoveHighlights = [];
    this._raycaster = null;

    this._rl = {
      agent: null,
      enabled: false,
      speed: 1,
      autoPlay: false,
      pendingState: null,
      pendingAction: null,
      pendingReward: 0,
      lastState: null,
      lastAction: null,
      episodes: 0,
      wins: { red: 0, black: 0, draw: 0 },
      trainSteps: 0,
      _restartTimer: null, // 对局结束自动开新局的定时器
    };

    this._llm = {
      enabled: false,
      strategy: null,
      waitingResponse: false,
      policyTimer: 0,
      policyInterval: 5,
      lastAdvice: '',
      stats: { requests: 0, responses: 0 },
    };

    this._panel = null;
    this._statusBar = null;
    this._dragCleanup = null;
    this._canvas = null;
    this._boundPointerDown = null;
  }

  // ==================== 生命周期 ====================

  generateScene() {
    const THREE = this.THREE;
    const scene = this.App.scene;

    // 棋盘组
    this._boardGroup = new THREE.Group();
    this.addToScene(this._boardGroup);

    // 棋盘底座
    const boardW = (BOARD_COLS - 1) * CELL + 1.2;
    const boardD = (BOARD_ROWS - 1) * CELL + 1.2;
    const baseGeo = new THREE.BoxGeometry(boardW, 0.3, boardD);
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0xd4a86a, roughness: 0.7, metalness: 0.05,
    });
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.position.y = -0.15;
    base.receiveShadow = true;
    this._boardGroup.add(base);

    // 棋盘边框（中空边框：四根条形沿底座四周，顶面与底座顶面 y=0 齐平但互不重叠，
    // 避免原实心方盒顶面与底座顶面共面导致整盘 z-fighting 闪烁）
    const frameMat = new THREE.MeshStandardMaterial({
      color: 0x5a3a1a, roughness: 0.8,
    });
    const FRAME_TH = 0.12;            // 边框厚度
    const FRAME_H = 0.35;             // 边框高度（顶面 y=0，与盘面齐平）
    const frameY = -FRAME_H / 2;
    const halfW = boardW / 2;
    const halfD = boardD / 2;
    const addStrip = (w, d, x, z) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, FRAME_H, d), frameMat);
      m.position.set(x, frameY, z);
      this._boardGroup.add(m);
    };
    addStrip(FRAME_TH, boardD, -(halfW + FRAME_TH / 2), 0);                 // 左
    addStrip(FRAME_TH, boardD,  halfW + FRAME_TH / 2, 0);                   // 右
    addStrip(boardW + FRAME_TH * 2, FRAME_TH, 0, -(halfD + FRAME_TH / 2)); // 上
    addStrip(boardW + FRAME_TH * 2, FRAME_TH, 0,  halfD + FRAME_TH / 2);   // 下

    // 网格线
    this._createGridLines();

    // 河界文字
    this._createRiverText();

    // 九宫格斜线
    this._createPalaceMarks();

    // 射线检测平面
    const planeGeo = new THREE.PlaneGeometry(boardW, boardD);
    const planeMat = new THREE.MeshBasicMaterial({ visible: false });
    this._boardPlane = new THREE.Mesh(planeGeo, planeMat);
    this._boardPlane.rotation.x = -Math.PI / 2;
    this._boardPlane.position.y = 0.01;
    this._boardPlane.userData.isBoard = true;
    this._boardGroup.add(this._boardPlane);

    // 灯光
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    this._boardGroup.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(3, 10, 5);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(1024, 1024);
    dirLight.shadow.camera.left = -8;
    dirLight.shadow.camera.right = 8;
    dirLight.shadow.camera.top = 8;
    dirLight.shadow.camera.bottom = -8;
    // 阴影偏移：消除平铺盘面上的阴影痤疮（暗斑/闪烁）
    dirLight.shadow.bias = -0.002;
    dirLight.shadow.normalBias = 0.02;
    this._boardGroup.add(dirLight);
    const fillLight = new THREE.PointLight(0xffcc88, 0.5, 15);
    fillLight.position.set(0, 5, 0);
    this._boardGroup.add(fillLight);

    // 创建棋子纹理
    for (let code = 1; code <= 14; code++) {
      this._pieceTextures.set(code, this._createPieceTexture(code));
    }

    // 创建棋子
    this._syncAllPieces();

    // 射线检测器
    this._raycaster = new THREE.Raycaster();

    // 交互事件：点击走子（pointerup 判定点击/拖拽，拖拽交给管理器旋转视角）
    this._canvas = this.App.renderer.domElement;
    this._boundPointerDown = (e) => this._onPointerDown(e);
    this._boundPointerUp = (e) => this._onPointerUp(e);
    this._canvas.addEventListener('pointerdown', this._boundPointerDown);
    this._canvas.addEventListener('pointerup', this._boundPointerUp);

    // UI
    this._createStatusBar();
    this._createRLPanel();

    console.log('[象棋] 场景已生成');
  }

  onStart() {
    super.onStart();
    this.state = 'playing';
    this._engine.reset();
    this._syncAllPieces();
    this._moveCount = 0;
    this._lastMove = null;
    this._setupObserver();
    this._updateStatusBar();
    this._updatePanelUI();
    this.userControlling = true;
    console.log('[象棋] 游戏开始');
  }

  /**
   * 布置观战角色：AI 角色随机立于红方或黑方一侧，面向棋盘观战。
   * 玩家附身该角色，保留用户操控权（WASD 自由走动 + 拖拽视角 + 滚轮缩放），
   * 与其它 3D 探索游戏控制方式完全一致。角色所在方决定 LLM 局势分析的视角。
   */
  _setupObserver() {
    const App = this.App;
    const avatar = App.currentAvatar || App.modelGroup;
    if (!avatar) return;
    // 随机立于红方或黑方一侧
    this._observerSide = Math.random() < 0.5 ? 'red' : 'black';
    const base = OBSERVER_POS[this._observerSide];
    const rx = base.x + (Math.random() - 0.5) * 3;
    const rz = base.z + (Math.random() - 0.5) * 1.5;
    avatar.position.set(rx, avatar.position.y, rz);
    // 面向棋盘中心
    const yaw = App.computeBodyFaceCam
      ? App.computeBodyFaceCam(avatar, { x: 0, y: 0, z: 0 })
      : -Math.PI / 2;
    App.smoothRotY = yaw;
    avatar.rotation.set(0, yaw, 0);
    // 保留玩家操控：用户可自由走动（不调用 releaseUserControl）
    const mgr = App.gameModeManager;
    if (mgr && mgr.controlBridge && !mgr.controlBridge.userControlling) {
      try { mgr.controlBridge.activateUserControl(); } catch (e) { /* 忽略 */ }
    }
    console.log(`[象棋] AI角色随机立于${this._observerSide === 'red' ? '红方' : '黑方'}一侧观战`);
  }

  update(dt) {
    if (this.state !== 'playing') return;
    super.update(dt);

    // 相机由 GameModeManager 统一接管：跟随角色轨道视角（与其他3D探索游戏一致）

    // 动画更新
    if (this._isAnimating) {
      this._updateAnimation(dt);
      return;
    }

    // 游戏结束
    if (this._engine.gameOver) return;

    // LLM 策略（AI 角色所在方回合时分析局势）
    if (this._llm.enabled && this._engine.turn === this._observerSide && !this._rl.autoPlay) {
      this._llmStepPolicy(dt);
    }

    // RL 机器人回合
    if (this._engine.turn === 'black' && !this._rl.autoPlay) {
      this._turnTimer += dt * 1000;
      if (this._turnTimer >= SPEED_DELAYS[this._rl.speed] || this._rl.speed >= 10) {
        this._turnTimer = 0;
        if (this._rl.enabled) {
          this._rlRobotMove();
        } else {
          this._heuristicMove('black');
        }
      }
    }

    // 自动对弈模式
    if (this._rl.autoPlay && !this._isAnimating) {
      this._turnTimer += dt * 1000;
      if (this._turnTimer >= SPEED_DELAYS[this._rl.speed]) {
        this._turnTimer = 0;
        this._autoPlayStep();
      }
    }

    this._updateStatusBar();
  }

  cleanup() {
    if (this._boundPointerDown && this._canvas) {
      this._canvas.removeEventListener('pointerdown', this._boundPointerDown);
    }
    if (this._boundPointerUp && this._canvas) {
      this._canvas.removeEventListener('pointerup', this._boundPointerUp);
    }
    if (this._dragCleanup) { this._dragCleanup(); this._dragCleanup = null; }
    if (this._panel) { this._panel.remove(); this._panel = null; }
    if (this._statusBar) { this._statusBar.remove(); this._statusBar = null; }
    this._pieceMeshes.clear();
    this._pieceTextures.clear();
    this._moveIndicators = [];
    this._lastMoveHighlights = [];
    this._pointerDownInfo = null;
    super.cleanup();
    console.log('[象棋] 资源已清理');
  }

  /**
   * 场景特效钩子（由 GameModeManager 在视角更新之后调用）。
   * 象棋使用管理器统一的跟随角色轨道视角，无需特殊处理。
   */
  updateSceneEffects(t) {}

  // ==================== 3D 渲染 ====================

  _boardX(col) { return (col - (BOARD_COLS - 1) / 2) * CELL; }
  _boardZ(row) { return (row - (BOARD_ROWS - 1) / 2) * CELL; }
  _worldToCol(x) { return Math.round(x / CELL + (BOARD_COLS - 1) / 2); }
  _worldToRow(z) { return Math.round(z / CELL + (BOARD_ROWS - 1) / 2); }

  _createGridLines() {
    const THREE = this.THREE;
    const points = [];
    // 网格线略高于盘面，与九宫斜线/河界文字拉开间距，避免斜视角下 z-fighting 闪烁
    const y = 0.02;

    // 横线
    for (let r = 0; r < BOARD_ROWS; r++) {
      const z = this._boardZ(r);
      points.push(this._boardX(0), y, z, this._boardX(BOARD_COLS - 1), y, z);
    }
    // 竖线（中间列在河界处断开）
    for (let c = 0; c < BOARD_COLS; c++) {
      const x = this._boardX(c);
      if (c === 0 || c === BOARD_COLS - 1) {
        points.push(x, y, this._boardZ(0), x, y, this._boardZ(BOARD_ROWS - 1));
      } else {
        points.push(x, y, this._boardZ(0), x, y, this._boardZ(4));
        points.push(x, y, this._boardZ(5), x, y, this._boardZ(BOARD_ROWS - 1));
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x3a2a0a, linewidth: 2 });
    const lines = new THREE.LineSegments(geo, mat);
    this._boardGroup.add(lines);
  }

  _createRiverText() {
    const THREE = this.THREE;
    const canvas = document.createElement('canvas');
    canvas.width = 512; canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0,0,0,0)';
    ctx.fillRect(0, 0, 512, 128);
    ctx.font = 'bold 48px "Microsoft YaHei", "SimHei", sans-serif';
    ctx.fillStyle = 'rgba(90,58,26,0.6)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('楚  河', 128, 64);
    ctx.fillText('漢  界', 384, 64);
    const tex = new THREE.CanvasTexture(canvas);
    const geo = new THREE.PlaneGeometry(8 * CELL, 1 * CELL);
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = 0.05;
    this._boardGroup.add(mesh);
  }

  _createPalaceMarks() {
    const THREE = this.THREE;
    const points = [];
    // 九宫斜线在网格线之上，避免交叉处 z-fighting
    const y = 0.035;
    // 红方九宫（row 7-9, col 3-5）
    points.push(this._boardX(3), y, this._boardZ(7), this._boardX(5), y, this._boardZ(9));
    points.push(this._boardX(5), y, this._boardZ(7), this._boardX(3), y, this._boardZ(9));
    // 黑方九宫（row 0-2, col 3-5）
    points.push(this._boardX(3), y, this._boardZ(0), this._boardX(5), y, this._boardZ(2));
    points.push(this._boardX(5), y, this._boardZ(0), this._boardX(3), y, this._boardZ(2));
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x3a2a0a });
    const lines = new THREE.LineSegments(geo, mat);
    this._boardGroup.add(lines);
  }

  _createPieceTexture(code) {
    const canvas = document.createElement('canvas');
    canvas.width = 128; canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const red = isRed(code);
    const char = PIECE_CHARS[code] || '?';

    ctx.clearRect(0, 0, 128, 128);
    // 背景圆
    ctx.fillStyle = '#f5deb3';
    ctx.beginPath(); ctx.arc(64, 64, 60, 0, Math.PI * 2); ctx.fill();
    // 内圈
    ctx.strokeStyle = red ? '#cc0000' : '#1a1a1a';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(64, 64, 48, 0, Math.PI * 2); ctx.stroke();
    // 文字：旋转 90° 使双方棋子从各自一方视角看为正立
    // 圆柱顶盖 UV：纹理 U 轴↔世界 Z 轴、V 轴↔世界 X 轴，CanvasTexture 又纵向翻转，
    // 若按画布原样绘制，字会侧躺 90°。红方(棋盘 +Z 侧)应字顶朝 -Z（画布左侧），
    // 黑方(棋盘 -Z 侧)应字顶朝 +Z（画布右侧），故红方逆时针转、黑方顺时针转。
    ctx.save();
    ctx.translate(64, 64);
    ctx.rotate(red ? -Math.PI / 2 : Math.PI / 2);
    ctx.translate(-64, -64);
    ctx.fillStyle = red ? '#cc0000' : '#1a1a1a';
    ctx.font = 'bold 56px "Microsoft YaHei", "SimHei", "PingFang SC", sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(char, 64, 66);
    ctx.restore();

    const tex = new this.THREE.CanvasTexture(canvas);
    tex.needsUpdate = true;
    return tex;
  }

  _createPieceMesh(code, c, r) {
    const THREE = this.THREE;
    const geo = new THREE.CylinderGeometry(PIECE_R, PIECE_R * 0.95, PIECE_H, 32);
    const sideMat = new THREE.MeshStandardMaterial({
      color: isRed(code) ? 0xf5deb3 : 0xe8c898, roughness: 0.6, metalness: 0.1,
    });
    const topTex = this._pieceTextures.get(code);
    const topMat = new THREE.MeshStandardMaterial({ map: topTex, roughness: 0.5 });
    const bottomMat = new THREE.MeshStandardMaterial({ color: 0xd4a86a, roughness: 0.6 });
    // [side, top, bottom]
    const mesh = new THREE.Mesh(geo, [sideMat, topMat, bottomMat]);
    mesh.position.set(this._boardX(c), PIECE_H / 2, this._boardZ(r));
    mesh.userData = { c, r, code, isPiece: true };
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  _syncAllPieces() {
    // 清除旧棋子
    for (const mesh of this._pieceMeshes.values()) {
      if (mesh.parent) mesh.parent.remove(mesh);
      this._disposeObject(mesh);
    }
    this._pieceMeshes.clear();

    // 创建新棋子
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        const code = this._engine.board[r][c];
        if (code === 0) continue;
        const mesh = this._createPieceMesh(code, c, r);
        this._boardGroup.add(mesh);
        this._pieceMeshes.set(`${c},${r}`, mesh);
      }
    }
  }

  _clearHighlights() {
    const THREE = this.THREE;
    if (this._selectedRing) {
      this._boardGroup.remove(this._selectedRing);
      this._disposeObject(this._selectedRing);
      this._selectedRing = null;
    }
    for (const ind of this._moveIndicators) {
      this._boardGroup.remove(ind);
      this._disposeObject(ind);
    }
    this._moveIndicators = [];
  }

  _showMoveIndicators(moves) {
    const THREE = this.THREE;
    for (const move of moves) {
      const [tc, tr] = move.to;
      const isCap = move.captured !== 0;
      const innerR = isCap ? PIECE_R * 0.95 : 0.15;
      const outerR = isCap ? PIECE_R * 1.15 : 0.28;
      const geo = new THREE.RingGeometry(innerR, outerR, 32);
      const mat = new THREE.MeshBasicMaterial({
        color: isCap ? 0xff4444 : 0x44ff44,
        transparent: true, opacity: 0.7, side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(geo, mat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(this._boardX(tc), 0.08, this._boardZ(tr));
      this._boardGroup.add(ring);
      this._moveIndicators.push(ring);
    }
  }

  _showLastMoveHighlight(move) {
    const THREE = this.THREE;
    for (const h of this._lastMoveHighlights) {
      this._boardGroup.remove(h);
      this._disposeObject(h);
    }
    this._lastMoveHighlights = [];
    if (!move) return;

    for (const [c, r] of [move.from, move.to]) {
      const geo = new THREE.PlaneGeometry(CELL * 0.85, CELL * 0.85);
      const mat = new THREE.MeshBasicMaterial({
        color: 0xffaa00, transparent: true, opacity: 0.25,
      });
      const sq = new THREE.Mesh(geo, mat);
      sq.rotation.x = -Math.PI / 2;
      sq.position.set(this._boardX(c), 0.065, this._boardZ(r));
      this._boardGroup.add(sq);
      this._lastMoveHighlights.push(sq);
    }
  }

  // ==================== 交互 ====================

  /**
   * 按下指针：仅记录位置。
   * 拖拽视角（由 GameModeManager 接管）与点击走子共存——
   * 若按下后发生明显位移，视为拖拽视角，不触发走子。
   */
  _onPointerDown(e) {
    if (this._isAnimating || this._engine.gameOver) return;
    if (this._rl.autoPlay) return;
    if (this._engine.turn !== 'red') return;
    this._pointerDownInfo = { x: e.clientX, y: e.clientY };
  }

  _onPointerUp(e) {
    const info = this._pointerDownInfo;
    this._pointerDownInfo = null;
    if (!info) return;
    if (this._isAnimating || this._engine.gameOver) return;
    if (this._rl.autoPlay) return;
    if (this._engine.turn !== 'red') return;
    // 位移超过阈值 = 拖拽视角，非点击
    if (Math.abs(e.clientX - info.x) + Math.abs(e.clientY - info.y) > 6) return;

    const THREE = this.THREE;
    const rect = this._canvas.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this._raycaster.setFromCamera(mouse, this.App.camera);

    // 检测棋子
    const meshes = Array.from(this._pieceMeshes.values());
    const pieceHits = this._raycaster.intersectObjects(meshes, false);
    if (pieceHits.length > 0) {
      const ud = pieceHits[0].object.userData;
      this._handleClick(ud.c, ud.r);
      return;
    }
    // 检测棋盘
    const boardHits = this._raycaster.intersectObject(this._boardPlane, false);
    if (boardHits.length > 0) {
      const p = boardHits[0].point;
      const c = this._worldToCol(p.x);
      const r = this._worldToRow(p.z);
      if (c >= 0 && c < BOARD_COLS && r >= 0 && r < BOARD_ROWS) {
        this._handleClick(c, r);
      }
    }
  }

  _handleClick(c, r) {
    const piece = this._engine.get(c, r);
    if (this._selectedPiece) {
      // 已选中棋子，尝试移动
      const move = this._validMoves.find(m => m.to[0] === c && m.to[1] === r);
      if (move) {
        this._clearHighlights();
        this._executePlayerMove(move);
        return;
      }
      // 点击了另一个己方棋子，切换选择
      if (piece !== 0 && isRed(piece)) {
        this._selectPiece(c, r);
        return;
      }
      // 点击了其他地方，取消选择
      this._clearHighlights();
      this._selectedPiece = null;
      this._validMoves = [];
      return;
    }
    // 未选中棋子，选择己方棋子
    if (piece !== 0 && isRed(piece)) {
      this._selectPiece(c, r);
    }
  }

  _selectPiece(c, r) {
    const THREE = this.THREE;
    this._clearHighlights();
    this._selectedPiece = { c, r };

    // 选中环
    const geo = new THREE.RingGeometry(PIECE_R * 1.05, PIECE_R * 1.25, 32);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xffaa00, transparent: true, opacity: 0.8, side: THREE.DoubleSide,
    });
    this._selectedRing = new THREE.Mesh(geo, mat);
    this._selectedRing.rotation.x = -Math.PI / 2;
    this._selectedRing.position.set(this._boardX(c), 0.095, this._boardZ(r));
    this._boardGroup.add(this._selectedRing);

    // 合法走法
    const rawMoves = this._engine.generatePieceMoves(c, r);
    this._validMoves = rawMoves.filter(m => !this._engine._movesIntoCheck(m, 'red'));
    this._showMoveIndicators(this._validMoves);
  }

  // ==================== 走法执行与动画 ====================

  _executePlayerMove(move) {
    this._selectedPiece = null;
    this._validMoves = [];

    // RL 奖励结算（如果黑方有待结算的经验）
    if (this._rl.enabled && this._rl.pendingState !== null) {
      this._rlOnOpponentMove(move);
    }

    this._executeMove(move, () => {
      this._lastMove = move;
      this._showLastMoveHighlight(move);
      this._moveCount++;

      if (this._engine.checkGameEnd()) {
        this._onGameEnd();
        return;
      }
      this._updateStatusBar();
      this._updatePanelUI();
    });
  }

  /**
   * 执行走法（含动画）。
   * @param {Object} move 走法对象
   * @param {Function} callback 落子完成回调
   * @param {boolean} instant true=跳过动画立即落子（自动对弈高速训练用）
   */
  _executeMove(move, callback, instant) {
    const [fc, fr] = move.from;
    const [tc, tr] = move.to;
    const fromKey = `${fc},${fr}`;
    const toKey = `${tc},${tr}`;

    const mesh = this._pieceMeshes.get(fromKey);
    if (!mesh) { if (callback) callback(); return; }

    // 移除被吃棋子
    if (move.captured) {
      const capMesh = this._pieceMeshes.get(toKey);
      if (capMesh) {
        this._pieceMeshes.delete(toKey);
        this._boardGroup.remove(capMesh);
        this._disposeObject(capMesh);
      }
    }

    // 更新映射
    this._pieceMeshes.delete(fromKey);
    this._pieceMeshes.set(toKey, mesh);

    // 执行引擎走法
    this._engine.makeMove(move);

    // 快速模式：跳过动画立即落子（自动对弈训练 / 高速倍速）
    if (instant || (this._rl.autoPlay && this._rl.speed >= 3)) {
      mesh.position.set(this._boardX(tc), PIECE_H / 2, this._boardZ(tr));
      mesh.userData.c = tc;
      mesh.userData.r = tr;
      this._isAnimating = false;
      this._anim = null;
      if (callback) callback();
      return;
    }

    // 动画（高速倍速下缩短动画时长）
    const dur = this._rl.speed >= 10 ? 0.08 : 0.4;
    this._isAnimating = true;
    this._anim = {
      mesh, timer: 0, duration: dur,
      fromX: this._boardX(fc), fromZ: this._boardZ(fr),
      toX: this._boardX(tc), toZ: this._boardZ(tr),
      baseY: PIECE_H / 2,
      callback,
    };
  }

  _updateAnimation(dt) {
    const a = this._anim;
    if (!a) { this._isAnimating = false; return; }
    a.timer += dt;
    let t = Math.min(a.timer / a.duration, 1);
    // 缓动
    const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    a.mesh.position.x = a.fromX + (a.toX - a.fromX) * ease;
    a.mesh.position.z = a.fromZ + (a.toZ - a.fromZ) * ease;
    // 抛物线高度
    a.mesh.position.y = a.baseY + Math.sin(t * Math.PI) * 0.8;

    if (t >= 1) {
      a.mesh.position.set(a.toX, a.baseY, a.toZ);
      a.mesh.userData.c = this._worldToCol(a.toX);
      a.mesh.userData.r = this._worldToRow(a.toZ);
      this._isAnimating = false;
      this._anim = null;
      if (a.callback) a.callback();
    }
  }

  // ==================== 游戏流程 ====================

  _newGame() {
    if (this._rl._restartTimer) {
      clearTimeout(this._rl._restartTimer);
      this._rl._restartTimer = null;
    }
    this._engine.reset();
    this._syncAllPieces();
    this._clearHighlights();
    this._moveCount = 0;
    this._lastMove = null;
    this._turnTimer = 0;
    this._showLastMoveHighlight(null);
    this._rl.pendingState = null;
    this._rl.pendingAction = null;
    this._rl.pendingReward = 0;
    this._updateStatusBar();
    this._updatePanelUI();
  }

  _onGameEnd() {
    const winner = this._engine.winner;
    const reason = this._engine.endReason;
    if (winner === 'red') this._rl.wins.red++;
    else if (winner === 'black') this._rl.wins.black++;
    else if (!this._rl.wins.draw) this._rl.wins.draw = 1;
    else this._rl.wins.draw++;

    // RL 回合结算（平局给黑方小负奖励，鼓励其争取胜利而非磨和）
    if (this._rl.enabled && this._rl.agent) {
      const terminalReward = winner === 'black' ? RL_REWARD.checkmate
        : winner === 'red' ? RL_REWARD.lose
        : RL_REWARD.draw;
      this._rl.pendingReward += terminalReward;
      if (this._rl.pendingState !== null) {
        const nextState = this._rlEncodeState();
        this._rl.agent.store(
          this._rl.pendingState, this._rl.pendingAction,
          this._rl.pendingReward, nextState, true,
        );
        this._rl.agent.train();
      }
      this._rl.agent.endEpisode(this._rl.pendingReward, winner || 'draw');
      this._rl.episodes++;
      this._rl.pendingState = null;
    }

    const winText = winner === 'red' ? '🎉 红方胜!'
      : winner === 'black' ? '🤖 黑方(RL)胜!'
      : '🤝 平局!';
    const reasonText = reason === 'checkmate' ? '将死'
      : reason === 'stalemate' ? '困毙'
      : reason === 'draw_50move' ? '60回合无吃子'
      : reason === 'draw_repetition' ? '局面重复三次'
      : reason === 'draw_material' ? '子力不足'
      : '和棋';

    // RL 开启或自动对弈时，对局结束自动开始新一局（连续训练）
    if (this._rl.enabled || this._rl.autoPlay) {
      this._setStatus(`${winText} (${reasonText}) | 步数: ${this._moveCount} | 1.5秒后自动开始新对局`);
      if (this._rl._restartTimer) clearTimeout(this._rl._restartTimer);
      this._rl._restartTimer = setTimeout(() => this._newGame(), 1500);
    } else {
      this._setStatus(`${winText} (${reasonText}) | 步数: ${this._moveCount} | 点击"新局"重开`);
    }

    this._updatePanelUI();
    console.log(`[象棋] 对局结束: ${winner || 'draw'} (${reason})`);
  }

  // ==================== RL 集成 ====================

  _rlEnsureAgent() {
    if (this._rl.agent) return this._rl.agent;
    try {
      this._rl.agent = RLAgentManager.get().getAgent('xiangqi', this);
      console.log('[象棋] RL Agent 已创建');
    } catch (e) {
      console.warn('[象棋] RL Agent 创建失败:', e);
    }
    return this._rl.agent;
  }

  _rlEncodeState() {
    const state = new Float64Array(STATE_SIZE);
    const board = this._engine.board;

    // 棋盘编码 (90维)
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        state[r * BOARD_COLS + c] = board[r][c] / 14;
      }
    }

    // 战略特征 (15维)
    const redMat = this._engine.materialValue('red');
    const blackMat = this._engine.materialValue('black');
    state[90] = (redMat - blackMat) / 40;

    const rKing = this._engine.findKing('red');
    const bKing = this._engine.findKing('black');
    state[91] = rKing ? rKing[0] / 8 : 0.5;
    state[92] = rKing ? rKing[1] / 9 : 0.5;
    state[93] = bKing ? bKing[0] / 8 : 0.5;
    state[94] = bKing ? bKing[1] / 9 : 0.5;

    const kingDist = rKing && bKing
      ? Math.abs(rKing[0] - bKing[0]) + Math.abs(rKing[1] - bKing[1]) : 17;
    state[95] = kingDist / 17;

    let redMoves = 0, blackMoves = 0;
    try {
      redMoves = this._engine.generateAllMoves('red').length;
      blackMoves = this._engine.generateAllMoves('black').length;
    } catch (e) {}
    state[96] = Math.min(redMoves / 50, 1);
    state[97] = Math.min(blackMoves / 50, 1);

    const cur = this._engine.turn;
    const enemy = cur === 'red' ? 'black' : 'red';
    state[98] = this._engine.isInCheck(cur) ? 1 : 0;
    state[99] = this._engine.isInCheck(enemy) ? 1 : 0;

    const totalP = this._engine.pieceCount('red') + this._engine.pieceCount('black');
    state[100] = totalP > 28 ? 0 : totalP > 16 ? 0.5 : 1;
    state[101] = cur === 'red' ? 0 : 1;
    state[102] = this._engine.pieceCount('red') / 16;
    state[103] = this._engine.pieceCount('black') / 16;
    state[104] = (this._lastMove && this._lastMove.captured) ? 1 : 0;

    return state;
  }

  /**
   * 判断棋盘格 (c, r) 是否被 side 的敌人方任意棋子攻击。
   * 即：(c, r) 是否处于 side 方的受威胁状态。
   */
  _isSquareAttacked(side, c, r) {
    const enemy = side === 'red' ? 'black' : 'red';
    const board = this._engine.board;
    for (let rr = 0; rr < BOARD_ROWS; rr++) {
      for (let cc = 0; cc < BOARD_COLS; cc++) {
        const code = board[rr][cc];
        if (code === 0 || pieceSide(code) !== enemy) continue;
        const moves = this._engine.generatePieceMoves(cc, rr);
        for (const m of moves) {
          if (m.to[0] === c && m.to[1] === r) return true;
        }
      }
    }
    return false;
  }

  /**
   * 判断 (c, r) 是否受 side 己方棋子保护（反吃能力）。
   * 走子后目标格被己方棋子占据，走法生成会排除己方占位，
   * 因此需临时清空该格，再检测己方棋子能否走到（被吃后可反吃）。
   */
  _isProtectedBySelf(side, c, r) {
    const board = this._engine.board;
    const saved = board[r][c];
    board[r][c] = 0; // 模拟该格空置，检测己方反吃能力
    let protectedFlag = false;
    for (let rr = 0; rr < BOARD_ROWS && !protectedFlag; rr++) {
      for (let cc = 0; cc < BOARD_COLS && !protectedFlag; cc++) {
        const code = board[rr][cc];
        if (code === 0 || pieceSide(code) !== side) continue;
        const moves = this._engine.generatePieceMoves(cc, rr);
        for (const m of moves) {
          if (m.to[0] === c && m.to[1] === r) { protectedFlag = true; break; }
        }
      }
    }
    board[r][c] = saved;
    return protectedFlag;
  }

  /**
   * 给走法打启发式评分（不执行，内部 makeMove/undo 模拟）。
   * 综合：将军、吃子净收益、送死惩罚、威胁对方高分棋、出子/中心/过河微奖励。
   * 供红方启发式选走法，也供 RL 动作空间过滤"无意义走法"。
   */
  _scoreHeuristicMove(move, side) {
    let score = 0;
    const enemy = side === 'red' ? 'black' : 'red';
    const [fc, fr] = move.from;
    const [tc, tr] = move.to;
    const type = pieceType(move.piece);
    const pieceVal = PIECE_VALUES[move.piece] || 0;

    this._engine.makeMove(move);

    // 1. 将军：最高优先
    if (this._engine.isInCheck(enemy)) score += 8;

    // 2. 吃子净收益（考虑反吃风险）
    if (move.captured) {
      score += PIECE_VALUES[move.captured] || 0;
    }

    // 3. 送死惩罚：移动到被攻击且无保护格
    const attackedByEnemy = this._isSquareAttacked(side, tc, tr);
    const protectedBySelf = this._isProtectedBySelf(side, tc, tr);
    if (attackedByEnemy && !protectedBySelf) {
      score -= pieceVal * 0.8;
    }

    // 4. 威胁对方高分棋（车/马/炮/将/卒）
    const threatMoves = this._engine.generatePieceMoves(tc, tr);
    for (const tm of threatMoves) {
      const target = this._engine.board[tm.to[1]][tm.to[0]];
      if (target !== 0 && pieceSide(target) === enemy) {
        score += (PIECE_VALUES[target] || 0) * 0.15;
      }
    }

    // 5. 出子奖励（车/马/炮离开原位）
    if (type >= 4 && type <= 6) {
      const startRow = side === 'red' ? 9 : 0;
      if (move.from[1] === startRow) score += 0.8;
    }
    // 6. 中心控制
    if (tc >= 3 && tc <= 5) score += 0.3;
    // 7. 兵/卒过河
    if (type === 7) {
      const crossed = side === 'red' ? tr <= 4 : tr >= 5;
      if (crossed) score += 0.5;
    }
    // 8. 前进奖励：棋子向对方半场移动（减少无意义的来回闲走）
    const forward = side === 'red' ? -1 : 1;
    if ((tr - fr) * forward > 0) score += 0.4;

    this._engine.undoMove();
    return score;
  }

  _rlGetSortedMoves() {
    const side = this._engine.turn;
    let moves = this._engine.generateAllMoves(side);
    // 过滤明显送死/无意义走法，缩小动作空间、减少无效探索
    const filtered = moves.filter(m => this._scoreHeuristicMove(m, side) > -8);
    if (filtered.length > 0) moves = filtered;
    // 按 from/to 方格排序，保证一致性
    moves.sort((a, b) => {
      const fa = a.from[1] * 9 + a.from[0];
      const fb = b.from[1] * 9 + b.from[0];
      if (fa !== fb) return fa - fb;
      const ta = a.to[1] * 9 + a.to[0];
      const tb = b.to[1] * 9 + b.to[0];
      return ta - tb;
    });
    return moves;
  }

  _rlRobotMove() {
    const agent = this._rlEnsureAgent();
    if (!agent) { this._heuristicMove('black'); return; }

    const moves = this._rlGetSortedMoves();
    if (moves.length === 0) {
      this._engine.checkGameEnd();
      if (this._engine.gameOver) this._onGameEnd();
      return;
    }

    // 编码状态
    const state = this._rlEncodeState();
    const validActions = moves.map((_, i) => i);

    // 训练早期混合启发式（行为引导）：随训练推进逐步交给 RL 接管。
    // 启发式走法同样作为真实经验入库，让网络学会模仿合理棋型。
    let move = null, moveIdx = -1;
    const progress = Math.min(this._rl.trainSteps / 6000, 1);
    if (Math.random() < (1 - progress) * 0.6) {
      const hm = this._pickHeuristicMove('black');
      if (hm) {
        moveIdx = moves.findIndex(m =>
          m.from[0] === hm.from[0] && m.from[1] === hm.from[1] &&
          m.to[0] === hm.to[0] && m.to[1] === hm.to[1]);
        if (moveIdx >= 0) move = moves[moveIdx];
      }
    }
    if (!move) {
      const { action } = agent.chooseAction(state, validActions);
      moveIdx = Math.min(action, moves.length - 1);
      move = moves[moveIdx];
    }

    // 保存 pending 经验
    this._rl.pendingState = state;
    this._rl.pendingAction = moveIdx;
    this._rl.pendingReward = this._rlCalcImmediateReward(move, 'black');
    this._rl.trainSteps++;

    // 检查游戏是否在黑方走完后结束
    this._executeMove(move, () => {
      this._lastMove = move;
      this._showLastMoveHighlight(move);
      this._moveCount++;

      if (this._engine.checkGameEnd()) {
        this._onGameEnd();
        return;
      }

      // 如果不是自动对弈，等待玩家走法
      if (!this._rl.autoPlay) {
        this._updateStatusBar();
        this._updatePanelUI();
      }
    });
  }

  _rlOnOpponentMove(opponentMove) {
    if (this._rl.pendingState === null) return;

    // 计算对手走法的奖励（从黑方视角）
    const oppReward = this._rlCalcImmediateReward(opponentMove, 'red');
    this._rl.pendingReward -= oppReward; // 对手的收益是黑方的损失

    // 判断对手走法是否会直接终结对局（将死/困毙/和棋）
    let done = this._engine.gameOver;
    if (!done) {
      try {
        this._engine.makeMove(opponentMove);
        if (this._engine.checkGameEnd()) done = true;
        this._engine.undoMove();
      } catch (e) { /* 忽略 */ }
    }

    // 对手终结对局时，把终局奖励计入该转移（将死=-100 / 平局=-10）
    if (done) {
      const w = this._engine.winner;
      this._rl.pendingReward += w === 'black' ? RL_REWARD.checkmate
        : w === 'red' ? RL_REWARD.lose
        : RL_REWARD.draw;
    }

    const nextState = this._rlEncodeState();

    try {
      this._rl.agent.store(
        this._rl.pendingState, this._rl.pendingAction,
        this._rl.pendingReward, nextState, done,
      );
      // 每步训练多次，提高样本利用率，加速收敛
      for (let i = 0; i < 3; i++) this._rl.agent.train();
    } catch (e) {
      console.warn('[象棋] RL store/train 失败:', e);
    }
    // 注意：endEpisode 由 _onGameEnd 统一结算（避免对局数重复计数）

    this._rl.pendingState = null;
    this._rl.pendingAction = null;
    this._rl.pendingReward = 0;
  }

  _rlCalcImmediateReward(move, side) {
    let r = RL_REWARD.step;
    const enemy = side === 'red' ? 'black' : 'red';
    const [fc, fr] = move.from;
    const [tc, tr] = move.to;
    // 吃子奖励
    if (move.captured) {
      r += (PIECE_VALUES[move.captured] || 0) * RL_REWARD.capture_mult;
    }

    this._engine.makeMove(move);

    // 将军奖励 / 被将军惩罚
    if (this._engine.isInCheck(enemy)) r += RL_REWARD.check;
    if (this._engine.isInCheck(side)) r += RL_REWARD.in_check;

    // 送死惩罚：移动子暴露在对方攻击下且无己方保护
    const pieceVal = PIECE_VALUES[move.piece] || 0;
    if (pieceVal > 0 && this._isSquareAttacked(side, tc, tr) && !this._isProtectedBySelf(side, tc, tr)) {
      r -= pieceVal * RL_REWARD.lose_piece;
    }

    // 威胁奖励：走后能威胁对方高分棋（车/马/炮/将/卒）
    const threatMoves = this._engine.generatePieceMoves(tc, tr);
    for (const tm of threatMoves) {
      const target = this._engine.board[tm.to[1]][tm.to[0]];
      if (target !== 0 && pieceSide(target) === enemy) {
        r += (PIECE_VALUES[target] || 0) * RL_REWARD.threat;
      }
    }

    // 出子奖励（棋子离开初始位置）
    const type = pieceType(move.piece);
    if (type >= 4 && type <= 6) { // 车/马/炮出子
      const startRow = side === 'red' ? 9 : 0;
      if (move.from[1] === startRow) r += RL_REWARD.develop;
    }
    // 中心控制
    if (tc >= 3 && tc <= 5) r += RL_REWARD.center_control;

    this._engine.undoMove();
    return r;
  }

  _autoPlayStep() {
    if (this._engine.gameOver) {
      return; // 新局由 _onGameEnd 的自动开新局定时器统一负责
    }

    if (this._engine.turn === 'black') {
      // RL 机器人走法
      if (this._rl.enabled) {
        this._rlRobotMove();
      } else {
        this._heuristicMove('black');
      }
    } else {
      // 红方：启发式 AI
      const move = this._pickHeuristicMove('red');
      if (!move) {
        this._engine.checkGameEnd();
        if (this._engine.gameOver) this._onGameEnd();
        return;
      }
      // 关键：结算黑方 RL 上一手的经验（必须先于走子，_rlOnOpponentMove 内部会 makeMove/undo 计算奖励）
      if (this._rl.enabled && this._rl.pendingState !== null) {
        this._rlOnOpponentMove(move);
      }
      this._executeMove(move, () => {
        this._lastMove = move;
        this._showLastMoveHighlight(move);
        this._moveCount++;
        if (this._engine.checkGameEnd()) {
          this._onGameEnd();
          return;
        }
        if (this._rl.speed < 10) {
          this._updateStatusBar();
          this._updatePanelUI();
        }
      });
    }
  }

  /** 只挑选启发式走法（不执行），供自动对弈红方先结算 RL 经验再走子 */
  _pickHeuristicMove(side) {
    const moves = this._engine.generateAllMoves(side);
    if (moves.length === 0) return null;

    // 评分制选走法：先排除白送高价值子的走法，再从高分走法中带扰动挑选
    const scored = [];
    for (const m of moves) {
      const s = this._scoreHeuristicMove(m, side);
      if (s <= -6) continue; // 明显送死/无意义，直接排除
      scored.push({ m, s: s + Math.random() * 1.2 }); // 加扰动避免重复单调
    }
    const pool = scored.length > 0 ? scored : moves.map(m => ({ m, s: 0 }));
    pool.sort((a, b) => b.s - a.s);
    // 明确优势走法（如将军、吃高价值子）直接采用，避免随机错过
    if (pool.length > 1 && pool[0].s >= 5 && pool[0].s - pool[1].s >= 2) {
      return pool[0].m;
    }
    const topN = Math.min(5, pool.length);
    return pool[Math.floor(Math.random() * topN)].m;
  }

  _heuristicMove(side) {
    const move = this._pickHeuristicMove(side);
    if (!move) {
      this._engine.checkGameEnd();
      if (this._engine.gameOver) this._onGameEnd();
      return null;
    }

    this._executeMove(move, () => {
      this._lastMove = move;
      this._showLastMoveHighlight(move);
      this._moveCount++;
      if (this._engine.checkGameEnd()) {
        this._onGameEnd();
      }
    });

    return move;
  }

  // ==================== RL 契约接口 ====================

  getObservationSpec() {
    const spec = [{ name: 'board', kind: 'grid', shape: [BOARD_COLS * BOARD_ROWS], scale: 14, offset: 0 }];
    const extras = [
      'mat_diff', 'r_king_c', 'r_king_r', 'b_king_c', 'b_king_r',
      'king_dist', 'red_moves', 'black_moves', 'cur_check', 'enemy_check',
      'phase', 'turn', 'red_count', 'black_count', 'last_capture',
    ];
    for (const n of extras) spec.push({ name: n, kind: 'scalar', scale: 1, offset: 0 });
    return spec;
  }

  getActionSpec() {
    return Array.from({ length: MAX_ACTIONS }, (_, i) => ({
      id: i, name: `move_${i}`, semantics: 'semantic', executable: true,
    }));
  }

  getObservation() { return this._rlEncodeState(); }

  applyAction(actionId) {
    const moves = this._rlGetSortedMoves();
    if (actionId < 0 || actionId >= moves.length) return false;
    this._executeMove(moves[actionId], () => {});
    return true;
  }

  rlStep(actionId) {
    const moves = this._rlGetSortedMoves();
    if (actionId < 0 || actionId >= moves.length) return null;
    const move = moves[actionId];
    const reward = this._rlCalcImmediateReward(move, this._engine.turn);
    this._engine.makeMove(move);
    this._moveCount++;
    const done = this._engine.checkGameEnd();
    return { obs: this._rlEncodeState(), reward, done, info: { move } };
  }

  getValidActions() {
    const moves = this._rlGetSortedMoves();
    return moves.map((_, i) => i);
  }

  rlDone() { return this._engine.gameOver; }

  rlReset() {
    this._engine.reset();
    this._syncAllPieces();
    this._moveCount = 0;
    return this._rlEncodeState();
  }

  getRLHyperparams() { return null; } // 使用 games-config 默认值

  // ==================== AI 助手（LLM） ====================

  _llmStepPolicy(dt) {
    if (!this._llm.enabled) return;
    this._llm.policyTimer += dt;
    if (this._llm.policyTimer >= this._llm.policyInterval && !this._llm.waitingResponse) {
      this._llm.policyTimer = 0;
      this._llmRequestAction();
    }
  }

  _llmBuildStateText() {
    const board = this._engine.board;
    const chars = { ...PIECE_NAMES, 0: '·' };
    let text = '';
    for (let r = 0; r < BOARD_ROWS; r++) {
      let row = '';
      for (let c = 0; c < BOARD_COLS; c++) {
        row += (chars[board[r][c]] || '·') + ' ';
      }
      if (r === 5) text += '  ── 楚河汉界 ──\n';
      text += row + '\n';
    }
    const turn = this._engine.turn === 'red' ? '红方' : '黑方(RL机器人)';
    const obsSide = this._observerSide === 'red' ? '红方' : '黑方';
    const step = this._engine.moveHistory.length;
    const inCheck = this._engine.isInCheck(this._engine.turn);
    const redMat = this._engine.materialValue('red');
    const blackMat = this._engine.materialValue('black');
    const lastMove = this._engine.moveHistory.length > 0
      ? this._engine.moveToString(this._engine.moveHistory[this._engine.moveHistory.length - 1])
      : '无';
    return `你是一位观战AI，正立于${obsSide}一侧观战，从${obsSide}视角分析局势。当前是${turn}回合，第${step}步。${inCheck ? '⚠️当前方被将军！' : ''}\n` +
      `材料对比：红方${redMat} vs 黑方${blackMat}。\n` +
      `上一步走法：${lastMove}。\n` +
      `当前棋盘快照：\n${text}`;
  }

  _llmBuildStateKey() {
    const redMat = this._engine.materialValue('red');
    const blackMat = this._engine.materialValue('black');
    const diff = redMat - blackMat;
    const diffB = diff > 10 ? 2 : diff > -10 ? 1 : 0;
    const totalP = this._engine.pieceCount('red') + this._engine.pieceCount('black');
    const phase = totalP > 28 ? 0 : totalP > 16 ? 1 : 2;
    const inCheck = this._engine.isInCheck(this._engine.turn) ? 1 : 0;
    return `${diffB}|${phase}|${inCheck}|${this._moveCount}`;
  }

  _llmRequestAction() {
    if (!this.App.ws || this.App.ws.readyState !== WebSocket.OPEN) return;
    const stateText = this._llmBuildStateText();
    const stateKey = this._llmBuildStateKey();
    this._llm.waitingResponse = true;
    this._llm.stats.requests++;
    this.App.ws.send(JSON.stringify({
      type: 'game_action_request',
      data: {
        game_type: 'xiangqi',
        state_text: stateText,
        state_key: stateKey,
        candidates: STRATEGY_CANDIDATES,
        last_reward: 0,
      },
    }));
  }

  onLLMActionResponse(data) {
    this._llm.waitingResponse = false;
    this._llm.stats.responses++;
    if (!data || !data.strategy) return;
    this._llm.strategy = data.strategy;
    this._llm.lastAdvice = data.reason || '';
    this._llmUpdateStrategyUI(data.strategy, data.speak || '', data.reason || '');
  }

  _llmUpdateStrategyUI(strategy, speak, reason) {
    const el = document.getElementById('xiangqi-llm-strategy');
    if (el) {
      el.textContent = STRATEGY_LABELS[strategy] || strategy;
    }
    const adviceEl = document.getElementById('xiangqi-llm-advice');
    if (adviceEl && reason) {
      adviceEl.textContent = reason;
    }
    const speakEl = document.getElementById('xiangqi-llm-speak');
    if (speakEl && speak) {
      speakEl.textContent = `💬 ${speak}`;
    }
  }

  // ==================== RL 面板 UI ====================

  _createRLPanel() {
    if (this._panel) return;
    const panel = document.createElement('div');
    panel.id = 'xiangqi-rl-panel';
    panel.style.cssText = `
      position:fixed; right:16px; top:90px; width:260px; z-index:9999;
      background:rgba(10,18,30,0.97);
      border:1px solid rgba(124,92,255,0.4); border-radius:12px;
      box-shadow:0 8px 32px rgba(0,0,0,0.5); color:#e0e0ff;
      font-family:system-ui,sans-serif; font-size:13px; user-select:none;
    `;
    panel.innerHTML = `
      <div id="xiangqi-rl-title" style="cursor:move;padding:10px 14px;border-bottom:1px solid rgba(124,92,255,0.2);display:flex;justify-content:space-between;align-items:center;">
        <span>🧠 象棋RL训练</span>
        <span id="xiangqi-rl-close" style="cursor:pointer;font-size:16px;opacity:0.6;">×</span>
      </div>
      <div style="padding:10px 14px;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;margin-bottom:8px;">
          <div>RL决策: <span id="xiangqi-rl-status" style="color:#ff6b6b;">关闭</span></div>
          <div>对局数: <span id="xiangqi-rl-episodes">0</span></div>
          <div>步数: <span id="xiangqi-rl-steps">0</span></div>
          <div>训练步: <span id="xiangqi-rl-train">0</span></div>
          <div>红胜: <span id="xiangqi-rl-redwin" style="color:#ff6666;">0</span></div>
          <div>黑胜: <span id="xiangqi-rl-blackwin" style="color:#6666ff;">0</span></div>
          <div>平局: <span id="xiangqi-rl-draw" style="color:#cccccc;">0</span></div>
          <div>观战方: <span id="xiangqi-observer-side">-</span></div>
          <div>倍速: <span id="xiangqi-rl-speed">1x</span></div>
          <div>自动对弈: <span id="xiangqi-rl-auto" style="color:#666;">否</span></div>
        </div>
        <div style="margin-bottom:4px;">AI指挥: <span id="xiangqi-rl-llm" style="color:#666;">关闭</span></div>
        <div style="margin-bottom:4px;">当前策略: <span id="xiangqi-llm-strategy" style="color:#00e5ff;">-</span></div>
        <div id="xiangqi-llm-advice" style="font-size:11px;color:#888;margin-bottom:4px;min-height:16px;line-height:1.3;"></div>
        <div id="xiangqi-llm-speak" style="font-size:11px;color:#7c5cff;margin-bottom:8px;min-height:14px;"></div>
        <div style="display:flex;flex-wrap:wrap;gap:4px;">
          <button id="xiangqi-btn-rl" style="flex:1;min-width:70px;padding:5px;border:none;border-radius:6px;background:rgba(124,92,255,0.3);color:#e0e0ff;cursor:pointer;font-size:12px;">开启RL</button>
          <button id="xiangqi-btn-speed" style="flex:1;min-width:70px;padding:5px;border:none;border-radius:6px;background:rgba(0,229,255,0.2);color:#e0e0ff;cursor:pointer;font-size:12px;">倍速</button>
          <button id="xiangqi-btn-auto" style="flex:1;min-width:70px;padding:5px;border:none;border-radius:6px;background:rgba(0,229,255,0.2);color:#e0e0ff;cursor:pointer;font-size:12px;">自动对弈</button>
          <button id="xiangqi-btn-llm" style="flex:1;min-width:70px;padding:5px;border:none;border-radius:6px;background:rgba(0,229,255,0.2);color:#e0e0ff;cursor:pointer;font-size:12px;">AI指挥</button>
          <button id="xiangqi-btn-new" style="flex:1;min-width:70px;padding:5px;border:none;border-radius:6px;background:rgba(255,170,0,0.2);color:#e0e0ff;cursor:pointer;font-size:12px;">新局</button>
          <button id="xiangqi-btn-reset" style="flex:1;min-width:70px;padding:5px;border:none;border-radius:6px;background:rgba(255,100,100,0.2);color:#e0e0ff;cursor:pointer;font-size:12px;">重置</button>
          <button id="xiangqi-btn-save" style="flex:1;min-width:70px;padding:5px;border:none;border-radius:6px;background:rgba(0,255,136,0.2);color:#e0e0ff;cursor:pointer;font-size:12px;">保存</button>
        </div>
      </div>
    `;
    document.body.appendChild(panel);
    this._panel = panel;

    // 关闭按钮
    panel.querySelector('#xiangqi-rl-close').addEventListener('click', () => {
      panel.style.display = 'none';
    });

    // 按钮事件
    panel.querySelector('#xiangqi-btn-rl').addEventListener('click', () => this._toggleRL());
    panel.querySelector('#xiangqi-btn-speed').addEventListener('click', () => this._cycleSpeed());
    panel.querySelector('#xiangqi-btn-auto').addEventListener('click', () => this._toggleAutoPlay());
    panel.querySelector('#xiangqi-btn-llm').addEventListener('click', () => this._toggleLLM());
    panel.querySelector('#xiangqi-btn-new').addEventListener('click', () => this._newGame());
    panel.querySelector('#xiangqi-btn-reset').addEventListener('click', () => this._resetRL());
    panel.querySelector('#xiangqi-btn-save').addEventListener('click', () => this._saveRL());

    // 拖动
    this._initPanelDrag();
  }

  _initPanelDrag() {
    const panel = this._panel;
    const title = panel.querySelector('#xiangqi-rl-title');
    let dragging = false, ox = 0, oy = 0, centered = false;

    const onStart = (e) => {
      if (e.target.closest('#xiangqi-rl-close')) return;
      dragging = true;
      const pt = e.touches ? e.touches[0] : e;
      ox = pt.clientX - panel.offsetLeft;
      oy = pt.clientY - panel.offsetTop;
      if (centered) {
        panel.style.transform = 'none';
        panel.style.left = panel.offsetLeft + 'px';
        panel.style.top = panel.offsetTop + 'px';
        centered = false;
      }
      e.preventDefault();
    };
    const onMove = (e) => {
      if (!dragging) return;
      const pt = e.touches ? e.touches[0] : e;
      let nx = pt.clientX - ox;
      let ny = pt.clientY - oy;
      const w = panel.offsetWidth;
      nx = Math.max(-w / 2 + 20, Math.min(window.innerWidth - 20, nx));
      ny = Math.max(0, Math.min(window.innerHeight - 30, ny));
      panel.style.left = nx + 'px';
      panel.style.top = ny + 'px';
    };
    const onEnd = () => { dragging = false; };

    title.addEventListener('mousedown', onStart);
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onEnd);
    title.addEventListener('touchstart', onStart, { passive: false });
    document.addEventListener('touchmove', onMove, { passive: false });
    document.addEventListener('touchend', onEnd);

    this._dragCleanup = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onEnd);
      document.removeEventListener('touchmove', onMove);
      document.removeEventListener('touchend', onEnd);
    };
  }

  _updatePanelUI() {
    const p = this._panel;
    if (!p) return;
    const setEl = (id, val) => {
      const el = p.querySelector(`#${id}`);
      if (el && val !== undefined) el.textContent = val;
      return el;
    };
    setEl('xiangqi-rl-status', this._rl.enabled ? '✅ 已开启' : '关闭');
    const statusEl = setEl('xiangqi-rl-status');
    if (statusEl) statusEl.style.color = this._rl.enabled ? '#00ff88' : '#ff6b6b';
    setEl('xiangqi-rl-episodes', this._rl.episodes);
    setEl('xiangqi-rl-steps', this._moveCount);
    setEl('xiangqi-rl-train', this._rl.trainSteps);
    setEl('xiangqi-rl-redwin', this._rl.wins.red);
    setEl('xiangqi-rl-blackwin', this._rl.wins.black);
    setEl('xiangqi-rl-draw', this._rl.wins.draw || 0);
    const obsEl = setEl('xiangqi-observer-side', this._observerSide === 'red' ? '🔴 红方' : '⚫ 黑方');
    if (obsEl) obsEl.style.color = this._observerSide === 'red' ? '#ff6666' : '#6666ff';
    setEl('xiangqi-rl-speed', this._rl.speed + 'x');
    setEl('xiangqi-rl-auto', this._rl.autoPlay ? '✅ 是' : '否');
    const autoStatusEl = p.querySelector('#xiangqi-rl-auto');
    if (autoStatusEl) autoStatusEl.style.color = this._rl.autoPlay ? '#00ff88' : '#666';
    setEl('xiangqi-rl-llm', this._llm.enabled ? '✅ 已开启' : '关闭');
    const llmStatusEl = p.querySelector('#xiangqi-rl-llm');
    if (llmStatusEl) llmStatusEl.style.color = this._llm.enabled ? '#00ff88' : '#666';

    // 更新按钮文字
    const rlBtn = p.querySelector('#xiangqi-btn-rl');
    rlBtn.textContent = this._rl.enabled ? '关闭RL' : '开启RL';
    rlBtn.style.background = this._rl.enabled ? 'rgba(255,100,100,0.3)' : 'rgba(124,92,255,0.3)';
    const autoBtn = p.querySelector('#xiangqi-btn-auto');
    autoBtn.textContent = this._rl.autoPlay ? '停止自动' : '自动对弈';
    autoBtn.style.background = this._rl.autoPlay ? 'rgba(255,170,0,0.3)' : 'rgba(0,229,255,0.2)';
    const llmBtn = p.querySelector('#xiangqi-btn-llm');
    llmBtn.textContent = this._llm.enabled ? '关闭指挥' : 'AI指挥';
    llmBtn.style.background = this._llm.enabled ? 'rgba(124,92,255,0.3)' : 'rgba(0,229,255,0.2)';
  }

  _toggleRL() {
    this._rl.enabled = !this._rl.enabled;
    if (this._rl.enabled) {
      this._rlEnsureAgent();
      this._newGame(); // 开启 RL 自动开始新对局
    }
    this._updatePanelUI();
  }

  _cycleSpeed() {
    const idx = SPEED_PRESETS.indexOf(this._rl.speed);
    this._rl.speed = SPEED_PRESETS[(idx + 1) % SPEED_PRESETS.length];
    this._updatePanelUI();
  }

  _toggleAutoPlay() {
    this._rl.autoPlay = !this._rl.autoPlay;
    if (this._rl.autoPlay) {
      if (this._rl.enabled) this._rlEnsureAgent();
      this._newGame();
    }
    this._updatePanelUI();
  }

  _toggleLLM() {
    this._llm.enabled = !this._llm.enabled;
    this._updatePanelUI();
  }

  _resetRL() {
    try {
      RLAgentManager.get().resetAgent('xiangqi');
      this._rl.agent = null;
      this._rl.episodes = 0;
      this._rl.wins = { red: 0, black: 0, draw: 0 };
      this._rl.trainSteps = 0;
      this._newGame();
      console.log('[象棋] RL 已重置');
    } catch (e) {
      console.warn('[象棋] RL 重置失败:', e);
    }
    this._updatePanelUI();
  }

  _saveRL() {
    if (this._rl.agent) {
      this._rl.agent.flush();
      console.log('[象棋] RL 已保存');
    }
    this._updatePanelUI();
  }

  // ==================== 状态栏 ====================

  _createStatusBar() {
    if (this._statusBar) return;
    const bar = document.createElement('div');
    bar.id = 'xiangqi-status';
    bar.style.cssText = `
      position:fixed; top:56px; left:50%; transform:translateX(-50%);
      z-index:9998; background:rgba(10,18,30,0.9);
      border:1px solid rgba(124,92,255,0.3); border-radius:20px;
      padding:6px 18px; color:#e0e0ff; font-size:13px;
      font-family:system-ui,sans-serif; white-space:nowrap;
      backdrop-filter:blur(10px);
    `;
    document.body.appendChild(bar);
    this._statusBar = bar;
    this._updateStatusBar();
  }

  _setStatus(text) {
    if (this._statusBar) this._statusBar.textContent = text;
  }

  _updateStatusBar() {
    if (!this._statusBar) return;
    if (this._engine.gameOver) return; // 游戏结束时由 _onGameEnd 设置
    const turn = this._engine.turn === 'red' ? '🔴 红方(你)' : '⚫ 黑方(RL)';
    const obs = this._observerSide === 'red' ? '红方' : '黑方';
    const check = this._engine.isInCheck(this._engine.turn) ? ' ⚠️ 将军!' : '';
    const lastMove = this._lastMove ? ` | 上一步: ${this._engine.moveToString(this._lastMove)}` : '';
    this._setStatus(`${turn}回合 | 观战: ${obs}侧 | 步数: ${this._moveCount}${check}${lastMove}`);
  }

  // ==================== 引擎兼容 ====================

  checkCollision() { return false; } // 允许角色在棋盘附近自由走动
  setPlayerSpeed(speed) { this._playerSpeed = speed; }

  getExtraState() {
    return {
      turn: this._engine.turn,
      observer_side: this._observerSide,
      move_count: this._moveCount,
      in_check: this._engine.isInCheck(this._engine.turn),
      game_over: this._engine.gameOver,
      winner: this._engine.winner,
      red_material: this._engine.materialValue('red'),
      black_material: this._engine.materialValue('black'),
    };
  }

  getPerceptionData() {
    const pos = this._getPlayerPosition() || { x: 0, y: 0, z: 0 };
    return {
      game_type: this.name,
      game_name: this.displayName,
      state: this.state,
      score: this._moveCount,
      elapsed_sec: Math.floor(this.elapsedTime),
      player: {
        x: pos.x, y: pos.y, z: pos.z,
        facing: this._getPlayerFacing(),
        speed: this._playerSpeed || 0,
      },
      board: {
        type: 'xiangqi',
        cols: BOARD_COLS,
        rows: BOARD_ROWS,
        cells: this._engine.board,
        turn: this._engine.turn,
        observer_side: this._observerSide,
        in_check: this._engine.isInCheck(this._engine.turn),
        move_count: this._moveCount,
        last_move: this._lastMove,
        game_over: this._engine.gameOver,
        winner: this._engine.winner,
      },
      objects: {},
      nearby: [],
      progress: this.getExtraState(),
      recent_events: this._getRecentEvents(),
    };
  }
}

export default XiangqiGame;
