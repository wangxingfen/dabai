/* ============================================================
 * 3D 场景模块 —— 棋盘/棋子渲染 + 第一人称探索控制器
 *
 * 布局约定（与规则引擎一致）：
 *   col 0..8  → x = col - 4          （红方底在 +z 侧）
 *   row 0..9  → z = row - 4.5        （黑方底在 -z 侧）
 * ============================================================ */
import * as THREE from 'three';
import { PIECE_NAMES, BOARD_COLS, BOARD_ROWS, pieceSide, isRed } from './engine.js';

const CELL = 1.0;
const PIECE_R = 0.38;
const PIECE_H = 0.18;
const PIECE_Y = 0.842;         // 棋子底面高度（棋盘表面）
const EYE_H = 1.7;             // 眼睛高度
const CHAR_R = 0.36;           // 角色碰撞半径
const MOVE_MS = 620;           // 走子动画时长

const PIECE_CHARS = {
  1: '帅', 2: '仕', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
  8: '将', 9: '士', 10: '象', 11: '马', 12: '车', 13: '炮', 14: '卒',
};

export class GameScene {
  constructor(canvas, engine) {
    this.canvas = canvas;
    this.engine = engine;
    this.keys = {};
    this.yaw = 0;
    this.pitch = -0.12;
    this.pos = new THREE.Vector3(0, 0, 7.2);
    this.bobPhase = 0;
    this.locked = false;
    this.animations = [];   // 走子动画
    this.effects = [];      // 淡出/缩放特效
    this.pieceMeshes = new Map(); // key "c,r" -> mesh
    this.colliders = [];    // {c,r,x,z,r,mesh}
    this.lastMove = null;
    this.checkMesh = null;
    this.callbacks = { onMoveDone: null, onLookChange: null };
    this.mySide = 'red';
    this._time = 0;
  }

  // ==================== 初始化 ====================
  init() {
    const canvas = this.canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x8fb8d8);
    this.scene.fog = new THREE.Fog(0x8fb8d8, 34, 90);

    this.camera = new THREE.PerspectiveCamera(72, window.innerWidth / window.innerHeight, 0.08, 200);
    this.camera.rotation.order = 'YXZ';

    // 灯光
    const hemi = new THREE.HemisphereLight(0xcfe6ff, 0x5a4a33, 0.85);
    this.scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xfff2d8, 1.5);
    sun.position.set(14, 22, 9);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -16; sun.shadow.camera.right = 16;
    sun.shadow.camera.top = 16; sun.shadow.camera.bottom = -16;
    sun.shadow.camera.far = 60;
    this.scene.add(sun);
    this.sun = sun;

    this._createGround();
    this._createBoard();
    this._createDecor();
    this._syncAllPieces();
    this._createCheckMarker();

    // 输入
    window.addEventListener('keydown', e => this._onKey(e, true));
    window.addEventListener('keyup', e => this._onKey(e, false));
    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
    canvas.addEventListener('click', () => {
      if (this.onCanvasClick) this.onCanvasClick();
      if (!this.locked) canvas.requestPointerLock();
    });
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === canvas;
      if (this.callbacks.onLookChange) this.callbacks.onLookChange(this.locked);
    });
    document.addEventListener('mousemove', e => {
      if (!this.locked) return;
      this.yaw -= e.movementX * 0.0021;
      this.pitch -= e.movementY * 0.0021;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch));
    });
    this.clock = new THREE.Clock();
  }

  // ==================== 地面 ====================
  _createGround() {
    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(120, 48),
      new THREE.MeshStandardMaterial({ color: 0x76926b, roughness: 1 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.02;
    ground.receiveShadow = true;
    this.scene.add(ground);
  }

  // ==================== 棋盘 ====================
  _createBoard() {
    // 木质底座（顶面高度 0.80）
    const base = new THREE.Mesh(
      new THREE.BoxGeometry(9.7, 0.8, 10.7),
      new THREE.MeshStandardMaterial({ color: 0x6b4a24, roughness: 0.85 })
    );
    base.position.y = 0.4;
    base.castShadow = true;
    base.receiveShadow = true;
    this.scene.add(base);

    // 棋盘面板（顶面高度 0.84）
    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(8.1, 0.07, 9.1),
      new THREE.MeshStandardMaterial({ color: 0xc79a54, roughness: 0.7 })
    );
    slab.position.y = 0.805;
    slab.castShadow = true;
    slab.receiveShadow = true;
    this.scene.add(slab);

    // 棋盘线画布（y=0.842）
    const art = this._createBoardArt();
    const artMat = new THREE.MeshStandardMaterial({ map: art, roughness: 0.62 });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(8, 9), artMat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = 0.842;
    plane.receiveShadow = true;
    this.scene.add(plane);

    // 碰撞盒（角色不可踏上棋盘，按底座外沿）
    this.platformBox = { minX: -4.85, maxX: 4.85, minZ: -5.35, maxZ: 5.35 };
  }

  _createBoardArt() {
    const W = 1024, H = 1152, M = 56;
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    // 底色
    ctx.fillStyle = '#e6c383';
    ctx.fillRect(0, 0, W, H);
    // 木纹
    ctx.globalAlpha = 0.14;
    for (let i = 0; i < 40; i++) {
      const y = Math.random() * H;
      ctx.strokeStyle = '#7a5a2c';
      ctx.lineWidth = 1.5 + Math.random() * 3;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.bezierCurveTo(W * 0.3, y + 12, W * 0.6, y - 10, W, y + 8);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const dx = (W - 2 * M) / 8, dy = (H - 2 * M) / 9;
    const X = c => M + c * dx, Y = r => M + r * dy;

    ctx.strokeStyle = '#4a2f12';
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    // 外框
    ctx.strokeRect(X(0) - 6, Y(0) - 6, dx * 8 + 12, dy * 9 + 12);
    // 横线
    for (let r = 0; r < BOARD_ROWS; r++) {
      ctx.beginPath(); ctx.moveTo(X(0), Y(r)); ctx.lineTo(X(8), Y(r)); ctx.stroke();
    }
    // 竖线（河界处断开）
    for (let c = 0; c < BOARD_COLS; c++) {
      if (c === 0 || c === 8) {
        ctx.beginPath(); ctx.moveTo(X(c), Y(0)); ctx.lineTo(X(c), Y(9)); ctx.stroke();
      } else {
        ctx.beginPath(); ctx.moveTo(X(c), Y(0)); ctx.lineTo(X(c), Y(4)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(X(c), Y(5)); ctx.lineTo(X(c), Y(9)); ctx.stroke();
      }
    }
    // 九宫斜线
    const palace = (r0, r1) => {
      ctx.beginPath(); ctx.moveTo(X(3), Y(r0)); ctx.lineTo(X(5), Y(r1)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(5), Y(r0)); ctx.lineTo(X(3), Y(r1)); ctx.stroke();
    };
    palace(0, 2); palace(9, 7);
    // 河界文字
    ctx.fillStyle = '#5a3a16';
    ctx.font = `bold ${Math.round(dy * 0.62)}px "STKaiti","KaiTi","Microsoft YaHei",serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('楚  河', X(2), Y(4.5) + 8);
    ctx.fillText('漢  界', X(6), Y(4.5) + 8);
    // 炮位标记
    ctx.lineWidth = 4;
    for (const [c, r] of [[1, 2], [7, 2], [1, 7], [7, 7]]) {
      ctx.beginPath(); ctx.arc(X(c), Y(r), 12, 0, Math.PI * 2); ctx.stroke();
    }
    // 兵位标记
    for (const [c, r] of [[0, 3], [2, 3], [4, 3], [6, 3], [8, 3], [0, 6], [2, 6], [4, 6], [6, 6], [8, 6]]) {
      ctx.beginPath();
      ctx.moveTo(X(c) - 12, Y(r)); ctx.lineTo(X(c) + 12, Y(r));
      ctx.moveTo(X(c), Y(r) - 12); ctx.lineTo(X(c), Y(r) + 12);
      ctx.stroke();
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.anisotropy = 8;
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  // ==================== 棋子 ====================
  _pieceTextureCache = new Map();
  _pieceTexture(code) {
    if (this._pieceTextureCache.has(code)) return this._pieceTextureCache.get(code);
    const canvas = document.createElement('canvas');
    canvas.width = 256; canvas.height = 256;
    const ctx = canvas.getContext('2d');
    const red = isRed(code);
    const color = red ? '#c22f2f' : '#232323';
    const char = PIECE_CHARS[code] || '?';
    ctx.clearRect(0, 0, 256, 256);
    ctx.fillStyle = '#f6e6bd';
    ctx.beginPath(); ctx.arc(128, 128, 122, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 10;
    ctx.beginPath(); ctx.arc(128, 128, 104, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = 'bold 130px "Microsoft YaHei","SimHei","PingFang SC",sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(char, 128, 134);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    this._pieceTextureCache.set(code, tex);
    return tex;
  }

  _createPieceMesh(code) {
    const group = new THREE.Group();
    const side = pieceSide(code);
    const rimColor = side === 'red' ? 0xc22f2f : 0x2b2b2b;
    // 棋子身
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(PIECE_R, PIECE_R * 0.9, PIECE_H, 28),
      new THREE.MeshStandardMaterial({ color: 0xf0dcae, roughness: 0.55, metalness: 0.08 })
    );
    body.position.y = PIECE_H / 2;
    body.castShadow = true;
    group.add(body);
    // 顶部文字
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(PIECE_R * 0.99, 28),
      new THREE.MeshStandardMaterial({ map: this._pieceTexture(code), roughness: 0.5 })
    );
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = PIECE_H - 0.004;
    group.add(disc);
    // 侧面色环
    const rim = new THREE.Mesh(
      new THREE.CylinderGeometry(PIECE_R * 1.04, PIECE_R * 1.04, PIECE_H * 0.22, 28),
      new THREE.MeshStandardMaterial({ color: rimColor, roughness: 0.4 })
    );
    rim.position.y = PIECE_H * 0.08;
    group.add(rim);
    const band = new THREE.Mesh(
      new THREE.CylinderGeometry(PIECE_R * 1.04, PIECE_R * 1.04, PIECE_H * 0.18, 28),
      new THREE.MeshStandardMaterial({ color: rimColor, roughness: 0.4 })
    );
    band.position.y = PIECE_H * 0.9;
    group.add(band);
    group.userData = { code, isPiece: true };
    return group;
  }

  boardX(c) { return c - 4; }
  boardZ(r) { return r - 4.5; }

  _syncAllPieces() {
    // 清除旧棋子
    for (const mesh of this.pieceMeshes.values()) {
      this.scene.remove(mesh);
      mesh.traverse(o => { if (o.geometry) o.geometry.dispose(); });
    }
    this.pieceMeshes.clear();
    this.colliders = [];
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        const code = this.engine.get(c, r);
        if (code === 0) continue;
        const mesh = this._createPieceMesh(code);
        mesh.position.set(this.boardX(c), PIECE_Y + PIECE_H / 2, this.boardZ(r));
        this.scene.add(mesh);
        this.pieceMeshes.set(`${c},${r}`, mesh);
        this.colliders.push({ c, r, x: this.boardX(c), z: this.boardZ(r), r: PIECE_R + 0.06, mesh });
      }
    }
    this._updateLastMoveMarker();
    this._updateCheckMarker();
  }

  /** 依据引擎状态更新单个棋子位置（走子后调用） */
  updatePieceFromEngine(move) {
    const [fc, fr] = move.from;
    const [tc, tr] = move.to;
    const key = `${fc},${fr}`;
    const mesh = this.pieceMeshes.get(key);
    if (!mesh) return;
    // 先取出目标格上被吃的棋子（若存在）
    const capMesh = (move.captured !== 0) ? this.pieceMeshes.get(`${tc},${tr}`) : undefined;
    // 移动棋子到目标格
    mesh.position.set(this.boardX(tc), PIECE_Y + PIECE_H / 2, this.boardZ(tr));
    this.pieceMeshes.delete(key);
    this.pieceMeshes.set(`${tc},${tr}`, mesh);
    // 更新碰撞体（from -> to）
    const col = this.colliders.find(x => x.c === fc && x.r === fr);
    if (col) { col.c = tc; col.r = tr; col.x = this.boardX(tc); col.z = this.boardZ(tr); }
    // 被吃棋子淡出（并移除其碰撞体）
    if (capMesh && capMesh !== mesh) {
      this.pieceMeshes.delete(`${tc},${tr}`);
      this.colliders = this.colliders.filter(x => (x.c === tc && x.r === tr) ? x === col : true);
      this.effects.push({ type: 'fade', mesh: capMesh, t0: this._time, dur: 0.32 });
    }
  }

  // ==================== 走子动画 ====================
  animateMove(move) {
    const [fc, fr] = move.from;
    const [tc, tr] = move.to;
    const mesh = this.pieceMeshes.get(`${fc},${fr}`);
    if (!mesh) return;
    this.animations.push({
      move, mesh,
      from: new THREE.Vector3(this.boardX(fc), PIECE_Y + PIECE_H / 2, this.boardZ(fr)),
      to: new THREE.Vector3(this.boardX(tc), PIECE_Y + PIECE_H / 2, this.boardZ(tr)),
      t0: this._time, dur: MOVE_MS / 1000,
    });
    this.lastMove = move;
    this._updateLastMoveMarker();
  }

  // ==================== 标记 ====================
  _createCheckMarker() {
    const geo = new THREE.RingGeometry(0.34, 0.5, 32);
    const mat = new THREE.MeshBasicMaterial({ color: 0xff3b30, transparent: true, opacity: 0.75, side: THREE.DoubleSide });
    this.checkMesh = new THREE.Mesh(geo, mat);
    this.checkMesh.rotation.x = -Math.PI / 2;
    this.checkMesh.visible = false;
    this.checkMesh.position.y = PIECE_Y + 0.02;
    this.scene.add(this.checkMesh);
  }

  _updateCheckMarker() {
    if (!this.checkMesh) return;
    const inCheck = this._inCheckSide;
    if (!inCheck) { this.checkMesh.visible = false; return; }
    const king = this.engine.findKing(inCheck);
    if (!king) { this.checkMesh.visible = false; return; }
    this.checkMesh.visible = true;
    this.checkMesh.position.x = this.boardX(king[0]);
    this.checkMesh.position.z = this.boardZ(king[1]);
  }

  _updateLastMoveMarker() {
    // 最近一步高亮
    if (this._lastMarkers) for (const m of this._lastMarkers) { this.scene.remove(m); m.material.dispose(); m.geometry.dispose(); }
    this._lastMarkers = [];
    if (!this.lastMove) return;
    const geo = new THREE.PlaneGeometry(0.88, 0.88);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffd75e, transparent: true, opacity: 0.5 });
    for (const [c, r] of [this.lastMove.from, this.lastMove.to]) {
      const m = new THREE.Mesh(geo, mat.clone());
      m.rotation.x = -Math.PI / 2;
      m.position.set(this.boardX(c), PIECE_Y + 0.015, this.boardZ(r));
      this.scene.add(m);
      this._lastMarkers.push(m);
    }
  }

  setCheckSide(side) { this._inCheckSide = side; }

  // ==================== 附身与出生点 ====================
  spawn(side) {
    this.mySide = side || (Math.random() < 0.5 ? 'red' : 'black');
    const redSide = this.mySide === 'red';
    const z = redSide ? 7.1 : -7.1;
    this.pos.set((Math.random() * 4 - 2), 0, z + (Math.random() * 0.6 - 0.3));
    this.yaw = redSide ? Math.PI : 0;
    this.pitch = -0.1;
    this._applyCamera(true);
  }

  // ==================== 第一人称控制器 ====================
  _onKey(e, down) {
    if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
    this.keys[e.code] = down;
  }

  _applyCamera(teleport) {
    const cam = this.camera;
    cam.rotation.set(this.pitch, this.yaw, 0);
    cam.position.set(this.pos.x, this.pos.y + EYE_H, this.pos.z);
    if (teleport) cam.position.y = EYE_H;
  }

  _collide(px, pz) {
    // 世界边界
    px = Math.max(-16, Math.min(16, px));
    pz = Math.max(-16, Math.min(16, pz));
    // 棋盘平台（AABB 圆推挤）
    const b = this.platformBox;
    if (px > b.minX - CHAR_R && px < b.maxX + CHAR_R && pz > b.minZ - CHAR_R && pz < b.maxZ + CHAR_R) {
      const cx = Math.max(b.minX, Math.min(b.maxX, px));
      const cz = Math.max(b.minZ, Math.min(b.maxZ, pz));
      const dx = px - cx, dz = pz - cz;
      const d2 = dx * dx + dz * dz;
      if (d2 < CHAR_R * CHAR_R) {
        if (d2 > 1e-6) {
          const d = Math.sqrt(d2);
          px = cx + dx / d * CHAR_R;
          pz = cz + dz / d * CHAR_R;
        } else {
          // 站在平台正中间（异常情况）向上推
          pz = b.maxZ + CHAR_R;
        }
      }
    }
    // 棋子碰撞（圆-圆推挤）
    for (const col of this.colliders) {
      const dx = px - col.x, dz = pz - col.z;
      const min = col.r + CHAR_R;
      const d2 = dx * dx + dz * dz;
      if (d2 < min * min && d2 > 1e-8) {
        const d = Math.sqrt(d2);
        px = col.x + dx / d * min;
        pz = col.z + dz / d * min;
      }
    }
    return [px, pz];
  }

  update(dt) {
    this._time += dt;
    // 走子动画
    for (let i = this.animations.length - 1; i >= 0; i--) {
      const a = this.animations[i];
      const t = Math.min(1, (this._time - a.t0) / a.dur);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // easeInOut
      a.mesh.position.lerpVectors(a.from, a.to, e);
      a.mesh.position.y = PIECE_Y + PIECE_H / 2 + Math.sin(t * Math.PI) * 0.38;
      if (t >= 1) {
        this.animations.splice(i, 1);
        this.updatePieceFromEngine(a.move);
        if (this.callbacks.onMoveDone) this.callbacks.onMoveDone(a.move);
      }
    }
    // 特效
    for (let i = this.effects.length - 1; i >= 0; i--) {
      const fx = this.effects[i];
      const t = Math.min(1, (this._time - fx.t0) / fx.dur);
      const s = 1 - t;
      fx.mesh.scale.setScalar(Math.max(0.01, s));
      fx.mesh.position.y = PIECE_Y + PIECE_H / 2 + t * 0.15;
      if (t >= 1) {
        this.scene.remove(fx.mesh);
        fx.mesh.traverse(o => { if (o.geometry) o.geometry.dispose(); });
        this.effects.splice(i, 1);
      }
    }
    // 被将军标记脉冲
    if (this.checkMesh && this.checkMesh.visible) {
      const pulse = 0.62 + Math.sin(this._time * 5) * 0.18;
      this.checkMesh.scale.setScalar(pulse);
    }
    // 移动
    const speed = this.keys['ShiftLeft'] || this.keys['ShiftRight'] ? 5.6 : 3.3;
    const f = new THREE.Vector3(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
    const rgt = new THREE.Vector3(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
    const move = new THREE.Vector3();
    if (this.keys['KeyW'] || this.keys['ArrowUp']) move.add(f);
    if (this.keys['KeyS'] || this.keys['ArrowDown']) move.sub(f);
    if (this.keys['KeyD'] || this.keys['ArrowRight']) move.add(rgt);
    if (this.keys['KeyA'] || this.keys['ArrowLeft']) move.sub(rgt);
    const moving = move.lengthSq() > 0.001;
    if (moving) {
      move.normalize().multiplyScalar(speed * dt);
      this.pos.x += move.x;
      this.pos.z += move.z;
      const [px, pz] = this._collide(this.pos.x, this.pos.z);
      this.pos.x = px; this.pos.z = pz;
      this.bobPhase += dt * speed * 1.9;
    }
    const bobY = moving ? Math.sin(this.bobPhase) * 0.05 : 0;
    this.camera.rotation.set(this.pitch, this.yaw, 0);
    this.camera.position.set(this.pos.x, this.pos.y + EYE_H + bobY, this.pos.z);
    // 渲染
    this.renderer.render(this.scene, this.camera);
  }

  // ==================== 场景装饰 ====================
  _createDecor() {
    // 树
    for (let i = 0; i < 14; i++) {
      const x = (Math.random() - 0.5) * 56;
      const z = (Math.random() - 0.5) * 56;
      if (Math.abs(x) < 5.5 && Math.abs(z) < 6.5) continue;
      const scale = 0.8 + Math.random() * 0.9;
      const tree = new THREE.Group();
      const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.22 * scale, 0.3 * scale, 1.6 * scale, 8),
        new THREE.MeshStandardMaterial({ color: 0x6d4c2f, roughness: 1 })
      );
      trunk.position.y = 0.8 * scale;
      trunk.castShadow = true;
      tree.add(trunk);
      const leaf = new THREE.Mesh(
        new THREE.ConeGeometry(1.05 * scale, 2.2 * scale, 10),
        new THREE.MeshStandardMaterial({ color: 0x3e7d3a, roughness: 0.9 })
      );
      leaf.position.y = 2.1 * scale;
      leaf.castShadow = true;
      tree.add(leaf);
      tree.position.set(x, 0, z);
      tree.rotation.y = Math.random() * Math.PI;
      this.scene.add(tree);
    }
    // 石块
    for (let i = 0; i < 10; i++) {
      const x = (Math.random() - 0.5) * 52;
      const z = (Math.random() - 0.5) * 52;
      if (Math.abs(x) < 5.5 && Math.abs(z) < 6.5) continue;
      const s = 0.25 + Math.random() * 0.5;
      const rock = new THREE.Mesh(
        new THREE.IcosahedronGeometry(s, 0),
        new THREE.MeshStandardMaterial({ color: 0x9a958c, roughness: 0.95 })
      );
      rock.position.set(x, s * 0.5, z);
      rock.rotation.set(Math.random() * 3, Math.random() * 3, 0);
      rock.castShadow = true;
      this.scene.add(rock);
    }
    // 边界围栏
    const fenceMat = new THREE.MeshStandardMaterial({ color: 0x8a6a42, roughness: 0.9 });
    const railGeo = new THREE.BoxGeometry(0.12, 0.5, 0.12);
    const railLong = new THREE.BoxGeometry(32, 0.12, 0.1);
    const points = [[-16, -16], [16, -16], [16, 16], [-16, 16]];
    for (let i = 0; i < 4; i++) {
      const a = points[i], b = points[(i + 1) % 4];
      const len = Math.abs(b[0] - a[0]) + Math.abs(b[1] - a[1]);
      const n = Math.floor(len / 2.4);
      for (let k = 0; k <= n; k++) {
        const t = k / n;
        const x = a[0] + (b[0] - a[0]) * t;
        const z = a[1] + (b[1] - a[1]) * t;
        const post = new THREE.Mesh(railGeo, fenceMat);
        post.position.set(x, 0.35, z);
        post.castShadow = true;
        this.scene.add(post);
      }
      const rail = new THREE.Mesh(railLong, fenceMat);
      rail.rotation.y = b[0] !== a[0] ? 0 : Math.PI / 2;
      rail.position.set((a[0] + b[0]) / 2, 0.62, (a[1] + b[1]) / 2);
      this.scene.add(rail);
    }
  }
}
