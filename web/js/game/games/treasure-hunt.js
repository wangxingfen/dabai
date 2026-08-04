/* ============================================================
 * 示例小游戏：迷宫寻宝 (Treasure Hunt)
 *
 * 玩法：
 * - 自动生成的迷宫场景
 * - 散落的宝箱和各种收集品
 * - 用户操控角色探索迷宫、收集物品、找到最终宝藏
 * - AI感知游戏状态，提供提示和情绪反应
 * - 一体双魂：用户控制身体探索，AI感受冒险的乐趣
 * ============================================================ */

import { BaseGame } from './base-game.js';
import { RLAgentManager } from '../rl/rl-agent-manager.js';
import { encodeObservation } from '../rl/observation-spec.js';
import { QuizService } from './quiz-service.js';
import { QuizUI } from './quiz-ui.js';

// ==================== RL 配置（P1-4：统一契约接入） ====================
// 决策节奏由 HumanInterfaceController 接管（P2-3）：人化反应延迟 + ≤20Hz 硬约束
const RL_MOVE_SPEED = 3.0;          // RL 移动速度 (m/s)
const RL_TAKEOVER_DELAY_MS = 5000;  // 用户停止操作 5s 后 RL 接管
// 奖励表（统一符号策略：正=收益，负=代价）
const RL_REWARD = {
  STEP: -0.02,        // 时间代价（负小）
  HIT_WALL: -0.3,     // 撞墙（负中）
  COLLECT: 10,        // 收集星光碎片（正大）
  WIN: 50,            // 找到宝藏通关（正大）
  APPROACH: 0.1,      // 靠近收集品（正小）
};

// 迷宫生成算法（深度优先搜索）
function generateMaze(rows, cols) {
  // 初始化全墙
  const maze = [];
  for (let r = 0; r < rows; r++) {
    maze[r] = [];
    for (let c = 0; c < cols; c++) {
      maze[r][c] = 1; // 1 = 墙
    }
  }

  // DFS生成路径
  const visited = new Set();
  function dfs(r, c) {
    const key = `${r},${c}`;
    if (visited.has(key)) return;
    visited.add(key);
    maze[r][c] = 0; // 0 = 通路

    // 随机方向
    const dirs = [[-2, 0], [2, 0], [0, -2], [0, 2]];
    for (let i = dirs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [dirs[i], dirs[j]] = [dirs[j], dirs[i]];
    }

    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited.has(`${nr},${nc}`)) {
        // 打通中间格子
        maze[r + dr / 2][c + dc / 2] = 0;
        dfs(nr, nc);
      }
    }
  }

  // 从(1,1)开始（奇数坐标）
  dfs(1, 1);

  return maze;
}

export class TreasureHuntGame extends BaseGame {
  constructor(app) {
    super(app);
    this.name = 'treasure_hunt';
    this.displayName = '迷宫寻宝';
    this.description = '在神秘迷宫中探索，触碰星光与线索需答对谜题才能收集，集齐所有星光与线索才能打开宝藏！';
    this.moveSpeed = 3.5;
    this.initialCameraRadius = 12;   // 相机初始化距离主角12米
    this.initialCameraHeight = 13;  // 相机初始化高度13米
    this.boundarySize = 0;

    // 迷宫参数
    this._mazeRows = 13;
    this._mazeCols = 13;
    this._cellSize = 2.5;
    this._mazeData = null;
    this._walls = [];

    // 游戏元素
    this._treasure = null;        // 最终宝藏
    this._collectibles = [];      // 收集品列表
    this._collectedCount = 0;
    this._totalCollectibles = 0;
    this._clues = [];             // 线索标记
    this._visitedClues = new Set();

    // 答题机制（P：收集线索/星光必须答对谜题，答错即挑战失败）
    this._starCount = 8;          // 星光碎片数量
    this._clueCount = 3;          // 线索数量
    this._quizService = new QuizService();   // 题库服务（天行数据 + 本地兜底）
    this._quizUI = new QuizUI();             // 答题弹窗
    this._quizQuestions = [];                // 预取的题目队列
    this._quizOpen = false;                  // 是否正在答题（暂停拾取/RL）
    this._quizPending = null;                // { kind: 'star'|'clue'|'treasure', ref }
    this._currentQuiz = null;                // 当前题目（AI 感知用）
    this._questionsAnswered = 0;             // 已答题目数
    this._questionsCorrect = 0;              // 答对题数
    this._treasureOpen = false;              // 所有线索/星光集齐后宝藏解锁
    this._treasureLockNotified = false;      // 已提示过宝藏被封印（防止重复刷屏）
    this._lightingUp = false;                // 亮灯动画进行中
    this._litObj = null;                     // 正在亮灯的对象
    this._litKind = null;                    // 'star' | 'clue'
    this._litTime = 0;
    this._litDuration = 1.2;

    // 碰撞检测
    this._wallBoxes = [];
    this._playerRadius = 0.4;

    // 物理状态（模拟重力+着地，与沙盒类似但简化为平面）
    this._playerVelocityY = 0;    // 垂直速度
    this._isGrounded = false;     // 是否在地面
    this._groundY = 0;            // 迷宫地面高度

    // UI提示
    this.uiHint = 'WASD 移动 · 拖拽转向 · 收集星光碎片 · 找到宝藏！';

    // RL 会话状态（P1-4）
    this._rl = null;              // 惰性初始化 { agent, decisionTimer, active, lastUserTime, hitWall }
  }

  generateScene() {
    const THREE = this.THREE;

    // 生成迷宫数据
    this._mazeData = generateMaze(this._mazeRows, this._mazeCols);
    const totalSize = this._mazeCols * this._cellSize * 1.1;
    this.boundarySize = totalSize;

    // 使用管理器传入的场景生成器
    const sceneGen = this.App._gameSceneGen;
    if (!sceneGen) return;

    // 生成场景环境
    sceneGen.generateEnvironment({
      style: 'dungeon',
      size: totalSize,
      colorScheme: 'mystical'
    });

    // 生成迷宫墙壁
    this._walls = sceneGen.generateMazeWalls(this._mazeData, this._cellSize);

    // 构建墙壁碰撞盒
    this._buildWallColliders();

    // 放置收集品（在迷宫通路中）
    this._placeCollectibles();

    // 放置线索
    this._placeClues();

    // 放置最终宝藏（迷宫最远处）
    this._placeTreasure();

    // 移动角色到迷宫入口（高空出生，自由落体入场）
    const avatar = this.App.currentAvatar;
    if (avatar) {
      const startPos = this._getCellWorldPos(1, 1);
      avatar.position.set(startPos.x, 20, startPos.z);
      this._playerVelocityY = 0;
      this._isGrounded = false;
    }
  }

  /** 标记为物理游戏（禁用 walk 动画的 Y 轴接管） */
  requestJump() {
    // 迷宫不需要二段跳，但需要此方法让 GameModeManager 识别为物理游戏
    return false;
  }

  onStart() {
    super.onStart();
    this.uiHint = `✨ 触碰星光/线索需答对谜题: 0/${this._totalCollectibles} · WASD移动 · 拖拽转向`;
    // 预取题目（天行数据百科题库 → 本地兜底）
    this._prefetchQuestions();
    if (this.App.sendAIAction) {
      this.App.sendAIAction('（你进入了一个神秘迷宫！周围是发光墙壁，空气中飘着魔法粒子。注意：每一颗星光和每一条线索都被谜题守护着，答对才能收集，答错可就挑战失败啦！集齐所有星光与线索才能打开宝藏，我们互相配合，一起加油！）');
    }
  }

  /** 预取本局所需题目（异步，不阻塞开局） */
  async _prefetchQuestions() {
    try {
      const count = this._starCount + this._clueCount + 3;
      this._quizQuestions = await this._quizService.fetchQuestions(count);
    } catch (e) {
      this._quizQuestions = [];
    }
  }

  /** 取下一道题（不足时尝试重新拉取） */
  async _nextQuestion() {
    if (this._quizQuestions.length === 0) {
      await this._prefetchQuestions();
    }
    return this._quizQuestions.length > 0 ? this._quizQuestions.shift() : null;
  }

  getExtraState() {
    return {
      collected: this._collectedCount,
      total_collectibles: this._totalCollectibles,
      treasure_found: this._treasure ? false : true,
      treasure_open: this._treasureOpen,
      questions_answered: this._questionsAnswered,
      questions_correct: this._questionsCorrect,
    };
  }

  // ==================== AI 感知数据（富快照） ====================

  _getMapData() {
    return {
      type: 'grid',
      rows: this._mazeRows,
      cols: this._mazeCols,
      cell_size: this._cellSize,
      size: this._mazeCols * this._cellSize * 1.1,
      cells: this._mazeData,
    };
  }

  _getObjectsData() {
    const data = {};

    // 收集品
    const collectibles = [];
    const collectibleColors = ['#ffdd44', '#ff44dd', '#44ddff', '#44ff88', '#ff8844', '#8844ff', '#ff8888', '#88ff44'];
    for (let i = 0; i < this._collectibles.length; i++) {
      const obj = this._collectibles[i];
      if (!obj.parent) continue;
      collectibles.push({
        id: `星光碎片${i + 1}`,
        x: +obj.position.x.toFixed(1),
        z: +obj.position.z.toFixed(1),
        color: ['金色','粉色','天蓝','翠绿','橙色','紫色','红色','黄绿'][i % 8] || '彩色',
        collected: false,
      });
    }
    if (collectibles.length > 0) data.collectible = collectibles;

    // 宝藏
    if (this._treasure && this._treasure.parent) {
      data.treasure = [{
        id: '神秘宝箱',
        x: +this._treasure.position.x.toFixed(1),
        z: +this._treasure.position.z.toFixed(1),
        found: false,
        locked: !this._treasureOpen,
      }];
    }

    // 当前谜题（AI 智能体通过感知触发即可看到题目与选项）
    if (this._currentQuiz) {
      data.quiz = [{
        id: 'current_question',
        index: this._currentQuiz.index,
        total: this._currentQuiz.total,
        kind: this._currentQuiz.kind,
        title: this._currentQuiz.title,
        options: this._currentQuiz.options,
      }];
    }

    // 线索
    const clues = [];
    const clueTexts = ['←', '↑', '✨'];
    for (let i = 0; i < this._clues.length; i++) {
      const c = this._clues[i];
      if (!c.parent) continue;
      clues.push({
        id: `线索${i + 1}`,
        x: +c.position.x.toFixed(1),
        z: +c.position.z.toFixed(1),
        text: clueTexts[i] || '?',
        visited: this._visitedClues.has(i),
      });
    }
    if (clues.length > 0) data.clue = clues;

    return data;
  }

  _getNearbyObjects() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return [];

    const px = avatar.position.x;
    const pz = avatar.position.z;
    const perceptionRange = 30.0;  // AI 感知范围 (m)
    const nearby = [];

    // 玩家朝向角
    const facing = this.App.smoothRotY || 0;

    // 附近的收集品
    for (let i = 0; i < this._collectibles.length; i++) {
      const obj = this._collectibles[i];
      if (!obj.parent) continue;
      const dx = obj.position.x - px;
      const dz = obj.position.z - pz;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < perceptionRange) {
        nearby.push({
          type: 'collectible',
          id: `星光碎片${i + 1}`,
          x: +obj.position.x.toFixed(1),
          z: +obj.position.z.toFixed(1),
          distance: +dist.toFixed(1),
          direction: this._relativeDir(dx, dz, facing),
          color: ['金色','粉色','天蓝','翠绿','橙色','紫色','红色','黄绿'][i % 8],
        });
      }
    }

    // 附近的宝藏
    if (this._treasure && this._treasure.parent) {
      const dx = this._treasure.position.x - px;
      const dz = this._treasure.position.z - pz;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < perceptionRange) {
        nearby.push({
          type: 'treasure',
          id: '神秘宝箱',
          x: +this._treasure.position.x.toFixed(1),
          z: +this._treasure.position.z.toFixed(1),
          distance: +dist.toFixed(1),
          direction: this._relativeDir(dx, dz, facing),
        });
      }
    }

    // 附近的墙壁（前方最近的墙壁）
    const wallDist = this._raycastWall(px, pz, facing);
    if (wallDist < perceptionRange && wallDist > 0) {
      nearby.push({
        type: 'wall',
        direction: '前方',
        distance: +wallDist.toFixed(1),
      });
    }

    // 附近的线索
    const clueTexts = ['←', '↑', '✨'];
    for (let i = 0; i < this._clues.length; i++) {
      const c = this._clues[i];
      if (!c.parent) continue;
      const dx = c.position.x - px;
      const dz = c.position.z - pz;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < perceptionRange) {
        nearby.push({
          type: 'clue',
          id: `线索标记`,
          x: +c.position.x.toFixed(1),
          z: +c.position.z.toFixed(1),
          distance: +dist.toFixed(1),
          direction: this._relativeDir(dx, dz, facing),
          text: clueTexts[i] || '?',
        });
      }
    }

    // 按距离排序（近的在前）
    nearby.sort((a, b) => a.distance - b.distance);
    return nearby;
  }

  /** 计算相对方向 */
  _relativeDir(dx, dz, facing) {
    // 将世界空间偏移旋转到玩家视角
    const cosA = Math.cos(-facing);
    const sinA = Math.sin(-facing);
    const rx = dx * cosA - dz * sinA;  // 玩家视角的左右
    const rz = dx * sinA + dz * cosA;  // 玩家视角的前后

    const angle = Math.atan2(rx, rz);  // atan2(左右, 前后)
    const deg = (angle * 180 / Math.PI + 360) % 360;

    if (deg < 22.5 || deg >= 337.5) return '正前方';
    if (deg >= 22.5 && deg < 67.5) return '右前方';
    if (deg >= 67.5 && deg < 112.5) return '右方';
    if (deg >= 112.5 && deg < 157.5) return '右后方';
    if (deg >= 157.5 && deg < 202.5) return '正后方';
    if (deg >= 202.5 && deg < 247.5) return '左后方';
    if (deg >= 247.5 && deg < 292.5) return '左方';
    return '左前方';
  }

  /** 前方射线检测墙壁距离 */
  _raycastWall(px, pz, facing) {
    const step = 0.5;
    const maxDist = 8.0;
    const cosA = Math.cos(facing);
    const sinA = Math.sin(facing);
    const halfCell = this._cellSize * 0.45;

    for (let d = step; d <= maxDist; d += step) {
      const cx = px + sinA * d;  // facing 方向 = 世界空间 +X 偏向 sinA
      const cz = pz + cosA * d;  // facing 方向 = 世界空间 +Z 偏向 cosA
      for (const w of this._wallBoxes) {
        if (cx >= w.minX && cx <= w.maxX && cz >= w.minZ && cz <= w.maxZ) {
          return d;
        }
      }
    }
    return maxDist + 1;
  }

  _getPlayerSpeed() {
    // 从 App 获取上一帧的移动速度
    return this._lastMoveSpeed || 0;
  }

  // ===== 在 update 中记录移动速度 =====
  update(dt) {
    super.update(dt);
    if (this.state !== 'playing') return;

    // 物理：重力 + 着地检测
    this._updatePlayerPhysics(dt);

    // 碰撞检测（阻止穿墙）
    this._checkWallCollision();

    // 检测收集品拾取（答对谜题才能收集）
    if (!this._quizOpen) {
      this._updateLightUp(dt);
      this._checkCollectiblePickup();
      // 检测线索触碰（同样需要答对谜题）
      this._checkClueTouch();
      // 检测宝藏触碰
      this._checkTreasureReach();
    }

    // 更新UI提示
    this._updateHint();

    // RL 决策驱动（用户停止操作后接管；答题期间不接管）
    this._rlUpdate(dt);

    // 记录移动速度（用于 AI 感知）
    this._lastMoveSpeed = 0;
  }

  // ==================== RL Env 契约（P1-4） ====================

  /** RL 环境规格版本 */
  rlSpecVersion() { return '1.0.0'; }

  /** 声明动作空间：4 方向世界移动 */
  getActionSpec() {
    if (!this._rlActionSpec) {
      this._rlActionSpec = ['up', 'down', 'left', 'right'].map((name, id) => ({
        id, name, semantics: 'semantic', executable: true,
      }));
    }
    return this._rlActionSpec;
  }

  /** 声明观察空间：13 维（墙 4 + 最近收集品 3 + 宝藏 3 + 进度 + 撞墙 + 速度） */
  getObservationSpec() {
    if (!this._rlObsSpec) {
      this._rlObsSpec = [
        { name: 'walls', kind: 'vector', dim: 4, scale: 1, offset: 0 },
        { name: 'nearest', kind: 'vector', dim: 3, scale: 1, offset: 0 },
        { name: 'treasure', kind: 'vector', dim: 3, scale: 1, offset: 0 },
        { name: 'progress', kind: 'scalar', scale: 1, offset: 0 },
        { name: 'hitWall', kind: 'scalar', scale: 1, offset: 0 },
        { name: 'speed', kind: 'scalar', scale: 5, offset: 0 },
      ];
    }
    return this._rlObsSpec;
  }

  /** 获取当前观察（各特征已归一化 0-1，经规格模块编码为 Float64Array） */
  getObservation() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return new Float64Array(13);
    const px = avatar.position.x, pz = avatar.position.z;
    const walls = this._getSurroundingWalls(px, pz);

    let nearest = [0, 0, 1];
    let nearestDist = Infinity;
    for (const obj of this._collectibles) {
      if (!obj.parent) continue;
      const dx = obj.position.x - px, dz = obj.position.z - pz;
      const d = Math.hypot(dx, dz);
      if (d < nearestDist) { nearestDist = d; nearest = [dx, dz, d]; }
    }
    // 归一化：dx/dz 除以 8m，dist 除以 12m
    if (nearestDist < Infinity) {
      nearest = [nearest[0] / 8, nearest[1] / 8, Math.min(1, nearest[2] / 12)];
    }

    let tre = [0, 0, 1];
    if (this._treasure && this._treasure.parent) {
      const dx = this._treasure.position.x - px, dz = this._treasure.position.z - pz;
      tre = [dx / 12, dz / 12, Math.min(1, Math.hypot(dx, dz) / 16)];
    }

    return encodeObservation(this.getObservationSpec(), {
      walls,
      nearest,
      treasure: tre,
      progress: this._totalCollectibles > 0 ? this._collectedCount / this._totalCollectibles : 0,
      hitWall: (this._rl && this._rl.hitWall) || 0,
      speed: Math.abs(this._lastMoveSpeed || 0),
    });
  }

  /** 执行 RL 动作：设置移动方向 */
  applyAction(actionId) {
    if (!this._rl) return false;
    switch (actionId) {
      case 0: this._rl.moveX = 0; this._rl.moveZ = -1; break;
      case 1: this._rl.moveX = 0; this._rl.moveZ = 1; break;
      case 2: this._rl.moveX = -1; this._rl.moveZ = 0; break;
      case 3: this._rl.moveX = 1; this._rl.moveZ = 0; break;
      default: return false;
    }
    return true;
  }

  /** 当前可用动作（迷宫无额外限制，全部可用） */
  getValidActions() { return [0, 1, 2, 3]; }

  /** RL 超参：注册表统一提供 */
  getRLHyperparams() { return null; }

  /** 回合是否结束（集齐全部星光/线索且打开宝藏才算完成） */
  rlDone() {
    if (this.state !== 'playing') return true;
    return this._treasureOpen
      && this._collectedCount >= this._totalCollectibles
      && (!this._treasure || !this._treasure.parent);
  }

  /** 惰性初始化 RL 智能体（注册表统一配置） */
  _rlEnsureAgent() {
    if (!this._rl) {
      this._rl = {
        agent: RLAgentManager.get().getAgent('treasure_hunt', this),
        active: false,
        lastUserTime: performance.now(),
        hitWall: 0,
        moveX: 0,
        moveZ: 0,
        // P2-3 接口节奏真实化：人化反应延迟控制器
        interfaceController: RLAgentManager.get().getInterfaceController(),
        // P2-2 评估：本局起始时间与最近决策时刻
        episodeStartTs: performance.now(),
        lastStepTs: performance.now(),
      };
      // P2-1b 从已采集的人类轨迹挂接行为克隆先验（异步、静默跳过）
      RLAgentManager.get().enableBehaviorCloning('treasure_hunt', this).catch(() => {});
      // P3-1 世界模型增强训练（想象回放，提升样本效率；异步、静默失败）
      RLAgentManager.get().enableWorldModel('treasure_hunt', 0.5, 8);
    }
    return this._rl.agent;
  }

  /** 每帧 RL 驱动：接管判定 + 决策循环（P2-3 人化节奏） */
  _rlUpdate(dt) {
    // 答题/亮灯期间暂停 RL 接管，避免 AI 在玩家答题或谜题触发时乱跑
    if (this._quizOpen || this._lightingUp) {
      if (this._rl) this._rl.lastUserTime = performance.now();
      return;
    }

    const agent = this._rlEnsureAgent();
    if (!agent) return;

    const gm = this.App.gameModeManager;
    const bridge = (gm && gm.controlBridge) || this.App._gameControlBridge || this.App.controlBridge;
    const userActive = !!(bridge && bridge.userControlling);

    if (userActive) {
      this._rl.lastUserTime = performance.now();
      this._rl.active = false;
      // P2-1a 人类轨迹采集（用户真实操控期间记录）
      const rec = RLAgentManager.get().getTrajectoryRecorder();
      if (!rec.isRecording()) rec.startRecording('treasure_hunt');
      rec.recordFrame(this, bridge);
      return;
    }
    // 用户停止操控 → 结束并保存一段人类轨迹
    const rec = RLAgentManager.get().getTrajectoryRecorder();
    if (rec.isRecording()) rec.stopRecording();

    if (!this._rl.active && performance.now() - this._rl.lastUserTime < RL_TAKEOVER_DELAY_MS) return;
    if (!this._rl.active) {
      // 刚接管：清除 idle walk 目标，避免 RL 位置控制与自动行走双驱动冲突
      this.App.idleWalkTarget = null;
      this.App._aiDrivenWalk = false;
      this.App.idleWalkProgress = 1;
    }
    this._rl.active = true;

    // P2-3：人化反应延迟触发决策（频率硬约束 ≤20Hz）
    if (this._rl.interfaceController.shouldAct(performance.now())) {
      this._rlStep();
    }
  }

  /** 单个决策步：感知 → 决策 → 执行 → 结算 → 存储/训练 */
  _rlStep() {
    const agent = this._rlEnsureAgent();
    const avatar = this.App.currentAvatar;
    if (!agent || !avatar) return;

    const stateVec = this.getObservation();
    const { action } = agent.chooseAction(stateVec, this.getValidActions());
    const prevPos = { x: avatar.position.x, z: avatar.position.z };
    const prevCount = this._collectedCount;
    const prevTreasure = !!(this._treasure && this._treasure.parent);
    const prevDist = this._nearestCollectibleDist();

    // 执行移动 + 碰撞修正（步长 = 实际经过的人化决策间隔，保持速度恒定）
    this.applyAction(action);
    const now = performance.now();
    const stepSec = Math.min(0.5, Math.max(0.05, (now - this._rl.lastStepTs) / 1000));
    this._rl.lastStepTs = now;
    avatar.position.x += this._rl.moveX * RL_MOVE_SPEED * stepSec;
    avatar.position.z += this._rl.moveZ * RL_MOVE_SPEED * stepSec;
    this._checkWallCollision();

    // 结算奖励（统一符号策略）
    let reward = RL_REWARD.STEP;
    const moved = Math.hypot(avatar.position.x - prevPos.x, avatar.position.z - prevPos.z);
    if (moved < 0.05) {
      reward += RL_REWARD.HIT_WALL;
      this._rl.hitWall = 1;
    } else {
      this._rl.hitWall = 0;
    }
    const gained = this._collectedCount - prevCount;
    if (gained > 0) reward += RL_REWARD.COLLECT * gained;
    if (prevTreasure && (!this._treasure || !this._treasure.parent)) reward += RL_REWARD.WIN;
    const nowDist = this._nearestCollectibleDist();
    if (prevDist !== Infinity && nowDist < prevDist - 0.15) reward += RL_REWARD.APPROACH;

    const nextState = this.getObservation();
    const done = this.rlDone();
    agent.store(stateVec, action, reward, nextState, done);
    agent.train();

    if (done) {
      const win = this._collectedCount >= this._totalCollectibles;
      agent.endEpisode(win ? RL_REWARD.WIN : 0, { win });
      // P2-2 人类限时评估基准：记录本局耗时与胜负
      RLAgentManager.get().getBaselineEvaluator().recordEpisode('treasure_hunt', {
        durationMs: performance.now() - this._rl.episodeStartTs,
        win,
      });
      this._rl.active = false;
      this._rl.lastUserTime = performance.now();
      this._rl.episodeStartTs = performance.now();
      this._rl.interfaceController.reset();
    }
  }

  /** 最近收集品距离（无则 Infinity） */
  _nearestCollectibleDist() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return Infinity;
    let min = Infinity;
    for (const obj of this._collectibles) {
      if (!obj.parent) continue;
      const d = Math.hypot(obj.position.x - avatar.position.x, obj.position.z - avatar.position.z);
      if (d < min) min = d;
    }
    return min;
  }

  /** 获取玩家所在格四周墙标志（世界方向：上 z-/下 z+/左 x-/右 x+） */
  _getSurroundingWalls(px, pz) {
    const cell = this._worldToCell(px, pz);
    if (!cell) return [1, 1, 1, 1];
    const { r, c } = cell;
    const isWall = (rr, cc) => {
      if (rr < 0 || rr >= this._mazeRows || cc < 0 || cc >= this._mazeCols) return 1;
      return this._mazeData[rr][cc] === 1 ? 1 : 0;
    };
    return [
      isWall(r - 1, c),  // 上 (z-)
      isWall(r + 1, c),  // 下 (z+)
      isWall(r, c - 1),  // 左 (x-)
      isWall(r, c + 1),  // 右 (x+)
    ];
  }

  /** 世界坐标 → 迷宫网格坐标 */
  _worldToCell(px, pz) {
    if (!this._mazeData) return null;
    const offsetX = -this._mazeCols * this._cellSize / 2 + this._cellSize / 2;
    const offsetZ = -this._mazeRows * this._cellSize / 2 + this._cellSize / 2;
    const c = Math.round((px - offsetX) / this._cellSize);
    const r = Math.round((pz - offsetZ) / this._cellSize);
    if (r < 0 || r >= this._mazeRows || c < 0 || c >= this._mazeCols) return null;
    return { r, c };
  }

  /** 重力+着地物理 */
  _updatePlayerPhysics(dt) {
    const avatar = this.App.currentAvatar;
    if (!avatar) return;

    const GRAVITY = 15;
    const MAX_FALL_SPEED = 25;

    // 应用重力
    if (!this._isGrounded) {
      this._playerVelocityY -= GRAVITY * dt;
      if (this._playerVelocityY < -MAX_FALL_SPEED) this._playerVelocityY = -MAX_FALL_SPEED;
    }

    avatar.position.y += this._playerVelocityY * dt;

    // 着地检测
    if (avatar.position.y <= this._groundY) {
      avatar.position.y = this._groundY;
      this._isGrounded = true;
      this._playerVelocityY = 0;
    } else {
      this._isGrounded = false;
    }

    // 同步行走动画基准高度
    if (!avatar.userData) avatar.userData = {};
    avatar.userData._baseY = this._isGrounded ? this._groundY : avatar.position.y;
  }

  /** 记录移动速度（由 GameModeManager._applyPlayerMovement 调用后设置） */
  setPlayerSpeed(speed) {
    this._lastMoveSpeed = speed;
  }

  // ==================== 内部实现 ====================

  _getCellWorldPos(r, c) {
    const offsetX = -this._mazeCols * this._cellSize / 2 + this._cellSize / 2;
    const offsetZ = -this._mazeRows * this._cellSize / 2 + this._cellSize / 2;
    return {
      x: offsetX + c * this._cellSize,
      z: offsetZ + r * this._cellSize
    };
  }

  _buildWallColliders() {
    this._wallBoxes = [];
    const THREE = this.THREE;
    const cellSize = this._cellSize;
    const halfCell = cellSize * 0.45;

    for (let r = 0; r < this._mazeRows; r++) {
      for (let c = 0; c < this._mazeCols; c++) {
        if (this._mazeData[r][c] === 1) {
          const pos = this._getCellWorldPos(r, c);
          this._wallBoxes.push({
            minX: pos.x - halfCell,
            maxX: pos.x + halfCell,
            minZ: pos.z - halfCell,
            maxZ: pos.z + halfCell,
          });
        }
      }
    }
  }

  _findEmptyCell() {
    const attempts = 200;
    for (let i = 0; i < attempts; i++) {
      const r = 1 + Math.floor(Math.random() * (this._mazeRows - 2));
      const c = 1 + Math.floor(Math.random() * (this._mazeCols - 2));
      if (this._mazeData[r][c] === 0) {
        return { r, c, pos: this._getCellWorldPos(r, c) };
      }
    }
    return { r: 1, c: 1, pos: this._getCellWorldPos(1, 1) };
  }

  _placeCollectibles() {
    this._totalCollectibles = this._starCount;
    const used = new Set();
    used.add('1,1'); // 入口不放

    for (let i = 0; i < this._starCount; i++) {
      const cell = this._findEmptyCell();
      const key = `${cell.r},${cell.c}`;
      if (used.has(key)) { i--; continue; }
      used.add(key);

      const pos = new this.THREE.Vector3(cell.pos.x, 1.2, cell.pos.z);
      const colors = [0xffdd44, 0xff44dd, 0x44ddff, 0x44ff88, 0xff8844, 0x8844ff, 0xff8888, 0x88ff44];
      const sceneGen = this.App._gameSceneGen;
      const obj = sceneGen.generateCollectible(pos, {
        color: colors[i % colors.length],
        size: 0.2,
        type: i % 3 === 0 ? 'gem' : 'star'
      });
      obj.userData.collectIndex = i;
      this._collectibles.push(obj);
    }
  }

  _placeClues() {
    const clues = [
      { r: 3, c: 5, text: '←' },
      { r: 7, c: 9, text: '↑' },
      { r: 9, c: 3, text: '✨' },
    ];

    for (const clue of clues) {
      // 确保线索在通路上
      if (this._mazeData[clue.r] && this._mazeData[clue.r][clue.c] === 0) {
        const pos = this._getCellWorldPos(clue.r, clue.c);
        const sceneGen = this.App._gameSceneGen;
        const marker = sceneGen.generateClueMarker(
          new this.THREE.Vector3(pos.x, 0, pos.z),
          clue.text
        );
        this._clues.push(marker);
      }
    }
    // 线索也是收集目标：全部集齐（星光 + 线索）才能打开宝藏
    this._totalCollectibles += this._clues.length;
  }

  _placeTreasure() {
    // 放在迷宫最深处的通路
    let bestCell = { r: this._mazeRows - 2, c: this._mazeCols - 2 };
    let maxDist = 0;
    for (let r = 1; r < this._mazeRows - 1; r++) {
      for (let c = 1; c < this._mazeCols - 1; c++) {
        if (this._mazeData[r][c] === 0) {
          const dist = Math.abs(r - 1) + Math.abs(c - 1);
          if (dist > maxDist) {
            maxDist = dist;
            bestCell = { r, c };
          }
        }
      }
    }

    const pos = this._getCellWorldPos(bestCell.r, bestCell.c);
    const sceneGen = this.App._gameSceneGen;
    this._treasure = sceneGen.generateTreasure(
      new this.THREE.Vector3(pos.x, 0, pos.z),
      { size: 0.8, color: 0xffd700, glowColor: 0xffaa00 }
    );
  }

  /**
   * 碰撞检测钩子 —— 由 GameModeManager._applyPlayerMovement 调用
   * 检查目标位置是否与迷宫墙壁相交
   * @param {number} newX - 目标 X 坐标
   * @param {number} newZ - 目标 Z 坐标
   * @returns {boolean} 返回 true 表示目标位置被墙壁阻挡
   */
  checkCollision(newX, newZ, options = {}) {
    const r = this._playerRadius;
    for (const w of this._wallBoxes) {
      // AABB vs circle 碰撞检测
      const closestX = Math.max(w.minX, Math.min(newX, w.maxX));
      const closestZ = Math.max(w.minZ, Math.min(newZ, w.maxZ));
      const dx = newX - closestX;
      const dz = newZ - closestZ;
      if (dx * dx + dz * dz < r * r) {
        return true;
      }
    }
    return false;
  }

  _checkWallCollision() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return;

    const px = avatar.position.x;
    const pz = avatar.position.z;
    const r = this._playerRadius;

    for (const w of this._wallBoxes) {
      // 扩展一点墙面碰撞范围
      const closestX = Math.max(w.minX - r, Math.min(px, w.maxX + r));
      const closestZ = Math.max(w.minZ - r, Math.min(pz, w.maxZ + r));
      const dx = px - closestX;
      const dz = pz - closestZ;
      const dist = Math.sqrt(dx * dx + dz * dz);

      if (dist < r && dist > 0.001) {
        const overlap = r - dist;
        const nx = dx / dist;
        const nz = dz / dist;
        avatar.position.x += nx * overlap;
        avatar.position.z += nz * overlap;
      }
    }
  }

  _checkCollectiblePickup() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return;
    const px = avatar.position.x;
    const pz = avatar.position.z;
    const pickupRange = 0.8;

    for (let i = this._collectibles.length - 1; i >= 0; i--) {
      const obj = this._collectibles[i];
      if (!obj.parent) {
        this._collectibles.splice(i, 1);
        continue;
      }
      const dx = px - obj.position.x;
      const dz = pz - obj.position.z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < pickupRange) {
        // 触碰星光 → 星光亮灯 → 亮灯结束弹出谜题（答对才能收进口袋）
        this._triggerLightUp('star', obj);
        return;
      }
    }
  }

  /** 触碰线索 → 线索亮灯 → 亮灯结束弹出谜题（答对才能收集线索） */
  _checkClueTouch() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return;
    const px = avatar.position.x;
    const pz = avatar.position.z;
    const touchRange = 1.0;

    for (let i = this._clues.length - 1; i >= 0; i--) {
      const marker = this._clues[i];
      if (!marker.parent) {
        this._clues.splice(i, 1);
        continue;
      }
      const dx = px - marker.position.x;
      const dz = pz - marker.position.z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < touchRange) {
        this._triggerLightUp('clue', marker);
        return;
      }
    }
  }

  // ==================== 亮灯动画 ====================

  /** 触碰星光/线索：先亮灯（发光脉冲），亮灯结束再弹出谜题 */
  _triggerLightUp(kind, obj) {
    if (this._lightingUp || this._quizOpen || this.state !== 'playing') return;
    this._lightingUp = true;
    this._litObj = obj;
    this._litKind = kind;
    this._litTime = 0;
    this._litDuration = 1.2;
    if (obj && obj.userData._baseY === undefined) {
      obj.userData._baseY = obj.position.y;
    }

    // 系统提示：触发亮灯
    const label = kind === 'clue' ? '线索' : '星光';
    if (this.App.showToast) {
      this.App.showToast(`💡 触碰到了${label}！谜题即将出现…`);
    }
    this._pushEvent('quiz_light_up', { kind });
  }

  /** 每帧驱动亮灯动画（发光脉冲 + 上浮），结束后弹出谜题 */
  _updateLightUp(dt) {
    if (!this._lightingUp) return;
    this._litTime += dt;
    const t = Math.min(1, this._litTime / this._litDuration);

    const obj = this._litObj;
    if (obj && obj.parent) {
      // 星光：增强自发光脉冲；线索：光圈透明度脉冲 + 放大
      obj.traverse && obj.traverse(child => {
        if (child.isMesh && child.material) {
          const mats = Array.isArray(child.material) ? child.material : [child.material];
          mats.forEach(m => {
            if (m.emissive && m.emissiveIntensity !== undefined) {
              m.emissiveIntensity = 0.5 + Math.sin(t * Math.PI * 3) * 0.8 + t * 1.2;
            }
            if (m.opacity !== undefined && m.blending !== undefined) {
              m.opacity = Math.min(1, 0.5 + Math.sin(t * Math.PI * 3) * 0.3 + t * 0.3);
            }
          });
        }
      });
      const base = 1 + t * 0.35;
      obj.scale.set(base, base, base);
      obj.position.y = (obj.userData._baseY ?? 0) + Math.sin(this._litTime * 6) * 0.08 + t * 0.25;
    }

    if (t >= 1) {
      // 亮灯结束 → 弹出谜题
      const kind = this._litKind;
      const litObj = this._litObj;
      this._lightingUp = false;
      this._litObj = null;
      this._litKind = null;
      if (litObj && litObj.parent) {
        this._openQuiz(kind, litObj);
      }
    }
  }

  // ==================== 答题机制 ====================

  /** 触发一道谜题（收集星光/线索必须答对；答错即挑战失败） */
  _openQuiz(kind, ref) {
    if (this._quizOpen || this.state !== 'playing') return;
    this._quizOpen = true;
    this._quizPending = { kind, ref };
    this._openQuizAsync(kind);
  }

  async _openQuizAsync(kind) {
    try {
      await this._openQuizFlow(kind);
    } catch (e) {
      // 任何异常都不能让答题流程卡死：安全复位，允许再次触发
      console.error('[寻宝] 答题流程异常:', e);
      this._currentQuiz = null;
      this._quizOpen = false;
      this._quizPending = null;
      this._updateHint();
    }
  }

  /** 答题主流程（被 _openQuizAsync 的 try/catch 包裹） */
  async _openQuizFlow(kind) {
    const question = await this._nextQuestion();
    if (!question) {
      // 题库不可用：直接放行收集，避免玩家被卡死
      this._quizOpen = false;
      this._quizPending = null;
      this._resolveQuiz(kind, null);
      this.sendAIAction('（题库好像迷路了…这次先放你过去，下次再来一起解谜吧！）');
      return;
    }

    this._questionsAnswered++;
    const index = this._questionsAnswered;
    this._currentQuiz = {
      index,
      total: this._totalCollectibles,
      kind,
      title: question.title,
      options: question.options,
    };
    this._pushEvent('quiz_question', {
      index, total: this._totalCollectibles, kind,
      title: question.title, options: question.options,
    });

    // AI 智能体通过触发即可看到题目与选项，陪伴玩家一起答题
    this.sendAIAction(
      `（一道谜题挡住了去路！题目：「${question.title}」，选项：A. ${question.options.A} ｜ B. ${question.options.B} ｜ C. ${question.options.C} ｜ D. ${question.options.D}。陪玩家一起解这道题吧：可以分享思路、排除干扰项，但先让玩家自己作答，除非Ta明确问你要答案。）`
    );
    this.uiHint = `📜 第 ${index}/${this._totalCollectibles} 题 · 答对才能收集！`;

    const result = await this._quizUI.present({
      index,
      total: this._totalCollectibles,
      kind,
      question,
      checkAnswer: async choice => this._quizService.checkAnswer(question, choice),
    });

    this._currentQuiz = null;
    this._quizUI.close();

    if (!result.correct) {
      // 答错 → 游戏挑战失败（先触发系统提示）
      this._pushEvent('quiz_wrong', { index, kind, choice: result.choice, answer: result.answer });
      if (this.App.showToast) {
        this.App.showToast(`❌ 答错了！正确答案：${result.answer}，挑战失败`);
      }
      this._quizOpen = false;
      this._quizPending = null;
      this.sendAIAction(
        `（答错了…这道题的正确答案是「${result.answer}」。解析：${result.analytic}。这次挑战失败了，没关系，失败是成功之母，我们重整旗鼓再来一次！）`
      );
      this.onFail(`答错谜题，挑战失败。正确答案：${result.answer}。${result.analytic}`);
      return;
    }

    this._questionsCorrect++;
    this._pushEvent('quiz_correct', { index, kind, choice: result.choice, answer: result.answer });
    if (this.App.showToast) {
      this.App.showToast(`✅ 答对了！解析已展示，${kind === 'clue' ? '线索' : '星光'}已收进口袋 (${this._collectedCount + 1}/${this._totalCollectibles})`);
    }
    this.sendAIAction(
      `（太棒了，答对了！正确答案是「${result.answer}」。${result.analytic}。又收获了一个${kind === 'clue' ? '线索' : '星光碎片'}，继续加油！）`
    );
    this._resolveQuiz(kind, this._quizPending ? this._quizPending.ref : null);
  }

  /** 答对后真正执行收集 */
  _resolveQuiz(kind, ref) {
    this._quizOpen = false;
    this._quizPending = null;
    if (kind === 'star') this._collectStar(ref);
    else if (kind === 'clue') this._collectClue(ref);
    this._checkTreasureUnlock();
    this._updateHint();
    // 答对后 AI 自动继续寻找下一个题目（无缝衔接，无需玩家操作）
    this._autoSeekNext();
  }

  /** AI 自动寻找下一个星光/线索（复用全局 idle walk 系统驱动） */
  _autoSeekNext() {
    // VR 沉浸模式或非游戏模式时不做自动行走
    if ((this.App.xrMode && this.App.xrMode !== 'off') || !this.App.gameModeActive) return;
    const target = this._nearestUncollected();
    if (!target) return;
    const avatar = this.App.currentAvatar || this.App.modelGroup;
    if (!avatar) return;

    const dx = target.position.x - avatar.position.x;
    const dz = target.position.z - avatar.position.z;
    const dist = Math.hypot(dx, dz);
    if (dist < 0.3) return; // 已在目标旁

    // 与 ai-autonomy-controller._injectWalkPath 一致的单段直达行走注入
    this.App._aiDrivenWalk = true;
    this.App.currentAction = { type: this.App.ActionType.WALK };
    this.App.walkPath = [];
    this.App.walkSegmentIndex = 0;
    this.App.idleWalkStart = { x: avatar.position.x, z: avatar.position.z };
    this.App.idleWalkTarget = new this.THREE.Vector3(target.position.x, avatar.position.y, target.position.z);
    this.App.idleWalkProgress = 0;
    this.App.walkFacingAngle = Math.atan2(dx, dz);
    const gameSpeed = (this.App.gameModeManager && this.App.gameModeManager.controlBridge)
      ? (this.App.gameModeManager.controlBridge._moveSpeed || 3.5)
      : 3.5;
    this.App.idleWalkSpeed = gameSpeed / Math.max(0.5, dist);

    if (this.App.sendAIAction) {
      this.App.sendAIAction('（走！我们继续出发，去找下一个被谜题守护的星光/线索！）');
    }
  }

  /** 查找最近的一个未收集目标（星光优先，其次线索） */
  _nearestUncollected() {
    const avatar = this.App.currentAvatar;
    if (!avatar) return null;
    let best = null;
    let bestDist = Infinity;
    const consider = arr => {
      arr.forEach(obj => {
        if (!obj.parent) return;
        const d = obj.position.distanceTo(avatar.position);
        if (d < bestDist) { bestDist = d; best = obj; }
      });
    };
    consider(this._collectibles);
    consider(this._clues);
    return best;
  }

  /** 收集星光碎片 */
  _collectStar(obj) {
    if (obj && obj.parent) obj.parent.remove(obj);
    const idx = this._collectibles.indexOf(obj);
    if (idx >= 0) this._collectibles.splice(idx, 1);
    this._collectedCount++;
    this.score += 10;

    this._pushEvent('item_collected', {
      collected: this._collectedCount,
      total: this._totalCollectibles,
      id: `星光碎片${idx + 1}`,
    });

    this.sendAIAction(
      `（答对了！又一颗星光碎片收集成功！现在收集了 ${this._collectedCount}/${this._totalCollectibles}，继续加油！）`
    );
  }

  /** 收集线索 */
  _collectClue(marker) {
    if (marker && marker.parent) marker.parent.remove(marker);
    const idx = this._clues.indexOf(marker);
    const clueTexts = ['←', '↑', '✨'];
    if (idx >= 0) {
      this._clues.splice(idx, 1);
      this._visitedClues.add(idx);
    }
    this._collectedCount++;
    this.score += 15;
    const text = clueTexts[idx] || '线索';

    this._pushEvent('clue_discovered', {
      collected: this._collectedCount,
      total: this._totalCollectibles,
      text,
    });

    this.sendAIAction(
      `（答对了！解开了一条神秘线索「${text}」！已经收集 ${this._collectedCount}/${this._totalCollectibles}，离宝藏越来越近了！）`
    );
  }

  /** 集齐全部星光与线索 → 宝藏解锁 */
  _checkTreasureUnlock() {
    if (this._treasureOpen || this._collectedCount < this._totalCollectibles) return;
    this._treasureOpen = true;
    this.uiHint = '🔓 所有星光与线索已集齐！快去打开宝藏宝箱！';
    if (this.App.showToast) {
      this.App.showToast('🔓 封印解除！所有星光与线索已集齐，快去打开宝藏！');
    }
    this._pushEvent('treasure_unlocked', { score: this.score });
    this.sendAIAction('（哇！所有星光碎片和线索都集齐了！宝箱的封印解除了，快去打开宝藏吧！）');
  }

  _checkTreasureReach() {
    if (!this._treasure || !this._treasure.parent) return;
    const avatar = this.App.currentAvatar;
    if (!avatar) return;

    const dx = avatar.position.x - this._treasure.position.x;
    const dz = avatar.position.z - this._treasure.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);

    if (dist < 1.2) {
      if (!this._treasureOpen) {
        // 尚未集齐：宝藏被封印，提示先完成收集（只提示一次，离开后再靠近会重新提示）
        const remain = this._totalCollectibles - this._collectedCount;
        this.uiHint = `🔒 宝藏被封印了，还需要收集 ${remain} 个星光/线索！`;
        if (!this._treasureLockNotified) {
          this._treasureLockNotified = true;
          if (this.App.showToast) {
            this.App.showToast(`🔒 宝藏被神秘封印锁住，还差 ${remain} 个星光/线索！`);
          }
          this._pushEvent('treasure_locked', { remain });
          this.sendAIAction('（宝藏宝箱就在眼前，但被神秘封印锁住了！看来得先集齐所有星光碎片和线索，才能打开它。）');
        }
        return;
      }
      // 集齐全部 → 打开宝藏，赢得游戏
      this._treasure.parent.remove(this._treasure);
      this._treasure = null;
      this.score += 50;
      this._pushEvent('treasure_found', { score: this.score });

      // 发送AI庆祝反应
      this.sendAIAction('（哇！！宝藏打开了！所有谜题都答对了，这是属于我们的胜利！好厉害的冒险伙伴！）');

      setTimeout(() => {
        this.onComplete({ treasure_found: true, all_questions_correct: true });
        this.uiHint = '🎉 恭喜赢得宝藏！得分: ' + this.score;
        if (this.App.showToast) {
          this.App.showToast('🎉 宝藏到手！得分: ' + this.score);
        }
      }, 500);
    } else if (dist < 3.5 && this._collectedCount >= 3) {
      this._treasureLockNotified = false; // 离开封印提示范围，允许再次提示
      this.uiHint = this._treasureOpen
        ? '🔥 宝藏已解锁，快去打开宝箱！'
        : '🔥 宝藏就在附近了！先收集所有星光与线索…';
    } else if (dist >= 3.5) {
      this._treasureLockNotified = false;
    }
  }

  _updateHint() {
    if (this.state === 'completed') return;
    if (this._quizOpen) return; // 答题中由答题弹窗接管提示
    if (this._collectedCount < this._totalCollectibles) {
      this.uiHint = `✨ 触碰星光/线索答对谜题即可收集: ${this._collectedCount}/${this._totalCollectibles} · WASD移动 · 拖拽转向`;
    } else if (this._treasureOpen) {
      this.uiHint = '🔓 所有星光与线索已集齐！快去打开宝藏宝箱！';
    }
  }

  cleanup() {
    // 关闭答题弹窗并重置答题状态
    if (this._quizUI) this._quizUI.close();
    this._quizOpen = false;
    this._quizPending = null;
    this._currentQuiz = null;
    this._quizQuestions = [];
    this._questionsAnswered = 0;
    this._questionsCorrect = 0;
    this._treasureOpen = false;
    this._treasureLockNotified = false;
    this._lightingUp = false;
    this._litObj = null;
    this._litKind = null;
    this._walls = [];
    this._collectibles = [];
    this._wallBoxes = [];
    this._clues = [];
    this._visitedClues.clear();
    this._mazeData = null;
    this._treasure = null;
    this._playerVelocityY = 0;
    this._isGrounded = false;
    super.cleanup();
  }
}
