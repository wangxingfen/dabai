/* ============================================================
 * 游戏控制权桥接 —— 用户操控与AI自主行为的切换
 *
 * 核心机制：
 * - 用户操控角色时，AI的自主动作系统(WALK/POSE/TURN/DANCE)被挂起
 * - AI仍然可以通过WebSocket说话、调用工具，但不能控制身体移动
 * - 用户停止操控后，AI恢复自主行动能力
 * - "一体双魂"：用户控制身体，AI通过感知体验游戏
 * ============================================================ */

export class GameControlBridge {
  declare App: any;
  declare userControlling: boolean;
  declare userInputActive: boolean;
  declare lastUserInputTime: number;
  declare AUTO_RECOVER_DELAY: number;
  declare _savedActions: any;       // 保存被挂起的AI动作状态
  declare _keys: any;               // 当前按下的键
  declare _moveSpeed: number;       // 游戏模式下的移动速度（统一）
  declare _moveDirection: { x: number, z: number };
  declare _aiOverrideBlocked: boolean; // 游戏模式下，AI不能覆写用户操控
  declare _mouseDragRot: number;    // 鼠标拖拽累积的旋转量（每帧消费）
  declare _DRAG_SENSITIVITY: number;  // 鼠标拖拽旋转灵敏度
  declare _PITCH_SENSITIVITY: number; // 鼠标拖拽俯仰灵敏度
  declare _virtualInput: any;       // 虚拟摇杆输入（移动端）

  constructor(app) {
    this.App = app;
    this.userControlling = false;
    this.userInputActive = false;
    this.lastUserInputTime = 0;
    this.AUTO_RECOVER_DELAY = 3000; // 用户停止输入3秒后，AI可能恢复控制
    this._savedActions = null;       // 保存被挂起的AI动作状态
    this._keys = {};                 // 当前按下的键
    this._moveSpeed = 3.5;           // 游戏模式下的移动速度（统一）
    this._moveDirection = { x: 0, z: 0 };
    this._aiOverrideBlocked = true;  // 游戏模式下，AI不能覆写用户操控
    this._mouseDragRot = 0;          // 鼠标拖拽累积的旋转量（每帧消费）
    this._DRAG_SENSITIVITY = 0.005;  // 鼠标拖拽旋转灵敏度
    this._PITCH_SENSITIVITY = 0.003; // 鼠标拖拽俯仰灵敏度
    this._virtualInput = null;       // 虚拟摇杆输入（移动端）
  }

  /**
   * 激活用户操控模式
   * 挂起AI自主动作系统
   */
  activateUserControl() {
    if (this.userControlling) return;
    this.userControlling = true;

    // 初始化摄像机方位角为角色正后方（解绑起点）：
    // 此后摄像机方位角独立于角色朝向，仅由拖拽改变；角色朝向由移动方向驱动
    const facing = this.App.smoothRotY || 0;
    this.App._gameCamAzimuth = facing + Math.PI;

    // 保存并挂起AI动作系统状态
    this._saveAIActionState();

    // 停止当前AI动作
    this.App.currentAction = null;
    this.App.nextActionTimer = 999999; // 一个很大的值，防止自动触发新动作
    this.App.idleWalkTarget = null;
    this.App.idleWalkProgress = 0;
    this.App.walkPath = [];
    this.App.walkSegmentIndex = 0;

    // 通知 AI 自主控制器：用户正在操控
    if (this.App.aiAutonomyController) {
      this.App.aiAutonomyController.notifyUserControlling();
    }

    // 通知AI：用户正在操控身体
    this._notifyAI('user_took_control', '用户开始操控身体行动');
  }

  /**
   * 释放用户操控，恢复AI自主
   */
  releaseUserControl() {
    if (!this.userControlling) return;
    this.userControlling = false;

    // 恢复AI动作系统
    this._restoreAIActionState();
    this.App.nextActionTimer = 1.5; // 很快恢复自主行动

    // 通知 AI 自主控制器：用户停止操控
    if (this.App.aiAutonomyController) {
      this.App.aiAutonomyController.notifyUserStopped();
    }

    // 通知AI：用户释放了控制
    this._notifyAI('user_released_control', '用户停止操控身体，我可以自由行动了');
  }

  /**
   * 处理键盘/触屏输入
   */
  handleKeyDown(key) {
    this._keys[key] = true;
    this.userInputActive = true;
    this.lastUserInputTime = Date.now();
  }

  handleKeyUp(key) {
    this._keys[key] = false;
    this.lastUserInputTime = Date.now();
  }

  /**
   * 每帧更新移动（游戏模式专用）
   * WASD/摇杆 = 相对摄像机视角的方向移动
   * 鼠标拖拽 = 轨道摄像机（独立于角色朝向）
   *
   * 摄像机已与角色朝向解绑：摄像机方位角(_gameCamAzimuth)仅由拖拽改变，
   * 角色朝向(smoothRotY)仅由移动方向驱动。移动方向以摄像机视角为基准计算，
   * 因摄像机不随角色转身而旋转，故无反馈打转问题。
   *
   * 本函数只负责产出 moveVec，角色朝向旋转交给 GameModeManager._applyPlayerMovement
   * 统一管道处理，与 AI 自主移动共用同一套旋转逻辑（最短角度路径插值）。
   * @returns {Object|null} 移动向量 {x, z, isMoving} 或 null
   */
  updateMovement(dt) {
    if (!this.userControlling) return null;

    let moveFwd = 0, moveRight = 0;

    // 虚拟摇杆输入（移动端）
    if (this._virtualInput && this._virtualInput.isMoving) {
      // x: -1(left)~1(right) → 左右，z: -1(down=前)~1(up=后) → 前后
      moveFwd = this._virtualInput.z || 0;
      moveRight = this._virtualInput.x || 0;
    } else {
      // 键盘输入
      if (this._keys['w'] || this._keys['arrowup']) moveFwd += 1;
      if (this._keys['s'] || this._keys['arrowdown']) moveFwd -= 1;
      if (this._keys['a'] || this._keys['arrowleft']) moveRight -= 1;
      if (this._keys['d'] || this._keys['arrowright']) moveRight += 1;
    }

    const hasInput = moveFwd !== 0 || moveRight !== 0;
    if (!hasInput) {
      if (this.lastUserInputTime > 0 && Date.now() - this.lastUserInputTime > this.AUTO_RECOVER_DELAY) {
        this.userInputActive = false;
      }
      return null;
    }

    // 归一化输入方向
    const len = Math.sqrt(moveFwd * moveFwd + moveRight * moveRight);
    const fwd = moveFwd / len;
    const right = moveRight / len;

    // 以摄像机视角为基准计算世界方向：
    // 摄像机方位角 A（绕角色的位置角），摄像机看向角色，
    // "屏幕前方"(W) = 摄像机→角色方向 = -(sinA, cosA)
    // "屏幕右方"(D) = (cosA, -sinA)
    const A = this.App._gameCamAzimuth;
    const camFwdX = -Math.sin(A);
    const camFwdZ = -Math.cos(A);
    const camRightX = Math.cos(A);
    const camRightZ = -Math.sin(A);

    const dirX = camFwdX * fwd + camRightX * right;
    const dirZ = camFwdZ * fwd + camRightZ * right;

    const speed = this._moveSpeed * dt;
    const worldX = dirX * speed;
    const worldZ = dirZ * speed;

    // 朝向旋转由 _applyPlayerMovement 统一管道处理（与 AI 共用），此处只产出位移向量
    return { x: worldX, z: worldZ, isMoving: true };
  }

  /** 移动速度设置 */
  setSpeed(speed) {
    this._moveSpeed = speed;
  }

  /** 设置虚拟摇杆输入（移动端） */
  setVirtualInput(val) {
    this._virtualInput = val;
    if (val && val.isMoving) {
      this.userInputActive = true;
      this.lastUserInputTime = Date.now();
    }
  }

  /** 鼠标拖拽：横滑环绕角色旋转摄像机，竖滑调整摄像机俯仰角 */
  addOrbitDrag(deltaX, deltaY) {
    // 水平拖拽 → 摄像机绕角色轨道旋转（独立于角色朝向）
    // 调用方传入的 deltaX 已对鼠标移动取反，这里 += 使"向右拖=摄像机向右绕"
    if (this.App._gameCamAzimuth !== undefined) {
      this.App._gameCamAzimuth += deltaX * this._DRAG_SENSITIVITY;
    }
    // 垂直拖拽 → 摄像机绕角色上下旋转
    if (this.App._gameCamPitch !== undefined) {
      this.App._gameCamPitch = Math.max(-0.5, Math.min(1.2,
        this.App._gameCamPitch + deltaY * this._PITCH_SENSITIVITY));
    }
    this.userInputActive = true;
    this.lastUserInputTime = Date.now();
  }

  /** 滚轮：拉近推远摄像机距离 */
  addZoom(deltaY) {
    if (this.App._gameCamRadius !== undefined) {
      // deltaY>0（下滑）→ 拉远，deltaY<0（上滑）→ 拉近
      this.App._gameCamRadius = Math.max(1.5, Math.min(15.0,
        this.App._gameCamRadius + deltaY * 0.01));
    }
    this.userInputActive = true;
    this.lastUserInputTime = Date.now();
  }

  /** 清理 */
  cleanup() {
    this._keys = {};
    this.userControlling = false;
    this.userInputActive = false;
    this._moveDirection = { x: 0, z: 0 };
  }

  // ==================== 内部方法 ====================

  _saveAIActionState() {
    this._savedActions = {
      nextActionTimer: this.App.nextActionTimer,
      currentAction: this.App.currentAction,
      idleWalkTarget: this.App.idleWalkTarget,
      idleWalkProgress: this.App.idleWalkProgress,
      walkPath: this.App.walkPath ? [...this.App.walkPath] : [],
      walkSegmentIndex: this.App.walkSegmentIndex,
    };
  }

  _restoreAIActionState() {
    if (!this._savedActions) return;
    // 不完全恢复旧状态（因为场景变了），只重置计时器
    this.App.nextActionTimer = Math.random() * 1 + 1;
    this._savedActions = null;
  }

  _notifyAI(eventType, description) {
    // 通过WebSocket通知AI游戏事件
    if (this.App.ws && this.App.ws.readyState === WebSocket.OPEN) {
      this.App.ws.send(JSON.stringify({
        type: 'game_event',
        event: eventType,
        data: { description }
      }));
    }
  }
}

export default GameControlBridge;