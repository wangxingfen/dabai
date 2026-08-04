/* ============================================================
 * HumanTrajectoryRecorder — 人类轨迹采集器（P2-1a）
 *
 * 目标（对应方案报告 P2-1）：
 * - 采集"用户真实操控"时的 (观察, 世界方向输入) 序列，作为人类行为先验数据
 * - 存入 IndexedDB（replays store，key: human_<gameKey>_<ts>）
 * - 供 BehaviorCloningPrior 构建 IDM 风格行为先验（AI 行为向人类对齐）
 *
 * 隐私与合规（对应报告 6. 风险）：仅采集本机演示数据，帧内不含用户身份信息；
 * 可通过 stopRecording 随时停止；数据仅存本地 IndexedDB，不上传。
 *
 * 用法：
 *   const rec = HumanTrajectoryRecorder.get();
 *   rec.startRecording('treasure_hunt');
 *   // 每帧（用户操控时）：
 *   rec.recordFrame(game, bridge);
 *   rec.stopRecording();  // 保存为一条轨迹
 * ============================================================ */

import { RLPersistence } from '../game/rl/rl-persistence.js';

/** 帧降采样间隔（ms）：与 RL 决策节奏同量级，避免冗余帧 */
const FRAME_MIN_INTERVAL_MS = 100;
/** 最短有效轨迹帧数（少于则丢弃，视为无效片段） */
const MIN_FRAMES = 5;

/**
 * 从桥接器按键状态推断用户输入方向（世界坐标系）。
 * WASD/方向键 = 相对摄像机视角；用 _gameCamAzimuth 旋转为世界方向
 * （与 GameControlBridge.updateMovement 的变换完全一致）。
 * @param {Object} bridge - GameControlBridge 实例
 * @param {Object} App - 全局 App（提供 _gameCamAzimuth）
 * @returns {{x:number, z:number}|null} 归一化世界方向；无输入返回 null
 */
export function inputVecFromBridge(bridge, App) {
  if (!bridge || !bridge._keys) return null;
  const k = bridge._keys;
  let right = 0, fwd = 0;
  if (k['w'] || k['arrowup']) fwd += 1;
  if (k['s'] || k['arrowdown']) fwd -= 1;
  if (k['a'] || k['arrowleft']) right -= 1;
  if (k['d'] || k['arrowright']) right += 1;

  // 虚拟摇杆（移动端）
  if (bridge._virtualInput && bridge._virtualInput.isMoving) {
    right = bridge._virtualInput.x || 0;
    fwd = bridge._virtualInput.z || 0;
  }

  if (!right && !fwd) return null;
  const len = Math.sqrt(right * right + fwd * fwd);
  right /= len; fwd /= len;

  // 摄像机相对 → 世界方向（与 updateMovement 一致）
  const A = (App && App._gameCamAzimuth) || 0;
  const camFwdX = -Math.sin(A);
  const camFwdZ = -Math.cos(A);
  const camRightX = Math.cos(A);
  const camRightZ = -Math.sin(A);
  return {
    x: camFwdX * fwd + camRightX * right,
    z: camFwdZ * fwd + camRightZ * right,
  };
}

export class HumanTrajectoryRecorder {
  static _instance = null;

  /** 全局单例 */
  static get() {
    if (!HumanTrajectoryRecorder._instance) {
      HumanTrajectoryRecorder._instance = new HumanTrajectoryRecorder();
    }
    return HumanTrajectoryRecorder._instance;
  }

  constructor() {
    this._persistence = new RLPersistence();
    this._recording = null;   // {gameKey, frames:[], startTs}
    this._active = false;
    this._lastFrameTs = 0;
    this._sessionCount = 0;
  }

  /** 开始采集 */
  startRecording(gameKey) {
    if (this._active) return false;
    this._active = true;
    this._lastFrameTs = 0;
    this._recording = {
      gameKey,
      frames: [],
      startTs: performance.now(),
    };
    return true;
  }

  /** 记录一帧（仅在用户操控期间调用） */
  recordFrame(game, bridge) {
    if (!this._active || !this._recording) return;
    const now = performance.now();
    if (now - this._lastFrameTs < FRAME_MIN_INTERVAL_MS) return;
    const vec = inputVecFromBridge(bridge, game && game.App);
    if (!vec) return;  // 无输入帧不记录（与 RL 动作空间对齐）

    this._lastFrameTs = now;
    let stateVec = null;
    try {
      if (game && typeof game.getObservation === 'function') {
        const obs = game.getObservation();
        stateVec = obs ? Array.from(obs) : null;
      }
    } catch (e) {
      stateVec = null;
    }
    this._recording.frames.push({
      t: +(now - this._recording.startTs).toFixed(0),
      s: stateVec,
      v: { x: +vec.x.toFixed(3), z: +vec.z.toFixed(3) },
    });
  }

  /** 结束并保存当前轨迹；返回存储 key 或 null（太短） */
  stopRecording() {
    if (!this._active || !this._recording) return null;
    const rec = this._recording;
    this._active = false;
    this._recording = null;
    this._lastFrameTs = 0;

    if (rec.frames.length < MIN_FRAMES) return null;
    const key = `human_${rec.gameKey}_${Date.now()}`;
    this._sessionCount++;
    const payload = {
      type: 'human_trajectory',
      gameKey: rec.gameKey,
      durationMs: rec.frames[rec.frames.length - 1].t,
      frames: rec.frames,
      ts: Date.now(),
    };
    this._persistence.save('replays', key, payload)
      .catch((e) => console.warn('[HumanTrajectory] 保存失败:', e.message));
    console.log(`[HumanTrajectory] 已保存轨迹: ${key} (${rec.frames.length} 帧)`);
    return key;
  }

  /** 放弃当前轨迹（不保存） */
  cancelRecording() {
    this._active = false;
    this._recording = null;
    this._lastFrameTs = 0;
  }

  /** 是否正在采集 */
  isRecording() { return this._active; }

  /** 本会话已保存轨迹数 */
  get sessionCount() { return this._sessionCount; }

  /**
   * 加载某游戏的全部人类轨迹（按时间升序）。
   * @param {string} gameKey
   * @returns {Promise<Array<{gameKey, durationMs, frames, ts}>>}
   */
  async getTrajectories(gameKey) {
    try {
      const keys = await this._persistence.listKeys('replays');
      const humanKeys = (keys || []).filter((k) =>
        typeof k === 'string' && k.startsWith('human_' + gameKey + '_'));
      if (!humanKeys.length) return [];
      const map = await this._persistence.loadMultiple('replays', humanKeys);
      const out = [];
      for (const k of humanKeys) {
        const d = map.get(k);
        if (d && d.type === 'human_trajectory' && Array.isArray(d.frames)) out.push(d);
      }
      return out.sort((a, b) => (a.ts || 0) - (b.ts || 0));
    } catch (e) {
      console.warn('[HumanTrajectory] 加载轨迹失败:', e.message);
      return [];
    }
  }

  /** 统计某游戏的人类轨迹帧数 */
  async countFrames(gameKey) {
    const trajs = await this.getTrajectories(gameKey);
    return trajs.reduce((s, t) => s + t.frames.length, 0);
  }
}

export default HumanTrajectoryRecorder;
