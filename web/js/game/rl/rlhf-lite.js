/* ============================================================
 * RLHFLite — 对比反馈奖励校准（P3-3，RLHF-lite）
 *
 * 目标（对应方案报告 P3-3：RLHF-lite 奖励校准）：
 * - 提供"好/坏片段"对比反馈通道（用户标记 → 偏好数据）
 * - 用 Bradley-Terry 偏好模型学习片段级偏好
 * - 输出奖励修正项（reward shaping），修正手写奖励覆盖不到的行为
 *
 * 结构：
 * - 片段（segment）：一段 (s, a, r) 轨迹
 * - 偏好对（pair）：{good: segment, bad: segment}
 * - 偏好模型：线性 logit = w·φ(segment)，φ 为片段聚合特征
 *   （均值状态 + 均值动作 one-hot + 均值奖励 + 方差）
 * - 训练：Bradley-Terry 交叉熵梯度上升
 * - 应用：shaping(stateVec, actionId) → 归一化偏好分数偏移
 *
 * 验收指标：
 *   - 好片段得分 > 坏片段得分（偏好模型学到对比信号）
 *   - 手写奖励未覆盖的行为能被修正（shaping 项可测、可叠加）
 * ============================================================ */

import { oneHot } from './world-model.js';

// ==================== 常量 ====================

const FEATURE_DIM_EXTRA = 3;              // [meanReward, rewardVar, length]
const PREF_LR = 0.1;                      // 偏好模型学习率
const PREF_EPOCHS = 12;                   // 每批训练轮数
const MAX_PAIRS = 500;                    // 偏好对池上限
const SHAPING_SCALE = 0.5;                // shaping 输出缩放
const L2 = 0.01;                          // 权重衰减

// ==================== 片段特征 ====================

/**
 * 把片段聚合为偏好模型特征向量
 * @param {Object} seg {s: Float64Array|number[], a: number|number[], r: number|number[]}
 * @param {number} stateSize
 * @param {number} nActions
 * @returns {Float64Array}
 */
export function segmentFeature(seg, stateSize, nActions) {
  const sArr = seg.s || [];
  const n = Math.min(stateSize, sArr.length || stateSize);

  // 均值状态
  const meanS = new Float64Array(n);
  // 动作 one-hot 均值
  const meanA = new Float64Array(nActions);
  // 奖励统计
  let meanR = 0, varR = 0;
  const rArr = seg.r;
  const rList = Array.isArray(rArr) ? rArr : [rArr];

  let count = Math.max(1, (Array.isArray(seg.s) ? seg.s.length : 0));
  if (Array.isArray(seg.s) && seg.s.length) {
    for (let i = 0; i < n; i++) {
      meanS[i] = sArr[i] / count;
    }
  } else if (sArr.length >= n) {
    for (let i = 0; i < n; i++) meanS[i] = sArr[i];
  }

  const aArr = Array.isArray(seg.a) ? seg.a : [seg.a];
  for (const a of aArr) {
    if (a >= 0 && a < nActions) meanA[a] += 1 / Math.max(1, aArr.length);
  }

  for (const r of rList) meanR += Number(r) || 0;
  meanR /= Math.max(1, rList.length);
  for (const r of rList) varR += (Number(r) - meanR) ** 2;
  varR /= Math.max(1, rList.length);

  const feat = new Float64Array(n + nActions + FEATURE_DIM_EXTRA);
  for (let i = 0; i < n; i++) feat[i] = meanS[i];
  for (let i = 0; i < nActions; i++) feat[n + i] = meanA[i];
  feat[n + nActions] = meanR;
  feat[n + nActions + 1] = varR;
  feat[n + nActions + 2] = Math.min(1, rList.length / 60);   // 片段长度归一化
  return feat;
}

// ==================== RLHFLite ====================

export class RLHFLite {
  /**
   * @param {Object} opts
   * @param {number} opts.stateSize
   * @param {number} opts.nActions
   */
  constructor(opts = {}) {
    this.stateSize = opts.stateSize ?? 0;
    this.nActions = opts.nActions ?? 0;
    if (!this.stateSize || !this.nActions) {
      throw new Error('RLHFLite: stateSize/nActions 必须为正');
    }
    this.featDim = this.stateSize + this.nActions + FEATURE_DIM_EXTRA;

    /** 偏好权重向量（Bradley-Terry logit 参数） */
    this.w = new Float64Array(this.featDim).fill(0.001);
    this.pairs = [];
    this.stats = { pairs: 0, updates: 0, trainLoss: 0, sep: 0 };
  }

  // ==================== 数据 ====================

  /**
   * 记录一条对比反馈
   * @param {Object} good 好片段 {s, a, r}
   * @param {Object} bad  坏片段 {s, a, r}
   */
  addPair(good, bad) {
    if (!good || !bad) return false;
    const g = this._feat(good);
    const b = this._feat(bad);
    if (!g || !b) return false;
    this.pairs.push({ g, b });
    if (this.pairs.length > MAX_PAIRS) this.pairs.shift();
    this.stats.pairs = this.pairs.length;
    return true;
  }

  /**
   * 批量添加对比反馈（离线校准服务）
   * @param {Array<{good:Object,bad:Object}>} list
   */
  addPairs(list = []) {
    let n = 0;
    for (const p of list) {
      if (this.addPair(p.good, p.bad)) n++;
    }
    return n;
  }

  /** 直接以片段特征对添加（供 Python 校准结果导入） */
  addFeaturePair(gFeat, bFeat) {
    this.pairs.push({ g: Float64Array.from(gFeat), b: Float64Array.from(bFeat) });
    if (this.pairs.length > MAX_PAIRS) this.pairs.shift();
    this.stats.pairs = this.pairs.length;
  }

  _feat(seg) {
    if (!seg || !seg.s) return null;
    return segmentFeature(seg, this.stateSize, this.nActions);
  }

  // ==================== 训练（Bradley-Terry） ====================

  /**
   * 训练偏好模型
   * @param {number} [epochs=PREF_EPOCHS]
   * @returns {{loss, sep}} sep = 好-坏平均得分差（验收指标）
   */
  train(epochs = PREF_EPOCHS) {
    const n = this.pairs.length;
    if (n < 1) return { loss: 0, sep: 0 };
    let totalLoss = 0;
    for (let ep = 0; ep < epochs; ep++) {
      const p = this.pairs[(Math.random() * n) | 0];
      const gLogit = this._logit(p.g);
      const bLogit = this._logit(p.b);
      // Bradley-Terry：P(good>bad) = σ(gLogit - bLogit)
      const diff = gLogit - bLogit;
      const loss = -Math.log(1 / (1 + Math.exp(-diff)));   // -log σ(diff)
      totalLoss += loss;

      // 梯度上升（最大化 P(good>bad)）
      const sig = 1 / (1 + Math.exp(-diff));
      const gGrad = (1 - sig) / Math.max(n, 1);
      for (let i = 0; i < this.featDim; i++) {
        const gw = this.w[i] + PREF_LR * (gGrad * (p.g[i] - p.b[i]) - L2 * this.w[i]);
        this.w[i] = gw;
      }
    }
    this.stats.updates++;
    this.stats.trainLoss = this.stats.trainLoss ? this.stats.trainLoss * 0.9 + (totalLoss / epochs) * 0.1 : totalLoss / epochs;
    this.stats.sep = this._measureSep();
    return { loss: totalLoss / epochs, sep: this.stats.sep };
  }

  _logit(feat) {
    let s = 0;
    for (let i = 0; i < this.featDim; i++) s += this.w[i] * feat[i];
    return s;
  }

  /** 好/坏片段平均得分差（≥0 表示模型区分出偏好） */
  _measureSep() {
    let gSum = 0, bSum = 0, cnt = 0;
    for (const p of this.pairs) {
      gSum += this._logit(p.g);
      bSum += this._logit(p.b);
      cnt++;
    }
    if (!cnt) return 0;
    return (gSum - bSum) / cnt;
  }

  // ==================== 应用 ====================

  /**
   * 奖励修正项：给定当前状态与动作，输出 shaping 分数偏移
   * @param {Float64Array|number[]} stateVec
   * @param {number} actionId
   * @param {number} [rewardHint] 可选：当前即时奖励（并入特征）
   * @returns {number} shaping 值（可加到 RL 奖励上）
   */
  shaping(stateVec, actionId, rewardHint = 0) {
    const feat = segmentFeature(
      { s: stateVec, a: actionId, r: rewardHint },
      this.stateSize, this.nActions
    );
    const raw = this._logit(feat);
    // 减去先验（初始 w≈0 时 shaping≈0，不扰动现有奖励）
    return Math.max(-1, Math.min(1, raw * SHAPING_SCALE));
  }

  /**
   * 动作偏好偏置：返回各动作的相对偏好偏移（供策略集成）
   * @returns {Float64Array|null} nActions 长度，或 null（无可区分信号）
   */
  actionBias(stateVec, rewardHint = 0) {
    if (this.stats.sep < 0.01) return null;
    const b = new Float64Array(this.nActions);
    for (let a = 0; a < this.nActions; a++) {
      b[a] = this.shaping(stateVec, a, rewardHint);
    }
    return b;
  }

  // ==================== 序列化 ====================

  toJSON() {
    return {
      version: 1, stateSize: this.stateSize, nActions: this.nActions,
      w: Array.from(this.w), stats: this.stats, pairs: this.pairs.length,
    };
  }

  fromJSON(data) {
    if (!data || !data.w) return false;
    this.stateSize = data.stateSize ?? this.stateSize;
    this.nActions = data.nActions ?? this.nActions;
    this.w = Float64Array.from(data.w);
    this.stats = { ...this.stats, ...(data.stats || {}) };
    return true;
  }
}

export default RLHFLite;
