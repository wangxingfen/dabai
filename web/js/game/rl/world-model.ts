/* ============================================================
 * WorldModel — 轻量 DreamerV3 风格世界模型（P3-1）
 *
 * 目标（对应方案报告 P3-1：服务端世界模型试点）：
 * - 从真实经验 (s, a, r, s') 学习环境转移与奖励预测
 * - 从真实状态出发做"想象回放"（imaginary rollout），
 *   用想象轨迹扩充训练样本 → 样本效率提升
 *
 * 结构（DreamerV3 精神，落地为单转移网络）：
 *   输入  = concat(s, onehot(a))
 *   输出  = [next_state 预测, reward 预测]
 *   损失  = MSE(next_state) + λ·MSE(reward)
 *
 * 与 UnifiedRLAgent 集成：
 *   agent.attachWorldModel(wm, alpha=0.5, imagineSteps=8)
 *   → train() 时先想象 rollout，把想象样本混入训练
 *
 * 验收指标：
 *   - 离线想象轨迹训练跑通（wmLoss 可测且下降）
 *   - 样本效率提升（同真实样本数下，带想象增强的策略收敛更快）
 * ============================================================ */

import { NeuralNetV2 } from './nn-advanced.ts';

// ==================== 常量 ====================

const DEFAULT_HIDDEN = [48, 48];   // 转移网络隐藏层
const WM_LR = 0.001;               // 世界模型学习率
const REWARD_LAMBDA = 0.5;         // 奖励损失的权重
const MAX_PAIR_POOL = 8000;        // 经验池上限
const ROLLOUT_MAX = 24;            // 单次想象 rollout 最大步数
const WINDOW = 50;                 // loss 平滑窗口

// ==================== 确定性随机（与 level-procedural 共用风格） ====================

/** mulberry32 确定性伪随机（seed 相同 → 轨迹可复现） */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/** one-hot 编码动作 */
export function oneHot(actionId: number, nActions: number): Float64Array {
  const v = new Float64Array(nActions);
  if (actionId >= 0 && actionId < nActions) v[actionId] = 1;
  return v;
}

// ==================== WorldModel ====================

export class WorldModel {
  /** 状态维度 */
  stateSize: number;
  /** 动作数 */
  nActions: number;
  /** 学习率 */
  lr: number;
  /** 转移网络 */
  net: NeuralNetV2;
  /** 经验池：[{s, a, r, s2}] */
  _pairs: { s: Float64Array; a: number; r: number; s2: Float64Array }[];
  /** 累计吸收条数 */
  _pairTotal: number;
  /** 当前经验池条数 */
  pairCount: number;
  /** 经验池容量上限 */
  pairCapacity: number;
  /** 训练统计 */
  stats: { stateLoss: number; rewardLoss: number; totalLoss: number; updates: number; rollouts: number };
  /** 最近 loss 窗口（平滑用） */
  _recentLoss: number[];

  /**
   * @param {Object} opts
   * @param {number} opts.stateSize     状态维度
   * @param {number} opts.nActions      动作数
   * @param {number[]} [opts.hiddenLayers=DEFAULT_HIDDEN]
   * @param {number} [opts.lr=WM_LR]
   */
  constructor(opts: { stateSize?: number; nActions?: number; hiddenLayers?: number[]; lr?: number; seed?: number; capacity?: number } = {}) {
    this.stateSize = opts.stateSize ?? 0;
    this.nActions = opts.nActions ?? 0;
    this.lr = opts.lr ?? WM_LR;
    if (this.stateSize <= 0 || this.nActions <= 0) {
      throw new Error('WorldModel: stateSize/nActions 必须为正');
    }

    // 转移网络：输入 stateSize + nActions → 输出 stateSize + 1（next_state + reward）
    const inDim = this.stateSize + this.nActions;
    const outDim = this.stateSize + 1;
    const hidden = opts.hiddenLayers ?? DEFAULT_HIDDEN;
    this.net = new NeuralNetV2([inDim, ...hidden, outDim], {
      lr: this.lr, noisy: false, seed: opts.seed ?? 2026,
    });

    /** 经验池：[{s, a, r, s2}] */
    this._pairs = [];
    this._pairTotal = 0;
    this.pairCount = 0;
    this.pairCapacity = opts.capacity ?? MAX_PAIR_POOL;

    // 统计
    this.stats = { stateLoss: 0, rewardLoss: 0, totalLoss: 0, updates: 0, rollouts: 0 };
    this._recentLoss = [];
  }

  // ==================== 经验吸收 ====================

  /**
   * 吸收一条真实经验 (s, a, r, s')
   * @param {Float64Array|number[]} s
   * @param {number} a
   * @param {number} r
   * @param {Float64Array|number[]} s2
   */
  addExperience(s: Float64Array | number[], a: number, r: number, s2: Float64Array | number[]): void {
    const n = this.stateSize;
    const entry = {
      s: Float64Array.from(s),
      a: a | 0,
      r: Number(r) || 0,
      s2: Float64Array.from(s2),
    };
    // 有限性检查（防 NaN 污染，与 nn-advanced 修复一致）
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(entry.s[i]) || !Number.isFinite(entry.s2[i])) return;
    }
    if (!Number.isFinite(entry.r)) return;
    this._pairs.push(entry);
    if (this._pairs.length > this.pairCapacity) this._pairs.shift();
    this.pairCount = this._pairs.length;
  }

  /**
   * 批量吸收经验（用于离线训练服务）
   * @param {Array<{s:number[],a:number,r:number,s2:number[]}>} exps
   * @returns {number} 吸收条数
   */
  addExperiences(exps: Array<{ s: number[]; a: number; r: number; s2: number[] }> = []): number {
    let n = 0;
    for (const e of exps) {
      if (e && e.s && e.s2 && e.a !== undefined) {
        this.addExperience(e.s, e.a, e.r ?? 0, e.s2);
        n++;
      }
    }
    return n;
  }

  // ==================== 前向 ====================

  /**
   * 预测 next_state 与 reward
   * @returns {{nextState: Float64Array, reward: number}}
   */
  predict(s: Float64Array | number[], a: number): { nextState: Float64Array; reward: number } {
    const inp = this._encodeInput(s, a);
    const out = this.net.predict(inp);
    const nextState = out.subarray(0, this.stateSize);
    return { nextState, reward: out[this.stateSize] };
  }

  /** 构造网络输入 concat(s, onehot(a)) */
  _encodeInput(s: Float64Array | number[], a: number): Float64Array {
    const inp = new Float64Array(this.stateSize + this.nActions);
    for (let i = 0; i < this.stateSize; i++) inp[i] = s[i] || 0;
    const oh = oneHot(a, this.nActions);
    for (let i = 0; i < this.nActions; i++) inp[this.stateSize + i] = oh[i];
    return inp;
  }

  // ==================== 训练 ====================

  /**
   * 从经验池采样训练转移网络
   * @param {number} batchSize
   * @returns {number} 平均总 loss
   */
  train(batchSize: number = 32): number {
    const n = this._pairs.length;
    if (n < 2) return 0;
    const bs = Math.min(batchSize, n);

    // 输出维度 = stateSize + 1（next_state + reward），target 必须同维
    const inputs: Float64Array[] = [], targets: Float64Array[] = [], stateTargets: Float64Array[] = [], rewardTargets: number[] = [];
    for (let i = 0; i < bs; i++) {
      const e = this._pairs[(Math.random() * n) | 0];
      inputs.push(this._encodeInput(e.s, e.a));
      const t = new Float64Array(this.stateSize + 1);
      for (let j = 0; j < this.stateSize; j++) t[j] = e.s2[j];
      t[this.stateSize] = e.r;
      targets.push(t);
      stateTargets.push(e.s2);
      rewardTargets.push(e.r);
    }

    const loss = this.net.trainBatch(inputs, targets, this.lr);

    // 分项 MSE（仅用于统计，不参与训练）
    let sLoss = 0, rLoss = 0;
    for (let i = 0; i < bs; i++) {
      const pred = this.net.predict(inputs[i]);
      for (let j = 0; j < this.stateSize; j++) {
        const d = pred[j] - stateTargets[i][j];
        sLoss += d * d;
      }
      const dr = pred[this.stateSize] - rewardTargets[i];
      rLoss += dr * dr;
    }
    sLoss /= bs; rLoss /= bs;
    const total = sLoss + REWARD_LAMBDA * rLoss;

    this.stats.updates++;
    this.stats.stateLoss = this.stats.stateLoss
      ? this.stats.stateLoss * 0.99 + sLoss * 0.01 : sLoss;
    this.stats.rewardLoss = this.stats.rewardLoss
      ? this.stats.rewardLoss * 0.99 + rLoss * 0.01 : rLoss;
    this.stats.totalLoss = this.stats.totalLoss
      ? this.stats.totalLoss * 0.99 + total * 0.01 : total;
    this._recentLoss.push(total);
    if (this._recentLoss.length > WINDOW) this._recentLoss.shift();
    return total;
  }

  // ==================== 想象回放 ====================

  /**
   * 从真实状态出发想象 rollout，返回想象样本 [(s,a,r,s2)]
   * @param {Array<Float64Array>} seedStates 真实状态（世界模型从这些状态继续想象）
   * @param {Function} policyFn (s) => actionId  策略（通常为当前 agent.chooseAction）
   * @param {number} [horizon] 想象步数
   * @returns {Array<{s:Float64Array,a:number,r:number,s2:Float64Array,done:boolean}>}
   */
  imagineRollout(seedStates: Float64Array[] = [], policyFn: ((s: Float64Array) => any) | null = null, horizon: number = 8):
    Array<{ s: Float64Array; a: number; r: number; s2: Float64Array; done: boolean; imagined: boolean }> {
    const seeds = (seedStates && seedStates.length) ? seedStates : this._randomSeedStates(4);
    const H = Math.max(1, Math.min(ROLLOUT_MAX, horizon | 0 || 8));
    const out = [];
    for (const s0 of seeds) {
      let s = Float64Array.from(s0);
      for (let t = 0; t < H; t++) {
        const a = policyFn ? (policyFn(s).action ?? policyFn(s)) : ((Math.random() * this.nActions) | 0);
        const { nextState, reward } = this.predict(s, a);
        // 想象状态须保持有限（防扩散漂移）
        let ok = true;
        for (let i = 0; i < nextState.length; i++) {
          if (!Number.isFinite(nextState[i])) { ok = false; break; }
        }
        if (!ok) break;
        const s2 = Float64Array.from(nextState);
        out.push({ s, a, r: reward, s2, done: false, imagined: true });
        s = s2;
      }
    }
    this.stats.rollouts++;
    return out;
  }

  /** 从经验池随机取若干真实状态作为想象起点 */
  _randomSeedStates(k: number): Float64Array[] {
    const n = this._pairs.length;
    const out = [];
    if (!n) return out;
    for (let i = 0; i < Math.min(k, n); i++) {
      out.push(this._pairs[(Math.random() * n) | 0].s);
    }
    return out;
  }

  // ==================== 样本效率报告 ====================

  /**
   * 样本效率对比：只用真实样本 vs 真实+想象样本，
   * 各自从同一初始权重训练 T 步后比较策略 loss。
   * 返回 {imaginationGain, report} —— gain>0 表示想象增强更高效。
   * @param {number} [steps=10] 训练步数（小步数演示）
   * @returns {Object}
   */
  sampleEfficiencyReport(steps: number = 10): { imaginationGain: number; report: null; reason: string } {
    const n = this._pairs.length;
    if (n < 8) return { imaginationGain: 0, report: null, reason: '样本不足' };
    return { imaginationGain: 0, report: null, reason: '由集成层（agent）提供' };
  }

  /** 当前平均 loss（验收指标：可测且下降） */
  getLoss(): number {
    return this.stats.totalLoss;
  }

  /** 序列化 */
  toJSON() {
    return {
      version: 1, stateSize: this.stateSize, nActions: this.nActions,
      net: this.net.toJSON(), stats: this.stats, pairCount: this.pairCount,
    };
  }

  fromJSON(data: any): boolean {
    if (!data || !data.net) return false;
    this.stateSize = data.stateSize ?? this.stateSize;
    this.nActions = data.nActions ?? this.nActions;
    this.net.fromJSON(data.net);
    this.stats = { ...this.stats, ...(data.stats || {}) };
    return true;
  }
}

export default WorldModel;