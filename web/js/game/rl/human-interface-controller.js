/* ============================================================
 * HumanInterfaceController — 接口节奏真实化（P2-3）
 *
 * 目标（对应方案报告 P2-3）：
 * - 让 RL 决策节奏呈现人类特征：反应延迟随机抖动，而非固定周期
 * - 硬性约束：决策频率 ≤20Hz（maxRateHz，对应 VPT 20Hz 人类接口上限）
 * - 反应延迟可配置：setLatencyProfile({minMs, maxMs})
 *
 * 用法：
 *   const hic = RLAgentManager.get().getInterfaceController();
 *   // 每帧（RL 接管时）：
 *   if (hic.shouldAct(performance.now())) this._rlStep();
 *   // 评估：
 *   const rate = hic.getRateHz();          // 实际决策频率
 *   hic.setLatencyProfile({ minMs: 180, maxMs: 520 });  // 可配置
 * ============================================================ */

/** 默认反应延迟区间（人类典型 150~450ms） */
const DEFAULT_MIN_DELAY_MS = 150;
const DEFAULT_MAX_DELAY_MS = 450;
/** 决策频率上限（VPT 人类接口 20Hz） */
const DEFAULT_MAX_RATE_HZ = 20;
/** 延迟记录窗口（频率统计） */
const DELAY_WINDOW = 20;
/** 绝对最小延迟（防抖，避免 0 延迟） */
const ABS_MIN_DELAY_MS = 50;

export class HumanInterfaceController {
  /**
   * @param {Object} opts
   * @param {number} [opts.minDelayMs] 最小反应延迟（ms）
   * @param {number} [opts.maxDelayMs] 最大反应延迟（ms）
   * @param {number} [opts.maxRateHz] 决策频率上限（Hz）
   */
  constructor({ minDelayMs = DEFAULT_MIN_DELAY_MS,
                maxDelayMs = DEFAULT_MAX_DELAY_MS,
                maxRateHz = DEFAULT_MAX_RATE_HZ } = {}) {
    this.minDelayMs = minDelayMs;
    this.maxDelayMs = maxDelayMs;
    this.maxRateHz = maxRateHz;
    this._lastAct = 0;
    this._nextDelay = null;
    this._recentDelays = [];
    this._decisions = 0;
  }

  /**
   * 生成下一个反应延迟（均匀随机 [min, max]，
   * 且不低于频率下限 1000/maxRateHz）。
   * @returns {number} 延迟（ms）
   */
  nextDelay() {
    const floorMs = 1000 / Math.max(1, this.maxRateHz);
    const minMs = Math.max(ABS_MIN_DELAY_MS, Math.min(this.minDelayMs, this.maxDelayMs));
    const maxMs = Math.max(minMs, this.maxDelayMs);
    let d = minMs + Math.random() * (maxMs - minMs);
    if (d < floorMs) d = floorMs;
    return d;
  }

  /**
   * 判定当前时刻是否可执行一次决策。
   * 采用"延迟生成 + 到期触发"：上次决策后再随机等待 nextDelay() 才放行。
   * @param {number} now 当前时间（performance.now()）
   * @returns {boolean}
   */
  shouldAct(now) {
    if (!this._lastAct) { this._lastAct = now; return false; }
    if (this._nextDelay === null) this._nextDelay = this.nextDelay();
    if (now - this._lastAct >= this._nextDelay) {
      const actual = now - this._lastAct;
      this._recentDelays.push(actual);
      if (this._recentDelays.length > DELAY_WINDOW) this._recentDelays.shift();
      this._lastAct = now;
      this._nextDelay = this.nextDelay();
      this._decisions++;
      return true;
    }
    return false;
  }

  /**
   * 实际决策频率（Hz）：最近窗口的平均延迟换算。
   * @returns {number} 频率；样本不足返回 0
   */
  getRateHz() {
    if (this._recentDelays.length < 2) return 0;
    const avg = this._recentDelays.reduce((s, d) => s + d, 0) / this._recentDelays.length;
    return avg > 0 ? +(1000 / avg).toFixed(2) : 0;
  }

  /**
   * 配置反应延迟区间（可运行时调整）。
   * @param {Object} profile
   * @param {number} [profile.minMs]
   * @param {number} [profile.maxMs]
   * @param {number} [profile.maxRateHz]
   */
  setLatencyProfile({ minMs, maxMs, maxRateHz } = {}) {
    if (minMs != null) this.minDelayMs = Math.max(ABS_MIN_DELAY_MS, minMs);
    if (maxMs != null) this.maxDelayMs = Math.max(this.minDelayMs, maxMs);
    if (maxRateHz != null) this.maxRateHz = Math.max(1, maxRateHz);
    this._nextDelay = null;  // 重采样
  }

  /** 重置状态（新局） */
  reset() {
    this._lastAct = 0;
    this._nextDelay = null;
    this._recentDelays = [];
  }

  /** 累计决策次数 */
  get decisionCount() { return this._decisions; }
}

export default HumanInterfaceController;
