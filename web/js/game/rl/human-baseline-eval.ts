/* ============================================================
 * HumanBaselineEvaluator — 人类限时评估基准（P2-2）
 *
 * 目标（对应方案报告 P2-2，SIMA 2 评估协议本地化）：
 * - 加载 humanBaseline.json（人类完成任务的平均时间与成功率）
 * - 追踪 RL 每局完成时间与胜负
 * - 报告"相对人类完成率" = AI成功率 / 人类成功率（达标线 ≥60%）
 *
 * 设计：
 * - fetch 静态 JSON，失败自动回退内置默认值（离线可用）
 * - 每游戏维护最近 N 局窗口（默认 30），滚动统计
 * - RLAgentManager.getAllStats() 中每个智能体附 eval 报告
 *
 * 用法：
 *   const ev = RLAgentManager.get().getBaselineEvaluator();
 *   ev.recordEpisode('treasure_hunt', { durationMs: 45200, win: true });
 *   const r = ev.report('treasure_hunt');  // {relativeSuccessRate, pass, ...}
 * ============================================================ */

/** 达标线：相对人类完成率 ≥ 60% */
export const PASS_THRESHOLD = 0.6;
/** 默认人类基线（fetch 失败时回退） */
export const DEFAULT_HUMAN_BASELINES = {
  treasure_hunt: { avgTimeSec: 90, successRate: 0.8 },
  sandbox: { avgTimeSec: 120, successRate: 0.7 },
  mario: { avgTimeSec: 60, successRate: 0.65 },
  moba_5v5: { avgTimeSec: 300, successRate: 0.5 },
};
/** 滚动窗口大小 */
const DEFAULT_WINDOW = 30;
/** 每游戏最多保留的历史局数 */
const MAX_EPISODES = 200;

export class HumanBaselineEvaluator {
  /** gameKey -> {avgTimeSec, successRate} */
  _baselines: Record<string, { avgTimeSec: number; successRate: number }>;
  /** gameKey -> [{durationMs, win, ts}] */
  _episodes: Record<string, { durationMs: number; win: boolean; ts: number }[]>;
  /** gameKey -> 最近报告 */
  _lastReport: Record<string, any>;
  /** 是否已成功加载外部基线 */
  _loaded: boolean;

  constructor() {
    /** gameKey -> {avgTimeSec, successRate} */
    this._baselines = {};
    for (const [k, v] of Object.entries(DEFAULT_HUMAN_BASELINES)) {
      this._baselines[k] = { ...v };
    }
    /** gameKey -> [{durationMs, win, ts}] */
    this._episodes = {};
    /** gameKey -> 最近报告 */
    this._lastReport = {};
    this._loaded = false;
  }

  /**
   * 加载人类基线（fetch humanBaseline.json）。
   * 失败时保留内置默认值（离线可用）。
   * @returns {Promise<Object>} 基线表
   */
  async loadBaseline(url: string = 'humanBaseline.json'): Promise<Record<string, { avgTimeSec: number; successRate: number }>> {
    try {
      const resp = await fetch(url, { cache: 'no-store' });
      if (resp.ok) {
        const data = await resp.json();
        if (data && typeof data === 'object') {
          for (const key of Object.keys(this._baselines)) {
            if (data[key]) {
              this._baselines[key] = {
                ...this._baselines[key],
                ...data[key],
              };
            }
          }
          this._loaded = true;
          console.log('[HumanBaseline] 已加载人类基线:', this._baselines);
        }
      }
    } catch (e) {
      console.warn('[HumanBaseline] 加载失败，使用内置默认基线:', e.message);
    }
    return this._baselines;
  }

  /** 是否已成功加载外部基线 */
  get loaded(): boolean { return this._loaded; }

  /** 人类基线表 */
  get baselines(): Record<string, { avgTimeSec: number; successRate: number }> { return this._baselines; }

  /**
   * 记录一局 RL 评估结果。
   * @param {string} gameKey
   * @param {Object} [opts]
   * @param {number} [opts.durationMs] 本局耗时（ms）
   * @param {boolean} [opts.win] 是否完成目标
   */
  recordEpisode(gameKey: string, { durationMs = 0, win = false }: { durationMs?: number; win?: boolean } = {}): void {
    if (!this._episodes[gameKey]) this._episodes[gameKey] = [];
    this._episodes[gameKey].push({ durationMs, win: !!win, ts: Date.now() });
    if (this._episodes[gameKey].length > MAX_EPISODES) {
      this._episodes[gameKey].shift();
    }
  }

  /** 某游戏已记录的局数 */
  episodeCount(gameKey: string): number {
    return (this._episodes[gameKey] || []).length;
  }

  /**
   * 生成评估报告（滚动窗口）。
   * @param {string} gameKey
   * @param {Object} [opts]
   * @param {number} [opts.windowSize] 窗口局数
   * @returns {Object} 报告；无数据时 available=false
   */
  report(gameKey: string, { windowSize = DEFAULT_WINDOW }: { windowSize?: number } = {}): any {
    const base = this._baselines[gameKey];
    const eps = (this._episodes[gameKey] || []).slice(-windowSize);
    if (!base || !eps.length) {
      const out = {
        gameKey,
        available: false,
        episodes: eps.length,
        humanSuccessRate: base ? base.successRate : null,
        humanAvgTimeSec: base ? base.avgTimeSec : null,
        aiSuccessRate: null,
        aiAvgTimeSec: null,
        relativeSuccessRate: null,
        relativeTime: null,
        pass: false,
      };
      this._lastReport[gameKey] = out;
      return out;
    }

    const aiSuccessRate = eps.reduce((s, e) => s + (e.win ? 1 : 0), 0) / eps.length;
    const aiAvgTimeSec = eps.reduce((s, e) => s + e.durationMs, 0) / eps.length / 1000;
    const relativeSuccessRate = base.successRate > 0
      ? aiSuccessRate / base.successRate : 0;
    const relativeTime = base.avgTimeSec > 0 ? aiAvgTimeSec / base.avgTimeSec : null;

    const out = {
      gameKey,
      available: true,
      episodes: eps.length,
      humanSuccessRate: base.successRate,
      humanAvgTimeSec: base.avgTimeSec,
      aiSuccessRate: +aiSuccessRate.toFixed(3),
      aiAvgTimeSec: +aiAvgTimeSec.toFixed(1),
      relativeSuccessRate: +relativeSuccessRate.toFixed(3),
      relativeTime: relativeTime != null ? +relativeTime.toFixed(2) : null,
      pass: relativeSuccessRate >= PASS_THRESHOLD,
    };
    this._lastReport[gameKey] = out;
    return out;
  }

  /** 全部游戏报告 */
  allReports(): any[] {
    return Object.keys(this._baselines).map((k) => this.report(k));
  }

  /** 最近一次报告（无则生成） */
  lastReport(gameKey: string): any {
    return this._lastReport[gameKey] || this.report(gameKey);
  }

  /** 达标线 */
  static get PASS_THRESHOLD(): number { return PASS_THRESHOLD; }
}

export default HumanBaselineEvaluator;