/* ============================================================
 * RLAgentManager — 统一 RL 智能体管理器（P1-1 / P1-3）
 *
 * 目标：
 * 1. 收敛到单一 UnifiedRLAgent 体系：所有游戏通过本管理器获取智能体，
 *    超参由 games-config.js 注册表统一提供（消灭各游戏私建实例与硬编码超参）
 * 2. 跨游戏 warm start：把已训练游戏的最佳策略权重迁移给目标游戏
 *    （结构兼容时才迁移，不兼容则记录原因）
 *
 * 用法：
 *   import { RLAgentManager } from './rl-agent-manager.js';
 *   const agent = RLAgentManager.get().getAgent('mario', this);
 * ============================================================ */

import { UnifiedRLAgent } from './unified-rl-agent.ts';
import { getGameConfig } from "./../games/games-config.ts";
import { HumanInterfaceController } from "./human-interface-controller.ts";
import { HumanBaselineEvaluator } from "./human-baseline-eval.ts";
import { HumanTrajectoryRecorder } from "./../../input/human-trajectory-recorder.ts";
import { BehaviorCloningPrior } from "./behavior-cloning.ts";
import { WorldModel } from "./world-model.ts";
import { RLHFLite } from "./rlhf-lite.ts";

export class RLAgentManager {
  static _instance = null;

  /** 全局单例 */
  static get() {
    if (!RLAgentManager._instance) RLAgentManager._instance = new RLAgentManager();
    return RLAgentManager._instance;
  }

  constructor() {
    /** gameKey -> UnifiedRLAgent */
    this._agents = new Map();
    /** warm start 迁移日志 */
    this._warmStartLog = [];
    // P2 共享设施（惰性创建）
    this._interface = null;       // P2-3 接口节奏真实化
    this._evaluator = null;       // P2-2 人类限时评估基准
    this._recorder = null;        // P2-1a 人类轨迹采集器
    // P3 共享设施（惰性创建）
    this._worldModels = new Map(); // gameKey -> WorldModel
    this._rlhfs = new Map();       // gameKey -> RLHFLite
    this._zeroShotEval = null;     // P3-2 零样本评测器
  }

  /**
   * 获取（或惰性创建）指定游戏的智能体。
   * 超参优先级：注册表 hyperparams < 游戏 getRLHyperparams() 覆盖
   * @param {string} gameKey - 游戏 key（games-config.js）
   * @param {Object|null} game - 游戏实例（用于读取超参覆盖）
   * @returns {UnifiedRLAgent|null} 未启用 RL 的游戏返回 null
   */
  getAgent(gameKey, game = null) {
    const cfg = getGameConfig(gameKey);
    if (!cfg || !cfg.rl || !cfg.rl.enabled) return null;
    if (this._agents.has(gameKey)) return this._agents.get(gameKey);

    const hp = cfg.rl.hyperparams || {};
    const overrides = (game && typeof game.getRLHyperparams === 'function')
      ? (game.getRLHyperparams() || {}) : {};
    const agent = new UnifiedRLAgent({
      mode: cfg.mode,
      stateSize: cfg.rl.stateSize,
      nActions: cfg.rl.nActions,
      storageKey: cfg.rl.storageKey,
      ...hp,
      ...overrides,
    });
    this._agents.set(gameKey, agent);
    console.log(`[RLAgentManager] 创建智能体: ${gameKey} (${cfg.rl.storageKey}, ${cfg.rl.stateSize}D/${cfg.rl.nActions}A)`);
    return agent;
  }

  /** 指定游戏是否已有智能体实例 */
  hasAgent(gameKey) { return this._agents.has(gameKey); }

  /** 按存储键查找智能体（用于外部模块定位） */
  getAgentByStorageKey(storageKey) {
    for (const agent of this._agents.values()) {
      if (agent.storageKey === storageKey) return agent;
    }
    return null;
  }

  /**
   * 跨游戏 warm start（P1-3）：把源游戏最佳权重迁移到目标游戏。
   * @param {string} fromKey - 源游戏 key（须已创建智能体并训练过）
   * @param {string} toKey   - 目标游戏 key
   * @param {Object|null} game - 目标游戏实例
   * @returns {Promise<{ok:boolean, reason?:string}>}
   */
  async warmStart(fromKey, toKey, game = null): Promise<{ ok: boolean; reason?: string; src?: number[]; dst?: number[] }> {
    const src = this._agents.get(fromKey);
    if (!src) return { ok: false, reason: 'source_not_loaded' };
    const dst = this.getAgent(toKey, game);
    if (!dst) return { ok: false, reason: 'target_not_enabled' };

    const srcLayers = JSON.stringify(src.onlineNet.layers);
    const dstLayers = JSON.stringify(dst.onlineNet.layers);
    if (srcLayers !== dstLayers) {
      return {
        ok: false, reason: 'structure_mismatch',
        src: src.onlineNet.layers, dst: dst.onlineNet.layers,
      };
    }
    try {
      const data = await src._persistence.load(src.storageKey + '_best');
      if (!data || !data.net) return { ok: false, reason: 'no_best_weight' };
      dst.onlineNet.fromJSON(data.net);
      dst.targetNet.copyFrom(dst.onlineNet);
      await dst.flush();
      this._warmStartLog.push({ from: fromKey, to: toKey, ts: Date.now() });
      console.log(`[RLAgentManager] warm start: ${fromKey} -> ${toKey}`);
      return { ok: true };
    } catch (e) {
      return { ok: false, reason: e.message };
    }
  }

  /** 获取 warm start 迁移日志 */
  getWarmStartLog() { return this._warmStartLog.slice(); }

  // ==================== P2 共享设施 ====================

  /** P2-3 接口节奏真实化（全局共享） */
  getInterfaceController() {
    if (!this._interface) this._interface = new HumanInterfaceController();
    return this._interface;
  }

  /** P2-2 人类限时评估基准（全局共享，惰性加载基线） */
  getBaselineEvaluator() {
    if (!this._evaluator) {
      this._evaluator = new HumanBaselineEvaluator();
      this._evaluator.loadBaseline().catch(() => {});
    }
    return this._evaluator;
  }

  /** P2-1a 人类轨迹采集器（全局共享） */
  getTrajectoryRecorder() {
    if (!this._recorder) this._recorder = HumanTrajectoryRecorder.get();
    return this._recorder;
  }

  /**
   * P2-1b 挂接行为克隆先验到指定游戏智能体。
   * @param {string} gameKey
   * @param {Object} prior - BehaviorCloningPrior 实例
   * @param {number} [alpha] BC 权重
   */
  attachBCPrior(gameKey, prior, alpha = 0.1) {
    const agent = this._agents.get(gameKey);
    if (!agent) return null;
    return agent.attachBCPrior(prior, alpha);
  }

  /**
   * P2-1b 从已采集的人类轨迹构建行为先验并挂接（异步、可重复调用，
   * 数据量不足时静默跳过）。
   * @param {string} gameKey
   * @param {Object} game 游戏实例（提供 getActionSpec）
   * @param {number} [alpha] BC 权重
   * @param {number} [minSamples] 启用所需的最小有效样本数
   * @returns {Promise<BehaviorCloningPrior|null>}
   */
  async enableBehaviorCloning(gameKey, game, alpha = 0.1, minSamples = 10) {
    const agent = this._agents.get(gameKey);
    if (!agent || !game || typeof game.getActionSpec !== 'function') return null;
    const recorder = this.getTrajectoryRecorder();
    const trajs = await recorder.getTrajectories(gameKey);
    if (!trajs.length) return null;
    const prior = new BehaviorCloningPrior({
      stateSize: agent.stateSize,
      nActions: agent.nActions,
    });
    let added = 0;
    for (const t of trajs) added += prior.addTrajectory(t, game.getActionSpec());
    if (added < minSamples) return null;
    agent.attachBCPrior(prior, alpha);
    console.log(`[RLAgentManager] ${gameKey}: 行为克隆先验已启用 (${added} 样本)`);
    return prior;
  }

  /** 所有智能体统计快照 {gameKey: {stats, eval, bcKL}} */
  getAllStats() {
    const out = {};
    for (const [k, a] of this._agents) {
      const entry: Record<string, any> = { stats: a.getStats() };
      // P2-1b 行为分布距离（验收指标）
      if (a.bcPrior && a.bcPrior.pairCount) {
        entry.bcKL = a.measureBCKL();
        entry.bcSamples = a.bcPrior.pairCount;
      }
      // P2-2 人类限时评估报告
      if (this._evaluator) entry.eval = this._evaluator.report(k);
      // P3-1 世界模型指标（验收：loss 可测且下降、想象样本注入量）
      const wm = this._worldModels.get(k);
      if (wm) {
        entry.wm = { loss: wm.getLoss(), updates: wm.stats.updates, pairs: wm.pairCount, rollouts: wm.stats.rollouts };
      }
      if (a.worldModel) {
        entry.wmImagined = a.wmImagined;
        entry.wmLoss = a.wmLoss;
      }
      // P3-3 RLHF-lite 指标（验收：好/坏分离度、shaping 累计）
      const rlhf = this._rlhfs.get(k);
      if (rlhf) {
        entry.rlhf = { pairs: rlhf.stats.pairs, sep: rlhf.stats.sep, trainLoss: rlhf.stats.trainLoss };
      }
      if (a.rlhf) entry.rlhfShaping = a.rlhfShaping;
      out[k] = entry;
    }
    return out;
  }

  // ==================== P3 共享设施 ====================

  /**
   * P3-1 世界模型（全局共享，惰性创建）。
   * @param {string} gameKey
   * @returns {WorldModel|null}
   */
  getWorldModel(gameKey) {
    const agent = this._agents.get(gameKey);
    if (!agent) return null;
    if (!this._worldModels.has(gameKey)) {
      const wm = new WorldModel({ stateSize: agent.stateSize, nActions: agent.nActions });
      this._worldModels.set(gameKey, wm);
    }
    return this._worldModels.get(gameKey);
  }

  /**
   * P3-1 启用世界模型增强训练（想象回放）。
   * @param {string} gameKey
   * @param {number} [alpha=0.5] 想象样本权重
   * @param {number} [imagineSteps=8]
   * @returns {WorldModel|null}
   */
  enableWorldModel(gameKey, alpha = 0.5, imagineSteps = 8) {
    const agent = this._agents.get(gameKey);
    const wm = this.getWorldModel(gameKey);
    if (!agent || !wm) return null;
    agent.attachWorldModel(wm, alpha, imagineSteps);
    console.log(`[RLAgentManager] ${gameKey}: 世界模型增强已启用 (alpha=${alpha}, steps=${imagineSteps})`);
    return wm;
  }

  /**
   * P3-3 RLHF-lite 对比反馈校准器（全局共享，惰性创建）。
   * @param {string} gameKey
   * @returns {RLHFLite|null}
   */
  getRLHF(gameKey) {
    const agent = this._agents.get(gameKey);
    if (!agent) return null;
    if (!this._rlhfs.has(gameKey)) {
      const r = new RLHFLite({ stateSize: agent.stateSize, nActions: agent.nActions });
      this._rlhfs.set(gameKey, r);
    }
    return this._rlhfs.get(gameKey);
  }

  /**
   * P3-3 启用 RLHF-lite 奖励校准（store() 时叠加 shaping）。
   * @param {string} gameKey
   * @param {number} [alpha=0.3]
   * @returns {RLHFLite|null}
   */
  enableRLHF(gameKey, alpha = 0.3) {
    const agent = this._agents.get(gameKey);
    const r = this.getRLHF(gameKey);
    if (!agent || !r) return null;
    agent.attachRLHF(r, alpha);
    console.log(`[RLAgentManager] ${gameKey}: RLHF-lite 奖励校准已启用 (alpha=${alpha})`);
    return r;
  }

  /** P3-2 零样本评测器（全局共享） */
  getZeroShotEvaluator() {
    if (!this._zeroShotEval) {
      const mod = import('./zero-shot-eval.ts');
      // 惰性动态加载，避免启动时引入评测依赖
      this._zeroShotEval = { pending: mod, instance: null };
    }
    if (this._zeroShotEval.instance) return this._zeroShotEval.instance;
    if (this._zeroShotEval.pending) {
      this._zeroShotEval.pending.then(m => {
        this._zeroShotEval.instance = new m.ZeroShotEvaluator();
        this._zeroShotEval.pending = null;
      }).catch(() => {});
      return this._zeroShotEval.instance;
    }
    return null;
  }

  /**
   * P3-2 零样本可玩性评测：加载已训练权重，在未见关卡上测门禁。
   * @param {string} gameKey
   * @param {Object} [opts] {episodes, seedCount, baseSeed, humanBaseline, passRatio}
   * @returns {Promise<{pass, summary}|null>}
   */
  async runZeroShotEval(gameKey, opts: { episodes?: number; seedCount?: number; baseSeed?: number; humanBaseline?: { successRate: number; avgTimeSec?: number }; passRatio?: number; evaluator?: any } = {}) {
    const agent = this._agents.get(gameKey);
    if (!agent) return null;
    const mod = await import('./zero-shot-eval.ts');
    const evaluator = opts.evaluator || new mod.ZeroShotEvaluator(opts);
    const weights = agent.onlineNet.toJSON();
    const cfg = {
      stateSize: agent.stateSize,
      nActions: agent.nActions,
      hiddenLayers: agent.hiddenLayers,
    };
    return evaluator.evaluate(weights, cfg);
  }

  /** 全部智能体立即持久化 */
  async flushAll() {
    const jobs = [];
    for (const a of this._agents.values()) jobs.push(a.flush());
    await Promise.all(jobs);
  }

  // ===== 类型声明（仅类型注解，无运行时副作用） =====
  declare _agents: Map<string, UnifiedRLAgent>;
  declare _warmStartLog: Array<{ from: string; to: string; ts: number }>;
  declare _interface: HumanInterfaceController | null;
  declare _evaluator: HumanBaselineEvaluator | null;
  declare _recorder: HumanTrajectoryRecorder | null;
  declare _worldModels: Map<string, WorldModel>;
  declare _rlhfs: Map<string, RLHFLite>;
  declare _zeroShotEval: any;
}

export default RLAgentManager;
