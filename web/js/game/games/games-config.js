/* ============================================================
 * 游戏注册表 —— 前端 RL 适配的单一事实源（对应方案报告 P0-2）
 *
 * 职责：
 * - 承载每个游戏的 RL 元信息（模式映射 / 存储键 / 观察维度 / 动作数 / 超参）
 * - 驱动 GameModeManager 的注册与 init-game-mode 的游戏入口
 * - 与后端 rl_coordinator.py 的 GAME_KEY_TO_MODE 保持一一对应
 *
 * 新增一款游戏时：在 GAME_CONFIGS 追加一项配置，并实现 BaseGame 的 RL 契约。
 * ============================================================ */

export const GAME_CONFIGS = [
  {
    key: 'treasure_hunt',
    displayName: '迷宫寻宝',
    description: '在神秘迷宫中探索，触碰星光与线索需答对谜题才能收集，集齐所有星光与线索才能打开宝藏！',
    // 统一模式映射（与后端 UnifiedMode 对齐：approach_game=1, date_game=2, creative_play=6）
    mode: 'approach_game',
    rl: {
      enabled: true,             // P1-4 已接入统一 RL 契约（13 维观察 / 4 动作）
      storageKey: 'treasure_unified_v1',
      stateSize: 13,
      nActions: 4,
      hyperparams: {
        hiddenLayers: [64, 64], lr: 0.001, gamma: 0.95, nStep: 3,
        useNoisy: true, noisyStd: 0.1, usePER: true, useDistributional: false,
        autoTune: true, usePBT: false, replayCapacity: 5000, batchSize: 32, useSymlog: true,
      },
    },
  },
  {
    key: 'sandbox',
    displayName: '沙盒世界',
    description: '无限随机生成的方块大世界，流浪探索、采集资源，和AI伙伴一起冒险。',
    mode: 'creative_play',
    rl: {
      enabled: true,             // P1-4 已接入统一 RL 契约（11 维观察 / 8 动作）
      storageKey: 'sandbox_unified_v1',
      stateSize: 11,
      nActions: 8,
      hyperparams: {
        hiddenLayers: [64, 64], lr: 0.001, gamma: 0.95, nStep: 3,
        useNoisy: true, noisyStd: 0.1, usePER: true, useDistributional: false,
        autoTune: true, usePBT: false, replayCapacity: 5000, batchSize: 32, useSymlog: true,
      },
    },
  },
  {
    key: 'moba_5v5',
    displayName: '王者峡谷 5v5',
    description: '5v5 推塔对战：三路兵线、野区、防御塔，复刻王者荣耀核心玩法。AI 英雄具备完整对线/gank/团战智能。',
    mode: 'date_game',
    rl: {
      enabled: true,
      storageKey: 'moba_unified_v1',
      stateSize: 12,
      nActions: 9,
      // 统一超参（与 moba-5v5.js 原 _rlEnsureAgent 保持一致）
      // useSymlog: 奖励 symlog 变换（DreamerV3 域无关技巧，P1-2）
      hyperparams: {
        hiddenLayers: [64, 48], lr: 0.001, gamma: 0.93, nStep: 2,
        useNoisy: true, noisyStd: 0.12, usePER: true, useDistributional: false,
        autoTune: true, usePBT: false, replayCapacity: 10000, batchSize: 32, useSymlog: true,
      },
    },
  },
  {
    key: 'mario',
    displayName: '马里奥无限跑酷',
    description: '随机地形无限跑酷：DQN 必须学会通用跳跃策略才能存活。玩家放手时 AI 自主闯关。',
    mode: 'approach_game',
    rl: {
      enabled: true,
      storageKey: 'mario_unified_v1',
      stateSize: 112,
      nActions: 8,
      // 统一超参（与 mario-game.js 原 _rlEnsureAgent 保持一致）
      // useSymlog: 奖励 symlog 变换（DreamerV3 域无关技巧，P1-2）
      hyperparams: {
        hiddenLayers: [128, 128], lr: 0.0003, gamma: 0.99, nStep: 3,
        useNoisy: true, noisyStd: 0.08, usePER: true, useDistributional: false,
        autoTune: true, usePBT: false, replayCapacity: 20000, useSymlog: true,
      },
    },
  },
  {
    key: 'xiangqi',
    displayName: '象棋对战',
    description: '玩家附身AI角色随机立于红黑一方，自由在棋盘附近走动观战。AI角色获取棋盘快照并实时分析局势，提供战术建议。RL机器人(黑方)通过DQN不断学习进化。',
    mode: 'approach_game',
    rl: {
      enabled: true,
      storageKey: 'xiangqi_unified_v1',
      stateSize: 105,
      nActions: 48,
      hyperparams: {
        hiddenLayers: [128, 128], lr: 0.0003, gamma: 0.99, nStep: 3,
        useNoisy: true, noisyStd: 0.1, usePER: true, useDistributional: false,
        autoTune: true, usePBT: false, replayCapacity: 20000, batchSize: 64, useSymlog: true,
      },
    },
  },
  {
    key: 'cyber_corp',
    displayName: '赛博公司',
    description: '就任 CEO 创建独立世界：无实体第一人称游走，大厅角色与 AI 对话完全隔离；每个员工持独立 RL 策略自主工作并互相协作，靠近员工显示其独立对话记录，对员工说「开会」「汇报」可召集全员会议。',
    mode: 'approach_game',
    rl: {
      enabled: false,
      storageKey: 'cyber_corp_unified_v1',
      stateSize: 0,
      nActions: 0,
      hyperparams: {},
      // 说明：赛博公司使用游戏内置的 SwarmRLAgent（每员工独立 Q 表 + 公司任务看板），
      // 不接入大厅统一 RL 管线，故 enabled=false；RL 元信息仅供大厅展示识别。
    },
  },
];

/** 按 key 取游戏配置 */
export function getGameConfig(key) {
  return GAME_CONFIGS.find(g => g.key === key) || null;
}

/** 获取支持 RL 的游戏配置列表 */
export function getRLEnabledGames() {
  return GAME_CONFIGS.filter(g => g.rl && g.rl.enabled);
}

export default GAME_CONFIGS;
