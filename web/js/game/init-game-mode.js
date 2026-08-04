/* ============================================================
 * 游戏模式初始化模块
 *
 * - 注册所有可用游戏
 * - 创建GameModeManager实例
 * - 绑定UI按钮
 * - 导出供其他模块使用
 * ============================================================ */

import { GameModeManager } from './game-mode-manager.js';
import { TreasureHuntGame } from './games/treasure-hunt.js';
import { SandboxGame } from './games/sandbox-game.js';
import { MobaGame } from './games/moba-5v5.js';
import { MarioGame } from './games/mario-game.js';
import { XiangqiGame } from './games/xiangqi-game.js';
// ?v= 版本参数：避免浏览器缓存旧版赛博公司蜂群引擎（改动后 bump 版本号即可强制刷新）
import { CyberCorpGame } from './games/cyber-corp.js?v=20260804e';
import { GAME_CONFIGS } from './games/games-config.js';

export default (function init(App) {
  const { THREE } = App;

  /* ---------- 注册游戏（由 games-config 注册表驱动，P0-2） ---------- */
  // 工厂映射：新增游戏只需在 games-config.js 追加配置并在下方追加工厂
  const GAME_FACTORIES = {
    treasure_hunt: (app) => new TreasureHuntGame(app),
    sandbox: (app) => new SandboxGame(app),
    moba_5v5: (app) => new MobaGame(app),
    mario: (app) => new MarioGame(app),
    xiangqi: (app) => new XiangqiGame(app),
    cyber_corp: (app) => new CyberCorpGame(app),
  };

  // 由注册表驱动注册：注册表是游戏身份与 RL 元信息的单一事实源
  for (const cfg of GAME_CONFIGS) {
    const factory = GAME_FACTORIES[cfg.key];
    if (!factory) {
      console.warn(`[GameMode] 注册表包含未实现的游戏: ${cfg.key}`);
      continue;
    }
    GameModeManager.registerGame(cfg.key, factory);
  }

  /* ---------- 创建管理器 ---------- */
  App.gameModeManager = new GameModeManager(App);
  window._gameManager = App.gameModeManager;

  // 启用非游戏模式的环境快照（供 AI 自主感知使用）
  setTimeout(() => {
    if (App.gameModeManager && App.gameModeManager.stateObserver) {
      App.gameModeManager.stateObserver.enableLobbySnapshots();
      console.log('[AI自主] 大厅模式环境快照已启用');
    }
  }, 5000); // 等 WebSocket 连接建立后再启动

  // 在每帧更新中调用大厅快照
  const origRenderLoop = App.renderLoop;
  if (origRenderLoop) {
    App._origRenderLoop = origRenderLoop;
  }
  // 注意：大厅快照更新在 game-mode-manager 的 stateObserver.updateLobby() 中
  // 需要在主渲染循环中调用。我们通过 App 暴露一个调用点。
  App._onFrameLobbyUpdate = function() {
    if (App.gameModeManager && !App.gameModeManager.active && App.gameModeManager.stateObserver) {
      App.gameModeManager.stateObserver.updateLobby();
    }
  };

  /* ---------- 添加游戏按钮 ---------- */
  const toolsBar = document.querySelector('.stage-tools');
  if (!toolsBar) {
    console.warn('[GameMode] 未找到工具栏元素');
    return;
  }

  const gameBtn = document.createElement('button');
  gameBtn.id = 'game-btn';
  gameBtn.className = 'stage-tool-btn';
  gameBtn.title = '游戏模式';
  gameBtn.innerHTML = `
    <svg viewBox="0 0 24 24" width="20" height="20">
      <path fill="currentColor" d="M21 6H3c-1.1 0-2 .9-2 2v8c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-10 7H8v3H6v-3H3v-2h3V8h2v3h3v2zm4.5 2c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm4-3c-.83 0-1.5-.67-1.5-1.5S18.67 9 19.5 9s1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/>
    </svg>
  `;
  gameBtn.addEventListener('click', () => {
    App.gameModeManager.enterGameMode();
  });

  // 插入到低功耗按钮之前
  const lowPowerBtn = document.getElementById('low-power-btn');
  if (lowPowerBtn) {
    toolsBar.insertBefore(gameBtn, lowPowerBtn);
  } else {
    toolsBar.appendChild(gameBtn);
  }

  console.log('[GameMode] 游戏模式已初始化，已注册游戏:', GameModeManager.getAvailableGames().map(g => g.name));
});
