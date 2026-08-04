export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  WebSocket
   * ============================================================ */
  /* ============================================================
   *  场景状态持久化（相机/模型位置/缩放）
   * ============================================================ */
  App.SCENE_KEY = 'dabai.sceneState';
  App.saveSceneState = function saveSceneState() {
    const state = {
      camZoom: App.camZoom,
      camOffsetX: App.camOffsetX,
      camOffsetY: App.camOffsetY,
      camOffsetZ: App.camOffsetZ,
      xrMode: App.xrMode,
      moveMode: App.moveMode,
      backgroundAutoRotate: App.backgroundAutoRotate
    };
    // 保存角色位置/缩放
    const avatarTarget = App.modelGroup;
    if (avatarTarget) {
      state.charPos = {
        x: avatarTarget.position.x,
        z: avatarTarget.position.z
      };
      state.charScale = avatarTarget.scale.x;
    }
    // 保存背景位置/缩放
    if (App.backgroundGroup) {
      state.bgPos = {
        x: App.backgroundGroup.position.x,
        z: App.backgroundGroup.position.z
      };
      state.bgScale = App.backgroundGroup.scale.x;
    }
    try {
      localStorage.setItem(App.SCENE_KEY, JSON.stringify(state));
    } catch (e) {}
  };
  App._saveTimer = null;
  App.debouncedSaveScene = function debouncedSaveScene() {
    clearTimeout(App._saveTimer);
    App._saveTimer = setTimeout(App.saveSceneState, 500);
  };
  App.restoreSceneState = function restoreSceneState() {
    try {
      const s = JSON.parse(localStorage.getItem(App.SCENE_KEY));
      if (!s) return;
      if (typeof s.camZoom === 'number') App.camZoom = THREE.MathUtils.clamp(s.camZoom, App.MIN_ZOOM, App.MAX_ZOOM);
      if (typeof s.camOffsetX === 'number') {
        App.camOffsetX = s.camOffsetX;
        App.camOffsetY = s.camOffsetY || 0;
        App.camOffsetZ = s.camOffsetZ || 0;
      }
      // 背景自转
      if (typeof s.backgroundAutoRotate === 'boolean') App.backgroundAutoRotate = s.backgroundAutoRotate;
      // 角色/背景位置在模型加载后恢复
      window._restoredScene = s;
    } catch (e) {}
  };
  App.loadCameraSettings = function loadCameraSettings() {
    try {
      const s = JSON.parse(localStorage.getItem(App.CAM_SETTINGS_KEY));
      if (!s) return;
      if (typeof s.cameraHeight === 'number') {
        App.cameraHeight = THREE.MathUtils.clamp(s.cameraHeight, 0.1, 10.0);
      }
      if (typeof s.cameraTiltDeg === 'number') {
        App.cameraTiltDeg = THREE.MathUtils.clamp(s.cameraTiltDeg, -80, 80);
      }
      if (typeof s.cameraDistance === 'number') {
        App.cameraDistance = THREE.MathUtils.clamp(s.cameraDistance, 0.1, 10.0);
        App.DEFAULT_CAM_POS.z = App.cameraDistance;
        App.targetCamPos.z = App.cameraDistance;
      }
      // VR 注视归位开关
      if (typeof s.gazeAssist === 'boolean') App.gazeAssistEnabled = s.gazeAssist;
    } catch (e) {}
  };
  App._camSettingsSaveTimer = null;
  App.saveCameraSettings = function saveCameraSettings() {
    clearTimeout(App._camSettingsSaveTimer);
    App._camSettingsSaveTimer = setTimeout(() => {
      try {
        localStorage.setItem(App.CAM_SETTINGS_KEY, JSON.stringify({
          cameraHeight: App.cameraHeight,
          cameraTiltDeg: App.cameraTiltDeg,
          cameraDistance: App.cameraDistance,
          gazeAssist: App.gazeAssistEnabled
        }));
      } catch (e) {}
    }, 300);
  };
  App.resetAvatarToOrigin = function resetAvatarToOrigin() {
    const avatar = App.currentAvatar || App.modelGroup;
    if (avatar) {
      let targetX = 0, targetY = avatar.position.y, targetZ = 0;
      if (App.backgroundGroup) {
        // 从背景包围盒中心往下射线，取最低命中点 = 真正地板
        const cx = App._bgCenterX || 0;
        const cz = App._bgCenterZ || 0;
        const floorY = App._findFloorY(cx, cz);
        targetX = cx; targetY = floorY + 2.0; targetZ = cz;
      }
      // 平滑过渡回原点（替代硬瞬移），距离过近直接落位
      const dx = targetX - avatar.position.x;
      const dz = targetZ - avatar.position.z;
      const dist = Math.hypot(dx, dz);
      if (dist < 0.3) {
        avatar.position.set(targetX, targetY, targetZ);
      } else {
        App._smoothTeleport = {
          x0: avatar.position.x, y0: avatar.position.y, z0: avatar.position.z,
          x1: targetX, y1: targetY, z1: targetZ,
          t: 0, dur: Math.min(0.7, 0.3 + dist * 0.04),
        };
      }
    }
    // 清除可能恢复旧位置的缓存
    if (window._restoredScene) {
      delete window._restoredScene.charPos;
    }
    // 停止当前 idle 移动，避免下一帧把角色拉回旧路径
    App.idleWalkTarget = null;
    App.idleWalkProgress = 0;
    App.walkPath = [];
    App.walkSegmentIndex = 0;
    App.currentAction = null;
    App.nextActionTimer = 0;
    App.saveSceneState();
  };
  App.applySavedPositions = function applySavedPositions() {
    const s = window._restoredScene;
    if (!s) return;
    // 恢复角色位置（平滑过渡，替代硬瞬移）
    const avatar = App.modelGroup;
    if (s.charPos && avatar) {
      const dx = s.charPos.x - avatar.position.x;
      const dz = s.charPos.z - avatar.position.z;
      const dist = Math.hypot(dx, dz);
      if (dist < 0.3) {
        avatar.position.x = s.charPos.x;
        avatar.position.z = s.charPos.z;
      } else {
        App._smoothTeleport = {
          x0: avatar.position.x, y0: avatar.position.y, z0: avatar.position.z,
          x1: s.charPos.x, y1: avatar.position.y, z1: s.charPos.z,
          t: 0, dur: Math.min(0.7, 0.3 + dist * 0.04),
        };
      }
      if (s.charScale) avatar.scale.setScalar(s.charScale);
    }
    // 恢复背景位置
    if (s.bgPos && App.backgroundGroup) {
      App.backgroundGroup.position.x = s.bgPos.x;
      App.backgroundGroup.position.z = s.bgPos.z;
      if (s.bgScale) App.backgroundGroup.scale.setScalar(s.bgScale);
    }
    // 恢复移动模式
    if (s.moveMode) App.setMoveMode(true);
  };
  App.connectWS = function connectWS() {
    // 清理旧的定时器
    if (App.wsHeartbeat) {
      clearInterval(App.wsHeartbeat);
      App.wsHeartbeat = null;
    }
    if (App.rlHeartbeat) {
      clearTimeout(App.rlHeartbeat);
      App.rlHeartbeat = null;
    }
    if (App.wsReconnectTimer) {
      clearTimeout(App.wsReconnectTimer);
      App.wsReconnectTimer = null;
    }
    if (App.wsConnTimeout) {
      clearTimeout(App.wsConnTimeout);
      App.wsConnTimeout = null;
    }
    App.statusBadge.textContent = '连接中…';
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';

    // 连接超时：3秒还没连上就触发重连
    App.wsConnTimeout = setTimeout(() => {
      if (App.ws && App.ws.readyState === WebSocket.CONNECTING) {
        App.ws.close();
        App.ws = null;
      }
      App.statusBadge.textContent = '连接超时，重连中…';
      App.wsReconnectTimer = setTimeout(App.connectWS, 2000);
    }, 3000);
    try {
      App.ws = new WebSocket(`${proto}://${location.host}/ws`);
    } catch (e) {
      App.statusBadge.textContent = '连接失败，重连中…';
      App.wsReconnectTimer = setTimeout(App.connectWS, 2000);
      return;
    }
    App.ws.onopen = () => {
      clearTimeout(App.wsConnTimeout);
      App.wsConnTimeout = null;
      App.statusBadge.textContent = '已连接';
      App.setState(App.State.IDLE);
      App.addSystemMsg('已连接到 AI');
      // 发送用户标识以恢复历史
      const uid = localStorage.getItem('dabai.userId') || 'u_' + Date.now().toString(36);
      localStorage.setItem('dabai.userId', uid);
      App.ws.send(JSON.stringify({
        type: 'set_user',
        user_id: uid
      }));

      // 同步当前角色形象与背景场景给 AI
      const savedModel = JSON.parse(localStorage.getItem('dabai.currentModel') || 'null');
      if (savedModel && savedModel.name) {
        App.ws.send(JSON.stringify({
          type: 'set_avatar',
          name: savedModel.name
        }));
        App._sentAvatarName = savedModel.name;
      }
      const savedBg2 = JSON.parse(localStorage.getItem('dabai.currentBackground') || 'null');
      if (savedBg2 && savedBg2.name) {
        App.ws.send(JSON.stringify({
          type: 'set_background',
          name: savedBg2.name
        }));
        App._sentBgName = savedBg2.name;
      }

      // 启动 WebSocket 心跳（每25秒发一次 ping，防止 NAT/代理超时断连）
      App.wsHeartbeat = setInterval(() => {
        if (App.ws && App.ws.readyState === WebSocket.OPEN) {
          // 尝试发送心跳，如果失败则触发重连
          try {
            App.ws.send(JSON.stringify({
              type: 'ping'
            }));
          } catch (e) {
            if (App.ws) {
              App.ws.close();
              App.ws = null;
            }
            App.connectWS();
          }
        } else if (!App.ws || App.ws.readyState > WebSocket.OPEN) {
          // 连接已不在 OPEN 状态，发起重连
          if (App.wsHeartbeat) {
            clearInterval(App.wsHeartbeat);
            App.wsHeartbeat = null;
          }
          App.connectWS();
        }
      }, 25000);

      // 如果之前在自动对话模式，重连后恢复 VAD
      if (App.voiceMode === 'auto' && !App.vadStream) {
        console.log('[WS] 重连后恢复自动对话模式');
        App.startVADMode().then(ok => {
          if (ok) {
            App.vadState = 'idle';
            App.vadLoop();
          }
        });
      }

      // 启动 RL 统一调度心跳（完全统摄：后端 proactive_tick 自动驱动）
      App.startRLHeartbeat();
    };
    App.ws.onmessage = e => {
      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch {
        return;
      }
      // 服务端 pong 响应，忽略
      if (msg.type === 'pong') return;
      App.handleWSMessage(msg);
    };
    App.ws.onerror = () => {
      clearTimeout(App.wsConnTimeout);
      App.wsConnTimeout = null;
      App.statusBadge.textContent = '连接出错，重连中…';
    };
    App.ws.onclose = () => {
      clearTimeout(App.wsConnTimeout);
      App.wsConnTimeout = null;
      if (App.wsHeartbeat) {
        clearInterval(App.wsHeartbeat);
        App.wsHeartbeat = null;
      }
      if (App.rlHeartbeat) {
        clearTimeout(App.rlHeartbeat);
        App.rlHeartbeat = null;
      }
      App.statusBadge.textContent = '连接断开，重连中…';
      App.wsReconnectTimer = setTimeout(App.connectWS, 2000);
    };
  };
  App.handleWSMessage = function handleWSMessage(msg) {
    switch (msg.type) {
      case 'ready':
        App.setState(App.State.IDLE);
        break;
      case 'user_set':
        console.log('[WS] 用户已设置:', msg.user_id);
        // 恢复聊天历史
        if (msg.history && msg.history.length > 0) {
          App.messagesEl.innerHTML = '';
          App.addSystemMsg('已连接');
          for (const h of msg.history) {
            if (h.user) App.addUserMsg(h.user);
            if (h.ai) App.addAIMsg(h.ai);
          }
        }
        break;
      case 'thinking':
        // 独立系统游戏（赛博公司）：吞掉后端 AI 回复启动信号。
        // currentReplySession 一旦设置，audio_chunk/audio_end 就会播放大厅 TTS
        // 并显示大厅 AI 回复气泡（乱入）。游戏内回复由蜂群 addAIMsg 独立渲染，
        // 不经过后端回复链路。
        if (App.gameModeManager && App.gameModeManager.currentGame && App.gameModeManager.currentGame.isIsolated) break;
        // 新一轮回复开始：清空旧队列与文本
        App.currentReplySession = msg.session_id || null;
        App.currentReplyText = '';
        App.audioQueue = [];
        App.removeTyping();
        App.setState(App.State.THINKING);
        App.showTyping();
        break;
      case 'listening':
        App.setState(App.State.LISTENING);
        break;
      case 'transcript':
        App.removeTyping();
        App.addUserMsg(msg.text, true);
        // 独立系统游戏（赛博公司）：玩家消息由游戏内蜂群系统接管（addUserMsg hook），
        // 不触发大厅 RL 系统（约会/表情/参与），否则大厅角色会"听到玩家说话"而乱入
        if (App.gameModeManager && App.gameModeManager.currentGame && App.gameModeManager.currentGame.isIsolated) break;
        // 记录用户活跃时间（语音消息也算，供 RL 统一状态取数）
        App._lastUserMessageTime = Date.now();
        App._lastUserInteractTime = Date.now();
        if (App._engagementRL) App._engagementRL.notifyUserMessage();
        if (App._datingSystem) App._datingSystem.notifyUserMessage(msg.text);
        if (App._expressionRL) App._expressionRL.notifyUserMessage();
        break;
      case 'audio_chunk':
        // 独立系统游戏（赛博公司）：吞掉后端 AI 回复音频流并清空播放队列。
        // handleAudioChunk 只在 currentReplySession 有值且不匹配时过滤，
        // 而游戏模式下 thinking 已被隔离（session 为 null），音频会照常播放 ——
        // 玩家未靠近任何角色也会听到大厅 AI 的"幽灵回复"声音。
        // 蜂群角色语音走游戏内独立 TTS（/api/game/speak），不受影响。
        if (App.gameModeManager && App.gameModeManager.currentGame && App.gameModeManager.currentGame.isIsolated) {
          if (App.clearAudioQueue) App.clearAudioQueue();
          break;
        }
        App.handleAudioChunk(msg);
        break;
      case 'audio_end':
        // 独立系统游戏（赛博公司）：同样吞掉结束信号，阻止队列继续播放
        if (App.gameModeManager && App.gameModeManager.currentGame && App.gameModeManager.currentGame.isIsolated) break;
        App.handleAudioEnd(msg);
        break;
      case 'interrupted':
        App.handleInterrupted(msg);
        break;
      case 'tool_call_start':
        // 工具调用开始
        App.removeTyping();
        App.addToolCallMsg(msg.tool_name, msg.arguments, 'start');
        break;
      case 'tool_call_result':
        // 工具调用结果
        App.addToolCallResult(msg.tool_name, msg.result, msg.success);
        break;
      case 'session_list':
        // 会话列表
        App.renderSessionList(msg.sessions);
        break;
      case 'session_switched':
        // 会话切换成功：清空面板并渲染历史
        App.messagesEl.innerHTML = '';
        if (msg.history && msg.history.length > 0) {
          App.addSystemMsg('已切换会话');
          for (const h of msg.history) {
            if (h.user) App.addUserMsg(h.user);
            if (h.ai) App.addAIMsg(h.ai);
          }
        }
        console.log('[WS] 切换到会话:', msg.session_id);
        break;
      case 'session_created':
        console.log('[WS] 新会话:', msg.session_id);
        break;
      case 'session_deleted':
        console.log('[WS] 会话已删除:', msg.session_id);
        break;
      case 'error':
        App.removeTyping();
        App.showToast(msg.message || '出错了');
        App.setState(App.State.IDLE);
        App.showSubtitle('');
        break;
      case 'restart_vad':
        // ffmpeg 转码失败 → 重建 VAD 自动对话模式（比刷新页面更快更轻量）
        console.warn('[WS] 重建 VAD:', msg.reason);
        App.showToast(msg.reason || '语音模式异常，正在重建…');
        App.stopVADMode();
        App.startVADMode().then(ok => {
          if (ok) {
            App.vadState = 'idle';
            App.vadLoop();
            App.showToast('语音模式已恢复');
          } else {
            // 麦克风也拿不到就没办法了
            App.showToast('无法恢复语音模式，请手动刷新页面');
          }
        });
        break;
      case 'screen_command':
        App.handleScreenCommand(msg);
        break;
      // === AI 自主行为命令 ===
      case 'ai_behavior_command':
        // 后端行为决策引擎发来的命令
        // 独立系统游戏（赛博公司）：拒绝大厅 AI 行为命令（大厅角色不感知游戏环境）
        if (App.gameModeManager && App.gameModeManager.currentGame && App.gameModeManager.currentGame.isIsolated) break;
        if (App.aiAutonomyController) {
          App.aiAutonomyController.receiveCommand(msg);
        } else {
          console.log('[AI自主] 收到命令但控制器未初始化:', msg.behavior);
        }
        break;
      // === LLM-as-policy 宏观策略响应 ===
      case 'game_action_response':
        if (App.gameModeManager && App.gameModeManager.currentGame &&
            typeof App.gameModeManager.currentGame.onLLMActionResponse === 'function') {
          App.gameModeManager.currentGame.onLLMActionResponse(msg.data || {});
        }
        break;
      // === RL 统一调度计划（完全统摄：游戏 + 非游戏 Agent 统一派发） ===
      case 'rl_dispatch':
        {
          const plan = msg.data || {};
          if (!plan.agent_choice) break;
          App._rlLastDispatchTime = Date.now();
          console.log('[RL调度]', plan.agent_choice, '|', plan.reason || '', '| 模式:', plan.mode_name || '');
          // engagement：执行行为指令（自主行为）
          // 独立系统游戏（赛博公司）：不执行大厅自主行为（游戏内由蜂群 + RL 独立驱动）
          if (plan.agent_choice === 'engagement' && plan.behavior_cmd && App.aiAutonomyController) {
            const gIsolated = App.gameModeManager && App.gameModeManager.currentGame && App.gameModeManager.currentGame.isIsolated;
            if (!gIsolated) App.aiAutonomyController.receiveCommand(plan.behavior_cmd);
          }
          // ai_agent：服务端已启动主动回复（rl_dispatch 仅作状态回显，避免重复触发）
          // game_agent：游戏内部已有独立 game_action_request 循环，此处同步策略即可
          if (plan.agent_choice === 'game_agent' && plan.strategy && App.gameModeManager) {
            App.gameModeManager.lastRlStrategy = plan.strategy;
          }
          // RL 决策的快照间隔（秒）→ 按模式应用到观察者（游戏/大厅）
          if (plan.snapshot_interval && App.gameModeManager && App.gameModeManager.stateObserver) {
            const obs = App.gameModeManager.stateObserver;
            if (plan.interval_mode === 'game') obs.applyRlGameInterval(plan.snapshot_interval);
            else obs.applyRlLobbyInterval(plan.snapshot_interval);
          }
        }
        break;
      // === RL 状态回显（rl_sync 的响应） ===
      case 'rl_status':
        {
          const st = msg.data || {};
          // RL 决策的快照间隔（秒）→ 按模式应用到观察者（游戏/大厅）
          if (st.snapshot_interval && App.gameModeManager && App.gameModeManager.stateObserver) {
            const obs = App.gameModeManager.stateObserver;
            if (st.interval_mode === 'game') obs.applyRlGameInterval(st.snapshot_interval);
            else obs.applyRlLobbyInterval(st.snapshot_interval);
          }
          if (App._rlStatusEl) App._rlStatusEl.textContent = JSON.stringify(st);
        }
        break;
    }
  }; // ========== 文件名字段模糊匹配（AI 给的名字可能与实际文件名略有差异） ==========
  /**
   * 模糊匹配文件名，返回服务器上的精确文件名，找不到返回 null。
   * @param {string} name - AI 传入的文件名
   * @param {string} endpoint - '/api/models' 或 '/api/backgrounds'
   */
  App.fuzzyMatchFile = async function fuzzyMatchFile(name, endpoint) {
    // 标准化比较：去扩展名、全小写
    const normalize = s => s.replace(/\.[^.]+$/, '').toLowerCase().trim();
    const target = normalize(name);
    if (!target) return null;
    try {
      const resp = await fetch(endpoint);
      if (!resp.ok) return null;
      const data = await resp.json();
      // API 返回格式: {models: [{name:..., ...}]} 或 {backgrounds: [{name:..., ...}]}
      const list = data.models || data.backgrounds || [];
      const filenames = list.map(f => f.name);

      // 1. 精确匹配
      if (filenames.includes(name)) return name;

      // 2. 忽略大小写精确匹配
      const lowerFiles = filenames.map(f => f.toLowerCase());
      const exactIdx = lowerFiles.indexOf(name.toLowerCase());
      if (exactIdx >= 0) return filenames[exactIdx];

      // 3. 去扩展名后匹配
      const noExtNames = filenames.map(f => normalize(f));
      const noExtIdx = noExtNames.indexOf(target);
      if (noExtIdx >= 0) return filenames[noExtIdx];

      // 4. 子串匹配（至少包含 3 个字符才匹配，避免过短关键词命中）
      if (target.length >= 3) {
        for (let i = 0; i < noExtNames.length; i++) {
          if (noExtNames[i].includes(target) || target.includes(noExtNames[i])) {
            console.log(`[FuzzyMatch] '${name}' → '${filenames[i]}'`);
            return filenames[i];
          }
        }
      }

      // 5. 中文关键词匹配：拆出中文部分，看是否被文件名包含
      const zhChars = target.replace(/[^\u4e00-\u9fff]/g, '');
      if (zhChars.length >= 2) {
        for (let i = 0; i < noExtNames.length; i++) {
          if (noExtNames[i].includes(zhChars)) {
            console.log(`[FuzzyMatch|ZH] '${name}' → '${filenames[i]}'`);
            return filenames[i];
          }
        }
      }
    } catch (e) {
      console.warn('[FuzzyMatch] 查询文件列表失败:', e);
    }
    return null;
  }; // ========== 屏幕控制命令处理（AI 通过 MCP 工具调用控制前端） ==========
  App.handleScreenCommand = async function handleScreenCommand(msg) {
    const {
      tool,
      args
    } = msg;
    console.log('[ScreenCmd] 收到 AI 屏幕指令:', tool, args);
    try {
      switch (tool) {
        case 'switch_character_model':
          {
            let name = args.model_name;
            // 模糊匹配：AI 给的文件名可能不精确，尝试找到最接近的实际文件
            const match = await App.fuzzyMatchFile(name, '/api/models');
            if (match) {
              name = match;
            } else {
              console.warn('[ScreenCmd] 未找到模型:', name, '，尝试直接加载');
            }
            const url = '/models/' + encodeURIComponent(name);
            await App.loadModelFromUrl(url, name);
            break;
          }
        case 'switch_background_scene':
          {
            let name = args.bg_name;
            // 模糊匹配：AI 给的文件名可能不精确
            const match = await App.fuzzyMatchFile(name, '/api/backgrounds');
            if (match) {
              name = match;
            } else {
              console.warn('[ScreenCmd] 未找到背景:', name, '，尝试直接加载');
            }
            const url = '/backgrounds/' + encodeURIComponent(name);
            await App.loadBackgroundFromUrl(url, name);
            break;
          }
        case 'switch_tts_settings':
          {
            if (args.engine) {
              App.switchTTSEngine(args.engine);
            }
            const cfg = {};
            if (args.voice) cfg.edge_voice = args.voice;
            if (args.rate) cfg.edge_rate = args.rate;
            if (args.engine) cfg.engine = args.engine;
            if (Object.keys(cfg).length > 0) {
              try {
                const res = await fetch('/api/tts/config', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json'
                  },
                  body: JSON.stringify(cfg)
                });
                if (res.ok) {
                  const voiceDesc = args.voice || '';
                  const rateDesc = args.rate ? ' 语速' + args.rate : '';
                  App.showToast('AI 切换了语音设置' + (voiceDesc ? ' · ' + voiceDesc : '') + rateDesc);
                }
              } catch (e) {
                console.warn('[ScreenCmd] TTS 配置保存失败:', e);
              }
            }
            break;
          }
        case 'switch_app_mode':
          {
            const mode = args.mode;
            switch (mode) {
              case 'auto_voice':
                App.setVoiceMode('auto');
                App.showToast('AI 切换到了自动对话模式 🎙️');
                break;
              case 'press_voice':
                App.setVoiceMode('press');
                App.showToast('AI 切换到了按住说话模式');
                break;
              case 'low_power':
                if (!App.lowPowerMode) App.toggleLowPowerMode();
                break;
              case 'normal':
                if (App.lowPowerMode) App.toggleLowPowerMode();
                break;
            }
            break;
          }
        case 'show_screen_toast':
          {
            App.showToast(args.message || '✨');
            break;
          }
        case 'play_bgm':
          {
            const bgmName = args.bgm_name;
            const vol = args.volume !== undefined ? args.volume : 0.3;
            App.setBGMVolume(vol);
            const url = '/bgm/' + encodeURIComponent(bgmName);
            App.playBGM(url, bgmName);
            App.showToast('AI 播放了背景音乐');
            break;
          }
        case 'stop_bgm':
          {
            App.stopBGM();
            App.showToast('AI 停止了背景音乐');
            break;
          }
        case 'launch_game':
          {
            const gameKey = args.game_key;
            if (!gameKey) {
              console.warn('[ScreenCmd] launch_game 缺少 game_key 参数');
              break;
            }
            // 使用游戏模式管理器进入游戏
            if (App.gameModeManager) {
              App.gameModeManager.enterGameMode(gameKey);
            } else {
              console.warn('[ScreenCmd] gameModeManager 未初始化');
            }
            break;
          }
      }
    } catch (e) {
      console.error('[ScreenCmd] 执行失败:', tool, e);
    }
  };
  App.addToolCallMsg = function addToolCallMsg(toolName, args, status) {
    const el = document.createElement('div');
    el.className = 'msg tool-call';
    el.id = 'tool-' + toolName;
    el.innerHTML = `
        <svg class="tool-icon" viewBox="0 0 24 24" width="16" height="16">
            <path fill="currentColor" d="M22.7 19l-9.1-9.1c.9-2.3.4-5-1.5-6.9-2-2-5-2.4-7.4-1.3L9 6 6 9 1.6 4.7C.4 7.1.9 10.1 2.9 12.1c1.9 1.9 4.6 2.4 6.9 1.5l9.1 9.1c.4.4 1 .4 1.4 0l2.3-2.3c.5-.4.5-1.1.1-1.4z"/>
        </svg>
        <span class="tool-name">🔧 ${toolName}</span>
        ${args ? `<span class="tool-args">参数: ${args}</span>` : ''}
    `;
    App.messagesEl.appendChild(el);
    App.scrollToBottom();
  };
  App.addToolCallResult = function addToolCallResult(toolName, result, success) {
    let el = document.getElementById('tool-' + toolName);
    if (!el) {
      el = document.createElement('div');
      el.className = 'msg tool-call' + (success === false ? ' tool-error' : '');
      el.id = 'tool-' + toolName;
      el.innerHTML = `<span class="tool-name">🔧 ${toolName}</span>`;
      App.messagesEl.appendChild(el);
    }
    if (success === false) {
      el.classList.add('tool-error');
    }
    const resultDiv = document.createElement('div');
    resultDiv.className = 'tool-result';
    resultDiv.textContent = '结果: ' + (result || '(无)');
    el.appendChild(resultDiv);
    App.scrollToBottom();
  }; // ========== 会话管理 ==========
  App.sessionModalEl = undefined;
  App.sessionListEl = undefined;
  App.sessionBtn = undefined;
  App.sessionModalClose = undefined;
  App.newSessionBtn = undefined;
  App.initSessionUI = function initSessionUI() {
    App.sessionModalEl = document.getElementById('session-modal');
    App.sessionListEl = document.getElementById('session-list');
    App.sessionBtn = document.getElementById('session-btn');
    App.sessionModalClose = document.getElementById('session-modal-close');
    App.newSessionBtn = document.getElementById('new-session-btn');
    App.sessionBtn.addEventListener('click', () => {
      App.sessionModalEl.classList.add('show');
      // 请求会话列表
      if (App.ws && App.ws.readyState === WebSocket.OPEN) {
        App.ws.send(JSON.stringify({
          type: 'list_sessions'
        }));
      }
    });
    App.sessionModalClose.addEventListener('click', () => {
      App.sessionModalEl.classList.remove('show');
    });
    App.sessionModalEl.querySelector('.modal-backdrop').addEventListener('click', () => {
      App.sessionModalEl.classList.remove('show');
    });
    App.newSessionBtn.addEventListener('click', () => {
      if (App.ws && App.ws.readyState === WebSocket.OPEN) {
        App.ws.send(JSON.stringify({
          type: 'new_session'
        }));
        // 清除当前对话显示
        App.messagesEl.innerHTML = '';
        App.sessionModalEl.classList.remove('show');
      }
    });
  };
  App.renderSessionList = function renderSessionList(sessions) {
    if (!App.sessionListEl) return;
    App.sessionListEl.innerHTML = '';
    if (!sessions || sessions.length === 0) {
      App.sessionListEl.innerHTML = '<div style="text-align:center;color:var(--text-dim);padding:20px">暂无历史对话</div>';
      return;
    }
    sessions.forEach(s => {
      const item = document.createElement('div');
      item.className = 'session-item' + (s.is_active ? ' active' : '');
      item.innerHTML = `
            <div class="session-icon">
                <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/></svg>
            </div>
            <div class="session-info">
                <div class="session-title">${s.title || '对话'}</div>
                <div class="session-time">${s.updated_at ? new Date(s.updated_at).toLocaleString() : ''}</div>
            </div>
            <button class="session-delete" data-sid="${s.id}" title="删除">
                <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
            </button>
        `;

      // 点击切换到该会话
      item.addEventListener('click', e => {
        if (e.target.closest('.session-delete')) return;
        if (App.ws && App.ws.readyState === WebSocket.OPEN) {
          App.ws.send(JSON.stringify({
            type: 'switch_session',
            session_id: s.id
          }));
          App.sessionModalEl.classList.remove('show');
        }
      });

      // 删除按钮
      item.querySelector('.session-delete').addEventListener('click', e => {
        e.stopPropagation();
        if (App.ws && App.ws.readyState === WebSocket.OPEN) {
          App.ws.send(JSON.stringify({
            type: 'delete_session',
            session_id: s.id
          }));
          item.remove();
          App.showToast('对话已删除');
        }
      });
      App.sessionListEl.appendChild(item);
    });
  };
  App.renderSessionListFromState = function renderSessionListFromState() {
    // 重新请求会话列表
    if (App.ws && App.ws.readyState === WebSocket.OPEN) {
      App.ws.send(JSON.stringify({
        type: 'list_sessions'
      }));
    }
  };
  App.sendText = function sendText(text) {
    if (!App.ws || App.ws.readyState !== WebSocket.OPEN) {
      App.showToast('未连接到服务器');
      return;
    }
    // 记录用户活跃时间（供 RL 统一状态/心跳取数）
    App._lastUserMessageTime = Date.now();
    App._lastUserInteractTime = Date.now();
    if (App._expressionRL) App._expressionRL.notifyUserMessage();
    // 用户文字输入可打断 AI 回复（唯一打断来源之一；AI action 一律不打断）
    if (App.currentState === App.State.SPEAKING || App.currentState === App.State.THINKING) {
      App.triggerInterrupt();
    }
    // AI 移动不再因用户发消息而打断（AI 可以边走边聊，活人感）
    App.ws.send(JSON.stringify({
      type: 'text',
      content: text
    }));
  };
  /* ============================================================
   *  RL 统一调度心跳（完全统摄：让后端 proactive_tick 自动跑起来）
   * ============================================================ */
  App._rlLastDispatchTime = 0;
  /** 构建统一状态快照并同步给后端 RL 协调器。
   *  wantDecision=true 时请求统一调度决策（rl_decision → rl_dispatch 闭环）。 */
  App.sendRLSync = function sendRLSync(wantDecision) {
    if (!App.ws || App.ws.readyState !== WebSocket.OPEN) return;
    // 关系状态快照（前端 UnifiedDatingSystem 与后端共享同一套语义）
    let affection = 0, trust = 0, intimacy = 0;
    const ds = App._datingSystem || App._engagementRL;
    if (ds && typeof ds.getDebugInfo === 'function') {
      try {
        const info = ds.getDebugInfo();
        const m = info.relationship && info.relationship.metrics;
        if (m) {
          affection = m.affection || 0;
          trust = m.trust || 0;
          intimacy = m.intimacy || 0;
        }
      } catch (e) {}
    }
    // 游戏模式状态
    const gm = App.gameModeManager;
    const gameActive = !!(gm && gm.currentGame && gm.currentGame.state !== 'completed' && gm.currentGame.state !== 'failed');
    const lastMsg = App._lastUserMessageTime || 0;
    const data = {
      affection,
      trust,
      intimacy,
      emotion: 0,
      want_decision: !!wantDecision,
      event: 'proactive_tick',
      game_state: gameActive ? 'playing' : 'idle',
      game_key: gameActive ? (gm.currentGame.key || gm.currentGame.name || '') : '',
      seconds_since_user_message: Math.max(0, (Date.now() - lastMsg) / 1000),
      user_engaged: (Date.now() - lastMsg) < 120000,
    };
    App.ws.send(JSON.stringify({ type: 'rl_sync', data }));
  };
  /** 启动 RL 统一调度心跳（动态间隔）：
   *  游戏模式每 30s 同步（让 RL 决策游戏快照间隔更及时）；
   *  大厅模式每 90s 同步，且仅当用户 5 分钟以上未互动时才请求主动决策，
   *  防止"自言自语"。 */
  App.startRLHeartbeat = function startRLHeartbeat() {
    if (App.rlHeartbeat) { clearTimeout(App.rlHeartbeat); App.rlHeartbeat = null; }
    const rlHeartbeatTick = () => {
      if (!App.ws || App.ws.readyState !== WebSocket.OPEN) { App.rlHeartbeat = null; return; }
      // AI 正在说话/思考时跳过，避免打断
      const busy = App.currentState === App.State.SPEAKING || App.currentState === App.State.THINKING;
      // 距上次派发不足 60 秒则跳过（服务端冷却 + 前端兜底，双保险）
      const canRequest = !busy && (Date.now() - (App._rlLastDispatchTime || 0)) >= 60000;
      // 游戏进行中由游戏内部 game_action_request 走同一协调器统一入口，不重复触发
      const gm = App.gameModeManager;
      const gameActive = !!(gm && gm.currentGame && gm.currentGame.state !== 'completed' && gm.currentGame.state !== 'failed');
      if (gameActive) {
        App.sendRLSync(false); // 仅同步状态（含游戏快照间隔决策），决策交给游戏内循环
        const next = 30000;
        App.rlHeartbeat = setTimeout(rlHeartbeatTick, next);
        return;
      }
      // 大厅：用户 5 分钟以上未互动才请求主动决策；刚互动过只同步状态（AI 不说话）
      if (canRequest) {
        const secSinceMsg = (Date.now() - (App._lastUserMessageTime || 0)) / 1000;
        App.sendRLSync(secSinceMsg >= 300);
      }
      App.rlHeartbeat = setTimeout(rlHeartbeatTick, 90000);
    };
    App.rlHeartbeat = setTimeout(rlHeartbeatTick, 30000);
  };
  App.sendAudioBase64 = function sendAudioBase64(b64, mimeType) {
    if (!App.ws || App.ws.readyState !== WebSocket.OPEN) {
      App.showToast('未连接到服务器');
      return;
    }
    App.ws.send(JSON.stringify({
      type: 'audio',
      data: b64,
      mime_type: mimeType || 'audio/webm'
    }));
  };
  /* ============================================================
   *  TTS 播放 + 口型同步
   * ============================================================ */
});