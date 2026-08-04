export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ---------- DOM ---------- */
  App.$ = id => document.getElementById(id);
  App.canvas = App.$('three-canvas');
  App.statusBadge = App.$('status-badge');
  App.subtitle = App.$('subtitle');
  App.messagesEl = App.$('messages');
  App.scrollHint = App.$('scroll-hint');
  App.textInput = App.$('text-input');
  App.sendBtn = App.$('send-btn');
  App.voiceBtn = App.$('voice-btn');
  App.toastEl = App.$('toast');
  App.resetCamBtn = App.$('reset-cam-btn');
  App.fullscreenBtn = App.$('fullscreen-btn');
  App.chatToggle = App.$('chat-toggle');
  App.dropHint = App.$('drop-hint');
  App.modelLoading = App.$('model-loading');
  App.modelLoadingText = App.$('model-loading-text'); // 背景场景
  App.bgBtn = App.$('bg-btn');
  App.bgModal = App.$('bg-modal');
  App.bgModalClose = App.$('bg-modal-close');
  App.bgListEl = App.$('bg-list');
  App.bgFileInput = App.$('bg-file-input'); // 移动模式
  App.moveBtn = App.$('move-btn'); // 第一人称探索
  App.fpvBtn = App.$('fpv-btn');
  App.fpvCrosshair = App.$('fpv-crosshair');
  App.fpvJoystick = App.$('fpv-joystick');
  App.fpvJoystickThumb = App.$('fpv-joystick-thumb');
  App.fpvExitBtn = App.$('fpv-exit'); // 对我的称呼（已集成到角色卡片）
  // 相机设置
  App.roleCardBtn = App.$('role-card-btn');
  App.roleCardModal = App.$('role-card-modal');
  App.roleCardModalClose = App.$('role-card-modal-close');
  App.roleCardList = App.$('role-card-list');
  App.roleCardCreateBtn = App.$('role-card-create-btn');
  App.roleCardEditModal = App.$('role-card-edit-modal');
  App.roleCardEditClose = App.$('role-card-edit-close');
  App.rcName = App.$('rc-name');
  App.rcRoleName = App.$('rc-role-name');
  App.rcUserName = App.$('rc-user-name');
  App.rcModelSelect = App.$('rc-model-select');
  App.rcModelUploadBtn = App.$('rc-model-upload-btn');
  App.rcModelFileInput = App.$('rc-model-file-input');
  App.rcLlmBaseUrl = App.$('rc-llm-base-url');
  App.rcLlmApiKey = App.$('rc-llm-api-key');
  App.rcLlmModel = App.$('rc-llm-model');
  App.rcLlmModelList = App.$('rc-llm-model-list');
  App.rcLlmRefreshBtn = App.$('rc-llm-refresh-btn');
  App.rcLlmTip = App.$('rc-llm-tip');
  App.rcTtsTabs = App.$('rc-tts-tabs');
  App.rcTtsEdgePanel = App.$('rc-tts-edge-panel');
  App.rcTtsGsoPanel = App.$('rc-tts-gso-panel');
  App.rcVoiceSelect = App.$('rc-voice-select');
  App.rcRateRange = App.$('rc-rate-range');
  App.rcRateVal = App.$('rc-rate-val');
  App.rcGsoUrl = App.$('rc-gso-url');
  App.rcGsoRef = App.$('rc-gso-ref');
  App.rcGsoChar = App.$('rc-gso-char');
  App.rcSystemPrompt = App.$('rc-system-prompt');
  App.rcApplyBtn = App.$('rc-apply-btn');
  App.rcDeleteBtn = App.$('rc-delete-btn');
  App.rcToolsEnabled = App.$('rc-tools-enabled');
  App.rcToolsField = App.$('rc-tools-field');
  App.rcToolsList = App.$('rc-tools-list'); // 相机设置
  App.camSettingsBtn = App.$('cam-settings-btn');
  App.camSettingsModal = App.$('cam-settings-modal');
  App.camSettingsModalClose = App.$('cam-settings-modal-close');
  App.camHeightRange = App.$('cam-height-range');
  App.camHeightVal = App.$('cam-height-val');
  App.camDistanceRange = App.$('cam-distance-range');
  App.camDistanceVal = App.$('cam-distance-val');
  App.camTiltRange = App.$('cam-tilt-range');
  App.camTiltVal = App.$('cam-tilt-val');
  App.camSettingsSaveBtn = App.$('cam-settings-save-btn');
  App.vrGazeAssistToggle = App.$('vr-gaze-assist-toggle');
  // BGM
  App.bgmBtn = App.$('bgm-btn');
  App.bgmModal = App.$('bgm-modal');
  App.bgmModalClose = App.$('bgm-modal-close');
  App.bgmListEl = App.$('bgm-list');
  App.bgmFileInput = App.$('bgm-file-input');
  App.bgmEmptyMsg = App.$('bgm-empty-msg');
  /* ---------- 状态 ---------- */
  App.State = {
    IDLE: 'idle',
    THINKING: 'thinking',
    LISTENING: 'listening',
    SPEAKING: 'speaking'
  };
  App.currentState = App.State.IDLE;
  App.gazeAssistEnabled = true; // VR 注视归位开关（相机设置弹窗可切换，默认开启）
  App.isRecording = false;
  App.mediaRecorder = null;
  App.audioChunks = [];
  App.ws = null;
  App.wsHeartbeat = null; // WebSocket 心跳定时器
  App.wsReconnectTimer = null; // 重连定时器
  App.wsConnTimeout = null; // 连接超时定时器
  App.currentAudio = null;
  App.audioCtx = null;
  App.analyser = null;
  App.analyserData = null; // --- 流式播放队列 ---
  App.audioQueue = []; // 待播放的句子 [{seq, text, audio_b64, audio_mime}]
  App.isPlayingQueue = false;
  App.currentReplyText = ''; // 当前回复累积文本
  App.currentReplySession = null; // 当前回复 session_id（用于判断过期消息）
  App.pendingAIMsgEl = null; // 流式回复占位 DOM
  // --- VAD 自动对话模式 ---
  App.voiceMode = 'auto'; // 'press' 按住说话 | 'auto' 自动对话（默认开启）
  App.vadStream = null; // VAD 持续监听的 MediaStream
  App.vadAnalyser = null;
  App.vadData = null;
  App.vadRAF = null;
  App.vadState = 'idle'; // 'idle' | 'recording'
  App.vadSilenceStart = 0;
  App.vadInterruptStart = 0;
  App.vadVoiceStart = 0; // 人声连续确认计时（防噪声单帧误判）
  App.vadRecorder = null;
  App.vadChunks = [];
  App._vadClonedTrack = null;
  App.VAD_THRESHOLD = 0.05; // 普通说话音量阈值（降低让小声也能触发）
  App.VAD_INTERRUPT_THRESHOLD = 0.09; // AI 说话时打断阈值（略高于说话阈值，避免 AI 外放声音/环境音自打断；用户开口必然超过）
  App.VAD_SILENCE_MS = 1800; // 静音多久判定结束（加长避免说话停顿被切断）
  App.VAD_INTERRUPT_MS = 250; // AI 说话时持续多久高音量判定打断（缩短加快响应）
  App.VAD_MIN_RECORD_MS = 350; // 最短录音时长，避免短噪音触发
  // --- 人声特性检测（VAD 噪音过滤：只有人声才触发聆听中）---
  App.VAD_VOICE_ENABLED = true; // 是否启用语音特性评分（true=严格人声过滤）
  App.VAD_VOICE_SCORE_THRESHOLD = 0.40; // 语音特性评分阈值：≥此值才视为人声
  App.VAD_VOICE_CONFIRM_MS = 120; // 人声连续确认时长：持续这么久才算开始说话（防噪声单帧误判）
  App.VAD_VOICE_MIN_F0 = 70; // 基频下限 Hz（低于此视为低频噪音/机械声）
  App.VAD_VOICE_MAX_F0 = 480; // 基频上限 Hz（高于此视为高频噪音/口哨）
  App.VAD_HARMONIC_BINS = 3; // 基频倍数谐波检测数（F0×2, ×3, ×4）
  // --- 低功耗模式 ---
  App.lowPowerMode = false; // 是否处于低功耗模式
  App.LP_KEY = 'dabai.lowPower';
  /* ============================================================
   *  互动强化学习（非游戏模式）
   * ============================================================ */
  App.engagementRLActive = false; // 默认关闭，稍后由 boot 根据配置开启
  App._engagementRL = null;

  /* ============================================================
   *  恋爱养成强化学习系统
   * ============================================================ */
  App.datingSystemActive = false; // 默认关闭，由 boot 或用户开启
  App._datingSystem = null;

  /* ============================================================
   *  Three.js 场景
   * ============================================================ */
});