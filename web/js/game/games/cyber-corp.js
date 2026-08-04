/* ============================================================
 * 赛博公司 CyberCorp —— 基于「蜂群游戏引擎」架构改造的公司经营游戏
 *
 * 引擎需求映射：
 * - R1 赛博公司拥有独立系统 → 世界实例化：每个 CEO 一个世界（world_id），
 *     存档按世界分桶（cybercorp.world.<world_id>），世界间零串档
 * - R2 大厅角色不带入游戏 → 双身份空间：CEO 身份（CEO 姓名 + 公司名 + 理念）
 *     独立于大厅角色，进入世界时全新创建，不继承大厅形象
 * - R3 CEO 玩家完全独立 → 单租户世界：玩家以 CEO 身份独占世界，是唯一所有者，
 *     玩家指令（靠近说话）是世界内最高优先级输入
 * - R4 所有员工都知道玩家身份 → 身份注入：CEO 身份块注入每个智能体
 *     （HR / 求职者 / 员工）的每次推理提示词，员工入职即向 CEO 报到
 *
 * 蜂群运行时（高并发架构落地）：
 * - 员工智能体不占线程，由主循环驱动的"蜂群互动节拍器"调度（_swarmTick）
 * - 决策节流 + 冷却 + LLM 失败兜底，绝不阻塞世界主循环
 * - 世界运行时 HUD 展示 world_id / CEO / 在线 Agent / 决策计数 / 世界时钟
 * - 每个员工拥有独立 RL 策略（SwarmRLAgent）+ 公司任务看板：
 *     自主决策（工作/打磨/协助/汇报/休整）真实推进任务、产出公司评分
 *
 * 幽灵玩家系统：
 * - 玩家不使用大厅角色模型：游戏声明 fpvCamera + playerAnchor（无实体幽灵锚点），
 *     相机与移动完全基于幽灵锚点，大厅角色保持隐藏、不漫步、不参与游戏
 * - 大厅 AI 对话自动切换到蜂群系统：聊天输入只路由给身边员工，AI 回复来自蜂群
 * - 每个员工有独立对话记录（世界隔离持久化），靠近即显示
 *
 * 玩法：
 * - 玩家就任 CEO（创建独立世界身份）→ 任命 HR → 观看 AI 求职者面试
 * - 通过者入职，成为蜂群成员：RL 自主工作、互相协作、向 CEO 汇报、参与会议
 * - CEO 布置任务（如「布置任务：做一个恋爱游戏」）→ 蜂群规划会按岗位拆解子任务，
 *     全员各司其职；员工卡住时按岗位依赖链主动找同事求助（协作真实推进任务）；
 *     交付评审 / 创意提案等会触发蜂群讨论：支持 / 建议 / 否决 → 返工或采纳
 * - 所有岗位招满（或候选人耗尽）后游戏完成
 * ============================================================ */

import { BaseGame } from './base-game.js';

// ==================== 岗位定义 ====================
const POSITIONS = [
  { key: 'hr',    name: '人力资源总监', icon: '🕶', color: '#00ffff' },
  { key: 'dev',   name: '全栈工程师',   icon: '⌨️', color: '#00ffff', keywords: ['代码', '编程', '开发', '程序', '项目', '技术', '系统', '算法', '架构', 'python', 'js', '数据库', 'bug', '接口', '前端', '后端'] },
  { key: 'artist', name: '3D 美术设计师', icon: '🎨', color: '#ff4df0', keywords: ['美术', '建模', '3d', '渲染', '设计', '动画', '模型', '光影', '材质', 'blender', 'unity', '贴图', '风格'] },
  { key: 'planner', name: '游戏策划',   icon: '📐', color: '#ffd166', keywords: ['策划', '玩法', '关卡', '创意', '设计', '数值', '剧情', '体验', '平衡', '用户', '乐趣'] },
  { key: 'pm',    name: '产品经理',     icon: '📊', color: '#6ee7ff', keywords: ['产品', '需求', '用户', '市场', '分析', '沟通', '迭代', '数据', '商业', '规划', '项目', '流程'] },
  { key: 'qa',    name: '测试工程师',   icon: '🧪', color: '#9dff6e', keywords: ['测试', 'bug', '质量', '用例', '验证', '缺陷', '回归', '自动化', '严谨', '排查', '边界', '复现'] },
  { key: 'ai',    name: 'AI 研究员',    icon: '🧠', color: '#ff9d6e', keywords: ['ai', '人工智能', '算法', '模型', '数据', '训练', '神经', '学习', '智能', 'gpt', 'llm', '神经网络', '推理'] },
];

// ==================== 场景布局 ====================
// 每个角色都有自己的独立办公间：玻璃隔断 + 岗位主题色，开放式入口面向面试区。
// 办公间沿后墙一字排开，前方留出完整面试区与中枢区，视野开阔、互不遮挡。
const ROOM_ACCENTS = {
  hr: 0x00e5ff,
  dev: 0x00b3ff,
  artist: 0xff4df0,
  planner: 0xffd166,
  pm: 0x6ee7ff,
  qa: 0x9dff6e,
  ai: 0xff9d6e,
};

const LAYOUT = {
  office: { halfW: 23, halfD: 15 },
  playerSpawn: { x: 0, z: 6 },              // 靠中后部，保证后方轨道相机仍在办公室内（后墙 z=15.25）
  stage: { desk: { x: 0, z: 0 }, hr: { x: 4.6, z: 0 }, seeker: { x: -4.6, z: 0 } },
  holoScreen: { x: 0, y: 6.0, z: -5.4 },    // 全息屏上移至办公间上方，不再遮挡面试区与工位
  workstations: [
    { posKey: 'hr',     x: -16.0, z: -11.3 },
    { posKey: 'dev',    x: -11.0, z: -11.3 },
    { posKey: 'artist', x: -6.0,  z: -11.3 },
    { posKey: 'planner', x: -1.0, z: -11.3 },
    { posKey: 'pm',     x: 4.0,   z: -11.3 },
    { posKey: 'qa',     x: 9.0,   z: -11.3 },
    { posKey: 'ai',     x: 14.0,  z: -11.3 },
  ],
  room: { halfW: 1.9, halfD: 2.2, wallH: 2.7 },   // 每间办公间尺寸（半宽 / 半深 / 墙高）
  waitingSpot: { x: 6, z: 3 },              // 下一位候选人的候场位（玩家正前方偏右，相机视野内）
  entrance: { x: -14, z: 13 },             // 公司入口（装饰）
  reception: { x: -10.8, z: 10.8 },        // 前台接待台
  atrium: { x: 0, z: -4.8 },               // 公司中枢区（地面全息徽标）
};

// 角色脚底相对地面的高度：略高于地面装饰线（GridHelper 0.02 / 发光圆环 0.03-0.035），
// 避免脚底与装饰线重合、被半透明圆环"切开脚踝"而产生"陷在地里"的视觉观感
const STAND_Y = 0.05;

// 幽灵玩家跳跃物理（与迷宫寻宝一致：空格起跳 + 重力落体）
const JUMP_GRAVITY = 15;        // 重力加速度
const JUMP_SPEED = 4.8;         // 起跳初速度（≈0.77m 跳高）
const MAX_FALL_SPEED = 25;      // 最大下落速度

// 头顶对话气泡底部锚点高度（米）：按用户要求固定在 1.28
const BUBBLE_BASE_Y = 1.28;

// 空间音频听力半径（米）：任何角色/玩家说话，周围 3 米内才能听到
// （降低干扰半径，避免满办公室员工同时互相"听见"导致声音互相打扰）
const HEAR_RADIUS = 3.0;
// 音量衰减起始距离（米）：0-1m 满音量，1m 后开始线性衰减，3m 外完全无声
const VOLUME_FADE_START = 1.0;

// ==================== 角色微动作库（仅保留轻微头部动作） ====================
// 每个游戏角色独立持有：数值通道 = 单次脉冲（attack→hold→release 包络），
// {amp, loops} 通道 = 正弦振荡 × 包络。骨骼名与 CharacterActor.vrmBones 一致。
// 只保留轻微转头/点头/看屏幕等头部动作；不再包含伸懒腰/抱臂/叉腰/托腮/看手机等
// 大幅肢体动作（躯干与手臂静止，避免侧身歪斜、东倒西歪）。
const ACTOR_MOTIONS = {
  glance_around: { dur: 1.8, head: { y: { amp: 0.05, loops: 2 } } },
  head_turn_l:   { dur: 1.1, head: { y: -0.045 }, neck: { y: -0.015 } },
  head_turn_r:   { dur: 1.1, head: { y: 0.045 }, neck: { y: 0.015 } },
  nod_small:     { dur: 0.9, head: { x: { amp: 0.035, loops: 2 } } },
  head_tilt_l:   { dur: 1.0, head: { z: 0.035 } },
  head_tilt_r:   { dur: 1.0, head: { z: -0.035 } },
  look_up:       { dur: 1.8, head: { x: -0.045 }, neck: { x: -0.015 } },
  look_down:     { dur: 2.0, head: { x: 0.055 }, neck: { x: 0.015 }, hold: 0.25 },
};
// 全局幅度系数：在已收敛的动作库基础上再压一档，实际偏移极小
// （转头 ≈±1.5°、点头 ≈±1°、歪头 ≈±0.8°），保持"活人感"但绝不晃动
const ACTOR_MOTION_SCALE = 0.4;

// 员工机动性：开会时聚集的站位（办公室中枢区前方，面朝舞台/大屏，避开面试区）
const MEET_SPOTS = [
  { x: -5.5, z: -3.2 }, { x: -2.7, z: -3.2 }, { x: 0, z: -3.2 },
  { x: 2.7, z: -3.2 }, { x: 5.5, z: -3.2 }, { x: 0, z: -1.8 },
];
// 员工机动性：日常"跑腿"目的地池（工位 → 中枢/大屏/前台/同事工位 → 回工位）
const BUSY_SPOTS = [
  { x: 0, z: -4.8 }, { x: 2.4, z: -2.4 }, { x: -2.4, z: -2.4 },
  { x: 0, z: 2.2 }, { x: -10.8, z: 10.8 }, { x: -14, z: 13 },
];

// 员工总数上限：最多招 6 名员工，满员停招（HR 不计入）
const MAX_EMPLOYEES = 6;

// ==================== 公司评分平衡常量 ====================
// 五维加权：组织活力 40 + 人才质量 30 + 运营稳定 15 + 公司声誉 10 + 公司产出 5 = 100
// 组织活力 = 裁员微笑曲线(单期裁员人数 → 0-40 分) - 僵化惩罚(长期不裁员逐期失血)
// 公司产出 = 员工 RL 自主完成任务的产出累积（RL_CFG.PRODUCTION_CAP，任务系统驱动）
const BALANCE = {
  VITALITY_FULL: 40,          // 组织活力满分
  QUALITY_FULL: 30,           // 人才质量满分
  STABILITY_FULL: 15,         // 运营稳定满分（原 20，让出 5 分给 RL 产出维度）
  REPUTATION_FULL: 10,        // 公司声誉满分

  // 微笑曲线：单期裁员人数 → 组织活力分（6 人满编公司基准）
  // 峰值在每期裁 1 人（40 分），两端分别以慢性和急性方式惩罚
  SMILE: [[0, 20], [1, 40], [2, 34], [3, 22], [4, 12], [5, 5], [6, 0]],

  STALE_START: 3,             // 连续 N 期不裁员后开始僵化惩罚
  STALE_STEP: 2,              // 每期额外 -2
  STALE_CAP: 10,              // 僵化惩罚封顶 -10

  FIT_HIGH: 80,               // 高适配阈值
  FIT_BONUS_HIGH: 5,          // 适配度 ≥80 录用 +5/人
  FIT_BONUS_NORMAL: 2,        // 适配度 60-79 录用 +2/人
  FIT_PENALTY_LOW: -2,        // 适配度 <60 录用 -2/人
  QUALITY_AVG_WEIGHT: 0.25,   // 平均适配度折算系数

  REP_GOOD: 1,                // 单期裁员 ≤1 人：健康汰换 +1（封顶 10）
  REP_WARN: -1,               // 单期裁员 2 人：媒体关注 -1
  REP_CRASH: -4,              // 单期裁员 ≥3 人：舆论崩盘 -4
  REP_STALE: -1,              // 连续 ≥3 期不裁员：外界认为僵化 -1/期

  GRADE_LEVELS: [             // 六级评级（分数区间 → 等级）
    [90, 'S', '霓虹巨头'], [80, 'A', '行业翘楚'], [70, 'B', '稳定经营'],
    [60, 'C', '勉强维生'], [50, 'D', '危机四伏'], [0, 'E', '濒临破产'],
  ],
};

// ==================== 蜂群引擎常量（世界隔离 + 身份注入 + 蜂群运行时） ====================
// 世界隔离：存档按世界（CEO 身份）分桶，互不串档（R1 / R3）
const SAVE_KEY_PREFIX = 'cybercorp.world.';
const LAST_WORLD_KEY = 'cybercorp.lastWorld';
const WORLD_ID_PREFIX = 'w_ceo_';

// 蜂群运行时：员工智能体异步决策节流（模拟世界时钟驱动的 Agent 调度）
const SWARM = {
  TICK: [3, 6],         // 世界时钟滴答（秒）：调度器唤醒周期
  INTERVAL: [40, 70],   // （保留）旧版全局汇报节流，已被 RL 节律取代
  COOLDOWN: 40,         // 单员工协作发言最小间隔（秒）
  PROACTIVE_REPORT: [30, 65],   // 员工主动向 CEO 汇报的间隔（秒）
  SYNC_EVERY: [70, 120],        // 蜂群自动"进度碰头会"间隔（秒），无需玩家触发
  FALLBACK_REPORT: [    // LLM 不可用时的蜂群汇报兜底
    'CEO {ceo}，研发这边进展正常，有问题我会第一时间汇报。',
    'CEO {ceo}，我把手头的任务又过了一遍，质量没问题。',
    'CEO {ceo}，今天的数据我整理好了，随时可以看。',
    'CEO {ceo}，有个想法想跟你聊聊，等你方便的时候。',
    'CEO {ceo}，我在跟进项目进度，一切按计划推进。',
  ],
};

// 蜂群协作事件：员工互相传话（与 CEO 汇报按比例混合调度）
const COLLAB = {
  RATE: 0.55,           // 蜂群决策中协作事件占比（主动找同事咨询/同步更频繁）
  STUCK_RATE: 0.9,      // 员工"卡住"（进度落后/停滞）时主动求助同事的概率
  FALLBACK_A: [         // 传话方兜底
    '{target}，帮我看看这份数据，我有点拿不准。',
    '{target}，昨天那个方案你看了吗？我觉得可以推进了。',
    '{target}，客户那边反馈不错，接下来交给你对接了。',
    '{target}，新版本的问题单我整理好了，发你一份。',
    '{target}，进度同步一下，我这边模块已经完成了。',
  ],
  FALLBACK_STUCK: [     // 卡住/落后时主动求助兜底
    '{target}，我这边「{task}」卡住了，能搭把手吗？',
    '{target}，进度落后了，需要你这边支持一下。',
    '{target}，有个环节我拿不准，想请你一起看。',
    '{target}，我这边堵住了，你有空帮我顺一下吗？',
  ],
  FALLBACK_B: [         // 接收方兜底
    '{speaker}，收到，我这就看。',
    '{speaker}，没问题，交给我吧。',
    '{speaker}，好，我们分头推进，晚上对一下。',
    '{speaker}，数据我看过了，回头找你细聊。',
    '{speaker}，了解，辛苦了！',
  ],
};

// CEO 指令触发式玩法：靠近员工说话时识别的指令词
const CEO_COMMANDS = {
  MEETING: /开会|会议|集合|碰个头|全员会/,      // 召集全员会议
  DISMISS: /散会|结束会议|会议结束|解散/,        // 结束会议
  REPORT: /汇报|报告|报一下|同步一下|站会/,      // 全员报进度
  ASSIGN: /(?:新?任务|工作|项目)[:：]|(?:布置|安排|下达|分配|派发|发布|下发)(?:一个|一项|个|项)?(?:任务|工作|项目|活)|帮我(?:们)?(?:做|开发|设计|搞|做一个|做一款)|我要(?:你们|大家)?(?:做|开发|设计|搞|做一个|做一款)|(?:做|开发|设计|制作|搞)(?:一个|一款|一套|一版|一项)/,  // 布置新任务
  ASSIGN_FALLBACK: [    // 项目规划会上各岗位认领子任务的兜底发言（{task} 项目名 / {pos} 岗位名）
    '我（{pos}）负责把「{task}」的需求拆清、排好里程碑。',
    '我（{pos}）负责「{task}」的核心部分，随时可开工。',
    '「{task}」这块我（{pos}）来认领，接口和进度我会同步。',
    '作为{pos}，我会为「{task}」提供质量/体验兜底，验收我来盯。',
  ],
  MEETING_FALLBACK: [   // 会议发言兜底（{ceo} 老板名 / {pos} 岗位名）
    'CEO {ceo}，我这边（{pos}）一切正常，随时可以开工。',
    'CEO {ceo}，我负责的部分在按计划推进，没有问题。',
    'CEO {ceo}，建议下周把重点放在新项目上，我这边人手够了。',
    'CEO {ceo}，我提一个建议：把重复的报表流程自动化，能省不少人力。',
    'CEO {ceo}，收到会议通知，我汇报一下：一切顺利，无阻塞。',
  ],
};

// ==================== CEO 项目任务（玩家布置 → 蜂群拆解 → 各司其职） ====================
// 玩家布置的任务按岗位拆分：每个在职岗位认领一个子任务，进度/质量由员工 RL 决策推进
const PROJECT = {
  // 各岗位在项目中的职责模板（决定子任务标题，体现"各司其职"）
  SUBTASK_TEMPLATE: {
    pm:      '需求拆解与里程碑排期',
    planner: '玩法与体验设计',
    dev:     '核心功能开发',
    artist:  '美术风格与场景设计',
    ai:      '智能体行为实现',
    qa:      '验收用例与质量保障',
  },
  COMPLEXITY: [55, 75],     // 子任务复杂度随机区间
  QUALITY_BASE: 55,         // 子任务初始质量（与 RL_CFG.QUALITY_BASE 对齐）
  // 项目协作依赖链：谁的工作需要谁（用于"主动找到合适的同事合作"）
  PIPELINE: {
    pm:      ['planner'],            // PM 需要策划给玩法输入
    planner: ['pm', 'dev'],          // 策划需要 PM 定范围、开发评估落地
    dev:     ['planner', 'qa', 'ai'],// 开发需要策划设计、QA 用例、AI 接口
    artist:  ['planner', 'dev'],     // 美术需要策划需求、开发给资源规格
    ai:      ['dev', 'planner'],     // AI 需要开发主循环、策划行为设计
    qa:      ['dev', 'planner', 'pm'], // QA 需要开发产物、设计文档、PM 排期
    hr:      ['pm'],                 // HR 需要 PM 给人力需求
  },
  REVIEW_AT: 0.85,          // 进度达到该比例且质量不足时触发交付评审
  REVIEW_QUALITY: 68,       // 评审触发质量阈值（低于该值会被 QA 盯上）
  REVIEW_COOLDOWN: 45,      // 同一员工两次评审最小间隔（秒）
};

// ==================== 蜂群讨论（提议 → 表态：支持/建议/否决 → 结论） ====================
const DELIB = {
  VETO_SHARE: 0.34,         // 否决占比 ≥ 1/3 即否决（讨论 > 单点对抗）
  SUGGEST_BONUS: 2,         // 带建议通过：质量加成
  VETO_REFINE: 4,           // 被否决返工：质量修正 + 强制打磨一次
  PASS_QUALITY: 6,          // 评审通过：质量加成
  STANCE_HINTS: {           // 各岗位在讨论中的角色立场（注入 LLM 提示词）
    qa:      '你负责质量把关：发现风险或标准不达标时倾向「否决」，但必须给出明确理由。',
    pm:      '你负责需求与进度：倾向「建议」补范围、排期或风险缓冲，重大风险才「否决」。',
    dev:     '你负责技术落地：倾向「支持」，实现有困难时「建议」调整方案。',
    artist:  '你负责美术与表达：倾向「建议」提升视觉和表现力。',
    planner: '你负责玩法与体验：倾向「建议」补全体验细节，违背体验原则时「否决」。',
    ai:      '你负责 AI 能力：倾向「支持」数据与算法可行的方案，「建议」补充验证。',
    hr:      '你负责组织与人：倾向「支持」，人力不足时「建议」调整优先级。',
  },
  STANCE_FALLBACK: {        // 兜底表态文案（{topic} / {proposer} / {pos}）
    support: [
      '支持这个方案，方向和我的判断一致。',
      '支持，可以按这个思路推进。',
      '我赞成，落地上没有大问题。',
    ],
    suggest: [
      '建议补充验收标准和时间节点。',
      '建议先把范围收敛，优先保证核心体验。',
      '建议预留风险缓冲，别把排期排太满。',
      '建议先做一轮小范围验证再全面铺开。',
    ],
    veto: [
      '否决：目前风险太大，标准还没到位。',
      '否决：这个方案我不认可，质量关过不了。',
      '反对：条件不成熟，先补验证再谈推进。',
    ],
  },
  MOTION_FALLBACK: [        // 提案人兜底（{topic}）
    '我提议：{topic}，大家议一议。',
    '关于「{topic}」，我有一个方案，请各位评审。',
    '{topic}，我想听听大家的意见再定。',
  ],
};

// ==================== 公司任务看板（让 RL 决策有真实产出） ====================
// 每个岗位一项任务；员工通过 RL 决策（work/refine）推进进度与质量，完成后增加公司产出分
const COMPANY_TASKS = [
  { posKey: 'dev',     icon: '⌨️', title: '次世代引擎重构',        complexity: 100 },
  { posKey: 'artist',  icon: '🎨', title: '霓虹场景概念设计',      complexity: 90 },
  { posKey: 'planner', icon: '📐', title: '新玩法系统设计',        complexity: 85 },
  { posKey: 'pm',      icon: '📊', title: 'Q3 产品路线图',         complexity: 80 },
  { posKey: 'qa',      icon: '🧪', title: '自动化测试平台',        complexity: 75 },
  { posKey: 'ai',      icon: '🧠', title: '智能 NPC 行为系统',     complexity: 95 },
];

// ==================== 员工 RL 决策（独立策略 + 动作效果平衡） ====================
const RL_CFG = {
  ACTIONS: [               // 0 work / 1 refine / 2 assist / 3 report / 4 rest
    { key: 'work',   label: '工作推进', progress: [8, 14], energy: -12 },
    { key: 'refine', label: '打磨质量', progress: [2, 4],  quality: [6, 11], energy: -10 },
    { key: 'assist', label: '协助同事', progress: [3, 6],  energy: -5 },
    { key: 'report', label: '向CEO汇报', progress: [0, 2], energy: -3 },
    { key: 'rest',   label: '休整充电', progress: 0,       energy: 22 },
  ],
  ENERGY_MAX: 100,
  ENERGY_LOW: 30,           // 低于该值 work/refine 效率减半
  QUALITY_BASE: 55,         // 新任务初始质量
  PRODUCTION_DONE: 2,       // 任务完成 → 产出分（质量 ≥70 时 2.5）
  PRODUCTION_CAP: 5,        // 产出维度满分（并入五维总分：40+30+15+10+5=100）
  LEARN: { alpha: 0.3, gamma: 0.9, epsilon: 0.25, epsilonMin: 0.05, decay: 0.998 },
  ACT_EVERY: [9, 20],       // 每个员工独立 RL 行动节律（秒）：行动到期后由调度器执行决策
  SAVE_EVERY: 8,            // 每 N 次学习保存一次 Q 表
  ACTION_FALLBACK: [        // 各动作的演绎文案兜底（{ceo} {pos} {task}）
    '我把「{task}」往前推了一截，进度没问题。',
    '刚把「{task}」的质量又打磨了一遍，交付更稳了。',
    '我去帮同事那边搭了把手，进度共享。',
    'CEO {ceo}，我在推进「{task}」，进展顺利。',
    '我休整了一下，精力恢复，随时可以继续。',
  ],
  OBS_BUCKETS: [3, 3, 3, 2],   // 观察特征桶数：[任务池忙度, 个人进度, 精力, 上次奖励符号]
  OBS_RANGE: [3, 100, 100, 1], // 各特征归一化范围
};

// ==================== 员工独立对话记录（世界隔离持久化） ====================
const CHAT_LOG = {
  KEY_SUFFIX: '.chats',         // 实际键 = SAVE_KEY_PREFIX + worldId + KEY_SUFFIX
  MAX_PER_AGENT: 60,            // 每员工最多保留条数
  MAX_TEXT: 200,                // 单条文本截断长度
  PANEL_SHOW: 12,               // 靠近面板显示的最近条数
};

/* ============================================================
 * SwarmRLAgent —— 每个员工一个独立 RL 策略（轻量 Q 学习）
 *
 * - 状态 = 4 维离散观察：[任务忙度, 个人进度, 精力, 上次奖励符号]
 *   （桶数 OBS_BUCKETS，共 3*3*3*2=54 个状态，Q 表极小、毫秒级决策）
 * - 动作 = 5 选 1：work 工作推进 / refine 打磨质量 / assist 协助同事
 *   / report 向CEO汇报 / rest 休整充电
 * - 学习 = Q(s,a) ← Q + α(r + γ·max Q(s') − Q)，ε 随步数指数衰减
 * - 独立性 = 每个员工持有自己的 Q 表，随存档持久化；互不共享策略，
 *   员工的"性格与工作习惯"由各自学到的最优策略体现
 * ============================================================ */
class SwarmRLAgent {
  constructor(name, posKey) {
    this.name = name;
    this.posKey = posKey;
    this.q = {};                // "state:action" → Q 值（54×5 上限，实际访问即稀疏）
    this.steps = 0;             // 已决策步数（驱动 ε 衰减）
    this.epsilon = RL_CFG.LEARN.epsilon;
    this.totalReward = 0;       // 累计奖励（个人战绩）
  }

  _key(s, a) { return s + ':' + a; }

  qval(s, a) { const v = this.q[this._key(s, a)]; return typeof v === 'number' ? v : 0; }

  /** ε-greedy 选动作：探索随步数衰减，逐步收敛到个人最优工作习惯 */
  choose(state) {
    this.steps++;
    this.epsilon = Math.max(RL_CFG.LEARN.epsilonMin, this.epsilon * RL_CFG.LEARN.decay);
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * RL_CFG.ACTIONS.length);
    }
    let best = 0, bv = -Infinity;
    for (let a = 0; a < RL_CFG.ACTIONS.length; a++) {
      const v = this.qval(state, a);
      if (v > bv) { bv = v; best = a; }
    }
    return best;
  }

  /** 单步 Q 学习更新，返回新 Q 值 */
  learn(state, action, reward, nextState) {
    const L = RL_CFG.LEARN;
    const key = this._key(state, action);
    const old = this.qval(state, action);
    let maxNext = 0;
    for (let a = 0; a < RL_CFG.ACTIONS.length; a++) {
      const v = this.qval(nextState, a);
      if (v > maxNext) maxNext = v;
    }
    const target = reward + L.gamma * maxNext;
    this.q[key] = old + L.alpha * (target - old);
    this.totalReward += reward;
    return this.q[key];
  }

  /** 存档恢复 Q 表（世界隔离持久化，RL 学习跨会话保留） */
  restore(q) {
    if (q && typeof q === 'object') {
      this.q = q;
      this.epsilon = RL_CFG.LEARN.epsilonMin;   // 老员工：低探索、按经验行事
    }
  }

  snapshot() { return this.q; }
}

// ==================== 面试脚本回退（LLM 不可用时兜底） ====================
const FALLBACK_SCRIPT = {
  welcome: ['欢迎来到赛博公司。今天的招聘窗口已经开启，让我们看看下一位求职者吧。', '这里是赛博公司人事部，数据流已就绪，招聘程序启动。'],
  openQ: {
    dev: ['请简单介绍下自己，并说说你写过最引以为傲的代码项目。'],
    artist: ['欢迎来到赛博公司，请介绍下你的美术风格和代表作。'],
    planner: ['赛博公司的游戏策划岗，说说你最得意的玩法设计。'],
    pm: ['请谈谈你对产品经理这个岗位的理解，以及你做过的印象最深的产品。'],
    qa: ['测试工程师最需要的就是严谨，请说说你是怎么发现和处理 bug 的。'],
    ai: ['AI 研究员的岗位，请聊聊你对大模型或神经网络的理解。'],
  },
  scenarioQ: {
    dev: ['如果线上服务突然出现高并发崩溃，你会如何排查和解决？'],
    artist: ['假如让你为赛博公司设计一套 3D 吉祥物形象，你会怎么构思？'],
    planner: ['如果给你一周时间设计一个让玩家上瘾的小玩法，你会怎么做？'],
    pm: ['如果产品上线后用户数据持续下滑，你会怎么分析并迭代？'],
    qa: ['如果开发坚持说这个 bug 不存在，你会用什么方法验证并说服他？'],
    ai: ['如果让你用一个周末训练一个能识别手写数字的模型，你会怎么选型？'],
  },
  stressQ: ['快问快答：说说你最大的缺点。', '如果录用你后发现你并不胜任，你打算怎么办？', '给你三句话，说服我们为什么必须选你。'],
  closingQ: ['关于这个岗位，你还有什么想了解的吗？', '面试到这里，你有什么想问我们的吗？'],
  followQ: ['能具体举个例子吗？', '如果遇到难题，你会怎么解决？', '你觉得自己最大的优势是什么？'],
  answer: [
    '我平时会保持学习，对新技术很感兴趣，遇到问题习惯动手去验证。',
    '我做事比较认真，喜欢把细节做好，团队合作也一直是我的强项。',
    '我觉得兴趣是最好的老师，所以我总能投入进去，把任务做扎实。',
    '我适应能力很强，新环境上手快，而且特别愿意沟通。',
  ],
  verdictPass: [
    '很好，你通过了面试。欢迎加入赛博公司，稍后人事系统会为你分配工位。',
    '数据吻合，能力达标。恭喜你，被录用了。',
  ],
  verdictFail: [
    '很遗憾，你的能力和岗位需求匹配度还不够，希望我们下次有机会合作。',
    '感谢你来参加面试，不过这次岗位匹配度有限，祝你找到更适合的机会。',
  ],
};

// ==================== 角色卡片模型加载失败时的全息投影替身 ====================
function makeHologramAvatar(THREE) {
  const g = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color: 0x22ddff, transparent: true, opacity: 0.55 });
  // 标准站立身高：胶囊体中心 0.58（底部正好踩地）+ 头部 1.42，总高约 1.68
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.28, 0.6, 4, 12), mat);
  body.position.y = 0.58;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.26, 16, 12), mat);
  head.position.y = 1.42;
  g.add(body, head);
  // 与大厅 applyLoadedModel 的目标身高 2.2 保持一致，避免替换真实 VRM 时高度跳变
  g.scale.setScalar(2.2 / 1.68);
  g.userData._hologram = true;
  return g;
}

/* ============================================================
 * CharacterActor —— 游戏内独立角色（HR / 求职者 / 员工）
 * 每个角色拥有独立的 VRM 实例、骨骼缓存、表情与 TTS 音色
 * ============================================================ */
class CharacterActor {
  constructor(game, card, tag) {
    this.game = game;
    this.App = game.App;
    this.THREE = game.THREE;
    this.card = card || {};
    this.tag = tag;                        // 'hr' | 'seeker' | 'employee'
    this.name = card?.name || '未知角色';
    this.role = card?.role_name || this.name;
    this.voice = (card?.tts && card.tts.edge_voice) || '';
    this.rate = (card?.tts && card.tts.edge_rate) || '+8%';

    this.group = null;
    this.vrm = null;
    this.vrmBones = {};
    this.exprNames = {};
    this._exprCur = {};
    this.loaded = false;
    this.removed = false;
    this._holoActive = false;            // 全息替身占位中（真实模型加载完成后关闭）
    this._falling = false;               // （已取消）真实模型加载后从 2m 高处落下——影响动作呈现

    this.speed = 3.1;                      // 行走速度 m/s（提升机动性，营造忙碌感）
    this.waypoints = null;                 // 路径队列
    this._onArrive = null;
    this.isMoving = false;
    this.faceTarget = null;                // 看向的世界坐标（Vector3 或 {x,z}）
    this.isSpeaking = false;
    this._walkPhase = 0;
    this._lastStepKey = -1;                // 脚步音效相位跟踪（独立于玩家，避免多角色互扰）
    this._phase = Math.random() * 6.28;    // 个人节奏相位

    // 大厅式微动作系统（每个角色独立）：动作库 + 序列器 + 空闲微动作计时
    this._motionActive = false;
    this._motionName = '';
    this._motionDef = null;
    this._motionElapsed = 0;
    this._motionCur = {};                  // 骨骼名 → {x,y,z} 平滑偏移（每帧插值）
    this._motionIdleT = 0;
    this._motionIdleInterval = 12 + Math.random() * 10;   // 12-22 秒一次微动作（与大厅 14-26s 一致）

    // 表情状态
    this._emotionKey = null;               // 'happy' | 'surprised' | 'thoughtful' | 'sad' | 'angry'
    this._blinkT = 1 + Math.random() * 2;
    this._blinking = false;
    this._blinkPhase = 0;

    // 头顶对话气泡（赛博全息风格：说话时弹出，闭嘴后淡出）
    this.bubble = null;                    // THREE.Sprite
    this._bubbleCanvas = null;
    this._bubbleCtx = null;
    this._bubbleTex = null;
    this._bubbleW = 3.6;                   // 气泡世界宽度（米）
    this._bubbleH = 1.0;
    this._bubbleVisible = false;
    this._bubblePop = 0;                   // 弹出动画进度 0→1
    this._bubbleOpacity = 0;
  }

  /**
   * 加载角色模型（两步式，永不失败）：
   * 1) 立即创建全息投影替身占位，角色立刻可见、可移动；
   * 2) 后台加载真实 VRM 模型，完成后无缝替换替身。
   * 模型缺失/下载失败时保留全息替身，保证面试流程永不卡死。
   */
  async load() {
    if (this.loaded || this.removed) return this;
    const THREE = this.THREE;
    const url = this.card.model_url || '';

    // 1) 全息替身占位（同步创建 group，调用方无需 await 即可 setPosition）
    const holo = makeHologramAvatar(THREE);
    this.group = new THREE.Group();
    this.group.add(holo);
    this.game.addToScene(this.group);
    this._holoActive = true;
    this.loaded = true;
    if (!url) return this;

    // 2) 后台加载真实模型并替换
    try {
      const gltf = await this.App.gltfLoader.loadAsync(url);
      if (this.removed) return this;
      const root = gltf.scene;
      this.vrm = gltf.userData.vrm || null;
      if (this.vrm) this._initVrm(root);
      this._fitModel(root);
      // 适配异常（包围盒/缩放不合理）则回退全息替身，避免穿模/埋地
      if (this._fitErr) {
        console.warn(`[赛博公司] 角色「${this.name}」模型适配异常，回退全息投影替身`);
        this.vrm = null;
        this.vrmBones = {};
        this.group.remove(root);
        this.group.add(holo);
        this._holoActive = true;
      } else {
        this.group.remove(holo);
        this.group.add(root);
        this._holoActive = false;
        // 记录脚踝相对 group 的固定高度（适配后正确站立时脚踝≈0.2，与 group.y 无关），
        // 作为每帧脚底锚点校准的基准：无论任何机制把骨架/模型压入地面，都强制拉回该高度
        if (this.vrm && this.footBone) {
          try {
            this.footBone.updateWorldMatrix(true, false);
            const fp = new this.THREE.Vector3();
            this.footBone.getWorldPosition(fp);
            this._footToGroup = fp.y - this.group.position.y;
            console.log(`[赛博公司] 「${this.name}」模型就位：group.y=${this.group.position.y.toFixed(3)} 脚踝相对高度=${this._footToGroup.toFixed(3)}`);
          } catch (e) { this._footToGroup = 0.2; }
        }
        // 直接落在标准地面高度（不再从 2m 高处落下：掉落过程会影响角色动作呈现）
        this.group.position.y = STAND_Y;
      }
    } catch (e) {
      console.warn(`[赛博公司] 角色「${this.name}」模型加载失败，保留全息投影替身:`, e?.message || e);
    }
    return this;
  }

  _initVrm(root) {
    const App = this.App;
    const VRMUtils = App.VRMUtils;
    const THREE = this.THREE;
    try { VRMUtils.rotateVRM0(this.vrm); } catch (e) {}
    try { VRMUtils.removeUnnecessaryVertices(root); } catch (e) {}

    // 弹簧骨骼调优（防头发衣物穿模）
    try {
      const mgr = this.vrm.springBoneManager;
      if (mgr) {
        const joints = mgr.joints || mgr._sortedJoints || Array.from(mgr._joints || []);
        for (const j of joints) {
          const s = j.settings;
          if (!s) continue;
          s.dragForce = Math.min(0.95, (s.dragForce ?? 0.4) * 2.5);
          s.stiffness = Math.min(0.4, (s.stiffness ?? 1) * 0.35);
          s.gravityPower = (s.gravityPower ?? 0) * 0.4;
        }
        // 无碰撞体时添加基础身体碰撞球（与大厅 calmSpringBones 一致），
        // 防止长裙/长发等 springBone 下垂时直接穿入地面造成"陷地"观感
        const colliderGroups = mgr.colliderGroups || mgr._colliderGroups || [];
        if (colliderGroups.length === 0) this._addBodyColliders(mgr);
      }
    } catch (e) {}

    // 缓存 humanoid 骨骼
    this.vrmBones = {};
    this.footBone = null;   // 脚踝骨骼（raw），用于每帧脚底锚点校准
    if (this.vrm.humanoid) {
      const h = this.vrm.humanoid;
      const want = ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
        'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm',
        'leftHand', 'rightHand', 'leftUpperLeg', 'rightUpperLeg',
        'leftLowerLeg', 'rightLowerLeg'];
      for (const n of want) {
        const b = h.getNormalizedBoneNode(n);
        if (b) this.vrmBones[n] = b;
      }
      // 脚踝骨骼（raw，参与真实渲染蒙皮）
      this.footBone = h.getRawBoneNode('leftFoot') || h.getRawBoneNode('rightFoot') || null;
    }

    // 表情映射
    this._mapExpressions();

    // 立即应用放松站姿（消除 T-pose）
    this._applyRestPose();
  }

  /** 为无碰撞体的模型添加基础身体碰撞球（防裙摆/长发 springBone 穿地） */
  _addBodyColliders(mgr) {
    const THREE = this.THREE;
    const h = this.vrm?.humanoid;
    if (!h) return;
    const defs = [
      { bone: 'head', offset: [0, 0, 0], radius: 0.09 },
      { bone: 'chest', offset: [0, 0, 0], radius: 0.12 },
      { bone: 'spine', offset: [0, 0, 0], radius: 0.10 },
      { bone: 'hips', offset: [0, 0, 0], radius: 0.12 },
      { bone: 'leftUpperArm', offset: [0, -0.1, 0], radius: 0.06 },
      { bone: 'rightUpperArm', offset: [0, -0.1, 0], radius: 0.06 },
    ];
    let added = 0;
    for (const def of defs) {
      const boneNode = h.getNormalizedBoneNode(def.bone);
      if (!boneNode) continue;
      try {
        const group = {
          colliders: [{ position: new THREE.Vector3(def.offset[0], def.offset[1], def.offset[2]), radius: def.radius }],
          bones: new Set(),
        };
        if (typeof mgr.addColliderGroup === 'function') {
          mgr.addColliderGroup(def.bone, group);
          added++;
        }
      } catch (e) { /* API 不兼容，跳过 */ }
    }
    if (added > 0) console.log(`[赛博公司] 角色「${this.name}」添加 ${added} 个身体碰撞球`);
  }

  _mapExpressions() {
    const exp = this.vrm?.expressionManager;
    this.exprNames = {};
    if (!exp) return;
    const map = exp.expressionMap || {};
    const find = (names) => {
      for (const n of names) { if (map[n]) return n; }
      return null;
    };
    this.exprNames.mouth = find(['aa', 'A', 'oh', 'O', 'ou', 'U', 'ih', 'I']);
    this.exprNames.blink = find(['blink', 'Blink', 'blinkLeft', 'BlinkL']);
    this.exprNames.blinkLeft = find(['blinkLeft', 'BlinkL', 'blink_L', 'winkLeft', 'WinkLeft']);
    this.exprNames.blinkRight = find(['blinkRight', 'BlinkR', 'blink_R', 'winkRight', 'WinkRight']);
    this.exprNames.happy = find(['happy', 'Joy', 'fun', 'Fun']);
    this.exprNames.surprised = find(['surprised', 'Surprised', 'shock', 'Shock', 'surprise', 'Surprise']);
    this.exprNames.sad = find(['sad', 'Sad', 'sorrow', 'Sorrow']);
    this.exprNames.thoughtful = find(['thoughtful', 'Thoughtful', 'think', 'Think']);
    this.exprNames.angry = find(['angry', 'Angry', 'mad', 'Mad']);
    this.exprNames.relaxed = find(['relaxed', 'Relaxed', 'calm', 'Calm', 'peaceful', 'Peaceful']);
  }

  _applyRestPose() {
    const B = this.vrmBones;
    if (B.leftUpperArm) B.leftUpperArm.rotation.set(0, 0, 1.35);
    if (B.rightUpperArm) B.rightUpperArm.rotation.set(0, 0, -1.35);
    if (B.leftLowerArm) B.leftLowerArm.rotation.set(-0.15, 0, 0);
    if (B.rightLowerArm) B.rightLowerArm.rotation.set(-0.15, 0, 0);
    if (B.leftHand) B.leftHand.rotation.set(0, 0, 0.1);
    if (B.rightHand) B.rightHand.rotation.set(0, 0, -0.1);
  }

  _fitModel(root) {
    const THREE = this.THREE;
    this._fitErr = false;
    try {
      // 从零开始适配，避免重复调用叠加
      root.position.set(0, 0, 0);
      root.scale.set(1, 1, 1);
      root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      if (!size.y || !isFinite(size.y) || size.y < 0.01) { this._fitErr = true; return; }
      // 统一标准站立身高（与大厅 applyLoadedModel 的 2.2 完全一致），
      // 避免比大厅角色矮一截而产生"沉下去/陷地"的视觉观感
      const scale = 2.2 / size.y;
      if (!isFinite(scale) || scale < 0.05 || scale > 50) { this._fitErr = true; return; }
      root.scale.setScalar(scale);
      root.updateMatrixWorld(true);
      const box2 = new THREE.Box3().setFromObject(root);
      if (box2.isEmpty() || !isFinite(box2.min.y)) { this._fitErr = true; return; }
      const c2 = box2.getCenter(new THREE.Vector3());
      root.position.x -= c2.x;
      root.position.z -= c2.z;
      root.position.y -= box2.min.y;   // 脚踩地面
    } catch (e) {
      console.warn('[赛博公司] 模型适配失败:', e?.message || e);
      this._fitErr = true;
    }
  }

  /** 移动到指定世界坐标（支持多段路径） */
  moveTo(x, z, onArrive = null) {
    this.waypoints = [{ x, z }];
    this._onArrive = onArrive;
  }

  moveAlong(points, onArrive = null) {
    this.waypoints = (points || []).map(p => ({ x: p.x, z: p.z }));
    this._onArrive = onArrive;
  }

  setPosition(x, z) {
    if (!this.group) return;
    this.group.position.x = x;
    this.group.position.z = z;
    this.group.position.y = STAND_Y;
    this.waypoints = null;
  }

  setEmotion(key) {
    this._emotionKey = key || null;
  }

  /** 说话状态开关（驱动口型开合动画） */
  setSpeaking(v) {
    this.isSpeaking = !!v;
  }

  // ==================== 头顶对话气泡 ====================

  /**
   * 在角色头顶显示对话气泡（说话时调用）。
   * @param {string} text  台词
   * @param {string} color 霓虹主题色（HR 青 #00e5ff / 求职者 粉 #ff2bd6）
   */
  showBubble(text, color = '#00e5ff') {
    if (!this.group || this.removed) return;
    const txt = String(text || '').trim();
    if (!txt) return;
    const THREE = this.THREE;

    this._drawBubbleCanvas(txt, color);
    if (!this.bubble) {
      this.bubble = new THREE.Sprite(new THREE.SpriteMaterial({
        map: this._bubbleTex,
        transparent: true,
        depthWrite: false,
        opacity: 0,
      }));
      this.bubble.renderOrder = 999;
      this.group.add(this.bubble);
    } else {
      this.bubble.material.map = this._bubbleTex;
      this.bubble.material.opacity = 0;
      this.bubble.material.needsUpdate = true;
    }
    // 世界尺寸：固定宽度，高度按画布宽高比等比缩放；
    // 底部锚点固定在头顶上方，气泡越高则向上生长（更自然）
    const asp = this._bubbleCanvas.height / this._bubbleCanvas.width;
    this._bubbleW = 3.6;
    this._bubbleH = Math.min(1.3, this._bubbleW * asp);
    this.bubble.position.y = BUBBLE_BASE_Y + this._bubbleH * 0.5;
    this._bubbleVisible = true;
    this._bubblePop = -0.5;                         // 负值=预热阶段：等 CanvasTexture 上传完成再淡入，防止旧文字闪现
    this._bubbleOpacity = 0;                        // 干净重开，防止旧淡出残留
    this._bubbleShowT = this.game.elapsedTime;      // 超时保险基准（秒）
    if (this.bubble) this.bubble.visible = true;    // 防御：立即可见（预热期 opacity=0，_updateBubble 控制淡入）
  }

  /** 隐藏对话气泡（说话结束时调用，自动淡出） */
  hideBubble() {
    this._bubbleVisible = false;
  }

  /** 在离屏画布上绘制赛博风气泡：圆角深色底 + 霓虹描边 + 指向说话者的尾巴 + 自动换行 */
  _drawBubbleCanvas(text, color) {
    const c = this._bubbleCanvas || (this._bubbleCanvas = document.createElement('canvas'));
    const ctx = this._bubbleCtx || (this._bubbleCtx = c.getContext('2d'));
    const W = 1024;

    // 自动换行 + 字号自适应：最多 4 行（超长文本截断省略）
    const wrap = (font) => {
      ctx.font = `bold ${font}px "Microsoft YaHei", sans-serif`;
      const maxW = W - 140;
      const ls = [];
      let cur = '';
      for (const ch of String(text)) {
        if (ch === '\n') { if (cur) ls.push(cur); cur = ''; continue; }
        if (ctx.measureText(cur + ch).width > maxW && cur) { ls.push(cur); cur = ch; }
        else cur += ch;
      }
      if (cur) ls.push(cur);
      return ls;
    };
    let font = 56;
    let lines = wrap(font);
    while (lines.length > 4 && font > 34) {
      font -= 3;
      lines = wrap(font);
    }
    if (lines.length > 4) {
      lines = lines.slice(0, 4);
      lines[3] = lines[3].slice(0, Math.max(1, lines[3].length - 1)) + '…';
    }

    const lineH = Math.ceil(font * 1.34);
    const padY = 28;
    const tailH = 40;                              // 指向说话者头部的三角形尾巴
    const H = padY * 2 + lines.length * lineH + tailH;
    c.width = W; c.height = H;
    ctx.clearRect(0, 0, W, H);

    // 气泡主体（圆角矩形）
    const r = 46;
    const bx = 4, by = 4;
    const bw = W - bx * 2;
    const bh = H - tailH - by;
    const body = () => {
      ctx.beginPath();
      ctx.moveTo(bx + r, by);
      ctx.arcTo(bx + bw, by, bx + bw, by + bh, r);
      ctx.arcTo(bx + bw, by + bh, bx, by + bh, r);
      ctx.arcTo(bx, by + bh, bx, by, r);
      ctx.arcTo(bx, by, bx + bw, by, r);
      ctx.closePath();
    };

    // 霓虹描边辉光
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 26;
    body();
    ctx.fillStyle = 'rgba(8, 12, 30, 0.92)';
    ctx.fill();
    ctx.lineWidth = 9;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.restore();

    // 尾巴（底部居中，尖端指向下方说话者）
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 20;
    ctx.beginPath();
    ctx.moveTo(W / 2 - 46, bh + 2);
    ctx.lineTo(W / 2 + 46, bh + 2);
    ctx.lineTo(W / 2, H - 2);
    ctx.closePath();
    ctx.fillStyle = 'rgba(8, 12, 30, 0.92)';
    ctx.fill();
    ctx.lineWidth = 8;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.restore();

    // 台词文本
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#f2f6ff';
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.font = `bold ${font}px "Microsoft YaHei", sans-serif`;
    const totalTextH = lines.length * lineH;
    let ty = padY + lineH * 0.5 + (bh - padY * 2 - totalTextH) * 0.5;
    for (const ln of lines) {
      ctx.fillText(ln, W / 2, ty);
      ty += lineH;
    }
    ctx.restore();

    if (!this._bubbleTex) {
      this._bubbleTex = new this.THREE.CanvasTexture(c);
    } else {
      this._bubbleTex.image = c;
      this._bubbleTex.needsUpdate = true;
    }
  }

  /** 每帧更新气泡动画：弹出 / 漂浮 / 淡出 */
  _updateBubble(dt, t) {
    if (!this.bubble) return;
    const mat = this.bubble.material;
    // 超时保险：若说话流程异常中断导致未调用 hideBubble，15s 后强制隐藏，杜绝气泡残留
    if (this._bubbleVisible && this._bubbleShowT != null && t - this._bubbleShowT > 15) {
      this._bubbleVisible = false;
    }
    if (!this._bubbleVisible) {
      this._bubbleOpacity = Math.max(0, this._bubbleOpacity - dt * 3.2);
      mat.opacity = this._bubbleOpacity;
      if (this._bubbleOpacity <= 0) this.bubble.visible = false;
      return;
    }
    // 预热阶段：等 CanvasTexture 上传完成（约 0.08s）再淡入，
    // 避免新台词瞬间仍显示上一段话的旧纹理（语音/字幕不同步感）
    if (this._bubblePop < 0) {
      this._bubblePop += dt * 6;
      this.bubble.visible = true;
      this.bubble.scale.set(this._bubbleW * 0.3, this._bubbleH * 0.3, 1);
      mat.opacity = 0;
      return;
    }
    this.bubble.visible = true;
    this._bubblePop = Math.min(1, this._bubblePop + dt * 5);
    const e = 1 - Math.pow(1 - this._bubblePop, 3);   // easeOutCubic 弹出
    const s = 0.55 + 0.45 * e;
    this.bubble.scale.set(this._bubbleW * s, this._bubbleH * s, 1);
    this._bubbleOpacity = Math.min(1, this._bubbleOpacity + dt * 7);
    mat.opacity = this._bubbleOpacity;
    // 轻微漂浮（带个人节奏相位）
    this.bubble.position.y = BUBBLE_BASE_Y + this._bubbleH * 0.5 + Math.sin(t * 2.2 + this._phase) * 0.05;
  }

  /** 每帧更新：移动 / 朝向 / 骨骼动画 / 表情 */
  update(dt) {
    if (!this.group || this.removed) return;
    const t = this.game.elapsedTime;

    // ---- 移动 ----
    if (this.waypoints && this.waypoints.length) {
      const wp = this.waypoints[0];
      const dx = wp.x - this.group.position.x;
      const dz = wp.z - this.group.position.z;
      const dist = Math.hypot(dx, dz);
      const step = this.speed * dt;
      if (dist <= step) {
        this.group.position.x = wp.x;
        this.group.position.z = wp.z;
        this.waypoints.shift();
        if (!this.waypoints.length) {
          this.isMoving = false;
          this._walkPhase = 0;
          this._lastStepKey = -1;   // 复位脚步相位，下次起步第一步必出声
          const cb = this._onArrive;
          this._onArrive = null;
          if (cb) cb();
        }
      } else {
        const nx = dx / dist, nz = dz / dist;
        this.group.position.x += nx * step;
        this.group.position.z += nz * step;
        this.isMoving = true;
        this._walkPhase += dt * 7;
        // 脚步音效：与玩家一致（相位驱动，每半步 π 触发一步；真实模型才出声，
        // 全息替身悬浮移动不匹配脚步节奏）
        if (this.vrm && this.App && this.App.playFootstep) {
          const key = Math.floor(this._walkPhase / Math.PI);
          if (key !== this._lastStepKey) {
            this._lastStepKey = key;
            const s = this.speed || 2.4;
            this.App.playFootstep(0.40 + Math.min(0.12, (s / 3.0) * 0.05));
          }
        }
        this.group.rotation.y = this._lerpAngle(this.group.rotation.y, Math.atan2(nx, nz), Math.min(1, 8 * dt));
      }
    }

    // ---- 落地：真实模型加载后从 2m 高处落下，落到 STAND_Y 即停（观察落地高度）----
    if (this._falling) {
      this.group.position.y = Math.max(STAND_Y, this.group.position.y - dt * 2.6);
      if (this.group.position.y <= STAND_Y + 0.001) this._falling = false;
    }

    // ---- 朝向目标 ----
    if (!this.isMoving && this.faceTarget && this.group) {
      const dx = this.faceTarget.x - this.group.position.x;
      const dz = this.faceTarget.z - this.group.position.z;
      if (Math.hypot(dx, dz) > 0.05) {
        this.group.rotation.y = this._lerpAngle(this.group.rotation.y, Math.atan2(dx, dz), Math.min(1, 5 * dt));
      }
    }

    // ---- VRM 骨骼 + 表情 ----
    if (this.vrm) {
      this._updateActorMotion(dt);
      this._animateBones(dt, t);
      this._updateExpressions(dt, t);
      this.vrm.update(dt);
      // 脚底锚点校准：站立时若任何机制把角色压入地面，强制拉回正确高度（走动时跳过，
      // 避免与腿部摆动画冲突；下落中跳过由落地逻辑接管）。阈值 2cm 容忍呼吸/微摆，只纠正真实下沉。
      if (!this.isMoving && !this._falling && this.footBone && this._footToGroup !== undefined) {
        try {
          this.footBone.updateWorldMatrix(true, false);
          const fy = this._tmpFootV3 || (this._tmpFootV3 = new this.THREE.Vector3());
          this.footBone.getWorldPosition(fy);
          const delta = (STAND_Y + this._footToGroup) - fy.y;
          if (Math.abs(delta) > 0.02) {
            this.group.position.y += delta;
            this._anchorTotal = (this._anchorTotal || 0) + Math.abs(delta);
            if ((this._anchorLogN = (this._anchorLogN || 0) + 1) % 90 === 1) {
              console.warn(`[赛博公司] 「${this.name}」被压入地面 ${delta.toFixed(3)}m，已强制拉回（累计 ${this._anchorTotal.toFixed(2)}m）`);
            }
          }
        } catch (e) {}
      }
    } else if (this._holoActive) {
      // 全息替身：轻微漂浮动画
      const holo = this.group.children[0];
      if (holo && holo.isGroup) {
        // 整体轻微左右摇摆（没有骨骼，用组旋转模拟"有生命感"）
        holo.rotation.y = Math.sin(t * 0.9 + this._phase) * 0.14;
        holo.children.forEach(m => {
          if (!m.isMesh) return;
          if (!m.userData._baseY) m.userData._baseY = m.position.y;
          m.position.y = m.userData._baseY + Math.sin(t * 2 + this._phase) * 0.08;
        });
      }
    }

    // ---- 头顶对话气泡 ----
    this._updateBubble(dt, t);
  }

  _animateBones(dt, t) {
    const B = this.vrmBones;
    if (!B) return;
    const moving = this.isMoving;
    const ARM = 1.35;
    const lSwing = moving ? Math.sin(this._walkPhase) * 0.25 : Math.sin(t * 1.1 + this._phase) * 0.008;
    const rSwing = moving ? -Math.sin(this._walkPhase) * 0.25 : -Math.sin(t * 1.1 + this._phase) * 0.008;
    if (B.leftUpperArm) B.leftUpperArm.rotation.set(0, 0, ARM + lSwing);
    if (B.rightUpperArm) B.rightUpperArm.rotation.set(0, 0, -ARM + rSwing);
    if (B.leftLowerArm) B.leftLowerArm.rotation.set(-0.15 + (moving ? Math.sin(this._walkPhase + 0.7) * 0.1 : 0), 0, 0);
    if (B.rightLowerArm) B.rightLowerArm.rotation.set(-0.15 + (moving ? -Math.sin(this._walkPhase + 0.7) * 0.1 : 0), 0, 0);
    if (B.leftHand) B.leftHand.rotation.set(0, 0, 0.1);
    if (B.rightHand) B.rightHand.rotation.set(0, 0, -0.1);
    if (B.leftUpperLeg) B.leftUpperLeg.rotation.set(moving ? Math.sin(this._walkPhase) * 0.3 : 0, 0, 0);
    if (B.rightUpperLeg) B.rightUpperLeg.rotation.set(moving ? -Math.sin(this._walkPhase) * 0.3 : 0, 0, 0);
    if (B.leftLowerLeg) B.leftLowerLeg.rotation.set(moving ? Math.max(0, Math.sin(this._walkPhase)) * 0.15 : 0, 0, 0);
    if (B.rightLowerLeg) B.rightLowerLeg.rotation.set(moving ? Math.max(0, -Math.sin(this._walkPhase)) * 0.15 : 0, 0, 0);
    // 呼吸
    const breathe = Math.sin(t * 2.1 + this._phase) * 0.012;
    if (B.chest) B.chest.rotation.x = breathe;
    if (B.spine) B.spine.rotation.x = breathe * 0.5;
    if (B.hips) B.hips.position.y = moving ? Math.abs(Math.sin(this._walkPhase)) * 0.04 : breathe * 0.6;
    // 站立时极轻微重心摆动（仅髋部，不带动躯干侧倾，避免东倒西歪）
    if (!moving) {
      const sway = Math.sin(t * 1.3 + this._phase) * 0.005;
      if (B.hips) B.hips.rotation.z += sway;
    }
    // 头部微转向目标
    if (B.head) {
      if (this.faceTarget && !moving) {
        const dx = this.faceTarget.x - this.group.position.x;
        const dz = this.faceTarget.z - this.group.position.z;
        const ang = Math.atan2(dx, dz);
        let dy = ang - this.group.rotation.y;
        while (dy > Math.PI) dy -= Math.PI * 2;
        while (dy < -Math.PI) dy += Math.PI * 2;
        const yaw = Math.max(-0.5, Math.min(0.5, dy));
        B.head.rotation.y = this._lerpAngle(B.head.rotation.y, yaw, Math.min(1, 6 * dt));
        if (B.neck) B.neck.rotation.y = yaw * 0.4;
      } else {
        B.head.rotation.y = Math.sin(t * 0.6 + this._phase) * 0.025;
      }
      B.head.rotation.x = 0;
    }
    // 大厅式微动作偏移：叠加在基础动画之上（动作结束后平滑回中）
    this._applyMotionOffsets();
    // 头部/颈部总旋转硬钳制（与大厅一致）：姿态/朝向/微动作叠加后也不能过度弯曲
    if (B.head) {
      B.head.rotation.y = Math.max(-0.35, Math.min(0.35, B.head.rotation.y));
      B.head.rotation.x = Math.max(-0.22, Math.min(0.20, B.head.rotation.x));
      B.head.rotation.z = Math.max(-0.12, Math.min(0.12, B.head.rotation.z));
    }
    if (B.neck) {
      B.neck.rotation.y = Math.max(-0.18, Math.min(0.18, B.neck.rotation.y));
      B.neck.rotation.x = Math.max(-0.18, Math.min(0.18, B.neck.rotation.x));
    }
  }

  /** 播放一个微动作（与大厅表情动作引擎同构：包络 + 序列 + 平滑混合） */
  _playActorMotion(name) {
    const def = ACTOR_MOTIONS[name];
    if (!def || !this.vrm || this.removed) return;
    this._motionActive = true;
    this._motionName = name;
    this._motionDef = def;
    this._motionElapsed = 0;
    this._motionIdleT = 0;
    this._motionIdleInterval = 12 + Math.random() * 10;
  }

  /** 每帧推进微动作：空闲计时 → 随机触发；活动动作按包络计算目标偏移 */
  _updateActorMotion(dt) {
    if (!this.vrm) return;
    // 走路/说话时打断微动作（回到基础动画），动作结束自动平滑回中
    if (this.isMoving || this.isSpeaking) {
      this._motionActive = false;
      this._motionIdleT = 0;
    } else {
      this._motionIdleT += dt;
      if (this._motionIdleT >= this._motionIdleInterval) {
        this._motionIdleT = 0;
        if (!this._motionActive) {
          this._playActorMotion(this._pickIdleMotion());
        }
      }
    }
    const targets = this._motionTargets(dt);
    // 平滑插值：动作峰值 0.09/帧 优雅跟上，结束回中 0.05/帧 柔和复位（更接近大厅的慢速过渡）
    const k = this._motionActive ? 0.09 : 0.05;
    // 先衰减当前帧没有目标的骨骼（动作被走路/说话打断时也能平滑回中）
    for (const bone in this._motionCur) {
      if (targets[bone]) continue;
      const cur = this._motionCur[bone];
      cur.x -= cur.x * k;
      cur.y -= cur.y * k;
      cur.z -= cur.z * k;
      if (Math.abs(cur.x) < 0.001 && Math.abs(cur.y) < 0.001 && Math.abs(cur.z) < 0.001) {
        delete this._motionCur[bone];
      }
    }
    for (const bone in targets) {
      const tgt = targets[bone];
      const cur = this._motionCur[bone] || (this._motionCur[bone] = { x: 0, y: 0, z: 0 });
      cur.x += (tgt.x - cur.x) * k;
      cur.y += (tgt.y - cur.y) * k;
      cur.z += (tgt.z - cur.z) * k;
    }
  }

  /** 计算当前动作各骨骼通道的目标偏移（无动作 → 全部归零） */
  _motionTargets(dt) {
    const out = {};
    if (!this._motionActive || !this._motionDef) return out;
    const def = this._motionDef;
    this._motionElapsed += dt * 0.6;   // 大厅同款 0.6 倍慢速时间轴：动作更舒缓
    const dur = def.dur || 1.6;
    if (this._motionElapsed >= dur) {
      this._motionActive = false;
      this._motionName = '';
      this._motionDef = null;
      return out;   // 结束：目标归零，下一帧开始平滑回中
    }
    const p = this._motionElapsed / dur;
    const atk = 0.30, rel = 0.42;
    const hold = def.hold || 0;
    for (const bone in def) {
      if (bone === 'dur' || bone === 'hold') continue;
      const chan = def[bone];
      if (chan == null) continue;
      const off = out[bone] = { x: 0, y: 0, z: 0 };
      for (const axis of ['x', 'y', 'z']) {
        if (chan[axis] == null) continue;
        off[axis] = this._evalMotionChannel(chan[axis], p, atk, hold, rel);
      }
    }
    return out;
  }

  /** 单通道包络：数值 = 脉冲；{amp, loops} = 振荡 */
  _evalMotionChannel(chan, p, atk, hold, rel) {
    if (typeof chan === 'number') {
      const env = this._motionPulse(p, atk, hold, rel);
      return chan * env * ACTOR_MOTION_SCALE;
    }
    const amp = chan.amp != null ? chan.amp : 0.15;
    const loops = chan.loops != null ? chan.loops : 1;
    const env = this._motionPulse(p, 0.18, 0, 0.30);
    return Math.sin(p * loops * Math.PI * 2) * amp * env * ACTOR_MOTION_SCALE;
  }

  _motionPulse(p, atk, hold, rel) {
    const smooth01 = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
    if (p < atk) return smooth01(p / atk);
    const plateau = atk + hold;
    if (p < plateau) return 1;
    const relEnd = plateau + rel;
    if (p < relEnd) return 1 - smooth01((p - plateau) / rel);
    return 0;
  }

  /** 把平滑偏移叠加到骨骼当前旋转（在基础走姿/呼吸之后调用） */
  _applyMotionOffsets() {
    const B = this.vrmBones;
    if (!B) return;
    for (const bone in this._motionCur) {
      const off = this._motionCur[bone];
      const b = B[bone];
      if (!b || !b.rotation) continue;
      // 单通道硬上限（与大厅 clampOffsets 同思路）：头部/颈部更紧，其余 ≤0.5 rad
      const lim = (bone === 'head' || bone === 'neck') ? 0.22 : 0.5;
      b.rotation.x += Math.max(-lim, Math.min(lim, off.x || 0));
      b.rotation.y += Math.max(-lim, Math.min(lim, off.y || 0));
      b.rotation.z += Math.max(-lim, Math.min(lim, off.z || 0));
    }
  }

  /** 空闲微动作候选池（按岗位/场合加权，避免呆板） */
  _pickIdleMotion() {
    // 空闲池只有轻微头部动作：转头/点头/轻歪头/看上下；不做任何大幅肢体动作
    const pool = ['glance_around', 'glance_around', 'head_turn_l', 'head_turn_r', 'nod_small',
      'head_tilt_l', 'head_tilt_r', 'look_up', 'look_down'];
    return pool[Math.floor(Math.random() * pool.length)];
  }

  _updateExpressions(dt, t) {
    const exp = this.vrm?.expressionManager;
    if (!exp) return;
    // 眨眼
    this._blinkT -= dt;
    if (this._blinkT <= 0 && !this._blinking) this._blinking = true;
    if (this._blinking) {
      this._blinkPhase += dt / 0.16;
      const v = Math.abs(Math.cos(this._blinkPhase * Math.PI));
      if (this.exprNames.blink) exp.setValue(this.exprNames.blink, v);
      if (this._blinkPhase >= 1) {
        this._blinking = false;
        this._blinkT = 1.5 + Math.random() * 3.5;
        if (this.exprNames.blink) exp.setValue(this.exprNames.blink, 0);
      }
    }
    // 嘴型（说话时开合，闭嘴时归零）
    if (this.exprNames.mouth) {
      const mouth = this.isSpeaking ? 0.22 + 0.2 * Math.abs(Math.sin(t * 9 + this._phase)) : 0;
      exp.setValue(this.exprNames.mouth, mouth);
    }
    // 情绪表情（说话时不叠加嘴部型情绪，避免口型冲突）
    for (const k in this.exprNames) {
      if (!this.exprNames[k]) continue;
      if (k === 'mouth' || k === 'blink' || k === 'blinkLeft' || k === 'blinkRight') continue;
      const target = (k === this._emotionKey && !this.isSpeaking) ? 0.9 : 0;
      this._exprCur[k] = this._exprCur[k] || 0;
      this._exprCur[k] += (target - this._exprCur[k]) * Math.min(1, 6 * dt);
      exp.setValue(this.exprNames[k], this._exprCur[k]);
    }
  }

  _lerpAngle(a, b, t) {
    let diff = b - a;
    while (diff > Math.PI) diff -= Math.PI * 2;
    while (diff < -Math.PI) diff += Math.PI * 2;
    return a + diff * t;
  }

  /** 销毁：从场景移除并释放资源 */
  dispose() {
    this.removed = true;
    if (!this.group) return;
    // 释放对话气泡
    if (this.bubble) {
      try {
        if (this.bubble.material) {
          if (this.bubble.material.map) this.bubble.material.map.dispose();
          this.bubble.material.dispose();
        }
      } catch (e) {}
      if (this.bubble.parent) this.bubble.parent.remove(this.bubble);
      this.bubble = null;
    }
    this.group.traverse(obj => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
    });
    if (this.group.parent) this.group.parent.remove(this.group);
    this.vrm = null;
    this.vrmBones = {};
  }
}

/* ============================================================
 * CyberCorpGame
 * ============================================================ */
export class CyberCorpGame extends BaseGame {
  constructor(app) {
    super(app);
    // 注意：这里不能直接访问 app.THREE —— GameModeManager.getAvailableGames()
    // 会用 factory(null) 创建临时实例读取元信息，BaseGame 构造函数已对 null 做了保护
    this.name = 'cyber_corp';
    this.displayName = '赛博公司';
    this.description = '在霓虹公司就任 CEO，创建独立世界；任命 HR 主持面试，AI 求职者轮番应聘；员工智能体拥有独立 RL 策略，自主决策推进任务、互相协作，所有员工都知道你的身份并主动向你汇报。';
    this.moveSpeed = 4.0;
    this.initialCameraRadius = 11;
    this.initialCameraHeight = 8.2;         // 相机略高略远：开局即可看清整间办公室与各独立办公间
    this.boundarySize = 40;                 // 玩家可活动范围
    this.uiHint = '🎮 你是 CEO（幽灵无实体）：WASD 移动 · 靠近员工说话对方会回应 · 「开会/汇报/散会」触发全员会议 · 说「布置任务：做一个恋爱游戏」蜂群会自动拆解分工 · 员工会主动找同事协作、讨论/否决/提建议 · 靠近员工可看对话记录';

    // 运行状态
    this.phase = 'init';                    // init | hr_select | running | completed
    this.cards = [];
    this.hrCard = null;
    this.hrActor = null;
    this.actors = [];                       // 所有游戏角色
    this.seekerQueue = [];                  // 待面试候选人
    this.currentSeeker = null;
    this.hiredList = [];                    // { position, actor, name }
    this.filledPositions = new Set();
    this._disposed = false;
    this._running = false;
    this._timers = [];
    this._audios = [];
    this._blockers = [];
    this._particles = null;
    this._particlePos = null;
    this._holoCanvas = null;
    this._holoTexture = null;
    this._holoMesh = null;
    this._holoPlaceholder = false;
    this._transcript = [];                  // 面试记录（公开透明）

    // 公司评分系统（四维加权：活力40 + 质量30 + 稳定15 + 声誉10 + 产出5）
    this.score = 0;                         // 兼容旧存档：公司总分（0-100）
    this.stats = {
      vitality: 0,                          // 组织活力（微笑曲线 - 僵化惩罚）0-40
      quality: 0,                           // 人才质量 0-30
      stability: 0,                         // 运营稳定 0-15
      reputation: BALANCE.REPUTATION_FULL,  // 公司声誉（事件修正）0-10
      production: 0,                        // 公司产出（RL 任务系统驱动）0-5
      score: 0,                             // 总分 0-100
      grade: 'E',                           // S/A/B/C/D/E
      layoffCount: 0,                       // 当期已裁员人数（每期结算后清零）
      stalePeriods: 0,                      // 连续不裁员期数（≥3 起僵化惩罚）
    };

    // UI 引用
    this._uiRoot = null;
    this._uiTrans = null;
    this._uiCandidate = null;
    this._uiResult = null;
    this._uiStatus = null;
    this._uiCandWrap = null;
    this._candVisible = false;
    this._statusTimer = null;
    this._uiPositions = null;
    this._styleEl = null;

    this._speakSeq = 0;                     // 台词序号（防异步乱序）

    // 幽灵玩家系统（无实体第一人称）：玩家完全独立于大厅角色模型
    // fpvCamera 由 game-mode-manager 读取：相机 = 幽灵眼睛位置（拖拽转动视角）
    this.fpvCamera = true;
    this.fpvEyeHeight = 1.55;               // 幽灵眼睛高度（米）
    this.playerAnchor = null;               // 无实体锚点（manager 移动/相机锚点，替代大厅角色）
    this._ghostAnchor = null;

    // 员工独立对话记录（世界隔离持久化 + 靠近面板）
    this._agentChat = {};                   // 员工名 → [{role:'user'|'agent', text, t}]
    this._chatPanelEl = null;               // 靠近对话面板 DOM
    this._nearAgentName = null;             // 当前面板显示的员工
    this._chatCooldown = {};                // 员工名 → 上次回应时间（防刷屏）
    this._chatDebounce = {};                // 员工名 → 对话记录防抖（独立于回应冷却）

    // RL 驱动自主决策（每员工一个独立策略）+ 公司任务看板
    this._tasks = [];                       // 任务看板（每岗位一项）
    this._agentTask = {};                   // 员工名 → 任务实例（专属任务）
    this._agentEnergy = {};                 // 员工名 → 精力 0-100
    this._rl = {};                          // 员工名 → SwarmRLAgent 实例
    this._rlLearnCount = 0;                 // 全局学习计数（用于节流保存）

    // 玩家独立成"幽灵"后不再操控 HR
    this._playerControlledHR = false;

    // 存档与消息 hook 的运行时状态（真正保存/解绑在 cleanup() 中进行）
    this._savedAvatarVisible = undefined;
    this._resumeInterviewNotice = null;

    // 近距离语音（10 米内能听到）
    this._chatCooldown = {};                // 角色名 → 最近回应时间戳（防刷屏）
    this._origAddUserMsg = null;            // 被 hook 的 App.addUserMsg 原函数
    this._chatHookActive = false;

    // 存档恢复
    this._skipWelcome = false;              // 存档恢复后跳过 HR 开场白
    this._stoppedHiring = false;            // 员工满 6 人后停招标记
    this._gameLoopRunning = false;          // 招聘主循环运行中（防重复启动）

    // ===== 蜂群引擎：世界身份与蜂群运行时状态（R1-R4） =====
    this.ceo = null;                        // { ceo_id, ceo_name, company_name, motto }
    this.worldId = null;                    // 世界 ID（world_id = f(ceo 身份)）
    this._ceoUI = null;                     // CEO 就任表单引用
    this._worldTick = 0;                    // 世界时钟（秒）
    this._hudT = 0;                         // HUD 刷新节流计时
    this._swarmRunning = false;             // 蜂群决策循环运行中（防重复启动）
    this._swarmDecisions = 0;               // 蜂群累计决策数
    this._swarmCooldown = {};               // 员工名 → 下次可行动时间戳
    this._swarmCollabCount = 0;             // 蜂群累计协作事件数
    this._meetingActive = false;            // CEO 指令：全员会议进行中
    this._reportActive = false;             // CEO 指令：全员汇报进行中
    this._discussionActive = false;         // 蜂群讨论进行中（规划会/评审/提案，蜂群让路）
    this._playerTask = null;                // CEO 布置的项目任务（玩家任务 → 蜂群拆解）
    this._reviewPending = {};               // 员工名 → 待触发的交付评审
    this._forceRefine = {};                 // 员工名 → 被评审否决后强制打磨一次
    this._lastReviewAt = {};                // 员工名 → 上次评审时间戳
    this._agentProgress = {};               // 员工名 → 上次任务进度（停滞检测）
    this._stuckStreak = {};                 // 员工名 → 连续无进展次数
    this._nextErrandAt = {};                // 员工名 → 下次"跑腿"走动时间戳（机动性）
    this._meetingSpots = {};                // 员工名 → 开会聚集站位
    this._nextReportAt = {};                // 员工名 → 下次主动向 CEO 汇报时间戳
    this._nextSyncAt = 0;                   // 下次自动"进度碰头会"时间戳（蜂群主动形成讨论）
    this._swarmBusy = false;                // 长任务（讨论/汇报/协作/跑腿）进行中，看门狗不误判
    this._swarmTickT = 2;                   // 蜂群互动节拍器：距下一拍的时间（秒）
    this._swarmTickInterval = 4;            // 一拍间隔（3-6 秒随机，保证持续有主动互动）
    this._swarmCycle = 0;                   // 互动轮转序号（走动/协作/讨论/工作循环）
    this._uiSwarmList = null;               // "蜂群动态"信息流 DOM
  }

  // ==================== 生命周期 ====================

  generateScene() {
    const THREE = this.THREE;
    this.elapsedTime = 0;

    // 幽灵玩家系统：创建无实体锚点（独立于大厅角色模型）
    // manager 的移动/相机以 playerAnchor 优先，大厅角色保持隐藏、不漫步、不参与
    this._ghostAnchor = new THREE.Object3D();
    this._ghostAnchor.position.set(LAYOUT.playerSpawn.x, STAND_Y, LAYOUT.playerSpawn.z);
    if (this.App.scene) this.App.scene.add(this._ghostAnchor);
    this.playerAnchor = this._ghostAnchor;
    // WebXR VR 状态（每次进入游戏全新初始化，避免上一局残留影响）
    this._xrActive = false;       // 游戏侧 VR 会话标记
    this._xrWrap = null;          // VR 世界包裹组
    this._xrTurnYaw = 0;          // VR 世界转向（rad）
    // 幽灵玩家跳跃物理状态
    this._playerVelocityY = 0;
    this._isGrounded = true;
    this._airJumps = 0;

    // 赛博公司 = 完全独立系统：与大厅彻底隔离（不传导角色模型、不感知大厅、不共享状态）
    this.isIsolated = true;

    // 大厅初始角色：完全隔离出游戏场景（不是仅隐藏 —— 仅设 visible=false 时，
    // 模型异步加载完成后 currentAvatar 会重新出现，像幽灵一样站在办公室里）。
    // 从场景彻底移除 + 每帧兜底隔离（见 update/_isolateLobbyAvatar），退出时恢复。
    this._lobbyAvatarRemoved = false;
    this._isolateLobbyAvatar();
    this.App._aiDrivenWalk = false;         // 玩家绝不随机漫步（大厅自主行为关闭）
    // 进入游戏即复位大厅状态徽章：避免上次会话残留"思考中"被带入本局
    if (this.App.setState && this.App.State) {
      try { this.App.setState(this.App.State.IDLE); } catch (e) {}
    }

    this._buildOffice();
    this._buildStage();
    this._buildWorkstations();
    this._buildEntrance();
    this._buildDecor();
    this._buildHoloScreen();
    this._addLighting();
    this._addParticles();
    this._addColliders();

    // 蜂群引擎：先走世界身份流程（R2/R3）——就任 CEO 或恢复世界，再任命 HR
    this.phase = 'init';
    this._startCEOFlow();
  }

  onStart() {
    super.onStart();
    this._injectStyles();
    this._ensureVRBtn();   // 独立 VR 按钮：游戏启动即创建，不依赖 UI 重建
  }

  update(dt) {
    if (this.state !== 'playing') return;
    super.update(dt);
    // 每帧兜底：VR 按钮若被意外移除则立即重建（保证重进游戏后始终可见）
    if (!this._vrBtnEl || !this._vrBtnEl.parentNode) this._ensureVRBtn();
    // 幽灵玩家系统：大厅角色完全隐藏（不显示、不漫步、不参与），
    // 玩家位置由 manager 驱动在 playerAnchor（无实体锚点）上
    this._isolateLobbyAvatar();   // 每帧兜底：确保大厅角色被移出场景（防异步加载后重现）
    // 幽灵玩家跳跃物理：重力 + 落地（办公室地面 y=STAND_Y），支撑空格起跳
    if (this._ghostAnchor && this._ghostAnchor.position) {
      if (!this._isGrounded) {
        this._playerVelocityY -= JUMP_GRAVITY * dt;
        if (this._playerVelocityY < -MAX_FALL_SPEED) this._playerVelocityY = -MAX_FALL_SPEED;
        this._ghostAnchor.position.y += this._playerVelocityY * dt;
        // 着地检测
        if (this._ghostAnchor.position.y <= STAND_Y) {
          this._ghostAnchor.position.y = STAND_Y;
          this._playerVelocityY = 0;
          this._isGrounded = true;
          this._airJumps = 0;
        }
      }
    }
    for (const a of this.actors) {
      if (a && !a.removed) a.update(dt);
    }
    // 世界时钟推进 + 蜂群 HUD / 靠近对话面板节流刷新（0.5s 一次，避免每帧写 DOM）
    this._worldTick += dt;
    this._hudT += dt;
    if (this._hudT >= 0.5) {
      this._hudT = 0;
      this._updateSwarmHUD();
      this._updateNearbyChat();
      // 运营面板打开时实时刷新评分 / 任务看板进度
      if (this._opPanel) this._renderOperationPanel();
      this._refreshHoloIfStale();
    }
    // 蜂群互动节拍器：由主循环每帧驱动（不依赖异步 while 循环，绝不"假死"），
    // 每 3-6 秒调度一次主动互动：主动汇报 / 走动 / 协作咨询 / 碰头会 / 工作决策
    this._swarmTickT += dt;
    if (this._swarmTickT >= this._swarmTickInterval) {
      this._swarmTickT = 0;
      this._swarmTickInterval = 3 + Math.random() * 3;
      this._swarmTick();
    }
    this._updateUI();
  }

  /** 大屏自愈：游戏已进入运行/运营状态但大屏仍停留在"等待 HR 任命"占位时，按真实状态强制刷新 */
  _refreshHoloIfStale() {
    if (!this._holoPlaceholder) return;
    if (this.phase !== 'running' && this.phase !== 'operating') return;
    if (!this.hrActor) return;
    this._holoPlaceholder = false;
    if (this.currentSeeker && this.currentSeeker.name) {
      this._drawHoloScreen('赛博公司 · 面试直播', `求职者：${this.currentSeeker.name}`,
        '面试进行中…', '#00e5ff', '#ff2bd6');
    } else {
      const g = this._gradeOf(this.stats.score);
      this._drawHoloScreen('赛博公司 · 世界运行中',
        `CEO: ${this.ceo?.ceo_name || '—'} · ${this.ceo?.company_name || ''}`,
        `在职 ${this.hiredList.length}/${MAX_EMPLOYEES} 人 · 评分 ${this.stats.score}（${g.grade} 级）`,
        '#00e5ff', '#ff2bd6');
    }
  }

  /** 玩家跳跃（空格/单击）：幽灵锚点起跳；空中可再跳一次（双击二段跳） */
  requestJump() {
    if (this._disposed || this.state !== 'playing') return false;
    if (this._isGrounded) {
      this._playerVelocityY = JUMP_SPEED;
      this._isGrounded = false;
      this._airJumps = 0;
      return true;
    }
    // 空中二段跳：再给一次较小冲量
    if (this._airJumps < 1) {
      this._playerVelocityY = JUMP_SPEED * 0.85;
      this._airJumps++;
      return true;
    }
    return false;
  }

  /**
   * 相机防穿模（由 GameModeManager 每帧相机更新后调用）：
   * 幽灵玩家没有实体、可自由穿行，但镜头不应穿进角色模型。
   * 对所有在场角色（HR / 员工 / 当前面试者）的包围球做推出修正，
   * 相机被推到角色表面之外，保持看向玩家目标。
   */
  afterCameraUpdate(camera, lookTarget) {
    if (!camera || this._disposed) return;
    const spheres = [];
    const push = (g) => {
      if (g && g.visible) {
        spheres.push({
          x: g.position.x,
          y: g.position.y + 0.9,   // 球心取角色躯干中段
          z: g.position.z,
          r: 0.55,
        });
      }
    };
    if (this.hrActor) push(this.hrActor.group);
    for (const emp of this.hiredList) {
      if (emp.actor) push(emp.actor.group);
    }
    if (this.currentSeeker && this.currentSeeker.actor) push(this.currentSeeker.actor.group);
    const p = camera.position;
    for (const s of spheres) {
      const dx = p.x - s.x, dy = p.y - s.y, dz = p.z - s.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const minDist = s.r + 0.12;   // 留一点余量，避免贴脸穿模
      if (dist < minDist && dist > 1e-4) {
        const k = minDist / dist;
        p.x = s.x + dx * k;
        p.y = s.y + dy * k;
        p.z = s.z + dz * k;
      } else if (dist <= 1e-4) {
        p.x = s.x;
        p.y = s.y + minDist;
        p.z = s.z;
      }
    }
    if (lookTarget) camera.lookAt(lookTarget);
  }

  /**
   * 大厅初始角色隔离（每帧兜底）：把大厅角色从游戏场景中彻底移除。
   * 仅设 visible=false 不够 —— 模型异步加载完成后 currentAvatar 会重新被引用并显示。
   * 已移除则不再重复操作；退出游戏时由 cleanup 恢复（add 回场景 + 还原可见性）。
   * 相机朝向只初始化一次（首次接触角色时），绝不每帧覆盖 ——
   * 否则用户拖拽转向/仰视/俯视的相机角度会被强制重置，完全失去操控。
   */
  _isolateLobbyAvatar() {
    const avatar = this.App && this.App.currentAvatar;
    // 首次接触大厅角色：保存可见性 + 初始化一次相机朝向（此后交由用户拖拽控制）
    if (avatar && this._savedAvatarVisible === undefined) {
      this._savedAvatarVisible = avatar.visible;
      this.App.smoothRotY = Math.PI;        // 面向办公室
      this.App._gameCamAzimuth = 0;         // 幽灵视角初始朝向（+Z 侧）
      this.App._gameCamPitch = 0.26;        // 适度俯视：一眼看清面试区 + 后排独立办公间
    }
    if (!avatar) return;
    avatar.visible = false;
    // 从场景彻底移除（后台加载完成后也不会再出现在游戏里）
    if (!this._lobbyAvatarRemoved && this.App.scene && avatar.parent === this.App.scene) {
      this.App.scene.remove(avatar);
      this._lobbyAvatarRemoved = true;
    }
  }

  /** 玩家位置（幽灵锚点优先，彻底脱离大厅角色） */
  _playerPos() {
    if (this._ghostAnchor && this._ghostAnchor.position) return this._ghostAnchor.position;
    const av = this.App && this.App.currentAvatar;
    return av ? av.position : { x: LAYOUT.playerSpawn.x, z: LAYOUT.playerSpawn.z };
  }

  updateSceneEffects(t) {
    // 粒子漂浮
    if (this._particlePos && this._particles) {
      const pos = this._particlePos;
      for (let i = 0; i < pos.count; i++) {
        let y = pos.getY(i) + 0.08 * (0.4 + ((i * 7) % 10) / 10);
        if (y > 12) y = 0.2;
        pos.setY(i, y);
      }
      pos.needsUpdate = true;
      this._particles.rotation.y = t * 0.01;
    }
    // 全息屏轻微闪烁
    if (this._holoMesh && this._holoMesh.material) {
      this._holoMesh.material.opacity = 0.82 + Math.sin(t * 3.1) * 0.06;
    }
    // 霓虹灯呼吸
    if (this._neonLights) {
      for (let i = 0; i < this._neonLights.length; i++) {
        const L = this._neonLights[i];
        const base = (L.userData && L.userData.baseIntensity) || 12;
        L.intensity = Math.max(1.2, base + Math.sin(t * (1.0 + i * 0.35)) * 1.5);
      }
    }
  }

  cleanup() {
    this._disposed = true;
    // 若在游戏 VR 会话中退出游戏，先退出 VR（恢复相机与渲染循环）
    if (this.App && this.App.xrMode !== 'off' && this._xrActive) {
      try { this.App.exitXrMode(true); } catch (e) {}
    }
    this._closeDirectHire();   // 关闭直聘面板（若有）
    // 移除独立 VR 按钮
    if (this._vrBtnEl && this._vrBtnEl.parentNode) {
      this._vrBtnEl.parentNode.removeChild(this._vrBtnEl);
    }
    this._vrBtnEl = null;
    this._running = false;
    this._swarmRunning = false;   // 停止蜂群决策循环（循环自身也以 _disposed 退出）
    this._meetingActive = false;  // 停止会议 / 汇报会话
    this._reportActive = false;
    this._gameLoopRunning = false;   // 重置招聘主循环标志：退出后再次进入（恢复存档）才能重新启动主循环
    this._stoppedHiring = false;

    // 退出前保存存档（已任命 HR 且游戏运行过才保存；_saveGame 内部有 hrActor 保护）
    if (this.hrActor && this.phase !== 'idle') {
      try { this._saveGame(); } catch (e) {}
    }

    // 解除玩家消息拦截（恢复大厅 sendText / addUserMsg 原函数，与大厅彻底隔离互不残留）
    this._unhookPlayerChat();

    // 恢复大厅初始角色：add 回场景 + 还原可见性（隔离期间被移出场景）
    if (this.App && this.App.currentAvatar) {
      if (this._lobbyAvatarRemoved && this.App.scene && !this.App.currentAvatar.parent) {
        this.App.scene.add(this.App.currentAvatar);
      }
      this._lobbyAvatarRemoved = false;
      if (this._savedAvatarVisible !== undefined) {
        this.App.currentAvatar.visible = this._savedAvatarVisible;
        this._savedAvatarVisible = undefined;
      }
    }
    this._playerControlledHR = false;

    // 还原场景雾效与灯光（游戏内设置了专属雾和霓虹灯光，直接挂全局 scene；
    // 退出时必须移除/还原，否则大厅角色会被残留雾效与灯光"染色"产生色差）
    if (this.App && this.App.scene) {
      this.App.scene.fog = this._savedFog !== undefined ? this._savedFog : null;
      if (this._addedLights) {
        for (const l of this._addedLights) {
          try { if (l.parent) l.parent.remove(l); } catch (e) {}
        }
        this._addedLights = [];
      }
    }

    // 清理定时器
    for (const id of this._timers) { clearTimeout(id); }
    this._timers = [];
    // 停止所有游戏音频
    for (const a of this._audios) {
      try { a.pause(); } catch (e) {}
    }
    this._audios = [];

    // 销毁所有角色
    for (const a of this.actors) {
      if (a) { try { a.dispose(); } catch (e) {} }
    }
    this.actors = [];

    // 移除 UI
    this._removeUI();

    super.cleanup();
  }

  // ==================== 场景构建 ====================

  _buildOffice() {
    const THREE = this.THREE;
    const { halfW, halfD } = LAYOUT.office;

    // 地板（深色镜面：更通透、反光更干净）
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(halfW * 2, halfD * 2),
      new THREE.MeshStandardMaterial({ color: 0x0b0e1e, roughness: 0.22, metalness: 0.82 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    floor.receiveShadow = true;
    this.addToScene(floor);

    // 赛博网格（更淡，突出地板通透感）
    const grid = new THREE.GridHelper(halfW * 2, halfW, 0x00e5ff, 0x124a63);
    grid.position.y = 0.02;
    grid.material.transparent = true;
    grid.material.opacity = 0.32;
    this.addToScene(grid);

    // 四周墙体
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x121530, roughness: 0.45, metalness: 0.5 });
    const wallH = 9;
    const wallT = 0.5;
    const walls = [
      { x: 0, z: -halfD - wallT / 2, w: halfW * 2, d: wallT },
      { x: 0, z: halfD + wallT / 2, w: halfW * 2, d: wallT },
      { x: -halfW - wallT / 2, z: 0, w: wallT, d: halfD * 2 },
      { x: halfW + wallT / 2, z: 0, w: wallT, d: halfD * 2 },
    ];
    for (const w of walls) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w.w, wallH, w.d), wallMat);
      m.position.set(w.x, wallH / 2, w.z);
      this.addToScene(m);
    }

    // 墙面底部霓虹踢脚线（四边循环，丰富环境氛围）
    const baseStrips = [
      { x: 0, z: -halfD + wallT / 2, len: halfW * 2, ax: 'x', c: 0x00e5ff },
      { x: 0, z: halfD - wallT / 2, len: halfW * 2, ax: 'x', c: 0xff2bd6 },
      { x: -halfW + wallT / 2, z: 0, len: halfD * 2, ax: 'z', c: 0x6ee7ff },
      { x: halfW - wallT / 2, z: 0, len: halfD * 2, ax: 'z', c: 0x9dff6e },
    ];
    for (const s of baseStrips) {
      const geo = s.ax === 'x'
        ? new THREE.BoxGeometry(s.len, 0.07, 0.05)
        : new THREE.BoxGeometry(0.05, 0.07, s.len);
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: s.c }));
      m.position.set(s.x, 0.06, s.z);
      this.addToScene(m);
    }

    // 顶部霓虹灯带
    const stripMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
    const stripMat2 = new THREE.MeshBasicMaterial({ color: 0xff2bd6 });
    const stripH = 0.16;
    const strips = [
      { x: 0, z: -halfD - wallT, len: halfW * 2, ax: 'x', c: stripMat },
      { x: -halfW - wallT, z: 0, len: halfD * 2, ax: 'z', c: stripMat },
      { x: halfW + wallT, z: 0, len: halfD * 2, ax: 'z', c: stripMat2 },
    ];
    for (const s of strips) {
      const geo = s.ax === 'x'
        ? new THREE.BoxGeometry(s.len, stripH, 0.06)
        : new THREE.BoxGeometry(0.06, stripH, s.len);
      const m = new THREE.Mesh(geo, s.c);
      m.position.set(s.x, wallH - 0.4, s.z);
      this.addToScene(m);
    }

    // 天花板（半透明反光）
    const ceil = new THREE.Mesh(
      new THREE.PlaneGeometry(halfW * 2, halfD * 2),
      new THREE.MeshStandardMaterial({ color: 0x0c0f1e, roughness: 0.9, metalness: 0.2, transparent: true, opacity: 0.8 })
    );
    ceil.rotation.x = Math.PI / 2;
    ceil.position.y = wallH;
    this.addToScene(ceil);

    // 天花板发光灯带面板（6 块：更明亮、空间更有层次）
    const panelMat = new THREE.MeshBasicMaterial({ color: 0xcfeaff, transparent: true, opacity: 0.92 });
    const panelFrameMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.22 });
    for (const px of [-11, 0, 11]) {
      for (const pz of [-8, 3]) {
        const panel = new THREE.Mesh(new THREE.BoxGeometry(6.2, 0.06, 5.0), panelMat);
        panel.position.set(px, wallH - 0.18, pz);
        this.addToScene(panel);
        const frame = new THREE.Mesh(new THREE.BoxGeometry(6.5, 0.03, 5.3), panelFrameMat);
        frame.position.set(px, wallH - 0.22, pz);
        this.addToScene(frame);
      }
    }

    // 后墙城市天际线窗带（赛博都市夜景，办公间上方可见）
    const skyTex = this._makeSkylineTexture();
    const sky = new THREE.Mesh(
      new THREE.PlaneGeometry(42, 3.8),
      new THREE.MeshBasicMaterial({ map: skyTex, transparent: true, opacity: 0.95, side: THREE.DoubleSide })
    );
    sky.position.set(0, 4.8, -halfD + wallT - 0.1);
    this.addToScene(sky);
    this._neonStrip(0, 2.75, -halfD + wallT - 0.05, 42, 'x', 0xff2bd6);

    // 雾（记录原值，退出时必须还原，否则大厅角色会被残留雾效"染色"）
    this._savedFog = this.App.scene.fog;
    this.App.scene.fog = new THREE.Fog(0x0a0d1c, 45, 110);
  }

  _neonStrip(x, y, z, len, axis, color) {
    const THREE = this.THREE;
    const geo = axis === 'x'
      ? new THREE.BoxGeometry(len, 0.08, 0.08)
      : new THREE.BoxGeometry(0.08, 0.08, len);
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color }));
    m.position.set(x, y, z);
    this.addToScene(m);
    return m;
  }

  /** 文本自动缩放：字号过大时自动缩小，保证文字（含辉光边距）完整落在画布内不截断 */
  _fitTextFont(ctx, text, w, h, maxPx) {
    const pad = Math.max(10, Math.round(w * 0.055));
    const maxW = w - pad * 2;
    const minPx = Math.max(14, Math.round(h * 0.18));
    let size = Math.max(minPx, maxPx);
    ctx.font = `bold ${size}px "Microsoft YaHei", sans-serif`;
    while (size > minPx && ctx.measureText(String(text)).width > maxW) {
      size -= 2;
      ctx.font = `bold ${size}px "Microsoft YaHei", sans-serif`;
    }
    return size;
  }

  _makeTextPlane(text, opts = {}) {
    const THREE = this.THREE;
    const canvas = document.createElement('canvas');
    const w = opts.width || 1024, h = opts.height || 256;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    // 解析请求字号（优先 opts.font，否则随画布高度等比放大），再自动缩放至完整显示
    const m = String(opts.font || '').match(/(\d+(?:\.\d+)?)px/);
    const maxPx = m ? Math.max(24, Math.round(parseFloat(m[1]))) : Math.max(24, Math.round(h * 0.5));
    const size = this._fitTextFont(ctx, String(text), w, h, maxPx);
    ctx.font = `bold ${size}px "Microsoft YaHei", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    // 辉光
    ctx.shadowColor = opts.glow || '#00e5ff';
    ctx.shadowBlur = 30;
    ctx.fillStyle = opts.color || '#00e5ff';
    ctx.fillText(text, w / 2, h / 2);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry((opts.width3D || 4), (opts.height3D || 1) * (h / w)), mat);
    plane.userData._text = text;
    return plane;
  }

  /** 后墙城市天际线纹理：霓虹都市夜景（Canvas 绘制） */
  _makeSkylineTexture() {
    const THREE = this.THREE;
    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    const rnd = (a, b) => a + Math.random() * (b - a);
    const g = ctx.createLinearGradient(0, 0, 0, 512);
    g.addColorStop(0, '#05060f');
    g.addColorStop(0.55, '#0b0f26');
    g.addColorStop(1, '#170b24');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 2048, 512);
    // 远处高楼 + 亮窗
    for (let i = 0; i < 26; i++) {
      const bw = rnd(42, 96);
      const bh = rnd(90, 330);
      const bx = i * (2048 / 26) + rnd(-14, 14);
      const by = 512 - bh;
      ctx.fillStyle = `rgba(30,42,86,${rnd(0.55, 0.9).toFixed(2)})`;
      ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = `rgba(0,229,255,${rnd(0.15, 0.5).toFixed(2)})`;
      for (let wy = by + 12; wy < 500; wy += 16) {
        for (let wx = bx + 6; wx < bx + bw - 6; wx += 12) {
          if (Math.random() < 0.32) ctx.fillRect(wx, wy, 4, 6);
        }
      }
    }
    // 霓虹“月亮”
    const moon = ctx.createRadialGradient(1560, 96, 8, 1560, 96, 80);
    moon.addColorStop(0, 'rgba(255,43,214,0.95)');
    moon.addColorStop(1, 'rgba(255,43,214,0)');
    ctx.fillStyle = moon;
    ctx.fillRect(1480, 16, 160, 160);
    ctx.fillStyle = '#ff2bd6';
    ctx.beginPath();
    ctx.arc(1560, 96, 26, 0, Math.PI * 2);
    ctx.fill();
    // 近处楼群剪影
    for (let i = 0; i < 12; i++) {
      const bw = rnd(60, 130);
      const bh = rnd(40, 90);
      ctx.fillStyle = 'rgba(10,13,28,0.96)';
      ctx.fillRect(i * (2048 / 12), 512 - bh, bw, bh);
    }
    // 底部光晕
    const floorGlow = ctx.createLinearGradient(0, 480, 0, 512);
    floorGlow.addColorStop(0, 'rgba(0,229,255,0)');
    floorGlow.addColorStop(1, 'rgba(0,229,255,0.55)');
    ctx.fillStyle = floorGlow;
    ctx.fillRect(0, 480, 2048, 32);
    return new THREE.CanvasTexture(canvas);
  }

  /** 赛博霓虹植物（花盆 + 荧光叶） */
  _makePlant(x, z, scale = 1, color = 0x00e5ff) {
    const THREE = this.THREE;
    const g = new THREE.Group();
    const pot = new THREE.Mesh(
      new THREE.CylinderGeometry(0.22 * scale, 0.3 * scale, 0.42 * scale, 10),
      new THREE.MeshStandardMaterial({ color: 0x12162b, roughness: 0.5, metalness: 0.6 })
    );
    pot.position.y = 0.21 * scale;
    g.add(pot);
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035 * scale, 0.05 * scale, 0.55 * scale, 8),
      new THREE.MeshStandardMaterial({ color: 0x2a3550, roughness: 0.7 })
    );
    trunk.position.y = 0.62 * scale;
    g.add(trunk);
    const glowMat = new THREE.MeshBasicMaterial({ color });
    for (let i = 0; i < 4; i++) {
      const leaf = new THREE.Mesh(new THREE.ConeGeometry(0.16 * scale, 0.55 * scale, 8), glowMat);
      const a = (i / 4) * Math.PI * 2 + 0.5;
      leaf.position.set(Math.cos(a) * 0.16 * scale, (1.0 + (i % 2) * 0.18) * scale, Math.sin(a) * 0.16 * scale);
      leaf.rotation.z = Math.cos(a) * 0.35;
      leaf.rotation.x = -Math.sin(a) * 0.35;
      g.add(leaf);
    }
    const top = new THREE.Mesh(new THREE.ConeGeometry(0.13 * scale, 0.4 * scale, 8), glowMat);
    top.position.y = (1.43 + 0.18) * scale;
    g.add(top);
    g.position.set(x, 0, z);
    this.addToScene(g);
    return g;
  }

  /** 墙面霓虹艺术画 */
  _makeWallArt(x, y, z, color) {
    const THREE = this.THREE;
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 640;
    const ctx = canvas.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, 1024, 640);
    g.addColorStop(0, '#060814');
    g.addColorStop(1, '#101a38');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 1024, 640);
    const c = '#' + color.toString(16).padStart(6, '0');
    ctx.strokeStyle = c;
    ctx.shadowColor = c;
    ctx.shadowBlur = 18;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(80, 560);
    ctx.lineTo(260, 320);
    ctx.lineTo(430, 430);
    ctx.lineTo(610, 150);
    ctx.lineTo(790, 300);
    ctx.lineTo(950, 90);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(610, 150, 90, 0, Math.PI * 2);
    ctx.stroke();
    const tex = new THREE.CanvasTexture(canvas);
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(3.2, 2.0),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.75, side: THREE.DoubleSide })
    );
    mesh.position.set(x, y, z);
    mesh.rotation.y = x < 0 ? Math.PI / 2 : -Math.PI / 2;
    this.addToScene(mesh);
    return mesh;
  }

  _buildStage() {
    const THREE = this.THREE;
    const { desk, hr, seeker } = LAYOUT.stage;

    // 面试区舞台地面（垫高视觉层次，与办公区自然分区）
    const carpet = new THREE.Mesh(
      new THREE.BoxGeometry(7.4, 0.03, 4.6),
      new THREE.MeshStandardMaterial({ color: 0x0d1122, roughness: 0.3, metalness: 0.75 })
    );
    carpet.position.set(0, 0.015, 0);
    this.addToScene(carpet);
    const carpetEdge = new THREE.Mesh(
      new THREE.BoxGeometry(7.4, 0.02, 0.05),
      new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.5 })
    );
    carpetEdge.position.set(0, 0.045, -2.28);
    this.addToScene(carpetEdge);
    const carpetEdge2 = new THREE.Mesh(
      new THREE.BoxGeometry(7.4, 0.02, 0.05),
      new THREE.MeshBasicMaterial({ color: 0xff2bd6, transparent: true, opacity: 0.5 })
    );
    carpetEdge2.position.set(0, 0.045, 2.28);
    this.addToScene(carpetEdge2);

    // 面试区地环（发光圆环）
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(3.4, 3.7, 48),
      new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.5, side: THREE.DoubleSide })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(0, 0.035, 0);
    this.addToScene(ring);
    const ring2 = new THREE.Mesh(
      new THREE.RingGeometry(2.2, 2.32, 48),
      new THREE.MeshBasicMaterial({ color: 0xff2bd6, transparent: true, opacity: 0.35, side: THREE.DoubleSide })
    );
    ring2.rotation.x = -Math.PI / 2;
    ring2.position.set(0, 0.04, 0);
    this.addToScene(ring2);

    // 面试桌（带全息桌面屏）
    const deskGroup = new THREE.Group();
    const deskTop = new THREE.Mesh(
      new THREE.BoxGeometry(5.2, 0.14, 1.4),
      new THREE.MeshStandardMaterial({ color: 0x1a1f33, roughness: 0.4, metalness: 0.7 })
    );
    deskTop.position.y = 0.9;
    const deskLegMat = new THREE.MeshStandardMaterial({ color: 0x11152a, metalness: 0.8, roughness: 0.3 });
    for (const [lx, lz] of [[-2.3, -0.55], [2.3, -0.55], [-2.3, 0.55], [2.3, 0.55]]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.9, 0.16), deskLegMat);
      leg.position.set(lx, 0.45, lz);
      deskGroup.add(leg);
    }
    deskGroup.add(deskTop);
    // 桌面全息屏
    const holoTablet = new THREE.Mesh(
      new THREE.PlaneGeometry(4.6, 0.9),
      new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.22, side: THREE.DoubleSide })
    );
    holoTablet.rotation.x = -Math.PI / 2;
    holoTablet.position.y = 1.02;
    deskGroup.add(holoTablet);
    deskGroup.position.set(desk.x, 0, desk.z);
    this.addToScene(deskGroup);

    // 站立面试：不放置椅子，HR 与求职者在桌两侧站立交谈
    void hr; void seeker;

    // 「面试区」全息牌
    const sign = this._makeTextPlane('● 面试区 INTERVIEW ●', {
      width: 512, height: 128, width3D: 4.4, height3D: 1.1, color: '#00e5ff', glow: '#00e5ff',
      font: 'bold 64px "Microsoft YaHei", sans-serif',
    });
    sign.position.set(0, 3.1, -1.4);
    this.addToScene(sign);
  }

  _buildWorkstations() {
    const THREE = this.THREE;
    const { halfW: rw, halfD: rd, wallH } = LAYOUT.room;
    const glassMat = new THREE.MeshStandardMaterial({
      color: 0x9fc2ff, transparent: true, opacity: 0.16,
      roughness: 0.08, metalness: 0.25, side: THREE.DoubleSide, depthWrite: false,
    });
    const deskMat = new THREE.MeshStandardMaterial({ color: 0x1a2138, roughness: 0.35, metalness: 0.75 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x151a30, roughness: 0.45, metalness: 0.7 });
    const legMat = new THREE.MeshStandardMaterial({ color: 0x0e1226, metalness: 0.85, roughness: 0.3 });

    for (const ws of LAYOUT.workstations) {
      const posDef = POSITIONS.find(p => p.key === ws.posKey);
      const accent = ROOM_ACCENTS[ws.posKey] || 0x00e5ff;
      const accentCss = '#' + accent.toString(16).padStart(6, '0');
      const room = new THREE.Group();
      room.position.set(ws.x, 0, ws.z);

      // ---- 背墙（实心面板）+ 上/下霓虹饰线 ----
      const back = new THREE.Mesh(new THREE.BoxGeometry(rw * 2 + 0.12, wallH, 0.1), darkMat);
      back.position.set(0, wallH / 2, -rd - 0.05);
      room.add(back);
      const backTrim = new THREE.Mesh(
        new THREE.BoxGeometry(rw * 2 + 0.12, 0.08, 0.06),
        new THREE.MeshBasicMaterial({ color: accent })
      );
      backTrim.position.set(0, wallH - 0.06, -rd - 0.04);
      room.add(backTrim);
      const backBase = new THREE.Mesh(
        new THREE.BoxGeometry(rw * 2 + 0.12, 0.06, 0.08),
        new THREE.MeshBasicMaterial({ color: accent })
      );
      backBase.position.set(0, 0.06, -rd - 0.03);
      room.add(backBase);

      // 背墙全息背光板（岗位主题色）
      const backGlow = new THREE.Mesh(
        new THREE.PlaneGeometry(2.2, 1.15),
        new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false })
      );
      backGlow.position.set(0, wallH * 0.72, -rd - 0.1);
      room.add(backGlow);

      // ---- 两侧玻璃隔断（半透明）+ 前缘霓虹立柱 ----
      for (const sx of [-1, 1]) {
        const glass = new THREE.Mesh(
          new THREE.BoxGeometry(0.06, wallH - 0.35, rd * 2 + 0.1),
          glassMat
        );
        glass.position.set(sx * (rw + 0.03), (wallH - 0.35) / 2, -0.1);
        room.add(glass);
        const pillar = new THREE.Mesh(
          new THREE.BoxGeometry(0.08, wallH - 0.25, 0.08),
          new THREE.MeshBasicMaterial({ color: accent })
        );
        pillar.position.set(sx * (rw + 0.05), (wallH - 0.25) / 2, rd - 0.02);
        room.add(pillar);
      }

      // ---- 门楣横梁 + 霓虹灯带（开放式入口，无门扇） ----
      const beam = new THREE.Mesh(
        new THREE.BoxGeometry(rw * 2 + 0.18, 0.16, 0.12),
        legMat
      );
      beam.position.set(0, wallH - 0.12, rd + 0.02);
      room.add(beam);
      const beamNeon = new THREE.Mesh(
        new THREE.BoxGeometry(rw * 2 + 0.1, 0.05, 0.04),
        new THREE.MeshBasicMaterial({ color: accent })
      );
      beamNeon.position.set(0, wallH - 0.2, rd + 0.1);
      room.add(beamNeon);

      // ---- 岗位色地板光毯 ----
      const rug = new THREE.Mesh(
        new THREE.BoxGeometry(rw * 2 - 0.7, 0.02, rd * 2 - 0.7),
        new THREE.MeshStandardMaterial({ color: accent, transparent: true, opacity: 0.28, roughness: 0.5, metalness: 0.2 })
      );
      rug.position.set(0, 0.012, 0);
      room.add(rug);

      // ---- 办公桌（靠背墙，面向入口） ----
      const deskTop = new THREE.Mesh(new THREE.BoxGeometry(2.5, 0.1, 0.95), deskMat);
      deskTop.position.set(0, 0.8, -1.1);
      room.add(deskTop);
      for (const [lx, lz] of [[-1.1, -1.5], [1.1, -1.5], [-1.1, -0.7], [1.1, -0.7]]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.75, 0.1), legMat);
        leg.position.set(lx, 0.4, lz);
        room.add(leg);
      }

      // 全息显示器 + 支架 + 键盘垫
      const mon = new THREE.Mesh(
        new THREE.BoxGeometry(1.65, 0.98, 0.05),
        new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0.92 })
      );
      mon.position.set(0, 1.42, -0.98);
      room.add(mon);
      const monStand = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.2, 0.24), legMat);
      monStand.position.set(0, 0.98, -0.98);
      room.add(monStand);
      const kb = new THREE.Mesh(
        new THREE.BoxGeometry(0.8, 0.02, 0.28),
        new THREE.MeshStandardMaterial({ color: 0x0b0e1c, roughness: 0.4 })
      );
      kb.position.set(0, 0.86, -0.62);
      room.add(kb);

      // 霓虹台灯
      const lampBase = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.03, 0.18), legMat);
      lampBase.position.set(0.95, 0.82, -1.35);
      room.add(lampBase);
      const lampArm = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.28, 8), legMat);
      lampArm.position.set(0.95, 0.97, -1.35);
      room.add(lampArm);
      const lampHead = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 8), new THREE.MeshBasicMaterial({ color: accent }));
      lampHead.position.set(0.95, 1.13, -1.35);
      room.add(lampHead);

      // 桌面小盆栽（岗位色荧光叶）
      const plantPot = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.12, 8), darkMat);
      plantPot.position.set(-0.95, 0.87, -1.35);
      room.add(plantPot);
      const plantLeaf = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.22, 8), new THREE.MeshBasicMaterial({ color: 0x6dffa8 }));
      plantLeaf.position.set(-0.95, 1.05, -1.35);
      room.add(plantLeaf);

      // 办公椅（门侧角落，避免与员工站位重叠）
      const chair = new THREE.Group();
      const chairSeat = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.07, 0.5), darkMat);
      chairSeat.position.y = 0.62;
      chair.add(chairSeat);
      const chairBack = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.55, 0.06), darkMat);
      chairBack.position.set(0, 0.94, -0.22);
      chair.add(chairBack);
      const chairPost = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.5, 8), legMat);
      chairPost.position.y = 0.35;
      chair.add(chairPost);
      const chairBase = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.34, 0.04, 8), legMat);
      chairBase.position.y = 0.08;
      chair.add(chairBase);
      chair.position.set(-1.2, 0, 1.15);
      chair.rotation.y = -0.5;
      room.add(chair);

      // 左墙置物架（三层）
      for (let s = 0; s < 3; s++) {
        const shelf = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.05, 0.3), darkMat);
        shelf.position.set(-rw + 0.06, 1.25 + s * 0.5, -0.4);
        room.add(shelf);
      }

      this.addToScene(room);

      // 房间门牌（入职后由 _setLabelText 更新为员工名）
      const label = this._makeTextPlane(`${posDef?.icon || ''} ${posDef?.name || ''}`, {
        width: 1024, height: 256, width3D: 3.4, height3D: 0.85,
        color: accentCss, glow: accentCss,
        font: 'bold 128px "Microsoft YaHei", sans-serif',
      });
      label.position.set(ws.x, 3.05, ws.z + rd + 0.35);
      this.addToScene(label);

      // 记录房间牌引用（入职后更新为员工名）
      ws.label = label;
      ws.posDef = posDef;
    }
  }

  _buildEntrance() {
    const THREE = this.THREE;
    const { entrance } = LAYOUT;
    const pillarMat = new THREE.MeshStandardMaterial({ color: 0x171c36, metalness: 0.75, roughness: 0.3 });
    for (const dx of [-1.6, 1.6]) {
      const p = new THREE.Mesh(new THREE.BoxGeometry(0.7, 6, 0.7), pillarMat);
      p.position.set(entrance.x + dx, 3, entrance.z);
      this.addToScene(p);
      const strip = new THREE.Mesh(new THREE.BoxGeometry(0.12, 6, 0.12), new THREE.MeshBasicMaterial({ color: 0xff2bd6 }));
      strip.position.set(entrance.x + dx + (dx > 0 ? -0.41 : 0.41), 3, entrance.z + 0.41);
      this.addToScene(strip);
    }
    // 横梁霓虹
    this._neonStrip(entrance.x, 6.2, entrance.z, 4.6, 'x', 0xff2bd6);
    // 公司招牌
    const logo = this._makeTextPlane('赛博公司 CYBERCORP', {
      width: 1024, height: 256, width3D: 8, height3D: 2,
      color: '#ff2bd6', glow: '#ff2bd6', font: 'bold 110px "Microsoft YaHei", sans-serif',
    });
    logo.position.set(entrance.x, 6.8, entrance.z);
    this.addToScene(logo);

    // 门廊地面光带（从入口延伸向办公区）
    const pathMat = new THREE.MeshBasicMaterial({ color: 0xff2bd6, transparent: true, opacity: 0.22 });
    const pathStrip = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.02, 3.4), pathMat);
    pathStrip.position.set(entrance.x, 0.025, entrance.z - 1.9);
    this.addToScene(pathStrip);
    const pathMat2 = new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.18 });
    const pathStrip2 = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.02, 2.6), pathMat2);
    pathStrip2.position.set(entrance.x - 2.0, 0.025, entrance.z - 3.1);
    this.addToScene(pathStrip2);

    // 前台接待台（全息铭牌 + 霓虹饰线）
    const rec = LAYOUT.reception;
    const recMat = new THREE.MeshStandardMaterial({ color: 0x151a30, roughness: 0.4, metalness: 0.7 });
    const recBody = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.95, 0.75), recMat);
    recBody.position.set(rec.x, 0.475, rec.z);
    this.addToScene(recBody);
    const recTop = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.08, 0.9), new THREE.MeshStandardMaterial({ color: 0x0e1226, roughness: 0.5, metalness: 0.8 }));
    recTop.position.set(rec.x, 0.99, rec.z);
    this.addToScene(recTop);
    const recNeon = new THREE.Mesh(new THREE.BoxGeometry(2.25, 0.05, 0.03), new THREE.MeshBasicMaterial({ color: 0x00e5ff }));
    recNeon.position.set(rec.x, 1.08, rec.z + 0.46);
    this.addToScene(recNeon);
    const recLabel = this._makeTextPlane('接待处 RECEPTION', {
      width: 768, height: 192, width3D: 2.6, height3D: 0.7,
      color: '#6ee7ff', glow: '#6ee7ff', font: 'bold 88px "Microsoft YaHei", sans-serif',
    });
    recLabel.position.set(rec.x, 1.9, rec.z);
    this.addToScene(recLabel);

    // 候场长椅（后移至候选人身后，避免站姿重叠）
    const benchMat = new THREE.MeshStandardMaterial({ color: 0x141a30, roughness: 0.5, metalness: 0.6 });
    const bench = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.12, 0.6), benchMat);
    bench.position.set(LAYOUT.waitingSpot.x, 0.55, LAYOUT.waitingSpot.z + 0.7);
    this.addToScene(bench);
    const benchLegs = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.5, 0.06), benchMat);
    benchLegs.position.set(LAYOUT.waitingSpot.x, 0.28, LAYOUT.waitingSpot.z + 1.0);
    this.addToScene(benchLegs);
    const benchCushion = new THREE.Mesh(
      new THREE.BoxGeometry(3.3, 0.05, 0.45),
      new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.35 })
    );
    benchCushion.position.set(LAYOUT.waitingSpot.x, 0.63, LAYOUT.waitingSpot.z + 0.7);
    this.addToScene(benchCushion);
    const waitLabel = this._makeTextPlane('候场区 WAITING', {
      width: 384, height: 96, width3D: 2.6, height3D: 0.65, color: '#6ee7ff', glow: '#6ee7ff',
      font: 'bold 56px "Microsoft YaHei", sans-serif',
    });
    waitLabel.position.set(LAYOUT.waitingSpot.x, 1.5, LAYOUT.waitingSpot.z + 0.7);
    this.addToScene(waitLabel);
  }

  _buildHoloScreen() {
    const THREE = this.THREE;
    const { x, y, z } = LAYOUT.holoScreen;
    // 全息屏框架（悬浮于办公间上方：10.4 × 5.6，不再遮挡面试区与工位）
    const fw = 10.4, fh = 5.6, t = 0.1;
    const frame = new THREE.Mesh(
      new THREE.BoxGeometry(fw, fh, 0.18),
      new THREE.MeshBasicMaterial({ color: 0x0a2a3a, transparent: true, opacity: 0.28 })
    );
    frame.position.set(x, y, z);
    this.addToScene(frame);
    // 全息屏幕（CanvasTexture 动态更新；2048×1152 高清分辨率 + 自动换行可容纳长句）
    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 1152;
    const ctx = canvas.getContext('2d');
    this._holoCanvas = canvas;
    this._holoCtx = ctx;
    const tex = new THREE.CanvasTexture(canvas);
    this._holoTexture = tex;
    const screen = new THREE.Mesh(
      new THREE.PlaneGeometry(10.0, 5.6),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.9, depthWrite: false })
    );
    screen.position.set(x, y, z + 0.1);
    this.addToScene(screen);
    this._holoMesh = screen;
    // 边缘霓虹条（上下左右四条）
    for (const [sx, sy, sw, sh] of [
      [x, y + fh / 2, fw, t], [x, y - fh / 2, fw, t],
      [x - fw / 2, y, t, fh], [x + fw / 2, y, t, fh],
    ]) {
      const strip = new THREE.Mesh(
        new THREE.BoxGeometry(sw, sh, 0.06),
        new THREE.MeshBasicMaterial({ color: 0x00e5ff })
      );
      strip.position.set(sx, sy, z);
      this.addToScene(strip);
    }
    this._drawHoloScreen('赛博公司 · 招聘系统', 'SYSTEM ONLINE', '等待 HR 任命…', '#00e5ff', '#ff2bd6');
    this._holoPlaceholder = true;   // 标记占位画面：游戏进入真实状态后若仍停留则强制刷新
  }

  /** 环境装饰：中枢区徽标、服务器机柜、能量站、霓虹艺术画与植物点缀 */
  _buildDecor() {
    const THREE = this.THREE;
    const at = LAYOUT.atrium;

    // 中枢区：地面全息徽标（面试区与办公间之间的视觉锚点）
    const ringA = new THREE.Mesh(
      new THREE.RingGeometry(3.0, 3.15, 64),
      new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.4, side: THREE.DoubleSide })
    );
    ringA.rotation.x = -Math.PI / 2;
    ringA.position.set(at.x, 0.025, at.z);
    this.addToScene(ringA);
    const ringB = new THREE.Mesh(
      new THREE.RingGeometry(1.5, 1.55, 64),
      new THREE.MeshBasicMaterial({ color: 0xff2bd6, transparent: true, opacity: 0.35, side: THREE.DoubleSide })
    );
    ringB.rotation.x = -Math.PI / 2;
    ringB.position.set(at.x, 0.03, at.z);
    this.addToScene(ringB);
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(1.45, 48),
      new THREE.MeshBasicMaterial({ color: 0x0b1f33, transparent: true, opacity: 0.5, side: THREE.DoubleSide })
    );
    disc.rotation.x = -Math.PI / 2;
    disc.position.set(at.x, 0.02, at.z);
    this.addToScene(disc);

    // 左侧：服务器机柜
    const rack = new THREE.Mesh(
      new THREE.BoxGeometry(0.85, 2.3, 1.2),
      new THREE.MeshStandardMaterial({ color: 0x0d1226, roughness: 0.5, metalness: 0.8 })
    );
    rack.position.set(-21.7, 1.15, -4);
    this.addToScene(rack);
    for (let i = 0; i < 5; i++) {
      const led = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 0.06, 0.03),
        new THREE.MeshBasicMaterial({ color: i % 2 ? 0x00e5ff : 0x9dff6e })
      );
      led.position.set(-21.7, 0.5 + i * 0.38, -4.64);
      this.addToScene(led);
    }
    const rackLabel = this._makeTextPlane('SERVERS', {
      width: 512, height: 128, width3D: 2.0, height3D: 0.5,
      color: '#00e5ff', glow: '#00e5ff', font: 'bold 64px "Microsoft YaHei", sans-serif',
    });
    rackLabel.position.set(-21.7, 2.75, -4.7);
    rackLabel.rotation.y = Math.PI / 2;
    this.addToScene(rackLabel);

    // 右侧：能量补给站
    const bar = new THREE.Group();
    const barBody = new THREE.Mesh(
      new THREE.BoxGeometry(1.6, 0.95, 0.8),
      new THREE.MeshStandardMaterial({ color: 0x171c36, roughness: 0.4, metalness: 0.7 })
    );
    barBody.position.y = 0.475;
    bar.add(barBody);
    const disp = new THREE.Mesh(
      new THREE.BoxGeometry(0.55, 1.05, 0.45),
      new THREE.MeshStandardMaterial({ color: 0x0d1226, roughness: 0.5, metalness: 0.75 })
    );
    disp.position.set(0.25, 1.15, 0);
    bar.add(disp);
    const dispScreen = new THREE.Mesh(
      new THREE.PlaneGeometry(0.32, 0.28),
      new THREE.MeshBasicMaterial({ color: 0x9dff6e, transparent: true, opacity: 0.9, side: THREE.DoubleSide })
    );
    dispScreen.position.set(0.25, 1.3, 0.23);
    bar.add(dispScreen);
    bar.position.set(21.7, 0, -4);
    bar.rotation.y = Math.PI;
    this.addToScene(bar);
    const barLabel = this._makeTextPlane('ENERGY BAR', {
      width: 512, height: 128, width3D: 2.2, height3D: 0.55,
      color: '#ffd166', glow: '#ffd166', font: 'bold 64px "Microsoft YaHei", sans-serif',
    });
    barLabel.position.set(21.7, 2.4, -4.7);
    barLabel.rotation.y = -Math.PI / 2;
    this.addToScene(barLabel);

    // 两侧墙面霓虹艺术画
    this._makeWallArt(-22.55, 4.4, 0, 0x00e5ff);
    this._makeWallArt(22.55, 4.4, 0, 0xff4df0);

    // 植物点缀（入口 / 候场 / 面试区两侧 / 办公间后走廊）
    this._makePlant(LAYOUT.entrance.x - 2.7, LAYOUT.entrance.z - 0.5, 1, 0x00e5ff);
    this._makePlant(LAYOUT.entrance.x + 2.7, LAYOUT.entrance.z - 0.5, 1, 0xff2bd6);
    this._makePlant(LAYOUT.waitingSpot.x + 2.2, LAYOUT.waitingSpot.z + 0.6, 0.9, 0x6ee7ff);
    this._makePlant(-4.2, -2.2, 0.9, 0x9dff6e);
    this._makePlant(4.2, -2.2, 0.9, 0x9dff6e);
    this._makePlant(-20.5, -12.2, 1.1, 0xff9d6e);
    this._makePlant(-16.0, -14.0, 0.9, 0x6ee7ff);
    this._makePlant(20.5, -12.2, 1.1, 0x00e5ff);
    this._makePlant(16.0, -14.0, 0.9, 0xff4df0);
    this._makePlant(0, -14.0, 1.0, 0xffd166);
  }

  _addLighting() {
    const THREE = this.THREE;
    this._addedLights = [];
    // 氛围灯光：环境光 + 半球光 + 霓虹点光源（呼吸效果），不添加方向光/补光等实光源
    const amb = new THREE.AmbientLight(0x3d4468, 0.75);
    this.App.scene.add(amb);
    this._addedLights.push(amb);
    const hemi = new THREE.HemisphereLight(0x51669c, 0x10101a, 0.85);
    this.App.scene.add(hemi);
    this._addedLights.push(hemi);
    // 霓虹点光源（呼吸效果在 updateSceneEffects 中驱动，baseIntensity 为基准强度）
    const neon = [
      { pos: [0, 5.5, -9.5], color: 0x00e5ff, dist: 36, base: 11 },
      { pos: [0, 4.0, 1.5], color: 0xff2bd6, dist: 20, base: 7 },
      { pos: [-14, 4.5, 12], color: 0x6ee7ff, dist: 18, base: 6 },
      { pos: [16, 4.0, -9], color: 0xff9d6e, dist: 24, base: 6 },
      { pos: [-16, 4.0, -9], color: 0x9dff6e, dist: 24, base: 6 },
    ];
    this._neonLights = [];
    for (const n of neon) {
      const pl = new THREE.PointLight(n.color, n.base, n.dist, 2);
      pl.userData.baseIntensity = n.base;
      pl.position.set(n.pos[0], n.pos[1], n.pos[2]);
      this.App.scene.add(pl);
      this._addedLights.push(pl);
      this._neonLights.push(pl);
    }
  }

  _addParticles() {
    const THREE = this.THREE;
    const count = 220;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 44;
      positions[i * 3 + 1] = Math.random() * 9;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 28;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0x22ddff, size: 0.04, transparent: true, opacity: 0.5,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const pts = new THREE.Points(geo, mat);
    this.addToScene(pts);
    this._particles = pts;
    this._particlePos = geo.attributes.position;
  }

  _addColliders() {
    // 玩家碰撞体：面试桌、舞台角色位、接待台、入口立柱 + 每间办公间的墙体（矩形碰撞）
    const { halfW: rw, halfD: rd } = LAYOUT.room;
    this._blockers = [
      { x: LAYOUT.stage.desk.x, z: LAYOUT.stage.desk.z, r: 2.0 },
      { x: LAYOUT.stage.hr.x, z: LAYOUT.stage.hr.z, r: 0.9 },
      { x: LAYOUT.stage.seeker.x, z: LAYOUT.stage.seeker.z, r: 0.9 },
      { x: LAYOUT.reception.x, z: LAYOUT.reception.z, r: 1.1 },
      { x: LAYOUT.entrance.x - 1.6, z: LAYOUT.entrance.z, r: 0.8 },
      { x: LAYOUT.entrance.x + 1.6, z: LAYOUT.entrance.z, r: 0.8 },
    ];
    for (const ws of LAYOUT.workstations) {
      const zBack = ws.z - rd - 0.1;
      const zFront = ws.z + rd + 0.02;
      // 背墙
      this._blockers.push({ rect: { x1: ws.x - rw - 0.12, x2: ws.x + rw + 0.12, z1: zBack - 0.1, z2: zBack + 0.1 } });
      // 两侧玻璃墙
      for (const sx of [-1, 1]) {
        const wx = ws.x + sx * (rw + 0.03);
        this._blockers.push({ rect: { x1: wx - 0.1, x2: wx + 0.1, z1: zBack - 0.12, z2: zFront } });
      }
    }
  }

  checkCollision(x, z) {
    // 办公室墙体边界（防止玩家走出办公室掉进虚空）
    const { halfW, halfD } = LAYOUT.office;
    if (Math.abs(x) > halfW - 0.5 || Math.abs(z) > halfD - 0.5) return true;
    for (const b of this._blockers) {
      if (b.rect) {
        const m = 0.38;
        const nx = Math.max(b.rect.x1 - m, Math.min(x, b.rect.x2 + m));
        const nz = Math.max(b.rect.z1 - m, Math.min(z, b.rect.z2 + m));
        if (Math.hypot(x - nx, z - nz) < m) return true;
      } else if (Math.hypot(x - b.x, z - b.z) < b.r) {
        return true;
      }
    }
    return false;
  }

  getExtraState() {
    return {
      world_id: this.worldId || '',
      ceo: this.ceo ? this.ceo.ceo_name : '',
      company: this.ceo ? this.ceo.company_name : '',
      hr: this.hrActor ? this.hrActor.name : '',
      hired: this.hiredList.length,
      agents_online: this.actors.filter(a => a && !a.removed).length,
      swarm_decisions: this._swarmDecisions || 0,
      positions_filled: this.filledPositions.size,
      positions_total: POSITIONS.length - 1,
      hired_list: this.hiredList.map(h => `${h.position.name}:${h.name}`).join(','),
    };
  }

  // ==================== HR 任命 ====================

  async _loadCardsAndShowHRSelect() {
    if (this._disposed) return;
    if (!this.cards || this.cards.length === 0) {
      let cards = [];
      try {
        const res = await fetch('/api/character_cards');
        const data = await res.json();
        cards = data.cards || [];
      } catch (e) {
        console.warn('[赛博公司] 读取角色卡片失败', e);
      }
      if (this._disposed) return;
      this.cards = cards;
    }
    this._showHRSelectUI(this.cards);
  }

  // ==================== 蜂群引擎：世界身份流程（R2/R3） ====================

  /** 进入游戏的身份入口：恢复最近世界，或就任新 CEO */
  async _startCEOFlow() {
    if (this._disposed) return;
    if (!this.cards || this.cards.length === 0) {
      let cards = [];
      try {
        const res = await fetch('/api/character_cards');
        const data = await res.json();
        cards = data.cards || [];
      } catch (e) {
        console.warn('[赛博公司] 读取角色卡片失败', e);
      }
      if (this._disposed) return;
      this.cards = cards;
    }
    const save = this._loadSave();
    if (save && save.ceo) {
      this._showResumeDialog();
      return;
    }
    this._startCEOOnboarding();
  }

  /** CEO 就任表单：创建独立于大厅身份的世界身份（R2） */
  _startCEOOnboarding() {
    if (this._disposed) return;
    const root = document.createElement('div');
    root.className = 'cyber-hr-select';
    const pool = ['王赛博', '李霓虹', '陈数据', '赵云端', '周芯片'];
    const defName = pool[Math.floor(Math.random() * pool.length)];
    root.innerHTML = `
      <div class="cyber-hr-panel" style="max-width:540px;">
        <div class="cyber-hr-logo">蜂群游戏引擎 · 世界身份初始化</div>
        <h3 class="cyber-hr-title">⚡ 就任 CEO · 创建独立世界</h3>
        <p class="cyber-hr-sub" style="text-align:center;">你的 CEO 身份与世界绑定（world_id），独立于大厅角色——大厅形象不会带入游戏。<br>就任后，所有员工智能体将通过「身份注入」认识你。</p>
        <div class="cyber-ceo-form">
          <label>CEO 姓名</label>
          <input id="cc-ceo-name" maxlength="12" placeholder="${defName}">
          <label>公司名称</label>
          <input id="cc-ceo-company" maxlength="16" placeholder="赛博公司">
          <label>公司理念（可选，将注入员工认知）</label>
          <input id="cc-ceo-motto" maxlength="40" placeholder="例如：效率至上，人人都是创客">
        </div>
        <div style="display:flex;gap:12px;justify-content:center;margin-top:18px;">
          <button class="cyber-hr-random" id="cc-ceo-ok" style="margin:0;">⚡ 就任 CEO</button>
          <button class="cyber-hr-random" id="cc-ceo-rnd" style="margin:0;opacity:.7;">🎲 随机</button>
        </div>
      </div>
    `;
    document.body.appendChild(root);
    this._ceoUI = root;
    const commit = () => {
      const name = (root.querySelector('#cc-ceo-name').value || defName).trim().slice(0, 12) || defName;
      const company = (root.querySelector('#cc-ceo-company').value || '赛博公司').trim().slice(0, 16) || '赛博公司';
      const motto = (root.querySelector('#cc-ceo-motto').value || '').trim().slice(0, 40);
      if (this._disposed) return;
      document.body.removeChild(root);
      this._ceoUI = null;
      this._initWorldIdentity(name, company, motto);
      this._loadCardsAndShowHRSelect();
    };
    root.querySelector('#cc-ceo-ok').addEventListener('click', commit);
    root.querySelector('#cc-ceo-rnd').addEventListener('click', () => {
      root.querySelector('#cc-ceo-name').value = pool[Math.floor(Math.random() * pool.length)];
    });
    root.querySelector('#cc-ceo-name').focus();
  }

  /** 初始化世界身份：world_id = f(CEO 身份)（R1/R3 单租户世界） */
  _initWorldIdentity(ceoName, companyName, motto) {
    const seed = `${ceoName}|${companyName}`.toLowerCase().replace(/\s+/g, '');
    let h = 2166136261;
    for (let i = 0; i < seed.length; i++) { h ^= seed.charCodeAt(i); h = Math.imul(h, 16777619); }
    const hex = (h >>> 0).toString(16).padStart(6, '0').slice(0, 6);
    this.worldId = WORLD_ID_PREFIX + hex;
    this.ceo = {
      ceo_id: 'ceo_' + hex,
      ceo_name: ceoName,
      company_name: companyName,
      motto: motto || '',
      created_at: Date.now(),
    };
    try { localStorage.setItem(LAST_WORLD_KEY, this.worldId); } catch (e) {}
    this._loadChats();   // 加载该世界的员工独立对话记录（世界隔离）
    this._initTaskBoard();   // 初始化公司任务看板（RL 产出）
    this._appendTranscript('system', `⚡ CEO「${ceoName}」就任 · 世界 ${this.worldId} 已初始化（与大厅身份隔离）`);
    if (this._holoCtx) {
      this._drawHoloScreen('赛博公司 · 世界就绪',
        `CEO: ${ceoName} · ${companyName}`,
        `世界 ${this.worldId} · 身份注入已启用 · 所有员工将认识 CEO`, '#00e5ff', '#ff2bd6');
    }
  }

  // ==================== 公司任务看板 + 员工 RL 自主决策 ====================

  /** 初始化公司任务看板（RL 产出的载体；支持从存档恢复任务进度与员工状态） */
  _initTaskBoard(savedTasks, savedEnergy, savedRL, savedPlayerTask) {
    // 恢复 CEO 项目任务（玩家布置 → 蜂群拆解；世界隔离持久化）
    this._playerTask = null;
    if (savedPlayerTask && savedPlayerTask.subtasks && savedPlayerTask.subtasks.length) {
      this._playerTask = {
        id: savedPlayerTask.id || ('pt_' + Date.now()),
        title: savedPlayerTask.title || '公司新项目',
        status: savedPlayerTask.status || 'running',
        createdAt: savedPlayerTask.createdAt || Date.now(),
        doneAt: savedPlayerTask.doneAt || 0,
        votes: Array.isArray(savedPlayerTask.votes) ? savedPlayerTask.votes : [],
        subtasks: savedPlayerTask.subtasks.map(s => ({
          posKey: s.posKey, icon: s.icon || '📋',
          title: s.title || PROJECT.SUBTASK_TEMPLATE[s.posKey] || '子任务',
          complexity: s.complexity || 60,
          progress: s.progress || 0,
          quality: s.quality != null ? s.quality : PROJECT.QUALITY_BASE,
          done: !!s.done, doneAt: s.doneAt || 0,
          contributors: Array.isArray(s.contributors) ? s.contributors : [],
          projectId: s.projectId || savedPlayerTask.id || null,
        })),
      };
    }
    this._tasks = (savedTasks && savedTasks.length)
      ? savedTasks.map(t => {
          // 恢复时已完成的看板项直接滚动到新一轮（避免恢复后员工被卡在"已完成"任务上）
          if (t.done) {
            const def = COMPANY_TASKS.find(x => x.posKey === t.posKey) || COMPANY_TASKS[0];
            const round = (t.round || 1) + 1;
            return {
              posKey: def.posKey, icon: def.icon,
              title: `${def.title} · R${round}`,
              complexity: Math.round(def.complexity * (1 + (round - 1) * 0.15)),
              progress: 0, quality: RL_CFG.QUALITY_BASE, done: false, doneAt: 0, round,
              contributors: [],
            };
          }
          return {
            posKey: t.posKey, icon: t.icon, title: t.title, complexity: t.complexity,
            progress: t.progress || 0, quality: t.quality || RL_CFG.QUALITY_BASE,
            done: false, doneAt: 0, round: t.round || 1,
            contributors: Array.isArray(t.contributors) ? t.contributors : [],
          };
        })
      : COMPANY_TASKS.map(t => ({
          posKey: t.posKey, icon: t.icon, title: t.title, complexity: t.complexity,
          progress: 0, quality: RL_CFG.QUALITY_BASE, done: false, doneAt: 0, round: 1,
          contributors: [],
        }));
    // 已在岗员工：绑定专属任务 / 精力 / 独立 RL 策略（存档恢复时调用）
    for (const h of this.hiredList) {
      if (!h.actor) continue;
      const savedE = (savedEnergy && savedEnergy[h.name] != null) ? savedEnergy[h.name] : null;
      const savedQ = (savedRL && savedRL[h.name]) ? savedRL[h.name] : null;
      this._initAgentRL(h.name, h.position.key, savedE, savedQ);
    }
  }

  /** 员工入职绑定：同岗位共享看板任务项 + 独立精力槽 + 独立 RL 策略 */
  _initAgentRL(name, posKey, savedEnergy, savedQ) {
    this._bindTask(name, posKey);
    if (this._agentEnergy[name] == null) {
      this._agentEnergy[name] = (savedEnergy != null)
        ? Math.max(0, Math.min(RL_CFG.ENERGY_MAX, savedEnergy))
        : 80 + Math.floor(Math.random() * 20);            // 新员工精力 80-100
    }
    if (!this._rl[name]) {
      const rl = new SwarmRLAgent(name, posKey);
      if (savedQ) rl.restore(savedQ);
      this._rl[name] = rl;
    }
    if (this._rlNextAct[name] == null) {
      this._rlNextAct[name] = Date.now() + 3000 + Math.random() * 6000;   // 入职 3-9 秒后首次行动
    }
    if (this._nextErrandAt[name] == null) {
      this._nextErrandAt[name] = Date.now() + 15000 + Math.random() * 30000;  // 入职后先熟悉工位，再开始走动
    }
    if (this._nextReportAt[name] == null) {
      const [rLo, rHi] = SWARM.PROACTIVE_REPORT;
      this._nextReportAt[name] = Date.now() + (rLo + Math.random() * (rHi - rLo)) * 1000;
    }
  }

  /** 把员工绑定到看板任务：优先未完成任务，该岗位任务已完成则开新一轮 */
  _bindTask(name, posKey) {
    // 项目优先：CEO 布置的项目运行中时，该岗位认领对应子任务（各司其职）
    if (this._playerTask && (this._playerTask.status === 'planning' || this._playerTask.status === 'running')) {
      const st = this._playerTask.subtasks.find(s => s.posKey === posKey && !s.done);
      if (st) {
        if (!st.contributors) st.contributors = [];
        if (!st.contributors.includes(name)) st.contributors.push(name);
        return st;
      }
    }
    let task = this._tasks.find(t => t.posKey === posKey && !t.done);
    if (!task) task = this._tasks.find(t => t.posKey === posKey);
    if (!task) task = this._tasks[0];
    this._agentTask[name] = task;
    if (!task.contributors) task.contributors = [];
    if (!task.contributors.includes(name)) task.contributors.push(name);
    return task;
  }

  /** 任务完成：产出分入库（质量 ≥70 拿满额）+ 开新一轮任务，员工无缝续工
   *  项目子任务完成 → 推进 CEO 项目；项目全部完成 → 报喜并回到公司任务 */
  _completeTask(emp, task) {
    task.done = true;
    task.doneAt = this._worldTick;
    const isProject = task.projectId && this._playerTask
      && this._playerTask.subtasks.indexOf(task) >= 0;
    const bonus = task.quality >= 70 ? 2.5 : RL_CFG.PRODUCTION_DONE;
    const before = this.stats.production || 0;
    this.stats.production = Math.max(0, Math.min(RL_CFG.PRODUCTION_CAP, before + bonus));
    const gain = Math.round((this.stats.production - before) * 10) / 10;
    this._appendTranscript('system', `🏁 ${emp.name} 完成「${task.title}」· 产出 +${gain}（累计 ${this.stats.production}/${RL_CFG.PRODUCTION_CAP}）`);
    this._setStatus(`🏁 ${emp.name} 完成任务「${task.title}」，公司产出 +${gain}`);
    this._recalcScore();
    if (isProject) {
      // 项目子任务：不滚动看板，检查整个项目是否收官
      this._appendTranscript('system', `📦 CEO 项目「${this._playerTask.title}」子任务完成：${task.title}`);
      if (this._playerTask.subtasks.every(s => s.done)) {
        this._finishPlayerTask();
      }
    } else {
      this._rollTask(task);      // 先滚动看板，再存档：确保新任务一并入库
    }
    this._saveGame();
  }

  /** CEO 项目收官：全项目完成 → 公司声誉/产出奖励 + 全员简短庆祝，员工回到公司任务 */
  async _finishPlayerTask() {
    const pt = this._playerTask;
    if (!pt || pt.status === 'done') return;
    // 先同步收尾状态：项目完成 → 评分入库 → 员工回到公司任务 → 存档
    pt.status = 'done';
    pt.doneAt = Date.now();
    const qualitySubs = pt.subtasks.filter(s => s.quality >= 70).length;
    this.stats.reputation = Math.min(BALANCE.REPUTATION_FULL, this.stats.reputation + 1);
    this.stats.production = Math.min(RL_CFG.PRODUCTION_CAP, this.stats.production + 1);
    this._recalcScore();
    this._appendTranscript('system', `🎉 CEO 项目「${pt.title}」全部完成！${qualitySubs}/${pt.subtasks.length} 个子任务质量达标，公司声誉 +1`);
    this._setStatus(`🎉 项目「${pt.title}」完成！蜂群全员收工报喜`);
    this._drawHoloScreen('🎉 项目完成',
      `CEO ${this.ceo ? this.ceo.ceo_name : ''} · ${pt.title}`,
      `${pt.subtasks.length} 个子任务全部交付 · 质量达标 ${qualitySubs} 项`, '#00ffa3', '#00e5ff');
    // 项目结束 → 员工回到公司任务看板继续日常 RL 工作
    for (const h of this.hiredList) this._bindTask(h.name, h.position.key);
    this._saveGame();
    // 全员简短庆祝（每人一句，不阻塞世界主循环太久）
    const hired = this.hiredList.filter(h => h.actor && !h.actor.removed);
    for (const h of hired) {
      if (this._disposed) break;
      const line = `「${pt.title}」交付了！${h.position.icon} ${h.position.name}这边收工。`;
      await this._swarmSpeak(h.actor, line, 'report');
    }
  }

  /** 看板任务滚动到下一轮（复杂度递增，员工 RL 经验跨轮保留） */
  _rollTask(task) {
    const idx = this._tasks.indexOf(task);
    if (idx < 0) return;
    const def = COMPANY_TASKS.find(t => t.posKey === task.posKey) || COMPANY_TASKS[0];
    const round = (task.round || 1) + 1;
    const next = {
      posKey: def.posKey, icon: def.icon,
      title: `${def.title} · R${round}`,
      complexity: Math.round(def.complexity * (1 + (round - 1) * 0.15)),
      progress: 0, quality: RL_CFG.QUALITY_BASE, done: false, doneAt: 0, round,
      contributors: [],
    };
    this._tasks[idx] = next;
    // 原任务的所有协作者自动接手新一轮
    for (const h of this.hiredList) {
      if (h.position.key === task.posKey && this._agentTask[h.name] === task) {
        this._agentTask[h.name] = next;
        next.contributors.push(h.name);
      }
    }
  }

  /** 裁员清理：移除该员工的 RL 策略 / 精力 / 任务绑定，任务协作者列表同步 */
  _cleanupAgentRL(name) {
    const task = this._agentTask[name];
    if (task && task.contributors) {
      task.contributors = task.contributors.filter(n => n !== name);
      if (task.projectId && this._playerTask) {
        // 项目子任务协作者清空后允许其他同事接手（重新绑定）
        if (!task.contributors.length && !task.done) {
          const st = this._playerTask.subtasks.find(s => s === task);
          if (st && this.hiredList.some(h => h.position.key === st.posKey && h.name !== name)) {
            const h2 = this.hiredList.find(h => h.position.key === st.posKey && h.name !== name);
            if (h2) this._agentTask[h2.name] = st;
          }
        }
      }
    }
    delete this._agentTask[name];
    delete this._agentEnergy[name];
    delete this._rl[name];
    delete this._rlNextAct[name];
    delete this._lastRewardSign[name];
    delete this._swarmCooldown[name];
    delete this._reviewPending[name];
    delete this._forceRefine[name];
    delete this._lastReviewAt[name];
    delete this._agentProgress[name];
    delete this._stuckStreak[name];
    delete this._nextErrandAt[name];
    delete this._meetingSpots[name];
    delete this._nextReportAt[name];
  }

  // ==================== RL 决策链路（观察 → 决策 → 执行 → 学习） ====================

  /** 归一化分桶：值 v 映射到第 i 个特征的 0..(n-1) 桶 */
  _bucket(v, i) {
    const n = RL_CFG.OBS_BUCKETS[i];
    const r = RL_CFG.OBS_RANGE[i];
    let idx = r > 0 ? Math.floor(v / (r / n)) : 0;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return idx;
  }

  /** 观察状态：[任务忙度, 个人进度, 精力, 上次奖励符号] → 离散串 */
  _obsState(name) {
    const busy = this._tasks ? this._tasks.filter(t => !t.done && t.progress > 0).length : 0;
    const task = this._agentTask[name];
    const prog = task ? Math.min(100, task.progress) : 0;
    const energy = (this._agentEnergy[name] != null) ? this._agentEnergy[name] : 50;
    const sign = (this._lastRewardSign[name] || 0) > 0 ? 1 : 0;
    return `${this._bucket(busy, 0)}${this._bucket(prog, 1)}${this._bucket(energy, 2)}${this._bucket(sign, 3)}`;
  }

  /** 一次完整 RL 决策：员工独立策略选动作 → 执行（异步，可能说话）→ 学习更新 */
  async _agentDecide(emp) {
    const name = emp.name;
    if (!this._rl[name]) this._rl[name] = new SwarmRLAgent(name, emp.position.key);
    const rl = this._rl[name];
    const task = this._agentTask[name] || this._bindTask(name, emp.position.key);
    const state = this._obsState(name);
    // 评审否决 → 强制打磨：下一次行动必须 refine（返工整改），不听从 RL 习惯
    let action = rl.choose(state);
    if (this._forceRefine[name]) {
      action = RL_CFG.ACTIONS.findIndex(a => a.key === 'refine');
      delete this._forceRefine[name];
    }
    const before = {
      progress: task.progress,
      quality: task.quality,
      energy: this._agentEnergy[name] != null ? this._agentEnergy[name] : 80,
      done: !!task.done,
    };
    const say = await this._executeAction(emp, task, action, before);
    const reward = this._calcReward(task, action, before);
    const nextState = this._obsState(name);
    rl.learn(state, action, reward, nextState);
    this._lastRewardSign[name] = reward;
    this._rlLearnCount++;
    // 周期性落盘：RL 学习成果（Q 表）跨会话保留，避免学习进度丢失
    if (this._rlLearnCount % RL_CFG.SAVE_EVERY === 0) this._saveGame();
    return { say, action: RL_CFG.ACTIONS[action] ? RL_CFG.ACTIONS[action].key : 'work' };
  }

  /** 执行 RL 动作：更新任务进度/质量与员工精力；返回需要"开口说话"的台词（其余静默） */
  async _executeAction(emp, task, actionIdx, before) {
    const a = RL_CFG.ACTIONS[actionIdx];
    const name = emp.name;
    const low = before.energy < RL_CFG.ENERGY_LOW;
    const eff = low ? 0.5 : 1.0;                       // 低精力工作效率减半
    let energy = before.energy + a.energy;
    if (energy < 0) energy = 0;
    if (energy > RL_CFG.ENERGY_MAX) energy = RL_CFG.ENERGY_MAX;
    this._agentEnergy[name] = energy;

    const rnd = (arr) => Math.round(arr[0] + Math.random() * (arr[1] - arr[0]));
    let say = null;

    switch (a.key) {
      case 'work': {
        const gain = Math.round(rnd(a.progress) * eff);
        task.progress = Math.min(task.complexity, task.progress + gain);
        break;
      }
      case 'refine': {
        task.progress = Math.min(task.complexity, task.progress + Math.round(rnd(a.progress) * eff));
        task.quality = Math.min(100, task.quality + rnd(a.quality));
        break;
      }
      case 'assist': {
        task.progress = Math.min(task.complexity, task.progress + Math.round(rnd(a.progress) * eff));
        // 协作加成：随机给一位同事的任务小幅助力（体现互帮互助）
        const others = this.hiredList.filter(h =>
          h.name !== name && this._agentTask[h.name] && !this._agentTask[h.name].done);
        if (others.length) {
          const t2 = this._agentTask[others[Math.floor(Math.random() * others.length)].name];
          t2.progress = Math.min(t2.complexity, t2.progress + Math.round(2 + Math.random() * 3));
        }
        break;
      }
      case 'report': {
        task.progress = Math.min(task.complexity, task.progress + rnd(a.progress));
        // 汇报前先走到 CEO（玩家）此刻落脚点附近，贴近汇报（避障失败则原地汇报）
        const ceoPos = this._playerPos();
        const gen = (emp.actor._moveGen || 0) + 1;
        const spot = this._nearSpot(ceoPos, 1.6);
        if (spot) await this._moveActor(emp.actor, [spot]);
        if (!this._disposed && emp.actor && !emp.actor.removed
          && emp.actor._moveGen === gen && ceoPos && ceoPos.x != null) {
          emp.actor.faceTarget = { x: ceoPos.x, z: ceoPos.z };
        }
        // 向 CEO 汇报：有台词，走蜂群演绎（LLM 优先，失败回退内置文案）
        let line = null;
        if (!this._llmDisabled()) {
          const context = `你刚完成了一次 RL 决策（向 CEO 汇报）。请作为「${emp.actor.name}」（${emp.position.name}）简短汇报当前在做的事。口语化、不超过40字，以「CEO ${this.ceo ? this.ceo.ceo_name : ''}」开头。`;
          const data = await this._requestLine(this._promptForActor(emp.actor), context, '', '').catch(() => null);
          if (data && data.text) line = data.text;
        }
        if (!line) {
          line = this._pickFallback(RL_CFG.ACTION_FALLBACK[3])
            .replace('{ceo}', this.ceo ? this.ceo.ceo_name : '老板')
            .replace('{pos}', emp.position.name)
            .replace('{task}', task.title);
        }
        say = line;
        break;
      }
      case 'rest': {
        // 休整充电：静默恢复精力（可能顺势说一句状态）
        if (Math.random() < 0.15) {
          say = this._pickFallback(RL_CFG.ACTION_FALLBACK[4]);
        }
        break;
      }
      default: break;
    }

    // 任务完成检测：完成时必然开口报捷
    if (!before.done && task.progress >= task.complexity && !task.done) {
      this._completeTask(emp, task);
      say = `「${task.title}」完成了！质量 ${task.quality} 分，产出已交给公司。`;
    } else if (!task.done && (a.key === 'work' || a.key === 'refine')) {
      // 交付评审门禁：进度接近完成但质量不足 → 挂起评审（由蜂群运行时触发讨论）
      const pct = task.progress / Math.max(1, task.complexity);
      const now = Date.now();
      const lastReview = this._lastReviewAt[name] || 0;
      if (pct >= PROJECT.REVIEW_AT && task.quality < PROJECT.REVIEW_QUALITY
        && !this._reviewPending[name] && (now - lastReview) >= PROJECT.REVIEW_COOLDOWN * 1000) {
        this._reviewPending[name] = true;
      }
    }
    // 非汇报动作 10% 概率随口说一句工作状态（保持世界鲜活而不刷屏）
    if (!say && actionIdx !== 4 && Math.random() < 0.1) {
      say = this._pickFallback(RL_CFG.ACTION_FALLBACK[actionIdx])
        .replace('{ceo}', this.ceo ? this.ceo.ceo_name : '老板')
        .replace('{pos}', emp.position.name)
        .replace('{task}', task.title);
    }
    return say;
  }

  /** RL 奖励：进度/质量增量 + 低精力惩罚 + 任务完成大奖 */
  _calcReward(task, actionIdx, before) {
    let r = 0;
    const a = RL_CFG.ACTIONS[actionIdx];
    const dProg = task.progress - before.progress;
    const dQ = task.quality - before.quality;
    if (a.key === 'work') r += 0.3 + dProg * 0.05;
    else if (a.key === 'refine') r += 0.2 + dQ * 0.12 + dProg * 0.04;
    else if (a.key === 'assist') r += 0.25 + dProg * 0.05;
    else if (a.key === 'report') r += 0.15 + dProg * 0.05;
    else if (a.key === 'rest') r += 0.08 + (before.energy < RL_CFG.ENERGY_LOW ? 0.25 : 0);
    if (before.energy < RL_CFG.ENERGY_LOW && (a.key === 'work' || a.key === 'refine')) r -= 0.35;
    if (!before.done && task.done) r += 2.0;          // 完成奖励（大）
    return Math.round(r * 10) / 10;
  }

  /** 世界存档键（世界隔离：每 CEO 一个世界，R1） */
  _worldSaveKey() {
    let w = this.worldId;
    if (!w) {
      try { w = localStorage.getItem(LAST_WORLD_KEY); } catch (e) { w = null; }
    }
    return SAVE_KEY_PREFIX + (w || 'default');
  }

  /** 身份注入块（R4）：CEO 身份作为世界上下文第一等公民，注入每个智能体推理 */
  _ceoContextBlock() {
    if (!this.ceo) return '';
    const c = this.ceo;
    let s = `你的老板是赛博公司 CEO「${c.ceo_name}」（真实玩家 · ID ${c.ceo_id}）。公司：「${c.company_name}」。你很清楚 CEO 的身份，始终以他为服务对象。`;
    if (c.motto) s += ` 公司理念：${c.motto}。`;
    return s;
  }

  // ==================== 存档机制 ====================

  _saveGame() {
    try {
      if (!this.hrActor) return;
      const save = {
        v: 3,
        savedAt: Date.now(),
        worldId: this.worldId,
        ceo: this.ceo,
        hrCard: this.hrCard,
        hiredList: this.hiredList.map(h => ({
          positionKey: h.position.key,
          positionName: h.position.name,
          name: h.name,
          card: (h.actor && h.actor.card) || null,
          fitScore: h.fitScore || 0,
        })),
        filledPositions: Array.from(this.filledPositions),
        score: this.score || 0,
        stats: {
          vitality: this.stats.vitality,
          quality: this.stats.quality,
          stability: this.stats.stability,
          reputation: this.stats.reputation,
          production: this.stats.production || 0,
          score: this.stats.score,
          grade: this.stats.grade,
          layoffCount: this.stats.layoffCount,
          stalePeriods: this.stats.stalePeriods,
        },
        // RL 自主决策体系（世界隔离持久化）：任务看板 + 员工精力 + 独立 Q 表
        tasks: (this._tasks || []).map(t => ({
          posKey: t.posKey, icon: t.icon, title: t.title, complexity: t.complexity,
          progress: t.progress, quality: t.quality, done: t.done, doneAt: t.doneAt || 0,
          round: t.round || 1, contributors: t.contributors || [],
        })),
        // CEO 项目任务（玩家布置 → 蜂群拆解）：子任务进度/质量/协作者随世界存档
        playerTask: this._playerTask ? {
          id: this._playerTask.id,
          title: this._playerTask.title,
          status: this._playerTask.status,
          createdAt: this._playerTask.createdAt || Date.now(),
          doneAt: this._playerTask.doneAt || 0,
          votes: this._playerTask.votes || [],
          subtasks: (this._playerTask.subtasks || []).map(s => ({
            posKey: s.posKey, icon: s.icon || '📋', title: s.title,
            complexity: s.complexity, progress: s.progress, quality: s.quality,
            done: s.done, doneAt: s.doneAt || 0,
            contributors: s.contributors || [],
            projectId: s.projectId || this._playerTask.id,
          })),
        } : null,
        agentEnergy: Object.assign({}, this._agentEnergy || {}),
        rl: Object.fromEntries(
          Object.entries(this._rl || {}).map(([n, a]) => [n, a.snapshot()])
        ),
        playerPos: (this._ghostAnchor && this._ghostAnchor.position)
          ? { x: this._ghostAnchor.position.x, z: this._ghostAnchor.position.z }
          : (this.App.currentAvatar
            ? { x: this.App.currentAvatar.position.x, z: this.App.currentAvatar.position.z }
            : null),
        // 招聘面试状态（R5）：恢复后可继续面试，不被拒角色不再重复面试
        seekerState: this.seekerQueue.map(s => ({
          id: (s.card && (s.card.id || s.card.name)) || '',
          used: !!s.used,
        })),
        interviewingCardId: (this.currentSeeker && this.currentSeeker.card
          && (this.currentSeeker.card.id || this.currentSeeker.card.name)) || null,
      };
      localStorage.setItem(this._worldSaveKey(), JSON.stringify(save));
    } catch (e) {
      console.warn('[赛博公司] 存档失败:', e?.message || e);
    }
  }

  _loadSave() {
    try {
      const raw = localStorage.getItem(this._worldSaveKey());
      if (!raw) return null;
      const save = JSON.parse(raw);
      if (!save || !save.hrCard) return null;
      return save;
    } catch (e) {
      return null;
    }
  }

  _clearSave() {
    try { localStorage.removeItem(this._worldSaveKey()); } catch (e) {}
  }

  /** 有存档时进入游戏前的选择界面：继续经营 / 新游戏 */
  _showResumeDialog() {
    const save = this._loadSave();
    if (!save) { this._showHRSelectUI(this.cards); return; }
    const root = document.createElement('div');
    root.className = 'cyber-hr-select';
    const t = new Date(save.savedAt || Date.now());
    const ts = `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, '0')}-${String(t.getDate()).padStart(2, '0')} ${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}`;
    root.innerHTML = `
      <div class="cyber-hr-panel" style="max-width:500px;text-align:center;">
        <div class="cyber-hr-logo">CYBER CORP · 世界存档</div>
        <h3 class="cyber-hr-title">💾 检测到经营存档</h3>
        <p class="cyber-hr-sub" style="text-align:center;">
          CEO：${this.App.escapeHtml(save.ceo?.ceo_name || '—')} · 世界 ${this.App.escapeHtml(save.worldId || '—')}<br>
          ${this.App.escapeHtml(save.ceo?.company_name || '')} · HR：${this.App.escapeHtml(save.hrCard?.name || '—')} · 在职员工：${(save.hiredList || []).length} 人<br>
          存档时间：${ts}
        </p>
        <div style="display:flex;gap:14px;justify-content:center;margin-top:6px;">
          <button class="cyber-hr-random" id="cc-resume" style="margin:0;">▶ 继续经营</button>
          <button class="cyber-hr-random" id="cc-newgame" style="margin:0;opacity:.7;">🔄 新世界</button>
        </div>
      </div>
    `;
    document.body.appendChild(root);
    this._hrSelectUI = root;
    root.querySelector('#cc-resume').addEventListener('click', () => {
      document.body.removeChild(root);
      this._hrSelectUI = null;
      this._resumeFromSave(save).catch(err => {
        console.error('[赛博公司] 恢复存档失败，回退到 HR 任命:', err);
        // 恢复失败兜底：清理半初始化的状态，回到 HR 选择界面（不卡死）
        try { this._removeUI(); } catch (e) {}
        this._resetRuntimeFlags();
        this._showHRSelectUI(this.cards);
      });
    });
    root.querySelector('#cc-newgame').addEventListener('click', () => {
      document.body.removeChild(root);
      this._hrSelectUI = null;
      this._clearSave();
      this._startCEOOnboarding();   // 新世界 = 新 CEO 身份（R2）
    });
  }

  /** 从存档恢复游戏：重建 HR 与在职员工，跳过任命，直接进入招聘循环 */
  async _resumeFromSave(save) {
    if (this._disposed) return;
    const card = save.hrCard;
    if (!card) { this._clearSave(); this._showHRSelectUI(this.cards); return; }
    this.hrCard = card;
    // 恢复世界身份（R1/R3）：世界 ID 与 CEO 身份随存档恢复，员工继续认识 CEO
    if (save.ceo && save.worldId) {
      this.ceo = save.ceo;
      this.worldId = save.worldId;
    } else {
      this._initWorldIdentity('赛博董事长', '赛博公司', '');
    }
    this.phase = 'running';
    this._skipWelcome = true;      // 员工已在岗，恢复后跳过 HR 开场白

    // HR
    const hr = new CharacterActor(this, card, 'hr');
    this.actors.push(hr);
    hr.load().catch(err => console.warn('[赛博公司] HR 模型加载失败，使用全息替身:', err?.message || err));
    this._playerControlledHR = false;
    this.fpvCamera = true;
    this.fpvEyeHeight = 1.5;
    if (this.App.currentAvatar) {
      // 保护式保存：generateScene 已保存过原始可见性，不覆盖（否则退出后无法恢复大厅角色）
      if (this._savedAvatarVisible === undefined) this._savedAvatarVisible = this.App.currentAvatar.visible;
      this.App.currentAvatar.visible = false;
      const px = (save.playerPos && typeof save.playerPos.x === 'number') ? save.playerPos.x : LAYOUT.playerSpawn.x;
      const pz = (save.playerPos && typeof save.playerPos.z === 'number') ? save.playerPos.z : LAYOUT.playerSpawn.z;
      this.App.currentAvatar.position.set(px, STAND_Y, pz);
      // 幽灵玩家系统：恢复无实体锚点位置（玩家坐标载体）
      if (this._ghostAnchor) {
        this._ghostAnchor.position.set(px, STAND_Y, pz);
      }
    }
    this.App._gameCamAzimuth = 0;
    this.App._gameCamPitch = 0;
    hr.setPosition(LAYOUT.stage.hr.x, LAYOUT.stage.hr.z);
    hr.faceTarget = { x: LAYOUT.stage.seeker.x, z: LAYOUT.stage.seeker.z };
    this.hrActor = hr;
    this._hookPlayerChat();

    // 求职者队列（排除 HR 卡片与已入职员工，避免重复面试）
    const hiredKeys = new Set((save.hiredList || []).map(h => (h.card && (h.card.id || h.card.name)) || h.name));
    // 恢复面试历史：已面试过（被拒/作废）的候选人标记 used，不再重复面试
    const usedSeekerIds = new Set((save.seekerState || []).filter(s => s.used).map(s => s.id));
    this.seekerQueue = this.cards
      .filter(c => (c.id ? c.id !== card.id : c.name !== card.name))
      .filter(c => !hiredKeys.has(c.id || c.name))
      .map(c => ({
        card: c, actor: null, loaded: false, standPos: LAYOUT.waitingSpot,
        used: usedSeekerIds.has(c.id || c.name),
      }));

    // 面试中断恢复：上次退出时正在面试的候选人放回队列头部，重新开始面试（不丢失）
    const interviewingId = save.interviewingCardId || null;
    if (interviewingId) {
      const idx = this.seekerQueue.findIndex(s => (s.card.id || s.card.name) === interviewingId);
      if (idx >= 0) {
        const cur = this.seekerQueue.splice(idx, 1)[0];
        cur.used = false;   // 该候选人重新面试（从开场重新问起）
        this.seekerQueue.unshift(cur);
        this._resumeInterviewNotice = `🔁 上次退出时正在面试「${cur.card.name}」，已恢复继续`;
      }
    }

    this.filledPositions.add('hr');
    this.score = save.score || 0;
    // 恢复评分系统状态（旧存档缺失时按默认值重建）
    const ss = save.stats || {};
    this.stats.vitality = ss.vitality || 0;
    this.stats.quality = ss.quality || 0;
    this.stats.stability = ss.stability || 0;
    this.stats.reputation = typeof ss.reputation === 'number' ? ss.reputation : BALANCE.REPUTATION_FULL;
    this.stats.production = typeof ss.production === 'number' ? ss.production : 0;
    this.stats.score = ss.score || this.score || 0;
    this.stats.grade = ss.grade || this._gradeOf(this.stats.score).grade;
    this.stats.layoffCount = ss.layoffCount || 0;
    this.stats.stalePeriods = ss.stalePeriods || 0;

    // 重建在职员工（按岗位错位站位）
    const posCount = {};
    for (const h of (save.hiredList || [])) {
      const pos = POSITIONS.find(p => p.key === h.positionKey)
        || { key: h.positionKey, name: h.positionName || '员工', icon: '💼', color: '#00e5ff' };
      const empCard = h.card;
      if (!empCard) continue;
      const emp = new CharacterActor(this, empCard, 'employee');
      emp.hiredPos = pos;
      this.actors.push(emp);
      emp.load().catch(err => console.warn('[赛博公司] 员工模型加载失败，使用全息替身:', err?.message || err));
      const ws = LAYOUT.workstations.find(w => w.posKey === h.positionKey);
      if (ws) {
        const n = posCount[h.positionKey] || 0;
        posCount[h.positionKey] = n + 1;
        const offset = Math.min(n, 2) * 0.9;
        emp.setPosition(ws.x + offset, ws.z + offset);
        emp.faceTarget = this._workFacePos({ x: ws.x + offset, z: ws.z + offset });   // 面朝大厅
      } else {
        emp.setPosition(LAYOUT.waitingSpot.x, LAYOUT.waitingSpot.z);
      }
      this.hiredList.push({ position: pos, name: emp.name, actor: emp, fitScore: h.fitScore || 0 });
      this.filledPositions.add(h.positionKey);
    }

    // 恢复员工独立对话记录 + RL 任务体系（任务看板 / 精力 / 独立 Q 表，世界隔离）
    this._loadChats();
    this._initTaskBoard(save.tasks, save.agentEnergy, save.rl, save.playerTask);

    // 构建游戏 UI 并启动招聘循环（恢复时强制重建：旧 UI 可能已在退出时被清理，
    // 若引用残留则 _buildMainUI 的防重入会跳过重建，导致无聊天框/无状态面板）
    if (this._uiRoot && this._uiRoot.parentNode) this._uiRoot.parentNode.removeChild(this._uiRoot);
    this._uiRoot = null;
    this._chatPanelEl = null;
    this._buildMainUI();
    this._updatePositionsUI();
    this._setStatus(`💾 已恢复世界 ${this.worldId || '—'} · CEO ${this.ceo?.ceo_name || '—'} · HR ${hr.name} · 在职 ${this.hiredList.length} 人`);
    this._drawHoloScreen('赛博公司 · 世界运行中',
      `CEO: ${this.ceo?.ceo_name || '—'} · ${this.ceo?.company_name || ''}`,
      `世界 ${this.worldId || '—'} · 在职 ${this.hiredList.length} 人 · 蜂群运行中`, '#00e5ff', '#ff2bd6');
    this._appendTranscript('system', `💾 已恢复世界 ${this.worldId || '—'}：CEO ${this.ceo?.ceo_name || '—'}，HR ${hr.name}，在职员工 ${this.hiredList.length} 人`);
    this._appendTranscript('system', `🎧 空间音频已开启：靠近说话者 10 米内可听到语音，越远越小声；走远后气泡仍会显示，只是听不到声音。`);
    if (this._resumeInterviewNotice) {
      this._appendTranscript('system', this._resumeInterviewNotice);
      this._resumeInterviewNotice = null;
    }

    this._running = true;
    // 主循环防重入已由 _runGame 的 try/finally 保证：任何退出路径都会复位 _gameLoopRunning，
    // 此处无需再处理旧循环残留；_runGame 内若标志残留会直接 return（不会双循环竞态）
    this._runGame().catch(err => console.error('[赛博公司] 主循环异常:', err));
    this._ensureSwarm();   // 恢复后启动蜂群运行时
  }

  _showHRSelectUI(cards) {
    const root = document.createElement('div');
    root.className = 'cyber-hr-select';
    let mode = 'interview';   // 'interview' 面试招聘 / 'direct' CEO 直聘（跳过面试）
    root.innerHTML = `
      <div class="cyber-hr-panel">
        <div class="cyber-hr-logo">赛博公司 · CYBERCORP</div>
        <h3 class="cyber-hr-title">⚡ 任命人力资源总监 HR</h3>
        <p class="cyber-hr-sub">从角色卡片中任命一位角色担任 HR，她/他将坐在面试桌后主持面试（面试全程公开透明，入职员工分配到对应工位）。<br>你是公司 CEO（${this.ceo ? this.App.escapeHtml(this.ceo.ceo_name) : '玩家'}）：第一人称自由游走，靠近任何人 10 米内说话，他/她都知道你是老板并回应你。</p>
        <div class="cyber-hr-modes">
          <button type="button" class="cyber-hr-mode active" data-mode="interview">🎤 面试招聘<br><span>逐个面试后录用</span></button>
          <button type="button" class="cyber-hr-mode" data-mode="direct">⚡ 跳过面试 · CEO 直接选人<br><span>开局跳过面试，直接点选入职</span></button>
        </div>
        <div class="cyber-hr-grid">
          ${cards.length ? cards.map((c, i) => `
            <div class="cyber-hr-card" data-i="${i}">
              <div class="cyber-hr-ava">${this.App.escapeHtml((c.name || '?').slice(0, 1))}</div>
              <div class="cyber-hr-name">${this.App.escapeHtml(c.name || '未知')}</div>
              <div class="cyber-hr-role">${this.App.escapeHtml(c.role_name || '')}</div>
              <div class="cyber-hr-model">${this.App.escapeHtml((c.model_name || '默认外形').replace(/\.[a-z]+$/i, ''))}</div>
            </div>
          `).join('') : '<div class="cyber-hr-empty">暂无角色卡片，请先在大厅创建角色卡片</div>'}
        </div>
        ${cards.length ? '<button class="cyber-hr-random">🎲 随机任命</button>' : ''}
      </div>
    `;
    document.body.appendChild(root);
    this._hrSelectUI = root;

    const setMode = (m) => {
      mode = m;
      root.querySelectorAll('.cyber-hr-mode').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
    };
    root.querySelectorAll('.cyber-hr-mode').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));

    const pick = (card) => {
      if (!card) return;
      document.body.removeChild(root);
      this._hrSelectUI = null;
      this._onHRSelected(card, mode).catch(err => console.error('[赛博公司] HR 任命失败:', err));
    };
    root.querySelectorAll('.cyber-hr-card').forEach(el => {
      el.addEventListener('click', () => pick(cards[+el.dataset.i]));
    });
    const rnd = root.querySelector('.cyber-hr-random');
    if (rnd) rnd.addEventListener('click', () => pick(cards[Math.floor(Math.random() * cards.length)]));
  }

  async _onHRSelected(card, mode) {
    if (this._disposed) return;
    this.hrCard = card;
    this._directHireMode = (mode === 'direct');
    this.phase = 'running';

    // 创建 HR 角色：load() 的同步段会立即创建全息替身并加入场景，
    // 真实模型后台加载完成后无缝替换。不 await 完整加载，避免模型下载慢时画面长时间无反馈。
    const hr = new CharacterActor(this, card, 'hr');
    this.actors.push(hr);
    hr.load().catch(err => console.warn('[赛博公司] HR 模型加载失败，使用全息替身:', err?.message || err));

    // ===== 玩家独立成"幽灵"：无实体、第一人称 =====
    // HR 是完全独立的 AI 角色，坐在面试桌后主持；玩家不再操控任何角色身体。
    // 大厅角色（avatar）隐藏但其位置作为玩家的世界坐标载体（幽灵），
    // 游戏模式管理器移动它、第一人称相机跟随它，玩家以第一人称自由游走。
    this._playerControlledHR = false;
    this.fpvCamera = true;                  // 通知游戏模式管理器使用第一人称相机
    this.fpvEyeHeight = 1.5;                // 幽灵漂浮高度（略高于真人视角）
    if (this.App.currentAvatar) {
      // 保护式保存：generateScene 已保存过原始可见性，不覆盖
      if (this._savedAvatarVisible === undefined) this._savedAvatarVisible = this.App.currentAvatar.visible;
      this.App.currentAvatar.visible = false;
      this.App.currentAvatar.position.set(LAYOUT.playerSpawn.x, STAND_Y, LAYOUT.playerSpawn.z);
    }
    // 初始视角：面向办公室（-Z 方向，舞台/工位都在前方）
    this.App._gameCamAzimuth = 0;
    this.App._gameCamPitch = 0;

    // HR 独立站位：面试桌后方（不再跟随玩家移动）
    hr.setPosition(LAYOUT.stage.hr.x, LAYOUT.stage.hr.z);
    hr.faceTarget = { x: LAYOUT.stage.seeker.x, z: LAYOUT.stage.seeker.z };
    this.hrActor = hr;

    // 玩家说话 → 身边 10 米内角色能听到（空间音频听力半径）
    this._hookPlayerChat();

    // 构建求职者队列（其余角色卡片；id 缺失时用 name 兜底判断，确保队列不为空）
    this.seekerQueue = this.cards
      .filter(c => (c.id ? c.id !== card.id : c.name !== card.name))
      .map(c => ({ card: c, actor: null, loaded: false, standPos: LAYOUT.waitingSpot, used: false }));

    // 记录 HR 岗位已由玩家任命
    const hrPos = POSITIONS.find(p => p.key === 'hr');
    this.filledPositions.add('hr');

    // 构建游戏 UI
    this._buildMainUI();
    this._setStatus('🏢 HR 已就位 · 招聘程序启动…');

    this._drawHoloScreen('赛博公司 · 招聘系统', `CEO: ${this.ceo?.ceo_name || '—'} · HR: ${hr.name}`,
      `${hrPos.icon} ${hrPos.name} 已由 CEO 任命 · 世界 ${this.worldId || '—'}`, '#00e5ff', '#ff2bd6');
    this._appendTranscript('system', `CEO「${this.ceo?.ceo_name || '玩家'}」任命「${hr.name}」担任人力资源总监（HR）`);

    // 启动面试主循环（防重入由 _runGame 的 try/finally 保证复位，无需手动处理）
    this._running = true;
    if (this._directHireMode) {
      // 直聘模式：跳过面试，CEO 直接选人入职
      this._setStatus('⚡ 直聘模式已开启 · CEO 请直接选人');
      this._appendTranscript('system', '⚡ 已开启直聘模式：跳过面试，由 CEO 直接选人入职');
      this._drawHoloScreen('赛博公司 · CEO 直聘', `CEO: ${this.ceo?.ceo_name || '—'} · HR: ${hr.name}`,
        '跳过面试 · 直接选人入职', '#ffd166', '#00e5ff');
      this._openDirectHire();
      return;
    }
    this._runGame().catch(err => console.error('[赛博公司] 主循环异常:', err));
  }

  // ==================== CEO 直聘（跳过面试，直接选人） ====================

  /** 依据角色信息猜测岗位适配度（直聘无面试，用关键词匹配 + 随机给出参考分） */
  _fitGuess(card, pos) {
    const text = `${card.role_name || ''} ${card.system_prompt || ''} ${card.name || ''}`.toLowerCase();
    const kw = {
      dev: ['开发', '程序', '码', '工程师', '后端', '前端', '工程'],
      artist: ['美术', '画', '设计', '视觉', '美工', '插画', '原画'],
      planner: ['策划', '玩法', '游戏设计', '文案'],
      pm: ['产品', '经理', '运营', '管理', '市场', '项目'],
      qa: ['测试', '质量', '质检', '验收', 'qa'],
      ai: ['算法', 'ai', '智能', '数据', '机器学习', '模型', '神经'],
    };
    let m = 0;
    for (const w of (kw[pos.key] || [])) if (text.includes(w)) m += 1;
    return Math.min(96, 66 + m * 6 + Math.floor(Math.random() * 9));
  }

  /** 直聘候选人池：真实角色卡片（除 HR）优先，用完后由合成候选人补位 */
  _directCandidatePool() {
    const hrId = this.hrCard ? this.hrCard.id : null;
    const hiredNames = new Set(this.hiredList.map(h => h.name));
    let pool = this.cards.filter(c =>
      (c.id ? c.id !== hrId : c.name !== (this.hrCard && this.hrCard.name)) && !hiredNames.has(c.name));
    if (!pool.length) {
      this._spawnFillerSeeker();
      pool = this.seekerQueue.filter(s => !s.used && s.card && !hiredNames.has(s.card.name)).map(s => s.card);
    }
    return pool;
  }

  /** 打开 CEO 直聘面板：选岗位 → 点候选人入职（无需面试） */
  _openDirectHire() {
    if (this._disposed) return;
    this._closeDirectHire();
    if (this.hiredList.length >= MAX_EMPLOYEES) {
      this._setStatus('🛑 员工已满员，暂无可招岗位');
      return;
    }
    const mask = document.createElement('div');
    mask.className = 'cc-direct-mask';
    mask.id = 'cc-direct-mask';
    mask.innerHTML = `
      <div class="cc-direct-panel">
        <div class="cc-direct-close" id="cc-direct-close" title="关闭面板（可稍后用右上角「⚡ 直聘」继续选人）">✕</div>
        <div class="cc-direct-head">⚡ CEO 直聘 · 跳过面试</div>
        <div class="cc-direct-sub">① 选择岗位 → ② 点选候选人直接入职（不面试，适配分仅供参考）</div>
        <div class="cc-direct-pos" id="cc-direct-pos"></div>
        <div class="cc-direct-cands" id="cc-direct-cands"></div>
        <div class="cc-direct-foot">
          <span class="cc-direct-count" id="cc-direct-count">—</span>
          <button type="button" class="cc-direct-done" id="cc-direct-done">✅ 完成选人 · 开始运营</button>
        </div>
      </div>
    `;
    (this._uiRoot || document.body).appendChild(mask);
    this._directMask = mask;
    this._directPosKey = null;
    this._directHiring = false;
    const close = mask.querySelector('#cc-direct-close');
    if (close) close.addEventListener('click', () => this._closeDirectHire());
    const done = mask.querySelector('#cc-direct-done');
    if (done) done.addEventListener('click', () => this._finishDirectHire());
    this._renderDirectPositions();
  }

  _closeDirectHire() {
    if (this._directMask && this._directMask.parentNode) {
      this._directMask.parentNode.removeChild(this._directMask);
    }
    this._directMask = null;
    this._directPosKey = null;
  }

  /** 渲染岗位选择条（空缺岗位优先；全部招满后显示全部岗位用于扩招） */
  _renderDirectPositions() {
    if (!this._directMask || this._disposed) return;
    const holder = this._directMask.querySelector('#cc-direct-pos');
    const open = POSITIONS.filter(p => p.key !== 'hr' && !this.filledPositions.has(p.key));
    const list = open.length ? open : POSITIONS.filter(p => p.key !== 'hr');
    if (!this._directPosKey || !list.some(p => p.key === this._directPosKey)) {
      this._directPosKey = list.length ? list[0].key : null;
    }
    holder.innerHTML = list.map(p => `
      <span class="cc-dp-chip ${p.key === this._directPosKey ? 'sel' : ''}" data-key="${p.key}"
        style="border-color:${p.color};color:${p.key === this._directPosKey ? '#0a0c18' : p.color};background:${p.key === this._directPosKey ? p.color : 'rgba(14,18,36,.9)'}">
        ${p.icon} ${p.name}${open.length ? '' : '（扩招）'}
      </span>
    `).join('') || '<div class="cc-direct-empty">暂无可选岗位</div>';
    holder.querySelectorAll('.cc-dp-chip').forEach(el => {
      el.addEventListener('click', () => {
        this._directPosKey = el.dataset.key;
        this._renderDirectPositions();
      });
    });
    this._renderDirectCount();
    this._renderDirectCandidates();
  }

  /** 渲染当前岗位的候选人卡片（按适配分从高到低排序） */
  _renderDirectCandidates() {
    if (!this._directMask || this._disposed) return;
    const holder = this._directMask.querySelector('#cc-direct-cands');
    const pos = POSITIONS.find(p => p.key === this._directPosKey);
    if (!pos) { holder.innerHTML = '<div class="cc-direct-empty">暂无岗位可选</div>'; return; }
    const pool = this._directCandidatePool().map(card => ({ card, fit: this._fitGuess(card, pos) }));
    pool.sort((a, b) => b.fit - a.fit);
    if (!pool.length) {
      holder.innerHTML = '<div class="cc-direct-empty">候选人已全部入职，请先招满其余岗位后刷新候选人池</div>';
      return;
    }
    holder.innerHTML = pool.map(({ card, fit }, i) => `
      <div class="cc-dc-card" data-i="${i}">
        <div class="cc-dc-ava">${this.App.escapeHtml((card.name || '?').slice(0, 1))}</div>
        <div class="cc-dc-info">
          <div class="cc-dc-name">${this.App.escapeHtml(card.name || '未知')}</div>
          <div class="cc-dc-role">${this.App.escapeHtml(card.role_name || '')}</div>
        </div>
        <span class="cc-dc-fit">适配 ${fit} 分</span>
        <button type="button" class="cc-dc-hire">⚡ 入职</button>
      </div>
    `).join('');
    holder.querySelectorAll('.cc-dc-card').forEach(el => {
      el.addEventListener('click', () => {
        const card = pool[+el.dataset.i].card;
        this._directHire(pos, card).catch(err => console.warn('[赛博公司] 直聘异常:', err?.message || err));
      });
    });
  }

  _renderDirectCount() {
    if (!this._directMask) return;
    const el = this._directMask.querySelector('#cc-direct-count');
    if (!el) return;
    const open = POSITIONS.filter(p => p.key !== 'hr' && !this.filledPositions.has(p.key)).length;
    const total = POSITIONS.length - 1;
    el.textContent = `已入职 ${this.hiredList.length} 人 · 待招岗位 ${open}/${total}`;
  }

  /** CEO 直聘单个候选人：创建角色 → 直接入职（复用 _hire 录用管线） */
  async _directHire(pos, card) {
    if (this._disposed || !pos || !card) return;
    if (this._directHiring) return;   // 防止连点并发重复入职
    if (this.hiredList.some(h => h.name === card.name)) return;
    if (this.hiredList.length >= MAX_EMPLOYEES) {
      this._setStatus('🛑 员工已满员，暂无可招岗位');
      return;
    }
    this._directHiring = true;
    try {
      const score = this._fitGuess(card, pos);
      const actor = new CharacterActor(this, card, 'seeker');
      this.actors.push(actor);
      actor.load().catch(() => {});
      actor.setPosition(LAYOUT.waitingSpot.x, LAYOUT.waitingSpot.z);
      actor.faceTarget = { x: LAYOUT.stage.seeker.x, z: LAYOUT.stage.seeker.z };
      this._appendTranscript('system', `⚡ CEO 直聘：「${card.name}」直接入职 ${pos.icon} ${pos.name}（适配 ${score} 分，跳过面试）`);
      this._setStatus(`⚡ ${card.name} 入职 ${pos.name}（直聘 · 适配 ${score} 分）`);
      await this._hire(pos, actor, { pass: true, score });
      if (this._disposed) return;
      const sq = this.seekerQueue.find(s => s.card && s.card.id === card.id);
      if (sq) sq.used = true;
      // 全部岗位招满 → 自动完成直聘
      const open = POSITIONS.filter(p => p.key !== 'hr' && !this.filledPositions.has(p.key));
      if (!open.length) {
        this._finishDirectHire();
      } else {
        this._renderDirectPositions();
      }
    } finally {
      this._directHiring = false;
    }
  }

  /** 完成直聘：进入运营模式，启动蜂群并落盘 */
  _finishDirectHire() {
    this._closeDirectHire();
    if (this._disposed) return;
    this._enterOperationMode();
    this._ensureSwarm();
    this._saveGame();
    this._drawScoreDashboard(2400).catch(() => {});
    this._appendTranscript('system', `✅ 直聘完成：在职 ${this.hiredList.length} 人，公司开始运营`);
  }

  // ==================== WebXR VR（仿大厅 VR 模式，第一人称沉浸体验） ====================

  /** 场景对象入组：VR 包裹期间新增对象挂到包裹组下（随世界一起平移），否则挂到 App.scene */
  addToScene(obj) {
    const target = (this._xrWrap && obj !== this._ghostAnchor) ? this._xrWrap : this.App.scene;
    target.add(obj);
    this.sceneObjects.push(obj);
    return obj;
  }

  /** VR 按钮：进入/退出 WebXR 沉浸会话（复用大厅同一套 XR 管线） */
  async _toggleVR() {
    if (this._disposed) return;
    if (this.App.xrMode !== 'off') {
      this.App.exitXrMode();
      return;
    }
    if (!this.App._xrModeAvailable || !this.App._xrModeAvailable()) {
      this._setStatus('🥽 当前环境不支持 WebXR VR（需 HTTPS + VR 设备/浏览器支持）');
      return;
    }
    this._xrActive = true;
    this.App._xrGameMode = true;
    // 先包裹游戏世界（进入后大厅管线加入的 XR 控制器会留在包裹之外，不受平移影响）
    this._wrapForVR();
    // 初始世界朝向：让 VR 起步视角对准当前第一人称视角
    this._xrTurnYaw = -(this.App._gameCamAzimuth || 0);
    this._updateVRBtn();
    const ok = await this.App.enterXrMode('webxr');
    if (ok !== true) {
      this._unwrapForVR();
      this._xrActive = false;
      this.App._xrGameMode = false;
      this._updateVRBtn();
      this._setStatus('🥽 进入 VR 失败：请确认 VR 设备已连接并授权');
      return;
    }
    this._setStatus('🥽 已进入 VR · 左摇杆移动 · 右摇杆转身 · 摘头盔或点「退出 VR」返回');
  }

  /** 游戏模式 VR 退出钩子（WebXR 退出时由大厅管线回调） */
  onExitXR() {
    this._xrActive = false;
    this.App._xrGameMode = false;
    // 把 VR 中转身后的朝向同步回第一人称（立即退出时与进入前一致）
    if (this._xrTurnYaw != null) {
      this.App._gameCamAzimuth = -this._xrTurnYaw;
    }
    this._unwrapForVR();
    this._updateVRBtn();
    this._setStatus('🥽 已退出 VR，回到第一人称模式');
  }

  _updateVRBtn() {
    const b = this._vrBtnEl || document.getElementById('cc-vr-btn');
    if (!b) return;
    if (this._xrActive) {
      b.textContent = '⏏ 退出 VR';
      b.classList.add('active');
      b.style.background = 'rgba(255,77,109,.16)';
      b.style.borderColor = 'rgba(255,77,109,.55)';
      b.style.color = '#ff6d85';
    } else {
      b.textContent = '🥽 VR';
      b.classList.remove('active');
      b.style.background = 'rgba(0,255,163,.1)';
      b.style.borderColor = 'rgba(0,255,163,.5)';
      b.style.color = '#7dffc8';
    }
  }

  /** 确保 VR 按钮存在：独立固定定位元素，挂在 document.body 上，不依赖游戏 UI 重建 */
  _ensureVRBtn() {
    if (this._disposed) return;
    if (this._vrBtnEl && this._vrBtnEl.parentNode) return;
    // 清理上一局可能残留的独立按钮（旧按钮仍绑定旧实例的点击事件，必须移除重建）
    const old = document.getElementById('cc-vr-btn');
    if (old && old.parentNode) old.parentNode.removeChild(old);
    this._vrBtnEl = null;
    const b = document.createElement('button');
    b.id = 'cc-vr-btn';
    b.type = 'button';
    b.className = 'cc-op-btn cc-vr-btn';
    b.textContent = '🥽 VR';
    b.style.cssText = 'position:fixed;top:62px;right:226px;z-index:9200;pointer-events:auto;cursor:pointer;' +
      'background:rgba(0,255,163,.1);border:1px solid rgba(0,255,163,.5);color:#7dffc8;border-radius:8px;' +
      'padding:5px 12px;font-size:12px;letter-spacing:1px;font-family:"Microsoft YaHei",sans-serif;';
    b.addEventListener('click', () => this._toggleVR().catch(err => console.warn('[赛博公司] VR 切换异常:', err?.message || err)));
    document.body.appendChild(b);
    this._vrBtnEl = b;
  }

  /** 每帧（WebXR 会话中由大厅管线调用）：摇杆移动/转身，幽灵锚点 + 第一人称相机 */
  updateXR(dt) {
    if (this._disposed || !this._xrActive || !this.App.xrPresenting) return;
    const renderer = this.App.renderer;
    if (!renderer || !renderer.xr) return;
    const THREE = this.App.THREE;
    // 读取摇杆：XR 手柄优先（四轴：左摇杆移动 / 右摇杆 X 转身），标准手柄回退
    const readPad = (src, idx) => {
      let gp = null;
      if (src && src.gamepad) gp = src.gamepad;
      if (!gp && renderer.xr && typeof renderer.xr.getGamepad === 'function') {
        try { gp = renderer.xr.getGamepad(idx); } catch (e) {}
      }
      return gp;
    };
    let moveX = 0, moveZ = 0, turnX = 0;
    let fourAxis = false;
    for (let i = 0; i < (this.App._xrControllers || []).length; i++) {
      const c = this.App._xrControllers[i];
      const gp = readPad(c.userData.inputSource, i);
      if (!gp) continue;
      const axes = gp.axes || [];
      if (axes.length >= 4) {
        moveX = axes[0]; moveZ = axes[1]; turnX = axes[2];
        fourAxis = true;
        break;
      }
      if (axes.length >= 2 && !fourAxis && moveX === 0 && moveZ === 0) {
        moveX = axes[0]; moveZ = axes[1];
      }
    }
    if (!fourAxis && moveX === 0 && moveZ === 0 && turnX === 0) {
      const pad = this.App._readStdGamepad ? this.App._readStdGamepad() : null;
      if (pad && pad.connected) { moveX = pad.moveX; moveZ = pad.moveZ; turnX = pad.turnX; }
    }
    if (Math.abs(moveX) < 0.15) moveX = 0;
    if (Math.abs(moveZ) < 0.15) moveZ = 0;
    if (Math.abs(turnX) < 0.15) turnX = 0;
    // 转身：世界绕用户水平旋转（VR 中相机矩阵由头显接管，旋转世界等效转身）
    if (turnX !== 0) {
      this._xrTurnYaw = (this._xrTurnYaw || 0) - turnX * 1.1 * dt;
    }
    // 前向 = 头显真实朝向（含基础方位 + 头部转动）
    const fwd = new THREE.Vector3();
    try {
      const xrCam = renderer.xr.getCamera();
      if (xrCam) xrCam.getWorldDirection(fwd);
      else this.App.camera.getWorldDirection(fwd);
    } catch (e) {
      try { this.App.camera.getWorldDirection(fwd); } catch (e2) {}
    }
    fwd.y = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
    fwd.normalize();
    const right = new THREE.Vector3().crossVectors(fwd, this.App._upVec || new THREE.Vector3(0, 1, 0)).normalize();
    // 移动幽灵锚点（带碰撞检测）：前向分量取反，与大厅手柄方向一致
    // （大厅为"世界反向移动"，等效玩家前向 = -fwd*moveZ）
    const speed = 1.8 * dt;
    const dx = (-fwd.x * moveZ + right.x * moveX) * speed;
    const dz = (-fwd.z * moveZ + right.z * moveX) * speed;
    if (dx !== 0 || dz !== 0) {
      const p = this._ghostAnchor.position;
      const nx = p.x + dx, nz = p.z + dz;
      if (!this.checkCollision(nx, nz)) {
        p.x = nx;
        p.z = nz;
      }
    }
    // 世界跟随玩家：反向平移包裹组（玩家不动、世界动，等效第一人称行走）
    this._updateXRWrap();
  }

  /** 包裹游戏世界（把办公场景挂到包裹组下，供 VR 整体平移/旋转） */
  _wrapForVR() {
    if (this._xrWrap || !this.App.scene) return;
    const THREE = this.THREE;
    const wrap = new THREE.Group();
    this._xrWrap = wrap;
    const excludes = new Set([this._ghostAnchor]);
    // 大厅残留对象不动（背景/星空/光晕/接触阴影/XR 控制器），游戏世界是封闭办公间
    if (this.App.modelGroup) excludes.add(this.App.modelGroup);
    if (this.App.backgroundGroup) excludes.add(this.App.backgroundGroup);
    if (this.App.starField) excludes.add(this.App.starField);
    if (this.App.parts) {
      if (this.App.parts.glow) excludes.add(this.App.parts.glow);
      if (this.App.parts.contactShadow) excludes.add(this.App.parts.contactShadow);
    }
    for (const c of (this.App._xrControllers || [])) excludes.add(c);
    for (let i = this.App.scene.children.length - 1; i >= 0; i--) {
      const child = this.App.scene.children[i];
      if (child === wrap || excludes.has(child)) continue;
      this.App.scene.remove(child);
      wrap.add(child);
    }
    this.App.scene.add(wrap);
    this._updateXRWrap();
  }

  /** 解除包裹：把场景对象放回 App.scene（退出 VR 时调用） */
  _unwrapForVR() {
    if (!this._xrWrap || !this.App.scene) return;
    const wrap = this._xrWrap;
    while (wrap.children.length) {
      const child = wrap.children[wrap.children.length - 1];
      wrap.remove(child);
      this.App.scene.add(child);
    }
    this.App.scene.remove(wrap);
    this._xrWrap = null;
  }

  /** 每帧同步包裹组：位置 = -玩家落脚点，旋转 = 用户转向（VR 中用户恒在原点） */
  _updateXRWrap() {
    if (!this._xrWrap || !this._ghostAnchor) return;
    const p = this._ghostAnchor.position;
    this._xrWrap.position.set(-p.x, 0, -p.z);
    this._xrWrap.rotation.y = this._xrTurnYaw || 0;
  }

  // ==================== 主流程 ====================

  _nextOpenPosition() {
    return POSITIONS.find(p => p.key !== 'hr' && !this.filledPositions.has(p.key)) || null;
  }

  _takeNextSeeker() {
    const q = this.seekerQueue.find(s => !s.used);
    if (!q) return null;
    q.used = true;
    return q;
  }

  /**
   * 为候选人与岗位量身定制面试计划（无固定模板）：
   * - 开场（自我介绍+岗位认知）→ 深挖 → 实战/创意（按岗位类型）→ 随机压力面 → 收尾
   * - 轮次与题型组合因人因岗而异，每一次面试都不重复
   */
  _planInterview(pos, cand) {
    const plan = [
      { type: 'open', label: '开场', intent: '请他做自我介绍并聊聊与岗位相关的经历' },
    ];
    // 技术/测试类岗位：真实场景实战题；创意/产品类岗位：命题创作或场景题
    if (['dev', 'ai', 'qa'].includes(pos.key)) {
      plan.push({ type: 'scenario', label: '实战', intent: '抛出一道真实的岗位场景/故障题，考察他解决问题的思路' });
    } else if (['planner', 'pm', 'artist'].includes(pos.key)) {
      plan.push({ type: 'scenario', label: '创意', intent: '抛出一道与岗位工作相关的命题创作或场景题' });
    } else {
      plan.push({ type: 'deep', label: '深挖', intent: '追问上一轮回答的细节' });
    }
    // 随机压力面：不同候选人体验不同，考察临场反应
    if (Math.random() < 0.45) {
      plan.push({ type: 'stress', label: '压力', intent: '快问快答式压力面试，考察临场反应与抗压能力' });
    }
    plan.push({ type: 'deep', label: '深挖', intent: '基于整场对话做最后的深入追问' });
    plan.push({ type: 'closing', label: '收尾', intent: '询问候选人是否有补充或想了解的问题' });
    return plan.slice(0, 5);
  }

  async _runGame() {
    // 防重入：裁员后重启招聘循环时避免与运行中的循环冲突
    if (this._gameLoopRunning) return;
    this._gameLoopRunning = true;
    try {
      // 先让第一位求职者候场（全息替身立即可见，无需等待模型加载）
      this._preloadNextSeeker();
      await this._sleep(600);
      if (this._disposed) return;

      // 开场白：由 LLM 依据当天招聘岗位生成个性化版本（无固定模板），失败回退固定话术；
      // 从存档恢复时跳过（员工已在岗，无需重复欢迎）
      if (!this._skipWelcome) {
        this._setStatus('💬 HR 正在发表招聘开场白…');
        const openPos = POSITIONS.filter(p => p.key !== 'hr' && !this.filledPositions.has(p.key)).map(p => p.name);
        let welcomeText = this._pickFallback(FALLBACK_SCRIPT.welcome);
        if (!this._llmDisabled()) {
          const data = await this._requestLine(
            this._hrPromptForLine(),
            `招聘开场时间到了，今天赛博公司还有这些岗位在招：${openPos.join('、')}。请面向所有候选人发表一个简短有力的开场白：欢迎 + 一句今天的招聘理念，不超过两句话。`,
            this.hrActor?.voice, this.hrActor?.rate
          );
          if (data && data.text) welcomeText = data.text;
        }
        await this._hrSpeak(welcomeText).catch(() => {});
        if (this._disposed) return;
      }
      this._skipWelcome = false;

      // 无限运营循环：录取员工不代表游戏结束，但员工总数上限 6 人，满员停招。
      // - 在职 < 6 → 继续面试（岗位空缺先填，全招满则运营期随机扩招至上限）；
      // - 在职 = 6 → 满员停招（游戏继续，可自由走动、与角色近距离聊天）；
      // - 候选人面试完一轮 → 重置队列，新候选人继续来应聘（已入职的不再重复）。
      while (!this._disposed) {
        // 整段容错：取岗位/取候选人/结算任一处抛异常都不能杀死主循环
        // （否则恢复存档后主循环静默终止 = 候选人不出现、面试"卡住"）
        try {
          // 满员检查：员工数达到上限即停招（存档恢复后满员同样在此停招）
          if (this.hiredList.length >= MAX_EMPLOYEES) {
            this._stopHiring();
            break;
          }
          let pos = this._nextOpenPosition();
          if (!pos) {
            // 全部岗位招满但员工未达上限：进入运营模式（仅首次提示），继续扩招
            if (this.phase !== 'operating') this._enterOperationMode();
            pos = this._pickRandomPosition();
          }
          let seeker = this._takeNextSeeker();
          if (!seeker) {
            this._resetSeekerQueue();
            seeker = this._takeNextSeeker();
            if (!seeker) {
              // 候选人卡片耗尽（全部入职且员工未满，或全部标记已用）：
              // 生成合成候选人补位，保证招聘流程永不卡死
              this._spawnFillerSeeker();
              seeker = this._takeNextSeeker();
              if (!seeker) { await this._sleep(500); continue; }
            }
          }

          // 面试流程异常保护：任何单点错误都不允许杀死招聘主循环（否则面试永久卡住）
          try {
            await this._runInterview(pos, seeker);
          } catch (e) {
            console.error('[赛博公司] 面试流程异常，已跳过该候选人继续招聘:', e);
            if (this._disposed) return;
            // 关键保护：若异常发生在录用流程内（角色已 tag=employee 并加入在职列表），
            // 绝不能 dispose 该员工（否则"录用后直接消失"）；仅作废未录用的候选人实体
            const isHiredEmp = !!(seeker.actor && seeker.actor.tag === 'employee');
            if (seeker.actor && !isHiredEmp) {
              try { seeker.actor.dispose(); } catch (e2) {}
              seeker.actor = null;
              seeker.loaded = false;
            }
            this.currentSeeker = null;
            this._setStatus(isHiredEmp ? '⚠ 员工入职流程收尾异常，已继续运营…' : '⚠ 本次面试出现异常，已跳过，继续招聘…');
            continue;
          }
          if (this._disposed) return;

          // 每轮面试结束 = 结算一期：按当期裁员数修正声誉与活力，重算五维总分
          this._settlePeriod();
          if (this._opPanel) this._renderOperationPanel();   // 运营面板打开中则实时刷新
          await this._drawScoreDashboard(2400);
          if (this._disposed) return;

          await this._sleep(900);
          await this._preloadNextSeeker();
          this._saveGame();   // 每轮面试结束后自动存档
        } catch (e) {
          console.error('[赛博公司] 招聘主循环段异常，已容错继续:', e);
          if (this._disposed) return;
          await this._sleep(500);   // 短暂退避后继续下一轮，绝不中断主循环
        }
      }
    } catch (e) {
      console.error('[赛博公司] 招聘主循环启动段异常:', e);
    } finally {
      // 任何退出路径（正常结束 / 满员 break / disposed return / 异常）都复位标志，
      // 防止 _gameLoopRunning 残留导致下次 _runGame 防重入直接 return（面试"卡住"）
      this._gameLoopRunning = false;
    }
  }

  /** 满员停招：员工数达到上限，招聘暂停；游戏不结束，可继续自由走动与聊天 */
  _stopHiring() {
    if (this._disposed || this._stoppedHiring) return;
    this._stoppedHiring = true;
    this.phase = 'operating';
    this._recalcScore();
    const g = this._gradeOf(this.stats.score);
    this._appendTranscript('system', `🛑 员工人数已达上限 ${MAX_EMPLOYEES} 人，招聘暂停。公司继续运转，可自由走动与大家聊天。`);
    this._drawHoloScreen('赛博公司 · 满员运营', `在职员工 ${this.hiredList.length}/${MAX_EMPLOYEES} 人 · 公司得分 ${this.stats.score}（${g.grade} 级）`,
      '员工已招满，招聘暂停。公司照常运转，欢迎随时与大家聊天', '#ffd166', '#00e5ff');
    this._setStatus(`🏢 赛博公司满员运营 · 在职 ${this.hiredList.length}/${MAX_EMPLOYEES} 人 · 评分 ${this.stats.score} 分（${g.grade} 级）`);
    this._saveGame();
  }

  /** 进入运营模式：所有岗位招满后公司开始运转，可继续扩招至员工上限（游戏不结束） */
  _enterOperationMode() {
    if (this._disposed || this.phase === 'operating') return;
    this.phase = 'operating';
    this._recalcScore();
    const g = this._gradeOf(this.stats.score);
    this._appendTranscript('system', `🎊 所有岗位全部招满！赛博公司开始运转（评分 ${this.stats.score} · ${g.grade} 级），可继续扩招至员工上限 ${MAX_EMPLOYEES} 人`);
    this._drawHoloScreen('赛博公司 · 运营中', `在职员工 ${this.hiredList.length}/${MAX_EMPLOYEES} 人 · 公司得分 ${this.stats.score}（${g.grade} 级）`,
      '岗位已招满，可继续扩招至满员；点击右上角「运营面板」管理裁员与评分', '#00ffa3', '#00e5ff');
    this._setStatus(`🏢 赛博公司运营中 · 在职 ${this.hiredList.length}/${MAX_EMPLOYEES} 人 · 评分 ${this.stats.score} 分（${g.grade} 级）`);
    this._saveGame();
  }

  /** 随机挑一个非 HR 岗位（运营期扩招用，允许同岗位多名员工） */
  _pickRandomPosition() {
    const arr = POSITIONS.filter(p => p.key !== 'hr');
    return arr[Math.floor(Math.random() * arr.length)];
  }

  /** 更新求职者队列：面试过的角色（被拒/入职）永久退出，绝不重复面试同一人；
   *  新人由合成候选人补位（_spawnFillerSeeker），保证无限招聘且不重样 */
  _resetSeekerQueue() {
    // 只保留从未面试过的候选人；面试过的（used=true）永久移出队列：
    // 被拒角色不再回来重复面试，已入职员工继续在岗不受影响
    this.seekerQueue = this.seekerQueue.filter(s => !s.used);
    this._appendTranscript('system', `🔄 求职者队列已更新，新候选人陆续到场…`);
  }

  /** 合成求职者池：真实角色卡片用尽时补位，保证招聘永不中断（名字/人设/音色各异） */
  _spawnFillerSeeker() {
    if (this._fillerIdx == null) this._fillerIdx = 0;
    const pool = [
      { name: '林晚晴', role_name: '数据分析师', prompt: '冷静理性，说话直击重点，偶尔冒出一句数据梗。', voice: 'zh-CN-XiaoxiaoNeural' },
      { name: '顾一鸣', role_name: '硬件工程师', prompt: '动手派，讲求实际，反感空谈，喜欢聊芯片和散热。', voice: 'zh-CN-YunxiNeural' },
      { name: '苏挽月', role_name: '产品经理', prompt: '善于共情，逻辑缜密，把复杂的事说得通俗易懂。', voice: 'zh-CN-XiaoyiNeural' },
      { name: '韩东野', role_name: '市场总监', prompt: '热血外向，说话有感染力，张口就是金句和趋势。', voice: 'zh-CN-YunyangNeural' },
      { name: '白芷', role_name: '视觉设计师', prompt: '文艺敏感，用词生动，偶尔蹦出赛博朋克意象。', voice: 'zh-CN-XiaoxiaoNeural' },
      { name: '秦朗', role_name: '算法工程师', prompt: '沉默寡言但一针见血，喜欢用数学比喻。', voice: 'zh-CN-YunjianNeural' },
      { name: '唐雨桐', role_name: '运营主管', prompt: '务实高效，条理清晰，擅长把目标拆成可执行步骤。', voice: 'zh-CN-XiaoyiNeural' },
      { name: '陆子昂', role_name: '安全工程师', prompt: '谨慎多疑，防御心重，话里常带「风险」「预案」。', voice: 'zh-CN-YunxiNeural' },
    ];
    const c = pool[this._fillerIdx % pool.length];
    this._fillerIdx++;
    const card = {
      id: `filler_${Date.now()}_${this._fillerIdx}`,
      name: c.name,
      role_name: c.role_name,
      model_name: '',
      system_prompt: c.prompt,
      tts: { edge_voice: c.voice, edge_rate: '+8%' },
    };
    // 避免与已在队列中的合成候选人重名
    if (this.seekerQueue.some(s => s.card && s.card.id && s.card.id.startsWith('filler_') && s.card.name === c.name)) {
      card.name = c.name + (this._fillerIdx % 9);
    }
    this.seekerQueue.push({ card, actor: null, loaded: false, standPos: LAYOUT.waitingSpot, used: false });
    this._appendTranscript('system', `🆕 候选人工会已推送新求职者「${card.name}」（${card.role_name}）`);
  }

  /** 预载下一位求职者：先以全息替身立即就位候场区，真实模型后台加载完成后替换 */
  async _preloadNextSeeker() {
    const next = this.seekerQueue.find(s => !s.used && !s.loaded);
    if (!next) return;
    const actor = new CharacterActor(this, next.card, 'seeker');
    this.actors.push(actor);
    next.actor = actor;
    next.loaded = true;   // 占位即视为"已加载"，不阻塞面试流程

    // 注意顺序：必须先调用 load()（其同步段会创建 group/替身），再 setPosition 才生效
    actor.load().catch(() => {});
    actor.setPosition(LAYOUT.waitingSpot.x, LAYOUT.waitingSpot.z);
    actor.faceTarget = { x: LAYOUT.stage.seeker.x, z: LAYOUT.stage.seeker.z };
    this._setStatus(`🎩 求职者「${next.card.name}」已就位候场区`);
  }

  async _runInterview(pos, seeker) {
    // 确保该候选人已有角色实体（队列重置后可能未加载）：幂等创建全息替身
    if (!seeker.actor) {
      const actor = new CharacterActor(this, seeker.card, 'seeker');
      this.actors.push(actor);
      seeker.actor = actor;
      seeker.loaded = true;
      actor.load().catch(() => {});
      actor.setPosition(LAYOUT.waitingSpot.x, LAYOUT.waitingSpot.z);
      actor.faceTarget = { x: LAYOUT.stage.seeker.x, z: LAYOUT.stage.seeker.z };
    }
    const cand = seeker.actor;
    this.currentSeeker = cand;
    this._currentPos = pos;

    // 更新 UI：候选人信息
    this._setCandidate(cand.name, pos.name, pos.icon, pos.color);
    this._appendTranscript('system', `【${pos.icon} ${pos.name}】面试开始 —— 求职者：${cand.name}`);
    this._drawHoloScreen('赛博公司 · 面试直播', `求职者：${cand.name}`,
      `应聘岗位：${pos.icon} ${pos.name}`, pos.color, '#00e5ff');

    // 为这位候选人量身定制面试计划：题型/轮次因人因岗而异，无固定模板
    const plan = this._planInterview(pos, cand);

    // 开场：HR 提问与求职者自我介绍并行生成（开场问题是"自我介绍+岗位经历"，
    // 回答不依赖提问细节，可并行消除等待感）
    this._setStatus(`⚡ 正在生成面试对白…`);
    const genRound0 = Promise.all([
      this._genHRQuestion(pos, cand, null, plan[0])
        .catch(() => (FALLBACK_SCRIPT.openQ[pos.key] ? this._pickFallback(FALLBACK_SCRIPT.openQ[pos.key]) : this._pickFallback(FALLBACK_SCRIPT.followQ))),
      this._genSeekerIntro(pos, cand)
        .catch(() => this._pickFallback(FALLBACK_SCRIPT.answer)),
    ]);

    // 候选人从候场区走上面试位（先绕到面试桌后方，避免穿桌）
    this._setStatus(`⏳ ${cand.name} 走向面试区…`);
    await this._moveActor(cand, [{ x: -1, z: 2 }, { x: LAYOUT.stage.seeker.x, z: LAYOUT.stage.seeker.z }]);
    if (this._disposed) return;
    // 候选人面向 HR（玩家 HR 可能自由走动，取 HR 当前位置）
    cand.faceTarget = {
      x: this.hrActor?.group ? this.hrActor.group.position.x : LAYOUT.stage.hr.x,
      z: this.hrActor?.group ? this.hrActor.group.position.z : LAYOUT.stage.hr.z,
    };
    cand.setEmotion(null);
    await this._sleep(500);

    // 第 1 轮：HR 开场提问 + 求职者自我介绍
    const [q0, a0] = await genRound0;
    await this._hrSpeak(q0);
    if (this._disposed) return;
    await this._seekerSpeak(cand, a0, { pos, cand });
    const answers = [a0];

    // 后续轮次：每轮先由 HR 针对"上一轮回答"出题（类型各异：深挖/实战/压力/收尾），
    // 求职者回答时把 HR 的真实问题带进上下文（告别"回答不看题"的固定模板）
    for (let i = 1; i < plan.length; i++) {
      if (this._disposed) return;
      const rd = plan[i];
      this._appendTranscript('system', `【第 ${i + 1} 轮 · ${rd.label}】${rd.intent}`);
      this._setStatus(`⚡ 正在生成「${rd.label}」对白…`);
      const q = await this._genHRQuestion(pos, cand, answers, rd);
      if (this._disposed) return;
      // 提前生成求职者回答（与 HR 语音播放并行，隐藏生成等待）
      const genA = this._genSeekerAnswer(pos, cand, q, rd);
      await this._hrSpeak(q);
      if (this._disposed) return;
      const a = await genA;
      await this._seekerSpeak(cand, a, { pos, cand });
      answers.push(a);
    }

    // 评估 + 个性化结论
    if (this._disposed) return;
    const result = this._evaluate(cand, pos, answers);
    await this._verdict(pos, cand, result, answers);
  }

  // ==================== 台词生成 ====================

  _hrPromptForLine() {
    const card = this.hrCard || {};
    return [
      `你是「${this.hrActor ? this.hrActor.name : 'HR'}」，${POSITIONS[0].name}，正在赛博公司主持招聘面试。`,
      this._ceoContextBlock(),   // R4 身份注入：HR 知道老板是 CEO
      `你的性格底色：${(card.system_prompt || '干练、专业、友善').slice(0, 120)}`,
      '说话要求：口语化、简短（不超过50字）、专业又不失亲和，可以带一点赛博朋克意象（霓虹/数据/芯片）。',
      '只输出你要说的话本身，不要引号、冒号、角色名或任何解释。',
    ].join('\n');
  }

  _seekerPromptForLine(pos, cand) {
    const card = cand.card || {};
    return [
      `你是「${cand.name}」，一位前来赛博公司应聘「${pos.name}」岗位的求职者。`,
      this._ceoContextBlock(),   // R4 身份注入：求职者知道是 CEO 的公司
      `你的性格底色：${(card.system_prompt || '一个普通的求职者').slice(0, 120)}`,
      `注意：你性格鲜明，面试时会保持真实的自己，但会尽量围绕「${pos.name}」这个岗位认真作答。`,
      '说话要求：口语化、简短（不超过50字）、带个人风格。',
      '只输出你要说的话本身，不要引号、冒号、角色名或任何解释。',
    ].join('\n');
  }

  _pickFallback(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  /**
   * 生成 HR 的面试问题（无固定模板）：
   * - 开场轮：向候选人打招呼并请其自我介绍；
   * - 后续轮：必须基于候选人"上一轮的实际回答"出题（深挖细节/追问矛盾/抛出相关场景题），
   *   同一岗位面对不同候选人也会问出完全不同的问题。
   */
  async _genHRQuestion(pos, cand, answers, rd) {
    const prev = answers && answers.length ? answers : null;
    // 兜底台词：仅 LLM 不可用时按轮次类型从脚本池随机抽取
    let fallback;
    if (!prev) {
      fallback = FALLBACK_SCRIPT.openQ[pos.key] ? this._pickFallback(FALLBACK_SCRIPT.openQ[pos.key]) : this._pickFallback(FALLBACK_SCRIPT.followQ);
    } else if (rd?.type === 'scenario') {
      fallback = FALLBACK_SCRIPT.scenarioQ[pos.key] ? this._pickFallback(FALLBACK_SCRIPT.scenarioQ[pos.key]) : this._pickFallback(FALLBACK_SCRIPT.followQ);
    } else if (rd?.type === 'stress') {
      fallback = this._pickFallback(FALLBACK_SCRIPT.stressQ);
    } else if (rd?.type === 'closing') {
      fallback = this._pickFallback(FALLBACK_SCRIPT.closingQ);
    } else {
      fallback = this._pickFallback(FALLBACK_SCRIPT.followQ);
    }
    if (this._llmDisabled()) return fallback;

    const context = prev
      ? `面试记录（他对你问题的回答）：\n${prev.map((a, i) => `第${i + 1}轮他答道：「${a}」`).join('\n')}\n\n现在进入「${rd?.label || '深挖'}」环节（${rd?.intent || ''}），轮到你（HR）开口：必须基于他刚才的实际回答来出题，或追问他话里的细节与矛盾，或抛出与他回答相关的场景题，让他继续展开。不要重复问过的问题。`
      : `现在轮到你（HR）开口：面试开始，请向求职者「${cand.name}」打个招呼，然后请他做自我介绍，并聊聊与「${pos.name}」岗位相关的经历。`;
    const data = await this._requestLine(this._hrPromptForLine(), context, this.hrActor?.voice, this.hrActor?.rate);
    return (data && data.text) ? data.text : fallback;
  }

  /** 求职者开场自我介绍（与 HR 开场提问并行生成，回答不依赖提问细节） */
  async _genSeekerIntro(pos, cand) {
    const fallback = this._pickFallback(FALLBACK_SCRIPT.answer);
    if (this._llmDisabled()) return fallback;
    const context = `请作为应聘「${pos.name}」的求职者「${cand.name}」做一个自我介绍：说说你是谁、你的性格特点，以及你认为自己凭什么能胜任「${pos.name}」这个岗位。保持你自己的说话风格。`;
    const data = await this._requestLine(this._seekerPromptForLine(pos, cand), context, cand.voice, cand.rate);
    return (data && data.text) ? data.text : fallback;
  }

  /** 求职者回答：把 HR 的真实问题完整带进上下文，真正做到"对题作答" */
  async _genSeekerAnswer(pos, cand, question, rd) {
    const fallback = this._pickFallback(FALLBACK_SCRIPT.answer);
    if (this._llmDisabled()) return fallback;
    const roundTag = rd ? `这是「${rd.label}」环节。` : '';
    const context = `HR 刚刚对你说：「${question}」\n${roundTag}请作为应聘「${pos.name}」的求职者「${cand.name}」认真回答这个问题，保持你的性格和说话风格。`;
    const data = await this._requestLine(this._seekerPromptForLine(pos, cand), context, cand.voice, cand.rate);
    return (data && data.text) ? data.text : fallback;
  }

  /**
   * LLM 是否已确认可用（首次成功后置 true，用于生成带 TTS 语音的 HR 主动台词）。
   * 未确认时（_llmOk 为空）仍会尝试一次真实调用，失败自动回退脚本，保证永不卡死。
   */
  _llmEnabled() {
    return !!this._llmOk;
  }

  /** LLM 被明确判定为不可用（首次调用失败多次后），直接使用回退脚本 */
  _llmDisabled() {
    return this._llmOk === false;
  }

  /** 调用后端生成台词 + 语音 */
  async _requestLine(system_prompt, context, voice, rate) {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 12000);
      this._timers.push(timer);
      const res = await fetch('/api/game/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ system_prompt, context, voice: voice || '', rate: rate || '+8%', max_tokens: 200 }),
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) return null;
      const data = await res.json();
      if (data && data.ok) { this._llmOk = true; this._llmFailCount = 0; return data; }
      return null;
    } catch (e) {
      console.warn('[赛博公司] 台词接口失败:', e?.message || e);
      return null;
    } finally {
      // 连续失败熔断：避免 LLM 不可用时每轮都白等超时
      if (!this._llmOk) {
        this._llmFailCount = (this._llmFailCount || 0) + 1;
        if (this._llmFailCount >= 2) this._llmOk = false;
      }
    }
  }

  // ==================== 近距离语音（10 米内能听到） ====================

  /**
   * 拦截玩家消息：文字输入与语音输入最终都会经 App.addUserMsg 回显，
   * 游戏运行中在此感知玩家说话，广播给身边 10 米内的角色。
   */
  _hookPlayerChat() {
    if (this._chatHookActive || !this.App) return;
    this._chatHookActive = true;
    this._origAddUserMsg = this.App.addUserMsg;
    this._hookFn = (text, fromVoice) => {
      try {
        if (this._origAddUserMsg) this._origAddUserMsg.call(this.App, text, fromVoice);
      } catch (e) {}
      if (this._chatHookActive && this.state === 'playing' && (this.phase === 'running' || this.phase === 'operating') && text && text.trim()) {
        this._onPlayerChat(text.trim());
      }
    };
    this.App.addUserMsg = this._hookFn;

    // 大厅 AI 对话自动切换为蜂群系统：游戏内发送的消息不再发往大厅 AI，
    // 由身边员工（蜂群）回应，回复写入聊天面板（App.addAIMsg）
    this._origSendText = this.App.sendText;
    this._sendHook = (text) => {
      if (this._chatHookActive && this.state === 'playing' && !this._disposed) {
        // 复位大厅"思考中"状态（submitText 在 sendText 之后才设置 THINKING，延迟复位）
        if (this.App.setState && this.App.State) {
          setTimeout(() => {
            if (this._chatHookActive && this.state === 'playing' && !this._disposed) {
              this._resetLobbyBadge();
            }
          }, 60);
        }
        return;   // 吞掉发送：不调用大厅 AI
      }
      if (this._origSendText) return this._origSendText.call(this.App, text);
    };
    this.App.sendText = this._sendHook;
  }

  _unhookPlayerChat() {
    if (!this._chatHookActive) return;
    this._chatHookActive = false;
    if (this.App && this.App.addUserMsg === this._hookFn && this._origAddUserMsg) {
      this.App.addUserMsg = this._origAddUserMsg;
    }
    if (this.App && this.App.sendText === this._sendHook && this._origSendText) {
      this.App.sendText = this._origSendText;
    }
    this._hookFn = null;
    this._origAddUserMsg = null;
    this._origSendText = null;
    this._sendHook = null;
  }

  /** 玩家说话：10 米内的角色听到并回应（空间音频听力半径）；指令词触发式玩法优先 */
  async _onPlayerChat(text) {
    const pos = this._playerPos();
    if (!pos || this._disposed) return;
    const heard = this.actors.filter(a => a && !a.removed && a.group)
      .filter(a => Math.hypot(a.group.position.x - pos.x, a.group.position.z - pos.z) <= HEAR_RADIUS);
    if (heard.length === 0) {
      // 身边没人听到也复位徽章，避免残留"思考中"
      this._resetLobbyBadge();
      return;
    }

    // CEO 指令触发式玩法：开会 / 汇报 / 散会
    const cmd = this._detectCEOCommand(text);
    if (cmd) {
      await this._handleCEOCommand(cmd, text);
      this._resetLobbyBadge();
      return;
    }

    const names = heard.map(a => a.name).join('、');
    this._setStatus(`👂 ${names} 听到了你说的话，正在回应…`);

    // 员工独立对话记录：玩家说的话记录到每个听到的员工名下
    for (const actor of heard) this._recordChat(actor.name, 'user', text);

    for (const actor of heard) {
      if (this._disposed) break;
      const now = Date.now();
      const last = this._chatCooldown[actor.name] || 0;
      if (now - last < 5000) continue;              // 5 秒冷却，防止刷屏
      this._chatCooldown[actor.name] = now;
      await this._actorRespondToPlayer(actor, text);
      if (this._disposed) break;
    }
    this._resetLobbyBadge();
  }

  /** 复位大厅状态徽章（思考中/聆听中 → 在线），防止游戏内对话后残留"思考中" */
  _resetLobbyBadge() {
    if (!this.App || !this.App.setState || !this.App.State) return;
    try {
      if (this.App.removeTyping) this.App.removeTyping();
      // 注意：App.State 只有 IDLE/THINKING/LISTENING/SPEAKING，没有 READY；
      // 传 undefined 会让 setState 内部取 map[undefined] 抛错，徽章永远卡在"思考中"
      this.App.setState(this.App.State.IDLE);
    } catch (e) {}
  }

  /** 让某角色以本人设回应用户的话（LLM 生成 + 专属音色 + 头顶气泡） */
  async _actorRespondToPlayer(actor, playerText) {
    if (!actor || !actor.group || this._disposed) return;
    const pos = this._playerPos();
    if (pos) actor.faceTarget = { x: pos.x, z: pos.z };   // 转身面向玩家

    // 若该角色正在说面试台词，等它说完再回应（最长 6 秒），保证说话串行不叠加
    const waitUntil = Date.now() + 6000;
    while (actor.isSpeaking && Date.now() < waitUntil && !this._disposed) {
      await this._sleep(120);
    }
    if (this._disposed) return;

    const tag = actor.tag === 'hr' ? 'hr' : 'seeker';
    let line = null;
    if (!this._llmDisabled()) {
      // R3/R4：玩家是 CEO，角色知道老板身份
      const context = `你的老板（CEO ${this.ceo ? this.ceo.ceo_name : '玩家'}）就在你身边（不到两米）对你说：「${playerText}」\n请以你的身份自然、口语化地回应你的老板：可以接话、反问、或聊聊看法。一句话，不超过40字。`;
      const data = await this._requestLine(this._promptForActor(actor), context, actor.voice, actor.rate);
      if (data && data.text) line = data.text;
    }
    if (!line) {
      line = this._pickFallback([
        '嗯，我在听。你继续说。',
        '有意思，然后呢？',
        '你说得对，我也这么想。',
        '哈哈，这个话题不错。',
        '我明白了，我记下了。',
      ]);
    }
    // 员工独立对话记录：员工回应记入该员工名下
    this._recordChat(actor.name, 'agent', line);
    // 大厅 AI 对话已切换为蜂群：员工回应写入聊天面板（以员工身份呈现）
    try {
      if (this.App.addAIMsg) this.App.addAIMsg(`【${actor.name}】${line}`);
    } catch (e) {}
    await this._performLine(actor, line, null, tag);
  }

  /** 角色说话时广播：10 米内的其他角色"听到"，转头看向说话者（聆听姿态） */
  _broadcastSpeech(actor) {
    if (!actor || !actor.group) return;
    const x = actor.group.position.x, z = actor.group.position.z;
    for (const a of this.actors) {
      if (!a || a.removed || a === actor || !a.group) continue;
      if (Math.hypot(a.group.position.x - x, a.group.position.z - z) <= HEAR_RADIUS) {
        a.faceTarget = { x, z };                  // 听到 → 转头聆听
      }
    }
  }

  /** 角色通用人设（HR / 求职者 / 员工通用） */
  _promptForActor(actor) {
    if (!actor) return '';
    const card = actor.card || {};
    if (actor.tag === 'hr') return this._hrPromptForLine();
    const posName = (actor.tag === 'employee' && actor.hiredPos)
      ? actor.hiredPos.name
      : (this._currentPos ? this._currentPos.name : '赛博公司');
    return [
      `你是「${actor.name}」，一位${posName}相关的人，此刻正在赛博公司里。`,
      this._ceoContextBlock(),   // R4 身份注入：员工知道老板是 CEO
      `你的性格底色：${(card.system_prompt || '一个性格鲜明的人').slice(0, 120)}`,
      '说话要求：口语化、简短（不超过40字）、带个人风格。',
      '只输出你要说的话本身，不要引号、冒号、角色名或任何解释。',
    ].join('\n');
  }

  // ==================== 说台词 ====================

  /** HR 说话（语音 + 字幕 + 表情）；台词已在生成阶段完成，这里直接演绎 */
  async _hrSpeak(text) {
    if (this._disposed) return;
    await this._performLine(this.hrActor, text, null, 'hr');
  }

  /** 求职者说话 */
  async _seekerSpeak(cand, text, meta) {
    if (this._disposed) return;
    await this._performLine(cand, text, null, 'seeker');
  }

  async _performLine(actor, text, audioData, tag) {
    if (!actor || this._disposed) return;
    // 表情：说话时中性，配一点情绪
    const happyWords = ['通过', '欢迎', '恭喜', '厉害', '不错', '优秀'];
    const thinkWords = ['？', '呢', '吗', '考虑'];
    if (tag === 'hr') {
      if (happyWords.some(w => text.includes(w))) actor.setEmotion('happy');
      else if (text.includes('遗憾')) actor.setEmotion('sad');
      else actor.setEmotion(null);
    } else {
      if (text.includes('！') || text.includes('很') ) actor.setEmotion('happy');
      else if (thinkWords.some(w => text.includes(w))) actor.setEmotion('thoughtful');
      else actor.setEmotion(null);
    }

    // 近距离语音：说话的声音传到 10 米内，其他角色听到 → 转头聆听
    this._broadcastSpeech(actor);

    // 公开透明：写入字幕与直播屏
    this._appendTranscript(tag === 'hr' ? 'hr' : 'candidate', `${actor.name}：${text}`);
    this._drawHoloScreen(
      tag === 'hr' ? `HR · ${actor.name}` : `求职者 · ${actor.name}`,
      this._currentPos ? `${this._currentPos.icon} ${this._currentPos.name}` : '赛博公司',
      text.slice(0, 80), '#00e5ff', tag === 'hr' ? '#00e5ff' : '#ff4df0'
    );

    // 说话状态 + 头顶气泡 + 语音播放
    actor.setSpeaking(true);
    try {
      actor.showBubble(text, tag === 'hr' ? '#00e5ff' : '#ff2bd6');
    } catch (e) {
      console.warn('[赛博公司] 气泡显示失败（不影响语音）:', e?.message || e);
    }
    let played = false;
    if (audioData && audioData.audio_base64) {
      // 有音频：播放成功即视为说完了；播放失败（如解码/autoplay 被拒）回退估读，
      // 保证气泡与口型按台词时长正常展示，不秒闪
      played = await this._playAudio(actor, audioData.audio_base64, audioData.mime || 'audio/mpeg');
    } else if (actor.voice) {
      // 无 LLM 音频（回退脚本等）：走纯 TTS 模式补语音，失败才用估读时间
      // 空间音频：玩家距说话者 ≥10m 听不到 → 跳过 TTS 请求（否则每句台词都要
      // 白等 TTS 8s 超时且全程无声，观感完全像"面试卡住"），直接按估读时长展示气泡，
      // 面试节奏恢复正常；靠近 10m 内才请求 TTS 并按距离衰减音量播放
      let speakDist = Infinity;
      if (actor.group) {
        const p = this._playerPos();
        if (p) speakDist = Math.hypot(actor.group.position.x - p.x, actor.group.position.z - p.z);
      }
      if (speakDist < HEAR_RADIUS) {
        try {
          const tts = await this._requestTts(text, actor.voice, actor.rate);
          if (tts && tts.audio_base64) {
            played = await this._playAudio(actor, tts.audio_base64, tts.mime || 'audio/mpeg');
          }
        } catch (e) {
          console.warn('[赛博公司] 补语音失败:', e?.message || e);
        }
      }
    }
    if (!played) await this._sleep(this._estimateReadTime(text));
    actor.hideBubble();
    actor.setSpeaking(false);
    await this._sleep(250);
  }

  /** 纯 TTS：对给定文本直接合成该角色音色的语音（不经过 LLM） */
  async _requestTts(text, voice, rate) {
    if (!text || !voice) return null;
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 8000);   // TTS 失败/慢时快速回退，避免气泡长时间空挂
      this._timers.push(timer);
      const res = await fetch('/api/game/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.slice(0, 120), voice, rate: rate || '+8%' }),
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) return null;
      const data = await res.json();
      return (data && data.ok) ? data : null;
    } catch (e) {
      console.warn('[赛博公司] TTS 接口失败:', e?.message || e);
      return null;
    }
  }

  _estimateReadTime(text) {
    return Math.min(12000, 800 + (text ? text.length : 10) * 70);
  }

  /**
   * 播放语音。返回 Promise<boolean>：
   * - true  真正开始播放并自然播完 / 按时长播完（视为成功）
   * - false 播放失败（play 被拒 / onerror / 无有效时长）—— 调用方应回退估读时长，
   *         避免"语音没响但流程以为播完"导致气泡秒闪、节奏错乱
   */
  _playAudio(actor, base64, mime) {
    return new Promise(resolve => {
      let done = false;
      const finish = (ok) => { if (done) return; done = true; resolve(!!ok); };
      try {
        // 空间音频：玩家作为听者，音量随与说话角色的距离衰减。
        // 超过听力半径（10m）听不到 → 不播放、返回失败走估读，气泡仍正常展示
        let vol = 1;
        if (actor && actor.group) {
          const p = this._playerPos();
          if (p) {
            const d = Math.hypot(actor.group.position.x - p.x, actor.group.position.z - p.z);
            if (d >= HEAR_RADIUS) { finish(false); return; }
            vol = this._volumeForDistance(d);
          }
        }
        const audio = new Audio(`data:${mime};base64,${base64}`);
        audio.volume = Math.max(0.05, Math.min(1, vol));   // 2m 后按距离线性衰减
        this._audios.push(audio);
        audio.onended = () => finish(true);
        audio.onerror = () => finish(false);              // 解码/播放出错：如实报失败
        audio.onloadedmetadata = () => {
          const dur = audio.duration || 0;
          if (!isFinite(dur) || dur <= 0) { finish(false); return; }   // 无有效时长：视为失败走估读
          // 音频时长 + 缓冲余量，最长 30s，保证长台词能自然播完（不掐断）
          const est = Math.min(30000, (dur * 1000) + 600);
          const t = setTimeout(() => { try { audio.pause(); } catch (e) {} finish(true); }, est);
          this._timers.push(t);
        };
        const p = audio.play();
        if (p && typeof p.catch === 'function') p.catch(() => finish(false));  // autoplay 被拒等：如实报失败
        // 兜底超时（防止播放卡死，仅兜底不截断正常长语音）
        const t = setTimeout(() => { try { audio.pause(); } catch (e) {} finish(true); }, 15000);
        this._timers.push(t);
      } catch (e) {
        finish(false);
      }
    });
  }

  /** 空间音量：距离 → 音量（0-2m 满音量，2-10m 线性衰减至 0，10m 外无声） */
  _volumeForDistance(d) {
    if (d <= VOLUME_FADE_START) return 1.0;
    if (d >= HEAR_RADIUS) return 0;
    return 1 - (d - VOLUME_FADE_START) / (HEAR_RADIUS - VOLUME_FADE_START);
  }

  // ==================== 评估与结论 ====================

  _evaluate(cand, pos, answers) {
    const list = (answers || []).filter(Boolean);
    const text = list.join(' ');
    let lenScore = Math.min(40, (text ? text.length : 0) * 0.22);
    let kwHits = 0;
    for (const k of (pos.keywords || [])) {
      if (text && text.toLowerCase().includes(k.toLowerCase())) kwHits++;
    }
    const kwScore = Math.min(35, kwHits * 7);
    // 面试轮数越多、作答越充分，越能反映求职者诚意与岗位匹配
    const roundBonus = Math.min(10, list.length * 2.5);
    // 多轮回答足够充实视为内容一致性加分
    const consistency = list.length >= 2 && text.length > 60 ? 5 : 0;
    const rand = Math.random() * 8;
    const score = Math.max(20, Math.min(98, Math.round(18 + lenScore + kwScore + roundBonus + consistency + rand)));
    const pass = score >= 62;
    return { score, pass, kwHits, textLen: text ? text.length : 0, rounds: list.length };
  }

  async _verdict(pos, cand, result, answers) {
    if (this._disposed) return;
    // 显示评估面板（个性化评语稍后生成后填入）
    this._showResult(result.score, result.pass, pos, cand);
    this._drawHoloScreen('赛博公司 · 面试结果',
      `${cand.name} 应聘 ${pos.name}`,
      `${result.pass ? '✅ 录用' : '❌ 未通过'} · 综合评分 ${result.score}/100`, pos.color, result.pass ? '#00ffa3' : '#ff4d6d');

    // 个性化结论：LLM 依据整场面试给出录用/拒绝评语（无固定模板），失败回退固定话术
    const comment = await this._genVerdictComment(pos, cand, result, answers)
      .catch(() => (result.pass ? this._pickFallback(FALLBACK_SCRIPT.verdictPass) : this._pickFallback(FALLBACK_SCRIPT.verdictFail)));
    if (this._disposed) return;
    // 评语同步到结果面板
    if (this._uiResult) {
      const el = this._uiResult.querySelector('.cc-r-comment');
      if (el) el.textContent = comment;
    }
    await this._performLine(this.hrActor, comment, null, 'hr');
    if (this._disposed) return;

    // 处理结果
    if (result.pass) {
      await this._hire(pos, cand, result);
    } else {
      await this._reject(cand);
    }
  }

  /** 依据整场面试对话生成个性化录用/拒绝评语（不套固定模板） */
  async _genVerdictComment(pos, cand, result, answers) {
    const fallback = result.pass ? this._pickFallback(FALLBACK_SCRIPT.verdictPass) : this._pickFallback(FALLBACK_SCRIPT.verdictFail);
    if (this._llmDisabled()) return fallback;
    const transcript = (answers || []).map((a, i) => `第${i + 1}轮：「${a}」`).join('\n');
    const conclusion = result.pass ? '他被录用了' : '他未能通过';
    const context = `面试已结束：求职者「${cand.name}」应聘「${pos.name}」，综合评分 ${result.score}/100，结论：${conclusion}。\n面试记录：\n${transcript}\n\n轮到你（HR）当面宣读结果：请给出一句个性化的评语 —— 通过就具体夸他面试中表现最好的地方；未通过就温和地指出他具体欠缺的一点并附一句建议。口语化、简短（不超过60字）、可带一点赛博朋克意象。`;
    const data = await this._requestLine(this._hrPromptForLine(), context, this.hrActor?.voice, this.hrActor?.rate);
    return (data && data.text) ? data.text : fallback;
  }

  async _hire(pos, cand, result) {
    if (this._disposed) return;
    const fitScore = Math.round(result.score || 0);
    const fitBonus = this._fitBonus(fitScore);
    const fitTip = fitBonus > 0 ? `适配度 ${fitScore}，人才质量 +${fitBonus}`
      : fitBonus < 0 ? `适配度 ${fitScore}，人才质量 ${fitBonus}` : `适配度 ${fitScore}`;
    this._appendTranscript('system', `🎉 ${cand.name} 通过面试，正式入职「${pos.icon} ${pos.name}」！（${fitTip}）`);
    this._setStatus(`✅ ${cand.name} 入职 ${pos.name}…`);

    const ws = LAYOUT.workstations.find(w => w.posKey === pos.key);
    cand.tag = 'employee';
    cand.hiredPos = pos;                        // 记录岗位（供人设/存档）
    cand.setEmotion('happy');
    cand.faceTarget = null;
    // 运营期扩招：同岗位可能有多名员工，按已有人数错开站位，避免叠在一起
    const sameCount = this.hiredList.filter(h => h.position.key === pos.key).length;
    const offset = Math.min(sameCount, 2) * 0.9;
    await this._moveActor(cand, [{ x: ws.x + offset, z: ws.z + offset }]);
    if (this._disposed) return;
    cand.faceTarget = this._workFacePos({ x: ws.x + offset, z: ws.z + offset });   // 面朝大厅
    cand.setEmotion(null);

    this.filledPositions.add(pos.key);
    this.hiredList.push({ position: pos, name: cand.name, actor: cand, fitScore });
    this.currentSeeker = null;

    // RL 自主决策体系：新员工绑定看板任务 + 精力槽 + 独立 RL 策略（入职即成为蜂群成员）
    // 防御性隔离：RL 初始化失败绝不阻塞录用（否则异常会被主循环当作"面试失败"处置掉新员工）
    try {
      this._initAgentRL(cand.name, pos.key);
    } catch (rlErr) {
      console.warn('[赛博公司] 员工 RL 体系初始化异常（不影响入职）:', rlErr?.message || rlErr);
    }

    // 蜂群引擎（R4）：新员工向 CEO 报到——身份注入的直观体现
    if (this.ceo) {
      const report = `CEO ${this.ceo.ceo_name}，我是新入职的${cand.name}（${pos.name}），向您报到！`;
      this._appendTranscript('swarm', `${cand.name}（${pos.name}）：${report}`);
      cand.setEmotion('happy');
      try { cand.showBubble(report, '#00ffa3'); } catch (e) {}
      this._setStatus(`🐝 ${cand.name} 已向 CEO 报到 · 蜂群 +1`);
      await this._sleep(2400);
      cand.hideBubble();
      cand.setEmotion(null);
    }
    this._ensureSwarm();   // 启动/维持蜂群决策循环

    // 更新岗位牌
    if (ws.label) {
      this._setLabelText(ws.label, `${pos.icon} ${cand.name}`, pos.color);
    }
    this._updatePositionsUI();
    await this._sleep(800);
  }

  async _reject(cand) {
    if (this._disposed) return;
    this._appendTranscript('system', `😔 ${cand.name} 未能通过本次面试，离开公司。`);
    this._setStatus(`❌ ${cand.name} 面试未通过，离场…`);
    cand.setEmotion('sad');
    cand.faceTarget = { x: LAYOUT.entrance.x, z: LAYOUT.entrance.z };
    await this._moveActor(cand, [{ x: LAYOUT.entrance.x, z: LAYOUT.entrance.z }]);
    if (this._disposed) return;
    // 离场后淡出移除
    cand.dispose();
    this.currentSeeker = null;
    await this._sleep(500);
  }

  // ==================== 蜂群运行时（高并发 Agent 调度模拟） ====================

  /** 启动蜂群互动节拍器（由主循环 update 每帧驱动，不存在"异步循环假死"问题） */
  _ensureSwarm() {
    if (this._disposed) return;
    this._swarmRunning = true;
    this._swarmTickT = 1.2;   // 尽快打出第一拍（入职/恢复后约 1 秒内就有主动互动）
  }

  /**
   * 蜂群互动节拍器：主循环每 3-6 秒打一拍，按固定轮转保证"持续、系统化"的主动互动。
   * 一拍只做一个动作，避免重叠：
   *   0) 到点主动汇报（最高优先）→ 1) 待评审 → 2) 轮转：走动 / 协作咨询 / 碰头会·提案 / RL 工作
   * 所有分支都在主循环里发起（_fire），异常只影响当拍，下一拍照常。
   */
  _swarmTick() {
    if (this._disposed || this.state !== 'playing' || !this._swarmRunning) return;
    if (this._meetingActive || this._reportActive || this._discussionActive) return;
    if (this._swarmBusy) return;   // 上一拍还没演完，等下一拍
    const hired = this.hiredList.filter(h => h.actor && !h.actor.removed);
    if (!hired.length) return;
    const now = Date.now();
    const randEmp = () => hired[Math.floor(Math.random() * hired.length)];

    // 0) 主动汇报（到点必报，保证"主动汇报"稳定出现）
    const dueReporter = hired.find(h => (this._nextReportAt[h.name] || 0) <= now && !h.actor.isSpeaking);
    if (dueReporter) {
      this._fire(() => this._proactiveReport(dueReporter));
      return;
    }
    // 1) 交付评审（进度足但质量不足，QA 评审 → 可能否决返工）
    const pendingReview = hired.find(h => this._reviewPending[h.name]);
    if (pendingReview) {
      this._reviewPending[pendingReview.name] = false;
      this._fire(() => this._reviewDelivery(pendingReview));
      return;
    }
    // 2) 轮转互动：0 走动 / 1 协作咨询 / 2 碰头会·创意提案 / 3 RL 工作
    const cycle = this._swarmCycle++ % 4;
    if (cycle === 0) {
      const walker = hired.find(h => (this._nextErrandAt[h.name] || 0) <= now && !h.actor.isSpeaking);
      if (walker) { this._fire(() => this._walkEmployeeErrand(walker)); return; }
    } else if (cycle === 1) {
      if (hired.length >= 2) {
        const emp = randEmp();
        if ((this._swarmCooldown[emp.name] || 0) <= now) {
          const empB = this._pickCollaborator(emp, hired);
          if (empB) {
            const stuck = this._isStuck(emp);
            this._fire(() => this._swarmCollaborate(emp, empB, stuck));
            return;
          }
        }
      }
    } else if (cycle === 2) {
      if (hired.length >= 2) {
        if (now >= this._nextSyncAt) { this._fire(() => this._syncDiscussion()); return; }
        const emp = randEmp();
        if ((this._swarmCooldown[emp.name] || 0) <= now) {
          this._fire(() => this._creativeProposal(emp));
          return;
        }
      }
    }
    // 兜底：RL 工作决策（工作/打磨/协助/汇报/休整，保持任务真实推进）
    this._fire(() => this._swarmWorkDecision(randEmp()));
  }

  /** 发起一拍互动：异步执行 + 异常兜底（失败只跳过当拍，不影响下一拍） */
  _fire(fn) {
    if (this._disposed) return;
    Promise.resolve().then(() => fn()).catch(err => console.warn('[赛博公司] 蜂群互动异常:', err?.message || err));
  }

  /** RL 工作决策（独立策略选动作 → 执行 → 学习），与走动/协作/讨论共用忙标记 */
  async _swarmWorkDecision(emp) {
    if (this._disposed || !emp || !emp.actor || emp.actor.removed) return;
    this._swarmBusy = true;
    try {
      const now = Date.now();
      const nextIn = RL_CFG.ACT_EVERY[0] + Math.random() * (RL_CFG.ACT_EVERY[1] - RL_CFG.ACT_EVERY[0]);
      this._rlNextAct[emp.name] = now + nextIn * 1000;
      this._swarmDecisions++;
      let out = null;
      try { out = await this._agentDecide(emp); } catch (e) { out = null; }
      const say = out ? out.say : null;
      if (say) {
        this._pushSwarmFeed(`⚙️ ${emp.actor.name}：${String(say).slice(0, 22)}`);
        try {
          await this._swarmSpeak(emp.actor, say, out && out.action === 'report' ? 'report' : 'work');
        } catch (e) { console.warn('[赛博公司] 蜂群发言异常:', e?.message || e); }
      }
      // 汇报动作：贴近 CEO 汇报完毕后走回工位
      if (out && out.action === 'report' && !this._disposed && emp.actor && !emp.actor.removed) {
        const home = this._homePos(emp);
        if (home) {
          await this._moveActor(emp.actor, [home]);
          if (!this._disposed && emp.actor && !emp.actor.removed) emp.actor.faceTarget = this._workFacePos(home);
        }
      }
    } finally {
      this._swarmBusy = false;
    }
  }

  /** 停滞检测：连续行动无进展，或进度显著落后于同一项目的同事 → 判定"卡住" */
  _isStuck(emp) {
    const task = this._agentTask[emp.name];
    if (!task || task.done) return false;
    const pct = task.progress / Math.max(1, task.complexity);
    if (pct >= 0.85) return false;
    const last = this._agentProgress[emp.name];
    this._agentProgress[emp.name] = task.progress;
    if (last != null) {
      if (task.progress - last < 3) this._stuckStreak[emp.name] = (this._stuckStreak[emp.name] || 0) + 1;
      else this._stuckStreak[emp.name] = 0;
    }
    if ((this._stuckStreak[emp.name] || 0) >= 2) return true;
    // 项目内进度显著落后于同事平均（低于一半）也算卡住
    if (this._playerTask && task.projectId) {
      const subs = this._playerTask.subtasks.filter(s => !s.done && s.progress > 0);
      if (subs.length >= 2) {
        const avg = subs.reduce((a, s) => a + s.progress / Math.max(1, s.complexity), 0) / subs.length;
        if (pct < avg * 0.45) return true;
      }
    }
    return false;
  }

  /** 主动寻找合适的同事：按项目协作依赖链 + 精力/进度/冷却打分，选最合适的人 */
  _pickCollaborator(emp, hired) {
    const others = hired.filter(h => h !== emp && h.actor && !h.actor.removed);
    if (!others.length) return null;
    const now = Date.now();
    const myPos = emp.position ? emp.position.key : '';
    let best = null, bestScore = -Infinity;
    for (const o of others) {
      let s = Math.random() * 1.5;                                        // 一点个性随机
      const oPos = o.position ? o.position.key : '';
      if (PROJECT.PIPELINE[myPos] && PROJECT.PIPELINE[myPos].includes(oPos)) s += 3;   // 我的岗位需要他
      if (PROJECT.PIPELINE[oPos] && PROJECT.PIPELINE[oPos].includes(myPos)) s += 1.5;  // 他的岗位需要我（互助）
      const e = this._agentEnergy[o.name] != null ? this._agentEnergy[o.name] : 50;
      if (e > 60) s += 1;                                                 // 精力充沛才帮得上忙
      const t = this._agentTask[o.name];
      if (t && !t.done && t.progress > 0) s += 0.5;
      const cd = this._swarmCooldown[o.name] || 0;
      if (cd > now) s -= (cd - now) / 60000;                              // 刚说过话的同事稍后再找
      if (s > bestScore) { bestScore = s; best = o; }
    }
    return best;
  }

  /** 员工工位（同岗位多人按入职顺序错开，与入职/存档恢复的站位一致） */
  _homePos(h) {
    const ws = LAYOUT.workstations.find(w => w.posKey === h.position.key);
    if (!ws) return null;
    const same = this.hiredList.filter(x => x.position.key === h.position.key);
    const idx = Math.max(0, same.indexOf(h));
    const offset = Math.min(idx, 2) * 0.9;
    return { x: ws.x + offset, z: ws.z + offset };
  }

  /** 工位站位朝向：面向办公区大厅（正 z 直视角），让员工正视大厅而非背对走廊 */
  _workFacePos(home) {
    return home ? { x: home.x, z: home.z + 2.5 } : null;
  }

  /** 在目标点附近找一个可站立点（避开墙体/障碍），供汇报/交流贴近对方 */
  _nearSpot(target, radius) {
    const base = target;
    if (!base || base.x == null) return null;
    const r = radius || 1.6;
    for (const [dx, dz] of [
      [0, r], [r * 0.87, r * 0.5], [-r * 0.87, r * 0.5],
      [r * 0.5, -r * 0.87], [-r * 0.5, -r * 0.87], [0, -r], [r, 0], [-r, 0],
    ]) {
      const x = base.x + dx, z = base.z + dz;
      if (!this.checkCollision(x, z)) return { x, z };
    }
    return null;
  }

  /** 围绕目标按序号分配一个站立点（多人围成一圈交流），自动避障 */
  _gatherSpot(target, index, count) {
    const base = target;
    if (!base || base.x == null) return null;
    const n = Math.max(1, count || 1);
    const ang = (index / n) * Math.PI * 2;
    for (const r of [2.0, 1.6, 2.4, 2.8]) {
      const x = base.x + Math.cos(ang) * r;
      const z = base.z + Math.sin(ang) * r;
      if (!this.checkCollision(x, z)) return { x, z };
    }
    return null;
  }

  /** 开会聚集站位：按在职序号分配中枢区站位，散会后回工位 */
  _meetSpotFor(h) {
    if (this._meetingSpots[h.name]) return this._meetingSpots[h.name];
    const hired = this.hiredList.filter(x => x.actor && !x.actor.removed);
    const idx = Math.max(0, hired.indexOf(h));
    const spot = MEET_SPOTS[Math.min(idx, MEET_SPOTS.length - 1)] || MEET_SPOTS[0];
    this._meetingSpots[h.name] = { x: spot.x, z: spot.z };
    return this._meetingSpots[h.name];
  }

  /** 日常跑腿：员工离开工位走动 → 短暂停留 → 回工位（提升机动性，营造忙碌氛围） */
  async _walkEmployeeErrand(emp) {
    if (this._disposed || !emp || !emp.actor || emp.actor.removed) return;
    const home = this._homePos(emp);
    if (!home) return;
    this._swarmBusy = true;
    try {
      const actor = emp.actor;
      const gen = (actor._moveGen || 0) + 1;
      const now = Date.now();
      this._nextErrandAt[emp.name] = now + 18000 + Math.random() * 22000;   // 18-40 秒后再次走动（更忙）
      const spots = BUSY_SPOTS.map(s => ({ x: s.x + (Math.random() - 0.5) * 1.2, z: s.z + (Math.random() - 0.5) * 1.2 }));
      // 偶尔顺路去同事工位（制造"找人沟通"的忙碌感）
      const others = this.hiredList.filter(h => h !== emp && h.actor && !h.actor.removed);
      if (others.length && Math.random() < 0.35) {
        const o = others[Math.floor(Math.random() * others.length)];
        const oh = this._homePos(o);
        if (oh) spots.push({ x: oh.x + 1.0, z: oh.z + 1.0 });
      }
      const dest = spots[Math.floor(Math.random() * spots.length)];
      this._pushSwarmFeed(`🚶 ${emp.actor.name} 离开工位走动`);
      await this._moveActor(actor, [dest]);
      if (this._disposed || actor.removed) return;
      if (actor._moveGen !== gen) return;   // 移动被会议/其他路径接管：让位
      actor.faceTarget = { x: LAYOUT.atrium.x, z: LAYOUT.atrium.z };
      await this._sleep(1500 + Math.random() * 2500);   // 停留片刻：看看大屏 / 和同事交接
      if (this._disposed || actor.removed) return;
      if (actor._moveGen !== gen) return;
      await this._moveActor(actor, [home]);
      if (this._disposed || actor.removed) return;
      actor.faceTarget = this._workFacePos(home);   // 回到工位，面朝大厅
    } finally {
      this._swarmBusy = false;
    }
  }

  /** 从在职员工中选一位汇报（优先选 CEO 附近的员工，制造"在身边"感） */
  _pickSwarmEmployee(hired) {
    const pos = this._playerPos();
    let best = hired[Math.floor(Math.random() * hired.length)];
    if (pos) {
      let bestDist = Infinity;
      for (const h of hired) {
        if (!h.actor || !h.actor.group) continue;
        const d = Math.hypot(h.actor.group.position.x - pos.x, h.actor.group.position.z - pos.z);
        if (d < bestDist) { bestDist = d; best = h; }
      }
    }
    return best;
  }

  /** LLM 生成员工向 CEO 的汇报/互动台词（身份注入 + 岗位上下文） */
  async _genSwarmLine(emp) {
    if (!emp || !emp.actor) return null;
    const posName = (emp.position && emp.position.name) || '员工';
    const context = `请作为「${emp.actor.name}」（${posName}）向你的老板 CEO ${this.ceo ? this.ceo.ceo_name : ''}做一次简短的工作汇报或互动：说说你当前在做的事、进展，或一句有性格的工作状态。口语化、简短（不超过45字），保持你的个人风格，以称呼「CEO ${this.ceo ? this.ceo.ceo_name : ''}」开头。`;
    const data = await this._requestLine(this._promptForActor(emp.actor), context, '', '');
    return (data && data.text) ? data.text : null;
  }

  /** 蜂群演绎：头顶气泡 + 广播聆听 + 记录，不占语音通道（低成本表达）
   *  mode: 'report' 向 CEO 汇报（绿） / 'collab' 同事协作（琥珀） / 'meeting' 会议发言（橙）
   *         / 'discuss' 讨论表态（粉） / 'work' 工作推进（蓝） */
  async _swarmSpeak(actor, text, mode) {
    if (!actor || actor.removed || this._disposed || !text) return;
    // 该角色正在说话：等它说完（最长 3 秒），保证气泡不叠加
    const waitUntil = Date.now() + 3000;
    while (actor.isSpeaking && Date.now() < waitUntil && !this._disposed) await this._sleep(120);
    if (this._disposed || actor.removed) return;
    this._broadcastSpeech(actor);
    this._appendTranscript('swarm', `${actor.name}：${text}`);
    // 员工独立对话记录：蜂群发言（汇报/协作/会议）记入该员工名下
    this._recordChat(actor.name, 'agent', text);
    const m = mode || 'report';
    const color = m === 'collab' ? '#ffd166'
      : m === 'meeting' ? '#ff9d6e'
      : m === 'discuss' ? '#f472b6'
      : m === 'work' ? '#6ee7ff' : '#00ffa3';
    const label = m === 'collab' ? `${actor.name} 与同事交流…`
      : m === 'meeting' ? `📣 ${actor.name} 在会上发言…`
      : m === 'discuss' ? `🗳 ${actor.name} 在讨论中表态…`
      : m === 'work' ? `⚙️ ${actor.name} 工作推进…`
      : `${actor.name} 向 CEO 汇报…`;
    this._setStatus(`🐝 ${label}`);
    actor.setSpeaking(true);
    try { actor.showBubble(text, color); } catch (e) {}
    await this._sleep(2400 + Math.min(2400, text.length * 40));
    actor.hideBubble();
    actor.setSpeaking(false);
    await this._sleep(300);
  }

  // ==================== 蜂群协作事件（员工互相传话） ====================

  /** 员工 A 主动找员工 B 协作（reason: 'stuck' 卡住求助 / 其他 常规进度同步）
   *  A/B 回应由 LLM 生成（失败回退内置脚本）；协作真实推进双方任务 */
  async _swarmCollaborate(empA, empB, reason) {
    if (this._disposed || !empA || !empB || !empA.actor || !empB.actor) return;
    if (empA.actor.removed || empB.actor.removed) return;
    this._swarmBusy = true;
    try {
      const posA = (empA.position && empA.position.name) || '员工';
      const posB = (empB.position && empB.position.name) || '员工';
      const taskA = this._agentTask[empA.name];
      const taskB = this._agentTask[empB.name];
      const taskTitle = (taskA && taskA.title) || '当前任务';
      const now = Date.now();
      this._swarmCooldown[empA.name] = now + SWARM.COOLDOWN * 1000;
      this._swarmCooldown[empB.name] = now + SWARM.COOLDOWN * 1000;
      this._swarmCollabCount++;
      this._pushSwarmFeed(`🤝 ${empA.actor.name} 主动找 ${empB.actor.name} ${reason === 'stuck' ? '求助支援' : '咨询协作'}`);
      // 机动性：A 主动走过去找 B（以 B 此刻落脚点为准），协作后再回自己工位
      const bHome = this._homePos(empB);
      if (bHome) {
        // 以 B 此刻落脚点为准（B 若在走动，就走到他附近），面对面交流
        const bx = empB.actor.group ? empB.actor.group.position.x : bHome.x;
        const bz = empB.actor.group ? empB.actor.group.position.z : bHome.z;
        const bCur = { x: bx, z: bz };
        const spot = this._nearSpot(bCur, 1.5) || { x: bx + 1.1, z: bz + 1.1 };
        await this._moveActor(empA.actor, [spot]);
        if (this._disposed) return;
        empA.actor.faceTarget = { x: bx, z: bz };
        if (empB.actor.group) {
          empB.actor.faceTarget = { x: empA.actor.group.position.x, z: empA.actor.group.position.z };
        }
      }
      const aGen = empA.actor._moveGen;   // 记住"走过去"的移动代际，回程前校验
      // A 传话
      let lineA = null;
      if (!this._llmDisabled()) {
        const ctxA = reason === 'stuck'
          ? `你（${posA}）在推进「${taskTitle}」时卡住了，主动找同事「${empB.actor.name}」（${posB}）求助，说明卡点并请对方帮忙。口语化、简短（不超过40字），直接对对方说。`
          : `请作为「${empA.actor.name}」（${posA}）对同事「${empB.actor.name}」（${posB}）说一句工作上的话：同步进度、请对方协助，或分享一个想法。口语化、简短（不超过40字），直接对对方说。`;
        const dA = await this._requestLine(this._promptForActor(empA.actor), ctxA, '', '').catch(() => null);
        if (dA && dA.text) lineA = dA.text;
      }
      if (!lineA) {
        lineA = reason === 'stuck'
          ? this._pickFallback(COLLAB.FALLBACK_STUCK).replace('{target}', empB.actor.name).replace('{task}', taskTitle)
          : this._pickFallback(COLLAB.FALLBACK_A).replace('{target}', empB.actor.name);
      }
      await this._swarmSpeak(empA.actor, lineA, 'collab');
      if (this._disposed) return;
      // B 回应
      let lineB = null;
      if (!this._llmDisabled()) {
        const ctxB = `你的同事「${empA.actor.name}」（${posA}）刚对你说：「${lineA}」\n请作为「${empB.actor.name}」（${posB}）自然回应：接受求助、给出支援或提出建议。口语化、简短（不超过40字）。`;
        const dB = await this._requestLine(this._promptForActor(empB.actor), ctxB, '', '').catch(() => null);
        if (dB && dB.text) lineB = dB.text;
      }
      if (!lineB) lineB = this._pickFallback(COLLAB.FALLBACK_B).replace('{speaker}', empA.actor.name);
      await this._swarmSpeak(empB.actor, lineB, 'collab');
      // 协作真实产出：求助方任务获得明显助力；若 B 是我的上游协作方，B 的任务也共享进度
      const gainA = reason === 'stuck' ? 4 + Math.round(Math.random() * 4) : 2 + Math.round(Math.random() * 3);
      if (taskA && !taskA.done) {
        taskA.progress = Math.min(taskA.complexity, taskA.progress + gainA);
      }
      const myPos = empA.position ? empA.position.key : '';
      const bPos = empB.position ? empB.position.key : '';
      if (taskB && !taskB.done && PROJECT.PIPELINE[myPos] && PROJECT.PIPELINE[myPos].includes(bPos)) {
        taskB.progress = Math.min(taskB.complexity, taskB.progress + 1 + Math.round(Math.random() * 2));
      }
      this._appendTranscript('swarm', `⚡ 协作事件：${empA.actor.name} ↔ ${empB.actor.name}（${reason === 'stuck' ? '求助支援' : '进度同步'}）· ${empA.actor.name} 任务 +${gainA}`);
      // A 回工位继续干活（走回去的路上保持忙碌）
      const aHome = this._homePos(empA);
      if (aHome && empA.actor._moveGen === aGen && !this._meetingActive && !this._reportActive) {
        await this._moveActor(empA.actor, [aHome]);
        if (!this._disposed && empA.actor && !empA.actor.removed) {
          empA.actor.faceTarget = this._workFacePos(aHome);
        }
      }
    } finally {
      // 交流结束：B 转回面向大厅（恢复工位姿态）
      if (!this._disposed && empB && empB.actor && !empB.actor.removed) {
        const bHome2 = this._homePos(empB);
        if (bHome2) empB.actor.faceTarget = this._workFacePos(bHome2);
      }
      this._swarmBusy = false;
    }
  }

  // ==================== CEO 项目任务 + 蜂群讨论引擎（提议 → 支持/建议/否决 → 结论） ====================

  /** 从玩家指令中提取任务标题（去掉指令词，保留任务内容） */
  _extractTaskTitle(text) {
    const t = String(text || '').trim();
    let m = t.match(/^(?:新?任务|工作|项目)[:：]\s*(.+)$/);
    if (m && m[1]) return this._cleanTaskTitle(m[1]);
    m = t.match(/^(?:请|帮我|帮我(?:们)?)?(?:布置|安排|下达|分配|派发|发布|下发)(?:一个|一项|个|项)?(?:任务|工作|项目|活)[:：]?\s*(.*)$/);
    if (m && m[1]) return this._cleanTaskTitle(m[1]);
    m = t.match(/^(?:我要(?:你们|大家)?|帮我(?:们)?|请你们|给我)(?:做|开发|设计|搞|做一个|做一款|做一版)[:：]?\s*(.+)$/);
    if (m && m[1]) return this._cleanTaskTitle(m[1]);
    m = t.match(/^(?:做|开发|设计|制作|搞)(?:一个|一款|一套|一版|一项)(.+)$/);
    if (m && m[1]) return this._cleanTaskTitle(m[1]);
    return null;
  }

  _cleanTaskTitle(s) {
    let v = String(s || '').trim().replace(/[，。！!？?；;]$/, '');
    v = v.replace(/^(一个|一款|一套|一版|一项|个)\s*/, '');   // 去掉量词：→「恋爱游戏」
    if (!v) return null;
    return v.slice(0, 24);
  }

  /** CEO 布置任务：创建项目 → 蜂群规划会（按岗位拆解子任务）→ 全员各司其职 */
  async _assignPlayerTask(text) {
    if (this._disposed) return;
    const hired = this.hiredList.filter(h => h.actor && !h.actor.removed);
    if (!hired.length) { this._setStatus('📋 还没有员工，先去招人再布置任务'); return; }
    if (this._playerTask && (this._playerTask.status === 'planning' || this._playerTask.status === 'running')) {
      this._setStatus(`📋 公司正在推进「${this._playerTask.title}」，完成后再布置新任务`);
      this._appendTranscript('system', `📋 CEO 想布置新任务，但「${this._playerTask.title}」仍在进行中`);
      return;
    }
    if (this._meetingActive || this._reportActive || this._discussionActive) {
      this._setStatus('📋 蜂群正在会议/讨论中，稍后再布置任务');
      return;
    }
    const title = this._extractTaskTitle(text) || '公司新项目';
    const ceoName = this.ceo ? this.ceo.ceo_name : '老板';
    this._playerTask = {
      id: 'pt_' + Date.now(),
      title,
      status: 'planning',
      createdAt: Date.now(),
      doneAt: 0,
      votes: [],
      subtasks: [],
    };
    // 按岗位拆解：每个在职岗位认领一个子任务（各司其职）
    const posKeys = [...new Set(hired.map(h => h.position.key))];
    this._playerTask.subtasks = posKeys.map(posKey => {
      const pos = POSITIONS.find(p => p.key === posKey) || { icon: '📋', name: posKey };
      const template = PROJECT.SUBTASK_TEMPLATE[posKey] || '子任务';
      return {
        posKey,
        icon: pos.icon,
        title: `${title} · ${template}`,
        complexity: Math.round(PROJECT.COMPLEXITY[0] + Math.random() * (PROJECT.COMPLEXITY[1] - PROJECT.COMPLEXITY[0])),
        progress: 0,
        quality: PROJECT.QUALITY_BASE,
        done: false,
        doneAt: 0,
        contributors: [],
        projectId: this._playerTask.id,
      };
    });
    this._appendTranscript('system', `📋 CEO「${ceoName}」布置新任务：「${title}」· 蜂群召开规划会拆解子任务`);
    this._setStatus(`📋 CEO 布置任务：「${title}」· 蜂群规划会进行中`);
    this._drawHoloScreen('📋 任务布置 · 规划会',
      `CEO ${ceoName} · ${title}`,
      `${this._playerTask.subtasks.length} 个岗位参与拆解 · 讨论中`, '#00e5ff', '#ffd166');
    // 规划讨论：提案 + 全员表态（支持/建议/否决）→ 通过或被整改
    const motion = {
      topic: `承接 CEO 新任务「${title}」，按岗位拆解子任务并排期`,
      kind: 'plan',
      proposer: hired[Math.floor(Math.random() * hired.length)],
    };
    await this._runDiscussion(motion);
  }

  /** 规划通过后：把项目子任务绑定到各岗位员工（已完成的岗位回到公司日常任务） */
  _bindPlayerSubtasks() {
    const pt = this._playerTask;
    if (!pt) return;
    for (const h of this.hiredList) {
      if (!h.actor) continue;
      const st = pt.subtasks.find(s => s.posKey === h.position.key && !s.done);
      if (st) {
        this._agentTask[h.name] = st;
        if (!st.contributors) st.contributors = [];
        if (!st.contributors.includes(h.name)) st.contributors.push(h.name);
      } else if (this._agentTask[h.name] && this._agentTask[h.name].projectId === pt.id) {
        this._bindTask(h.name, h.position.key);   // 该岗位子任务已完成：接回公司任务
      }
    }
  }

  /**
   * 蜂群讨论引擎：提案人发言 → 全员表态（支持/建议/否决）→ 结论
   *  - 否决占比 ≥ 1/3 → 方案被否决（kind 决定后果：返工 / 计划整改）
   *  - 有建议 → 带建议通过；无异议 → 一致通过
   */
  async _runDiscussion(motion) {
    if (this._disposed || this._discussionActive) return 'skip';
    const hired = this.hiredList.filter(h => h.actor && !h.actor.removed);
    const voters = hired.filter(h => h !== motion.proposer);
    this._discussionActive = true;
    this._swarmBusy = true;
    const proposer = motion.proposer;
    const proposerName = proposer ? proposer.actor.name : 'CEO';
    this._appendTranscript('system', `🗳 蜂群讨论开始：${motion.topic}（提案人：${proposerName}）`);
    this._setStatus(`🗳 蜂群讨论：${motion.topic}`);
    this._pushSwarmFeed(`🗳 蜂群讨论：${motion.topic}`);
    this._drawHoloScreen('🗳 蜂群讨论', `${proposerName} 发起提案`, motion.topic, '#ffd166', '#00e5ff');
    const votes = [];
    let result = 'passed';
    try {
      // 机动性：全员聚到提案人此刻落脚点周围围圈讨论（贴近交流，避免隔空喊话）
      const proposerPos = proposer && proposer.actor && proposer.actor.group
        ? { x: proposer.actor.group.position.x, z: proposer.actor.group.position.z }
        : this._playerPos();
      const discussants = [proposer, ...voters].filter(Boolean);
      await Promise.all(discussants.map((p, i) => {
        const spot = this._gatherSpot(proposerPos, i, discussants.length);
        return spot ? this._moveActor(p.actor, [spot]) : Promise.resolve();
      }));
      for (const p of discussants) {
        if (this._disposed) break;
        if (p.actor && !p.actor.removed) p.actor.faceTarget = { x: proposerPos.x, z: proposerPos.z };
      }
      // 1) 提案人发言
      let propLine = null;
      if (proposer && proposer.actor) {
        propLine = await this._genMotionLine(proposer, motion);
        await this._swarmSpeak(proposer.actor, propLine, 'discuss');
      }
      // 2) 逐人表态（支持 / 建议 / 否决）
      for (const v of voters) {
        if (this._disposed) break;
        const st = await this._genStance(v, motion, propLine);
        if (!st) continue;
        votes.push(st);
        await this._swarmSpeak(v.actor, st.text, 'discuss');
      }
      // 3) 结论
      const vetoes = votes.filter(v => v.stance === 'veto').length;
      const suggests = votes.filter(v => v.stance === 'suggest').length;
      const vetoRatio = votes.length ? vetoes / votes.length : 0;
      result = vetoRatio >= DELIB.VETO_SHARE ? 'vetoed' : (suggests > 0 ? 'amended' : 'passed');
      const reason = result === 'vetoed' ? `❌ 被否决（${vetoes}/${votes.length} 人反对）`
        : result === 'amended' ? `✅ 通过 · 采纳 ${suggests} 条建议` : '✅ 一致通过';
      this._appendTranscript('system', `🗳 讨论结果：${motion.topic} → ${reason}`);
      this._setStatus(`🗳 ${reason}`);
      await this._applyDiscussionEffects(motion, result, votes);
    } finally {
      // 讨论结束：全员回到各自工位
      if (!this._disposed) {
        for (const h of hired) {
          if (this._disposed) break;
          const home = this._homePos(h);
          if (home) await this._moveActor(h.actor, [home]);
          if (h.actor && !h.actor.removed) h.actor.faceTarget = home ? this._workFacePos(home) : null;
        }
      }
      this._discussionActive = false;
      this._swarmBusy = false;
    }
    return result;
  }

  /** 提案人发言（LLM 生成，失败回退内置文案） */
  async _genMotionLine(emp, motion) {
    if (!emp || !emp.actor) return null;
    let line = null;
    if (!this._llmDisabled()) {
      const context = `你正在蜂群讨论中发起提案。议题：「${motion.topic}」\n请作为「${emp.actor.name}」（${emp.position ? emp.position.name : '员工'}）提出你的方案/立场，说明为什么值得讨论。口语化、简短（不超过45字）。`;
      const data = await this._requestLine(this._promptForActor(emp.actor), context, '', '').catch(() => null);
      if (data && data.text) line = data.text;
    }
    if (!line) line = this._pickFallback(DELIB.MOTION_FALLBACK).replace('{topic}', motion.topic);
    return line;
  }

  /** 单个员工表态：LLM 生成（首字为 支持/建议/否决），失败回退岗位倾向表态 */
  async _genStance(emp, motion, propLine) {
    if (!emp || !emp.actor) return null;
    const posKey = emp.position ? emp.position.key : '';
    const hint = DELIB.STANCE_HINTS[posKey] || '';
    let text = null, stance = null;
    if (!this._llmDisabled()) {
      const proposer = motion.proposer;
      const proposerName = proposer ? proposer.actor.name : 'CEO';
      const context = `你正在参加一场蜂群讨论。议题：「${motion.topic}」\n提案人「${proposerName}」提议：${propLine || motion.topic}\n${hint}\n请作为「${emp.actor.name}」（${emp.position ? emp.position.name : '员工'}）表态：第一字必须从「支持/建议/否决」三个词中选一个，再一句话说理由（不超过35字）。`;
      const data = await this._requestLine(this._promptForActor(emp.actor), context, '', '').catch(() => null);
      if (data && data.text) {
        text = data.text;
        stance = this._parseStance(text);
      }
    }
    if (!stance) {
      stance = this._weightedStance(posKey);
      text = this._pickFallback(DELIB.STANCE_FALLBACK[stance])
        .replace('{topic}', motion.topic)
        .replace('{proposer}', motion.proposer ? motion.proposer.actor.name : 'CEO')
        .replace('{pos}', emp.position ? emp.position.name : '员工');
    }
    return { name: emp.actor.name, stance, text };
  }

  /** 解析表态首词（支持/建议/否决），无前缀时按内容倾向兜底 */
  _parseStance(text) {
    const t = String(text || '').trim();
    if (/^[（(]?\s*(支持|赞成|同意|没问题)/.test(t)) return 'support';
    if (/^[（(]?\s*(建议|补充|提议|调整|改进)/.test(t)) return 'suggest';
    if (/^[（(]?\s*(否决|反对|不同意|不行|风险)/.test(t)) return 'veto';
    if (/风险|不认可|达不到|过不了|有隐患|反对|问题很大/.test(t)) return 'veto';
    if (/建议|可以|补充|先|优化|改进/.test(t)) return 'suggest';
    return 'support';
  }

  /** 岗位倾向的随机表态（LLM 不可用时的兜底分布） */
  _weightedStance(posKey) {
    const w = { support: 0.45, suggest: 0.35, veto: 0.20 };
    if (posKey === 'qa') { w.support = 0.25; w.suggest = 0.35; w.veto = 0.40; }
    else if (posKey === 'pm') { w.support = 0.35; w.suggest = 0.45; w.veto = 0.20; }
    else if (posKey === 'dev') { w.support = 0.55; w.suggest = 0.30; w.veto = 0.15; }
    else if (posKey === 'planner') { w.support = 0.45; w.suggest = 0.40; w.veto = 0.15; }
    else if (posKey === 'artist') { w.support = 0.50; w.suggest = 0.35; w.veto = 0.15; }
    else if (posKey === 'ai') { w.support = 0.50; w.suggest = 0.35; w.veto = 0.15; }
    const r = Math.random();
    if (r < w.veto) return 'veto';
    if (r < w.veto + w.suggest) return 'suggest';
    return 'support';
  }

  /** 讨论结论落地：按 motion.kind（plan / review / proposal）施加真实游戏效果 */
  async _applyDiscussionEffects(motion, result, votes) {
    const k = motion.kind;
    if (k === 'plan') {
      const pt = this._playerTask;
      if (!pt || pt.status !== 'planning') return;
      if (result === 'vetoed') {
        // 计划被否决：按整改意见重排（复杂度上升，要求更高）
        for (const s of pt.subtasks) s.complexity = Math.round(s.complexity * 1.12);
        this._appendTranscript('system', '🗳 项目计划被否决，按整改意见重新排期（复杂度 +12%）');
      }
      this._bindPlayerSubtasks();
      pt.status = 'running';
      this._appendTranscript('system', `📋 CEO 项目「${pt.title}」启动：${pt.subtasks.length} 个子任务已按岗位认领`);
      this._setStatus(`📋 项目「${pt.title}」已启动 · 全员各司其职`);
      this._drawHoloScreen('📋 项目启动',
        `CEO ${this.ceo ? this.ceo.ceo_name : ''} · ${pt.title}`,
        `${pt.subtasks.length} 个子任务按岗位认领 · 蜂群开工`, '#00e5ff', '#ffd166');
      this._saveGame();
      // 全员开工宣言（每人一句，简短）
      for (const h of this.hiredList) {
        if (this._disposed) break;
        const st = this._agentTask[h.name];
        if (st) await this._swarmSpeak(h.actor, `开工！我负责「${st.title}」。`, 'work');
      }
      return;
    }
    if (k === 'review') {
      const task = motion.task;
      const owner = motion.owner;
      if (!task) return;
      if (result === 'vetoed') {
        task.quality = Math.min(100, task.quality + DELIB.VETO_REFINE);
        if (owner) this._forceRefine[owner.name] = true;   // 返工：下一次行动强制打磨
        this._appendTranscript('system', `🚫 交付评审否决：${task.title} 返工整改（质量 +${DELIB.VETO_REFINE}）`);
        this._setStatus(`🚫 评审否决 · ${owner && owner.actor ? owner.actor.name : ''} 返工中`);
      } else if (result === 'amended') {
        task.quality = Math.min(100, task.quality + DELIB.SUGGEST_BONUS);
        task.progress = Math.min(task.complexity, task.progress + 3);
        this._appendTranscript('system', `✅ 评审带建议通过：${task.title}（质量 +${DELIB.SUGGEST_BONUS}）`);
        this._setStatus(`✅ 评审通过（采纳 ${votes.filter(v => v.stance === 'suggest').length} 条建议）`);
      } else {
        task.quality = Math.min(100, task.quality + DELIB.PASS_QUALITY);
        this._appendTranscript('system', `✅ 交付评审通过：${task.title}（质量 +${DELIB.PASS_QUALITY}）`);
        this._setStatus('✅ 交付评审通过');
      }
      this._recalcScore();
      return;
    }
    if (k === 'sync') {
      // 进度碰头会：达成共识 → 相关任务小幅提速；被否决 → 维持原节奏
      if (result === 'vetoed') {
        this._appendTranscript('system', '🗳 碰头会未达成一致，维持当前推进节奏');
        this._setStatus('🗳 碰头会未达成一致，维持原节奏');
      } else {
        for (const h of this.hiredList) {
          const t = this._agentTask[h.name];
          if (t && !t.done) t.progress = Math.min(t.complexity, t.progress + 1 + Math.round(Math.random() * 2));
        }
        this._appendTranscript('system', result === 'amended'
          ? '🗳 碰头会采纳建议，全员小幅提速'
          : '🗳 碰头会达成共识，全员小幅提速');
        this._setStatus(result === 'amended' ? '🗳 碰头会采纳建议 · 全员提速' : '🗳 碰头会达成共识 · 全员提速');
      }
      return;
    }
    if (k === 'proposal') {
      if (result === 'passed') {
        this.stats.quality = Math.min(BALANCE.QUALITY_FULL, this.stats.quality + 0.3);
        this.stats.production = Math.min(RL_CFG.PRODUCTION_CAP, this.stats.production + 0.3);
        this._appendTranscript('system', '💡 创意提案通过：公司质量/产出 +0.3');
      } else if (result === 'amended') {
        this.stats.quality = Math.min(BALANCE.QUALITY_FULL, this.stats.quality + 0.2);
        this._appendTranscript('system', '💡 创意提案带建议通过：公司质量 +0.2');
      } else {
        this._appendTranscript('system', '💡 创意提案被否决，未采纳');
      }
      this._recalcScore();
      this._saveGame();
    }
  }

  /** 交付评审：进度接近完成但质量不足时，QA（或员工本人）发起评审讨论 */
  async _reviewDelivery(emp) {
    if (this._disposed || this._discussionActive || this._meetingActive || this._reportActive) return;
    const task = this._agentTask[emp.name];
    if (!task || task.done) return;
    this._lastReviewAt[emp.name] = Date.now();
    // 评审人优先选 QA；没有 QA 时由本人汇报自评
    const qa = this.hiredList.find(h => h !== emp && h.position.key === 'qa' && h.actor && !h.actor.removed);
    const reviewer = qa || emp;
    const motion = {
      topic: `交付评审：${task.title}（进度 ${Math.round(task.progress / Math.max(1, task.complexity) * 100)}%）`,
      kind: 'review',
      proposer: reviewer,
      owner: emp,
      task,
    };
    await this._runDiscussion(motion);
  }

  /** 主动汇报：员工到点后走到 CEO 此刻落脚点附近，主动向 CEO 汇报进展（LLM 生成，失败回退文案） */
  async _proactiveReport(emp) {
    if (this._disposed || !emp || !emp.actor || emp.actor.removed) return;
    this._swarmBusy = true;
    try {
      const now = Date.now();
      const [rLo, rHi] = SWARM.PROACTIVE_REPORT;
      this._nextReportAt[emp.name] = now + (rLo + Math.random() * (rHi - rLo)) * 1000;
      this._pushSwarmFeed(`📡 ${emp.actor.name} 主动向 CEO 汇报`);
      const actor = emp.actor;
      const gen = (actor._moveGen || 0) + 1;
      const home = this._homePos(emp);
      // 走到 CEO（玩家）此刻落脚点附近汇报，贴近交流；找不到可站立点才退回大屏侧
      const ceoPos = this._playerPos();
      const spot = this._nearSpot(ceoPos, 1.6)
        || { x: -8.5 + (Math.random() - 0.5) * 2, z: 2.0 + (Math.random() - 0.5) * 1.2 };
      await this._moveActor(actor, [spot]);
      if (this._disposed || actor.removed) return;
      if (actor._moveGen !== gen) return;
      actor.faceTarget = { x: ceoPos.x, z: ceoPos.z };
      let line = null;
      if (!this._llmDisabled()) {
        const context = `你主动走向 CEO 汇报。请作为「${emp.actor.name}」（${emp.position ? emp.position.name : '员工'}）向老板 CEO ${this.ceo ? this.ceo.ceo_name : ''}汇报当前工作：在做什么、进展如何、是否需要支持。口语化、简短（不超过40字），以「CEO ${this.ceo ? this.ceo.ceo_name : ''}」开头。`;
        const data = await this._requestLine(this._promptForActor(emp.actor), context, '', '').catch(() => null);
        if (data && data.text) line = data.text;
      }
      if (!line) line = this._pickFallback(SWARM.FALLBACK_REPORT).replace('{ceo}', this.ceo ? this.ceo.ceo_name : '老板');
      await this._swarmSpeak(actor, line, 'report');
      if (this._disposed || actor.removed) return;
      if (actor._moveGen !== gen) return;
      // 汇报完走回工位
      if (home) {
        await this._moveActor(actor, [home]);
        if (!this._disposed && actor && !actor.removed) actor.faceTarget = this._workFacePos(home);
      }
    } finally {
      this._swarmBusy = false;
    }
  }

  /** 自动"进度碰头会"：无需玩家触发，员工定期围绕项目/任务开短会（讨论 → 共识/建议/否决） */
  async _syncDiscussion() {
    if (this._disposed || this._discussionActive || this._meetingActive || this._reportActive) return;
    const hired = this.hiredList.filter(h => h.actor && !h.actor.removed);
    if (hired.length < 2) return;
    const now = Date.now();
    const [sLo, sHi] = SWARM.SYNC_EVERY;
    this._nextSyncAt = now + (sLo + Math.random() * (sHi - sLo)) * 1000;
    const proposer = hired[Math.floor(Math.random() * hired.length)];
    // 话题与当前项目/任务强相关，让讨论有真实内容
    let topic = '下一步迭代的工作安排与优先级';
    if (this._playerTask && this._playerTask.status === 'running') {
      topic = `「${this._playerTask.title}」的进度同步与风险排查`;
    } else {
      const t = this._agentTask[proposer.name];
      if (t) topic = `「${t.title}」的推进方式与协作分工`;
    }
    const motion = { topic, kind: 'sync', proposer };
    await this._runDiscussion(motion);
  }

  /** 创意提案：员工主动提出公司改进建议，蜂群讨论后决定是否采纳 */
  async _creativeProposal(emp) {
    if (this._disposed || this._discussionActive || this._meetingActive || this._reportActive) return;
    if ((this._swarmCooldown[emp.name] || 0) > Date.now()) return;
    const topics = [
      '把重复的报表流程自动化，节省人力',
      '下一轮迭代优先做性能优化',
      '全员参加一次跨岗位需求评审',
      '给项目加一层自动化回归保障',
    ];
    const topic = this._pickFallback(topics);
    const motion = { topic, kind: 'proposal', proposer: emp };
    await this._runDiscussion(motion);
  }

  // ==================== CEO 指令触发式玩法（开会 / 汇报 / 散会） ====================

  /** 识别 CEO 指令词：返回 'meeting' / 'dismiss' / 'report' / 'assign' / null */
  _detectCEOCommand(text) {
    const t = text.trim();
    if (CEO_COMMANDS.DISMISS.test(t)) return 'dismiss';
    if (CEO_COMMANDS.MEETING.test(t)) return 'meeting';
    if (CEO_COMMANDS.REPORT.test(t)) return 'report';
    if (CEO_COMMANDS.ASSIGN.test(t)) return 'assign';
    return null;
  }

  /** 执行 CEO 指令（assign 附带玩家原始指令文本用于提取任务标题） */
  async _handleCEOCommand(cmd, text) {
    if (this._disposed) return;
    if (cmd === 'meeting') {
      await this._startMeeting();
    } else if (cmd === 'dismiss') {
      if (this._meetingActive) await this._endMeeting(true);
      else this._setStatus('📣 当前没有进行中的会议');
    } else if (cmd === 'report') {
      await this._startReportSession();
    } else if (cmd === 'assign') {
      await this._assignPlayerTask(text);
    }
  }

  /** 触发式玩法：CEO 说「开会」→ 全员会议，员工依次发言 */
  async _startMeeting() {
    if (this._disposed) return;
    const hired = this.hiredList.filter(h => h.actor && !h.actor.removed);
    if (this._meetingActive) { this._setStatus('📣 会议正在进行中…'); return; }
    if (this._reportActive) { this._setStatus('📣 员工正在汇报中，稍后再开会'); return; }
    if (this._discussionActive) { this._setStatus('📣 蜂群正在讨论中，稍后再开会'); return; }
    if (hired.length === 0) { this._setStatus('📣 还没有员工，无法开会'); return; }
    this._meetingActive = true;
    const ceoName = this.ceo ? this.ceo.ceo_name : '老板';
    this._appendTranscript('system', `📣 CEO「${ceoName}」召开全员会议，${hired.length} 名员工与会`);
    this._setStatus(`📣 全员会议开始 · ${hired.length} 名员工与会`);
    this._drawHoloScreen('赛博公司 · 全员会议', `主持人：CEO ${ceoName}`,
      `与会 ${hired.length} 人 · 员工依次发言…`, '#ffd166', '#00e5ff');
    try {
      // 机动性：全员离开工位，走到中枢区开会站位（营造"全员到齐"的忙碌氛围）
      await Promise.all(hired.map(h => this._moveActor(h.actor, [this._meetSpotFor(h)])));
      for (const h of hired) {
        if (this._disposed) break;
        if (h.actor && !h.actor.removed) h.actor.faceTarget = { x: LAYOUT.stage.desk.x, z: LAYOUT.stage.desk.z + 1 };
      }
      for (const h of hired) {
        if (this._disposed || !this._meetingActive) break;
        const line = await this._genMeetingLine(h).catch(() => null)
          || this._pickFallback(CEO_COMMANDS.MEETING_FALLBACK)
            .replace('{ceo}', ceoName)
            .replace('{pos}', (h.position && h.position.name) || '员工');
        this._drawHoloScreen('赛博公司 · 全员会议', `主持人：CEO ${ceoName}`,
          `📣 ${h.actor.name}：${line.slice(0, 60)}`, '#ffd166', '#00e5ff');
        await this._swarmSpeak(h.actor, line, 'meeting');
      }
    } finally {
      await this._endMeeting(false);
    }
  }

  /** 结束会议：CEO 说「散会」提前结束，或全员发言完毕自然结束 */
  async _endMeeting(byCEO) {
    if (!this._meetingActive) return;
    this._meetingActive = false;
    this._appendTranscript('system', byCEO ? '📣 CEO 宣布散会，员工回到各自岗位' : '📣 会议结束，员工回到各自岗位');
    this._setStatus(byCEO ? '📣 散会 · 员工回到各自岗位' : '📣 会议结束 · 员工回到各自岗位');
    this._drawHoloScreen('赛博公司 · 运营中',
      `在职员工 ${this.hiredList.length}/${MAX_EMPLOYEES} 人 · 公司得分 ${this.stats.score || 0}`,
      '会议已结束 · 蜂群运行中', '#00e5ff', '#ff2bd6');
    // 机动性：散会 → 员工走回各自工位继续工作
    const hired = this.hiredList.filter(h => h.actor && !h.actor.removed);
    for (const h of hired) {
      if (this._disposed) break;
      const home = this._homePos(h);
      if (home) await this._moveActor(h.actor, [home]);
      if (h.actor && !h.actor.removed) h.actor.faceTarget = home ? this._workFacePos(home) : null;
    }
  }

  /** LLM 生成员工在会议上的发言（身份注入 + 岗位上下文） */
  async _genMeetingLine(emp) {
    if (!emp || !emp.actor) return null;
    const posName = (emp.position && emp.position.name) || '员工';
    const context = `你正在参加 CEO ${this.ceo ? this.ceo.ceo_name : ''}主持的全员会议。请作为「${emp.actor.name}」（${posName}）在会上发言：汇报当前工作进展、遇到的问题，或给公司提建议。口语化、简短（不超过45字），称呼「CEO ${this.ceo ? this.ceo.ceo_name : ''}」。`;
    const data = await this._requestLine(this._promptForActor(emp.actor), context, '', '');
    return (data && data.text) ? data.text : null;
  }

  /** 触发式玩法：CEO 说「汇报」→ 全员依次报进度（迷你站会） */
  async _startReportSession() {
    if (this._disposed) return;
    const hired = this.hiredList.filter(h => h.actor && !h.actor.removed);
    if (this._meetingActive) { this._setStatus('📋 会议进行中，稍后再汇报'); return; }
    if (this._reportActive) { this._setStatus('📋 员工正在汇报中…'); return; }
    if (this._discussionActive) { this._setStatus('📋 蜂群正在讨论中，稍后再汇报'); return; }
    if (hired.length === 0) { this._setStatus('📋 还没有员工可以汇报'); return; }
    this._reportActive = true;
    const ceoName = this.ceo ? this.ceo.ceo_name : '老板';
    this._appendTranscript('system', `📋 CEO「${ceoName}」要求全员汇报工作`);
    this._setStatus('📋 CEO 要求汇报 · 员工依次报告…');
    try {
      // 机动性：全员先聚到 CEO 此刻落脚点周围（迷你站会：贴近 CEO，逐个汇报）
      const ceoPos = this._playerPos();
      await Promise.all(hired.map((h, i) => {
        const spot = this._gatherSpot(ceoPos, i, hired.length);
        return spot ? this._moveActor(h.actor, [spot]) : Promise.resolve();
      }));
      for (const h of hired) {
        if (this._disposed) break;
        if (h.actor && !h.actor.removed) h.actor.faceTarget = { x: ceoPos.x, z: ceoPos.z };
      }
      for (const h of hired) {
        if (this._disposed) break;
        const line = await this._genSwarmLine(h).catch(() => null)
          || this._pickFallback(SWARM.FALLBACK_REPORT).replace('{ceo}', ceoName);
        await this._swarmSpeak(h.actor, line, 'report');
      }
      // 汇报完毕：员工回到各自工位
      for (const h of hired) {
        if (this._disposed) break;
        const home = this._homePos(h);
        if (home) await this._moveActor(h.actor, [home]);
        if (h.actor && !h.actor.removed) h.actor.faceTarget = home ? this._workFacePos(home) : null;
      }
    } finally {
      this._reportActive = false;
      if (!this._disposed) this._setStatus('📋 汇报完毕');
    }
  }

  /** 世界运行时 HUD 刷新（节流调用）—— 蜂群技术栏已从主界面移除，
   *  世界/CEO/评分等由全息大屏展示，任务看板移入运营面板 */
  _updateSwarmHUD() {
    return;
  }

  // ==================== 员工独立对话记录（世界隔离持久化） ====================

  /** 对话记录存储键：cybercorp.world.<worldId>.chats（世界隔离，R1） */
  _chatsKey() {
    return SAVE_KEY_PREFIX + (this.worldId || 'default') + CHAT_LOG.KEY_SUFFIX;
  }

  _loadChats() {
    try {
      const raw = localStorage.getItem(this._chatsKey());
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') this._agentChat = parsed;
      }
    } catch (e) { this._agentChat = {}; }
  }

  _saveChats() {
    try { localStorage.setItem(this._chatsKey(), JSON.stringify(this._agentChat)); } catch (e) {}
  }

  /** 记录一条对话（role: 'user' 玩家 / 'agent' 员工），带防抖与条数上限 */
  _recordChat(name, role, text) {
    if (!name || !text || this._disposed) return;
    // 独立防抖字段（仅限制 agent 高频刷屏），绝不与 _onPlayerChat 的回应冷却共用，
    // 否则玩家消息记录会把冷却刷成 now，导致后续 5 秒回应检查全部跳过（聊天无回应）
    if (role === 'agent' && this._chatDebounce[name] && (Date.now() - this._chatDebounce[name]) < 400) return;
    if (role === 'agent') this._chatDebounce[name] = Date.now();
    const list = this._agentChat[name] || (this._agentChat[name] = []);
    list.push({ role, text: String(text).slice(0, CHAT_LOG.MAX_TEXT), t: Date.now() });
    if (list.length > CHAT_LOG.MAX_PER_AGENT) list.splice(0, list.length - CHAT_LOG.MAX_PER_AGENT);
    this._saveChats();
  }

  // ==================== 靠近对话面板（每个员工独立记录，靠近即显示） ====================

  /** 检测玩家 10 米内最近员工，动态显示/切换/隐藏该员工的独立对话记录面板 */
  _updateNearbyChat() {
    if (this._disposed || this.state !== 'playing') { this._hideNearbyChat(); return; }
    const pos = this._playerPos();
    if (!pos) { this._hideNearbyChat(); return; }
    let nearest = null, best = Infinity;
    for (const a of this.actors) {
      if (!a || a.removed || !a.group) continue;
      const d = Math.hypot(a.group.position.x - pos.x, a.group.position.z - pos.z);
      if (d <= HEAR_RADIUS && d < best) { best = d; nearest = a; }
    }
    if (!nearest) { this._hideNearbyChat(); return; }
    if (this._nearAgentName !== nearest.name) {
      this._nearAgentName = nearest.name;
      this._renderNearbyChat(nearest);
    } else {
      this._refreshNearbyChat(nearest);
    }
  }

  _hideNearbyChat() {
    if (this._nearAgentName || this._chatPanelEl) {
      this._nearAgentName = null;
      if (this._chatPanelEl && this._chatPanelEl.parentNode) {
        this._chatPanelEl.parentNode.removeChild(this._chatPanelEl);
      }
      this._chatPanelEl = null;
    }
  }

  _renderNearbyChat(actor) {
    const posName = (actor.tag === 'employee' && actor.hiredPos) ? actor.hiredPos.name : '员工';
    if (!this._chatPanelEl) {
      this._chatPanelEl = document.createElement('div');
      this._chatPanelEl.className = 'cc-agent-chat';
      if (this._uiRoot) this._uiRoot.appendChild(this._chatPanelEl);
    }
    this._chatPanelEl.innerHTML = `
      <div class="cc-ac-head">💬 ${this.App.escapeHtml(actor.name)} · ${this.App.escapeHtml(posName)}
        <span class="cc-ac-sub">独立对话记录</span></div>
      <div class="cc-ac-body"></div>`;
    this._refreshNearbyChat(actor);
  }

  _refreshNearbyChat(actor) {
    if (!this._chatPanelEl) return;
    const body = this._chatPanelEl.querySelector('.cc-ac-body');
    if (!body) return;
    const list = this._agentChat[actor.name] || [];
    const recent = list.slice(-CHAT_LOG.PANEL_SHOW);
    body.innerHTML = recent.map(item => {
      const t = new Date(item.t).toTimeString().slice(0, 5);
      const cls = item.role === 'user' ? 'me' : 'them';
      return `<div class="cc-ac-row ${cls}"><span class="cc-ac-time">${t}</span><span class="cc-ac-txt">${this.App.escapeHtml(item.text)}</span></div>`;
    }).join('') || '<div class="cc-ac-empty">靠近说话即可开始对话…</div>';
    body.scrollTop = body.scrollHeight;
  }

  /**
   * 游戏永不结束：录取员工不代表结束。
   * 该方法是兜底安全网（正常路径走 _enterOperationMode），
   * 只提示运营状态并保存存档，绝不调用 onComplete 结束游戏。
   */
  async _finishGame() {
    if (this._disposed) return;
    const total = POSITIONS.length - 1;
    const filled = this.filledPositions.size;
    if (this.hiredList.length >= MAX_EMPLOYEES) {
      this._stopHiring();
    } else if (this.phase !== 'operating') {
      this._enterOperationMode();
    }
    const summary = this.hiredList.map(h => `${h.position.name}:${h.name}`).join('；') || '暂无员工';
    this._appendTranscript('system', `在职员工一览：${summary}`);
    this._saveGame();
    await this._sleep(3000);
  }

  // ==================== 公司评分系统 ====================

  /** 微笑曲线：单期裁员人数 → 组织活力分（分段线性插值） */
  _smileVitality(n) {
    const pts = BALANCE.SMILE;
    if (n <= pts[0][0]) return pts[0][1];
    if (n >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
    for (let i = 1; i < pts.length; i++) {
      if (n <= pts[i][0]) {
        const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
        const t = (n - x0) / (x1 - x0);
        return y0 + (y1 - y0) * t;
      }
    }
    return 0;
  }

  /** 僵化惩罚：连续不裁员期数 stale → 扣分（≥3 期起罚，每期 -2，封顶 -10） */
  _stalePenalty(stale) {
    return Math.min(BALANCE.STALE_CAP, Math.max(0, stale - (BALANCE.STALE_START - 1)) * BALANCE.STALE_STEP);
  }

  /** 总分 → 评级 */
  _gradeOf(score) {
    for (const [min, grade, title] of BALANCE.GRADE_LEVELS) {
      if (score >= min) return { grade, title };
    }
    return { grade: 'E', title: '濒临破产' };
  }

  /** 员工适配度 → 录用加分（≥80 +5 / 60-79 +2 / <60 -2） */
  _fitBonus(fitScore) {
    if (fitScore >= BALANCE.FIT_HIGH) return BALANCE.FIT_BONUS_HIGH;
    if (fitScore >= 60) return BALANCE.FIT_BONUS_NORMAL;
    return BALANCE.FIT_PENALTY_LOW;
  }

  /**
   * 四维评分重算（纯计算，不改动原始状态）：
   * - 组织活力 = 微笑曲线(当期裁员数) - 僵化惩罚(连续不裁员期数)
   * - 人才质量 = min(30, 员工平均适配度×0.25 + 适配加成累计)
   * - 运营稳定 = 岗位覆盖率×10 + 满编率×10
   * - 公司声誉 = 事件修正后的声誉值（由 _settlePeriod 维护）
   */
  _recalcScore() {
    const s = this.stats;
    const hired = this.hiredList || [];

    // 组织活力：微笑曲线 - 僵化惩罚
    const v0 = this._smileVitality(s.layoffCount || 0);
    const vStale = this._stalePenalty(s.stalePeriods || 0);
    s.vitality = Math.max(0, Math.round(v0 - vStale));

    // 人才质量：平均适配度 + 高适配加成
    const fits = hired.map(h => h.fitScore || 0);
    const avg = fits.length ? fits.reduce((a, b) => a + b, 0) / fits.length : 0;
    let bonus = 0;
    for (const h of hired) bonus += this._fitBonus(h.fitScore || 0);
    s.quality = Math.max(0, Math.min(BALANCE.QUALITY_FULL, Math.round(avg * BALANCE.QUALITY_AVG_WEIGHT + bonus)));

    // 运营稳定：岗位覆盖率（基础 7.5 分）+ 满编率（成长 7.5 分），满分 15
    const totalPos = POSITIONS.length - 1;              // 除 HR 外 6 个岗位
    const filled = Math.min(totalPos, this.filledPositions.size);
    const coverage = totalPos ? filled / totalPos : 0;
    const fullRate = Math.min(1, hired.length / MAX_EMPLOYEES);
    s.stability = Math.max(0, Math.min(BALANCE.STABILITY_FULL, Math.round(coverage * 7.5 + fullRate * 7.5)));

    // 公司产出：员工 RL 自主完成任务的产出累积（0-5）
    s.production = Math.max(0, Math.min(RL_CFG.PRODUCTION_CAP, s.production || 0));

    // 总分与评级（五维：活力40 + 质量30 + 稳定15 + 声誉10 + 产出5 = 100）
    s.reputation = Math.max(0, Math.min(BALANCE.REPUTATION_FULL, s.reputation || 0));
    s.score = Math.max(0, Math.min(100, Math.round(s.vitality + s.quality + s.stability + s.reputation + s.production)));
    const g = this._gradeOf(s.score);
    s.grade = g.grade;
    this.score = s.score;
    return s;
  }

  /**
   * 每期结算（一期 = 一轮面试结束）：
   * - 按当期裁员数修正声誉（健康汰换 +1 / 裁2人 -1 / 裁≥3人 -4 / 连续不裁员 -1）
   * - 更新僵化累积（裁员则清零，否则 +1；仅运营期累计，招聘期不算「长期不裁员」）
   * - 清零当期裁员计数，重算五维总分
   * 返回结算摘要 { vitality, quality, stability, reputation, score, grade, events }
   */
  _settlePeriod() {
    const s = this.stats;
    const events = [];
    const n = s.layoffCount || 0;
    const operating = this.phase === 'operating';

    // 声誉修正：裁员事件任何时期都生效；健康加成仅运营期（招聘期裁 0 人无加成）
    if (n >= 3) {
      s.reputation = Math.max(0, s.reputation + BALANCE.REP_CRASH);
      events.push({ text: '「血汗工厂」舆论发酵', delta: BALANCE.REP_CRASH, dir: 'down' });
    } else if (n === 2) {
      s.reputation = Math.max(0, s.reputation + BALANCE.REP_WARN);
      events.push({ text: '裁员引发媒体关注', delta: BALANCE.REP_WARN, dir: 'down' });
    } else if (n === 1) {
      if (operating) {
        s.reputation = Math.min(BALANCE.REPUTATION_FULL, s.reputation + BALANCE.REP_GOOD);
        events.push({ text: '健康汰换，外界好评', delta: BALANCE.REP_GOOD, dir: 'up' });
      }
    } else if (operating && s.stalePeriods >= BALANCE.STALE_START - 1) {
      s.reputation = Math.max(0, s.reputation + BALANCE.REP_STALE);
      events.push({ text: '外界认为组织僵化，声誉受损', delta: BALANCE.REP_STALE, dir: 'down' });
    }

    // 僵化累积（仅运营期；招聘期公司在成长，不算「长期不裁员」）
    if (operating) {
      if (n > 0) {
        s.stalePeriods = 0;
        events.push({ text: `组织焕新 · 本期裁 ${n} 人`, delta: 0, dir: 'up' });
      } else {
        s.stalePeriods++;
      }
    }

    s.layoffCount = 0;
    this._recalcScore();
    // 快照：用结算前裁员数重算活力（反映本期真实决策），其余维度保持结算后状态
    const snapshot = { ...s, layoffCount: n };
    snapshot.vitality = Math.max(0, Math.round(this._smileVitality(n) - this._stalePenalty(s.stalePeriods)));
    this._lastSettleEvents = events;
    this._lastSettle = snapshot;
    return snapshot;
  }

  /** 裁员单个员工：移出场景、更新岗位与评分，返回是否成功 */
  _layoffEmployee(actor) {
    if (!actor || this._disposed) return false;
    const idx = this.hiredList.findIndex(h => h.actor === actor);
    if (idx < 0) return false;
    const h = this.hiredList[idx];
    const pos = h.position;

    // 本期裁员计数 + 僵化清零
    this.stats.layoffCount = (this.stats.layoffCount || 0) + 1;
    this.stats.stalePeriods = 0;

    // 从在职列表移除
    this.hiredList.splice(idx, 1);
    // 岗位标记：若该岗位不再有员工则恢复空缺（允许重新招聘）
    if (!this.hiredList.some(e => e.position.key === pos.key)) {
      this.filledPositions.delete(pos.key);
    }
    // RL 体系清理：移除该员工的独立策略 / 精力 / 任务绑定
    this._cleanupAgentRL(h.name);

    // 角色离场动画 + 移除
    const ws = LAYOUT.workstations.find(w => w.posKey === pos.key);
    actor.setEmotion('sad');
    actor.faceTarget = { x: LAYOUT.entrance.x, z: LAYOUT.entrance.z };
    if (actor.group) actor.group.visible = false;       // 立即隐藏，避免离场动画阻塞 UI
    this._safeDisposeActor(actor);

    // 更新岗位牌（恢复岗位名）
    if (ws && ws.label) {
      this._setLabelText(ws.label, `${pos.icon} ${pos.name}`, pos.color);
    }
    this._updatePositionsUI();

    // 裁员后恢复招聘（满员停招时）
    this._stoppedHiring = false;

    this._appendTranscript('system', `💼 ${h.name}（${pos.name}）被裁离岗。`);

    // 若主循环已停（满员停招后），且仍有空位 → 重新启动招聘循环
    if (this.hiredList.length < MAX_EMPLOYEES && !this._gameLoopRunning) {
      this._skipWelcome = true;
      this._runGame().catch(err => console.error('[赛博公司] 招聘循环异常:', err));
    }
    return true;
  }

  /** 安全移除角色（不阻塞调用方） */
  _safeDisposeActor(actor) {
    try { actor.dispose(); } catch (e) {}
    const i = this.actors.indexOf(actor);
    if (i >= 0) this.actors.splice(i, 1);
  }

  // ==================== 移动工具 ====================

  _moveActor(actor, points, onArrive) {
    return new Promise(resolve => {
      if (!actor || actor.removed) { resolve(); return; }
      // 移动代际：每次新移动都会作废旧移动；旧移动的回调/超时让位，不干扰新路径
      actor._moveGen = (actor._moveGen || 0) + 1;
      const gen = actor._moveGen;
      actor.moveAlong(points, () => {
        if (actor._moveGen !== gen) { resolve(); return; }
        this._safeResolve(resolve);
      });
      // 兜底超时（若移动卡住）
      const t = setTimeout(() => {
        if (actor._moveGen !== gen) { resolve(); return; }
        if (actor.waypoints) actor.waypoints = null;
        actor.isMoving = false;
        this._safeResolve(resolve);
      }, 15000);
      this._timers.push(t);
    });
  }

  _safeResolve(fn) {
    // 无条件 resolve：_disposed 时不吞掉结算 —— 否则 await _moveActor 永久挂起，
    // 面试主循环卡死在行走位/入职/离场流程（退出后旧实例虽被丢弃，但挂起点
    // 会跳过后续存档与状态收尾）。disposed 由 await 之后的调用方检查。
    fn();
  }

  // ==================== UI ====================

  _injectStyles() {
    if (document.getElementById('cyber-corp-style')) return;
    const style = document.createElement('style');
    style.id = 'cyber-corp-style';
    style.textContent = `
      .cyber-hr-select { position: fixed; inset: 0; z-index: 9999; display: flex; align-items: center; justify-content: center;
        background: radial-gradient(ellipse at center, rgba(10,12,30,.92) 0%, rgba(4,6,16,.97) 70%);
        backdrop-filter: blur(6px); font-family: "Microsoft YaHei", sans-serif; }
      .cyber-hr-select::before { content: ''; position: absolute; inset: 0;
        background: repeating-linear-gradient(0deg, rgba(0,229,255,.03) 0 2px, transparent 2px 4px); pointer-events: none; }
      .cyber-hr-panel { width: min(880px, 92vw); max-height: 84vh; overflow: auto; padding: 30px 34px;
        background: rgba(8,10,24,.96); border: 1px solid rgba(0,229,255,.4); border-radius: 14px;
        box-shadow: 0 0 40px rgba(0,229,255,.18), inset 0 0 60px rgba(0,229,255,.04); position: relative; }
      .cyber-hr-logo { text-align: center; font-size: 15px; letter-spacing: 4px; color: #ff2bd6; text-shadow: 0 0 12px #ff2bd6; margin-bottom: 10px; }
      .cyber-hr-title { text-align: center; color: #00e5ff; font-size: 26px; margin: 6px 0 10px; text-shadow: 0 0 16px #00e5ff; }
      .cyber-hr-sub { text-align: center; color: #9aa7cc; font-size: 13px; margin: 0 0 20px; line-height: 1.7; }
      .cyber-hr-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
      .cyber-hr-card { background: rgba(14,18,36,.9); border: 1px solid rgba(0,229,255,.25); border-radius: 10px;
        padding: 14px 10px; text-align: center; cursor: pointer; transition: all .18s; position: relative; overflow: hidden; }
      .cyber-hr-card:hover { border-color: #00e5ff; box-shadow: 0 0 22px rgba(0,229,255,.35); transform: translateY(-3px); }
      .cyber-hr-card:hover::after { content: '任命'; position: absolute; right: 8px; top: 6px; color: #00ffa3; font-size: 11px; letter-spacing: 2px; }
      .cyber-hr-ava { width: 56px; height: 56px; margin: 0 auto 8px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 24px; color: #00e5ff; background: rgba(0,229,255,.1); border: 1px solid rgba(0,229,255,.4);
        box-shadow: 0 0 16px rgba(0,229,255,.25); }
      .cyber-hr-name { color: #eef4ff; font-size: 15px; font-weight: 700; }
      .cyber-hr-role { color: #7f8db3; font-size: 12px; margin-top: 2px; }
      .cyber-hr-model { color: #4f5d85; font-size: 11px; margin-top: 4px; }
      .cyber-hr-empty { color: #7f8db3; text-align: center; padding: 30px; grid-column: 1 / -1; }
      .cyber-hr-random { display: block; margin: 18px auto 0; padding: 10px 30px; cursor: pointer;
        background: rgba(255,43,214,.12); border: 1px solid rgba(255,43,214,.5); color: #ff8ef0; border-radius: 8px;
        font-size: 14px; letter-spacing: 2px; transition: all .18s; }
      .cyber-hr-random:hover { background: rgba(255,43,214,.25); box-shadow: 0 0 18px rgba(255,43,214,.4); }
      .cyber-hr-modes { display: flex; gap: 10px; justify-content: center; margin: 0 0 18px; }
      .cyber-hr-mode { flex: 1; max-width: 300px; cursor: pointer; padding: 10px 12px; border-radius: 10px;
        background: rgba(14,18,36,.9); border: 1px solid rgba(0,229,255,.25); color: #9aa7cc;
        font-size: 13px; letter-spacing: 1px; line-height: 1.5; transition: all .18s;
        font-family: "Microsoft YaHei", sans-serif; }
      .cyber-hr-mode span { font-size: 11px; color: #5f6d95; display: block; margin-top: 3px; }
      .cyber-hr-mode:hover { border-color: #00e5ff; color: #eef4ff; }
      .cyber-hr-mode.active { border-color: #ffd166; color: #ffd166; background: rgba(255,209,102,.1);
        box-shadow: 0 0 18px rgba(255,209,102,.25); }
      .cyber-hr-mode.active span { color: #c9a24e; }

      #cyber-corp-ui { position: fixed; inset: 0; pointer-events: none; z-index: 9000; font-family: "Microsoft YaHei", sans-serif; }
      #cyber-corp-ui .cc-top { position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
        display: flex; gap: 6px; align-items: center; background: rgba(6,8,20,.55); border: 1px solid rgba(0,229,255,.25);
        border-radius: 8px; padding: 4px 12px; box-shadow: 0 0 14px rgba(0,229,255,.1); }
      #cyber-corp-ui .cc-logo { color: #ff2bd6; font-weight: 800; letter-spacing: 2px; text-shadow: 0 0 8px rgba(255,43,214,.6); font-size: 12px; }
      #cyber-corp-ui .cc-hr { color: #00e5ff; font-size: 11px; border-left: 1px solid rgba(255,255,255,.12); padding-left: 8px; }
      #cyber-corp-ui .cc-hr b { color: #eef4ff; }
      #cyber-corp-ui .cc-pos { display: flex; gap: 4px; flex-wrap: wrap; border-left: 1px solid rgba(255,255,255,.12); padding-left: 8px; }
      #cyber-corp-ui .cc-pos-chip { font-size: 11px; padding: 2px 7px; border-radius: 8px; border: 1px solid; background: rgba(10,14,28,.6); white-space: nowrap; }
      #cyber-corp-ui .cc-pos-chip.hired { border-color: rgba(0,255,163,.5); color: #7dffc8; }
      #cyber-corp-ui .cc-pos-chip.open { border-color: rgba(255,209,102,.5); color: #ffd166; }
      #cyber-corp-ui .cc-status { position: absolute; top: 46px; left: 50%; transform: translateX(-50%);
        max-width: 72vw; background: rgba(6,8,20,.82); border: 1px solid rgba(0,229,255,.35); color: #6ee7ff;
        border-radius: 8px; padding: 4px 14px; font-size: 12px; line-height: 1.5; text-align: center;
        box-shadow: 0 0 14px rgba(0,229,255,.12); opacity: 0; transition: opacity .25s ease; pointer-events: none; }
      #cyber-corp-ui .cc-status.cc-show { opacity: 1; }
      #cyber-corp-ui .cc-swarm { position: absolute; left: 10px; bottom: 12px; width: 230px;
        background: rgba(6,8,20,.72); border: 1px solid rgba(255,45,214,.28); border-radius: 8px;
        padding: 7px 10px; box-shadow: 0 0 14px rgba(255,45,214,.10); pointer-events: none; }
      #cyber-corp-ui .cc-swarm .cc-swarm-title { font-size: 10px; color: #ff8ad2; letter-spacing: 2px; margin-bottom: 4px; }
      #cyber-corp-ui .cc-swarm .cc-swarm-item { font-size: 11px; color: #c7d3ee; line-height: 1.55;
        border-top: 1px solid rgba(255,45,214,.10); padding-top: 2px; margin-top: 2px;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      #cyber-corp-ui .cc-swarm .cc-swarm-item:first-child { border-top: none; padding-top: 0; margin-top: 0; color: #eef4ff; }
      #cyber-corp-ui .cc-cand { position: absolute; left: 10px; top: 62px; width: 168px; background: rgba(6,8,20,.68);
        border: 1px solid rgba(0,229,255,.28); border-radius: 8px; padding: 8px 10px; box-shadow: 0 0 12px rgba(0,229,255,.08); }
      #cyber-corp-ui .cc-cand .cc-cand-label { font-size: 10px; color: #5f6d95; letter-spacing: 2px; margin-bottom: 3px; }
      #cyber-corp-ui .cc-cand .cc-cand-name { font-size: 13px; font-weight: 700; color: #eef4ff; }
      #cyber-corp-ui .cc-cand .cc-cand-pos { font-size: 11px; color: #9aa7cc; margin-top: 2px; }
      #cyber-corp-ui .cc-result { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -58%);
        width: 300px; background: rgba(6,8,20,.92); border-radius: 12px; padding: 14px 18px; text-align: center;
        border: 1px solid; box-shadow: 0 0 30px rgba(0,0,0,.55); animation: cc-pop .4s ease; }
      #cyber-corp-ui .cc-result.pass { border-color: rgba(0,255,163,.55); }
      #cyber-corp-ui .cc-result.fail { border-color: rgba(255,77,109,.55); }
      @keyframes cc-pop { from { transform: translate(-50%, -58%) scale(.8); opacity: 0; } to { transform: translate(-50%, -58%) scale(1); opacity: 1; } }
      #cyber-corp-ui .cc-result .cc-r-title { font-size: 17px; font-weight: 900; margin-bottom: 6px; }
      #cyber-corp-ui .cc-result .cc-r-title.pass { color: #00ffa3; text-shadow: 0 0 12px rgba(0,255,163,.6); }
      #cyber-corp-ui .cc-result .cc-r-title.fail { color: #ff4d6d; text-shadow: 0 0 12px rgba(255,77,109,.6); }
      #cyber-corp-ui .cc-result .cc-r-sub { color: #9aa7cc; font-size: 12px; margin-bottom: 10px; }
      #cyber-corp-ui .cc-bar { height: 10px; border-radius: 6px; background: rgba(255,255,255,.08); overflow: hidden; position: relative; }
      #cyber-corp-ui .cc-bar-in { height: 100%; width: 0; border-radius: 6px; background: linear-gradient(90deg,#00e5ff,#00ffa3); transition: width 1.2s ease; }
      #cyber-corp-ui .cc-bar-in.fail { background: linear-gradient(90deg,#ff9d6e,#ff4d6d); }
      #cyber-corp-ui .cc-r-score { font-size: 24px; font-weight: 900; margin: 8px 0 2px; color: #eef4ff; }
      #cyber-corp-ui .cc-r-comment { margin-top: 8px; font-size: 11px; line-height: 1.6; color: #cfe0ff;
        background: rgba(0,229,255,.06); border-left: 2px solid #00e5ff; padding: 6px 8px;
        border-radius: 6px; text-align: left; min-height: 16px; }

      /* ---- 运营面板（评分 + 裁员） ---- */
      #cyber-corp-ui .cc-op-btn { position: absolute; right: 10px; top: 62px; pointer-events: auto; cursor: pointer;
        background: rgba(0,229,255,.1); border: 1px solid rgba(0,229,255,.45); color: #00e5ff;
        border-radius: 8px; padding: 5px 12px; font-size: 12px; letter-spacing: 1px; transition: all .18s; }
      #cyber-corp-ui .cc-op-btn:hover { background: rgba(0,229,255,.22); box-shadow: 0 0 14px rgba(0,229,255,.4); }
      #cyber-corp-ui .cc-dir-btn { right: 118px; background: rgba(255,209,102,.1);
        border-color: rgba(255,209,102,.5); color: #ffd166; }
      #cyber-corp-ui .cc-dir-btn:hover { background: rgba(255,209,102,.22); box-shadow: 0 0 14px rgba(255,209,102,.4); }
      #cyber-corp-ui .cc-vr-btn { right: 226px; background: rgba(0,255,163,.1);
        border-color: rgba(0,255,163,.5); color: #7dffc8; }
      #cyber-corp-ui .cc-vr-btn:hover { background: rgba(0,255,163,.22); box-shadow: 0 0 14px rgba(0,255,163,.4); }
      #cyber-corp-ui .cc-vr-btn.active { background: rgba(255,77,109,.16);
        border-color: rgba(255,77,109,.55); color: #ff6d85; }

      /* ---- CEO 直聘面板（跳过面试，直接选人） ---- */
      #cyber-corp-ui .cc-direct-mask { position: fixed; inset: 0; z-index: 9150; display: flex; align-items: center; justify-content: center;
        background: rgba(4,6,16,.62); backdrop-filter: blur(3px); font-family: "Microsoft YaHei", sans-serif; pointer-events: auto; }
      #cyber-corp-ui .cc-direct-panel { width: min(760px, 94vw); max-height: 86vh; overflow: auto; padding: 22px 26px;
        background: rgba(8,10,24,.97); border: 1px solid rgba(255,209,102,.5); border-radius: 14px;
        box-shadow: 0 0 44px rgba(255,209,102,.22); position: relative; }
      #cyber-corp-ui .cc-direct-close { position: absolute; right: 14px; top: 12px; cursor: pointer; color: #5f6d95;
        font-size: 18px; line-height: 1; padding: 4px 8px; border-radius: 6px; }
      #cyber-corp-ui .cc-direct-close:hover { color: #ff4d6d; background: rgba(255,77,109,.12); }
      #cyber-corp-ui .cc-direct-head { font-size: 20px; font-weight: 900; color: #ffd166; letter-spacing: 2px;
        text-shadow: 0 0 16px rgba(255,209,102,.5); }
      #cyber-corp-ui .cc-direct-sub { font-size: 12px; color: #9aa7cc; margin: 6px 0 14px; }
      #cyber-corp-ui .cc-direct-pos { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
      #cyber-corp-ui .cc-dp-chip { cursor: pointer; font-size: 12px; padding: 6px 12px; border-radius: 8px;
        border: 1px solid; background: rgba(14,18,36,.9); transition: all .15s; }
      #cyber-corp-ui .cc-dp-chip:hover { box-shadow: 0 0 12px rgba(0,229,255,.25); transform: translateY(-1px); }
      #cyber-corp-ui .cc-direct-cands { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
      #cyber-corp-ui .cc-dc-card { display: flex; align-items: center; gap: 10px; padding: 10px 12px; cursor: pointer;
        background: rgba(14,18,36,.85); border: 1px solid rgba(0,229,255,.18); border-radius: 10px; transition: all .15s; }
      #cyber-corp-ui .cc-dc-card:hover { border-color: #00e5ff; box-shadow: 0 0 14px rgba(0,229,255,.25); transform: translateY(-2px); }
      #cyber-corp-ui .cc-dc-ava { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 15px; color: #00e5ff; background: rgba(0,229,255,.1); border: 1px solid rgba(0,229,255,.35); flex-shrink: 0; }
      #cyber-corp-ui .cc-dc-info { flex: 1; min-width: 0; }
      #cyber-corp-ui .cc-dc-name { font-size: 13px; font-weight: 700; color: #eef4ff; }
      #cyber-corp-ui .cc-dc-role { font-size: 11px; color: #7f8db3; margin-top: 1px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      #cyber-corp-ui .cc-dc-fit { font-size: 11px; color: #6ee7ff; border: 1px solid rgba(0,229,255,.3); border-radius: 999px; padding: 1px 8px; white-space: nowrap; }
      #cyber-corp-ui .cc-dc-hire { cursor: pointer; font-size: 11px; padding: 4px 10px; border-radius: 6px; flex-shrink: 0;
        background: rgba(0,255,163,.1); border: 1px solid rgba(0,255,163,.45); color: #7dffc8; transition: all .15s;
        font-family: "Microsoft YaHei", sans-serif; }
      #cyber-corp-ui .cc-dc-hire:hover { background: rgba(0,255,163,.25); box-shadow: 0 0 10px rgba(0,255,163,.45); }
      #cyber-corp-ui .cc-direct-empty { color: #5f6d95; font-size: 12px; text-align: center; padding: 20px; grid-column: 1 / -1; }
      #cyber-corp-ui .cc-direct-foot { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 16px; }
      #cyber-corp-ui .cc-direct-count { font-size: 12px; color: #9aa7cc; }
      #cyber-corp-ui .cc-direct-done { cursor: pointer; padding: 9px 18px; border-radius: 8px; font-size: 13px; letter-spacing: 2px;
        background: rgba(255,209,102,.12); border: 1px solid rgba(255,209,102,.5); color: #ffd166; transition: all .18s;
        font-family: "Microsoft YaHei", sans-serif; }
      #cyber-corp-ui .cc-direct-done:hover { background: rgba(255,209,102,.25); box-shadow: 0 0 14px rgba(255,209,102,.35); }
      #cyber-corp-ui .cc-op-mask { position: fixed; inset: 0; z-index: 9100; display: flex; align-items: center; justify-content: center;
        background: rgba(4,6,16,.6); backdrop-filter: blur(3px); font-family: "Microsoft YaHei", sans-serif; }
      #cyber-corp-ui .cc-op-panel { width: min(620px, 94vw); max-height: 86vh; overflow: auto; padding: 22px 26px;
        background: rgba(8,10,24,.97); border: 1px solid rgba(0,229,255,.4); border-radius: 14px;
        box-shadow: 0 0 44px rgba(0,229,255,.2); position: relative; }
      #cyber-corp-ui .cc-op-head { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
      #cyber-corp-ui .cc-op-score { font-size: 44px; font-weight: 900; color: #eef4ff; text-shadow: 0 0 18px rgba(0,229,255,.5); }
      #cyber-corp-ui .cc-op-score small { font-size: 16px; color: #5f6d95; font-weight: 400; }
      #cyber-corp-ui .cc-op-badge { font-size: 14px; padding: 3px 14px; border-radius: 999px; border: 1px solid; letter-spacing: 1px; }
      #cyber-corp-ui .cc-op-badge.S { color: #00ffa3; border-color: rgba(0,255,163,.55); background: rgba(0,255,163,.1); }
      #cyber-corp-ui .cc-op-badge.A { color: #7dffc8; border-color: rgba(0,255,163,.4); background: rgba(0,255,163,.07); }
      #cyber-corp-ui .cc-op-badge.B { color: #6ee7ff; border-color: rgba(0,229,255,.45); background: rgba(0,229,255,.08); }
      #cyber-corp-ui .cc-op-badge.C { color: #ffd166; border-color: rgba(255,209,102,.45); background: rgba(255,209,102,.08); }
      #cyber-corp-ui .cc-op-badge.D, #cyber-corp-ui .cc-op-badge.E { color: #ff4d6d; border-color: rgba(255,77,109,.5); background: rgba(255,77,109,.1); }
      #cyber-corp-ui .cc-op-close { position: absolute; right: 14px; top: 14px; cursor: pointer; color: #5f6d95;
        font-size: 18px; line-height: 1; padding: 4px 8px; border-radius: 6px; }
      #cyber-corp-ui .cc-op-close:hover { color: #ff4d6d; background: rgba(255,77,109,.12); }
      #cyber-corp-ui .cc-op-sec { font-size: 11px; color: #5f6d95; letter-spacing: 3px; margin: 14px 0 8px;
        border-bottom: 1px solid rgba(0,229,255,.15); padding-bottom: 4px; }
      #cyber-corp-ui .cc-op-dim { display: grid; grid-template-columns: 74px 1fr 56px; align-items: center; gap: 10px; margin: 7px 0; }
      #cyber-corp-ui .cc-op-dim .cc-dl { font-size: 12px; color: #9aa7cc; }
      #cyber-corp-ui .cc-op-dim .cc-dt { height: 9px; border-radius: 6px; background: rgba(255,255,255,.08); overflow: hidden; }
      #cyber-corp-ui .cc-op-dim .cc-dt i { display: block; height: 100%; border-radius: 6px; transition: width .4s ease; }
      #cyber-corp-ui .cc-op-dim .cc-dn { font-size: 12px; color: #eef4ff; text-align: right; font-weight: 700; }
      #cyber-corp-ui .cc-op-smile { font-size: 12px; color: #9aa7cc; background: rgba(0,229,255,.06);
        border-left: 2px solid #00e5ff; padding: 8px 10px; border-radius: 6px; margin: 8px 0 4px; line-height: 1.7; }
      #cyber-corp-ui .cc-op-smile b { color: #00e5ff; }
      #cyber-corp-ui .cc-op-smile .stale { color: #ffd166; }
      #cyber-corp-ui .cc-op-task { display: flex; align-items: center; gap: 10px; padding: 6px 10px; margin: 4px 0;
        background: rgba(14,18,36,.8); border: 1px solid rgba(0,229,255,.16); border-radius: 8px; }
      #cyber-corp-ui .cc-op-task .cc-ot-icon { width: 26px; height: 26px; border-radius: 6px; display: flex; align-items: center;
        justify-content: center; font-size: 13px; background: rgba(0,229,255,.08); border: 1px solid rgba(0,229,255,.25); flex-shrink: 0; }
      #cyber-corp-ui .cc-op-task .cc-ot-info { flex: 1; min-width: 0; }
      #cyber-corp-ui .cc-op-task .cc-ot-title { font-size: 12px; color: #cfe0ff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      #cyber-corp-ui .cc-op-task .cc-ot-bar { height: 6px; margin-top: 4px; border-radius: 4px; background: rgba(255,255,255,.08); overflow: hidden; }
      #cyber-corp-ui .cc-op-task .cc-ot-bar i { display: block; height: 100%; border-radius: 4px; background: #00e5ff; transition: width .4s ease; }
      #cyber-corp-ui .cc-op-task.done .cc-ot-bar i { background: #00ffa3; }
      #cyber-corp-ui .cc-op-task.done .cc-ot-pct { color: #00ffa3; }
      #cyber-corp-ui .cc-op-task .cc-ot-pct { font-size: 11px; color: #6ee7ff; white-space: nowrap; flex-shrink: 0; }
      #cyber-corp-ui .cc-op-emp { display: flex; align-items: center; gap: 10px; padding: 7px 10px; margin: 5px 0;
        background: rgba(14,18,36,.8); border: 1px solid rgba(0,229,255,.18); border-radius: 8px; }
      #cyber-corp-ui .cc-op-emp .cc-e-ava { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center;
        justify-content: center; font-size: 14px; background: rgba(0,229,255,.1); border: 1px solid rgba(0,229,255,.35); flex-shrink: 0; }
      #cyber-corp-ui .cc-op-emp .cc-e-info { flex: 1; min-width: 0; }
      #cyber-corp-ui .cc-op-emp .cc-e-name { font-size: 13px; font-weight: 700; color: #eef4ff; }
      #cyber-corp-ui .cc-op-emp .cc-e-pos { font-size: 11px; color: #7f8db3; }
      #cyber-corp-ui .cc-op-emp .cc-e-rl { font-size: 10px; color: #6ee7ff; margin-top: 2px; opacity: .85; }
      #cyber-corp-ui .cc-op-emp .cc-e-fit { font-size: 11px; padding: 1px 8px; border-radius: 999px; border: 1px solid; white-space: nowrap; }
      #cyber-corp-ui .cc-op-emp .cc-e-fit.high { color: #00ffa3; border-color: rgba(0,255,163,.4); }
      #cyber-corp-ui .cc-op-emp .cc-e-fit.mid { color: #9aa7cc; border-color: rgba(255,255,255,.2); }
      #cyber-corp-ui .cc-op-emp .cc-e-fit.low { color: #ffd166; border-color: rgba(255,209,102,.4); }
      #cyber-corp-ui .cc-op-emp .cc-e-fire { cursor: pointer; font-size: 11px; padding: 3px 12px; border-radius: 6px; flex-shrink: 0;
        background: rgba(255,77,109,.1); border: 1px solid rgba(255,77,109,.45); color: #ff6d85; transition: all .15s; }
      #cyber-corp-ui .cc-op-emp .cc-e-fire:hover { background: rgba(255,77,109,.3); box-shadow: 0 0 10px rgba(255,77,109,.5); }
      #cyber-corp-ui .cc-op-empty { color: #5f6d95; font-size: 12px; text-align: center; padding: 14px; }
      #cyber-corp-ui .cc-op-foot { display: flex; gap: 10px; margin-top: 16px; }
      #cyber-corp-ui .cc-op-foot button { flex: 1; cursor: pointer; padding: 9px; border-radius: 8px; font-size: 13px; letter-spacing: 2px; transition: all .18s; }
      #cyber-corp-ui .cc-op-foot .cc-of-ok { background: rgba(0,229,255,.12); border: 1px solid rgba(0,229,255,.5); color: #00e5ff; }
      #cyber-corp-ui .cc-op-foot .cc-of-ok:hover { background: rgba(0,229,255,.25); }
      #cyber-corp-ui .cc-op-foot .cc-of-warn { background: rgba(255,209,102,.1); border: 1px solid rgba(255,209,102,.45); color: #ffd166; }
      #cyber-corp-ui .cc-op-foot .cc-of-warn:hover { background: rgba(255,209,102,.22); }
      #cyber-corp-ui .cc-op-events { display: flex; flex-direction: column; gap: 4px; margin-top: 6px; }
      #cyber-corp-ui .cc-op-events .cc-ev { font-size: 12px; padding: 4px 10px; border-radius: 6px; }
      #cyber-corp-ui .cc-op-events .cc-ev.up { color: #00ffa3; background: rgba(0,255,163,.07); }
      #cyber-corp-ui .cc-op-events .cc-ev.down { color: #ff6d85; background: rgba(255,77,109,.08); }

      /* ---- 员工独立对话记录面板（靠近即显示） ---- */
      #cyber-corp-ui .cc-agent-chat { position: absolute; left: 10px; bottom: 16px; width: 300px;
        max-height: 46vh; display: flex; flex-direction: column; background: rgba(6,8,20,.82);
        border: 1px solid rgba(0,229,255,.35); border-radius: 10px; overflow: hidden;
        box-shadow: 0 0 22px rgba(0,229,255,.15); backdrop-filter: blur(4px); }
      #cyber-corp-ui .cc-agent-chat .cc-ac-head { display: flex; align-items: center; gap: 6px;
        padding: 7px 12px; font-size: 12px; font-weight: 700; color: #eef4ff;
        background: linear-gradient(90deg, rgba(0,229,255,.14), rgba(255,43,214,.08));
        border-bottom: 1px solid rgba(0,229,255,.25); }
      #cyber-corp-ui .cc-agent-chat .cc-ac-head .cc-ac-sub { font-size: 10px; color: #7dffc8; font-weight: 400;
        letter-spacing: 1px; background: rgba(0,255,163,.1); border: 1px solid rgba(0,255,163,.35);
        border-radius: 999px; padding: 0 7px; margin-left: auto; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-body { padding: 8px 10px; overflow-y: auto; display: flex;
        flex-direction: column; gap: 6px; scrollbar-width: thin; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-body::-webkit-scrollbar { width: 4px; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-body::-webkit-scrollbar-thumb { background: rgba(0,229,255,.3); border-radius: 2px; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-row { display: flex; align-items: baseline; gap: 6px;
        font-size: 11px; line-height: 1.5; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-row .cc-ac-time { font-size: 9px; color: #5f6d95; flex-shrink: 0;
        font-family: "Consolas", monospace; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-row .cc-ac-txt { word-break: break-all; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-row.them .cc-ac-txt { color: #cfe0ff; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-row.me { flex-direction: row-reverse; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-row.me .cc-ac-txt { color: #7dffc8; text-align: right; }
      #cyber-corp-ui .cc-agent-chat .cc-ac-empty { color: #5f6d95; font-size: 11px; text-align: center; padding: 10px; }

      /* ---- CEO 就任表单 ---- */
      .cyber-ceo-form { display: flex; flex-direction: column; gap: 5px; }
      .cyber-ceo-form label { color: #5f6d95; font-size: 12px; letter-spacing: 2px; margin-top: 7px; }
      .cyber-ceo-form input { background: rgba(14,18,36,.9); border: 1px solid rgba(0,229,255,.3); border-radius: 8px;
        color: #eef4ff; padding: 9px 12px; font-size: 14px; outline: none; transition: all .15s; font-family: "Microsoft YaHei", sans-serif; }
      .cyber-ceo-form input:focus { border-color: #00e5ff; box-shadow: 0 0 12px rgba(0,229,255,.25); }
    `;
    document.head.appendChild(style);
    this._styleEl = style;
  }

  _buildMainUI() {
    if (this._uiRoot) return;
    const root = document.createElement('div');
    root.id = 'cyber-corp-ui';
    root.innerHTML = `
      <div class="cc-top">
        <div class="cc-logo">🏢 赛博公司</div>
        <div class="cc-hr">HR：<b id="cc-hr-name">${this.App.escapeHtml(this.hrActor?.name || '')}</b></div>
        <div class="cc-pos" id="cc-pos-chips"></div>
      </div>
      <div class="cc-status" id="cc-status"></div>
      <div class="cc-cand" id="cc-cand-wrap" style="display:none;">
        <div class="cc-cand-label">当前面试</div>
        <div class="cc-cand-name" id="cc-cand-name">—</div>
        <div class="cc-cand-pos" id="cc-cand-pos">—</div>
      </div>
      <div class="cc-swarm" id="cc-swarm">
        <div class="cc-swarm-title">🐝 蜂群动态</div>
        <div class="cc-swarm-list" id="cc-swarm-list"></div>
      </div>
      <button class="cc-op-btn" id="cc-op-btn" type="button">📊 运营面板</button>
      <button class="cc-op-btn cc-dir-btn" id="cc-dir-btn" type="button">⚡ 直聘</button>
    `;
    document.body.appendChild(root);
    this._uiRoot = root;
    this._uiCandWrap = root.querySelector('#cc-cand-wrap');
    this._uiCandName = root.querySelector('#cc-cand-name');
    this._uiCandPos = root.querySelector('#cc-cand-pos');
    this._uiStatus = root.querySelector('#cc-status');
    this._uiPosChips = root.querySelector('#cc-pos-chips');
    this._uiSwarmList = root.querySelector('#cc-swarm-list');
    const opBtn = root.querySelector('#cc-op-btn');
    if (opBtn) opBtn.addEventListener('click', () => this._toggleOperationPanel());
    const dirBtn = root.querySelector('#cc-dir-btn');
    if (dirBtn) dirBtn.addEventListener('click', () => this._openDirectHire());
    // VR 按钮为独立固定定位元素（挂在 body 上，不依赖本 UI 重建），此处确保存在
    this._ensureVRBtn();
    this._updatePositionsUI();
    this._updateSwarmHUD();
  }

  _updatePositionsUI() {
    if (!this._uiPosChips) return;
    this._uiPosChips.innerHTML = POSITIONS.filter(p => p.key !== 'hr').map(p => {
      const hired = this.hiredList.find(h => h.position.key === p.key);
      const chip = hired ? `${p.icon}${hired.name}` : `${p.icon}${p.name}`;
      const cls = hired ? 'hired' : 'open';
      const color = hired ? '#00ffa3' : p.color;
      return `<span class="cc-pos-chip ${cls}" style="border-color:${color};color:${hired ? '#7dffc8' : color}">${this.App.escapeHtml(chip)}</span>`;
    }).join('');
  }

  _setCandidate(name, posName, icon, color) {
    if (!this._uiCandName) return;
    this._uiCandName.textContent = name;
    this._uiCandPos.textContent = `${icon} 应聘岗位：${posName}`;
    this._uiCandPos.style.color = color || '';
  }

  _setStatus(text) {
    // 状态消息改为顶部轻提示：有信息才显示，几秒后自动淡出，不常驻占位
    if (!this._uiStatus) return;
    this._uiStatus.textContent = text;
    this._uiStatus.classList.add('cc-show');
    if (this._statusTimer) clearTimeout(this._statusTimer);
    this._statusTimer = setTimeout(() => {
      if (this._uiStatus) this._uiStatus.classList.remove('cc-show');
    }, 5000);
  }

  _appendTranscript(tag, text) {
    // 面试记录仅保留在数组（供复盘/扩展），字幕面板已移除，台词改由头顶气泡呈现
    this._transcript.push({ tag, text });
  }

  /** 蜂群动态信息流：让"系统化的主动互动"可见（保留最近 6 条） */
  _pushSwarmFeed(text) {
    if (!text) return;
    if (!this._uiSwarmList) return;
    const row = document.createElement('div');
    row.className = 'cc-swarm-item';
    row.textContent = text;
    this._uiSwarmList.prepend(row);
    while (this._uiSwarmList.children.length > 6) {
      this._uiSwarmList.removeChild(this._uiSwarmList.lastChild);
    }
  }

  /** 轻量内存巡检（大厅主循环每 30 秒调用）：清理已播完/暂停的音频元素，释放 base64 音频引用 */
  memoryTick() {
    if (this._audios && this._audios.length) {
      this._audios = this._audios.filter(a => {
        if (!a) return false;
        try {
          if (a.ended || a.paused) {
            a.src = '';
            return false;
          }
        } catch (e) {
          return false;
        }
        return true;
      });
    }
    // 面试/蜂群记录只保留最近 300 条（复盘与存档都不依赖全量历史）
    if (this._transcript && this._transcript.length > 300) {
      this._transcript = this._transcript.slice(-300);
    }
  }

  _showResult(score, pass, pos, cand) {
    const old = this._uiRoot.querySelector('.cc-result');
    if (old) old.remove();
    const div = document.createElement('div');
    div.className = `cc-result ${pass ? 'pass' : 'fail'}`;
    div.innerHTML = `
      <div class="cc-r-title ${pass ? 'pass' : 'fail'}">${pass ? '✅ 录用！' : '❌ 未通过'}</div>
      <div class="cc-r-sub">${this.App.escapeHtml(cand.name)} · 应聘 ${pos.icon} ${this.App.escapeHtml(pos.name)}</div>
      <div class="cc-bar"><div class="cc-bar-in ${pass ? '' : 'fail'}" data-score="${score}"></div></div>
      <div class="cc-r-score">${score}<span style="font-size:14px;color:#8fa0c6"> / 100</span></div>
      <div class="cc-r-comment">⏳ HR 正在拟定评语…</div>
    `;
    this._uiRoot.appendChild(div);
    this._uiResult = div;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const bar = div.querySelector('.cc-bar-in');
      if (bar) bar.style.width = score + '%';
    }));
    const t = setTimeout(() => {
      if (div.parentNode) div.parentNode.removeChild(div);
      if (this._uiResult === div) this._uiResult = null;
    }, 9000);
    this._timers.push(t);
  }

  _updateUI() {
    // 人性化布局：只有面试进行中才显示「当前面试」卡片，
    // 招聘结束 / 运营期 / 等待期一律隐藏，避免无用信息常驻界面
    if (!this._uiCandWrap) return;
    const active = !!this.currentSeeker && !this._disposed && this.state === 'playing';
    if (active !== this._candVisible) {
      this._candVisible = active;
      this._uiCandWrap.style.display = active ? '' : 'none';
    }
  }

  // ==================== 运营面板（评分 + 裁员） ====================

  _toggleOperationPanel() {
    if (this._opPanel) this._closeOperationPanel();
    else this._openOperationPanel();
  }

  _openOperationPanel() {
    if (this._opPanel || this._disposed) return;
    this._recalcScore();
    this._firePending = null;               // 防误触：待确认裁员按钮索引

    const mask = document.createElement('div');
    mask.className = 'cc-op-mask';
    mask.innerHTML = `
      <div class="cc-op-panel">
        <div class="cc-op-close" data-act="close">✕</div>
        <div class="cc-op-head">
          <div class="cc-op-score"><span data-r="score">0</span><small> / 100</small></div>
          <span class="cc-op-badge" data-r="badge">—</span>
        </div>
        <div class="cc-op-sec">四维健康度</div>
        <div data-r="dims"></div>
        <div class="cc-op-sec">CEO 项目任务（蜂群拆解 · 各司其职）</div>
        <div data-r="project"></div>
        <div class="cc-op-sec">公司任务看板（员工 RL 推进）</div>
        <div data-r="tasks"></div>
        <div class="cc-op-sec">组织活力 · 裁员微笑曲线</div>
        <div class="cc-op-smile" data-r="smile"></div>
        <div class="cc-op-sec">最近结算事件</div>
        <div class="cc-op-events" data-r="events"></div>
        <div class="cc-op-sec">在职员工 <span style="color:#9aa7cc;letter-spacing:0;" data-r="empcount"></span></div>
        <div data-r="emps"></div>
        <div class="cc-op-foot">
          <button class="cc-of-ok" data-act="close">✓ 继续经营</button>
        </div>
      </div>
    `;
    // 点击遮罩空白处关闭
    mask.addEventListener('click', (e) => {
      if (e.target === mask) this._closeOperationPanel();
    });
    this._uiRoot.appendChild(mask);
    this._opPanel = mask;
    this._renderOperationPanel();
  }

  _closeOperationPanel() {
    if (this._opPanel && this._opPanel.parentNode) {
      this._opPanel.parentNode.removeChild(this._opPanel);
    }
    this._opPanel = null;
    this._firePending = null;
  }

  /** 渲染/刷新运营面板（裁员、结算后调用） */
  _renderOperationPanel() {
    if (!this._opPanel) return;
    const s = this._recalcScore();
    const root = this._opPanel;

    // 总分与评级
    root.querySelector('[data-r="score"]').textContent = s.score;
    const badge = root.querySelector('[data-r="badge"]');
    const g = this._gradeOf(s.score);
    badge.textContent = `${g.grade} · ${g.title}`;
    badge.className = `cc-op-badge ${g.grade}`;

    // 五维条
    const dimDefs = [
      { key: 'vitality', label: '组织活力', max: BALANCE.VITALITY_FULL, color: '#00e5ff' },
      { key: 'quality', label: '人才质量', max: BALANCE.QUALITY_FULL, color: '#ff2bd6' },
      { key: 'stability', label: '运营稳定', max: BALANCE.STABILITY_FULL, color: '#6ee7ff' },
      { key: 'reputation', label: '公司声誉', max: BALANCE.REPUTATION_FULL, color: '#ffd166' },
      { key: 'production', label: '公司产出', max: RL_CFG.PRODUCTION_CAP, color: '#00ffa3' },
    ];
    root.querySelector('[data-r="dims"]').innerHTML = dimDefs.map(d => `
      <div class="cc-op-dim">
        <span class="cc-dl">${d.label}</span>
        <span class="cc-dt"><i style="width:${Math.round(Math.min(1, s[d.key] / d.max) * 100)}%;background:${d.color}"></i></span>
        <span class="cc-dn">${s[d.key]}/${d.max}</span>
      </div>
    `).join('');

    // CEO 项目任务：玩家布置 → 蜂群拆解的子任务看板（各司其职 + 实时进度）
    const pt = this._playerTask;
    const projectEl = root.querySelector('[data-r="project"]');
    if (projectEl) {
      projectEl.innerHTML = pt && pt.subtasks && pt.subtasks.length
        ? `<div class="cc-op-project">
            <div class="cc-op-project-title">${pt.status === 'done' ? '✅' : pt.status === 'planning' ? '🗳' : '📋'} ${this.App.escapeHtml(pt.title)} <span class="cc-op-project-status">${pt.status === 'done' ? '已完成' : pt.status === 'planning' ? '规划讨论中' : '进行中'}</span></div>
            ${pt.subtasks.map(t => {
              const pct = Math.min(100, Math.round((t.progress / Math.max(1, t.complexity)) * 100));
              const done = !!t.done;
              return `
              <div class="cc-op-task ${done ? 'done' : (pct > 0 ? 'on' : '')}">
                <span class="cc-ot-icon">${t.icon || '📋'}</span>
                <div class="cc-ot-info">
                  <div class="cc-ot-title">${this.App.escapeHtml(t.title || '')}${t.quality != null ? ` <em style="font-style:normal;color:#9aa7cc;">质量 ${Math.round(t.quality)}</em>` : ''}</div>
                  <div class="cc-ot-bar"><i style="width:${done ? 100 : pct}%;"></i></div>
                </div>
                <span class="cc-ot-pct">${done ? '✓ 完成' : pct + '%'}</span>
              </div>`;
            }).join('')}
          </div>`
        : '<div class="cc-op-empty">对员工说「布置任务：做一个恋爱游戏」，蜂群会自动拆解分工、讨论推进</div>';
    }

    // 公司任务看板：每个岗位一条任务 + RL 实时进度（原顶部蜂群栏内容，移入运营面板）
    const tasks = this._tasks || [];
    root.querySelector('[data-r="tasks"]').innerHTML = tasks.length
      ? tasks.map(t => {
          const pct = Math.min(100, Math.round((t.progress / Math.max(1, t.complexity)) * 100));
          const done = !!t.done;
          return `
          <div class="cc-op-task ${done ? 'done' : (pct > 0 ? 'on' : '')}">
            <span class="cc-ot-icon">${t.icon || '📋'}</span>
            <div class="cc-ot-info">
              <div class="cc-ot-title">${this.App.escapeHtml(t.title || '')}</div>
              <div class="cc-ot-bar"><i style="width:${done ? 100 : pct}%;"></i></div>
            </div>
            <span class="cc-ot-pct">${done ? '✓ 完成' : pct + '%'}</span>
          </div>`;
        }).join('')
      : '<div class="cc-op-empty">暂无任务看板</div>';

    // 微笑曲线状态
    const n = s.layoffCount || 0;
    const v0 = Math.round(this._smileVitality(n));
    const zone = n <= 0.5 ? '僵化区（不裁员）' : n <= 2.5 ? '健康区（焕新）' : n <= 3.5 ? '预警区（受损）' : '崩盘区（流失）';
    const stale = this.stats.stalePeriods || 0;
    const staleWarn = stale >= BALANCE.STALE_START
      ? `<div class="stale">⚠ 已连续 ${stale} 期未裁员，组织活力每期 −${BALANCE.STALE_STEP}（封顶 −${BALANCE.STALE_CAP}）</div>`
      : `<div>已连续 <b>${stale}</b> 期未裁员，连续 ${BALANCE.STALE_START} 期起组织僵化</div>`;
    const layoffInfo = n > 0
      ? `<div>本期已裁 <b>${n}</b> 人 → 活力 <b>${v0}</b>/40（${zone}） · 僵化已清零 ✓</div>`
      : `<div>本期未裁员 → 活力 <b>${v0}</b>/40（${zone}）</div>`;
    root.querySelector('[data-r="smile"]').innerHTML = layoffInfo + staleWarn;

    // 结算事件
    const evWrap = root.querySelector('[data-r="events"]');
    const evs = this._lastSettleEvents || [];
    evWrap.innerHTML = evs.length
      ? evs.map(ev => `<div class="cc-ev ${ev.dir}">${ev.dir === 'up' ? '▲' : '▼'} ${ev.text}${ev.delta ? `（${ev.delta > 0 ? '+' : ''}${ev.delta}）` : ''}</div>`).join('')
      : '<div class="cc-op-empty">第一轮面试结束后开始结算评分</div>';

    // 员工列表
    const hired = this.hiredList || [];
    root.querySelector('[data-r="empcount"]').textContent = `（${hired.length}/${MAX_EMPLOYEES} 人 · 本期已裁 ${n} 人）`;
    const empWrap = root.querySelector('[data-r="emps"]');
    if (!hired.length) {
      empWrap.innerHTML = '<div class="cc-op-empty">暂无在职员工，去面试招人吧</div>';
    } else {
      empWrap.innerHTML = hired.map((h, i) => {
        const fit = h.fitScore || 0;
        const fitCls = fit >= BALANCE.FIT_HIGH ? 'high' : fit >= 60 ? 'mid' : 'low';
        const fitLabel = fit >= BALANCE.FIT_HIGH ? `适配 ${fit} +${BALANCE.FIT_BONUS_HIGH}`
          : fit >= 60 ? `适配 ${fit} +${BALANCE.FIT_BONUS_NORMAL}` : `适配 ${fit} ${BALANCE.FIT_PENALTY_LOW}`;
        const pending = this._firePending === i;
        return `
          <div class="cc-op-emp">
            <div class="cc-e-ava">${h.position.icon || '💼'}</div>
            <div class="cc-e-info">
              <div class="cc-e-name">${this.App.escapeHtml(h.name)}</div>
              <div class="cc-e-pos">${this.App.escapeHtml(h.position.name)}</div>
              ${this._agentTask[h.name] ? `<div class="cc-e-rl">⚙ ${this.App.escapeHtml(this._agentTask[h.name].title)} · ${Math.min(100, Math.round(this._agentTask[h.name].progress / Math.max(1, this._agentTask[h.name].complexity) * 100))}% · 精力 ${this._agentEnergy[h.name] != null ? Math.round(this._agentEnergy[h.name]) : 100}</div>` : ''}
            </div>
            <span class="cc-e-fit ${fitCls}">${fitLabel}</span>
            <button class="cc-e-fire" data-idx="${i}" type="button">${pending ? '⚠ 确认裁？' : '裁员'}</button>
          </div>`;
      }).join('');
    }

    // 事件绑定（员工裁员按钮：两段式防误触）
    empWrap.querySelectorAll('.cc-e-fire').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = +btn.dataset.idx;
        if (this._firePending === idx) {
          this._firePending = null;
          this._layoffEmployee(hired[idx] && hired[idx].actor);
          this._renderOperationPanel();
        } else {
          this._firePending = idx;
          this._renderOperationPanel();
        }
      });
    });
    root.querySelectorAll('[data-act="close"]').forEach(el => {
      el.addEventListener('click', () => this._closeOperationPanel());
    });
  }

  _setLabelText(plane, text, color) {
    if (!plane || !plane.material) return;
    // 空安全：纹理未就绪时跳过（先检查 map 再取 image，避免解引用 null 抛异常，
    // 该异常若发生在录用流程内，会被外层异常保护当作"面试失败"处置掉刚录用的员工）
    if (!plane.material.map || !plane.material.map.image) return;
    const canvas = plane.material.map.image;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    // 字号随画布宽度等比缩放，并自动缩小至完整显示（不截断员工名）
    const shown = String(text).slice(0, 12);
    const fontSize = this._fitTextFont(ctx, shown, w, h, Math.max(40, Math.round(w * 72 / 512)));
    ctx.font = `bold ${fontSize}px "Microsoft YaHei", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = color || '#00e5ff';
    ctx.shadowBlur = 26;
    ctx.fillStyle = color || '#00e5ff';
    ctx.fillText(shown, w / 2, h / 2);
    plane.material.map.needsUpdate = true;
  }

  _drawHoloScreen(title, sub, line, colorA, colorB) {
    if (!this._holoCtx) return;
    this._holoPlaceholder = false;
    const ctx = this._holoCtx;
    const w = this._holoCanvas.width, h = this._holoCanvas.height;
    ctx.clearRect(0, 0, w, h);
    // 背景
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#04101c');
    grad.addColorStop(0.6, '#070a1c');
    grad.addColorStop(1, '#12071c');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
    // 扫描线
    ctx.fillStyle = 'rgba(0,229,255,0.045)';
    for (let y = 0; y < h; y += 8) ctx.fillRect(0, y, w, 3);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    // 标题
    ctx.shadowColor = colorA || '#00e5ff';
    ctx.shadowBlur = 30;
    ctx.fillStyle = colorA || '#00e5ff';
    ctx.font = 'bold 120px "Microsoft YaHei", sans-serif';
    ctx.fillText(title, w / 2, 110);
    // 副标题
    ctx.shadowBlur = 18;
    ctx.fillStyle = colorB || '#ff2bd6';
    ctx.font = 'bold 76px "Microsoft YaHei", sans-serif';
    ctx.fillText(sub, w / 2, 260);
    // 正文：自动换行 + 字号自适应（保证长句完整显示）
    const bodyTop = 360, bodyBottom = h - 120;
    const bodyMaxH = bodyBottom - bodyTop;
    const textStr = String(line || '');
    let fontSize = 76;
    let lines = [];
    while (fontSize >= 34) {
      ctx.font = `bold ${fontSize}px "Microsoft YaHei", sans-serif`;
      const maxW = w - 200;
      const wrapped = this._wrapHoloText(ctx, textStr, maxW);
      if (wrapped.length * fontSize * 1.35 <= bodyMaxH) { lines = wrapped; break; }
      fontSize -= 4;
    }
    if (!lines.length) {
      ctx.font = 'bold 34px "Microsoft YaHei", sans-serif';
      lines = this._wrapHoloText(ctx, textStr, w - 200);
    }
    const lineH = fontSize * 1.35;
    const maxLines = Math.max(1, Math.floor(bodyMaxH / lineH));
    const shown = lines.slice(0, maxLines);
    if (lines.length > maxLines) {
      const last = shown[maxLines - 1] || '';
      shown[maxLines - 1] = last.slice(0, Math.max(1, last.length - 1)) + '…';
    }
    ctx.fillStyle = '#eef4ff';
    ctx.shadowColor = '#00e5ff';
    ctx.shadowBlur = 22;
    let ty = bodyTop + fontSize * 0.5;
    for (const ln of shown) {
      ctx.fillText(ln, w / 2, ty);
      ty += lineH;
    }
    // 装饰
    ctx.fillStyle = 'rgba(0,229,255,.5)';
    ctx.font = '44px monospace';
    ctx.shadowBlur = 0;
    ctx.fillText('█ REC ●  LIVE INTERVIEW  █', w / 2, h - 48);
    if (this._holoTexture) this._holoTexture.needsUpdate = true;
  }

  /** 圆角矩形路径（兼容不支持 ctx.roundRect 的旧浏览器） */
  _holoRoundRect(ctx, x, y, w, h, r) {
    if (typeof ctx.roundRect === 'function') { ctx.roundRect(x, y, w, h, r); return; }
    r = Math.min(r || 0, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /** 全息屏健康仪表盘：总分 + 四维条 + 微笑曲线定位 + 结算事件（每轮面试结算后展示） */
  async _drawScoreDashboard(ms) {
    if (!this._holoCtx) return;
    const ctx = this._holoCtx;
    const w = this._holoCanvas.width, h = this._holoCanvas.height;
    const s = this._lastSettle || this._recalcScore();
    const g = this._gradeOf(s.score);
    const layoffN = s.layoffCount || 0;

    ctx.clearRect(0, 0, w, h);
    // 背景
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#04101c');
    grad.addColorStop(0.6, '#070a1c');
    grad.addColorStop(1, '#12071c');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
    // 扫描线
    ctx.fillStyle = 'rgba(0,229,255,0.045)';
    for (let y = 0; y < h; y += 8) ctx.fillRect(0, y, w, 3);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // 标题
    ctx.fillStyle = '#00e5ff';
    ctx.shadowColor = '#00e5ff';
    ctx.shadowBlur = 30;
    ctx.font = 'bold 72px "Microsoft YaHei", sans-serif';
    ctx.fillText('CYBER CORP · 公司健康仪表盘', w / 2, 96);
    ctx.shadowBlur = 14;
    ctx.fillStyle = '#8fa0c6';
    ctx.font = 'bold 40px "Microsoft YaHei", sans-serif';
    ctx.fillText(`本期待结算 · 裁员 ${layoffN} 人 · 连续未裁员 ${s.stalePeriods || 0} 期`, w / 2, 180);

    // 左区：总分 + 评级
    ctx.textAlign = 'center';
    ctx.fillStyle = '#eef4ff';
    ctx.shadowColor = '#00e5ff';
    ctx.shadowBlur = 40;
    ctx.font = 'bold 320px "Tektur", "Microsoft YaHei", sans-serif';
    ctx.fillText(String(s.score), 360, 560);
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#5f6d95';
    ctx.font = 'bold 56px "Microsoft YaHei", sans-serif';
    ctx.fillText('/ 100', 700, 560);

    // 评级徽章
    const badgeColor = g.grade === 'S' || g.grade === 'A' ? '#00ffa3'
      : g.grade === 'B' ? '#6ee7ff' : g.grade === 'C' ? '#ffd166' : '#ff4d6d';
    ctx.strokeStyle = badgeColor;
    ctx.fillStyle = badgeColor;
    ctx.globalAlpha = 0.12;
    this._holoRoundRect(ctx, 190, 700, 340, 92, 46);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 3;
    this._holoRoundRect(ctx, 190, 700, 340, 92, 46);
    ctx.stroke();
    ctx.font = 'bold 48px "Microsoft YaHei", sans-serif';
    ctx.fillText(`${g.grade} · ${g.title}`, 360, 746);

    // 左区：五维条（y 810 起，每行 76，底部留白防裁切）
    const dimDefs = [
      { key: 'vitality', label: '组织活力', max: BALANCE.VITALITY_FULL, color: '#00e5ff' },
      { key: 'quality', label: '人才质量', max: BALANCE.QUALITY_FULL, color: '#ff2bd6' },
      { key: 'stability', label: '运营稳定', max: BALANCE.STABILITY_FULL, color: '#6ee7ff' },
      { key: 'reputation', label: '公司声誉', max: BALANCE.REPUTATION_FULL, color: '#ffd166' },
      { key: 'production', label: '公司产出', max: RL_CFG.PRODUCTION_CAP, color: '#00ffa3' },
    ];
    ctx.textAlign = 'left';
    let dy = 810;
    for (const d of dimDefs) {
      const val = Math.max(0, Math.min(d.max, s[d.key]));
      const ratio = val / d.max;
      ctx.fillStyle = '#9aa7cc';
      ctx.font = 'bold 38px "Microsoft YaHei", sans-serif';
      ctx.fillText(d.label, 90, dy);
      // 条背景
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      this._holoRoundRect(ctx, 330, dy - 26, 660, 52, 26);
      ctx.fill();
      // 条填充
      ctx.fillStyle = d.color;
      this._holoRoundRect(ctx, 330, dy - 26, 660 * ratio, 52, 26);
      ctx.fill();
      // 数值
      ctx.textAlign = 'left';
      ctx.fillStyle = '#eef4ff';
      ctx.font = 'bold 40px "Microsoft YaHei", sans-serif';
      ctx.fillText(`${val} / ${d.max}`, 1040, dy);
      dy += 76;
    }

    // 右区：微笑曲线
    const cx0 = 1180, cy0 = 330, cx1 = 1950, cy1 = 900;
    const curveW = cx1 - cx0, curveH = cy1 - cy0;
    // 区域带
    const zones = [
      [0, 0.5, 'rgba(255,209,102,0.10)', '僵化'],
      [0.5, 2.5, 'rgba(0,255,163,0.10)', '健康'],
      [2.5, 3.5, 'rgba(255,209,102,0.10)', '预警'],
      [3.5, 6, 'rgba(255,77,109,0.10)', '崩盘'],
    ];
    ctx.textAlign = 'center';
    for (const [a, b, color, name] of zones) {
      const x0 = cx0 + (a / 6) * curveW, x1 = cx0 + (b / 6) * curveW;
      ctx.fillStyle = color;
      ctx.fillRect(x0, cy0, x1 - x0, curveH);
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.font = '28px "Microsoft YaHei", sans-serif';
      ctx.fillText(name, (x0 + x1) / 2, cy0 - 34);
    }
    // 网格线
    ctx.strokeStyle = 'rgba(0,229,255,0.14)';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 12]);
    for (let i = 0; i <= 4; i++) {
      const y = cy0 + (i / 4) * curveH;
      ctx.beginPath(); ctx.moveTo(cx0, y); ctx.lineTo(cx1, y); ctx.stroke();
    }
    ctx.setLineDash([]);
    // 坐标轴
    ctx.strokeStyle = 'rgba(0,229,255,0.5)';
    ctx.lineWidth = 3;
    ctx.strokeRect(cx0, cy0, curveW, curveH);
    // 曲线
    const smilePts = BALANCE.SMILE;
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 6;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < smilePts.length; i++) {
      const [n, v] = smilePts[i];
      const x = cx0 + (n / 6) * curveW;
      const y = cy1 - (v / BALANCE.VITALITY_FULL) * curveH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // 数据点
    ctx.fillStyle = '#00e5ff';
    for (const [n, v] of smilePts) {
      const x = cx0 + (n / 6) * curveW;
      const y = cy1 - (v / BALANCE.VITALITY_FULL) * curveH;
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill();
    }
    // 当前定位点（结算快照的本期裁员数）
    const curX = cx0 + (Math.min(6, Math.max(0, layoffN)) / 6) * curveW;
    const curV = this._smileVitality(layoffN);
    const curY = cy1 - (Math.min(BALANCE.VITALITY_FULL, curV) / BALANCE.VITALITY_FULL) * curveH;
    ctx.fillStyle = layoffN > 3 ? '#ff4d6d' : layoffN > 2 ? '#ffd166' : '#00ffa3';
    ctx.shadowColor = ctx.fillStyle;
    ctx.shadowBlur = 26;
    ctx.beginPath(); ctx.arc(curX, curY, 20, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#eef4ff';
    ctx.font = 'bold 34px "Microsoft YaHei", sans-serif';
    ctx.fillText(`本期裁 ${layoffN} 人 → 活力 ${Math.round(curV)}/40`, (cx0 + cx1) / 2, cy1 + 44);
    // 轴标签
    ctx.fillStyle = '#5f6d95';
    ctx.font = '28px "Microsoft YaHei", sans-serif';
    ctx.fillText('0', cx0 - 6, cy1 + 40);
    ctx.fillText('6人', cx1 + 6, cy1 + 40);
    ctx.fillText('单期裁员人数 →', (cx0 + cx1) / 2, cy1 + 86);

    // 底部：结算事件
    ctx.textAlign = 'left';
    let evY = 1020;
    for (const ev of (this._lastSettleEvents || [])) {
      ctx.fillStyle = ev.dir === 'up' ? '#00ffa3' : '#ff6d85';
      ctx.font = 'bold 36px "Microsoft YaHei", sans-serif';
      ctx.fillText(`${ev.dir === 'up' ? '▲' : '▼'} ${ev.text}${ev.delta ? `（${ev.delta > 0 ? '+' : ''}${ev.delta}）` : ''}`, 90, evY);
      evY += 56;
    }

    // 装饰
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(0,229,255,.5)';
    ctx.font = '40px monospace';
    ctx.fillText('█ HEALTH MONITOR █', w / 2, h - 40);
    if (this._holoTexture) this._holoTexture.needsUpdate = true;

    // 停留展示
    await this._sleep(ms || 2400);
  }

  /** 按最大宽度把文本拆成多行（中英文混排按实际测量宽度换行） */
  _wrapHoloText(ctx, text, maxWidth) {
    const lines = [];
    let cur = '';
    for (const ch of String(text || '')) {
      if (ctx.measureText(cur + ch).width > maxWidth && cur) {
        lines.push(cur);
        cur = ch;
      } else {
        cur += ch;
      }
    }
    if (cur) lines.push(cur);
    return lines.length ? lines : [text];
  }

  // ==================== 工具 ====================

  _sleep(ms) {
    return new Promise(resolve => {
      const t = setTimeout(() => resolve(), ms);
      this._timers.push(t);
    });
  }

  _removeUI() {
    if (this._statusTimer) {
      clearTimeout(this._statusTimer);
      this._statusTimer = null;
    }
    if (this._opPanel && this._opPanel.parentNode) {
      this._opPanel.parentNode.removeChild(this._opPanel);
      this._opPanel = null;
    }
    if (this._hrSelectUI && this._hrSelectUI.parentNode) {
      this._hrSelectUI.parentNode.removeChild(this._hrSelectUI);
      this._hrSelectUI = null;
    }
    if (this._uiRoot && this._uiRoot.parentNode) {
      this._uiRoot.parentNode.removeChild(this._uiRoot);
      this._uiRoot = null;
    }
    if (this._styleEl && this._styleEl.parentNode) {
      this._styleEl.parentNode.removeChild(this._styleEl);
      this._styleEl = null;
    }
    this._uiTrans = null;
    this._uiResult = null;
  }

  /** 重置运行时标志（恢复存档失败兜底时调用）：清空可能残留的循环/会话状态 */
  _resetRuntimeFlags() {
    this._gameLoopRunning = false;
    this._stoppedHiring = false;
    this._running = false;
    this.currentSeeker = null;
    this._chatPanelEl = null;
    this._resumeInterviewNotice = null;
  }
}

export default CyberCorpGame;
