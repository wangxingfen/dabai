import type { AppKernel, MixamoClipInfo, AnimLibraryConfig, AnimCategory } from '../types/app-kernel.js';

export default (function init(App: AppKernel) {
  /* ============================================================
   *  Mixamo 动作库加载器 —— 批量加载 + 情绪映射 + 分类索引
   *
   *  定位：在 mixamo_retarget 基础上的"库管理层"。
   *  - 从 animation-library.json 读取动作元数据
   *  - 批量加载 FBX 并重定向注册到 App.mixamoClips
   *  - 提供按情绪/分类/名称的播放 API
   *  - 与情绪控制器联动：情绪变化时自动切换待机动作
   *
   *  用法：
   *    // 加载整个库（VRM 加载完成后调用）
   *    await App.loadAnimationLibrary()
   *
   *    // 按名称播放
   *    App.playLibraryClip('laugh')
   *
   *    // 按情绪随机播放一个动作
   *    App.playEmotionClip('happy')
   *
   *    // 按分类随机播放
   *    App.playCategoryClip('dance')
   * ============================================================ */

  // ==================== 运行时状态 ====================
  App._animLibraryConfig = null;
  App._animLibraryLoaded = false;
  App._animLibraryLoading = false;
  App._animLibraryStats = { total: 0, loaded: 0, failed: 0 };
  App._animLibraryGen = 0; // 模型切换代数：加载过程中换模型则中止旧加载
  // 角色专属动作配置：null = 未配置/关闭 → 执行全部动作
  App._roleAnimationConfig = null;

  /**
   * 重置动作库状态（模型销毁/切换时调用，避免旧模型的片段残留）
   */
  App.resetAnimationLibrary = function resetAnimationLibrary() {
    App._animLibraryGen++;
    App.mixamoClips = {};
    if (App.mixamoMixer) {
      App.mixamoMixer.stopAllAction();
      App.mixamoMixer = null;
    }
    App._mixamoActiveClip = null;
    App._mixamoTailActive = false;
    App._mixamoTailName = null;
    App._mixamoTailRem = 0;
    App._mixamoTailTotal = 0;
    App._animLibraryConfig = null;
    App._animLibraryLoaded = false;
    App._animLibraryLoading = false;
    App._animLibraryStats = { total: 0, loaded: 0, failed: 0 };
    App._mixamoActiveClipLoop = false;
    App._mixamoActiveClipStart = 0;
    App._mixamoSwitchTimer = null;
    App._mixamoEmotionEnabled = false;
    App._mixamoHipsRestPos = null; // 模型切换 → 旧模型的 hips 静息位失效，下次加载片段时重新捕获
    App.clearAnimState(); // 模型切换 → 上报无动作
    App._mixamoLastEmotion = 'neutral';
    App._mixamoEmotionCooldown = 0;
  };

  // ==================== 角色专属动作过滤 ====================
  /**
   * 设置当前角色的专属动作配置（由角色卡片应用时调用）。
   * @param config  { enabled, allowed }；enabled=false 或 allowed 为空 → 执行全部动作
   */
  App.setRoleAnimationConfig = function setRoleAnimationConfig(config) {
    if (config && config.enabled && Array.isArray(config.allowed) && config.allowed.length > 0) {
      App._roleAnimationConfig = { enabled: true, allowed: [...config.allowed] };
    } else {
      App._roleAnimationConfig = null;
    }
  };

  /**
   * 判断动作是否允许当前角色执行（未配置专属动作时全部允许）
   */
  App.isAnimAllowed = function isAnimAllowed(name: string): boolean {
    if (!App._roleAnimationConfig) return true;
    return App._roleAnimationConfig.allowed.includes(name);
  };

  /**
   * 返回当前角色允许的所有已加载动作信息
   */
  App.getAllowedClips = function getAllowedClips(): any[] {
    return Object.values(App.mixamoClips).filter(c => App.isAnimAllowed(c.name));
  };

  /**
   * 从允许的动作中随机选一个循环动作（用于待机），优先情绪匹配
   */
  App.pickAllowedLoopClip = function pickAllowedLoopClip(preferEmotion?: string): string | null {
    const allowed = App.getAllowedClips().filter(c => c.loop);
    if (!allowed.length) return null;
    if (preferEmotion) {
      const match = allowed.filter(c => c.emotion === preferEmotion);
      if (match.length) return match[Math.floor(Math.random() * match.length)].name;
    }
    return allowed[Math.floor(Math.random() * allowed.length)].name;
  };

  /**
   * 从允许的动作中按情绪选一个表达动作（无匹配返回 null）
   */
  App.pickAllowedClipByEmotion = function pickAllowedClipByEmotion(emotion: string): string | null {
    const allowed = App.getAllowedClips();
    if (!allowed.length) return null;
    const match = allowed.filter(c => c.emotion === emotion);
    if (match.length) return match[Math.floor(Math.random() * match.length)].name;
    return null;
  };

  // ==================== 配置加载 ====================
  /**
   * 加载动作库配置文件
   */
  App.loadAnimLibraryConfig = async function loadAnimLibraryConfig(): Promise<AnimLibraryConfig | null> {
    if (App._animLibraryConfig) return App._animLibraryConfig;
    try {
      const res = await fetch('/anim/animation-library.json');
      if (!res.ok) {
        console.warn('[AnimLib] 配置文件加载失败:', res.status);
        return null;
      }
      const config = await res.json();
      App._animLibraryConfig = config;
      console.log(`[AnimLib] 配置加载完成，共 ${countTotalAnimations(config)} 个动作`);
      return config;
    } catch (e) {
      console.error('[AnimLib] 配置加载异常:', e);
      return null;
    }
  };

  /**
   * 统计配置中的动作总数
   */
  function countTotalAnimations(config: AnimLibraryConfig): number {
    let total = 0;
    for (const key in config.categories) {
      total += config.categories[key].animations.length;
    }
    return total;
  }

  // ==================== 批量加载 ====================
  /**
   * 加载单个动作：优先烘焙缓存（/anim/baked/<name>.json，秒开零重定向），
   * 无烘焙缓存或加载失败时回退 FBX 实时重定向。
   */
  async function loadClip(anim: { name: string; file: string }, fbxUrl: string) {
    const bakedUrl = '/anim/baked/' + anim.name + '.json';
    const baked = await App.loadBakedMixamoClip(anim.name, bakedUrl);
    if (baked) return baked;
    return App.loadMixamoAnimation(fbxUrl, anim.name);
  }
  /**
   * 加载整个动作库（VRM 加载完成后调用）
   * @param lazy  是否懒加载（false=全部预加载，true=只加载待机类）
   */
  App.loadAnimationLibrary = async function loadAnimationLibrary(lazy = false): Promise<number> {
    if (App._animLibraryLoaded || App._animLibraryLoading) {
      return Object.keys(App.mixamoClips).length;
    }
    App._animLibraryLoading = true;
    const gen = App._animLibraryGen; // 记录当前模型代数，加载中途换模型则中止

    const config = await App.loadAnimLibraryConfig();
    if (!config) {
      App._animLibraryLoading = false;
      return 0;
    }

    const baseUrl = config.baseUrl || '/anim/';
    const allAnims: { name: string; file: string; emotion?: string; loop?: boolean }[] = [];

    // 收集所有动作
    for (const catKey in config.categories) {
      const cat = config.categories[catKey];
      for (const anim of cat.animations) {
        allAnims.push(anim);
      }
    }

    App._animLibraryStats = { total: allAnims.length, loaded: 0, failed: 0 };

    // 懒加载模式：只加载 idle 分类
    let toLoad = allAnims;
    if (lazy && config.categories['idle']) {
      toLoad = config.categories['idle'].animations;
      console.log(`[AnimLib] 懒加载模式：优先加载 ${toLoad.length} 个待机动作`);
    }

    // 逐个加载（避免并发过多）
    for (const anim of toLoad) {
      // 加载中途模型被切换 → 中止（旧模型的片段绑定已失效）
      if (App._animLibraryGen !== gen) {
        console.warn('[AnimLib] 模型已切换，中止动作库加载');
        App._animLibraryLoading = false;
        return App._animLibraryStats.loaded;
      }
      // VR 会话期间暂停在盘加载：单条 FBX 解析 + 骨骼重定向可能占用主线程
      // 数百毫秒，会掐断 XR 帧循环造成画面冻结/抖动。等到退出 VR 再继续；
      // 等待期间模型切换同样中止。
      while ((App.xrPresenting || (App.xrMode && App.xrMode !== 'off')) && App._animLibraryGen === gen) {
        await new Promise(r => setTimeout(r, 500));
      }
      if (App._animLibraryGen !== gen) {
        console.warn('[AnimLib] 模型已切换，中止动作库加载');
        App._animLibraryLoading = false;
        return App._animLibraryStats.loaded;
      }
      const url = baseUrl + anim.file;
      try {
        // 优先烘焙缓存（秒开零重定向），无缓存/失败回退 FBX 实时重定向
        let clip = await loadClip(anim, url);
        if (clip && App.mixamoClips[anim.name]) {
          // 补充元数据
          App.mixamoClips[anim.name].emotion = anim.emotion;
          App.mixamoClips[anim.name].loop = anim.loop ?? false;
          App._animLibraryStats.loaded++;
        } else {
          App._animLibraryStats.failed++;
        }
      } catch (e) {
        console.warn(`[AnimLib] 加载失败: ${anim.name}`, e);
        App._animLibraryStats.failed++;
      }
    }

    App._animLibraryLoaded = true;
    App._animLibraryLoading = false;

    console.log(`[AnimLib] 加载完成: ${App._animLibraryStats.loaded}/${App._animLibraryStats.total} 成功, ${App._animLibraryStats.failed} 失败`);

    // 如果是懒加载，后台继续加载剩余动作
    if (lazy) {
      const remaining = allAnims.filter(a => !App.mixamoClips[a.name]);
      if (remaining.length > 0) {
        backgroundLoadRemaining(remaining, baseUrl, gen);
      }
    }

    return App._animLibraryStats.loaded;
  };

  /**
   * 后台懒加载剩余动作
   */
  function backgroundLoadRemaining(anims: any[], baseUrl: string, gen: number) {
    console.log(`[AnimLib] 后台加载剩余 ${anims.length} 个动作...`);
    let idx = 0;
    const loadNext = async () => {
      if (idx >= anims.length) {
        console.log(`[AnimLib] 后台加载完成，总计 ${Object.keys(App.mixamoClips).length} 个动作`);
        return;
      }
      // 模型已切换 → 中止后台加载
      if (App._animLibraryGen !== gen) {
        console.warn('[AnimLib] 模型已切换，中止后台加载');
        return;
      }
      // VR 会话期间暂停后台加载：FBX 解析 + 重定向阻塞主线程会掐断 XR 帧循环
      if (App.xrPresenting || (App.xrMode && App.xrMode !== 'off')) {
        setTimeout(loadNext, 500);
        return;
      }
      const anim = anims[idx++];
      const url = baseUrl + anim.file;
      try {
        // 优先烘焙缓存（秒开零重定向），无缓存/失败回退 FBX 实时重定向
        let clip = await loadClip(anim, url);
        if (clip && App.mixamoClips[anim.name]) {
          App.mixamoClips[anim.name].emotion = anim.emotion;
          App.mixamoClips[anim.name].loop = anim.loop ?? false;
          App._animLibraryStats.loaded++;
        }
      } catch (e) {
        // 静默失败，后台加载不打断主流程
      }
      // 间隔加载，避免阻塞主线程
      setTimeout(loadNext, 200);
    };
    loadNext();
  }

  // ==================== 播放 API ====================
  /**
   * 播放库中的动作片段
   * @param name  动作名称
   * @param opts  播放选项
   */
  App.playLibraryClip = function playLibraryClip(name: string, opts?: any): boolean {
    if (!App.isAnimAllowed(name)) {
      console.warn(`[AnimLib] 动作 ${name} 不在当前角色专属动作列表，已拦截`);
      return false;
    }
    const info = App.mixamoClips[name];
    if (!info) {
      console.warn(`[AnimLib] 动作不存在: ${name}`);
      return false;
    }
    const loop = opts?.loop ?? info.loop ?? false;
    App.playMixamoClip(name, { ...opts, loop });
    App.updateAnimState(name); // 上报当前动作（说话时大白知道自己在做什么）
    return true;
  };

  /**
   * 按情绪随机播放一个关联动作
   * @param emotion  情绪标签
   * @param opts     播放选项
   */
  App.playEmotionClip = function playEmotionClip(emotion: string, opts?: any): string | null {
    const config = App._animLibraryConfig;
    if (!config?.emotionMap) return null;

    const pool = config.emotionMap[emotion];
    if (!pool || pool.length === 0) return null;

    // 筛选已加载且当前角色允许的动作
    const available = pool.filter(name => App.mixamoClips[name] && App.isAnimAllowed(name));
    if (available.length === 0) {
      console.warn(`[AnimLib] 情绪 ${emotion} 没有当前角色允许的动作`);
      return null;
    }

    // 强度分层：emotionMap 数组按 [爆发 → 手势 → 待机] 排序，
    // 按当前唤醒度选段——高唤醒偏爆发动作，低唤醒偏安静待机，中段偏手势表达。
    const arousal = (App.pad && App.pad.arousal) || 0.5;
    const n = available.length;
    let slice;
    if (arousal > 0.6) {
      slice = available.slice(0, Math.max(1, Math.ceil(n * 0.4)));
    } else if (arousal > 0.3) {
      slice = available.slice(Math.floor(n * 0.3), Math.max(1, Math.ceil(n * 0.7)));
    } else {
      slice = available.slice(Math.floor(n * 0.6));
    }
    const name = slice[Math.floor(Math.random() * slice.length)];
    App.playLibraryClip(name, opts);
    return name;
  };

  /**
   * 按分类随机播放一个动作
   * @param category  分类名（idle/gesture/emotion/walk/dance/pose）
   * @param opts      播放选项
   */
  App.playCategoryClip = function playCategoryClip(category: string, opts?: any): string | null {
    const config = App._animLibraryConfig;
    if (!config?.categories?.[category]) return null;

    const cat = config.categories[category];
    const available = cat.animations.filter(a => App.mixamoClips[a.name] && App.isAnimAllowed(a.name));
    if (available.length === 0) return null;

    const anim = available[Math.floor(Math.random() * available.length)];
    App.playLibraryClip(anim.name, opts);
    return anim.name;
  };

  /**
   * 获取指定分类下所有可用动作名
   */
  App.getCategoryClips = function getCategoryClips(category: string): string[] {
    const config = App._animLibraryConfig;
    if (!config?.categories?.[category]) return [];
    return config.categories[category].animations
      .filter(a => App.mixamoClips[a.name] && App.isAnimAllowed(a.name))
      .map(a => a.name);
  };

  /**
   * 获取指定情绪关联的所有可用动作名
   */
  App.getEmotionClips = function getEmotionClips(emotion: string): string[] {
    const config = App._animLibraryConfig;
    if (!config?.emotionMap?.[emotion]) return [];
    return config.emotionMap[emotion].filter(name => App.mixamoClips[name] && App.isAnimAllowed(name));
  };

  /**
   * 获取加载统计
   */
  App.getAnimLibraryStats = function getAnimLibraryStats() {
    return { ...App._animLibraryStats, available: Object.keys(App.mixamoClips).length };
  };

  // ==================== 统一动作调度 ====================
  // 统一规则：当前情绪 + 场景分类 → 在该分类的在盘动作里随机挑一个播放。
  // - 场景分类按情绪应景偏好选择（LIBRARY_EMOTION_SCENES：情绪→候选分类，越靠前越应景）；
  // - 分类内优先情绪匹配的子池，无匹配再整类随机；并避免与上一次连续重复；
  // - 与手写程序式动作（POSE/WALK/TURN/DANCE）共用同一个调度口：
  //   库可用时优先播在盘动作，库未加载或无可用动作时由调用方回退程序式动作；
  // - walk 分类可由外部显式指定（pickLibraryActionByScene('walk')），但自主轮换
  //   默认不选 walk：FBX 行走带位移轨道，原地播放会让角色滑步，行走应景由
  //   既有的移动系统（程序式走路）负责。
  App._lastScheduledClip = null; // 防连续重复：上次统一调度播放的动作名

  // ==================== 动作状态上报 ====================
  // 前端每次播放/停止库动作时维护当前动作状态，并节流上报后端（anim_state 消息）；
  // 后端在生成回复时注入 LLM 上下文（【你现在的动作】），
  // 大白说话时就知道自己正在做什么动作，回复内容可以自然地配合当前动作。
  App._currentAnimState = null;       // 当前动作 { name, category, emotion }
  App._lastAnimAnnounce = 0;          // 上报节流时间戳（600ms 防抖）

  function animCategoryOf(name: string): string {
    const config = App._animLibraryConfig;
    if (!config || !config.categories) return '';
    for (const key in config.categories) {
      const cat = config.categories[key];
      if (cat.animations.some(a => a.name === name)) return key;
    }
    return '';
  }

  /** 更新当前动作状态并触发上报（分类与情绪自动从动作库配置补齐） */
  App.updateAnimState = function updateAnimState(name: string) {
    const info = App.mixamoClips[name];
    App._currentAnimState = {
      name,
      category: animCategoryOf(name),
      emotion: (info && info.emotion) || ''
    };
    App._announceAnimState();
  };

  /** 清空当前动作状态并上报（动作停止/模型切换时） */
  App.clearAnimState = function clearAnimState() {
    App._currentAnimState = null;
    App._announceAnimState();
  };

  /** 节流上报：动作变化时同步给后端，1.5s 内合并 */
  App._announceAnimState = function _announceAnimState() {
    const now = performance.now();
    if (now - App._lastAnimAnnounce < 600) return;
    App._lastAnimAnnounce = now;
    if (!App.ws || App.ws.readyState !== WebSocket.OPEN) return;
    const st = App._currentAnimState;
    try {
      App.ws.send(JSON.stringify({
        type: 'anim_state',
        anim: st ? { name: st.name, category: st.category, emotion: st.emotion } : null
      }));
    } catch (e) {
      // 上报失败不影响动作播放
    }
  };

  // 情绪 → 应景场景分类候选（越靠前越应景；不指定场景时按当前情绪在此随机选一个）
  App.LIBRARY_EMOTION_SCENES = {
    happy:      ['idle', 'gesture', 'emotion', 'dance'],
    excited:    ['gesture', 'emotion', 'dance'],
    sad:        ['idle', 'pose', 'emotion'],
    angry:      ['idle', 'gesture'],
    surprised:  ['gesture', 'emotion'],
    shy:        ['idle', 'pose', 'gesture', 'emotion'],
    thoughtful: ['idle', 'gesture'],
    tired:      ['pose', 'idle', 'emotion'],
    calm:       ['pose', 'idle'],
    proud:      ['pose', 'gesture'],
    playful:    ['gesture', 'dance', 'emotion'],
    love:       ['gesture', 'emotion'],
    neutral:    ['idle', 'pose', 'gesture', 'emotion', 'dance']
  };

  // 各场景循环动作的播放时长上限（秒区间）：循环动作播够时间就释放回程序式动作，
  // 避免一个循环动作永久霸占全身骨骼
  // 循环动作限时已整体缩短：与调度器的“播够即轮换”双保险，任何循环动作
  // 都不会长时间霸占（最长约 5~6s 必换），角色始终处于小幅轮换状态
  App.LIBRARY_SCENE_HOLD = {
    idle: [3, 6], pose: [3, 6], gesture: [2.5, 5],
    emotion: [2.5, 5], dance: [3, 5], walk: [3, 6]
  };

  /**
   * 统一规则核心：在指定场景分类的在盘动作里随机挑一个播放。
   * @param scene  分类名（idle/gesture/emotion/walk/dance/pose）
   * @param preferEmotion  优先匹配的情绪（缺省取当前情绪）
   * @param opts  { hold?: number[秒区间], anyEmotion?: bool }
   * @returns 播放的动作名；该分类没有可播动作时返回 null
   */
  App.pickLibraryActionByScene = function pickLibraryActionByScene(scene, preferEmotion, opts) {
    const config = App._animLibraryConfig;
    if (!config || !config.categories || !config.categories[scene]) return null;
    if (App._mixamoActiveClip) return null; // 已有动作在播，不打断

    const cat = config.categories[scene];
    // 只播“已加载 + 当前角色允许”的在盘动作
    let available = cat.animations.filter(a => App.mixamoClips[a.name] && App.isAnimAllowed(a.name));
    if (!available.length) return null;

    const emotion = preferEmotion || App.emotionSource || 'neutral';
    let pool = available.filter(a => a.emotion === emotion);
    if (!pool.length || (opts && opts.anyEmotion)) pool = available;
    // 避免与上一次连续重复
    if (pool.length > 1 && App._lastScheduledClip) {
      const noRepeat = pool.filter(a => a.name !== App._lastScheduledClip);
      if (noRepeat.length) pool = noRepeat;
    }
    const anim = pool[Math.floor(Math.random() * pool.length)];

    // 循环动作限时持有（到点释放回程序式动作）；单次动作播完自动释放
    const hold = (opts && opts.hold) || App.LIBRARY_SCENE_HOLD[scene] || [4, 7];
    const holdMs = (hold[0] + Math.random() * (hold[1] - hold[0])) * 1000;
    if (anim.loop) {
      setTimeout(() => {
        if (App._mixamoActiveClip === anim.name) App.stopMixamoClip();
      }, holdMs);
    }

    App.playLibraryClip(anim.name, { loop: anim.loop });
    App._lastScheduledClip = anim.name;
    console.log('[AnimLib] 统一调度: [' + scene + '] ' + anim.name +
      ' (动作情绪=' + anim.emotion + ', 当前情绪=' + emotion + ', 限时=' + (holdMs / 1000).toFixed(1) + 's)');
    return anim.name;
  };

  /**
   * 统一调度入口：按当前情绪选应景场景分类，在该分类随机播一个在盘动作。
   * 播放成功时预置好程序式调度的间隔——动作播完（单次 finished / 循环限时到）
   * 释放回调度口后，按常规间隔自然恢复轮换。
   * @param scene  可选：显式指定场景分类（如 'walk'）；缺省按当前情绪应景选择
   */
  // 场景“动态感”权重：视觉幅度小的分类适当压低，动作感强的分类提高——
  // 避免轮换来轮换去都是小幅待机/姿势（看起来“一直没动”）
  // 2026-08-28 修正：gesture 分类（wave/salute/praying/point/thumbs_up/
  // come_here/stop_gesture/phone_call 等）几乎全是举手动作，权重过高导致
  // 角色频繁举手（“总是举起手”）。大幅压低 gesture，提高 idle/pose，
  // 让角色以自然待机为主、偶尔才做手势。
  const SCENE_VIVIDNESS = { idle: 1.2, pose: 1.1, gesture: 0.45, emotion: 0.8, dance: 0.7 };
  let _lastSceneUsed = ''; // 上次实际播放的场景：避免连续两次同场景（尤其是小幅动作）

  App.tryStartLibraryAction = function tryStartLibraryAction(scene, opts) {
    if (!App._animLibraryLoaded || !App._animLibraryConfig) return null;
    const emotion = App.emotionSource || App._mixamoLastEmotion || 'neutral';
    const scenes = scene ? [scene] : (App.LIBRARY_EMOTION_SCENES[emotion] || ['idle', 'gesture', 'pose']);
    // 场景分类加权随机（顺序即偏好：靠前权重高，但仍有机会选后面的分类，
    // 避免“永远只播第一个分类”的死板）：位置权重 × 动态感权重
    const weights = scenes.map((s, i) => Math.max(0.08, 0.5 - i * 0.14) * (SCENE_VIVIDNESS[s] || 1));
    // 加权不重复抽取：一次生成候选顺序（权重越高越靠前），按序尝试
    const pool = scenes.slice();
    const order = [];
    while (pool.length) {
      const ws = pool.map(s => weights[scenes.indexOf(s)]);
      const totalW = ws.reduce((a, b) => a + b, 0);
      let r = Math.random() * totalW;
      let idx = 0;
      for (let i = 0; i < pool.length; i++) {
        r -= ws[i];
        if (r <= 0) { idx = i; break; }
      }
      order.push(pool.splice(idx, 1)[0]);
    }
    for (const s of order) {
      // 不连续两次落在同一场景（小幅动作的 idle/pose 尤其避免连播）
      if (s === _lastSceneUsed && order.length > 1) continue;
      const name = App.pickLibraryActionByScene(s, emotion, opts);
      if (!name) continue; // 该分类无可播动作 → 换场景
      if (name === App._lastScheduledClip && order.length > 1) continue; // 撞重复动作 → 换场景
      _lastSceneUsed = s;
      App.nextActionTimer = App.ACTION_GAP_MIN + Math.random() * (App.ACTION_GAP_MAX - App.ACTION_GAP_MIN);
      return name;
    }
    // 全部场景都跳过（比如上一轮已播过该场景且没有其他选择）→ 允许重复，保证一直在动
    for (const s of order) {
      const name = App.pickLibraryActionByScene(s, emotion, opts);
      if (name) {
        _lastSceneUsed = s;
        App.nextActionTimer = App.ACTION_GAP_MIN + Math.random() * (App.ACTION_GAP_MAX - App.ACTION_GAP_MIN);
        return name;
      }
    }
    return null; // 应景分类全无可播动作 → 调用方回退程序式动作
  };

  // ==================== 情绪联动 ====================
  // 情绪变化时自动切换待机动作的逻辑挂在 emotion_controller 扩展中
});
