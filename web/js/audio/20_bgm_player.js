// ========== 背景音乐播放器 (BGM Player) ==========

export default function init_20_bgm_player(App) {
  let bgmAudio = null;
  let currentBgmName = null;
  let volume = 0.3;
  let lastTime = 0;       // 上次记录的播放位置
  let loopCount = 0;       // 循环次数

  /**
   * 通知服务器当前 BGM 状态
   */
  function notifyBgmState(name) {
    if (!App.ws || App.ws.readyState !== WebSocket.OPEN) return;
    App.ws.send(JSON.stringify({ type: 'set_bgm', name: name || null }));
  }

  // 创建 Audio 元素
  function ensureAudio() {
    if (!bgmAudio) {
      bgmAudio = new Audio();
      bgmAudio.loop = true;
      bgmAudio.volume = volume;
      // 通过 timeupdate 检测循环（loop=true 时 ended 不会触发）
      bgmAudio.addEventListener('timeupdate', () => {
        const t = bgmAudio.currentTime;
        // 当前时间比上次记录的时间显著回跳 → 说明完成了一次循环
        if (t < lastTime - 1.0) {
          loopCount++;
          console.log(`[BGM] 第 ${loopCount} 次循环结束`);
          App.sendAIAction && App.sendAIAction(`（《${currentBgmName}》已经循环播放了第${loopCount}遍，音乐还在继续陪伴着你）`);
        }
        lastTime = t;
      });
    }
    return bgmAudio;
  }

  /**
   * 播放背景音乐
   * @param {string} url - 音频文件 URL
   * @param {string} name - 音乐文件名（用于显示）
   */
  App.playBGM = function (url, name) {
    const audio = ensureAudio();
    // 如果正在播放同一首，不做任何事
    if (currentBgmName === name && !audio.paused) {
      return;
    }
    // 停止当前播放
    audio.pause();
    audio.src = url;
    audio.load();
    audio.play().then(() => {
      currentBgmName = name;
      lastTime = 0;
      loopCount = 0;
      console.log('[BGM] 开始播放:', name);
      notifyBgmState(name);
    }).catch(err => {
      console.warn('[BGM] 播放失败:', err);
    });
  };

  /**
   * 停止背景音乐
   */
  App.stopBGM = function () {
    if (bgmAudio) {
      bgmAudio.pause();
      bgmAudio.src = '';
      currentBgmName = null;
      console.log('[BGM] 已停止');
      notifyBgmState(null);
    }
  };

  /**
   * 设置背景音乐音量
   * @param {number} vol - 音量 0.0 ~ 1.0
   */
  App.setBGMVolume = function (vol) {
    volume = Math.max(0, Math.min(1, vol));
    if (bgmAudio) {
      bgmAudio.volume = volume;
    }
  };

  /**
   * 获取当前正在播放的 BGM 名称
   */
  App.getCurrentBGM = function () {
    return currentBgmName;
  };

  /**
   * BGM 是否正在播放
   */
  App.isBGMPlaying = function () {
    return bgmAudio && !bgmAudio.paused;
  };
}
