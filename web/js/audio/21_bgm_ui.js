// ========== 背景音乐管理 UI ==========

export default function init_21_bgm_ui(App) {
  /* ============================================================
   *  BGM 弹窗
   * ============================================================ */
  App.openBgmModal = function openBgmModal() {
    App.bgmModal.classList.add('show');
    App.refreshBgmList();
  };
  App.closeBgmModal = function closeBgmModal() {
    App.bgmModal.classList.remove('show');
  };

  App.refreshBgmList = async function refreshBgmList() {
    try {
      const res = await fetch('/api/bgm');
      const data = await res.json();
      App.renderBgmList(data.bgm || []);
    } catch (e) {
      App.showToast('获取BGM列表失败');
    }
  };

  App.renderBgmList = function renderBgmList(items) {
    App.bgmListEl.innerHTML = '';
    const currentName = App.getCurrentBGM();
    if (items.length === 0) {
      App.bgmListEl.innerHTML = '<div style="text-align:center;color:var(--text-dim);padding:20px" id="bgm-empty-msg">暂无音乐，请上传</div>';
      return;
    }

    // 添加停止按钮（总是可用）
    const stopCard = document.createElement('div');
    stopCard.className = 'model-card';
    if (!currentName) stopCard.classList.add('active');
    stopCard.innerHTML = `
      <div class="model-preview default-preview">
        <svg viewBox="0 0 24 24" width="32" height="32"><path fill="currentColor" d="M6 6h12v12H6z"/></svg>
      </div>
      <div class="model-info">
        <div class="model-name">无音乐 · 停止播放</div>
        <div class="model-meta">静音模式</div>
      </div>
      <div class="model-check" style="display:${!currentName ? 'block' : 'none'}">✓</div>
    `;
    stopCard.addEventListener('click', () => {
      App.stopBGM();
      App.refreshBgmList();
    });
    App.bgmListEl.appendChild(stopCard);

    for (const m of items) {
      const card = document.createElement('div');
      card.className = 'model-card';
      card.dataset.url = m.url;
      card.dataset.name = m.name;
      const isActive = currentName === m.name;
      if (isActive) card.classList.add('active');
      const sizeKB = (m.size / 1024).toFixed(0);
      const sizeStr = sizeKB > 1024 ? `${(sizeKB / 1024).toFixed(1)}MB` : `${sizeKB}KB`;
      card.innerHTML = `
        <div class="model-preview">${m.type.toUpperCase()}</div>
        <div class="model-info">
          <div class="model-name">${App.escapeHtml(m.name)}</div>
          <div class="model-meta">${m.type.toUpperCase()} · ${sizeStr}</div>
        </div>
        <div class="model-actions">
          <button class="model-delete" title="删除">
            <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
          </button>
        </div>
        <div class="model-check" style="display:${isActive ? 'block' : 'none'}">✓</div>
      `;

      // 点击播放
      card.addEventListener('click', e => {
        if (e.target.closest('.model-delete')) return;
        App.playBGM(m.url, m.name);
        App.refreshBgmList();
        App.showToast('正在播放: ' + m.name);
      });

      // 删除
      card.querySelector('.model-delete').addEventListener('click', async e => {
        e.stopPropagation();
        if (!confirm(`删除音乐 ${m.name}？`)) return;
        try {
          const res = await fetch(`/api/bgm/${encodeURIComponent(m.name)}`, { method: 'DELETE' });
          if (!res.ok) throw new Error('删除失败');
          if (currentName === m.name) App.stopBGM();
          App.refreshBgmList();
          App.showToast('已删除');
        } catch (err) {
          App.showToast('删除失败');
        }
      });

      App.bgmListEl.appendChild(card);
    }
  };

  App.uploadBgmFile = async function uploadBgmFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac'].includes(ext)) {
      App.showToast('仅支持 mp3 / wav / ogg / m4a / aac / flac');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      App.showToast('文件超过 50MB 上限');
      return;
    }
    App.showModelLoading(`上传 ${file.name} …`);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/api/bgm/upload', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      App.refreshBgmList();
      App.showToast('上传成功');
    } catch (err) {
      App.showToast('上传失败: ' + (err.message || err));
    } finally {
      App.hideModelLoading();
    }
  };
}
