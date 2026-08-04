export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  背景场景管理 UI
   * ============================================================ */
  App.openBgModal = function openBgModal() {
    App.bgModal.classList.add('show');
    App.refreshBackgroundList();
  };
  App.closeBgModal = function closeBgModal() {
    App.bgModal.classList.remove('show');
  };
  App.refreshBackgroundList = async function refreshBackgroundList() {
    try {
      const res = await fetch('/api/backgrounds');
      const data = await res.json();
      App.renderBackgroundList(data.backgrounds || []);
    } catch (e) {
      App.showToast('获取背景列表失败');
    }
  };
  App.renderBackgroundList = function renderBackgroundList(items) {
    App.bgListEl.innerHTML = '';
    const currentSaved = JSON.parse(localStorage.getItem('dabai.currentBackground') || 'null');
    for (const m of items) {
      const card = document.createElement('div');
      card.className = 'model-card';
      card.dataset.url = m.url;
      card.dataset.name = m.name;
      const isActive = currentSaved && currentSaved.url === m.url;
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
                <button class="model-rename" title="重命名">
                    <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>
                </button>
                <button class="model-delete" title="删除">
                    <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
                </button>
            </div>
            <div class="model-check" style="display:${isActive ? 'block' : 'none'}">✓</div>
        `;
      const nameEl = card.querySelector('.model-name');
      const renameBtn = card.querySelector('.model-rename');

      // 点击切换背景
      card.addEventListener('click', e => {
        if (e.target.closest('.model-delete, .model-rename, .model-name-edit')) return;
        App.loadBackgroundFromUrl(m.url, m.name);
        App.closeBgModal();
      });

      // 重命名
      renameBtn.addEventListener('click', async e => {
        e.stopPropagation();
        const oldName = nameEl.textContent;
        const input = document.createElement('input');
        input.className = 'model-name-edit';
        input.value = oldName;
        nameEl.replaceWith(input);
        input.focus();
        input.select();
        const doRename = async () => {
          const newName = input.value.trim();
          input.replaceWith(nameEl);
          if (!newName || newName === oldName) return;
          try {
            const res = await fetch(`/api/background/${encodeURIComponent(oldName)}/rename`, {
              method: 'PUT',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                new_name: newName
              })
            });
            if (!res.ok) {
              const err = await res.json().catch(() => ({}));
              throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            card.dataset.name = data.new_name;
            card.dataset.url = data.url;
            nameEl.textContent = data.new_name;
            // 如果重命名的是当前使用的背景，更新 localStorage
            if (currentSaved && currentSaved.url === `/backgrounds/${oldName}`) {
              currentSaved.url = data.url;
              currentSaved.name = data.new_name;
              localStorage.setItem('dabai.currentBackground', JSON.stringify(currentSaved));
            }
            App.showToast('已重命名');
          } catch (err) {
            App.showToast('重命名失败：' + (err.message || err));
          }
        };
        input.addEventListener('keydown', e => {
          if (e.key === 'Enter') {
            e.preventDefault();
            doRename();
          }
          if (e.key === 'Escape') {
            input.value = oldName;
            input.replaceWith(nameEl);
          }
        });
        input.addEventListener('blur', doRename);
      });

      // 删除
      card.querySelector('.model-delete').addEventListener('click', async e => {
        e.stopPropagation();
        if (!confirm(`删除背景 ${m.name}？`)) return;
        try {
          await fetch(`/api/background/${encodeURIComponent(m.name)}`, {
            method: 'DELETE'
          });
          if (currentSaved && currentSaved.url === m.url) localStorage.removeItem('dabai.currentBackground');
          App.refreshBackgroundList();
          App.showToast('已删除');
        } catch (err) {
          App.showToast('删除失败');
        }
      });
      App.bgListEl.appendChild(card);
    }
    App.refreshBgListSelection(currentSaved ? currentSaved.name : null);
  };
  App.refreshBgListSelection = function refreshBgListSelection(activeName) {
    const cards = App.bgListEl.querySelectorAll('.model-card');
    cards.forEach(c => {
      const name = c.dataset.name;
      const isActive = activeName === null ? false : name === activeName;
      c.classList.toggle('active', isActive);
      const check = c.querySelector('.model-check');
      if (check) check.style.display = isActive ? 'block' : 'none';
    });
  };
  App.uploadBackgroundFile = async function uploadBackgroundFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['glb', 'gltf', 'vrm'].includes(ext)) {
      App.showToast('仅支持 .glb / .gltf / .vrm');
      return;
    }
    if (file.size > 80 * 1024 * 1024) {
      App.showToast('文件超过 80MB 上限');
      return;
    }
    App.showModelLoading(`上传背景 ${file.name} …`);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/api/background/upload', {
        method: 'POST',
        body: fd
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      App.showToast('上传成功，正在加载…');
      await App.loadBackgroundFromUrl(data.url, data.name);
      App.closeBgModal();
    } catch (err) {
      App.showToast('上传失败：' + (err.message || err));
    } finally {
      App.hideModelLoading();
    }
  };
  /* ============================================================
   *  事件绑定
   * ============================================================ */
});