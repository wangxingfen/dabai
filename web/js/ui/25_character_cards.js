export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  角色卡片：外形（模型）+ 声音（TTS）+ 系统提示词（人设）
   *           + 称呼 + 工具调用（开关 + 工具白名单）
   *           + 大语言模型（model / base_url / api_key，留空沿用默认）
   *  一键切换整套角色配置
   * ============================================================ */
  App.roleCardActiveId = localStorage.getItem('dabai.activeRoleCard') || null;

  App.initRoleCards = function initRoleCards() {
    // 打开 / 关闭列表弹窗
    App.roleCardBtn?.addEventListener('click', App.openRoleCardModal);
    App.roleCardModalClose?.addEventListener('click', () => App.roleCardModal.classList.remove('show'));
    App.roleCardModal?.querySelector('.modal-backdrop')?.addEventListener('click', () => App.roleCardModal.classList.remove('show'));
    // 新建卡片
    App.roleCardCreateBtn?.addEventListener('click', () => App.openRoleCardEditor(null));
    // 编辑弹窗关闭
    App.roleCardEditClose?.addEventListener('click', () => App.roleCardEditModal.classList.remove('show'));
    App.roleCardEditModal?.querySelector('.modal-backdrop')?.addEventListener('click', () => App.roleCardEditModal.classList.remove('show'));
    // 引擎 tab
    App.rcTtsTabs?.querySelectorAll('.tts-tab').forEach(tab => {
      tab.addEventListener('click', () => App.switchRcTTSEngine(tab.dataset.engine));
    });
    // 语速滑块实时显示
    App.rcRateRange?.addEventListener('input', () => {
      const v = parseInt(App.rcRateRange.value);
      App.rcRateVal.textContent = (v >= 0 ? '+' : '') + v + '%';
    });
    // 保存并切换
    App.rcApplyBtn?.addEventListener('click', App.saveRoleCard);
    // 删除
    App.rcDeleteBtn?.addEventListener('click', App.deleteRoleCard);
    // 编辑弹窗内上传新模型
    App.rcModelUploadBtn?.addEventListener('click', () => App.rcModelFileInput?.click());
    App.rcModelFileInput?.addEventListener('change', e => {
      const f = e.target.files[0];
      if (f) App.uploadModelFile(f);
      e.target.value = '';
    });
    // LLM 配置：手动刷新模型列表；修改 base_url / api_key 时自动刷新
    App.rcLlmRefreshBtn?.addEventListener('click', () => App.loadRcLlmModels());
    let rcLlmRefreshTimer = null;
    [App.rcLlmBaseUrl, App.rcLlmApiKey].forEach(el => {
      el?.addEventListener('input', () => {
        clearTimeout(rcLlmRefreshTimer);
        rcLlmRefreshTimer = setTimeout(() => App.loadRcLlmModels(), 600);
      });
    });
    // 工具开关：关闭时禁用工具多选列表
    App.rcToolsEnabled?.addEventListener('change', () => {
      App.rcToolsField?.classList.toggle('disabled', !App.rcToolsEnabled.checked);
    });
  };

  /* ---------- 列表弹窗 ---------- */
  App.openRoleCardModal = async function openRoleCardModal() {
    App.roleCardModal?.classList.add('show');
    await App.refreshRoleCardList();
  };

  App.refreshRoleCardList = async function refreshRoleCardList() {
    try {
      const res = await fetch('/api/character_cards');
      const data = await res.json();
      App.renderRoleCardList(data.cards || []);
    } catch (e) {
      App.roleCardList.innerHTML = '<div style="text-align:center;color:var(--text-dim);padding:20px">加载失败</div>';
    }
  };

  App.renderRoleCardList = function renderRoleCardList(cards) {
    App.roleCardList.innerHTML = '';
    if (!cards.length) {
      App.roleCardList.innerHTML = '<div style="text-align:center;color:var(--text-dim);padding:20px">还没有角色卡片，点击下方按钮从当前配置创建</div>';
      return;
    }
    for (const c of cards) {
      const card = document.createElement('div');
      card.className = 'role-card';
      card.dataset.id = c.id;
      const isActive = App.roleCardActiveId === c.id;
      if (isActive) card.classList.add('active');
      const tts = c.tts || {};
      const voiceLabel = tts.engine === 'gpt_sovits'
        ? (tts.gptsovits_character || 'GPT-SoVITS')
        : (tts.edge_voice || '默认音色');
      const modelLabel = c.model_name ? c.model_name.replace(/\.(glb|gltf|vrm)$/i, '') : '默认外形';
      const promptBrief = (c.system_prompt || '').slice(0, 24) || '（未填写系统提示词）';
      const toolsCfg = c.tools;
      const toolsLabel = toolsCfg
        ? (toolsCfg.enabled === false
            ? '工具：关'
            : (toolsCfg.allowed && toolsCfg.allowed.length ? `工具：${toolsCfg.allowed.length}个` : '工具：全开'))
        : '';
      const llmCfg = c.llm || {};
      const llmLabel = llmCfg.model ? `大模型：${llmCfg.model}` : '';
      card.innerHTML = `
                <div class="role-card-avatar">${App.escapeHtml((c.name || '?').slice(0, 1))}</div>
                <div class="role-card-info">
                    <div class="role-card-name">${App.escapeHtml(c.name)}</div>
                    <div class="role-card-role">角色：${App.escapeHtml(c.role_name || c.name || 'AI助手')}</div>
                    <div class="role-card-meta">
                        <span class="role-card-tag">${App.escapeHtml(modelLabel)}</span>
                        <span class="role-card-tag">${App.escapeHtml(voiceLabel)}</span>
                        ${c.user_name ? `<span class="role-card-tag">称呼：${App.escapeHtml(c.user_name)}</span>` : ''}
                        ${toolsLabel ? `<span class="role-card-tag">${App.escapeHtml(toolsLabel)}</span>` : ''}
                        ${llmLabel ? `<span class="role-card-tag">${App.escapeHtml(llmLabel)}</span>` : ''}
                        <span class="role-card-tag">${App.escapeHtml(promptBrief)}</span>
                    </div>
                </div>
                <div class="role-card-actions">
                    <button class="role-card-edit" title="编辑">
                        <svg viewBox="0 0 24 24" width="15" height="15"><path fill="currentColor" d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>
                    </button>
                </div>
                <div class="role-card-check" style="display:${isActive ? 'block' : 'none'}">✓</div>
            `;
      // 点击卡片 → 一键切换
      card.addEventListener('click', e => {
        if (e.target.closest('.role-card-edit')) return;
        App.applyRoleCard(c.id);
      });
      // 编辑
      card.querySelector('.role-card-edit').addEventListener('click', e => {
        e.stopPropagation();
        App.openRoleCardEditor(c.id);
      });
      App.roleCardList.appendChild(card);
    }
  };

  /* ---------- 编辑弹窗 ---------- */
  App.openRoleCardEditor = async function openRoleCardEditor(cardId) {
    App.rcEditingId = cardId || null;
    App.rcDeleteBtn.style.display = cardId ? '' : 'none';
    document.getElementById('role-card-edit-title').textContent = cardId ? '编辑角色卡片' : '新建角色卡片';
    App.roleCardEditModal?.classList.add('show');

    // 懒加载模型列表
    await App.loadRcModels();
    // 懒加载音色列表
    await App.loadRcVoices();
    // 懒加载工具列表
    await App.loadRcTools();

    if (cardId) {
      // 编辑：用卡片数据回填
      try {
        const res = await fetch('/api/character_cards');
        const data = await res.json();
        const c = (data.cards || []).find(x => x.id === cardId);
        if (c) {
          App.fillRoleCardForm(c);
          return;
        }
      } catch (e) { /* 回退到当前配置 */ }
    }
    // 新建：捕获当前配置回填
    const current = await App.captureCurrentRole();
    App.fillRoleCardForm(current);
  };

  App.captureCurrentRole = async function captureCurrentRole() {
    const current = {
      name: '',
      role_name: 'AI助手',
      user_name: '',
      model_url: '',
      model_name: '',
      system_prompt: '',
      tts: {
        engine: 'edge_tts',
        edge_voice: '',
        edge_rate: '+8%',
        gptsovits_url: 'http://127.0.0.1:7860/',
        gptsovits_ref_audio: '',
        gptsovits_character: ''
      },
      tools: { enabled: true, allowed: [] },
      llm: { model: '', base_url: '', api_key: '' }
    };
    try {
      const [ttsRes, roleRes, nameRes, toolsRes, llmRes] = await Promise.all([
        fetch('/api/tts/config').then(r => r.json()),
        fetch('/api/config/role').then(r => r.json()),
        fetch('/api/config/user_name').then(r => r.json()),
        fetch('/api/config/tools').then(r => r.json()),
        fetch('/api/llm/config').then(r => r.json())
      ]);
      // 当前模型（localStorage 中持久化）
      const savedModel = JSON.parse(localStorage.getItem('dabai.currentModel') || 'null');
      if (savedModel && savedModel.url) {
        current.model_url = savedModel.url;
        current.model_name = savedModel.name || savedModel.url.split('/').pop();
      }
      if (ttsRes) {
        current.tts.engine = ttsRes.engine || 'edge_tts';
        current.tts.edge_voice = ttsRes.edge_voice || '';
        current.tts.edge_rate = ttsRes.edge_rate || '+8%';
        current.tts.gptsovits_url = ttsRes.gptsovits_url || '';
        current.tts.gptsovits_ref_audio = ttsRes.gptsovits_ref_audio || '';
        current.tts.gptsovits_character = ttsRes.gptsovits_character || '';
      }
      if (roleRes) {
        current.role_name = roleRes.role_name || 'AI助手';
        current.system_prompt = roleRes.system_prompt || '';
      }
      if (nameRes) {
        current.user_name = nameRes.user_name || '';
      }
      if (toolsRes) {
        current.tools = {
          enabled: !!toolsRes.enable_tools,
          allowed: toolsRes.allowed_tools || []
        };
      }
      if (llmRes) {
        current.llm = {
          model: llmRes.model || '',
          base_url: llmRes.base_url || '',
          api_key: llmRes.api_key || ''
        };
      }
    } catch (e) {
      console.warn('捕获当前角色配置失败', e);
    }
    return current;
  };

  App.fillRoleCardForm = function fillRoleCardForm(data) {
    const tts = data.tts || {};
    App.rcName.value = data.name || '';
    App.rcRoleName.value = data.role_name || '';
    App.rcUserName.value = data.user_name || '';
    App.rcSystemPrompt.value = data.system_prompt || '';
    // 模型
    if (data.model_url) {
      App.rcModelSelect.value = data.model_url;
    }
    // 大语言模型
    const llm = data.llm || {};
    App.rcLlmBaseUrl.value = llm.base_url || '';
    App.rcLlmApiKey.value = llm.api_key || '';
    App.rcLlmModel.value = llm.model || '';
    // 从服务商自动加载模型列表（留空时回退到全局默认配置的提供方）
    App.loadRcLlmModels();
    // TTS
    App.switchRcTTSEngine(tts.engine || 'edge_tts');
    const rate = parseInt(tts.edge_rate) || 0;
    App.rcRateRange.value = rate;
    App.rcRateVal.textContent = (rate >= 0 ? '+' : '') + rate + '%';
    if (tts.edge_voice) {
      App.rcVoiceSelect.value = tts.edge_voice;
    }
    App.rcGsoUrl.value = tts.gptsovits_url || '';
    App.rcGsoRef.value = tts.gptsovits_ref_audio || '';
    App.rcGsoChar.value = tts.gptsovits_character || '';
    // 工具配置
    const toolsCfg = data.tools || {};
    App.rcToolsEnabled.checked = toolsCfg.enabled !== false;
    App.rcToolsField?.classList.toggle('disabled', toolsCfg.enabled === false);
    App.renderRcTools(toolsCfg.allowed || []);
  };

  App.loadRcModels = async function loadRcModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      App.rcModelSelect.innerHTML = '<option value="">默认（不切换外形）</option>';
      for (const m of data.models || []) {
        const opt = document.createElement('option');
        opt.value = m.url;
        opt.textContent = m.name;
        App.rcModelSelect.appendChild(opt);
      }
    } catch (e) {
      App.rcModelSelect.innerHTML = '<option value="">加载失败</option>';
    }
  };

  App.loadRcVoices = async function loadRcVoices() {
    if (App.rcVoicesLoaded) return;
    App.rcVoiceSelect.innerHTML = '<option value="">加载中…</option>';
    try {
      const res = await fetch('/api/tts/voices');
      const data = await res.json();
      App.rcVoiceSelect.innerHTML = '';
      for (const v of data.voices || []) {
        const opt = document.createElement('option');
        opt.value = v.name;
        opt.textContent = v.friendly || v.name;
        App.rcVoiceSelect.appendChild(opt);
      }
      App.rcVoicesLoaded = true;
    } catch (e) {
      App.rcVoiceSelect.innerHTML = '<option value="">加载失败</option>';
    }
  };

  /** 从模型提供方自动加载可用大模型列表，填充到 datalist（支持手动输入模型名） */
  App.loadRcLlmModels = async function loadRcLlmModels() {
    if (!App.rcLlmModelList) return;
    const baseUrl = App.rcLlmBaseUrl?.value.trim() || '';
    const apiKey = App.rcLlmApiKey?.value.trim() || '';
    const tipEl = App.rcLlmTip;
    if (tipEl) tipEl.textContent = '正在加载模型列表…';
    try {
      const params = new URLSearchParams();
      if (baseUrl) params.set('base_url', baseUrl);
      if (apiKey) params.set('api_key', apiKey);
      const res = await fetch('/api/llm/models?' + params.toString());
      const data = await res.json();
      const list = data.models || [];
      App.rcLlmModelList.innerHTML = '';
      for (const m of list) {
        const opt = document.createElement('option');
        opt.value = m;
        App.rcLlmModelList.appendChild(opt);
      }
      if (tipEl) {
        tipEl.textContent = data.error
          ? `模型列表加载失败：${data.error}（可手动输入模型名）`
          : (list.length
              ? `已从服务商加载 ${list.length} 个模型，可直接选择或手动输入`
              : '服务商未返回模型，可手动输入模型名');
      }
    } catch (e) {
      if (tipEl) tipEl.textContent = '模型列表加载失败（可手动输入模型名）';
    }
  };

  App.loadRcTools = async function loadRcTools() {
    if (App.rcToolsLoaded) return;
    App.rcToolsList.innerHTML = '<div class="tts-tip" style="padding:8px">加载中…</div>';
    try {
      const res = await fetch('/api/tools');
      const data = await res.json();
      App.allRcTools = data.tools || [];
      App.rcToolsLoaded = true;
    } catch (e) {
      App.allRcTools = [];
      App.rcToolsList.innerHTML = '<div class="tts-tip" style="padding:8px">工具列表加载失败</div>';
    }
  };

  App.renderRcTools = function renderRcTools(selected) {
    const sel = new Set(selected || []);
    if (!App.allRcTools || !App.allRcTools.length) {
      App.rcToolsList.innerHTML = '<div class="tts-tip" style="padding:8px">暂无可用工具</div>';
      return;
    }
    const allChecked = App.allRcTools.every(t => sel.has(t.name));
    let html = `<label class="tools-check-all"><input type="checkbox" id="rc-tools-all" ${allChecked ? 'checked' : ''}> 全选</label>`;
    for (const t of App.allRcTools) {
      const checked = sel.has(t.name);
      const sourceTag = t.source === 'mcp' ? '（MCP）' : '';
      html += `<label class="tools-item" title="${App.escapeHtml(t.description || '')}">` +
        `<input type="checkbox" value="${App.escapeHtml(t.name)}" ${checked ? 'checked' : ''}>` +
        `<span class="tools-item-name">${App.escapeHtml(t.name)}${sourceTag}</span>` +
        `<span class="tools-item-desc">${App.escapeHtml(t.description || '')}</span>` +
        `</label>`;
    }
    App.rcToolsList.innerHTML = html;
    const allBox = document.getElementById('rc-tools-all');
    allBox?.addEventListener('change', () => {
      App.rcToolsList.querySelectorAll('.tools-item input').forEach(cb => { cb.checked = allBox.checked; });
    });
  };

  App.collectRcTools = function collectRcTools() {
    const enabled = App.rcToolsEnabled?.checked ?? true;
    if (!enabled) return { enabled: false, allowed: [] };
    const checked = [];
    App.rcToolsList?.querySelectorAll('.tools-item input:checked').forEach(cb => checked.push(cb.value));
    // 全选（或全部勾选）时清空白名单 = 全部可用，语义更简洁
    const allChecked = checked.length === (App.allRcTools || []).length;
    return { enabled: true, allowed: allChecked ? [] : checked };
  };

  App.switchRcTTSEngine = function switchRcTTSEngine(engine) {
    App.rcTtsTabs?.querySelectorAll('.tts-tab').forEach(t => {
      t.classList.toggle('active', t.dataset.engine === engine);
    });
    if (App.rcTtsEdgePanel) App.rcTtsEdgePanel.style.display = engine === 'edge_tts' ? '' : 'none';
    if (App.rcTtsGsoPanel) App.rcTtsGsoPanel.style.display = engine === 'gpt_sovits' ? '' : 'none';
  };

  App.collectRoleCardForm = function collectRoleCardForm() {
    const engine = App.rcTtsTabs?.querySelector('.tts-tab.active')?.dataset.engine || 'edge_tts';
    return {
      name: App.rcName.value.trim(),
      role_name: App.rcRoleName.value.trim(),
      user_name: App.rcUserName?.value.trim() || '',
      model_url: App.rcModelSelect.value,
      model_name: App.rcModelSelect.selectedOptions[0]?.textContent || '',
      system_prompt: App.rcSystemPrompt.value.trim(),
      tts: {
        engine,
        edge_voice: App.rcVoiceSelect?.value || '',
        edge_rate: (parseInt(App.rcRateRange?.value) || 0) >= 0
          ? '+' + (parseInt(App.rcRateRange?.value) || 0) + '%'
          : parseInt(App.rcRateRange?.value) + '%',
        gptsovits_url: App.rcGsoUrl?.value.trim() || '',
        gptsovits_ref_audio: App.rcGsoRef?.value.trim() || '',
        gptsovits_character: App.rcGsoChar?.value.trim() || ''
      },
      tools: App.collectRcTools(),
      llm: {
        model: App.rcLlmModel?.value.trim() || '',
        base_url: App.rcLlmBaseUrl?.value.trim() || '',
        api_key: App.rcLlmApiKey?.value.trim() || ''
      }
    };
  };

  /* ---------- 保存 / 应用 / 删除 ---------- */
  App.saveRoleCard = async function saveRoleCard() {
    const payload = App.collectRoleCardForm();
    if (!payload.name) {
      App.showToast('请填写卡片名称');
      App.rcName?.focus();
      return;
    }
    let cardId = App.rcEditingId;
    try {
      if (cardId) {
        const res = await fetch(`/api/character_cards/${cardId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        cardId = data.card.id;
      } else {
        const res = await fetch('/api/character_cards', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        cardId = data.card.id;
      }
      App.roleCardEditModal?.classList.remove('show');
      App.showToast('卡片已保存');
      await App.applyRoleCard(cardId);
    } catch (err) {
      App.showToast('保存失败：' + (err.message || err));
    }
  };

  App.applyRoleCard = async function applyRoleCard(cardId) {
    try {
      const res = await fetch(`/api/character_cards/${cardId}/apply`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      App.roleCardActiveId = cardId;
      localStorage.setItem('dabai.activeRoleCard', cardId);
      // 1. 切换外形
      if (data.model_url) {
        await App.loadModelFromUrl(data.model_url, data.model_name);
      }
      // 2. 刷新 TTS 设置（后端已切换，同步前端弹窗状态）
      try {
        const ttsRes = await fetch('/api/tts/config');
        const ttsCfg = await ttsRes.json();
        App.applyTTSConfig(ttsCfg);
      } catch (e) { /* 忽略 */ }
      // 3. 切换到该卡片独立记忆空间的会话（WS 按序处理：先切会话，后续 AI 动作落在新卡片记忆里）
      if (App.ws && App.ws.readyState === WebSocket.OPEN) {
        App.ws.send(JSON.stringify({ type: 'list_sessions' }));
        if (data.session_id) {
          App.ws.send(JSON.stringify({ type: 'switch_session', session_id: data.session_id }));
        }
      }
      // 4. 通知 AI 人设已切换
      const card = data.card || {};
      const roleName = card.role_name || card.name || 'AI助手';
      const voiceDesc = (card.tts && card.tts.engine === 'gpt_sovits')
        ? (card.tts.gptsovits_character || '新的声音')
        : '新的声音';
      const nameHint = card.user_name
        ? `，称呼用户为「${card.user_name}」`
        : `，不用特定称呼，称呼用户为"你"即可`;
      const toolsCfg = card.tools || {};
      let toolsHint = '';
      if (toolsCfg.enabled === false) {
        toolsHint = '，同时你当前不能调用任何工具（能力已被关闭），需要实时信息时如实告知用户即可';
      } else if (toolsCfg.allowed && toolsCfg.allowed.length) {
        toolsHint = `，你当前只能调用以下工具：${toolsCfg.allowed.join('、')}`;
      }
      App.sendAIAction(
        `（用户为你切换了完整的角色设定：你现在叫「${roleName}」，用「${voiceDesc}」说话，人设也更新了${nameHint}${toolsHint}。` +
        `你依然是用户身边最亲近的人，请自然地以这个新身份重新认识自己，带着新性格和新语气开始和用户聊天，` +
        `不要刻意提"用户切换了角色"这件事）`, true);
      // 4. 刷新列表选中态
      App.refreshRoleCardList();
      App.showToast(`已切换角色：${card.name || roleName}`);
      App.closeRoleCardModalIfOpen();
    } catch (err) {
      App.showToast('切换失败：' + (err.message || err));
    }
  };

  /** 启动时静默恢复上次使用的角色卡片配置（仅返回卡片数据，不触发切换动作/AI 消息） */
  App.restoreActiveRoleCard = async function restoreActiveRoleCard() {
    if (!App.roleCardActiveId) return null;
    try {
      const res = await fetch('/api/character_cards');
      const data = await res.json();
      const card = (data.cards || []).find(c => c.id === App.roleCardActiveId);
      if (!card) {
        // 卡片已被删除 → 清除持久化的活动卡片标记
        App.roleCardActiveId = null;
        localStorage.removeItem('dabai.activeRoleCard');
        return null;
      }
      return card;
    } catch (e) {
      console.warn('恢复角色卡片配置失败', e);
      return null;
    }
  };

  App.deleteRoleCard = async function deleteRoleCard() {
    if (!App.rcEditingId) return;
    if (!confirm('删除这张角色卡片？')) return;
    try {
      await fetch(`/api/character_cards/${App.rcEditingId}`, { method: 'DELETE' });
      if (App.roleCardActiveId === App.rcEditingId) {
        App.roleCardActiveId = null;
        localStorage.removeItem('dabai.activeRoleCard');
      }
      App.roleCardEditModal?.classList.remove('show');
      await App.refreshRoleCardList();
      App.showToast('已删除');
    } catch (err) {
      App.showToast('删除失败');
    }
  };

  App.closeRoleCardModalIfOpen = function closeRoleCardModalIfOpen() {
    if (App.roleCardModal?.classList.contains('show')) {
      App.roleCardModal.classList.remove('show');
    }
  };
  /* ============================================================
   *  启动
   * ============================================================ */
});
