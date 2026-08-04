export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  消息渲染
   * ============================================================ */
  // 聊天面板收起时新消息到达 → 提示切换按钮
  App.notifyFullscreenChat = function notifyFullscreenChat() {
    const panel = document.getElementById('chat-panel');
    if (panel.classList.contains('collapsed')) {
      App.chatToggle.classList.add('has-new');
    }
  };
  App.isFullscreen = false;

  // 限制消息 DOM 节点数量，防止长时间聊天导致 DOM 膨胀卡顿
  App._trimMessages = function _trimMessages() {
    const all = App.messagesEl.querySelectorAll('.msg:not(.streaming):not(.typing)');
    if (all.length > 200) {
      // 移除最旧的一半消息
      const removeCount = Math.floor(all.length / 2);
      for (let i = 0; i < removeCount; i++) {
        all[i].remove();
      }
    }
  };

  App.addUserMsg = function addUserMsg(text, fromVoice = false) {
    const el = document.createElement('div');
    el.className = 'msg user';
    el.textContent = text;
    if (fromVoice) el.title = '来自语音输入';
    App.messagesEl.appendChild(el);
    App._trimMessages();
    App.scrollToBottom();
    App.notifyFullscreenChat();
  };
  App.addAIMsg = function addAIMsg(text) {
    const el = document.createElement('div');
    el.className = 'msg ai';
    el.textContent = text;
    App.messagesEl.appendChild(el);
    App._trimMessages();
    App.scrollToBottom();
    App.notifyFullscreenChat();
  };
  App.addSystemMsg = function addSystemMsg(text) {
    const el = document.createElement('div');
    el.className = 'msg system';
    el.textContent = text;
    App.messagesEl.appendChild(el);
    App._trimMessages();
  };
  App.showTyping = function showTyping() {
    App.removeTyping();
    const el = document.createElement('div');
    el.className = 'msg ai typing';
    el.id = 'typing-indicator';
    el.innerHTML = '<span></span><span></span><span></span>';
    App.messagesEl.appendChild(el);
    App.scrollToBottom();
  };
  App.removeTyping = function removeTyping() {
    const t = App.$('typing-indicator');
    if (t) t.remove();
  };
  App.scrollToBottom = function scrollToBottom() {
    App.messagesEl.scrollTop = App.messagesEl.scrollHeight;
    App.updateScrollHint();
  };
  App.updateScrollHint = function updateScrollHint() {
    const atBottom = App.messagesEl.scrollHeight - App.messagesEl.scrollTop - App.messagesEl.clientHeight < 60;
    App.scrollHint.classList.toggle('show', !atBottom);
  };
  App.messagesEl.addEventListener('scroll', App.updateScrollHint);
  App.scrollHint.addEventListener('click', App.scrollToBottom);

  /* ============================================================
   *  Toast
   * ============================================================ */
});