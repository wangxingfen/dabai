export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  Toast
   * ============================================================ */
  App.toastTimer = null;
  App.showToast = function showToast(text) {
    App.toastEl.textContent = text;
    App.toastEl.classList.add('show');
    clearTimeout(App.toastTimer);
    App.toastTimer = setTimeout(() => App.toastEl.classList.remove('show'), 2500);
  };
  /* ============================================================
   *  模型管理 UI
   * ============================================================ */
});