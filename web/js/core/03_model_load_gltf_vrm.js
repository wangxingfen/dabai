export default (function init(App) {
  const {
    THREE: THREE,
    GLTFLoader: GLTFLoader,
    VRMLoaderPlugin: VRMLoaderPlugin,
    VRMUtils: VRMUtils
  } = App;
  /* ============================================================
   *  3D 模型加载 (GLTF / VRM)
   * ============================================================ */
  App.showModelLoading = function showModelLoading(text = '加载模型中…') {
    App.modelLoadingText.textContent = text;
    App.modelLoading.classList.add('show');
  };
  App.hideModelLoading = function hideModelLoading() {
    App.modelLoading.classList.remove('show');
  };
  App.loadModelFromUrl = async function loadModelFromUrl(url, name) {
    App.showModelLoading(`加载 ${name} …`);
    try {
      const gltf = await App.gltfLoader.loadAsync(url);
      await App.applyLoadedModel(gltf, url, name);
      // 持久化
      localStorage.setItem('dabai.currentModel', JSON.stringify({
        url,
        name
      }));
      // 恢复上次场景中的位置/缩放（VR 模式下跳过：出生点已在 applyLoadedModel
      // 中设为当前角色位置，此处若恢复会把新角色拉回旧坐标，覆盖新出生点）
      const inVR = !!(App.xrPresenting || (App.xrMode && App.xrMode !== 'off'));
      if (!inVR) setTimeout(App.applySavedPositions, 200);
      App.saveSceneState();
      App.showToast(`已切换为 ${name}`);

      // 防重：仅当模型名与上次发送不一致时才通知服务器和触发 AI
      const isNewAvatar = App._sentAvatarName !== name;
      if (isNewAvatar) {
        App._sentAvatarName = name;
        if (App.ws && App.ws.readyState === WebSocket.OPEN) {
          App.ws.send(JSON.stringify({
            type: 'set_avatar',
            name
          }));
        }
        // 启动阶段不触发 AI 动作（避免页面刷新时模型恢复加载触发 AI 说话）
        if (!App._isBooting) {
          App.sendAIAction(`（你换上了新造型${name}，看看自己的新样子，感受一下现在的心情和气质）`, true);
        }
      }
    } catch (err) {
      console.error('模型加载失败:', err);
      App.showToast('模型加载失败：' + (err.message || err));
    } finally {
      App.hideModelLoading();
    }
  };
  App.applyLoadedModel = async function applyLoadedModel(gltf, url, name) {
    // VR 模式下切换角色：出生点沿用当前角色位置（而非背景中心原点），
    // 否则新角色会瞬移到原点导致用户眼前角色消失；非 VR 模式不受影响
    const inVR = !!(App.xrPresenting || (App.xrMode && App.xrMode !== 'off'));
    const prevPos = inVR && App.modelGroup
      ? { x: App.modelGroup.position.x, y: App.modelGroup.position.y, z: App.modelGroup.position.z }
      : null;
    // VR 中清空进行中的平滑传送（注视归位等），避免加载期间把新角色瞬移走
    if (inVR) App._smoothTeleport = null;
    // 清理上一个模型
    App.disposeModel();
    const root = gltf.scene;
    App.modelGroup = new THREE.Group();
    App.modelGroup.add(root);
    App.scene.add(App.modelGroup);

    // 处理 VRM
    App.vrm = gltf.userData.vrm || null;
    if (App.vrm) {
      App.modelType = 'vrm';
      // VRM 0.x 朝向修正：将模型转到面向 +Z，与相机保持一致
      VRMUtils.rotateVRM0(App.vrm);
      VRMUtils.removeUnnecessaryVertices(root);

      // 调优弹簧骨骼：增大阻尼、降低刚度，防止头发衣物乱摆穿模
      App.calmSpringBones();

      // 缓存 humanoid 骨骼 (使用 normalized 空间，可叠加旋转)
      App.vrmBones = {};
      if (App.vrm.humanoid) {
        const h = App.vrm.humanoid;
        const want = ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head', 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand', 'leftUpperLeg', 'rightUpperLeg', 'leftLowerLeg', 'rightLowerLeg'];
        for (const name of want) {
          const bone = h.getNormalizedBoneNode(name);
          if (bone) App.vrmBones[name] = bone;
        }
        App.headBone = App.vrmBones.head || null;
        console.log('[VRM] 骨骼缓存:', Object.keys(App.vrmBones).join(', '));
        // 立即应用放松站姿（手臂自然下垂），避免 T-pose 闪烁
        App.applyVrmRestPose();
      }

      // 列出所有可用表情并缓存映射名
      if (App.vrm.expressionManager) {
        const allExpr = Object.keys(App.vrm.expressionManager.expressionMap || {});
        console.log('[VRM] 加载成功:', name);
        console.log('[VRM] 可用表情 (' + allExpr.length + '):', allExpr.join(', ') || '(无)');
        App.refreshExprNames();
      } else {
        console.log('[VRM] 加载成功但无表情管理器:', name);
      }
    } else {
      App.modelType = 'gltf';
      // 检索 morph targets 用于口型同步
      App.morphTargets = [];
      root.traverse(obj => {
        if (obj.isMesh && obj.morphTargetInfluences && obj.morphTargetDictionary) {
          for (const key in obj.morphTargetDictionary) {
            const lk = key.toLowerCase();
            if (lk.includes('mouth') || lk.includes('jaw') || lk.includes('open') || lk.includes('aa') || lk.includes('viseme')) {
              App.morphTargets.push({
                mesh: obj,
                index: obj.morphTargetDictionary[key],
                name: key
              });
            }
          }
        }
      });
      // 找头部骨骼 (常见命名)
      App.headBone = null;
      root.traverse(obj => {
        if (obj.isBone && !App.headBone) {
          const n = obj.name.toLowerCase();
          if (n === 'head' || n === 'head_' || n.includes('head')) App.headBone = obj;
        }
      });
      console.log('[GLTF] loaded', name, 'morphTargets:', App.morphTargets.length, 'headBone:', !!App.headBone);
    }

    // 自动缩放与定位：让模型高度约 2.2 单位，脚踩 y=0
    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const targetHeight = 2.2;
    const scale = targetHeight / Math.max(0.001, size.y);
    root.scale.setScalar(scale);
    // 重新计算 box
    root.updateMatrixWorld(true);
    const box2 = new THREE.Box3().setFromObject(root);
    const center2 = box2.getCenter(new THREE.Vector3());
    root.position.x += -center2.x;
    root.position.z += -center2.z;
    root.position.y += -box2.min.y; // 脚踩地面

    // 从背景包围盒中心往下射线检测地板，以地板+2m 作为出生高度；
    // VR 模式下切换角色时沿用当前角色位置（出生点=当前位置），高度按新位置地板重算
    App._modelGroupBaseY = root.position.y;
    if (App.backgroundGroup) {
      if (prevPos) {
        const floorY = App._findFloorY ? App._findFloorY(prevPos.x, prevPos.z) : 0;
        App.modelGroup.position.set(prevPos.x, floorY + 2.0, prevPos.z);
      } else {
        const floorY = App._findFloorY ? App._findFloorY(App._bgCenterX, App._bgCenterZ) : 0;
        App.modelGroup.position.set(App._bgCenterX, floorY + 2.0, App._bgCenterZ);
      }
    }

    if (App.proceduralChar) App.proceduralChar.visible = false;
    App.currentAvatar = App.modelGroup;
    // 立即面朝相机（避免首帧背对用户后再缓慢转过来）
    App.smoothRotY = App.computeBodyFaceCam(App.modelGroup);
    App.smoothRotX = 0;
    // VR 中重新捕获世界平移对象集合：_xrWorldObjs 进入 VR 时缓存的还是旧模型
    // 引用，若不同步更新，新角色不参与手柄移动/转身/缩放的世界平移
    if (inVR && App._xrCaptureWorld) App._xrCaptureWorld();

    // 更新选中状态
    App.refreshModelListSelection(name);
  };
  App.disposeModel = function disposeModel() {
    if (App.modelGroup) {
      if (App.selectedTarget === App.modelGroup) App.clearSelection();
      App.modelGroup.traverse(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());else obj.material.dispose();
        }
      });
      App.scene.remove(App.modelGroup);
      App.modelGroup = null;
    }
    App.vrm = null;
    App.morphTargets = [];
    App.headBone = null;
    App.vrmBones = {};
    App.exprNames = {
      mouth: null,
      blink: null,
      happy: null
    };
    App.smoothRotY = 0;
    App.smoothRotX = 0;
    App.smoothWalkFaceOff = 0; // 重置行走朝向偏移
    App.modelType = null;
    App.currentAvatar = null;
  };
  /* ============================================================
   *  3D 背景场景加载
   * ============================================================ */
});