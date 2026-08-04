/* ============================================================
 * 3D 虚拟 AI 角色陪聊 · 前端逻辑 (ES Module)
 *
 * 功能：
 * - 导入 GLB/GLTF/VRM 模型
 * - VRM 表情驱动口型同步 (viseme 'aa')
 * - GLTF Morph Target 口型同步 (自动检索 mouth/jaw 形变键)
 * - 模型自动居中、缩放、朝向修正
 * - 拖拽导入 + 文件选择 + 历史模型管理
 * - WebSocket 实时通信 + 语音录制 + TTS 播放
 * ============================================================ */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

export const App = {
  THREE,
  GLTFLoader,
  VRMLoaderPlugin,
  VRMUtils,
};
