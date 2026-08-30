// 实测：加载 VRM 后打印 normalized 骨骼方向（不用 process.exit，让事件循环自然结束）
import fs from 'fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';

const file = process.argv[2] || 'models/莉丽拉.vrm';
const buf = fs.readFileSync(file);
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

const write = (s) => fs.appendFileSync('C:/Users/WANGXI~1/AppData/Local/Temp/opencode/rig_log.txt', s + '\n');

write('=== ' + file + ' ===');
try {
  loader.parse(buf, '', (gltf) => {
    try {
      const vrm = gltf.userData.vrm;
      if (!vrm) { write('VRM 未注册'); return; }
      const h = vrm.humanoid;
      for (const n of ['leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'leftUpperLeg', 'spine']) {
        const raw = h.getRawBoneNode(n);
        const norm = h.getNormalizedBoneNode(n);
        if (!raw || !norm) { write(n + ' missing'); continue; }
        raw.updateWorldMatrix(true, false);
        norm.updateWorldMatrix(true, false);
        const q = new THREE.Quaternion();
        raw.getWorldQuaternion(q);
        const dirR = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
        norm.getWorldQuaternion(q);
        const dirN = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
        write(n + ' raw_Y=[' + [dirR.x, dirR.y, dirR.z].map(v => v.toFixed(2)).join(',') + '] norm_Y=[' + [dirN.x, dirN.y, dirN.z].map(v => v.toFixed(2)).join(',') + ']');
      }
      // 模拟 applyVrmRestPose：leftUpperArm.rotation.z = +1.35
      const la = h.getNormalizedBoneNode('leftUpperArm');
      if (la) {
        la.rotation.set(0, 0, 1.35);
        h.update();
        h.getRawBoneNode('leftUpperArm').updateWorldMatrix(true, false);
        const q = new THREE.Quaternion();
        h.getRawBoneNode('leftUpperArm').getWorldQuaternion(q);
        const d = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
        write('after rest pose (z=1.35): raw leftUpperArm Y=[' + [d.x, d.y, d.z].map(v => v.toFixed(2)).join(',') + ']');
      }
    } catch (e) {
      write('verify fail: ' + (e && e.message));
    }
  }, undefined, (e) => {
    write('load fail: ' + (e && e.message));
  });
} catch (e) {
  write('sync fail: ' + (e && e.message));
}

// 不 process.exit，等待异步回调
setTimeout(() => { write('--- done ---'); }, 30000);