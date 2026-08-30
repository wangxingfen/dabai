// 检查 three-vrm normalized rig 的实际方向：把 VRM 加载后，看 normalized leftUpperArm 世界方向
import fs from 'fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';

const file = process.argv[2] || 'models/莉丽拉.vrm';
const buf = fs.readFileSync(file);
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

loader.parse(buf, '', (gltf) => {
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error('VRM 未注册');
  const h = vrm.humanoid;
  const names = ['leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','leftUpperLeg','rightUpperLeg','spine','head'];
  for (const n of names) {
    const raw = h.getRawBoneNode(n);
    const norm = h.getNormalizedBoneNode(n);
    if (!raw || !norm) { console.log(n, 'missing'); continue; }
    raw.updateWorldMatrix(true, false);
    norm.updateWorldMatrix(true, false);
    const q = new THREE.Quaternion();
    raw.getWorldQuaternion(q);
    const dirR = new THREE.Vector3(0,1,0).applyQuaternion(q);
    norm.getWorldQuaternion(q);
    const dirN = new THREE.Vector3(0,1,0).applyQuaternion(q);
    console.log(`${n}: raw_Y=[${dirR.x.toFixed(2)},${dirR.y.toFixed(2)},${dirR.z.toFixed(2)}] norm_Y=[${dirN.x.toFixed(2)},${dirN.y.toFixed(2)},${dirN.z.toFixed(2)}]`);
  }
  // 检查 normalized hips 世界位置
  const hips = h.getNormalizedBoneNode('hips');
  if (hips) {
    hips.updateWorldMatrix(true, false);
    const p = new THREE.Vector3();
    hips.getWorldPosition(p);
    console.log('norm hips pos:', p.toArray().map(v=>v.toFixed(3)).join(','));
  }
  process.exit(0);
}, undefined, (e) => {
  console.error('LOAD_FAIL', e && e.message);
  process.exit(1);
});