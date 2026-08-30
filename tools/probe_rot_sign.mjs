// 纯 three.js 数学验证：normalized 手臂沿 +X 时，rotation.z=+1.35 会转去哪？
import * as THREE from 'three';

// 模拟 normalized leftUpperArm：父为 identity，子臂沿 local +Y = world +X（与模型一致）
const parent = new THREE.Object3D();
parent.updateMatrixWorld(true);
const arm = new THREE.Object3D();
parent.add(arm);
// 让 arm 的 local +Y 指向 world +X（模拟 tpose_rest 后的骨骼方向）
arm.rotation.set(0, 0, 0);
arm.rotation.setFromRotationMatrix(new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(0, 1, 0), Math.PI / 2));
parent.updateMatrixWorld(true);
const q = new THREE.Quaternion();
arm.getWorldQuaternion(q);
console.log('初始 arm 世界 Y 方向:', new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray().map(v => v.toFixed(2)).join(','));

// 应用前端 ARM_REST_Z = 1.35（Euler XYZ 顺序，rotation.z 绕局部 Z）
arm.rotation.set(0, 0, 1.35);
parent.updateMatrixWorld(true);
arm.getWorldQuaternion(q);
console.log('rotation.z=+1.35 后 arm 世界 Y 方向:', new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray().map(v => v.toFixed(2)).join(','));
console.log('=> 正值若使 Y 朝上说明手高举；朝下说明自然下垂');

// 也测试 -1.35（右臂用负值）
arm.rotation.set(0, 0, -1.35);
parent.updateMatrixWorld(true);
arm.getWorldQuaternion(q);
console.log('rotation.z=-1.35 后 arm 世界 Y 方向:', new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray().map(v => v.toFixed(2)).join(','));