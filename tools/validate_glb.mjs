import fs from 'fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const file = process.argv[2] || 'models/莉丽拉.vrm';
const buf = fs.readFileSync(file);
console.log('buf bytes:', buf.length);
const loader = new GLTFLoader();
loader.parse(buf, '', (gltf) => {
  console.log('GLTF_PARSE_OK scenes:', gltf.scene.children.length);
  console.log('extensionsUsed:', JSON.stringify(gltf.parser.json.extensionsUsed));
  process.exit(0);
}, undefined, (e) => {
  console.error('GLTF_PARSE_FAIL', e && e.message);
  process.exit(1);
});