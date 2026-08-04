/* ============================================================
 * 游戏场景自动生成器
 *
 * 根据游戏类型自动生成3D场景元素：
 * - 地形/地面
 * - 障碍物/墙壁
 * - 装饰物/粒子效果
 * - 光照氛围
 *
 * 所有生成的场景对象都会追踪，退出游戏时自动清理。
 * ============================================================ */

export class GameSceneGenerator {
  constructor(app) {
    this.App = app;
    this.THREE = app.THREE;
    this.generatedObjects = [];
    this.sceneLighting = null; // 保存原始光照引用，退出时恢复
    this._savedAmbientIntensity = null;
  }

  /**
   * 生成通用游戏场景环境
   * @param {Object} config - 场景配置
   * @param {string} config.style - 风格: 'maze' | 'open_field' | 'dungeon' | 'space'
   * @param {number} config.size - 场景大小
   * @param {string} config.colorScheme - 色调: 'warm' | 'cool' | 'mystical'
   */
  generateEnvironment(config = {}) {
    const { style = 'open_field', size = 20, colorScheme = 'mystical' } = config;
    const THREE = this.THREE;

    // 保存当前光照状态
    this._saveLighting();

    // 设置游戏氛围光照
    this._applyGameLighting(colorScheme);

    // 生成地面
    this._generateGround(size, style, colorScheme);

    // 生成粒子效果
    this._generateParticles(style, colorScheme, size);

    // 生成边界
    this._generateBoundary(size, style, colorScheme);

    // 清空角色位置
    const avatar = this.App.currentAvatar;
    if (avatar) {
      avatar.position.set(0, 0, 0);
    }

    return this.generatedObjects;
  }

  /**
   * 生成迷宫墙壁
   * @param {Array<Array<number>>} mazeData - 迷宫数据 (0=空地, 1=墙壁)
   * @param {number} cellSize - 单元格大小
   */
  generateMazeWalls(mazeData, cellSize = 2) {
    const THREE = this.THREE;
    const walls = [];
    const rows = mazeData.length;
    const cols = mazeData[0].length;
    const wallHeight = 3;
    const offsetX = -cols * cellSize / 2 + cellSize / 2;
    const offsetZ = -rows * cellSize / 2 + cellSize / 2;

    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x3a3a6a,
      roughness: 0.4,
      metalness: 0.3,
      emissive: 0x111133,
      emissiveIntensity: 0.3
    });
    const wallGeo = new THREE.BoxGeometry(cellSize * 0.9, wallHeight, cellSize * 0.9);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (mazeData[r][c] === 1) {
          const wall = new THREE.Mesh(wallGeo, wallMat);
          wall.position.set(offsetX + c * cellSize, wallHeight / 2, offsetZ + r * cellSize);
          wall.castShadow = true;
          wall.receiveShadow = true;
          wall.userData.isWall = true;
          this.App.scene.add(wall);
          this.generatedObjects.push(wall);
          walls.push(wall);
        }
      }
    }
    return walls;
  }

  /**
   * 生成宝箱/目标点
   * @param {THREE.Vector3} position
   * @param {Object} options
   */
  generateTreasure(position, options = {}) {
    const THREE = this.THREE;
    const { color = 0xffd700, size = 0.5, glowColor = 0xffaa00 } = options;

    const group = new THREE.Group();
    group.position.copy(position);

    // 宝箱主体
    const boxGeo = new THREE.BoxGeometry(size, size * 0.7, size * 0.7);
    const boxMat = new THREE.MeshStandardMaterial({
      color: color,
      roughness: 0.3,
      metalness: 0.6,
      emissive: glowColor,
      emissiveIntensity: 0.4
    });
    const box = new THREE.Mesh(boxGeo, boxMat);
    box.position.y = size * 0.35;
    group.add(box);

    // 宝箱盖
    const lidGeo = new THREE.BoxGeometry(size * 0.9, size * 0.15, size * 0.6);
    const lid = new THREE.Mesh(lidGeo, boxMat);
    lid.position.set(0, size * 0.7, 0);
    group.add(lid);

    // 发光环
    const ringGeo = new THREE.TorusGeometry(size * 0.8, 0.05, 16, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: glowColor, transparent: true, opacity: 0.6 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.05;
    group.add(ring);

    // 粒子光点
    const particleGeo = new THREE.BufferGeometry();
    const particleCount = 20;
    const positions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * size * 2;
      positions[i * 3 + 1] = Math.random() * size * 3;
      positions[i * 3 + 2] = (Math.random() - 0.5) * size * 2;
    }
    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: glowColor,
      size: 0.08,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    group.add(particles);

    group.userData.isTreasure = true;
    group.userData.ring = ring;
    group.userData.particles = particles;

    this.App.scene.add(group);
    this.generatedObjects.push(group);
    return group;
  }

  /**
   * 生成提示标记
   * @param {THREE.Vector3} position
   * @param {string} text - 提示文字
   */
  generateClueMarker(position, text = '?') {
    const THREE = this.THREE;
    const group = new THREE.Group();
    group.position.copy(position);

    // 旋转问号标记
    const canvas = this._createTextCanvas(text, 64, '#00e5ff');
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(1, 1, 1);
    sprite.position.y = 1.5;
    group.add(sprite);

    // 地面光圈
    const ringGeo = new THREE.RingGeometry(0.5, 0.6, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.5,
      depthWrite: false
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.02;
    group.add(ring);

    group.userData.isClue = true;
    group.userData.sprite = sprite;

    this.App.scene.add(group);
    this.generatedObjects.push(group);
    return group;
  }

  /**
   * 生成收集物（金币/星星）
   * @param {THREE.Vector3} position
   * @param {Object} options
   */
  generateCollectible(position, options = {}) {
    const THREE = this.THREE;
    const { color = 0xffdd44, size = 0.2, type = 'star' } = options;

    let geometry;
    if (type === 'star') {
      const starShape = new THREE.Shape();
      const outerR = size;
      const innerR = size * 0.4;
      const points = 5;
      for (let i = 0; i < points * 2; i++) {
        const r = i % 2 === 0 ? outerR : innerR;
        const angle = (i * Math.PI) / points - Math.PI / 2;
        const x = Math.cos(angle) * r;
        const y = Math.sin(angle) * r;
        if (i === 0) starShape.moveTo(x, y);
        else starShape.lineTo(x, y);
      }
      starShape.closePath();
      geometry = new THREE.ExtrudeGeometry(starShape, { depth: size * 0.3, bevelEnabled: true, bevelThickness: 0.02 });
    } else {
      geometry = new THREE.SphereGeometry(size, 16, 16);
    }

    const material = new THREE.MeshStandardMaterial({
      color: color,
      roughness: 0.2,
      metalness: 0.5,
      emissive: color,
      emissiveIntensity: 0.5
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    mesh.userData.isCollectible = true;

    this.App.scene.add(mesh);
    this.generatedObjects.push(mesh);
    return mesh;
  }

  /** 清理所有生成的场景对象 */
  cleanup() {
    for (const obj of this.generatedObjects) {
      if (obj.parent) obj.parent.remove(obj);
      obj.traverse(child => {
        if (child.geometry && child.geometry !== obj.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(m => { if (m.map) m.map.dispose(); m.dispose(); });
          } else {
            if (child.material.map) child.material.map.dispose();
            child.material.dispose();
          }
        }
      });
    }
    this.generatedObjects = [];
    this._restoreLighting();
  }

  // ==================== 内部方法 ====================

  _saveLighting() {
    const scene = this.App.scene;
    this.sceneLighting = { ambient: null, dirLights: [], pointLights: [] };
    scene.traverse(child => {
      if (child.isAmbientLight) this.sceneLighting.ambient = child;
      else if (child.isDirectionalLight) this.sceneLighting.dirLights.push(child);
      else if (child.isPointLight) this.sceneLighting.pointLights.push(child);
    });
    if (this.sceneLighting.ambient) {
      this._savedAmbientIntensity = this.sceneLighting.ambient.intensity;
    }
  }

  _restoreLighting() {
    if (this.sceneLighting && this.sceneLighting.ambient && this._savedAmbientIntensity !== null) {
      this.sceneLighting.ambient.intensity = this._savedAmbientIntensity;
    }
  }

  _applyGameLighting(colorScheme) {
    const schemes = {
      warm: { ambient: 0x332211, ambientIntensity: 0.6, fog: 0x221100, fogDensity: 0.02 },
      cool: { ambient: 0x112233, ambientIntensity: 0.5, fog: 0x001122, fogDensity: 0.015 },
      mystical: { ambient: 0x111133, ambientIntensity: 0.5, fog: 0x0a0a22, fogDensity: 0.018 },
    };
    const s = schemes[colorScheme] || schemes.mystical;
    if (this.sceneLighting && this.sceneLighting.ambient) {
      this.sceneLighting.ambient.intensity = s.ambientIntensity;
    }
    const scene = this.App.scene;
    if (scene.fog) {
      scene.fog.color.set(s.fog);
      scene.fog.density = s.fogDensity;
    }
  }

  _generateGround(size, style, colorScheme) {
    const THREE = this.THREE;
    const colors = { warm: 0x3a2a1a, cool: 0x1a2a3a, mystical: 0x1a1a2e };
    const groundGeo = new THREE.PlaneGeometry(size, size);
    const groundMat = new THREE.MeshStandardMaterial({
      color: colors[colorScheme] || 0x1a1a2e,
      roughness: 0.8,
      side: THREE.DoubleSide
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    ground.userData.isGround = true;
    this.App.scene.add(ground);
    this.generatedObjects.push(ground);

    // 网格线（装饰）
    if (style === 'maze' || style === 'dungeon') {
      const gridHelper = new THREE.PolarGridHelper(size / 2, Math.floor(size / 2), Math.floor(size / 4), 64, 0x333366, 0x222244);
      gridHelper.position.y = 0.005;
      this.App.scene.add(gridHelper);
      this.generatedObjects.push(gridHelper);
    }
  }

  _generateParticles(style, colorScheme, size) {
    const THREE = this.THREE;
    const colors = { warm: [0xff8844, 0xffaa66, 0xffcc88], cool: [0x4488ff, 0x66aaff, 0x88ccff], mystical: [0x8844ff, 0xaa66ff, 0xcc88ff] };
    const palette = colors[colorScheme] || colors.mystical;
    const count = Math.floor(size * 15);

    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colorArr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * size;
      positions[i * 3 + 1] = Math.random() * 5 + 0.5;
      positions[i * 3 + 2] = (Math.random() - 0.5) * size;
      const c = new THREE.Color(palette[Math.floor(Math.random() * palette.length)]);
      colorArr[i * 3] = c.r;
      colorArr[i * 3 + 1] = c.g;
      colorArr[i * 3 + 2] = c.b;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colorArr, 3));

    const mat = new THREE.PointsMaterial({
      size: 0.06,
      vertexColors: true,
      transparent: true,
      opacity: 0.5,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const particles = new THREE.Points(geo, mat);
    particles.userData.isParticles = true;
    this.App.scene.add(particles);
    this.generatedObjects.push(particles);
  }

  _generateBoundary(size, style, colorScheme) {
    // 半透明边界，防止角色走出去太远
    const THREE = this.THREE;
    const bSize = size * 1.05;
    const colors = { warm: 0xff6622, cool: 0x2266ff, mystical: 0x6622ff };

    // 用Line画出边界框
    const halfSize = bSize / 2;
    const points = [
      new THREE.Vector3(-halfSize, 0.05, -halfSize),
      new THREE.Vector3(halfSize, 0.05, -halfSize),
      new THREE.Vector3(halfSize, 0.05, halfSize),
      new THREE.Vector3(-halfSize, 0.05, halfSize),
      new THREE.Vector3(-halfSize, 0.05, -halfSize),
    ];
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const lineMat = new THREE.LineBasicMaterial({
      color: colors[colorScheme] || 0x6622ff,
      transparent: true,
      opacity: 0.3,
      depthWrite: false
    });
    const line = new THREE.Line(lineGeo, lineMat);
    this.App.scene.add(line);
    this.generatedObjects.push(line);
  }

  _createTextCanvas(text, size, color) {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = color || '#ffffff';
    ctx.font = `bold ${size * 0.7}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, size / 2, size / 2);
    return canvas;
  }
}

export default GameSceneGenerator;
