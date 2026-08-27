/* ============================================================
 * 观察空间规格 —— 声明式观察定义 + 通用归一化编码器（P0-3）
 *
 * 目标：消灭各游戏私有编码器里的临时归一化写法（如 Math.min(1, x/5)），
 * 观察特征一律以规格声明（scale/offset），由本模块统一归一化，
 * 使跨游戏观察可比较、可迁移、可复现。
 *
 * 规格格式（BaseGame.getObservationSpec() 返回值）：
 *   [
 *     { name: 'player_hp', kind: 'scalar', scale: 100, offset: 0 },
 *     { name: 'player',    kind: 'vector', dim: 22, scale: 1, offset: 0 },
 *     { name: 'vision',    kind: 'grid',   shape: [15, 6], scale: 1, offset: 0 },
 *   ]
 *
 * kind:
 *   'scalar' — 单个标量
 *   'vector' — 一组同量纲标量（如玩家状态段），raw[name] 为数组
 *   'grid'   — 二维网格（视觉扫描），raw[name] 为行主序数组，展平归一化
 *
 * 编码结果：Float64Array，长度 = 标量数 + 向量维数 + 网格元素数。
 * ============================================================ */

/** 单个标量归一化： (value + offset) / scale，裁剪到 [0,1]（scale<=0 时原样返回） */
export function normalizeScalar(value, scale = 1, offset = 0) {
  if (!scale || scale <= 0) return Number(value) || 0;
  const v = (Number(value) + (offset || 0)) / scale;
  return Math.max(0, Math.min(1, v));
}

/**
 * 按规格把原始观察编码为归一化 Float64Array
 * @param {Array} spec - getObservationSpec() 返回的特征规格
 * @param {Object|Array|Float64Array} raw - 原始观察
 *   - 标量特征：raw[name] 或按声明顺序传入数组
 *   - 网格特征：raw[name] 为数组/TypedArray，按行主序展平
 * @returns {Float64Array} 归一化观察向量
 */
export function encodeObservation(spec, raw) {
  if (!spec || !spec.length) return new Float64Array(0);
  const parts = [];

  for (const f of spec) {
    const value = Array.isArray(raw)
      ? undefined
      : (raw ? raw[f.name] : undefined);

    if (f.kind === 'grid') {
      // 网格特征（视觉扫描、地图切片等）：展平并逐元素归一化
      const rows = f.shape ? f.shape[0] : 0;
      const cols = f.shape ? (f.shape[1] || 1) : 1;
      const flat = value || [];
      for (let i = 0; i < rows * cols; i++) {
        parts.push(normalizeScalar(flat[i], f.scale, f.offset));
      }
    } else if (f.kind === 'vector') {
      // 向量特征（一组同量纲标量）：按 dim 展平
      const vec = value || [];
      const dim = f.dim || 0;
      for (let i = 0; i < dim; i++) {
        parts.push(normalizeScalar(vec[i], f.scale, f.offset));
      }
    } else {
      // 标量特征：支持 raw 对象取值或按声明顺序的数组
      let v;
      if (Array.isArray(raw)) {
        v = parts.length < raw.length ? raw[parts.length] : 0;
      } else {
        v = value !== undefined ? value : 0;
      }
      parts.push(normalizeScalar(v, f.scale, f.offset));
    }
  }

  return Float64Array.from(parts);
}

/** 规格描述文本（供调试/日志输出） */
export function describeObservationSpec(spec) {
  if (!spec) return 'null';
  return spec.map(f =>
    f.kind === 'grid'
      ? `${f.name}[${f.shape[0]}x${(f.shape[1] || 1)}]`
      : `${f.name}(scale=${f.scale},offset=${f.offset || 0})`
  ).join(', ');
}

/** 计算规格对应的观察向量维度 */
export function observationDim(spec) {
  if (!spec) return 0;
  return spec.reduce((sum, f) => {
    if (f.kind === 'grid' && f.shape) {
      return sum + f.shape[0] * (f.shape[1] || 1);
    }
    if (f.kind === 'vector') {
      return sum + (f.dim || 0);
    }
    return sum + 1;
  }, 0);
}

export default { normalizeScalar, encodeObservation, describeObservationSpec, observationDim };