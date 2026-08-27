/* ============================================================
 * 动作空间规格 —— 声明式动作表 + 统一解析器（P0-3）
 *
 * 目标：消灭各游戏 switch-case 硬编码动作分发。
 * 动作表由 BaseGame.getActionSpec() 声明，本模块提供：
 * - resolveAction：按索引解析动作定义
 * - describeActions：动作表描述（调试）
 * - guardActions：用安全掩码过滤当前可用动作
 *
 * 动作定义格式：
 *   { id: 0, name: 'move_left', semantics: 'primitive', executable: true }
 *   { id: 7, name: 'team_fight', semantics: 'semantic',  executable: true }
 * semantics: 'primitive'（底层原语：方向/跳跃）| 'semantic'（高层语义动作）
 * ============================================================ */

/** 按动作索引解析动作定义；越界返回 null */
export function resolveAction(spec, actionId) {
  if (!spec || !Array.isArray(spec)) return null;
  return spec[actionId] || null;
}

/** 按动作名查找动作定义；未找到返回 null */
export function findActionByName(spec, name) {
  if (!spec || !Array.isArray(spec)) return null;
  return spec.find(a => a.name === name) || null;
}

/**
 * 应用安全掩码：由游戏规则层（如血量过低强制撤退）给出的允许动作索引集合，
 * 与动作表取交集，返回合法动作索引数组。
 * @param {Array} spec - getActionSpec()
 * @param {Array<number>|null} mask - 允许的动作索引；null 表示全部允许
 * @returns {Array<number>}
 */
export function maskActions(spec, mask) {
  if (!spec || !spec.length) return [];
  if (!mask) return spec.map((_, i) => i);
  const valid = new Set(mask.filter(i => i >= 0 && i < spec.length));
  return spec.map((_, i) => i).filter(i => valid.has(i));
}

/** 动作表描述（调试/日志） */
export function describeActions(spec) {
  if (!spec || !spec.length) return '[]';
  return spec.map(a => `${a.id}:${a.name}(${a.semantics || 'primitive'})`).join(', ');
}

/** 动作数 */
export function actionCount(spec) {
  return spec && Array.isArray(spec) ? spec.length : 0;
}

export default { resolveAction, findActionByName, maskActions, describeActions, actionCount };