/* ============================================================
 * 虚拟摇杆组件 —— 移动端方向控制
 *
 * 功能：
 * - 左下角半透明圆环摇杆
 * - 拖拽内部拇指控制移动方向
 * - 输出标准化的方向向量 (x, z)，对应WASD移动
 * - 支持自动隐藏（无操作时降低透明度）
 * ============================================================ */

export class VirtualJoystick {
  declare container: HTMLElement;
  declare onMove: any;
  declare onStart: any;
  declare onEnd: any;

  declare _size: number;            // 外圈直径 (px)
  declare _thumbSize: number;       // 拇指直径 (px)
  declare _margin: number;          // 距屏幕边距

  declare _active: boolean;
  declare _touchId: number | null;
  declare _center: { x: number, y: number };
  declare _thumbPos: { x: number, y: number };
  declare _maxRadius: number;
  declare _value: { x: number, z: number, isMoving: boolean };

  declare _outer: HTMLElement;
  declare _thumb: HTMLElement;

  constructor(options: any = {}) {
    this.container = options.container || document.body;
    this.onMove = options.onMove || null;       // 回调: ({ x, z, isMoving }) =>
    this.onStart = options.onStart || null;
    this.onEnd = options.onEnd || null;

    this._size = options.size || 140;            // 外圈直径 (px)
    this._thumbSize = options.thumbSize || 60;   // 拇指直径 (px)
    this._margin = options.margin || 30;         // 距屏幕边距

    this._active = false;
    this._touchId = null;
    this._center = { x: 0, y: 0 };
    this._thumbPos = { x: 0, y: 0 };
    this._maxRadius = (this._size - this._thumbSize) / 2;
    this._value = { x: 0, z: 0, isMoving: false };

    this._createDOM();
    this._bindEvents();
  }

  // ==================== DOM ====================

  _createDOM() {
    const size = this._size;
    const thumbSize = this._thumbSize;

    // 外圈
    this._outer = document.createElement('div');
    this._outer.className = 'virtual-joystick-outer';
    Object.assign(this._outer.style, {
      width: size + 'px',
      height: size + 'px',
      bottom: this._margin + 'px',
      left: this._margin + 'px',
    });

    // 拇指
    this._thumb = document.createElement('div');
    this._thumb.className = 'virtual-joystick-thumb';
    Object.assign(this._thumb.style, {
      width: thumbSize + 'px',
      height: thumbSize + 'px',
    });

    this._outer.appendChild(this._thumb);
    this.container.appendChild(this._outer);

    // 初始位置：圆心
    this._center.x = size / 2;
    this._center.y = size / 2;
    this._thumbPos.x = size / 2;
    this._thumbPos.y = size / 2;
    this._updateThumbStyle();
  }

  // ==================== 事件 ====================

  _bindEvents() {
    this._outer.addEventListener('touchstart', (e) => this._onTouchStart(e), { passive: false });
    this._outer.addEventListener('touchmove', (e) => this._onTouchMove(e), { passive: false });
    this._outer.addEventListener('touchend', (e) => this._onTouchEnd(e));
    this._outer.addEventListener('touchcancel', (e) => this._onTouchEnd(e));
  }

  _onTouchStart(e) {
    e.preventDefault();
    if (this._active) return;
    const touch = e.changedTouches[0];
    this._touchId = touch.identifier;
    this._active = true;
    this._outer.classList.add('active');
    this._updateThumbFromTouch(touch);
    if (this.onStart) this.onStart();
  }

  _onTouchMove(e) {
    e.preventDefault();
    if (!this._active) return;
    // 找到匹配的touch
    let touch = null;
    for (let i = 0; i < e.changedTouches.length; i++) {
      if (e.changedTouches[i].identifier === this._touchId) {
        touch = e.changedTouches[i];
        break;
      }
    }
    if (!touch) return;
    this._updateThumbFromTouch(touch);
  }

  _onTouchEnd(e) {
    if (!this._active) return;
    let found = false;
    for (let i = 0; i < e.changedTouches.length; i++) {
      if (e.changedTouches[i].identifier === this._touchId) {
        found = true;
        break;
      }
    }
    if (!found) return;

    this._active = false;
    this._touchId = null;
    this._outer.classList.remove('active');

    // 复位
    this._thumbPos.x = this._center.x;
    this._thumbPos.y = this._center.y;
    this._value = { x: 0, z: 0, isMoving: false };
    this._updateThumbStyle();

    if (this.onEnd) this.onEnd();
    if (this.onMove) this.onMove(this._value);
  }

  _updateThumbFromTouch(touch) {
    const rect = this._outer.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;

    let dx = touch.clientX - cx;
    let dy = touch.clientY - cy;
    const dist = Math.sqrt(dx * dx + dy * dy);

    // 限制在最大半径内
    if (dist > this._maxRadius) {
      dx = (dx / dist) * this._maxRadius;
      dy = (dy / dist) * this._maxRadius;
    }

    this._thumbPos.x = this._center.x + dx;
    this._thumbPos.y = this._center.y + dy;

    // 计算归一化值：拖动方向映射到游戏世界移动
    // 上推(dy<0) = 前进(Z+)，下推(dy>0) = 后退(Z-)
    // 左推(dx<0) = 左移(X-)，右推(dx>0) = 右移(X+)
    const normX = dx / this._maxRadius;
    const normZ = -dy / this._maxRadius;

    const isMoving = Math.abs(normX) > 0.05 || Math.abs(normZ) > 0.05;

    this._value = { x: normX, z: normZ, isMoving };

    this._updateThumbStyle();

    if (this.onMove) this.onMove(this._value);
  }

  _updateThumbStyle() {
    this._thumb.style.transform = `translate(${this._thumbPos.x - this._thumbSize / 2}px, ${this._thumbPos.y - this._thumbSize / 2}px)`;
  }

  // ==================== 公开方法 ====================

  /** 获取当前摇杆方向值 */
  getValue() {
    return this._value;
  }

  /** 是否正在被拖拽 */
  isActive() {
    return this._active;
  }

  /** 销毁 */
  destroy() {
    this._outer.removeEventListener('touchstart', this._onTouchStart);
    this._outer.removeEventListener('touchmove', this._onTouchMove);
    this._outer.removeEventListener('touchend', this._onTouchEnd);
    this._outer.removeEventListener('touchcancel', this._onTouchEnd);
    if (this._outer.parentNode) {
      this._outer.parentNode.removeChild(this._outer);
    }
  }
}

export default VirtualJoystick;