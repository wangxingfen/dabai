"""
vrm_rest_final.py —— VRM JSON 后处理（v29 最终方案，实测验证：莉丽拉 手臂自然下垂、动作正常）

输入：Blender 导出的原始 VRM（MMD 原始 rest，骨骼局部旋转编码方向）
输出：标准 VRM（蔚蓝妖姬式：骨骼局部旋转 identity、手水平 bind、前端驱动直接生效）

三步：
1. **X 轴完整镜像**（照镜子式）：
   MMD 左臂在 +X，而 VRM/前端约定左臂在 -X；镜像后左臂到 -X，
   前端 applyVrmRestPose（左臂绕 Z +77°、右臂 -77°）才能正确把手转到自然下垂。
   镜像变换 M = diag(-1,1,1)：translation.x→-x、rotation→(x,-y,-z,w)、
   网格 POSITION/NORMAL/TANGENT x→-x、三角形索引翻转保持绕向。
2. **骨骼清 I + 重定向**：
   - humanoid 骨骼（含父链，不含 spring）：局部旋转 = identity，
     translation 平移补偿保持世界位置 → 前端每帧驱动直接生效（boneRot=I、P=I）
   - 非 humanoid 骨骼（twist 中间骨骼/手指/装饰/发饰）：局部 TRS 重定向
     （T'=R父原·T，R'=R父原·R）→ 世界变换不变，无变形
3. **重算 IBM = inverse(新世界矩阵)**，网格顶点不动（REST 渲染 = bind 原样，零变形）
"""
import struct
import json
import numpy as np
import sys

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


def read_glb(path):
    with open(path, 'rb') as f:
        data = f.read()
    offset = 12
    json_dict = None
    bin_chunk = b''
    while offset + 8 <= len(data):
        cl, ct = struct.unpack_from('<II', data, offset)
        offset += 8
        chunk = data[offset:offset + cl]
        offset += cl
        if ct == CHUNK_JSON:
            json_dict = json.loads(chunk.decode('utf-8'))
        elif ct == CHUNK_BIN:
            bin_chunk = chunk
    return json_dict, bin_chunk


def write_glb(path, gltf, bin_chunk):
    jb = json.dumps(gltf, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    jp = (4 - (len(jb) % 4)) % 4
    jb += b' ' * jp
    bp = (4 - (len(bin_chunk) % 4)) % 4
    bo = bin_chunk + b'\x00' * bp
    total = 12 + 8 + len(jb) + 8 + len(bo)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', GLB_MAGIC, 2, total))
        f.write(struct.pack('<II', len(jb), CHUNK_JSON) + jb)
        f.write(struct.pack('<II', len(bo), CHUNK_BIN) + bo)


def quat_to_mat(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def mat_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    return [qx/n, qy/n, qz/n, qw/n]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    gltf, bin_chunk = read_glb(src)
    nodes = gltf['nodes']
    parent_of = {}
    children_of = {}
    for i, n in enumerate(nodes):
        children_of[i] = n.get('children', [])
        for c in children_of[i]:
            parent_of[c] = i

    hb = gltf['extensions']['VRMC_vrm']['humanoid']['humanBones']
    spring_nodes = set()
    spring = gltf.get('extensions', {}).get('VRMC_springBone', {})
    for s in spring.get('springs', []):
        for j in s.get('joints', []):
            n = j.get('node')
            if n is not None:
                spring_nodes.add(n)

    humanoid_node_set = set(hb[s]['node'] for s in hb)
    norm_nodes = set()
    for idx in humanoid_node_set:
        i = idx
        while i is not None:
            norm_nodes.add(i)
            i = parent_of.get(i)
    norm_nodes -= spring_nodes

    # ---------- 1. X 轴完整镜像（节点 + 网格） ----------
    for n in nodes:
        t = n.get('translation')
        if t:
            n['translation'] = [-t[0], t[1], t[2]]
        r = n.get('rotation')
        if r:
            n['rotation'] = [r[0], -r[1], -r[2], r[3]]

    for mesh in gltf.get('meshes', []):
        for prim in mesh.get('primitives', []):
            attrs = prim.get('attributes', {})
            for attr_name in ('POSITION', 'NORMAL', 'TANGENT'):
                idx = attrs.get(attr_name)
                if idx is None:
                    continue
                acc = gltf['accessors'][idx]
                bv = gltf['bufferViews'][acc['bufferView']]
                off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
                ncomp = 3 if acc.get('type') == 'VEC3' else (2 if acc.get('type') == 'VEC2' else 4)
                count = acc['count']
                arr = np.frombuffer(bin_chunk, dtype=np.float32, count=count * ncomp, offset=off).copy()
                arr = arr.reshape(-1, ncomp)
                arr[:, 0] = -arr[:, 0]
                buf = bytearray(bin_chunk)
                newb = arr.astype(np.float32).tobytes()
                buf[off:off + len(newb)] = newb
                bin_chunk = bytes(buf)

            idx = prim.get('indices')
            if idx is None:
                continue
            acc = gltf['accessors'][idx]
            bv = gltf['bufferViews'][acc['bufferView']]
            off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
            count = acc['count']
            ct = acc['componentType']
            if ct == 5121:
                arr = np.frombuffer(bin_chunk, dtype=np.uint8, count=count, offset=off).copy()
            elif ct == 5123:
                arr = np.frombuffer(bin_chunk, dtype=np.uint16, count=count, offset=off).copy()
            elif ct == 5125:
                arr = np.frombuffer(bin_chunk, dtype=np.uint32, count=count, offset=off).copy()
            else:
                continue
            ntri = count // 3
            tri = arr.reshape(ntri, 3)
            tri[:, 1], tri[:, 2] = tri[:, 2].copy(), tri[:, 1].copy()
            buf = bytearray(bin_chunk)
            newb = tri.reshape(-1).tobytes()
            buf[off:off + len(newb)] = newb
            bin_chunk = bytes(buf)

    # ---------- 2. 清 I + 重定向 ----------
    order, visited = [], set()
    def dfs(i):
        if i in visited:
            return
        visited.add(i)
        p = parent_of.get(i)
        if p is not None:
            dfs(p)
        order.append(i)
    for i in range(len(nodes)):
        dfs(i)

    def local_trs(i):
        n = nodes[i]
        t = n.get('translation', [0, 0, 0])
        r = n.get('rotation', [0, 0, 0, 1])
        s = n.get('scale', [1, 1, 1])
        return np.array(t), quat_to_mat(r), np.array(s)

    world_rot_orig = {}
    world_pos_orig = {}
    for i in order:
        t, R, s = local_trs(i)
        p = parent_of.get(i)
        if p is None:
            world_rot_orig[i] = R
            world_pos_orig[i] = t
        else:
            world_rot_orig[i] = world_rot_orig[p] @ R
            world_pos_orig[i] = world_pos_orig[p] + world_rot_orig[p] @ t

    for i in order:
        n = nodes[i]
        t, R, s = local_trs(i)
        p = parent_of.get(i)
        if p is None:
            continue
        if i in norm_nodes:
            p_new_pos = world_pos_orig[p]
            new_t = world_pos_orig[i] - p_new_pos
            n['rotation'] = [0, 0, 0, 1]
            n['translation'] = [round(float(v), 6) for v in new_t]
        else:
            Rp = world_rot_orig[p]
            new_t = Rp @ t
            new_R = Rp @ R
            q = mat_to_quat(new_R)
            n['translation'] = [round(float(v), 6) for v in new_t]
            n['rotation'] = [round(float(v), 6) for v in q]

    # ---------- 3. 重算 IBM ----------
    world_new = {}
    for i in order:
        t, R, s = local_trs(i)
        p = parent_of.get(i)
        if p is None:
            world_new[i] = np.eye(4)
            world_new[i][:3, :3] = R
            world_new[i][:3, 3] = t
        else:
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = t
            world_new[i] = world_new[p] @ M

    buf = bytearray(bin_chunk)
    for skin in gltf.get('skins', []):
        acc = gltf['accessors'][skin['inverseBindMatrices']]
        bv = gltf['bufferViews'][acc['bufferView']]
        base = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        for k, j in enumerate(skin.get('joints', [])):
            m = world_new.get(j)
            if m is None:
                continue
            inv = np.linalg.inv(m)
            col = struct.pack('<16f', *[inv[c, r] for r in range(4) for c in range(4)])
            off = base + k * 64
            buf[off:off + 64] = col
    bin_chunk = bytes(buf)

    write_glb(dst, gltf, bin_chunk)
    print('VRM_FINAL_OK ->', dst)


if __name__ == '__main__':
    main()