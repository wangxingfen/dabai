"""
PMX/PMD → VRM1 转换脚本（由 appearance 技能 pmx_to_vrm 工具驱动，Blender 后台执行）

流程（v29 最终方案，实测验证：莉丽拉 手臂自然下垂、动作正常）：
1. 高手 setup（mmd_tools 导入 + VRM1 humanoid/表情/meta/MToon）
2. hips = 腰（骨盆，层级合法；默认 auto 会把 hips 填成 センター，腿位移被乘 0 → 腿僵）
3. 导出原始 VRM（rest 保持 MMD 原始姿态，humanoid 映射正确）
4. 由 pmx_impl.py 调用 vrm_rest_final.py 做 JSON 层后处理：
   - X 轴完整镜像（MMD 左臂 +X → VRM 左臂 -X，前端旋转方向匹配）
   - humanoid 骨骼清 I + 位置保持（前端驱动直接生效）
   - 非 humanoid 骨骼重定向（世界变换不变，无变形）
   - 重算 IBM（REST 渲染零变形）

注意：不要在 Blender 内做 X 镜像（scale.x=-1 apply 只翻根骨骼、网格覆盖不全，
会导致手臂竖直/网格错位）。镜像必须在 JSON 层做（节点+网格+索引完整镜像）。

用法：
    blender --background --python blender_pmx_to_vrm.py -- <pmx_path> <vrm_path>

说明：
    高手工具（run_pmx_to_vrm1_setup 等）位于本文件同目录（pmx_tools/），
    本技能自包含，不依赖外部仓库。
"""
import os
import sys

SKILL_TOOLS = os.path.dirname(os.path.abspath(__file__))

bpy = __import__("bpy")


def _load_tool(name):
    path = os.path.join(SKILL_TOOLS, name)
    ns = {"__file__": path}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    return ns


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 2:
        print("USAGE: blender --background --python blender_pmx_to_vrm.py -- <pmx_path> <vrm_path>")
        sys.exit(1)
    pmx_path = args[0]
    vrm_path = args[1]

    if not os.path.isfile(pmx_path):
        print(f"ERROR: PMX 文件不存在：{pmx_path}")
        sys.exit(1)

    vrm_dir = os.path.dirname(vrm_path)
    if vrm_dir and not os.path.isdir(vrm_dir):
        os.makedirs(vrm_dir, exist_ok=True)

    # ---------- 1. 清场 + 启用插件 ----------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="mmd_tools")
    bpy.ops.preferences.addon_enable(module="vrm")

    # ---------- 2. 高手完整 setup ----------
    run_ns = _load_tool("run_pmx_to_vrm1_setup.py")
    run_pmx_to_vrm1_setup = run_ns["run_pmx_to_vrm1_setup"]

    result = run_pmx_to_vrm1_setup(
        filepath=pmx_path,
        dry_run=False,
        scale=0.08,
    )
    print("SETUP_RESULT:", result)

    if not result.get("applied"):
        print("SETUP_FAILED:", result.get("error"))
        sys.exit(1)

    arm_name = result["armature_object_name"]
    arm = bpy.data.objects[arm_name]
    bpy.context.view_layer.objects.active = arm

    # ---------- 3. hips = 腰（骨盆，层级合法） ----------
    hb = arm.data.vrm_addon_extension.vrm1.humanoid.human_bones
    before_hips = hb.hips.node.bone_name
    print("HIPS_BEFORE:", before_hips)

    target_hips = None
    for b in arm.data.bones:
        bare = b.name.split(" (")[0].strip()
        if bare == "腰":
            target_hips = b.name
            break
    if target_hips is None:
        print("ERROR: 未找到「腰」骨骼")
        sys.exit(1)

    if before_hips != target_hips:
        hb.hips.node.bone_name = target_hips
        print("HIPS_FIXED:", before_hips, "->", target_hips)
    else:
        print("HIPS_OK:", target_hips)

    # ---------- 4. 导出原始 VRM（rest 保持 MMD 原始姿态，由 pmx_impl.py 调用 vrm_rest_final.py 后处理） ----------
    try:
        bpy.ops.export_scene.vrm(filepath=vrm_path)
        print("EXPORT_DONE:", vrm_path, "exists=", os.path.isfile(vrm_path))
    except Exception as e:
        print("EXPORT_EXC:", repr(e))

    if not os.path.isfile(vrm_path):
        print("EXPORT_FAILED: 文件未生成")
        sys.exit(1)

    print("CONVERT_OK:", vrm_path)


main()