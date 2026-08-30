"""
莉丽拉 PMX → VRM1 转换（修复版：不做 T-pose rest 标准化，保留 MMD 原始 rest）
原因（实测验证）：前端 applyVrmRestPose 在 normalized 空间对手臂设 rotation.z=±1.35，
期望 rest 手臂沿 +Y（骨骼局部 Y 指向子骨骼的自然骨架，如蔚蓝妖姬 VRM0）。
v19 的 T-pose rest 标准化把手臂转成 ±X 水平，叠加 1.35 → 手臂高举且随动作严重变形。
"""
import os
import sys

SKILL_TOOLS = r"D:\AI\dabai\_ref_miramocha\skills\mmd-pmx-to-vrm1\tools"

bpy = __import__("bpy")


def _load_tool(name):
    path = os.path.join(SKILL_TOOLS, name)
    ns = {"__file__": path}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    return ns


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 2:
        print("USAGE: blender --background --python pmx_to_vrm.py -- <pmx_path> <vrm_path>")
        sys.exit(1)
    pmx_path = args[0]
    vrm_path = args[1]

    if not os.path.isfile(pmx_path):
        print(f"ERROR: PMX 文件不存在：{pmx_path}")
        sys.exit(1)

    vrm_dir = os.path.dirname(vrm_path)
    if vrm_dir and not os.path.isdir(vrm_dir):
        os.makedirs(vrm_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="mmd_tools")
    bpy.ops.preferences.addon_enable(module="vrm")

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

    # hips = 腰（骨盆，层级合法）
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

    # 导出 VRM（保留 MMD 原始 rest，不做 T-pose rest 标准化）
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