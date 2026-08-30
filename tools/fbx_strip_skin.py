"""Blender 无界面脚本：剥离 FBX 网格几何，仅保留骨架(armature)+动画。
用法: blender --background --factory-startup --python fbx_strip_skin.py -- <输入.fbx> <输出.fbx>
"""
import sys
import os

idx = sys.argv.index('--') if '--' in sys.argv else -1
if idx < 0 or len(sys.argv) < idx + 3:
    print('usage: blender --background --python fbx_strip_skin.py -- IN.fbx OUT.fbx')
    sys.exit(2)
inp, out = sys.argv[idx + 1], sys.argv[idx + 2]

import bpy  # noqa: E402


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=inp)
    except Exception as e:
        print('IMPORT_ERR', e)
        sys.exit(3)

    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()

    bpy.ops.object.select_all(action='DESELECT')
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if not arms:
        print('NO_ARMATURE')
        sys.exit(4)
    arm = arms[0]
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)

    maxf = 1
    for a in bpy.data.actions:
        fr = getattr(a, 'frame_range', None)
        if fr:
            maxf = max(maxf, int(fr[1]))
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = maxf

    try:
        bpy.ops.export_scene.fbx(filepath=out, use_selection=True, add_leaf_bones=False)
    except Exception as e:
        print('EXPORT_ERR', e)
        sys.exit(5)
    print('OK', os.path.getsize(out), 'bones', len(arm.data.bones), 'actions', len(bpy.data.actions))


main()