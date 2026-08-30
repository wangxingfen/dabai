import sys
idx = sys.argv.index('--') if '--' in sys.argv else -1
if idx < 0:
    print('badargs'); sys.exit(2)
inp = sys.argv[idx + 1]
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=inp)
arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
arm = arms[0] if arms else None
ad = arm.animation_data if arm else None
act = ad.action if ad else None
n_fcurves = len(act.fcurves) if act else 0
n_keys = sum(len(fc.keyframe_points) for fc in act.fcurves) if act else 0
fr = act.frame_range if act else None
print('bones', len(arm.data.bones) if arm else 0,
      'meshes', len([o for o in bpy.data.objects if o.type == 'MESH']),
      'action', act.name if act else 'NONE',
      'frames', [int(fr[0]), int(fr[1])] if fr else None,
      'fcurves', n_fcurves, 'keys', n_keys)