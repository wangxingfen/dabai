import bpy
import os

print("BLENDER_VERSION:", bpy.app.version_string)

# check addons
for mod in ("mmd_tools", "vrm"):
    try:
        bpy.ops.preferences.addon_enable(module=mod)
        print("ADDON_OK:", mod)
    except Exception as e:
        print("ADDON_FAIL:", mod, repr(e))

print("MMD_TOOLS_AVAILABLE:", hasattr(bpy.ops.mmd_tools, "import_model"))
print("VRM_EXPORT_AVAILABLE:", hasattr(bpy.ops.export_scene, "vrm"))
print("VRM_TPOSE_AVAILABLE:", hasattr(bpy.ops.vrm, "make_estimated_humanoid_t_pose"))
print("VRM_AUTO_HUMANOID:", hasattr(bpy.ops.vrm, "assign_vrm1_humanoid_human_bones_automatically"))

# verify mmd import operator props
try:
    import inspect
    print("IMPORT_MODEL_OPS:", [o.identifier for o in bpy.ops.mmd_tools.import_model.get_rna_type().properties])
except Exception as e:
    print("IMPORT_PROPS_EXC:", repr(e))