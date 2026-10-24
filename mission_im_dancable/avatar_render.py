"""Blender-only offline renderer called by avatar.render_choreography.

Executed inside Blender's bundled Python; never imported by the game loop.
"""

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def render(target_file: Path, output_dir: Path) -> None:
    data = np.load(target_file, allow_pickle=False)
    targets, roots = data["targets"], data["roots"]
    scene = bpy.context.scene
    rig = bpy.data.objects["Armature"]
    scene.camera = bpy.data.objects["Cam ThreeQuarter"]
    scene.camera.data.lens = 78
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = True
    scene.render.resolution_x = 320
    scene.render.resolution_y = 400
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    for obj in bpy.data.collections["Presentation"].objects:
        if obj.type == "MESH":
            obj.hide_render = True
    if rig.animation_data:
        rig.animation_data_clear()
    bone_names = (
        "Spine2", "Neck",
        "LeftArm", "LeftForeArm", "RightArm", "RightForeArm",
        "LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg",
    )
    inverse = rig.matrix_world.inverted().to_3x3()
    for i, (pose, root) in enumerate(zip(targets, roots)):
        scene.frame_set(i + 1)
        for bone in rig.pose.bones:
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion.identity()
            bone.location.zero()
        rig.pose.bones["mixamorig:Hips"].location = (float(root[0]), 0, float(root[1]))
        bpy.context.view_layer.update()
        for name, direction in zip(bone_names, pose):
            if not np.isfinite(direction).all():
                continue
            bone = rig.pose.bones["mixamorig:" + name]
            current = (bone.tail - bone.head).normalized()
            desired = (inverse @ Vector(tuple(float(x) for x in direction))).normalized()
            rotation = current.rotation_difference(desired).to_matrix().to_4x4()
            origin = Matrix.Translation(bone.head)
            bone.matrix = origin @ rotation @ origin.inverted() @ bone.matrix
            bpy.context.view_layer.update()
        scene.render.filepath = str(output_dir / f"frame_{i + 1:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"AMY_RENDER {i + 1}/{len(targets)}", flush=True)


if __name__ == "__main__":
    index = sys.argv.index("--")
    render(Path(sys.argv[index + 1]), Path(sys.argv[index + 2]))
