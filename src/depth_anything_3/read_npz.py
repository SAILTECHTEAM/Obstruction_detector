import numpy as np

npz_path = r"workspace\input_images\session_20260818_165022_741771\predictions.npz"

with np.load(npz_path, allow_pickle=False) as data:
    print("包含字段：", data.files)

    depth = data["depths"]
    print("深度形状：", depth.shape)
    print("深度类型：", depth.dtype)

    if "conf" in data.files:
        confidence = data["conf"]
        print("置信度形状：", confidence.shape)

    if "intrinsics" in data.files:
        intrinsics = data["intrinsics"]
        print("第一张图片的相机内参：\n", intrinsics[0])

    if "extrinsics" in data.files:
        extrinsics = data["extrinsics"]
        print("第一张图片的相机外参：\n", extrinsics[0])

# 第一张图片的深度图
first_depth = depth[0]

valid_mask = np.isfinite(first_depth) & (first_depth > 0)
valid_depth = first_depth[valid_mask]

print("有效像素数：", valid_depth.size)
print("最小深度：", valid_depth.min())
print("最大深度：", valid_depth.max())
print("中位深度：", np.median(valid_depth))