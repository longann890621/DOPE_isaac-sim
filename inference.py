#!/usr/bin/env python

import argparse
import cv2
import numpy as np
from PIL import Image
import os
import simplejson as json
import sys
import yaml

sys.path.append("../common/")
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from utils import loadimages_inference, loadweights, Draw


class DopeNode(object):
    """ROS node that listens to image topic, runs DOPE, and publishes DOPE results"""

    def __init__(
        self,
        config,   # config yaml loaded eg dict
        weight,   # path to weight file
        parallel, # was it trained using DDP
        class_name,
    ):
        self.input_is_rectified = config["input_is_rectified"]
        self.downscale_height = config["downscale_height"]

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config["thresh_angle"]
        self.config_detect.thresh_map = config["thresh_map"]
        self.config_detect.sigma = config["sigma"]
        self.config_detect.thresh_points = config["thresh_points"]

        # load network model, create PNP solver
        self.model = ModelData(
            name=class_name,
            net_path=weight,
            parallel=parallel
        )
        self.model.load_net_model()
        print("Model Loaded")

        try:
            self.draw_color = tuple(config["draw_colors"][class_name])
        except:
            self.draw_color = (0, 255, 0)

        self.dimension = tuple(config["dimensions"][class_name])
        self.class_id = config["class_ids"][class_name]

        self.pnp_solver = CuboidPNPSolver(
            class_name, cuboid3d=Cuboid3d(config["dimensions"][class_name])
        )
        self.class_name = class_name

        print("Ctrl-C to stop")

    def image_callback(
        self,
        img,
        camera_info,
        img_name,
        output_folder,
        weight,
        debug=False
    ):
        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            P = np.matrix(camera_info["projection_matrix"]["data"], dtype="float64").copy()
            P.resize((3, 4))
            camera_matrix = P[:, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            if isinstance(camera_info, dict):
                fx = camera_info.get("fx", 1066.778)
                fy = camera_info.get("fy", 1067.487)
                cx = camera_info.get("cx", 312.9869)
                cy = camera_info.get("cy", 241.3109)
               
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype="float64")
                if "D" in camera_info:
                    dist_coeffs = np.array(camera_info["D"], dtype="float64").reshape((-1, 1))
                else:
                    dist_coeffs = np.zeros((5, 1), dtype="float64")  # fallback
            else:
                raise ValueError("Unsupported camera_info format")
                

        # Downscale image if necessary
        #height, width, _ = img.shape
        #scaling_factor = float(self.downscale_height) / height
        #if scaling_factor < 1.0:
            #camera_matrix[:2] *= scaling_factor
            #img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))

        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)

        # dictionary for the final output
        dict_out = {"camera_data": {}, "objects": []}

        # Detect object
        print("[DEBUG] Running detect_object_in_image...")
        results, belief_imgs = ObjectDetector.detect_object_in_image(
            self.model.net, self.pnp_solver, img, self.config_detect,
            grid_belief_debug=debug,
        )
        print(f"[DEBUG] Detected result count = {len(results)}")

        # Publish pose and overlay cube on image
        for _, result in enumerate(results):
            print(f"[DEBUG] result = {result}")
            if result["location"] is None:
                continue

            loc = result["location"]
            ori = result["quaternion"]

            dict_out["objects"].append(
                {
                    "class": self.class_name,
                    "location": np.array(loc).tolist(),
                    "quaternion_xyzw": np.array(ori).tolist(),
                    "projected_cuboid": np.array(result["projected_points"]).tolist(),
                }
            )

            # Draw the cube
            if None not in result["projected_points"]:
                points2d = []
                for pair in result["projected_points"]:
                    points2d.append(tuple(pair))
                draw.draw_cube(points2d, self.draw_color)

        # create directory to save image if it does not exist
        img_name_base = img_name.split("/")[-1]
        img_stem = ".".join(img_name_base.split(".")[:-1])
        output_path = os.path.join(
        output_folder,
        weight.split("/")[-1].replace(".pth", ""),
        img_stem,
        )
        
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        im.save(os.path.join(output_path, img_name_base))
        if belief_imgs is not None:
            belief_map_path = os.path.join(output_path, "belief_maps.png")
            belief_imgs.save(belief_map_path)
            
            peak = None
            if len(results) > 0:
                for i_r, r in enumerate(results):
                    # 通常這裡會計算 peak = something...
                    peak = r['location']  # 假設這是你要用的值
                    # 你可以在這裡處理其他推論結果的邏輯
                    
            if results:
                import matplotlib.pyplot as plt
                belief_np = np.array(belief_imgs)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(belief_np)
                
                for result in results:
                    projected_points = result.get("projected_points", [])
        
                    if isinstance(projected_points, np.ndarray) and projected_points.ndim == 2:
                        for pt in projected_points:
                            if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2:
                                ax.plot(pt[0], pt[1], 'ro', markersize=4)
                            else:
                                print(f"[警告] 異常的 pt：{pt}")
                    else:
                        print(f"[警告] 無效的 projected_points 結構：{projected_points}")

                ax.set_title("Belief Map with Projected Points")
                ax.axis("off")

                peak_out_path = os.path.join(output_path, "belief_with_peaks.png")
                fig.savefig(peak_out_path, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

        json_path = os.path.join(
            output_path, ".".join(img_name_base.split(".")[:-1]) + ".json"
        )
        # save the json files
        with open(json_path, "w") as fp:
            json.dump(dict_out, fp, indent=2)
            
        # 回傳是否成功推論（有物件）
        return len(dict_out["objects"]) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outf",
        default="output",
        help="Where to store the output images and inference results.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="folder for data images to load.",
    )
    parser.add_argument(
        "--config",
        default="../config/config_pose.yaml",
        help="Path to inference config file",
    )
    parser.add_argument(
        "--camera",
        default="../config/camera_info.yaml",
        help="Path to camera info file",
    )
    parser.add_argument(
        "--weights",
        "--weight",
        "-w",
        required=True,
        help="Path to weights or folder containing weights. If path is to a folder, then script "
        "will run inference with all of the weights in the folder. This could take a while if "
        "the set of test images is large.",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help="Were the weights trained using DDP; if set to true, the names of later weights "
        " will be altered during load to match the model"
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        type=str,
        default=["png"],
        help="Extensions for images to use. Can have multiple entries separated by space. "
        "e.g. png jpg",
    )
    parser.add_argument(
        "--object",
        required=True,
        help="Name of class to run detections on.",
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Generates debugging information, including raw belief maps and annotation of "
        "the results"
    )

    opt = parser.parse_args()

    # Load configuration
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(opt.outf, exist_ok=True)

    # Load model weights
    weights = loadweights(opt.weights)
    if len(weights) < 1:
        print("❌ No weights found. Please check --weights flag.")
        exit()
    else:
        print(f"🎯 Found {len(weights)} weights.")

    # Load inference images
    imgs, imgsname = loadimages_inference(opt.data, extensions=opt.exts)
    # 修正順序問題：依照圖片編號排序
    imgsname_sorted = sorted(
        zip(imgs, imgsname),
        key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x[1]))))
    )
    imgs, imgsname = zip(*imgsname_sorted)
    imgs = list(imgs)
    imgsname = list(imgsname)
    
    if len(imgs) == 0 or len(imgsname) == 0:
        print("❌ No input images found. Please check --data and --exts flags.")
        exit()

    for w_i, weight in enumerate(weights):
        dope_node = DopeNode(config, weight, opt.parallel, opt.object)

        total_success = 0
        total_fail = 0
        success_images = []
        failed_images = []

        for i in range(len(imgs)):
            print(f"({w_i + 1}/{len(weights)}) frame {i + 1}/{len(imgs)}: {imgsname[i]}")
            img_name = imgsname[i]
            frame = cv2.imread(imgs[i])
            frame = frame[..., ::-1].copy()

            success = dope_node.image_callback(
                img=frame,
                camera_info=camera_info,
                img_name=img_name,
                output_folder=opt.outf,
                weight=weight,
                debug=opt.debug
            )

            if success:
                total_success += 1
                success_images.append(img_name)
            else:
                total_fail += 1
                failed_images.append(img_name)

        # --- 總結 ---
        print("\n==============================")
        print("🎯 推論總結")
        print(f"✅ 成功預測數量: {total_success}")
        print(f"❌ 失敗預測數量: {total_fail}")

        if success_images:
            print("\n✅ 成功圖片列表:")
            for name in success_images:
                print(f" + {name}")

        #if failed_images:
            #print("\n❌ 失敗圖片列表:")
            #for name in failed_images:
                #print(f" - {name}")
        print("==============================\n")


