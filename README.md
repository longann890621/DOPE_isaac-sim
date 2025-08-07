# DOPE_isaac-sim

# 🧠 DOPE：使用 Isaac Sim 建立的深度物件姿態估計專案（6D Pose）

本專案完整實作了 6D 物件姿態估計流程，從合成資料生成、模型訓練到推論，皆參考並結合以下官方資源所製作：

- NVIDIA Isaac Sim 教學文件：  
  https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_pose_estimation.html  
- DOPE 原始碼 (Deep Object Pose Estimation)：  
  https://github.com/NVlabs/Deep_Object_Pose  

---

## 📁 專案架構說明

這個 repo 包含以下三大模組：

### 🔧 1. 合成資料生成（`pose_generation_formouse.py`）

使用 Isaac Sim 搭配 Replicator API 來產生 DOPE 格式的標註資料集。  
你可以指定任意 `.usd` 格式的 3D 模型（例如滑鼠模型），並產生下列輸出：
- RGB 影像（.png）
- 姿態標註（.json，包括 `location`, `quaternion_xyzw`, `projected_cuboid`）
- 附加除錯圖（可選）

使用方式範例：
```bash
./python.sh standalone_examples/replicator/pose_generation/pose_generation_formouse.py --num_mesh 0 --num_dome 3

```
# 🏋️‍♂️ 模型訓練（train.py）

本專案也完整時做了訓練 DOPE 模型的主腳本。

- 使用 belief map 與 affinity map 進行監督訓練
- 可自訂 batch size、epoch 數與儲存間隔
- 支援 TensorboardX 監看 loss 曲線與結果

---

## 📁 使用說明

以下是使用合成資料來訓練 DOPE 模型的流程。每個模型會輸出 .pth 權重檔，可以用於後續推論。

### 🔧 1. 執行指令（`train.py`）

使用方式範例：
```bash
python3 train.py \
  --data ~/AI_Nvidia/Isaac_Sim/output_split/train \
  --object mouse \
  --epochs 500 \
  --batchsize 32 \
  --save_every 50 \
  --outf ~/AI_Nvidia/Isaac_Sim/output_split/training_output

```

# 🧪 模型推論（inference.py）

此流程會載入訓練完成的 DOPE 模型，對資料夾中的圖片進行推論，並輸出 .json 及對應視覺化圖片。

### 🔧 1. 執行指令（`inference.py`）

使用方式範例：
```bash
python3 inference.py \
  --data /output_split_mouse/test3\
  --weights /training_output/net_epoch_0400.pth \
  --config ../config/config_pose.yaml \
  --object mouse \
  --camera ../config/camera_info.yaml \
  --outf /test_predictions3/mouse \
  --debug 

```
🧾 推論結果輸出：
圖像加上 projected cuboid 的疊圖（.png）

推論結果 JSON（含 location, quaternion_xyzw, projected_cuboid）

可選擇輸出 belief map 以除錯

📌 常用參數說明：
--data：要進行推論的圖像資料夾

--weights：DOPE 訓練好的模型權重（.pth）

--object：要推論的物件類別名稱

--config：模型對應的設定檔

--camera：相機內參檔案（通常與合成資料使用的參數相同）

--outf：推論結果儲存路徑

--debug：是否輸出 belief map 與點圖（可協助除錯）

