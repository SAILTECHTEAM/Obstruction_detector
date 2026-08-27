# 基於 Depth Anything 3 的影片遮擋偵測

本專案基於 [Depth Anything 3（DA3）](https://github.com/ByteDance-Seed/Depth-Anything-3) 開發，用於判斷固定機位畫面中是否出現大面積的新遮擋物。系統將正常場景參考影像與影片畫格逐幀比較，並結合：

- **DA3Metric-Large**：估算具有真實尺度的單目深度；
- **YOLO 實例分割**：獨立處理畫面中的人員區域；
- **連通區域與面積判定**：過濾小範圍雜訊並輸出遮擋區域；
- **深度正規化面積**：依照物體距離動態調整面積門檻，降低透視造成的影響。

目前專案是離線影片分析程式，不是 Web 服務。輸入為一張正常參考影像與一段待偵測影片；輸出包含標註後的影片、逐幀偵測結果及摘要報告。

## 偵測邏輯

程式會對每個待分析畫格執行以下步驟：

1. 分別推論參考影像與目前畫格的 metric depth；
2. 計算「目前深度 - 參考深度」，深度明顯減少的像素視為新遮擋候選；
3. 使用形態學處理清除雜訊，並擷取連通區域；
4. 使用 YOLO 分割人員，人員區域不納入一般深度遮擋統計；
5. 人員覆蓋比例達到門檻時輸出 `PERSON_OCCLUSION`；
6. 非人員的深度變化區域達到面積門檻時輸出 `DEPTH_OCCLUSION`；
7. 否則輸出 `NO_LARGE_OCCLUSION`。

建議使用 `detect_video_occlusion_depth_ratio`。此模式會依照區域深度修正面積門檻，比固定像素面積更適合遠近尺度差異明顯的監控畫面。

## 專案結構

```text
depth_anythingv3/
├── assets/
│   ├── images/normal.png        # 範例正常參考影像
│   └── videos/input.mp4         # 範例輸入影片
├── models/
│   └── yolo11n-seg.pt           # 專案內附的 YOLO 人員分割權重
├── outputs/                     # 執行後自動產生，不提交至 Git
├── src/depth_anything_3/
│   ├── detect_depth_occlusion.py
│   ├── detect_video_occlusion.py
│   ├── detect_video_occlusion_depth_ratio.py
│   └── person_segmentation.py
├── pyproject.toml
└── uv.lock
```

## Linux 環境需求

以下指令以 Ubuntu/Debian 為例。

- 64 位元 Linux；
- Python **3.11**；
- 建議使用 NVIDIA GPU；
- NVIDIA 驅動程式需支援專案鎖定的 PyTorch CUDA 12.8 執行環境；
- 可連線至 PyPI 與 Hugging Face，或事先準備本機 DA3 模型目錄；
- 請預留足夠空間存放 Python 環境、DA3 權重與輸出影片。

CPU 也可執行：將 `--device` 與 `--yolo-device` 都設為 `cpu`，但 DA3 逐幀影片推論速度會很慢。

先安裝系統相依套件：

```bash
sudo apt update
sudo apt install -y git curl ffmpeg libgl1 libglib2.0-0
```

如使用 GPU，請先確認驅動程式可用：

```bash
nvidia-smi
```

## Linux 部署

### 1. 取得專案

```bash
git clone https://github.com/xin-2005/Obstruction_detector.git
cd Obstruction_detector
```

### 2. 安裝 uv

專案已提交 `uv.lock`，建議透過 uv 建立並同步一致的 Python 3.11 環境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

若目前終端機仍找不到 `uv`，請重新登入，或執行：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 3. 安裝專案相依套件

```bash
uv python install 3.11
uv sync
```

目前鎖定環境包含 PyTorch、TorchVision、xFormers、OpenCV、Ultralytics、Matplotlib 與 DA3 所需套件。PyTorch、TorchVision 和 xFormers 使用 `pyproject.toml` 中設定的 CUDA 12.8 軟體來源。

驗證安裝與 GPU 狀態：

```bash
uv run python -c "import torch; import cv2; import ultralytics; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 4. 準備模型

YOLO 權重已位於：

```text
models/yolo11n-seg.pt
```

DA3 預設使用 Hugging Face 模型 `depth-anything/DA3METRIC-LARGE`。第一次執行時會自動下載並快取權重。若伺服器無法在執行時連線網路，可預先下載至專案目錄：

```bash
uv run hf download depth-anything/DA3METRIC-LARGE \
  --local-dir models/DA3METRIC-LARGE
```

之後將執行參數改為：

```bash
--model-dir models/DA3METRIC-LARGE
```

## 輸入資料需求

正常參考影像對偵測效果非常重要：

- 參考影像應來自與影片相同的固定機位；
- 參考影像中不應包含需要偵測的新遮擋物；
- 參考影像與影片應具有相同長寬比，建議解析度也保持一致；
- 相機移動、明顯晃動、變焦或場景結構變化會產生額外深度差異；
- 光線變化通常比機位變化的影響小，但過暗、反光或透明區域仍可能影響深度估算。

## 執行影片遮擋偵測

### 建議模式：依深度正規化面積

可直接使用專案內的範例資料：

```bash
uv run python -m depth_anything_3.detect_video_occlusion_depth_ratio \
  assets/images/normal.png \
  assets/videos/input.mp4 \
  --yolo-model models/yolo11n-seg.pt \
  --model-dir depth-anything/DA3METRIC-LARGE \
  --device cuda \
  --yolo-device cuda:0 \
  --area-mode depth
```

若使用預先下載的本機 DA3 權重，請將 `--model-dir` 改為 `models/DA3METRIC-LARGE`。

### 固定面積門檻模式

若鏡頭中目標距離的變化很小，可使用較簡單的固定面積模式：

```bash
uv run python -m depth_anything_3.detect_video_occlusion \
  assets/images/normal.png \
  assets/videos/input.mp4 \
  --yolo-model models/yolo11n-seg.pt \
  --model-dir depth-anything/DA3METRIC-LARGE \
  --device cuda \
  --yolo-device cuda:0 \
  --depth-threshold 0.5 \
  --min-area-ratio 0.03
```

### 快速測試

部署後建議先處理少量畫格，確認模型、顯示卡和影片編解碼皆正常：

```bash
uv run python -m depth_anything_3.detect_video_occlusion_depth_ratio \
  assets/images/normal.png \
  assets/videos/input.mp4 \
  --yolo-model models/yolo11n-seg.pt \
  --max-frames 10
```

## 圖片比對模式

除了影片之外，也可直接比較正常圖片與遮擋圖片：

```bash
uv run python -m depth_anything_3.detect_depth_occlusion \
  assets/images/normal.png \
  assets/images/2.png \
  --yolo-model models/yolo11n-seg.pt \
  --model-dir depth-anything/DA3METRIC-LARGE \
  --device cuda \
  --yolo-device cuda:0
```

若已有兩張圖片對應的 DA3 `results.npz`，可透過 `--normal-depth` 與 `--occluded-depth` 重複使用深度結果，略過 DA3 推論。兩個參數必須同時提供。

## 主要參數

### 共用參數

| 參數 | 預設值 | 說明 |
| --- | ---: | --- |
| `--model-dir` | `depth-anything/DA3METRIC-LARGE` | Hugging Face 模型名稱或本機模型目錄 |
| `--device` | `cuda` | DA3 推論裝置，例如 `cuda`、`cuda:0` 或 `cpu` |
| `--yolo-model` | 影片模式必填 | YOLO segmentation 權重路徑 |
| `--yolo-device` | 自動選擇 | YOLO 裝置，例如 `cuda:0` 或 `cpu` |
| `--yolo-confidence` | `0.25` | YOLO 偵測信心門檻 |
| `--yolo-image-size` | `640` | YOLO 推論影像尺寸 |
| `--process-res` | `504` | DA3 處理解析度上限；降低可減少顯示記憶體用量 |
| `--depth-threshold` | `0.5` | 判定為遮擋候選所需的最小深度減少值；使用預設 metric 模型時可近似以公尺理解 |
| `--person-alert-ratio` | `0.3333` | 人員遮罩占深度圖比例達到此值時發出警報 |
| `--person-dilate` | `7` | 清除深度變化時，人員遮罩向外擴張的像素半徑 |
| `--frame-step` | `1` | 每隔多少畫格分析一次；略過的畫格會原樣寫入輸出影片 |
| `--max-frames` | `0` | 最多讀取多少畫格，`0` 代表處理完整影片 |
| `--save-alert-frames` | 關閉 | 儲存警報畫格、遮擋遮罩及忽略的人員遮罩 |
| `--output-root` | 依模式而定 | 指定每次執行結果的根目錄 |

### 深度正規化面積參數

| 參數 | 預設值 | 說明 |
| --- | ---: | --- |
| `--area-mode` | `depth` | `depth` 使用深度正規化，`fixed` 使用固定面積 |
| `--reference-depth` | `3.0` | 校準面積門檻時的參考距離 |
| `--reference-area-ratio` | `0.03` | 物體位於參考距離時所需的最小畫面面積比例 |
| `--dynamic-min-area-ratio` | `0.003` | 動態面積門檻下限 |
| `--dynamic-max-area-ratio` | `0.30` | 動態面積門檻上限 |
| `--component-min-area-ratio` | `0.0005` | 計算區域深度前捨棄的微小連通區域門檻 |
| `--region-depth-erode` | `2` | 計算區域中位深度前的遮罩侵蝕半徑 |

### 固定面積參數

| 參數 | 預設值 | 說明 |
| --- | ---: | --- |
| `--min-area-ratio` | `0.03` | 連通區域至少占完整深度圖的比例 |
| `--blur-size` | `5` | 深度差高斯模糊核心大小，必須為正奇數 |
| `--open-size` | `3` | 形態學開運算核心大小，必須為正奇數 |
| `--close-size` | `7` | 形態學閉運算核心大小，必須為正奇數 |

查看所有參數：

```bash
uv run python -m depth_anything_3.detect_video_occlusion_depth_ratio --help
uv run python -m depth_anything_3.detect_video_occlusion --help
uv run python -m depth_anything_3.detect_depth_occlusion --help
```

## 輸出說明

深度正規化影片模式預設寫入：

```text
outputs/video_occlusion_depth_ratio_runs/<時間戳記_影片名稱>/
├── annotated.mp4             # 含判定文字、紅色遮罩及外框的結果影片
├── frame_results.jsonl       # 每個已分析畫格一行 JSON 結果
├── summary.json              # 本次影片分析摘要
└── alert_frames/             # 僅使用 --save-alert-frames 時產生
    ├── frame_XXXXXXXX.png
    ├── frame_XXXXXXXX_mask.png
    └── frame_XXXXXXXX_ignored_person.png
```

固定面積影片模式預設寫入 `outputs/video_occlusion_runs/`。

圖片比對模式預設寫入：

```text
outputs/occlusion_detection/
├── report.json
├── overview.png
├── occlusion_overlay.png
├── occlusion_mask.png
├── ignored_person_mask.png
└── depth_difference.npy
```

`frame_results.jsonl` 與 `report.json` 中的主要結論如下：

- `PERSON_OCCLUSION`：人員覆蓋比例達到警報門檻；
- `DEPTH_OCCLUSION`：存在符合門檻的非人員深度遮擋區域；
- `NO_LARGE_OCCLUSION`：未偵測到大面積遮擋。

## 門檻調整建議

- 誤報較多：增加 `--depth-threshold`、`--reference-area-ratio` 或 `--min-area-ratio`；
- 小型遮擋漏報：降低上述門檻，但會增加對雜訊及背景變化的敏感度；
- 遠處物體經常漏報：使用 `--area-mode depth`，並適度降低 `--dynamic-min-area-ratio`；
- 人員誤報較多：增加 `--yolo-confidence` 或 `--person-alert-ratio`；
- 顯示記憶體不足：優先降低 `--process-res` 和 `--yolo-image-size`；
- 處理速度太慢：增加 `--frame-step`。此參數會減少分析畫格數，但輸出影片仍保留未分析畫格。

建議從預設值開始，使用實際固定機位所收集的正常及遮擋樣本調整參數，不要只依照範例影片決定正式環境的門檻。

## 常見問題

### `torch.cuda.is_available()` 顯示 `False`

請確認 `nvidia-smi` 可正常執行，且 NVIDIA 驅動程式支援 CUDA 12.8 執行環境，再重新執行 `uv sync --frozen`。本專案不需要另外安裝完整 CUDA Toolkit，但需要可用的 NVIDIA 驅動程式。

### DA3 模型下載失敗

請在可連線網路的電腦上使用 `uv run hf download` 下載，再將完整模型目錄複製至伺服器，並透過 `--model-dir` 指向該目錄。

### 提示參考影像與影片畫格的深度尺寸不同

請確認參考影像與影片來自相同機位且具有相同長寬比。最穩定的方式是直接從同一影片來源擷取正常狀態的一幀作為參考影像。

### 無法讀取影片或建立輸出影片

請確認輸入影片路徑正確且已安裝 FFmpeg。預設輸出編碼為 `mp4v`，也可透過 `--codec` 傳入目前 OpenCV 環境支援的四字元編碼。

### 結果只有人員遮擋，沒有一般深度遮擋

這是預期邏輯：所有人員遮罩都會從一般深度變化候選區域中移除；只有人員覆蓋率達到 `--person-alert-ratio` 時，才會以 `PERSON_OCCLUSION` 發出警報。

## 致謝與授權條款

本專案建立於 ByteDance Seed 開源的 [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)，並使用 [Ultralytics](https://github.com/ultralytics/ultralytics) 提供的 YOLO segmentation 功能。

專案程式碼授權條款請參閱 [LICENSE](LICENSE)。部署及散布前，請另外確認所使用的 DA3 模型權重與 YOLO 權重授權條款是否符合你的使用情境。
