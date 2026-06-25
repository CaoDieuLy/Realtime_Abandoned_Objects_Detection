# Realtime Abandoned-Object Detection

Phát hiện **đồ bị bỏ quên / để lại** trong video camera cố định, chạy **thời-gian-thực trên CPU**.

> Nguyên lý: một vật bị bỏ lại = **khác nền sạch** + **đứng yên đủ lâu** + **không phải người/xe**.

Pipeline tách **2 nền** (đúng nguyên lý abandoned-object detection):
- **BGS ngắn hạn, thích nghi** (ViBe) → mask "đang chuyển động".
- **Nền sạch dài hạn** (`clean_bg`, median ~vài chục giây đầu, gần như đóng băng) → vật mới luôn khác nó nên **không bị nuốt** như BGS thường.
- **FSM tĩnh** (`core/static_state.py`): `static_fg = (khác clean_bg) AND (không moving)`, tích **tuổi-tĩnh**, lọc **semantic** (loại người/xe, giữ balo/túi…) → `abandoned` → matcher → **cảnh báo**.

---

## Cài đặt
```bash
git clone <repo-url> && cd <repo>
python -m pip install -r requirements.txt
```
- Python ≥ 3.10. Core: `numpy`, `opencv-python`, `ultralytics` (YOLO-seg).
- **Tăng tốc CPU**: `pip install openvino` (mặc định pipeline chạy YOLO bằng OpenVINO, ~1.5× nhanh hơn PyTorch — xem [Tốc độ](#tốc-độ)). Thiếu cũng chạy được (tự fallback PyTorch).
- **Tùy chọn**: `pybgs` (ViBe C++ nhanh), `torch`+`transformers` (chế độ dense SegFormer/PSPNet).

## Chạy
Đưa vào **đường dẫn một video camera cố định bất kỳ**, hoặc một **camera trực tiếp**:
```bash
# 1) Trên một file video bất kỳ
python run_rtsbs_aod.py --video /path/to/your_video.mp4 --outdir out/

# 2) Trên camera trực tiếp (webcam = index 0; RTSP/USB qua OpenCV)
python run_rtsbs_aod.py --camera-index 0 --outdir out/

# 3) Lưu thêm mask debug mỗi 300 frame
python run_rtsbs_aod.py --video /path/to/your_video.mp4 --outdir out/ --save-masks-every 300
```
Bắt buộc truyền **`--video`** hoặc **`--camera-index`**.

> **Tái lập benchmark**: các kết quả trong [`solution_analysis.md`](solution_analysis.md) đo trên bộ **ABODA** (Abandoned Objects DAtaset, công khai). Tải về rồi trỏ `--video /path/to/ABODA/video1.avi …` là chạy lại được.
**Default đã là cấu hình tốt nhất** (YOLO-seg `yolo26s` chạy OpenVINO + relight + heal). Mỗi cảnh thường **chỉ cần chỉnh một cờ**:

| Cờ | Ý nghĩa | Khi nào chỉnh |
|---|---|---|
| `--bg-learn-seconds` | thời gian học **nền sạch** | **phải kết thúc TRƯỚC khi vật xuất hiện**. Vật vào sớm → đặt nhỏ (vd 3–8s); cảnh trống lúc đầu → để 20s |
| `--video` / `--camera-index` | nguồn video hay camera | bắt buộc chọn 1 |
| `--heal-revealed 0/1` | trị "ghost" khi **xe/người đỗ-rồi-đi** | mặc định ON; đặt 0 nếu muốn tối đa thận trọng an ninh |
| `--proc-width` / `--sem-proc-width` | độ phân giải xử lý / cho YOLO | camera ≥1080p giữ mặc định 640/960 là tốt |

`python run_rtsbs_aod.py --help` chia 2 nhóm **common** (hay dùng) và **advanced** (đã tuned, ít khi đụng).
**Output** (`--outdir`): `events.json` (mỗi cảnh báo: frame, thời điểm, tâm, bbox) + ảnh `alert_*.jpg`.

---

## Chạy tốt nhất cho trường hợp nào
Vì dựa trên "khác nền sạch + đứng yên + không-phải-người/xe":

**Kích thước vật** (đo ở độ phân giải xử lý `--proc-width 640`; camera khác thì quy đổi theo tỉ lệ)
- ✅ **Tin cậy (≈0 FP)**: vật **≥ ~50×50 px (~2.500 px²)** tương phản rõ với nền. ABODA bắt SẠCH các vật từ ~60×35 đến 208×67 px (balo, vali, hộp, túi, thùng, máy giặt…).
- ⚠️ **Nhỏ tới ~20×36 px (~700 px²)** vẫn bắt được (vd cái ô gập ở video11) **nhưng FP cao + model dễ nhầm là người**.
- Bộ lọc diện tích blob: `--area-min` (mặc định **60 px²**) … `--area-max` (**30.000 px²**) — không phải nút thắt cho vật nhỏ; nút thắt là việc **phân biệt vật↔người**.

**Điều kiện camera**
- ✅ **Camera CỐ ĐỊNH** (bắt buộc — nền sạch giả định góc nhìn không đổi); mọi độ phân giải (proc-width tách trục → 1080p/4K vẫn nhanh); góc cao/chéo (hành lang, sảnh, bãi đỗ, lớp học, ngõ).
- ❌ **KHÔNG** dùng cho **PTZ / camera rung / di chuyển**.
- Cảnh có **xe/người đỗ-rồi-đi**: giữ `--heal-revealed 1`.

**Ánh sáng**
- ✅ Đổi sáng **ngày↔đêm, bật/tắt đèn, sáng tăng dần**: `relight` tự dựng lại nền ở mức sáng **đã ổn định** → không báo giả lúc chuyển sáng. Bóng đổ ngoài trời + hồng ngoại/đêm OK.
- ⚠️ Đổi sáng **cực không đều theo vùng** (bật đèn từng góc) còn vài FP cục bộ nhỏ.

**Mật độ người**
- ✅ Đã test tới **đám đông >40 người** (video11: vừa có **khu tụ tập** vừa **rải rác**) — **vẫn bắt được vật trong khu đông MIỄN LÀ vị trí vật ít người qua lại + không bị che lấp**.
- **owner-gate** (hoãn báo khi **chủ còn đứng cạnh** vật) **tự bật khi cảnh THƯA** (trung bình **< `--crowd-n` = 10** blob người/xe); đám đông **dày hơn** → tắt owner-gate (vì kiểm-tra-chủ-từng-vật không còn tin cậy giữa đám đông).
- FP tăng theo mật độ người (giới hạn nhận-diện): cảnh **thưa–vừa sạch nhất**; đám đông dày FP cao hơn nhưng **vật thật vẫn được bắt**.

## Tốc độ
Đo trên **CPU** (chưa dùng GPU), YOLO-seg `yolo26s` OpenVINO, semantic mỗi 10 frame:

| Cấu hình | FPS |
|---|---|
| Camera 640×480 | **~7–9 FPS** (video7 8.4, video11 7.2) |
| Camera ≥1080p (proc-width 640 + sem-proc 960) | ~4–6 FPS |
| Nếu đổi sang PyTorch `.pt` (bỏ OpenVINO) | chậm hơn ~1.5× |

→ Không phải 30fps, nhưng **đủ cho bài toán này** (vật phải đứng yên vài giây mới báo). Nhanh hơn nữa: `--semantic-every 15`, `--yolo-imgsz 640`, hoặc GPU `--yolo-device 0`.

---

## Cơ chế (5 khối, thay model semantic không phải viết lại FSM)
1. **Warm-up `clean_bg`**: median ~`--bg-learn-seconds` đầu → nền sạch dài hạn.
2. **Moving mask**: ViBe (`pybgs` C++) hoặc `ControlledViBE` (numba, nhận feedback).
3. **Semantic source**: `online-yoloseg` (mặc định) · `online-segformer` · `online-pspnet` · `dense` (map offline) · `none`.
4. **RT-SBS feedback** (tùy chọn): semantic sửa mask ViBe trước khi cập nhật nền.
5. **AOD FSM**: `clean_bg`-diff + `không moving` + tuổi-tĩnh + cổng semantic (`animate` loại, `object` giữ, `stuff` reject) + matcher / dedup / owner-gate.

### Thay model semantic (dense pluggable)
Phần semantic **không buộc vào một model**. Nguồn mới chỉ cần xuất các bản đồ điểm (0..`SEMANTIC_MAX`) đúng vai trò:
- `animate_score`: người/xe/động vật → loại khỏi AOD (và bảo vệ FG nếu feedback instance).
- `object_score`: balo/vali/túi/ô/hộp… → tăng tín hiệu giữ vật.
- `stuff_score` (nếu có): floor/wall/water… → reject (chỉ bật với model đủ tin cậy).

Lớp `OnlineYoloSeg` / `OnlineSegFormer` / `OnlinePSPNet` trong `run_rtsbs_aod.py` + `core/dense_semantic.py` là các ví dụ; thêm lớp mới có `infer()` trả về bản đồ điểm là dùng được.

### Chế độ (`--mode`)
| `--mode` | Semantic → BGS? | BGS backend | Ý nghĩa |
|---|---|---|---|
| **`no-feedback`** (mặc định) | KHÔNG | ViBe C++ (pybgs) | Nhanh nhất; semantic **chỉ loại người/giữ vật ở hậu kỳ**. |
| **`instance-feedback`** | một chiều (force-FG) | ControlledViBE | YOLO-seg bảo vệ vật động khỏi bị ViBe nuốt. |
| **`dense-feedback`** | hai chiều (force-BG + force-FG) | ControlledViBE | RT-SBS đầy đủ: dense pixel seg điều khiển cả update nền (SegFormer/PSPNet/dense maps). |
| `custom` | tùy cờ | tùy `--bgs-backend` | tự ghép `--semantic-mode` / `--semantic-feedback` / `--aod-motion-source`. |

## Tăng tốc YOLO trên CPU — OpenVINO (mặc định)
Default `--yolo-weights yolo26s-seg_openvino_model`: lần chạy đầu **tự export** OpenVINO từ `yolo26s-seg.pt` (ultralytics tự tải `.pt`); thiếu package `openvino` → **fallback PyTorch** (vẫn chạy). Benchmark CPU (độ chính xác **không đổi**, events identical):

| Backend | video7 | video11 |
|---|---|---|
| PyTorch `.pt` | 5.6 FPS | 4.6 FPS |
| ONNX FP32 (`.onnx`) | 6.5 (1.16×) | 4.9 (1.07×) |
| **OpenVINO FP32 (mặc định)** | **8.4 (1.50×)** | **7.2 (1.57×)** |

Đổi backend: `--yolo-weights yolo26s-seg.pt` (PyTorch) · `yolo26n-seg.pt` (nano, nhanh nhưng recall kém) · `*.onnx`. **Không dùng INT8** (giảm recall → hỏng đúng nút thắt nhận-diện-người).

## Cấu trúc
```
run_rtsbs_aod.py        # orchestrator: CLI + preset --mode + vòng lặp chính
core/
  clean_bg_prior.py     # nền sạch median warm-up + proc-width
  controlled_vibe.py    # ViBe (numba JIT) — cho phép chèn mask feedback
  semantic_feedback.py  # luật RT-SBS τBG/τFG (một/hai chiều)
  static_state.py       # FSM AOD: clean_bg-diff + framediff/tight + persist + tuổi-tĩnh + semantic gate + relight + heal
  semantic_classes.py   # tập lớp animate/object/stuff (label → ids) + scale điểm
  static_matching.py    # gom blob → theo vết → ứng viên
  dense_semantic.py     # đọc dense semantic map (mode dense)
tools/                  # sinh dense semantic (SegFormer/PSPNet)
requirements.txt
solution_analysis.md    # phân tích chi tiết + quá trình phát triển + kết quả
```

## Kết quả & giới hạn (tóm tắt)
Chi tiết đầy đủ + quá trình phát triển: **[`solution_analysis.md`](solution_analysis.md)**.

- **ABODA full-sweep (13 video, 1 cấu hình)**: **recall 12/12** (bắt được mọi vật). FP tập trung ở vài cảnh khó (đổi-sáng-từ-từ, đám đông); 6 cảnh sạch 0 FP.
- **Đã trị**: car-ghost (xe đỗ-rồi-đi, `--heal-revealed`); khe-hở đổi-sáng-từ-từ (tune relight); owner-gate báo-khi-chủ-đứng (model recall mạnh hơn + person-presence memory).
- **Giới hạn cố hữu (perception)**: model **không phân biệt được vật nhỏ-mảnh đứng (vd ô gập) với người** → mọi heuristic chỉ là đánh đổi recall↔precision. Lối ra thật sự = model **nhận ra vật** + bắt sự kiện **"chủ rời đi"** (open-vocab / fine-tune detector).
