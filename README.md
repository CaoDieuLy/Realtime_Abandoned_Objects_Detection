# Realtime Abandoned-Object Detection

Phát hiện **đồ bị bỏ quên / để lại** trong video camera cố định, chạy **thời-gian-thực trên CPU**.

> Nguyên lý: một vật bị bỏ lại = **khác nền sạch** + **đứng yên đủ lâu** + **không phải người/xe**.

Pipeline tách **2 nền** (đúng nguyên lý abandoned-object detection):
- **BGS ngắn hạn, thích nghi** (ViBe) → mask "đang chuyển động".
- **Nền sạch dài hạn** (`clean_bg`, median ~vài chục giây đầu, gần như đóng băng) → vật mới luôn khác nó nên **không bị nuốt** như BGS thường.
- **FSM tĩnh** (`core/static_state.py`): `static_fg = (khác clean_bg) AND (không moving)`, tích **tuổi-tĩnh**, lọc **semantic** (loại **animate = người/xe/động vật**, giữ balo/túi…) → `abandoned` → matcher → **cảnh báo**. → tức loại **mọi motion-object** (qua mask moving + class animate), không chỉ người.

> 📄 **Hồ sơ kỹ thuật + quy trình phát triển + kết quả đầy đủ**: **[`solution_analysis.md`](solution_analysis.md)**. README này là giới thiệu/hướng dẫn chạy; trong đó có chi tiết **§3.13 tỷ-lệ-foreground**, **§3.14 các-chế-độ-cập-nhật-sáng/nền**, **§3.15 tham-số-trade-off-recall/precision**.

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
| `--warmup-motion-mask 0/1` | nền sạch loại **transient ĐỘNG** lúc học (người/cửa đi qua không bị nướng vào nền) | **mặc định ON**; đặt 0 nếu thấy FP lạ ở cảnh khác. Tốt nhất vẫn là chọn **cửa-sổ-warmup lúc cảnh trống** (qua `--bg-learn-seconds`) |
| `--proc-width` / `--sem-proc-width` | độ phân giải xử lý / cho YOLO | camera ≥1080p giữ mặc định 640/960 là tốt |
| `--gather-px` | gộp mảnh mask để bbox **ôm vật** đầy đủ hơn | **mặc định 0** (box bám lõi vật, có thể nhỏ). Muốn box to/đầy hơn → đặt 5–10 (đánh đổi: cảnh đông dễ gộp nhiều người thành 1 box) |
| `--async-semantic auto/on/off` | chạy YOLO ở **thread nền** để vòng chính không nghẽn | **mặc định auto** (ON cho no-feedback): **+~20% FPS**, parity event, hơi nhỉnh recall. Đặt off để về cách đồng bộ cũ |
| `--clip-verify 0/1` | **cổng xác minh open-vocab** (MobileCLIP2) mỗi cảnh báo, lọc FP đám-đông/đổi-sáng | **mặc định OFF**. Bật 1 nếu cần ÍT FP crowd/lighting (đánh đổi ~20–25% FPS ở cảnh đông). Xem mục dưới |

`python run_rtsbs_aod.py --help` chia 2 nhóm **common** (hay dùng) và **advanced** (đã tuned, ít khi đụng).
**Output** (`--outdir`): `events.json` (mỗi cảnh báo: frame, thời điểm, tâm, bbox) + ảnh `alert_*.jpg`.

## Giao diện desktop (`app.py`)
Ứng dụng local có giao diện (tkinter + Pillow) bọc quanh pipeline:
```bash
python app.py            # mở GUI: chọn Video file / Camera trực tiếp
```
- **Chọn nguồn**: file video (hiện **FPS xử lý** để biết tốc độ thực) hoặc camera (tự chọn **FPS** — đặt ≈ FPS xử lý để khung hình không bị trễ/đệm).
- **Cảnh báo trực tiếp**: từ frame phát hiện trở đi, vẽ **bbox đỏ** lên vật bỏ quên; bbox **tự biến mất** khi vật **được lấy đi** (vị trí trở lại nền sạch — ngưỡng `--taken-clear-s`).
- **Nút "Vật bỏ quên (N)"** góc trên-phải mở **pop-up danh sách** vật đang được xác nhận.
- **Bấm vào bbox** để xác nhận/loại: chọn *No* → loại khỏi danh sách (không vẽ nữa).
- **Mỗi vật → 1 file JSON** `object_<id>.json` (trong `aod_sessions/<phiên>/`): id, frame/thời-điểm báo, tâm, bbox, trạng thái (present/taken/rejected).
- Kiểm thử không cần màn hình: `python app.py --selftest --video ABODA/video6.avi` (chạy headless, in + ghi JSON từng vật).

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

**Mức chiếm dụng foreground theo instance mask**

Đây là thước đo cảnh theo **diện tích pixel bị đối tượng chiếm** (mask **detector/YOLO**), giữ kèm số lượng. **% diện tích ≠ số lượng**: vài vật to & gần cam lấp nhiều % hơn rất nhiều người ở xa — vd **vid0355 (≈1 xe) = 6,33%** animate-mask > **video11 (≈40 người) = 3,31%**. ⚠ Đây chỉ là **1 trong 3 đại lượng mật-độ** (mắt-người ước-lượng · detector-count · foreground-khác-nền) — audit đầy đủ + so sánh ba đại lượng ở **[REPORT §6.3](REPORT.md)** (và [solution_analysis §3.13](solution_analysis.md)).

```text
animate-FG = union(mask người / xe / động vật được detector nhận ra)
object-FG  = union(mask balo / ô / túi / vali / chai được detector nhận ra)
semantic-FG = animate-FG union object-FG
coverage = area(mask) / area(frame)
```

Đã chạy tuần tự cả 13 video một lần bằng `yolo26n-seg.pt`, `imgsz=640`, `conf=0,15`; semantic được lấy mẫu **1 frame/giây**, batch 16. Vì CPU chỉ đạt ~2,3 semantic inference/giây, các số **TB/min/max là trên sample 1 Hz**, không phải cực trị tuyệt đối của mọi frame gốc. Tập lệnh và report thô có thể tái lập ở [`measure_instance_occupancy.py`](tools/measure_instance_occupancy.py) và [`aboda_instance_occupancy_1fps.json`](metrics/aboda_instance_occupancy_1fps.json).

- `Semantic-FG TB/min/max`: union pixel mask của animate + static-object.
- `Animate n TB/max`: số instance animate detector nhìn thấy; đây là **detected count**, không phải số người thật trong cảnh.
- `GT target mask`: diện tích mask vật bỏ quên từ annotation ABODA, chính xác theo nhãn; không phụ thuộc YOLO.
- `FG khi GT xuất hiện`: semantic-FG trong các sample từ `start_frame` đến `end_frame` của vật GT. `vid0103`/`vid0355` không có annotation event trong `aboda_gt.json`.

| Video | Semantic-FG<br>TB / min / max | Animate n<br>TB / max | Animate mask<br>TB / max | Static-object mask<br>TB / max | GT target mask | FG khi GT xuất hiện<br>TB / min / max |
|---|---:|---:|---:|---:|---:|---:|
| video1 | 2,72 / 0,00 / 8,26% | 1,1 / 3 | 2,45 / 7,83% | 0,27 / 0,74% | 0,57% | 3,05 / 0,62 / 8,26% |
| video2 | 1,73 / 0,00 / 10,12% | 2,6 / 11 | 1,73 / 10,12% | 0,01 / 0,36% | 0,11% | 0,92 / 0,31 / 2,48% |
| video3 | 0,91 / 0,30 / 3,01% | 1,4 / 3 | 0,77 / 2,74% | 0,15 / 2,57% | 0,25% | 1,44 / 0,33 / 2,88% |
| video4 | 1,85 / 0,00 / 13,23% | 1,6 / 3 | 1,78 / 13,23% | 0,06 / 0,75% | 0,46% | 2,34 / 0,00 / 13,23% |
| video5 | 1,07 / 0,00 / 9,03% | 0,7 / 4 | 1,07 / 9,03% | 0,00 / 0,00% | 0,22% | 0,78 / 0,00 / 3,87% |
| video6 | 0,63 / 0,00 / 19,43% | 0,2 / 5 | 0,62 / 19,43% | 0,01 / 1,19% | 0,24% + 0,33% | 2,12 / 0,00 / 19,43% |
| video7 | 0,42 / 0,00 / 9,00% | 0,1 / 2 | 0,36 / 9,00% | 0,06 / 1,49% | 0,73% | 0,43 / 0,00 / 2,54% |
| video8 | 0,63 / 0,00 / 8,43% | 0,1 / 2 | 0,42 / 8,43% | 0,21 / 2,63% | 2,27% | 1,10 / 0,00 / 6,76% |
| video9 | 1,41 / 0,00 / 10,26% | 0,3 / 2 | 1,31 / 9,51% | 0,10 / 2,41% | 1,03% | 0,38 / 0,00 / 4,40% |
| video10 | 1,62 / 0,00 / 10,79% | 0,3 / 2 | 1,07 / 10,79% | 0,56 / 1,66% | 1,03% | 1,74 / 1,11 / 5,13% |
| video11 | 3,32 / 1,19 / 5,44% | 10,5 / 15 | 3,31 / 5,36% | 0,02 / 0,33% | 0,08% | 3,39 / 1,19 / 5,44% |
| vid0103 | 6,05 / 0,00 / 10,39% | 1,7 / 4 | 5,69 / 9,74% | 1,09 / 9,31% | — | — |
| vid0355 | 6,34 / 0,00 / 11,33% | 2,4 / 5 | 6,33 / 11,33% | 0,03 / 0,50% | — | — |

Ví dụ đọc đúng số liệu:

- **video11**: detector thấy trung bình **10,5**, tối đa **15** animate instance; mask animate chiếm TB **3,31%**, cao nhất **5,36%** khung hình. Cảnh thực tế có thể có khoảng 40 người/vật như quan sát bằng mắt, nhưng **không được ghi thành 40 trong metric**: model bỏ sót/ghép các người xa hoặc che nhau. Đây là giới hạn perception cần báo cáo, không phải bằng chứng cảnh thưa.
- **vid0355**: animate mask chiếm TB **6,33%**, tối đa **11,33%**. Trong đó model gán `car` TB **5,47%** (max **9,96%**) và `person` TB **0,87%** (max **2,67%**); xe đôi lúc bị đổi nhãn `truck`, nên không cộng hai cột này. COCO YOLO không có lớp **washing machine**, nên static-object mask **không phải** diện tích máy giặt; cần annotation riêng hoặc model open-vocabulary/fine-tune để đo vật đó.

`owner-gate` vẫn dùng semantic person/vehicle để kiểm tra **chủ đứng gần vật**; đây là tín hiệu cục bộ cho quyết định alert. Với product, nên log đồng thời `semantic-FG coverage` và detected count; khi cần audit/đánh giá chính xác, thêm GT mask hoặc model nhận diện được đúng class vật mục tiêu.

## Tốc độ
Đo trên **CPU** (chưa dùng GPU), YOLO-seg `yolo26s` OpenVINO, semantic mỗi 10 frame:

| Cấu hình | FPS |
|---|---|
| Camera 640×480 | **~7–9 FPS** (video7 8.4, video11 7.2) |
| Camera ≥1080p (proc-width 640 + sem-proc 960) | ~4–6 FPS |
| Nếu đổi sang PyTorch `.pt` (bỏ OpenVINO) | chậm hơn ~1.5× |

→ Không phải 30fps, nhưng **đủ cho bài toán này** (vật phải đứng yên vài giây mới báo). Nhanh hơn nữa: `--semantic-every 15`, `--yolo-imgsz 640`, hoặc GPU `--yolo-device 0`.

**`--async-semantic` (mặc định auto, ON cho no-feedback):** YOLO chạy ở **thread nền** nên vòng BGS/FSM không nghẽn chờ inference. Đo trên video11 (CPU idle): sync 5.0→**async 6.1 FPS = +20%** (cảnh nhẹ lợi nhiều hơn vì YOLO chiếm tỉ lệ lớn hơn), **không đổi event** (parity), còn nhỉnh recall do map tươi hơn. Chỉ áp dụng mode no-feedback (RT-SBS feedback giữ đồng bộ). Số liệu: [`eval_async_clip/SUMMARY.md`](eval_async_clip/SUMMARY.md).

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
| **`no-feedback`** (mặc định) | KHÔNG | ViBe C++ (pybgs) | Nhanh nhất; loại **mọi motion-object**: ViBe loại pixel **đang chuyển động** + semantic loại **class animate (người/xe/động vật) kể cả đứng yên**; giữ vật (balo/túi…) ở hậu kỳ. |
| **`instance-feedback`** | một chiều (force-FG) | ControlledViBE | YOLO-seg bảo vệ vật động khỏi bị ViBe nuốt. |
| **`dense-feedback`** | hai chiều (force-BG + force-FG) | ControlledViBE | RT-SBS đầy đủ: dense pixel seg điều khiển cả update nền (SegFormer/PSPNet/dense maps). |
| `custom` | tùy cờ | tùy `--bgs-backend` | tự ghép `--semantic-mode` / `--semantic-feedback` / `--aod-motion-source`. |

## Các chế độ cập nhật SÁNG & cập nhật NỀN
4 lớp đụng tới nền/ánh sáng (chi tiết + tương tác: [solution_analysis §3.14](solution_analysis.md)):

| Lớp | Cơ chế (cờ) | Làm gì |
|---|---|---|
| **Dựng nền** | warmup median + **B** (`--warmup-motion-mask` 1) | nền sạch FROZEN; B loại pixel-motion (người/cửa đi qua không nướng vào nền) |
| **Bảo trì nền** | slow-update (`--clean-update-lr`) · **relight** (`--relight-dv`) · **light-comp** (`--heal-cov`) | hấp thụ trôi-nền chậm; **rebuild toàn bộ** khi đổi sáng lớn (ngày↔đêm); re-baseline khi đổi-sáng phủ rộng |
| **Khử sáng tại alert** | **light-struct** (`--light-struct`, NCC) · **C** (`--alert-min-support`) | patch nền-vân chỉ-đổi-sáng → bỏ; candidate "ma" (đã bị hấp thụ, hết newdiff) → defer |
| **Agent-ghost** | **heal-revealed** (`--heal-revealed`) | YOLO trên clean_bg tìm xe/người baked → heal khi chúng rời đi |

→ Mặc định **bật hết**. Persist-protect chặn slow-update nuốt vật tĩnh. Giới hạn: mở-cửa/đổi-trạng-thái-cảnh runtime + warmup-nhiễm (người bất-động suốt warmup) — chưa lớp nào trị.

## Tham số trade-off Recall ↔ Precision
Stance an ninh: **bỏ-sót-vật tệ hơn báo-nhầm** → default lệch recall. Núm chính (đầy đủ + 2 recipe đối cực: [solution_analysis §3.15](solution_analysis.md)):

| Tăng **RECALL** (bắt nhiều, chấp nhận FP) | Tăng **PRECISION** (ít FP, chấp nhận miss) |
|---|---|
| `--th-diff`↓ · `--area-min`↓ · `--ts-static`↓ · `--proc-width`↑ | ngược lại |
| `--tau-animate`↑ (ít nhầm vật=người) | `--tau-animate`↓ · `--person-overlap-max` 0.5 · `--owner-clear-s`↑ |
| `--heal-revealed 0` (an ninh tối đa) | nhóm suppress sáng (`--light-struct-ncc`↓, `--relight-dv`↓, `--heal-cov`↓) — ⚠ **nuốt vật** |

⚠ **Nhóm suppress-sáng nguy hiểm** (có thể nuốt vật thật) — chỉ siết khi chấp nhận rủi ro. Đòn bẩy recall sạch nhất = **`--proc-width`↑** (vật nhỏ rõ hơn, không nuốt vật).

## Tăng tốc YOLO trên CPU — OpenVINO (mặc định)
Default `--yolo-weights yolo26s-seg_openvino_model`: lần chạy đầu **tự export** OpenVINO từ `yolo26s-seg.pt` (ultralytics tự tải `.pt`); thiếu package `openvino` → **fallback PyTorch** (vẫn chạy). Benchmark CPU (độ chính xác **không đổi**, events identical):

| Backend | video7 | video11 |
|---|---|---|
| PyTorch `.pt` | 5.6 FPS | 4.6 FPS |
| ONNX FP32 (`.onnx`) | 6.5 (1.16×) | 4.9 (1.07×) |
| **OpenVINO FP32 (mặc định)** | **8.4 (1.50×)** | **7.2 (1.57×)** |

Đổi backend: `--yolo-weights yolo26s-seg.pt` (PyTorch) · `yolo26n-seg.pt` (nano, nhanh nhưng recall kém) · `*.onnx`. **Không dùng INT8** (giảm recall → hỏng đúng nút thắt nhận-diện-người).

## Cổng xác minh CLIP — lọc FP nâng cao (`--clip-verify`, tùy chọn)
Cổng **open-vocabulary** chạy **một CLIP zero-shot (MobileCLIP2-B) mỗi cảnh báo** (1 lần/ứng-viên, có cache) để loại các FP mà logic pixel không tách được khỏi vật-bỏ-quên: **đám đông** (người detector sót), **đổi-trạng-thái-cảnh/đổi-sáng** (tường/sàn/cửa). Nó phán *nội dung crop* so 2 nhóm prompt OBJECT (balo/vali/hộp/ô…) vs NOT-OBJECT (người/đám đông/sàn/tường/cửa/xe/bóng/lóa).

**Recall-safe (abstain):** chỉ suppress khi crop **chắc chắn không phải vật** (`P(object) < 0.25` **và** top-1 không-phải-vật `≥ 0.30`); mơ hồ → **giữ** (báo). Chạy SAU owner-gate nên lúc đó chủ thường đã rời → crop là vật, không phải người.

```bash
pip install open_clip_torch        # cần thêm (torch đã có); weights tự tải HF lần đầu
python run_rtsbs_aod.py --video X --outdir out/ --clip-verify 1
```

Kết quả đo (ABODA, async ON, [chi tiết + bằng chứng](eval_async_clip/SUMMARY.md)):

| Video | clip OFF → ON | |
|---|---|---|
| **video7** (đổi sáng) | FP **3 → 0** | giữ handbag (p_obj 0.99) |
| **video1** (sạch) | FP 0 → 0 | **0 regression** |
| **video11** (đám đông) | FP **9 → 3 (−67%)** | umbrella vẫn giữ (đã chứng minh recall-safe) |

**Full-sweep 13 video (async + CLIP on)** — số liệu [`results_full_clip/`](results_full_clip/), phân tích [REPORT §9](REPORT.md): **FP 18 → 4 (−78%)**, 8/11 video GT về **0 FP**, vid0355 **hết báo-lặp máy giặt** (2→1). ⚠ Recall **12/12 → 10/12** (mất video6-túi2 + video11-ô — đang điều tra căn nguyên async/CLIP/pybgs, xem REPORT §9.5). Vì stance *bỏ-sót tệ hơn báo-nhầm*, **2 cờ để opt-in** cho tới khi khép khe recall.

| Cờ | Default | Ý nghĩa |
|---|---|---|
| `--clip-verify` | 0 | bật/tắt cổng |
| `--clip-keep-conf` | 0.25 | giữ (không suppress) nếu P(object) ≥ ngưỡng — **cao hơn = an toàn recall hơn** |
| `--clip-suppress-conf` | 0.30 | chỉ suppress nếu top-1 (không-phải-vật) ≥ ngưỡng — cao hơn = suppress ít hơn |
| `--clip-recheck-s` | 2.0 | re-verify ứng-viên sau N giây (nâng 4–5s để **giảm chi phí ở cảnh đông**) |
| `--clip-model` | MobileCLIP2-B | đổi `MobileCLIP2-S2` để nhanh ~2–3× (đánh đổi accuracy) |

**Chi phí:** ~600ms/crop CPU, 1 lần/ứng-viên → ~**20–25% FPS ở cảnh đông** (nhiều ứng-viên re-verify), ít hơn ở cảnh thưa. Giảm bằng `--clip-recheck-s` cao hơn hoặc `--clip-model MobileCLIP2-S2`. Vì có chi phí nên **default OFF**; bật cho deployment cần ít FP crowd/lighting.

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
  clip_verifier.py      # cổng xác minh open-vocab MobileCLIP2 (--clip-verify)
tools/                  # sinh dense semantic (SegFormer/PSPNet)
eval_async_clip/ # kết quả đo async + CLIP (events.json + alert jpg + log + SUMMARY.md)
requirements.txt
solution_analysis.md    # phân tích chi tiết + quá trình phát triển + kết quả
```

## Kết quả & giới hạn (tóm tắt)
Chi tiết đầy đủ + quá trình phát triển: **[`solution_analysis.md`](solution_analysis.md)**.

- **ABODA full-sweep (13 video, 1 cấu hình mặc định)** — lần đầu đủ **11 video GT** (gồm video4/5): **Recall 12/12 · FP 18** ([chi tiết §3.11](solution_analysis.md)). FP dồn video11 (11, đám đông) + video7 (3, lighting-residual); 5 cảnh sạch 0 FP. (2 video ngoài-GT: vid0355 = 1 máy giặt báo-lặp-2-vị-trí; vid0103 = 1 FP warmup-nhiễm.)
- **Đã trị**: car-ghost (xe đỗ-rồi-đi, `--heal-revealed` detect-trên-plain-median); warmup-ghost ĐỘNG (`--warmup-motion-mask`, video6 f780); ma-sàn/cột sau đổi-sáng (`--alert-min-support` C); khe-hở đổi-sáng-từ-từ (tune relight); owner-gate báo-khi-chủ-đứng (yolo26s + person-presence memory). Đã thử & **loại**: A regional-shift, targeted-B (overfit) — xem §3.12.
- **Giới hạn cố hữu**: (1) **perception** — model không phân biệt vật-nhỏ-mảnh-đứng (ô gập) với người → mọi heuristic chỉ đánh đổi recall↔precision; (2) **scene-state-change** (mở cửa/ghế xê dịch runtime) + **warmup-nhiễm** (người bất-động suốt warmup) → chưa lớp nào trị (§4). Lối ra: model **nhận ra vật** + bắt **"chủ rời đi"** (open-vocab / fine-tune).
