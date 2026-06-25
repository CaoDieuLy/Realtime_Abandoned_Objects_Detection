# demov2 — Phát hiện đồ bỏ quên (RT-SBS feedback + clean-background FSM)

Vật bị bỏ lại = **khác nền sạch** + **đứng yên đủ lâu** + **không phải người/xe**.
Pipeline tách 2 nền (đúng nguyên lý AOD):

- **ViBE (ngắn hạn, adaptive)** → mask "đang chuyển động"; có thể nhận **RT-SBS semantic feedback**.
- **clean_bg (dài hạn, median ~20s đầu, gần như đóng băng)** → vật mới luôn khác nó, không bị nuốt.
- **FSM tĩnh** (`core/static_state.py`): `static_fg = (khác clean_bg) AND NOT moving`, tích tuổi-tĩnh, lọc semantic → `abandoned` → matcher → cảnh báo.

## Cài đặt
```
python -m pip install -r demov2/requirements.txt
```
Chạy **từ thư mục gốc repo** (để thấy `ABODA/`, `yolo26n-seg.pt`, và tự tạo `config/`).

## Cơ chế chạy hiện tại

`demov2/run_rtsbs_aod.py` hiện chạy theo 5 khối độc lập, nên có thể thay semantic model mà không phải viết lại FSM AOD:

1. **Warm-up clean_bg**: lấy median khoảng `--bg-learn-seconds` đầu video để tạo nền sạch dài hạn.
2. **Moving mask**: lấy từ `pybgs` ViBe hoặc `ControlledViBE`.
3. **Semantic source**: một trong `none`, `online-yoloseg`, `online-segformer`, `online-pspnet`, hoặc `dense` map offline.
4. **RT-SBS feedback**: nếu bật, semantic sửa mask ViBE trước khi cập nhật nền. YOLO-seg chỉ dùng được feedback một chiều `force-FG`; dense semantic dùng đủ `force-BG + force-FG`.
5. **AOD FSM**: `clean_bg` diff + `NOT moving` + tuổi tĩnh + semantic gate (`animate` reject, `object` keep, optional `stuff` reject) + matcher/dedup/owner-gate.

Phần semantic trong demov2 không bị buộc vào một model duy nhất. Điều kiện để thay semantic source là nguồn mới phải cung cấp được mask/probability cùng vai trò:

- `animate_prob16`: người/xe/động vật để loại khỏi AOD và, nếu feedback instance, bảo vệ FG.
- `object_prob16`: balo/vali/túi/ô/hộp... để tăng tín hiệu giữ vật.
- `stuff_prob16` nếu có: nền cảnh như floor/wall/water để reject, chỉ nên bật với model đủ tin cậy.
- Với feedback hai chiều kiểu RT-SBS gốc cần dense semantic theo pixel; với instance mask chỉ nên dùng một chiều `force-FG` hoặc dùng semantic gate hậu kỳ.

## 3 chế độ (`--mode`) — đặt tên theo CƠ CHẾ, không theo "demov1"

| `--mode` | Semantic vào BGS? | BGS backend | Nguồn semantic | Ý nghĩa |
|---|---|---|---|---|
| **`no-feedback`** | KHÔNG | pybgs (C++) | YOLO-seg | ViBE nhanh, semantic **chỉ trừ người ở hậu kỳ**, không cập nhật vào nền. Nhanh nhất. (= cấu trúc demov1) |
| **`instance-feedback`** | CÓ — **một chiều** (force-FG protect) | ControlledViBE | YOLO-seg | Instance seg + RT-SBS feedback bảo vệ vật động khỏi bị ViBE nuốt. |
| **`dense-feedback`** | CÓ — **hai chiều** (force-BG + force-FG) | ControlledViBE | SegFormer online mặc định; hoặc dense maps/PSPNet nếu chỉ định | RT-SBS gốc đầy đủ: dense pixel seg điều khiển cả update nền. |
| `custom` | tùy cờ | tùy `--bgs-backend` | tùy `--semantic-mode` | dùng các cờ lẻ như nhập. |

> `--mode` **ghi đè** `--bgs-backend` / `--semantic-feedback` / `--aod-motion-source`.
> Với `no-feedback` và `instance-feedback`, nếu `--semantic-mode` đang để mặc định `dense` thì code tự chuyển sang `online-yoloseg`.
> Với `dense-feedback`, nếu chưa truyền `--semantic-dir` thì code tự chuyển sang `online-segformer`.
> Vì sao tách: `pybgs.apply()` gộp segment+update (không nhận mask sửa) nên **không feedback được**; muốn feedback phải dùng `ControlledViBE`.

## Chạy
```bash
# Nhanh, FP thấp, KHÔNG semantic-feedback
python demov2/run_rtsbs_aod.py --video ABODA/video11.avi --mode no-feedback

# Instance seg + feedback một chiều
python demov2/run_rtsbs_aod.py --video ABODA/video11.avi --mode instance-feedback

# Dense pixel seg + RT-SBS hai chiều (SegFormer online)
python demov2/run_rtsbs_aod.py --video ABODA/video11.avi --mode dense-feedback --segformer-local-files-only

# Dense maps offline: sinh map trước rồi chạy RT-SBS hai chiều từ file
python demov2/run_rtsbs_aod.py --video ABODA/video11.avi --mode dense-feedback --semantic-mode dense --semantic-dir demov2/semantics_aboda/video11
```
Lưu mask debug: thêm `--save-masks-every 300 --outdir <dir>`.

## Các trục cấu hình lẻ (mode `custom`, hoặc override)
- `--bgs-backend {controlled|pybgs}` — ControlledViBE (numba, hỗ trợ feedback) hay ViBe C++ (nhanh, không feedback).
- `--semantic-feedback {on|off}` — bật/tắt RT-SBS feedback. `online-yoloseg` tắt BG-rule nên chỉ feedback một chiều; dense/SegFormer/PSPNet có thể feedback hai chiều.
- `--aod-motion-source {raw-vibe|rtsbs|framediff}` — cổng "moving" của FSM. **raw-vibe** (ViBE) lọc clutter động tốt; `rtsbs` dùng mask sau feedback; `framediff` bắt vật đặt sớm nhưng dễ FP hơn.
- `--semantic-mode {none|online-yoloseg|online-segformer|online-pspnet|dense}`.
- Class roles: `--yolo-animate-classes`, `--yolo-object-classes` (tên COCO; vd `--yolo-animate-classes car,bus,truck` bỏ person giữ car).
- Lọc/đếm: `--dilate-animate`, `--area-min`, `--ts-static`, `--owner-gate-local`, `--dedup-dist`, `--motion-to-static`, `--stuff-reject`, …

- `--proc-width N` (mặc định 640) / `--sem-proc-width N` (mặc định 960) — **TÁCH TRỤC độ phân giải**: BGS/FSM chạy ở `proc-width` (nhẹ, bậc-hai theo pixel → nhanh + area-threshold có nghĩa); YOLO/semantic chạy ở `sem-proc-width` (chi tiết cao, mask resize về proc). Cho camera độ phân giải cao (vd 2560px) → ~10× nhanh mà vẫn detect tốt.
- `--heal-revealed {0|1}` (+`--heal-lr` 0.15, `--heal-release-s` 5.0) — **adaptive dual-bg** cho camera có xe/người ra-vào. Chạy YOLO TRÊN clean_bg tìm agent (xe/người) bị "nướng" vào nền → cho clean_bg **EMA thích nghi tại pixel đó**:
  - **Heal vô điều kiện** mỗi frame → xe đỗ thì clean_bg=xe (newdiff~0); xe đi thì clean_bg hóa nền-thật trong ~0.5s → **ghost chết trước ngưỡng 5s** (diệt car-ghost).
  - **Tự kết thúc (release)**: gỡ pixel về frozen sau `heal-release-s` giây khi (**đã có motion** = agent thật sự rời, **VÀ** YOLO xác nhận không-còn-agent, **VÀ** clean_bg settled). → **không có điểm-mù vĩnh viễn** (vali-bom đặt sau ở chỗ xe cũ vẫn bị bắt). Tín hiệu **motion+YOLO kép** tránh release sớm (xe đỗ YOLO sót → motion=0 → không gỡ; người đi ngang → YOLO thấy xe → không gỡ).
  - **B — recompute sau relight**: khi clean_bg dựng lại (đổi sáng), chạy lại YOLO để cập nhật mask.
  - Vật thật không nằm trong mask → không đụng. **Default ON** (no-op nơi không có agent baked như ABODA = 0%; trị car-ghost nơi có; release tự-kết-thúc bound blind-spot ~`--heal-release-s`). Đặt `0` nếu muốn tối đa thận trọng an ninh.
  - KQ: vid0355 car-ghost biến mất (2 ev = máy giặt) · vid0103 sạch (1 ev) · video8 HIT/0FP (heal no-op).
  - ⚠️ Giới hạn (perception): YOLO **sót** agent lúc warmup → vẫn ghost; vật-tĩnh **nhận nhầm** là người (không bao giờ "rời") → blind-spot cục bộ tại đó. Cảnh an ninh cao: cân nhắc ảnh nền trống chụp sẵn.


## Gợi ý chọn mode

- **Baseline nhanh, ít phụ thuộc semantic**: dùng `--mode no-feedback`. BGS chạy bằng `pybgs`, semantic chỉ dùng ở tầng AOD để loại người/xe và giữ vật.
- **Muốn thử feedback nhưng vẫn nhẹ**: dùng `--mode instance-feedback`. YOLO-seg điều khiển feedback một chiều `force-FG`, không dùng BG-rule.
- **Muốn chạy gần RT-SBS gốc nhất**: dùng `--mode dense-feedback`. SegFormer/PSPNet/dense maps cung cấp semantic theo pixel để dùng đủ `force-BG + force-FG`.
- **Muốn tự ghép cấu hình**: dùng `--mode custom` rồi đặt riêng `--bgs-backend`, `--semantic-mode`, `--semantic-feedback`, và `--aod-motion-source`.

## Cấu trúc
```
run_rtsbs_aod.py        # orchestrator: parse + preset --mode + vòng lặp
core/
  clean_bg_prior.py     # build_warmup_background (nền sạch median 20s đầu)
  controlled_vibe.py    # ViBE Python (numba JIT) — cho phép chèn mask feedback
  semantic_feedback.py  # luật RT-SBS τBG/τFG (một/hai chiều)
  static_state.py       # FSM: clean_bg-diff + framediff/tight + persist + tuổi-tĩnh + cổng semantic + relight
  semantic_classes.py   # tập lớp moving/object/stuff (LUT → ids)
  static_matching.py    # gom blob → theo vết → ứng viên
  dense_semantic.py     # đọc dense semantic map (mode dense)
tools/                  # sinh dense semantic (SegFormer/PSPNet), batch driver
```

## Output (mỗi `--save-masks-every`)
`events.json` · `alert_f*.jpg` · `raw_vibe_f*` (ViBE) · `rtsbs_f*` (sau feedback) · `semantic_f*.png`/`semantic_vis_f*` (SegFormer) · `statnew/newdiff/fg/staticfg/moving/keep/stuff/tight/age` · `cleanbg_f*` · `frame_f*`/`cleanbg_color_f*`. (`--debug-owner 1` in chẩn đoán owner-gate mỗi alert.)

## Tăng tốc YOLO trên CPU — OpenVINO (DEFAULT)
Model `yolo26s-seg` (recall người tốt cho owner-gate) chậm ~2× so với nano. Default đã chuyển sang **OpenVINO FP32** để bù: `--yolo-weights yolo26s-seg_openvino_model`.
- **Tự lo**: lần chạy đầu nếu chưa có folder → **tự export** từ `yolo26s-seg.pt` (ultralytics tự tải .pt); nếu thiếu package `openvino` → **fallback PyTorch** (`.pt`) để vẫn chạy. Fresh checkout không lỗi.
- Cài để có tốc độ: `pip install openvino` (hoặc `onnxruntime` cho .onnx).

Benchmark CPU (v7/v11), **độ chính xác không đổi** (v7 events identical cả 3):
| Backend | video7 | video11 |
|---|---|---|
| PyTorch `.pt` | 5.6 FPS | 4.6 FPS |
| ONNX FP32 (`.onnx`) | 6.5 (1.16×) | 4.9 (1.07×) |
| **OpenVINO FP32 (default)** | **8.4 (1.50×)** | **7.2 (1.57×)** |
- **OpenVINO ~1.5× = tốt nhất trên CPU** — xóa gần hết penalty nano→s (được recall model-s ở tốc độ ~nano). ONNX-CPU chỉ ~1.1×.
- Đổi backend: `--yolo-weights yolo26s-seg.pt` (PyTorch thuần) · `yolo26n-seg.pt` (nano, nhanh nhưng recall kém) · `*.onnx`.
- **KHÔNG dùng INT8** (nhanh hơn nhưng giảm recall → hỏng đúng nút thắt person-recall). Đòn lớn hơn: `--yolo-imgsz 960→640` (phải test lại recall).

## Kết quả & giới hạn

> **`no-feedback` CHÍNH LÀ demov1** — cùng cấu trúc (pybgs ViBE + clean_bg FSM + trừ-người hậu kỳ, không feedback). Vì vậy không cần so sánh demov1 riêng: nó là **cùng một pipeline**.

### Full-sweep ABODA (toàn bộ 13 video, 1 cấu hình chung) — xem [`results_full/BAOCAO_FULL_SWEEP.md`](results_full/BAOCAO_FULL_SWEEP.md)
Chạy `no-feedback --heal-revealed 1 --semantic-every 10` + defaults, chỉ khác warmup (vid0103=8s, vid0355=3s, còn lại 20s):
- **Recall 12/12** (không miss thật). 3 "MISS" ban đầu (v2/v4/v9) là **lỗi chấm điểm** (center-distance tol quá chặt; box bắt được nhưng tâm lệch 7–26px). → chấm AOD phải dùng **box-overlap**, không tol cứng.
- **FP ≈ 42**, tập trung **video7 (21) + video11 (12) + video6 (7) = 40/42**. Sáu video sạch (1,2,3,4,8,9) = **0 FP**. vid0103/0355 sạch (car-ghost đã trị).
- **video7 = đổi sáng TỪ TỪ** (tối→bật đèn tăng dần→sáng hẳn→mới đặt vật): relight cũ rebuild clean_bg lúc còn **tối (V=133)** giữa ramp, rồi sáng dần dV≈24 **< relight-dv 30** & rải rác **< light-comp 15%** → nền-tối-đóng-băng vs cảnh-sáng → 13 FP nổ ở 5s. → **ĐÃ SỬA (xem dưới).**
- **video9**: bắt vật nhưng **báo 2 lần**, 1 lần lúc **chủ còn đứng cạnh** = lỗ hổng **owner-leaves** (cùng họ với video7, đã cải thiện bằng presence-memory).

### 2 FIX sau full-sweep (defaults mới) — kết quả `results_relight/*_s`
**FIX 1 — khe hở đổi-sáng-từ-từ:** đã thử `--light-norm` (global affine) → **THẤT BẠI** (sáng không-đồng-đều + ~2× chậm → default OFF). Đòn đúng = **tune relight**: `--relight-dv 30→20` + **`--relight-stable-dv 2.0`** (chỉ rebuild khi sáng ĐÃ ổn định/plateau; đang ramp thì pause, không rebuild → không khóa nền tối). → **video7 21→2 FP, video6 7→5 FP, không tốn tốc độ.**

**FIX 2 — owner-gate báo khi chủ còn đứng = YOLO person-recall** (không phải logic gate): nano `yolo26n-seg` emit map RỖNG tại frame alert → gate bị bỏ đói (tau downstream vô dụng). Đòn: (a) **`--yolo-weights yolo26n-seg → yolo26s-seg`** default (recall tốt hơn, **~2× chậm CPU**); (b) **person-presence memory** (owner-gate nhớ "có người gần đây trong `owner_clear_s`" theo không-gian, bền với YOLO flicker). → **túi video7 hoãn đến khi chủ rời** (báo hợp lệ); mọi vật vẫn HIT. `--debug-owner 1` để in chẩn đoán owner-gate mỗi alert.

**`--vibe-timeout`** (mặc định **150f ≈ 5s**, chỉ áp ControlledViBE = `instance`/`dense`; `no-feedback` dùng pybgs nên không đụng): ép hấp thụ FG kẹt → vật bị FG-protect nổi lên được, nhưng nhả **KHÔNG chọn lọc** nên clutter cũng nổi → tăng FP. Đo trên 3 cảnh:

| video | no-feedback (=demov1) | instance · timeout OFF | instance · timeout 150f |
|---|---|---|---|
| video1 (thưa) | HIT · 1 FP | ❌ MISS · 0 FP | ✅ HIT · 0 FP |
| video6 (đổi sáng) | HIT 2/2 · 7 FP | HIT 2/2 · 1 FP | HIT 2/2 · 3 FP |
| video11 (đông) | HIT · 11 FP | HIT · 15 FP | HIT · 18 FP |

- Thưa (v1) → timeout sửa miss (**WIN**); nhiều clutter (v6/v11) → timeout **tăng FP**. → `no-feedback` đạt recall đầy đủ với FP thấp nhất, **vẫn là default tốt nhất**.
- Cốt lõi giảm FP = **cổng moving ViBE** (lọc clutter động) + clean_bg hấp thụ clutter; `framediff` làm FP cao hơn.
- **Nguồn FP v11 (soi 11 ảnh alert)**: **~4 người đứng thuần** (person-recall trị được) · **~3 cụm "owner-present"** = người + **xe đẩy/túi dưới sàn** bị `MORPH_CLOSE` gộp thành 1 box to (vật-tĩnh-thật nhưng **chủ đứng cạnh trong hàng** → lỗ hổng **owner-leaves**, không phải person-miss) · **~3 lóa/sàn/bóng** · 1 đốm mơ hồ. **0 vật-bị-dời, 0 ghost-warmup.** Box-cụm to là **artifact over-merge** (nhiều vật-tĩnh liền kề + 2 lần CLOSE → 1 blob). → đòn: tăng recall người (~4) + **owner-leaves reasoning** (3 cụm) + xử lý lóa/bóng (~3); *không* phải scene-memory (suppress cả box → nuốt vật thật) hay person-aware-warmup.
- **Giới hạn thật**: YOLO/SegFormer **không phân biệt được cái ô gập đứng với người** (mask "person" trùm lên ô) → mọi heuristic (dilate-người, ngưỡng, timeout) chỉ là **tấm chăn / đánh đổi recall↔precision**, không tổng quát. Lối ra thật sự = model **nhận ra vật** + bắt sự kiện **"chủ rời đi"**.

> `rt-sbs/` (gốc paper) và `demov1/` giữ làm tham chiếu — demov2 **độc lập**, không import gì từ demov1; `no-feedback` đã tái hiện đúng cấu trúc demov1.
