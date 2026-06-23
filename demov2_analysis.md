# Phân tích chi tiết code demov2 — RT-SBS faithful + FSM AOD

> **Scope**: toàn bộ `demov2/` — 900 dòng orchestrator + 7 module core + 5 tool scripts.
> **Dataset**: ABODA (video 1,2,3,7,8,11). **Eval**: `eval/summary.json`.

---

## 1. Kiến trúc tổng quan

```mermaid
graph TD
    V[Video Input] --> WU[clean_bg_prior: Warmup 20s median]
    V --> ViBE[ControlledViBE / pybgs]
    V --> SEM["Semantic Engine<br/>(YOLO-seg / SegFormer / PSPNet)"]
    
    ViBE --> RAW[raw_vibe mask]
    RAW --> FB{SemanticFeedback}
    SEM --> FB
    FB --> RTSBS[rtsbs_mask]
    RTSBS --> ViBE_UP["vibe.update(rtsbs_mask)<br/>RT-SBS feedback loop"]
    
    WU --> FSM[StaticForegroundState]
    RAW --> FSM
    RTSBS --> FSM
    SEM --> FSM
    
    FSM --> AOD["abandoned mask<br/>(static_age ≥ T)"]
    AOD --> MATCH[StaticMatcher]
    MATCH --> GATE["Owner-gate + Dedup<br/>+ Person-hull + Prop1/2/3"]
    GATE --> EVENT[Alert Events]
```

**Triết lý**: Giữ đúng vòng RT-SBS gốc (ViBe + decision table + dense semantic feedback), gắn thêm nhánh AOD riêng dùng `clean_bg` cố định.

**3 mode preset**:
| Mode | BGS | Feedback | Motion gate | Semantic |
|------|-----|----------|-------------|----------|
| `no-feedback` | pybgs C++ | ❌ | raw-vibe | YOLO-seg (chỉ trừ người) |
| `instance-feedback` | ControlledViBE | ✅ 1 chiều (FG-protect) | raw-vibe | YOLO-seg |
| `dense-feedback` | ControlledViBE | ✅ 2 chiều (BG+FG) | rtsbs | SegFormer dense |

---

## 2. Phân tích từng module

### 2.1. `run_rtsbs_aod.py` — Orchestrator (900 dòng)

**Chức năng**: Parse args → init tất cả component → main loop → event output.

**Nhận xét**:

> [!WARNING]
> **God-file anti-pattern**: 900 dòng trong 1 file, gộp 5 wrapper class (`OnlineSegFormer`, `OnlinePSPNet`, `OnlineYoloSeg`, `CrowdEstimator`, `PybgsViBe`) + `parse_args()` 175 dòng / 70+ tham số + main loop 240 dòng. Rất khó maintain và test.

- **70+ CLI args**: configuration space quá lớn, nhiều tham số chỉ dùng cho 1 mode cụ thể. Không có validation chéo (vd `--stuff-reject` + `online-yoloseg` = vô nghĩa vì YOLO-COCO không có stuff class).
- **Main loop (L662–881)**: logic alert (L756–836) lồng 4 cấp `if`, trộn lẫn bbox refinement + dedup + owner-gate + person-overlap trong 1 block → rất khó trace flow.
- `read_frame()` (L49–58): mở/đóng `VideoCapture` mỗi lần đọc 1 frame → **chậm không cần thiết** (chỉ dùng cho initial semantic).

**Điểm tốt**:
- Mode preset system (`--mode`) gói 3 trục config thành 1 lựa chọn → UX tốt.
- Debug output đầy đủ (`--save-masks-every`): 14 loại mask → dễ debug visual.
- Crowd estimator + smoothing window → adaptive behavior.

---

### 2.2. `core/controlled_vibe.py` — ControlledViBE (169 dòng)

**Chức năng**: Port CPU của ViBE GPU gốc RT-SBS, cho phép tách `segmentation()` và `update()` để chèn semantic feedback.

**Nhận xét**:

| Aspect | Đánh giá |
|--------|----------|
| Numba JIT | ✅ `_vibe_segment_njit` parallel + early-exit, ~10× vs cv2 loop |
| Fallback | ✅ Non-numba path dùng `cv2.absdiff` + `cv2.transform` |
| Init | ✅ Đúng RT-SBS: exact copy + noisy samples |
| Update | ⚠️ Pre-computed schedule (`update_vector`, `position`) rolled mỗi frame — đúng logic nhưng **`np.roll` trên mảng lớn mỗi frame = chi phí ẩn** |

> [!NOTE]
> **Update schedule concern**: `update_vector` có size = H×W pixels. Mỗi frame gọi `np.roll()` 4 lần trên các mảng lớn (L151–154). Với 480p → 345K elements × 4 rolls = chi phí đáng kể. RT-SBS GPU gốc dùng random per-pixel trên GPU nên không có vấn đề này.

- **Neighbor update** (L165–168): `dst_y/dst_x = clip(ys + neighbor_row)` — đúng ViBE spec (propagate to neighbor), nhưng dùng `pos_neighbor = np.roll(self.position, ...)` tạo correlation giữa self-position và neighbor-position → **không hoàn toàn independent random** như paper.

---

### 2.3. `core/semantic_feedback.py` — RT-SBS Decision Table (98 dòng)

**Chức năng**: Implement luật BG/FG của RT-SBS dựa trên semantic map 16-bit.

**Nhận xét**:

> [!IMPORTANT]
> **Đây là module trung thành nhất với paper RT-SBS.** Logic decision table đúng:
> - `rule_BG = sem ≤ τ_BG` (pixel không phải moving object → force BG)
> - `rule_FG = (sem − model) ≥ τ_FG × 256` (semantic tăng đột biến → force FG)
> - Color-reuse inter-frame: `tau_bg_star` / `tau_fg_star` (decision table B/S/C → D_t)

- **Sparse instance map handling** (L52–57): `enable_bg_rule=False` cho YOLO-seg (sem=0 không phải "confident BG" → abstain). Đây là adaptation quan trọng và đúng.
- `_update_semantic_model()` (L92–97): random 1/256 update chỉ tại BG pixels — đúng paper.
- **Vấn đề**: `color_map` (L49) lưu toàn bộ frame BGR mỗi lần có semantic → **tốn RAM** (3× frame size) chỉ để so color diff ở frame tiếp.

---

### 2.4. `core/static_state.py` — StaticForegroundState (301 dòng)

**Chức năng**: FSM per-pixel: clean_bg diff → static FG → age → semantic gate → abandoned.

**Đây là module quan trọng nhất cho AOD và cũng phức tạp nhất.**

**Logic flow**:
```
gray = cvtColor(frame, GRAY)
    ↓
[relight check] → nếu đang rebuild clean_bg → return zeros
    ↓
newdiff = |gray − clean_bg| ≥ th_diff → morphOpen
    ↓
[light_comp] nếu coverage > heal_cov → absorb vào clean_bg (trừ persist)
    ↓
framediff = |gray − prev_gray| ≥ th_diff → dilate
    ↓
moving = framediff (hoặc external rtsbs/raw-vibe mask)
static_fg = newdiff AND NOT moving
    ↓
[motion_to_static latch] _moved |= dilate(moving); clear after sustained BG
    ↓
[semantic gate] animate ≥ τ → reject; object ≥ τ → keep; stuff ≥ τ → reject
valid = static_fg AND keep AND NOT stuff AND (moved if prop3)
    ↓
static_age[valid] += dt;  age[gone] = 0;  age[decaying] -= decay×dt
abandoned = (static_age ≥ t_static_s)
```

**Nhận xét chi tiết**:

| Feature | Đánh giá |
|---------|----------|
| Relight (ngày↔đêm) | ✅ Port từ demov1, dùng V+S divergence + hold counter |
| Light-comp (heal) | ✅ Adaptive alpha (sáng=1.0, tối=0.05) |
| Persist-protect | ✅ Chặn slow-update nuốt vật tĩnh |
| Motion-to-static latch (Prop3) | ✅ Ý tưởng đúng: vật phải "được mang vào" trước khi static |
| Tight mask | ✅ `fgbg AND NOT framediff` → giữ full object shape cho bbox |

> [!WARNING]
> **Race condition trong relight**: `_relight_step()` trả `True` ngay khi phát hiện diverging (L136), khiến toàn bộ frame bị skip (`return self._zeros_result()`). Nếu lighting thay đổi từ từ (gradual), `_switch_count` increment nhưng **mỗi frame diverging đều bị drop** → miss real events trong window đó.

> [!WARNING]
> **`clean_bg` update leak**: L276–283, update condition = `(NOT static_fg) OR stuff_b` AND `NOT protect_persist`. Nhưng `static_fg = newdiff AND NOT moving` — nếu ViBE moving gate flickering (người đi qua nhanh), một pixel vật tĩnh có thể bị moving=True nhất thời → `static_fg=False` → **clean_bg bị update tại pixel vật** → vật dần bị absorb.

---

### 2.5. `core/static_matching.py` — StaticMatcher (113 dòng)

**Chức năng**: Gom blob từ abandoned mask, track qua frame bằng IoU + distance.

**Nhận xét**:
- **Greedy matching** (L77–95): O(cands × blobs) greedy, không phải Hungarian → suboptimal khi nhiều candidate gần nhau. Chấp nhận được vì typical candidate count < 10.
- **EMA bbox smoothing** (L90): `0.7 * old + 0.3 * new` — hardcoded, giúp ổn định nhưng làm chậm phản ứng với shape change.
- **`max_cands=100`**: sort theo `first_seen` rồi cắt → ưu tiên candidate cũ. Đúng cho AOD (vật cũ quan trọng hơn).
- **Thiếu**: Không có confidence score per candidate; không có track-level semantic voting.

---

### 2.6. `core/clean_bg_prior.py` — Warmup Background (51 dòng)

**Nhận xét**: Đơn giản, đúng. Median warmup qua N sampled frames → robust hơn mean.

> [!NOTE]
> Dùng `cap.set(CAP_PROP_POS_FRAMES, fi)` để seek → **không reliable** trên một số codec (seek to nearest keyframe, không exact). Tốt hơn nên đọc tuần tự và skip.

---

### 2.7. `core/semantic_lut.py` — Class Lookup (184 dòng)

**Nhận xét**: Clean, well-structured.
- `MOVING_OBJECT_TERMS`, `STATIC_OBJECT_TERMS`, `STUFF_BACKGROUND_TERMS`: 3 tập từ vựng phủ rộng.
- `norm_label()`: xử lý `/`, `,`, `-`, `_` + split multi-word → robust matching.
- **Hạn chế**: ADE20K-150 không có `umbrella/handbag` → SegFormer KHÔNG bao giờ cho positive object signal cho ô/túi trên ABODA. YOLO-COCO có `umbrella` nhưng confidence thấp cho ô gập nhỏ.

---

### 2.8. `core/dense_semantic.py` — Dense Map Reader (80 dòng)

**Nhận xét**: Hỗ trợ `.png/.tif/.npy/.npz`, auto-resize, strict/sequential index. Clean code.

---

## 3. Phân tích kết quả Evaluation

Từ `eval/summary.json` (6 video × 2 mode):

| Video | Mode | HIT | FP | Events | Latency (s) | FPS |
|-------|------|-----|----|--------|-------------|-----|
| video1 | no-feedback | ✅ | 1 | 2 | 0.6 | 10.7 |
| video1 | instance-feedback | ❌ | 0 | 0 | — | 8.0 |
| video2 | no-feedback | ✅ | 0 | 1 | 3.5 | 10.9 |
| video2 | instance-feedback | ✅ | 0 | 1 | 3.6 | 7.8 |
| video3 | no-feedback | ✅ | 0 | 1 | -12.0 | 10.8 |
| video3 | instance-feedback | ✅ | 0 | 1 | -9.8 | 7.7 |
| video7 | no-feedback | ✅ | 3 | 5 | -13.1 | 10.4 |
| video7 | instance-feedback | ✅ | 2 | 3 | -7.3 | 7.2 |
| video8 | no-feedback | ✅ | 4 | 5 | -5.5 | 0.9 |
| video8 | instance-feedback | ✅ | 4 | 5 | -5.5 | 7.6 |
| video11 | no-feedback | ✅ | 11 | 12 | -33.6 | 10.3 |
| video11 | instance-feedback | ✅ | 15 | 16 | -35.2 | 7.4 |

### Tổng hợp:

| Mode | HIT | Total Objects | FP tổng | Avg FPS |
|------|-----|---------------|---------|---------|
| **no-feedback** | 6/6 | 6 | **19** | 9.0 |
| **instance-feedback** | 5/6 | 6 | **21** | 7.6 |

> [!CAUTION]
> **instance-feedback MISS video1** (0 events) — RT-SBS feedback FORCE-FG lên vùng person → ViBE giữ person area as FG → khi person rời, ViBE vẫn cho FG → moving gate không tắt → vật không bao giờ chuyển sang static. **Feedback 1 chiều gây hại ở cảnh đơn giản.**
> **CẬP NHẬT (vibe-timeout, 2026-06-19)**: đã SỬA được — `--vibe-timeout 150` (≈5s) ép hấp thụ vùng FG-protect kẹt lại → balo nổi lên → v1 instance: 0ev/MISS → **1ev/HIT/0FP**. Nhưng cùng cơ chế làm v6/v11 tăng FP — xem §3.5.

### Key observations:

1. **Latency âm (< 0)**: 4/6 video có latency **âm** → hệ thống báo TRƯỚC `abandon_frame` GT. Nghĩa là `ts_static=5s` + pipeline nhanh hơn annotation. Đây là **cảnh-báo-sớm**, không phải lỗi.

2. **video8 no-feedback: 0.9 FPS** — anomaly rõ. Có thể do `pybgs` ViBe C++ initialization chậm hoặc video8 resolution/codec đặc biệt.

3. **video11 FP explosion**: 11–15 FP ở cảnh đông. Gốc: YOLO-nano bỏ sót người đứng im trong đám đông → static + not-animate → false abandoned.

4. **instance-feedback không hơn no-feedback**: FP cao hơn (21 vs 19), miss 1 video, chậm hơn 16%. **Feedback vào ViBE không giúp AOD, còn gây hại.**

---

## 3.5. Ảnh hưởng `--vibe-timeout` cho ControlledViBE (cập nhật 2026-06-19)

Bảng §3 ở trên là eval TRƯỚC khi thêm `--vibe-timeout` (mặc định **150f ≈ 5s**, chỉ áp ControlledViBE = instance/dense; `no-feedback` dùng pybgs nên KHÔNG đụng). Đo lại v1/v6/v11 với 3 cấu hình:

| Video | no-feedback (pybgs) | instance · timeout OFF | instance · timeout 150f |
|-------|---------------------|------------------------|-------------------------|
| **video1** (thưa) | HIT · 1 FP (2 ev) | ❌ MISS · 0 FP (0 ev) | ✅ HIT · 0 FP (1 ev) |
| **video6** (đổi sáng, ~ko người) | HIT 2/2 · 7 FP (9 ev) | HIT 2/2 · 1 FP (3 ev) | HIT 2/2 · 3 FP (5 ev) |
| **video11** (đông) | HIT · 11 FP (12 ev) | HIT · 15 FP (16 ev) | HIT · 18 FP (19 ev) |

**Cơ chế** — `static_fg = newdiff(clean_bg ĐÓNG BĂNG) AND NOT moving(raw_vibe)`: mask `moving` (ViBE) là thứ DUY NHẤT đè clutter sống-lâu khỏi `static_fg` (clean_bg đóng băng vĩnh viễn coi clutter là "khác"). `--vibe-timeout` ép hấp thụ FG→BG sau 150f = **bộ gia tốc hấp thụ KHÔNG chọn lọc**:

- **v1 (thưa)**: nhả đúng vùng balo bị feedback FG-protect → `static_fg` bật → **MISS→HIT, 0 FP. WIN.**
- **v6/v11 (nhiều clutter)**: nhả LUÔN clutter (đổi-sáng ở v6 / người-bóng đám đông ở v11) → lọt `static_fg` → **+FP** (v6 +2, v11 +3). Cảnh càng nhiều clutter, phạt FP càng lớn.

> [!IMPORTANT]
> **Đánh đổi recall ↔ precision**: cùng một nút (timeout) tăng recall (nhả vật thật → v1 HIT) thì giảm precision (nhả cả clutter → v6/v11 +FP), vì moving-gate **không phân biệt "vật cần nhả để bắt" với "clutter cần giữ đè"**. Post-timeout instance đạt recall đầy đủ (HIT mọi vật) nhưng FP tổng (21) > `no-feedback` (19) **cùng recall** → `no-feedback` vẫn là default tốt nhất. Fix CÓ CHỌN LỌC = timeout gated theo `animate` (chỉ nhả nơi `animate_prob` thấp — giữ người ở FG) → xem §5 khuyến nghị.

---

## 3.6. Phân tích ảnh FP video11 (thực địa, no-feedback) — xác định NGUỒN lỗi

Soi 11 ảnh `alert_*.jpg` (1 HIT cái ô f1189 + 11 FP):

| FP (frame · tâm) | Khung đỏ là gì | Loại |
|---|---|---|
| f670 ·496,186 | đốm xám trên sàn (vật nhỏ/phản chiếu — KHÔNG rõ người) | clutter mơ hồ |
| f695 ·342,366 | tile sàn tối, KHÔNG vật | sàn/bóng |
| f869 ·193,243 | **người + xe đẩy/túi** trong hàng (1 box gộp) | cụm owner-present |
| f929 ·341,218 | người đứng giữa sàn | người đứng |
| f1307 ·51,146 | vùng tối góc trái (tường/cửa) | bóng/vùng tối |
| f1309 ·604,178 | **vệt lóa sáng mép phải (cạnh shop)** | **lóa sáng** |
| f1394 ·90,246 | người đứng trong hàng | người đứng |
| f1932 ·154,233 | **người + xe đẩy/túi** trong hàng (1 box gộp) | cụm owner-present |
| f2539 ·375,254 | 2–3 người đứng nói chuyện | người đứng |
| f2791 ·402,149 | người đứng xa cuối sảnh | người đứng |
| f3629 ·162,278 | **người + xe đẩy/túi** trong hàng (1 box gộp) | cụm owner-present |

> [!IMPORTANT]
> Phân loại lại (soi kỹ + cơ chế merge): **~4/11 người đứng thuần** (f929,f1394,f2539,f2791 — person-recall trị được); **~3/11 cụm "owner-present"** (f869,f1932,f3629 = người + **xe đẩy/túi đặt dưới sàn**, bị `MORPH_CLOSE`+`gather_k` GỘP thành 1 box to — KHÔNG phải person-miss thuần; xe đẩy/túi là vật-tĩnh-THẬT nhưng **chủ đang đứng cạnh trong hàng** → đây là lỗ hổng **owner-leaves**, person-recall chỉ co nhỏ box chứ không xóa); **~3/11 lóa/sàn/bóng** (f1309,f695,f1307); **1/11 đốm mơ hồ** (f670). **0/11 = vật-bị-dời, 0/11 = ghost-warmup.** → box-cụm to là **artifact của over-merge** (nhiều vật-tĩnh liền kề + 2 lần CLOSE → 1 connected component → 1 bbox), không phải "1 vật".

**Hệ quả:**
- **Person-aware warmup** (loại người khỏi median warmup) cho **~0 lợi** trên v11 — đúng vì 0/11 là ghost (người v11 đi-ngang nên median đã tự loại; coverage mask 7.6% nhưng FP 11→11). **So kỹ baseline-OFF vs ON theo VỊ TRÍ**: net 11=11 nhưng KHÔNG cùng tập — nó gỡ **đúng 1 FP sàn (f695)** (chỗ có người warmup cạnh tile → clean_bg đổi) nhưng **khu xếp hàng bắn dư 1 lần** → bù trừ; **10/11 vùng lỗi gốc (người + lóa) y nguyên, chỉ jitter frame/dedup**. → xác nhận không trị nguồn FP thật. Feature đúng & an toàn, **default OFF**, để dành cảnh có người đứng-yên-suốt-warmup thật; KHÔNG bật trong preset.
- **SCENE_FEATURE_MEMORY — ĐÃ build + test thực nghiệm** (`core/scene_feature_memory.py`, 2 mode `relocated`/`background`, default OFF, unit 4 ca PASS). Kết quả v11 no-feedback: **`relocated` = no-op** (suppress 1 FP sàn trùng hợp, net 12ev/11FP — đúng vì 0 vật-dời). **`background` FP 11→6 NHƯNG suppress 35 candidate = XÓA TRẮNG khu xếp hàng** (gọi nhầm đám đông là "background_glare 1.00" do edge-hist 9-bin quá thô → đám đông ≈ nền). HIT (ô) sống **chỉ vì ô nằm NGOÀI đám đông** (309,270); **vật bỏ quên TRONG đám đông sẽ bị nuốt → UNSAFE**. → scene-memory không giải được "đám đông tan thì im, vật trong đám đông thì báo" (ở mức pixel/blob người-đứng ≡ vật). Chỉ `relocated` hợp lý cho camera có vật-nền-bị-xê-dịch — không phải ABODA.
- **Đòn hiệu quả thật cho v11** (theo ảnh + cơ chế): (1) **tăng recall người** (YOLO lớn/imgsz cao/RT-DETR GPU) → loại ~4 người đứng thuần + **co nhỏ** cụm owner-present (vỡ blob, chỉ còn xe đẩy/túi); (2) **owner-leaves reasoning** (track người + gắn người↔vật) → diệt 3 cụm owner-present — đây mới là gốc, person-recall không đủ; (3) **xử lý lóa/bóng** (relight/light-comp/stuff-reject) → gỡ ~3 clutter ánh sáng; (4) **ROI khu chờ** — rẻ, đặc thù camera cố định. ⚠️ Không "suppress theo vùng" (scene-memory background): box-cụm là khối GỘP, suppress cả box sẽ nuốt vật thật nằm trong.

### gather-px: đổi default 15 → 5 (2026-06)
Sweep gather trên v11: box to nhất **931px (g0) → 38 220px (g15) → 48 600px+MISS (g31)** — gather to gộp cả đám đông thành box khổng lồ, quá tay (g31) còn trôi tâm → MISS. Trên v8 (vali rộng) gather giúp **IoU 0.32→0.77** (phủ vật). Vì **gather cố định không thể đúng cả 2** (cần cho vali, hại cho đám đông) → chọn **default 5** làm trung gian: v8 vali IoU~0.74 (vẫn HIT), v11 chỉ còn **1 box vừa (~9 700px)** thay vì 4 box tới 38 000px — FP/HIT y nguyên. (gather = công cụ IoU cho vật rộng, lợi của nó là **cosmetic** — không đổi việc có HIT hay không; hại over-merge đám đông là thật → để nhỏ.)
- **Dedup**: đã thử thay center-distance bằng **IoU-dedup + detector-extent (B)** → **kết quả y hệt** trên v8 (mảnh nhỏ f3892: IoU 0.018<<0.3 nên không gộp; B im vì YOLO không nhận ra vali tối) → **REVERT về center-distance gốc** (bằng kết quả, nhanh + gọn hơn). Mảnh ghép đúng-mà-thiếu = **containment** (tâm-mới nằm-trong-box-đã-báo); chưa làm. Báo-lặp vật-rộng còn lại là ±1 noise (dedup-ngưỡng), gốc vẫn = perception/owner-leaves.

---

## 4. Các vấn đề chính (tổng hợp)

### 🔴 Nghiêm trọng

| # | Vấn đề | File | Impact |
|---|--------|------|--------|
| 1 | **instance-feedback miss video1** — FG-protect giữ person area quá lâu, vật không chuyển static | `semantic_feedback.py` L59 | Miss detection |
| 2 | **clean_bg leak khi ViBE flicker** — moving gate nhất thời True → static_fg False → clean_bg update tại pixel vật | `static_state.py` L276 | Vật dần bị absorb |
| 3 | **God-file 900 dòng** — untestable, unmaintainable | `run_rtsbs_aod.py` | Engineering debt |

### 🟡 Quan trọng

| # | Vấn đề | File | Impact |
|---|--------|------|--------|
| 4 | **Relight skip ALL frames** khi diverging trước hold threshold | `static_state.py` L136 | Miss events during lighting transition |
| 5 | **np.roll overhead** mỗi frame trên H×W array | `controlled_vibe.py` L151 | Performance (est. 5-15% of frame time) |
| 6 | **70+ CLI args** không validated chéo | `run_rtsbs_aod.py` | Config errors silent |
| 7 | **Video seek** dùng `CAP_PROP_POS_FRAMES` (unreliable) | `clean_bg_prior.py` L37 | Potential wrong warmup frames |
| 8 | **Greedy matching** không Hungarian | `static_matching.py` | Suboptimal in crowded scenes |

### 🟢 Minor

| # | Vấn đề | Impact |
|---|--------|--------|
| 9 | `read_frame()` mở/đóng VideoCapture mỗi lần | Slow init |
| 10 | `color_map` lưu full BGR frame mỗi semantic step | RAM waste |
| 11 | Hardcoded EMA 0.7/0.3 bbox smoothing | Inflexible |
| 12 | `CrowdEstimator` dùng blob count (không phải detection count) | Inaccurate density |

---

## 5. Nhận xét tổng thể

### Điểm mạnh
1. **Modular core**: 7 module tách rõ responsibility (ViBE, feedback, FSM, matcher, semantic LUT, bg prior, dense reader).
2. **Flexible mode system**: 3 preset + `custom` mode cover nhiều experimental axis.
3. **Prop1/2/3 innovations**: motion-to-static latch, person-hull punch, local owner-gate with timeout — ý tưởng tốt cho crowded scenes.
4. **Debug output phong phú**: 14 loại mask + semantic vis → dễ diagnose.
5. **RT-SBS faithfulness**: Decision table + color-reuse + semantic model update đúng paper.

### Điểm yếu
1. **Feedback RT-SBS KHÔNG giúp AOD**: instance-feedback miss video1, FP không giảm. Dense-feedback cần GPU (SegFormer ~2 FPS). **Kết luận: RT-SBS feedback là overhead không mang lại giá trị cho bài toán abandoned object.**
2. **Orchestrator quá phức tạp**: 900 dòng, 70+ params, logic alert lồng sâu → khó test, khó extend.
3. **YOLO-nano bottleneck ở cảnh đông**: Bỏ sót 25/33 người → FP cứng, không trị bằng logic.
4. **Thiếu unit test**: Không có test nào trong repo cho demov2.
5. **Config coupling**: Nhiều tham số tương tác ngầm (vd `tau_animate` × `dilate_animate` × `person_hull` × `person_overlap_max`).

### Khuyến nghị

> [!TIP]
> 1. **Bỏ feedback mode**: `no-feedback` là mode tốt nhất (HIT 6/6, FP thấp hơn, nhanh hơn). Giữ code feedback cho nghiên cứu nhưng default = off.
> 2. **Tách orchestrator**: Extract `AlertDecisionEngine` class chứa logic dedup + owner-gate + person-hull + bbox-refine (hiện 80 dòng inline).
> 3. **Fix clean_bg leak**: Thêm `persist_protect` vào update condition, hoặc dùng hysteresis (pixel phải moving liên tục N frame mới cho update).
> 4. **Config profiles**: Thay 70+ args bằng YAML config files (`aboda_sparse.yaml`, `aboda_crowded.yaml`, ...).
> 5. **GPU person detector**: FP cảnh đông là bottleneck cứng → cần YOLO-L hoặc RT-DETR trên GPU cho person segmentation.
> 6. **`--vibe-timeout` có chọn lọc**: timeout hiện nhả FG mù (sửa recall v1 nhưng +FP v6/v11, xem §3.5). Gate theo `animate`: chỉ ép hấp thụ nơi `animate_prob` THẤP (giữ người ở FG, vẫn nhả vật) → giữ được v1 mà không kéo clutter người ở cảnh đông. Truyền `animate16` (runner đã có) vào `ControlledViBE.update()` làm no-absorb mask.
