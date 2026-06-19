# So sánh demov1 vs demov2 — Kiến trúc, Logic & Lịch sử kết quả (vật bỏ quên / ABODA)

> Cập nhật: **2026-06-17**. File này gộp đầy đủ kiến trúc + logic của **demov1** (RT-SBS-v2c) và
> **demov2** (RT-SBS faithful + FSM, 2 biến thể semantic), cùng **lịch sử kết quả chạy**.
> Bài toán: phát hiện đồ bỏ quên, CPU-only, realtime, video-agnostic. Dataset: ABODA (GT = `ABODA/aboda_gt.json`).

---

## 0. Tóm tắt 1 trang

| | **demov1 (RT-SBS-v2c)** | **demov2 (RT-SBS faithful + FSM)** |
|---|---|---|
| Triết lý | THAY lõi BGS bằng frame-diff nền-sạch; semantic chỉ trừ người | Giữ ĐÚNG vòng RT-SBS gốc (ViBe + decision table + dense semantic), gắn nhánh AOD |
| BGS lõi | `StaticDiffBG` (clean_bg cố định) + ViBe chỉ làm tầng moving | `ControlledViBE` (port RT-SBS) là lõi + clean_bg AOD riêng |
| Semantic | YOLO-seg **thưa 1/10f, chỉ trừ NGƯỜI** | **SegFormer dense** (đúng RT-SBS) **HOẶC** YOLO-seg (thay sau) |
| Đã validate | **4 video** (7,8,3,11) — sáng/tối + thưa/đông | **chỉ video11** (cảnh đông) |
| Bắt vật | ✅ cả 4 (balo/vali/ô), 1 báo/vật | ✅ ô (sau khi thêm FSM) |
| FP (video11) | **12** | 59–70 |
| FPS CPU | ~7–9 (realtime) | 12 (precompute semantic) / **~2 (SegFormer online — nghẽn)** |
| Trạng thái | **Pipeline production** | Dòng nghiên cứu "trung thành RT-SBS" |

**Kết luận:** `demov1` phù hợp triển khai hơn (đủ 4 cảnh, FP thấp, realtime, 1 báo/vật). `demov2` trung thành RT-SBS gốc nhất và **đã bắt được ô** sau khi vá tầng matcher (FSM), nhưng FP cao, mới chỉ video11, và SegFormer online quá nặng CPU. Bài học xuyên suốt: **RT-SBS tối ưu segment foreground-ĐỘNG, còn AOD cần giữ vật-TĨNH** — hai mục tiêu lệch nhau, nên demov1 (clean_bg cố định + matcher AOD) thắng về chất lượng phát hiện vật bỏ quên.

---

## 1. demov1 — Kiến trúc & Logic (RT-SBS-v2c)

**File:** `demov1/run_rtsbs_v2c.py` + `demov1/core/static_diff_bg.py`. **Doc gốc:** `demov1/BAOCAO_PIPELINE_RTSBS_V2C.md`.

**Nguyên tắc:** vật bỏ quên = *bất kỳ thứ gì KHÁC NỀN + TĨNH ≥5s + KHÔNG-người*. Không định nghĩa vật bằng lớp/hình dạng.

### Pipeline mỗi frame
```
frame ─► StaticDiffBG.apply():
          clean_bg (median warmup 25s, CỐ ĐỊNH)
          newdiff   = |gray − clean_bg| ≥ th_diff(40)        ← vật khác nền (giữ vật tĩnh, không bị nuốt)
          moving    = ViBe(frame)                            ← đang chuyển động (ViBe phủ TOÀN THÂN người)
          stat_new  = newdiff AND NOT moving                 ← vật MỚI đã ĐỨNG YÊN  ◄ output chính
     ─► YOLO-seg (thưa 1/10f) ─► person mask ─► fg = stat_new AND NOT dilate(person)
     ─► StaticMatcher (gom blob→candidate theo IoU+dist) ─► owner-gate ─► EVENT + keyframe
```

### 6 cơ chế lõi (đã thêm/validate)
1. **`StaticDiffBG` + clean_bg cố định** — thay MOG2 adaptive (vốn HỌC vật tĩnh vào nền → mù). Nền-sạch cố định → vật mới luôn lộ.
2. **ViBe-moving (mặc định)** — tầng `moving` dùng pybgs ViBe (auto-tìm `config/ViBe.xml` qua chdir nội bộ → chạy từ thư mục nào cũng được). **ViBe phủ toàn-thân người-động sạch hơn MOG2 → FP ≈ 1/2 MOG2** (đo trực tiếp).
3. **`relight` — nền-sạch-theo-CHẾ-ĐỘ-SÁNG** — dò bước nhảy toàn cục (|Δ meanV|>30 HOẶC |Δ meanS|>12, giữ ≥15f) → **dựng lại clean_bg chế độ mới**, NGƯNG báo suốt ramp. Trị camera đổi ngày↔đêm (video7 tối→sáng, video8 sáng→tối IR). Không dùng chromaticity → đúng cả pha màu lẫn đen-trắng.
4. **`heal-alpha` THÍCH NGHI theo chế độ** — pha TỐI (ref_S<15, IR) heal **mềm 0.05** (không nuốt vật mong manh); pha SÁNG heal **mạnh 1.0** (hấp thụ FP bóng). *Hóa giải xung đột video7(sáng)↔video8(tối).*
5. **`persist-protect`** — vùng khác-nền **bền bỉ ≥persist_s(1s)** → CHẶN slow-update/heal hấp thụ; tự nhả khi về nền. Chống "vật bị slow-update ăn dần khi chủ đứng cạnh lâu".
6. **owner-gate PRESENCE** (chỉ cảnh THƯA) — chờ tầm-tay vật VẮNG người ≥3s mới báo (định nghĩa abandoned kinh điển); dùng density YOLO ổn định (KHÔNG churn/moving_frac — chúng hiểu nhầm 1-chủ-cử-động là đông). + **`gather`** (close tight_mask) cho bbox sát vật + **dedup refine-trước-dedup** (1 báo/vật).

**Cấu hình chốt:** `--moving-backend vibe --relight 1 --owner-gate 1 --owner-clear-s 3 --gather-px 15 --persist-s 1.0 --heal-alpha 1.0 --heal-alpha-dark 0.05 --bg-learn-seconds 25 --th-diff 40 --seg-imgsz 1280`.

---

## 2. demov2 — Kiến trúc & Logic (RT-SBS faithful + FSM)

**File:** `demov2/run_rtsbs_aod.py` + `core/{controlled_vibe, semantic_feedback, dense_semantic, static_state, clean_bg_prior, static_matching, semantic_lut}.py`.

**Nguyên tắc:** giữ ĐÚNG vòng điều khiển RT-SBS gốc (ICIP2020), gắn thêm nhánh phát hiện vật bỏ quên.

### Pipeline mỗi frame
```
frame ─► [1] ControlledViBE.segmentation() ─► raw_vibe   (port CPU của ViBeGPU.py: 30 samples, thr10,
          color×4.5, match2, update8; ĐÃ vector hóa OpenCV → giữ BIT-IDENTICAL, 2.7→12 FPS)
     ─► [2] SemanticFeedback (LUẬT RT-SBS):
              có map mới : rule_BG = sem ≤ tau_BG ; rule_FG = sem − model ≥ tau_FG×256
              frame giữa : tái dùng rule cũ nơi color_diff ≤ tau_BG*/tau_FG*  (decision table B/S/C → D_t)
           ─► rtsbs_mask (foreground-ĐỘNG đã semantic-correct)
     ─► [3] vibe.update(frame, rtsbs_mask)                  ← VÒNG PHẢN HỒI RT-SBS gốc
     ─► [4] NHÁNH AOD:
              stat_new = |gray − clean_bg| ≥ th_diff  AND NOT moving   (clean_bg = median warmup riêng)
              StaticForegroundState (FSM tuổi-tĩnh per-pixel):
                 moving FG = rtsbs bật ; static FG = khác clean_bg & motion tắt
                 ABANDONED = static_age ≥ T  AND  animate_prob < tau_animate(0.25)  [hoặc object_prob ≥ tau_object]
              ─► StaticMatcher ─► EVENT
```

### Điểm khác cốt lõi so demov1
- **Có decision table RT-SBS đầy đủ** (rule_BG/rule_FG + color-reuse tau*) — demov1 KHÔNG có.
- **Có feedback semantic vào ViBe** (`vibe.update(rtsbs_mask)`) mỗi frame — demov1 KHÔNG.
- **FSM `StaticForegroundState`** (thêm sau để vá miss-ô): tuổi-tĩnh per-pixel + cổng semantic `animate_prob` (người/xe = animate cao → loại; vật = animate thấp → giữ). Đây là cái giúp demov2 **bắt được ô** (trước đó miss tại tầng matcher, KHÔNG phải tại BGS).

### 2 biến thể semantic của demov2
| Biến thể | Mô tả | Tốc độ |
|---|---|---|
| **SegFormer dense** (ban đầu — đúng RT-SBS) | dense scalar map mỗi-N-frame; ADE20K-150. **Lưu ý: ADE20K KHÔNG có lớp umbrella/handbag** → ô chỉ giữ bằng PHẦN BÙ (animate thấp), không có object-class | online ~2 FPS (NGHẼN CPU), precompute 12 FPS |
| **YOLO-seg** (thay sau + đổi logic) | thay SegFormer bằng YOLO-seg cho tín hiệu vật | nhanh hơn (không nghẽn SegFormer) |

---

## 3. So sánh kiến trúc

| Tiêu chí | demov1 v2c | demov2 (FSM) | RT-SBS gốc |
|---|---|---|---|
| Mục tiêu | phát hiện vật bỏ quên | RT-SBS core + AOD extension | background subtraction |
| Nền chính AOD | `clean_bg` warmup cố định | `clean_bg` warmup + suppression bằng rtsbs_mask | không có clean-bg AOD |
| BGS lõi | StaticDiffBG; ViBe chỉ làm short-motion | ControlledViBE là lõi RT-SBS | ViBe GPU |
| Semantic | YOLO-seg person mask thưa | SegFormer dense / YOLO-seg | PSPNet dense 16-bit |
| Decision table RT-SBS | ❌ không | ✅ có | ✅ có |
| Color reuse (tau*) | ❌ | ✅ | ✅ |
| Feedback vào ViBe | ❌ | ✅ `vibe.update(rtsbs_mask)` | ✅ |
| Giữ vật TĨNH | clean_bg cố định + persist-protect | clean_bg + FSM tuổi-tĩnh | (không phải mục tiêu) |
| Đổi sáng ngày/đêm | ✅ relight + heal-adaptive | ❌ chưa xử | — |
| dedup (1 báo/vật) | ✅ occupancy + refine-trước-dedup | ❌ (→ ô bị lặp ở bản YOLO-seg) | — |

---

## 4. LỊCH SỬ KẾT QUẢ CHẠY

### 4.1. demov1 — tiến hóa & kết quả CHỐT (ViBe)

Eval bằng `demov1/tools/eval_vs_gt.py` vs `ABODA/aboda_gt.json` (output `outputs_v*_vibe2`):

| video | Cảnh | Bắt vật? | Δcenter | IoU box | FP (ViBe) | FP (MOG2) | FPS |
|---|---|---|---|---|---|---|---|
| **video7** | tối→sáng, thưa | ✅ balo, 1 báo | 2px | 0.71 | **3** | 9 | 9.1 |
| **video8** | sáng→tối IR, vali 208×67 | ✅ vali, 1 báo | 2px | 0.77 | **4** | 9 | 8.8 |
| **video3** | balo, thưa | ✅ balo, 1 báo | 13px | 0.49 | **0** | 2 | 9.1 |
| **video11** | ô treo, ĐÔNG | ✅ ô, 1 báo | 1px | 0.31 | **12** | 26 | 7.0 |

Mốc tiến hóa (đo thực, đừng lặp lại): MOG2-only (nuốt ô) → StaticDiffBG (cứu ô) → +relight (video7 FP23→9) → +heal-thích-nghi (giải video8 vali IoU0.02→0.78) → +persist + owner-gate-presence (video7/3 timing) → +dedup-refine-trước (video8 ô-vali 3→1 báo, video11 32→26) → **đổi default ViBe (FP ≈ 1/2 MOG2)**. Đối soát thời gian: tất cả báo SỚM 6–34s so `abandon_frame` GT (cảnh-báo-sớm: trigger "tĩnh 5s + chủ rời tầm-tay" sớm hơn mốc "bỏ quên chính thức" của GT).

### 4.2. demov2 — tiến hóa semantic & kết quả (video11, ô GT≈(309,271))

| Giai đoạn | Output | Tổng events | **Ô THẬT bắt** (tâm sát ≤14px + box nhỏ) | FP/clutter | Ghi chú |
|---|---|---|---|---|---|
| **SegFormer-B0 batch** (cũ, chưa FSM) | `outputs_aboda_demo2/video11_segformer_b0` | **175** | có ô (f913 ~1px) nhưng lẫn FP bùng | ~174 | precision rất kém; SegFormer-ADE20K FG chỉ 1.49% ở cảnh đông |
| **SegFormer online + FSM** | `outputs_v11_online_fsm` | 70 | **1 lần** (f3220, t≈107s — MUỘN) | 69 | FSM cứu ô nhưng SegFormer chỉ bắt 1 lần & muộn; online ~2 FPS (nghẽn) |
| **YOLO-seg + FSM (đổi logic)** | `outputs_v11_yoloseg` | 67 | **5 lần** (f1142,2769,2959,3269,3729 — đều c[309,270] wh13×17) | 62 | YOLO-seg bắt ô **SỚM (t38s) + ổn định hơn** SegFormer, NHƯNG **LẶP 5 lần** (thiếu dedup-occupancy như demov1) |
| FSM precompute (default / tuned) | `outputs_v11_aod_fsm` / `_tuned` | 70 / **59** | có ô | 69 / ~52 | tuned th_diff=50,tstatic=2 |

**Nhận xét demov2:**
- **SegFormer (đúng RT-SBS) KHÔNG hơn YOLO-seg** trên ABODA: SegFormer bắt ô **1 lần & muộn (t107)**; YOLO-seg bắt ô **sớm (t38) + 5 lần**, nhanh hơn nhiều (không nghẽn SegFormer). → ADE20K không có lớp vật-bỏ-quên (umbrella/handbag) nên dense semantic không cho tín hiệu hữu ích cảnh ABODA. **Khớp kết luận demov1: dense semantic không đáng giá ở domain này.**
- **YOLO-seg bị LẶP ô 5 lần** (cùng [309,270] suốt ~88s) vì demov2 **thiếu dedup-theo-occupancy** mà demov1 có. → cùng 1 ô đứng yên, matcher tái tạo candidate → bắn lại nhiều lần.
- FP demov2 (59–70) ≫ demov1 (12) trên cùng video11: FP demov2 là **clutter phi-người** (bóng/phản chiếu/vật-tĩnh-khác-nền-20s-đông-cứng, animate≈0 nên semantic gate không loại được); matcher 5s là chốt chặn duy nhất (thiếu owner-gate/ViBe-person-subtract/dedup của demov1).

### 4.3. Tốc độ
- demov1: ~7–9 FPS CPU (realtime), mọi cảnh.
- demov2: dense precompute **12 FPS** (sau khi vector hóa ControlledViBE, giữ bit-identical); **online SegFormer-b0 ~2 FPS** (nghẽn ở model semantic — cần GPU/precompute). YOLO-seg nhanh hơn SegFormer-online.

---

## 5. Phân tích FP hiện tại

**demov1 (ViBe `outputs_v*_vibe2`) — 3 kiểu gốc:**
- ① **Người-đứng YOLO bỏ sót** (video11 toàn bộ 12 FP = cụm queue + nhóm đứng; video7 chân-người). Gốc: YOLO mù ~25/33 người đông → **cần GPU**.
- ② **Artifact ánh sáng bề mặt** (video7 phản chiếu tường/rèm pha-sáng; video8 đốm bàn lúc chuyển đèn).
- ③ **Nhiễu pha tối IR** (video8 low-SNR + phản chiếu sàn).
- video3 = **0 FP**.

**demov2 (video11):** FP là **clutter phi-người tĩnh** (khác clean_bg-20s-đông) + (bản YOLO-seg) ô bị lặp. Cùng gốc đám đông + thiếu verification object-level (dedup/owner-gate/scene-noise) mà demov1 đã có.

→ Cả hai chung nút thắt cảnh đông: **YOLO mù người đông trên CPU** — giới hạn cứng cần GPU/person-seg mạnh + tracker đám đông. Không phải lỗi logic.

---

## 6. RT-SBS vs AOD — vì sao lệch (phân tích nền tảng, vẫn đúng)

- **RT-SBS tối ưu segment foreground-ĐỘNG** + update background tốt. **AOD cần giữ vật-MỚI-TĨNH** không bị học vào nền/không bị semantic BG xóa. Hai mục tiêu ngược nhau.
- Trong RT-SBS, dòng L6 decision table: `B=FG, S=BG, No-Change → D=BG` (xóa ghost) — tốt cho BGS, nhưng với AOD "No-Change lâu" lại CHÍNH LÀ dấu hiệu vật bỏ lại → có thể làm mất vùng vật tĩnh.
- `rtsbs_mask (D_t)` là *foreground-đã-semantic-correct*, KHÔNG phải *mask-đang-chuyển-động* → dùng nó làm moving-gate AOD (`stat_new = clean_bg_diff AND NOT rtsbs_mask`) là ghép sai chức năng (nguồn FP demov2 bản đầu).
- demov1 né bằng: **clean_bg cố định** (giữ vật) + **short-motion riêng (ViBe)** + **không feedback semantic vào BGS** + **matcher object-level**.

---

## 7. Kết luận & hướng

**Chốt:** dùng **demov1 (RT-SBS-v2c + ViBe)** cho triển khai — đủ 4 cảnh (sáng/tối, thưa/đông), FP thấp (0–12), realtime, 1 báo/vật. **demov2** giữ làm dòng nghiên cứu RT-SBS trung thành; đã chứng minh **bắt được ô nhờ FSM** (miss trước đó là tại matcher, không phải BGS), nhưng:
- SegFormer online quá nặng CPU (~2 FPS) + ADE20K không hợp domain → **YOLO-seg thực dụng hơn** (đã xác nhận: bắt ô sớm hơn, nhanh hơn).
- Cần port các verification của demov1 sang: **dedup-occupancy** (chống lặp ô), owner-gate, scene-noise.

**Giới hạn cứng chung:** FP cảnh đông (video11) = người-đứng YOLO bỏ sót trên CPU → cần GPU/person-seg mạnh + tracker đám đông (BoT-SORT-ReID). Logic-level không trị hết FP rải.

**Nếu làm tiếp:** giữ AOD-core demov1, mượn RT-SBS ở vai trò PHỤ (protect/update background, KHÔNG dùng D_t làm moving-gate); thêm scene-noise memory + motion-to-static evidence cho cảnh đông.
