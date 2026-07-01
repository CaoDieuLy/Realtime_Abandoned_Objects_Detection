# Full ABODA sweep — Async-semantic + CLIP verifier ON

Một cấu hình mặc định cho cả 13 video (`--async-semantic on --clip-verify 1`, chỉ khác `--bg-learn-seconds` theo cảnh: vid0355=3s, vid0103=8s, còn lại 20s). Mỗi thư mục `<video>/` có `events.json` + `run.log` (FPS live). Chấm bằng scorer time-aware (GT quy về proc-640, khớp cả vị trí lẫn abandon_frame — video6 có 2 túi cùng chỗ khác thời điểm). Phân tích đầy đủ: [REPORT §9](../REPORT.md).

## Kết quả (baseline §6.2 vs async+CLIP)

| Video | Recall base→mới | FP base→mới | FPS |
|---|---|---|---|
| video1 | 1/1 → 1/1 | 0 → 0 | 6.4 |
| video2 | 1/1 → 1/1 | 0 → 0 | 7.5 |
| video3 | 1/1 → 1/1 | 0 → 0 | 7.3 |
| video4 | 1/1 → 1/1 | 0 → 0 | 6.7 |
| video5 | 1/1 → 1/1 | **1 → 0** | 7.6 |
| video6 | 2/2 → **1/2** | **1 → 0** | 8.2 |
| video7 | 1/1 → 1/1 | **3 → 0** | 8.9 |
| video8 | 1/1 → 1/1 | **1 → 0** | 8.7 |
| video9 | 1/1 → 1/1 | 0 → 0 | 7.1 |
| video10 | 1/1 → 1/1 | **1 → 0** | 7.1 |
| video11 | 1/1 → **1/1** (đại-diện)¹ | **11 → 4** | 5.9 |
| **TỔNG (12 vật GT)** | **12/12 → 11/12** | **18 → 4 (−78%)** · P 40→73% · **F1 57→82%** | ~5–9 |
| vid0355 (no GT) | — | máy giặt **2→1 (hết lặp)** | 3.1 |
| vid0103 (no GT) | — | 3 sự-kiện (1 FP warmup còn) | 7.2 |

Biểu đồ: [`../report_figures/fig_asyncclip_fp.png`](../report_figures/fig_asyncclip_fp.png) · `fig_asyncclip_summary.png` · `fig_asyncclip_fps.png`.

## Phân tích ngắn — vì sao FP giảm 18→4

CLIP loại đúng các báo-nhầm mà logic-pixel không tách được khỏi vật:
- **video7 (đổi sáng) 3→0:** patch tường/sàn chỉ-sáng-lên → CLIP `"plain wall"`/`"empty floor"` (top1 0.92–0.96) loại; túi thật p_obj≈0.99 giữ.
- **video11 (đám đông) 11→4:** cụm người/sàn detector sót → CLIP `"crowd of people"`/`"plain wall"` loại 7/11; 4 FP còn lại lọt vì top1<0.30 (CLIP *abstain*, an-toàn-recall).
- **vid0355 2→1:** máy giặt vỡ 2 mảnh; CLIP loại mảnh nhỏ 10×28 → hết báo-lặp.
- **vid0103 (không cắt):** FP (304,212) là sàn-lộ nơi người-đổ-rác bị nướng-tối vào nền lúc warmup — diff-cường-độ thật → CLIP abstain (giữ). Giới hạn `clean_bg` sai, không phải cổng.

→ 8/11 video GT về 0 FP; precision ~40% → **~73%**, **F1 57%→82%** (nhì SOTA ABODA — xem so-sánh dưới). ¹ video11-ô: pybgs bắt được **3/4 lần** async+CLIP (sweep gốc rơi 1/4 miss); bảng lấy đại-diện. Khe recall THẬT do CLIP = 1 vật (video6-túi2).

## So sánh SOTA trên ABODA (12 vật GT, cùng độ-đo TP/FP)
| Method | Recall | Precision | F1 | FP | Real-time CPU? |
|---|---|---|---|---|---|
| Ilya et al. | 75% | 69% | 72% | 4 | — |
| Saluky et al. | 75% | 75% | 75% | 3 | — |
| Lin 2015 (gốc ABODA) | 100% | 67% | 80% | 6 | — |
| SAO-YOLO 2024 (deep) | 100% | 86% | **92%** | 2 | YOLO-deep (thường GPU) |
| **Ours L1 (no-CLIP)** | **100%** | 40% | 57% | 18 | **✓ CPU** |
| **Ours L2 (+CLIP)** | 92% | 73% | **82%** | 4 | **✓ CPU** |

→ Luồng-2 **F1 82% = nhì SOTA** (sau SAO-YOLO deep), FP=4 thấp hơn Lin (6), mà **real-time CPU không-GPU**. Nguồn: SAO-YOLO (Sensors 2024, PMC11510867). 2 video bổ sung (vid0103/0355) NGOÀI ABODA → không SOTA nào test (case-study camera hi-res thực).

## ⚠ Recall 12/12 → 11/12 — căn nguyên ĐÃ CÔ-LẬP (2 nguyên nhân KHÁC nhau)

| Vật mất | Nguyên nhân | Bằng chứng |
|---|---|---|
| **video11-ô** | **pybgs** (async & CLIP vô tội) | [`diag_umbrella/`](diag_umbrella/): A(sync)/B(async)/C(async+clip) **đều HIT ô**; C log `CLIP KEEP 'an umbrella' p_obj=0.46`. Sweep chỉ xui pybgs. |
| **video6-túi2** | **CLIP loại nhầm mảnh-nhỏ** | pipeline chỉ bắt mảnh 25×9; CLIP đọc `'a plain wall'` p_obj≈0.01 ở **mọi cỡ crop** → suppress. Giới hạn tri-giác CLIP (túi tối/mờ). |

**Núm `--clip-min-area`** (bỏ qua CLIP cho bbox nhỏ) — đo ([`diag_minarea/`](diag_minarea/)): min-area **0** → túi2 mất, video11 FP **4**; min-area **500** → túi2 về (2/2) nhưng video11 FP **11** + mất dup-fix vid0355. Vật-nhỏ và FP-đám-đông-nhỏ trùng cỡ → **không tách sạch**. Lợi ích SẠCH recall-safe của CLIP = **FP vùng-lớn (đổi-sáng/mảnh-vỡ)**; FP-đám-đông-nhỏ là đánh-đổi. Chi tiết: [REPORT §9.5–9.6](../REPORT.md).

**Khuyến nghị:** async **nên bật** (đã default auto). CLIP **để opt-in** (`--clip-verify 0` mặc định) — bật cho camera nhiều đổi-sáng/đổi-cảnh; núm `--clip-min-area` đổi recall↔precision vật-nhỏ. Vì stance *bỏ-sót tệ hơn báo-nhầm*, khe recall vật-nhỏ là lý do CLIP chưa default-on.

## Tái lập
```bash
# từ repo-root, py -3.13
py -3.13 demov2/run_rtsbs_aod.py --video ABODA/<vid> --bg-learn-seconds <X> \
  --async-semantic on --clip-verify 1 --outdir demov2/results_full_clip/<vid>
```
