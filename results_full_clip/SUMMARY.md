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
| video11 | 1/1 → **0/1** | **11 → 4** | 5.9 |
| **TỔNG (12 vật GT)** | **12/12 → 10/12** | **18 → 4 (−78%)** | ~5–9 |
| vid0355 (no GT) | — | máy giặt **2→1 (hết lặp)** | 3.1 |
| vid0103 (no GT) | — | 3 sự-kiện (1 FP warmup còn) | 7.2 |

Biểu đồ: [`../report_figures/fig_asyncclip_fp.png`](../report_figures/fig_asyncclip_fp.png) · `fig_asyncclip_summary.png` · `fig_asyncclip_fps.png`.

## Phân tích ngắn — vì sao FP giảm 18→4

CLIP loại đúng các báo-nhầm mà logic-pixel không tách được khỏi vật:
- **video7 (đổi sáng) 3→0:** patch tường/sàn chỉ-sáng-lên → CLIP `"plain wall"`/`"empty floor"` (top1 0.92–0.96) loại; túi thật p_obj≈0.99 giữ.
- **video11 (đám đông) 11→4:** cụm người/sàn detector sót → CLIP `"crowd of people"`/`"plain wall"` loại 7/11; 4 FP còn lại lọt vì top1<0.30 (CLIP *abstain*, an-toàn-recall).
- **vid0355 2→1:** máy giặt vỡ 2 mảnh; CLIP loại mảnh nhỏ 10×28 → hết báo-lặp.
- **vid0103 (không cắt):** FP (304,212) là sàn-lộ nơi người-đổ-rác bị nướng-tối vào nền lúc warmup — diff-cường-độ thật → CLIP abstain (giữ). Giới hạn `clean_bg` sai, không phải cổng.

→ 8/11 video GT về 0 FP; precision (11 GT) ~40% → **~71%**.

## ⚠ Recall tụt 12/12 → 10/12 — ĐANG ĐIỀU TRA

Mất **video6-túi2** (@f5679) và **video11-ô**. Thí nghiệm cô-lập đang chạy: [`diag_umbrella/`](diag_umbrella/) — A(sync,no-clip)/B(async,no-clip)/C(async,clip) ×2 trên video11 để tách nguyên nhân **async** (YOLO dày hơn → mask người reset `static_age` của vật) vs **CLIP** (loại nhầm crop mảnh-nhỏ) vs **pybgs** (không tất định). Xem `diag_umbrella/_diag.log`. Kết luận + đề xuất sửa sẽ cập nhật sau.

Vì stance **"bỏ-sót tệ hơn báo-nhầm"**, cả hai cơ chế để **opt-in** (`--clip-verify 0`, `--async-semantic off`) cho tới khi khép được khe recall.

## Tái lập
```bash
# từ repo-root, py -3.13
py -3.13 demov2/run_rtsbs_aod.py --video ABODA/<vid> --bg-learn-seconds <X> \
  --async-semantic on --clip-verify 1 --outdir demov2/results_full_clip/<vid>
```
