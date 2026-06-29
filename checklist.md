# Checklist — việc còn lại / đang chờ test / đề xuất

> Quy ước test: **mỗi thay đổi test ≥3 video đang có vấn đề; OK mới giữ + làm tiếp.** HIT phải giữ (không miss vật).
> Trạng thái: ✅ xong · 🔄 đang test · ⏳ chưa làm · ⏸️ hoãn (cần model/lớn) · ❌ pipeline CHƯA giải được.

---

## 0. ❌ GIỚI HẠN CỐ HỮU — pipeline HIỆN TẠI KHÔNG giải quyết được

### ❌ Thay đổi-trạng-thái-CẢNH ở RUNTIME (vd MỞ CỬA) → BÁO NHẦM
- **Triệu chứng**: cửa đóng lúc warmup (baked vào clean_bg), sau đó **MỞ + đứng yên** → vùng cửa khác clean_bg + static + **không-phải-người** → tích tuổi → **báo nhầm sau ~5s**. Cùng họ: ghế kéo ra, nội thất xê dịch, poster gỡ xuống, đèn-cố-định bật, thùng rác di dời.
- **Gốc**: pipeline chỉ biết *"khác nền sạch + đứng yên + không-người/xe = nghi đồ bỏ quên"* → **KHÔNG phân biệt "cảnh đổi trạng thái" với "vật bị để lại"**.
- **Vì sao các lớp hiện có KHÔNG cứu**: relight (chỉ global) · light-comp (cần >15% coverage) · **light-struct** (mở cửa LỘ cảnh sau lưng = **đổi CẤU TRÚC**, NCC thấp → không suppress, đúng vì cũng có thể là vật) · heal-revealed (cửa ≠ agent). clean_bg slow-update rốt cuộc hấp thụ nhưng **quá chậm (~phút)** → báo trước.
- **Ngoại lệ**: cửa RẤT TO (blob > `area-max` 30000) → bị lọc kích thước.
- **Hướng xử (đều đánh đổi, CHƯA làm)**: (a) `--stuff-reject` mode dense + model tốt (gán cửa/tường=stuff→reject) · (b) **ROI-mask** vùng cửa (per-deployment, an toàn nhất) · (c) scene-change classifier (khó) · (d) chấp nhận + người vận hành bỏ qua.
- ⚠️ **Phân biệt với warmup-baked**: nếu cửa/người ĐỘNG trong WARMUP → baked → **B (motion-warmup) trị được** (xem §1). Còn cửa mở ở RUNTIME (sau warmup) thì **chưa có cách**.

### ❌ Warmup bị NHIỄM — nền KHÔNG bao giờ sạch lúc học (vid0103 FP 302,212)
- **Triệu chứng**: trong warmup có **người/vật hiện-diện gần-như-bất-động SUỐT** (vid0103: người đổ/bới rác đứng tại (302,212) cả 8s) → median **bake họ thành vùng TỐI** vào clean_bg → khi họ rời đi, frame sáng hơn nền-baked-tối → **diff dương → FP** (truy vết: clean_bg vệt tối lưỡi liềm, frame +30, newdiff 42.5%, support cao nên C giữ đúng).
- **Vì sao B KHÔNG cứu**: motion-mask loại framediff>20, nhưng người ĐỨNG YÊN (cúi) → framediff<20 → không loại (vid0103 chỉ 2.4% loại) + chiếm >50% mẫu → median vẫn tối. **B chỉ trị transient DI CHUYỂN NHANH** (cửa/người-đi-ngang như f780), KHÔNG trị người-bất-động-suốt-warmup.
- **Vì sao heal KHÔNG cứu**: heal-revealed YOLO chỉ bắt agent rõ (xe vid0103 10.7%), MÙ với bóng/người-cúi-tối-nhập-nhằng.
- **Hướng xử (đánh đổi, CHƯA làm)**: (a) **`--clean-bg-image`** nạp ảnh nền trống chụp sẵn (= P2c, tốt nhất cho DEPLOYMENT) · (b) chọn cửa-sổ-warmup lúc cảnh trống · (c) chấp nhận.
- **→ QUYẾT (2026-06): CHẤP NHẬN** — vid0103 KHÔNG có frame sạch trong video (người đổ rác từ giây 0). Ghi nhận là FP-đã-biết của warmup-nhiễm. `--clean-bg-image` để dành cho deployment thật (camera có lúc cảnh trống).
- ⚠️ KHÁC f780: f780 = người/cửa ĐỘNG lúc warmup → B trị. vid0103 = người BẤT ĐỘNG suốt → B bó tay. Cùng họ "warmup-baked" nhưng B chỉ cứu vế ĐỘNG.

### ❌ Khác (cần model/đổi backend)
- **Perception ceiling**: leg-FP, FP đám đông, vật-nhỏ-mảnh ≡ người (yolo26s sót/nhầm) → cần model to/open-vocab (xem §3).
- **pybgs non-deterministic**: FP dao động ±1 giữa các lần chạy (no-feedback dùng pybgs C++ không seed).

---

## 1. ✅ ĐÃ XONG (mới)

### P1a — dedup cooldown (`--dedup-cooldown-s` 30s)
- Entry `[cx,cy,last_seen,alert_frame]`; giữ entry trong cooldown bất kể cand-churn/occupancy. **Test: video9 2→1 (diệt double), video6 HIT 2/2, video11 no-regress.**

### P1b — `--light-struct` (đổi-sáng-cục-bộ, NCC tại lúc alert) — GIỮ default ON
- `is_lighting_artifact()`: patch vs clean_bg → nền CÓ VÂN (std≥8) + **NCC≥0.85** (cùng cấu trúc, chỉ đổi sáng) → bỏ alert. Nền phẳng/NCC thấp → KHÔNG bỏ (an toàn).
- **Test: video6 FP góc cầu thang BỊ DIỆT (NCC 0.96), HIT 2/2; video7 4→4 (no regress); video8 HIT Y (vật to không bị nuốt).** → an toàn 0 mất recall.
- **Giới hạn (đã phân tích)**: chỉ bắt đổi-sáng-THUẦN trên nền vân. **Reflection trên KÍNH + brightening không-đều (NCC<0.85)** → cấu trúc thay đổi → KHÔNG bắt (đúng — trông như vật). Đây là residual của môi trường nhiều kính.

### B — `--warmup-motion-mask` (clean_bg median LOẠI motion lúc warmup) — GIỮ làm OPT-IN (default OFF)
- **Trị**: warmup-baked **transient ĐỘNG** (người/cửa cử-động trong warmup → nướng vào nền → ghost FP sau). clean_bg = median chỉ trên mẫu ĐỨNG-YÊN của từng pixel (loại framediff>20), fallback median-thường nơi luôn-động. Thuật toán **CHUNG** (không hard-code).
- **Kết quả test (4 video)**:
  - video6: **f780 BIẾN MẤT** (8→6 ev), HIT 2/2 ✓
  - video7: 4→4, HIT 1/1 (no change) ✓
  - video11: 12→12, HIT 1/1 (no change) ✓
  - **vid0355: 2→3 ev — B THÊM 1 FP** (f389 (403,252)) — **đã ĐO cơ chế + ĐÃ FIX**: tái lập 100% (plain 2/2 sạch, B 2/2 FP, không phải noise). Cơ chế gián tiếp: B cắt pixel-motion-xe khỏi clean_bg → heal-revealed YOLO (chạy TRÊN clean_bg) nhận vùng xe nhỏ hơn (12.1%→10.7%) → vùng suppress co → (403,252) rìa xe lọt ra → FP.
  - **✅ FIX (heal-detect trên plain median)**: tách 2 vai trò clean_bg — heal-revealed detect trên `clean_bg_color_for_heal` (plain, lưu trước khi B ghi đè); newdiff/state dùng clean_bg-B. **KQ B+C(fix): vid0355 2ev (FP HẾT), video6 f780 vẫn hết (3ev HIT2/2), heal về 12.1%.** Behavior-preserving khi B tắt.
  - ⚠️ **NHƯNG B+C(fix) đẻ FP MỚI video11 (194,222)**: full-sweep cho thấy video11 12→13. Truy: (194,222) có ở MỌI config bật B (không C-only) → **B gây qua newdiff→relight-divergence**, deterministic (repeat 2/2). Trước fix bị CHE bởi heal-on-B tình cờ nén 1 FP khác (342,366); fix gỡ che → lộ.
  - ❌ **Đã THỬ "targeted-B" (chỉ áp B nơi region|B−plain|>ngưỡng) → LOẠI**: (a) ngưỡng vô căn cứ (overfit, chọn 6 nằm giữa f780=31.6 và vid11=5.3); (b) **đo principled (noise-floor) cho thấy (194,222)=5.3 là B-correction THẬT, không phải noise**; (c) **test: targeted-B KHÔNG khử được (194,222)** (vẫn 13ev) vì FP seed từ correction KHÁC (>ngưỡng, bị giữ). → **lợi ích B (f780) và hại B (relight-FP) KHÔNG tách rời** — cùng từ correction của B. Đã revert về B+C(fix).
  - **VERDICT B (cập nhật): DEFAULT ON** (`--warmup-motion-mask` 1) theo quyết định người dùng — fix f780 out-of-the-box, đổi lấy video11 +1 FP (đã hiểu, chấp nhận). Tắt bằng `--warmup-motion-mask 0`. Cho cam thật: **ưu tiên `--clean-bg-image`/cửa-sổ-warmup-sạch** (né nhiễm tại gốc, ổn định hơn B); B perturbs clean_bg → có thể đẻ FP khó lường ở cảnh khác → **nên A/B test trên footage thật**.
- **Rủi ro của B = thêm FP (false-POSITIVE)**, KHÔNG nuốt vật (B chỉ đổi nền tham chiếu, không suppress alert) → **an toàn hơn A** cho an ninh.
- **Giới hạn**: chỉ trị transient-ĐỘNG; trạng-thái-SAI-tĩnh (cửa giữ mở yên >50% warmup) → không cứu.
- **Đã thử + LOẠI**: mở rộng heal-revealed quét warmup-frames bằng YOLO → vô ích (YOLO MÙ với người-tối-cúi ở cửa video6: 0% baked) → đã revert.

### C — `--alert-min-support` (evidence-gate lúc alert) — GIỮ default ON (0.05) ⭐ FIX CHÍNH cho ma-sàn/cột
- **Trị**: candidate "MA" — đã hết bằng chứng nhưng static-FG còn giữ trễ → vẫn alert. Điển hình: **đổi sáng vùng → light-comp HẤP THỤ vào clean_bg → newdiff về 0 → nhưng FSM còn giữ → báo nhầm** (FP sàn/cột video6 f5157/f5171).
- **Cơ chế**: lúc alert đòi bbox còn **≥5% pixel newdiff hiện tại**; nếu ~0 → **DEFER** (continue, KHÔNG mark alerted → bằng chứng quay lại thì alert tiếp). Vật-frozen LUÔN khác clean_bg → support cao; ma đã-hấp-thụ → 0.
- **AN TOÀN (đo cả 11 video, 0 MISS)**: SUPPORT-DBG xác nhận f5157/f5171×2 support=**0.00%** suốt → defer; vật thật f3629/f5679 support **25–58%** → giữ. **Recall 12/12** toàn ABODA; người che thoáng qua chỉ TRỄ (re-age tức thì), không mất.
- **Kết quả**: video6 **8→4** (3 ma sàn/cột biến mất), video9 2→0, video10 2→0; video7/8/11/0103/0355 không đụng vật/FP-sống. **FP tổng ~20→16.**
- **Rủi ro = false-NEGATIVE rất hẹp**: chỉ khi clean_bg hấp thụ NHẦM vật thật (light-comp >15% cov / relight nuốt vật chưa-alert) → nhưng đó là rủi ro light-comp/relight CÓ SẴN, C không thêm.

### ❌ A — regional-shift (patch vs ring) — ĐÃ THỬ, VÔ HIỆU, GỠ KHỎI CODE
- **Lý do gỡ**: A đo "độ dịch sáng hiện tại" của blob, NHƯNG FP sàn/cột video6 lúc alert có **blob_n=0** (newdiff RỖNG — light-comp đã hấp thụ) → A không có gì để đo → bail. Test thật: video6 +A = **bỏ 0 FP**. f780/f5148 thì |diff|/std lớn (A đúng khi giữ). → A sai bài: các FP này KHÔNG phải lighting-sống lúc alert mà là **ma cũ** → **C mới là fix đúng**.
- **Rủi ro A nếu giữ**: nuốt vật-ngụy-trang (vật trùng độ-sáng-sàn-quanh) = false-negative → không hợp an ninh.
- Phân tích đầy đủ patch-vs-ring (video6 3/4, video7 1-2/2 "lý thuyết") vẫn lưu ở memory; nhưng do blob_n=0 nên không áp dụng được thực tế.

---

## 2. ⏳ CHƯA LÀM — ACTIONABLE

### P2-bbox — bbox cảnh báo ÔM VẬT (gather-0 cho box tí xíu) ⭐ ưu tiên
- **Triệu chứng**: với `--gather-px 0` box chỉ 1 góc tí của vật (video8: 23×11 vs GT 208×67). Ngay cả gather-5 chỉ ~52% vật (mask khác-nền không phủ chỗ vật trùng-sáng-nền).
- **Phương án (tăng dần)**: (a) gather-px ≥5 + fill-holes; (b) bbox-refine = **union(tight,newdiff) + CLOSE 10px + fill-holes**; (c) **ưu tiên mask YOLO-seg** của instance phủ candidate (khít nhất, nhưng chỉ vật COCO → fallback diff-box).
- **Test**: video8 (máy giặt, non-COCO) + video1/4 (vật rõ) — box phải ôm ≥80% vật, không over-grow dính bóng.

### P2-dedup-vật-to — vid0355: 1 máy giặt báo LẶP 2 vị trí
- **Triệu chứng**: máy giặt to → 2 alert (459,253)+(512,204) cách ~75px > `--dedup-dist` 40 → dedup không gộp → báo lặp.
- **Phương án**: dedup-dist **co giãn theo kích thước vật** (vd max(40, 0.5·√area)) hoặc gộp alert cùng-frame gần nhau. **Test**: vid0355 (1 alert) + ABODA (không gộp nhầm 2 vật riêng).

### P2a — Điều tra FP vid0103 (302,212)
- **Hiện trạng**: tất định 4/4 lần, **không phải crowd-n, không phải noise** → đến từ **đổi nano→yolo26s**. Là FP đường-gần-rác riêng biệt (không phải dedup).
- **Cần làm**: xem ảnh + mask quanh f1017 → là **rác/vật thật** (thì là HIT, không phải FP) hay **bóng/nhiễu s-model**.
- **Phương án nếu là nhiễu**: (a) tăng `--area-min` (nếu blob nhỏ) · (b) `--tau-object`/animate gate · (c) chấp nhận (1 FP). **KHÔNG chỉnh chung làm hại video khác** → cần per-scene hoặc rất bảo thủ.
- **Test**: vid0103 + 2 video sạch (đảm bảo không thêm FP/miss).

### ✅ FP đổi-sáng SÀN/CỘT video6 (f5157/f5171×2) — ĐÃ GIẢI QUYẾT bằng C (§1); A đã gỡ
> Hóa ra KHÔNG phải lighting-sống mà là **ma cũ** (light-comp hấp thụ → blob_n=0 lúc alert) → **C diệt sạch an toàn**. A (regional-shift) vô hiệu vì blob_n=0 → đã gỡ. Phân tích A bên dưới giữ làm lịch sử.
- **Phân tích lại (data std-of-diff)**: KHÔNG phải reflection (tôi từng nhầm). Đây là **brightening ĐỀU theo VÙNG**: `std(gray−clean_bg)` nhỏ (f5157=10.1, f5171b=13.2) = dịch sáng đồng đều, **không đổi cấu trúc** (đúng như user nói). NCC sai metric vì patch phẳng/mỏng. (f5148 khác: std-diff 55 = đèn TẮT = đổi cấu trúc thật.)
- **Gốc**: đổi sáng **CỤC BỘ** → mean toàn cục +5 < `relight_dv`20 → relight không pause → vùng sáng lọt FP cho tới khi relight rebuild (f5200).
- **Hướng A (regional uniform-shift)**: tại alert, so độ-dịch-sáng patch với **VÙNG XUNG QUANH (ring)**. Lighting = patch ≈ ring (cả vùng sáng đều) → bỏ; vật = patch ≫ ring → giữ.
  - **ĐÃ ĐO (lý thuyết, chưa implement)**: video6 → A bỏ **3/4** (f5157 |diff|13.5, f5171a 12.3, f5171b 2.1; GIỮ f5148 |diff|74=đèn-tắt-local); video7 → bỏ f2052 (11.1), f2035 sát ngưỡng (15.0). → **A trị đúng nhóm sàn/cột/tường (~4–5/6 FP lighting)**.
  - ⚠️ **RỦI RO = NUỐT VẬT (false-NEGATIVE)**: A là suppress. Vật bị bỏ nếu (a) độ-dịch-sáng vật **tình cờ bằng** dịch-sáng-vùng (vật-phẳng độ-sáng-bằng-sàn-đã-lit, đặt ĐÚNG lúc đổi sáng) → patch≈ring; hoặc (b) vật TO lấp cả ring; hoặc (c) ring nhiễm người/vật khác. → **nguy hiểm hơn B** (B chỉ thêm-FP).
  - **Safeguard**: chỉ bỏ khi `|patch−ring|<~17` **VÀ** `std(patch−clean_bg)` nhỏ (vật có cạnh/vân → std cao → GIỮ). Cứu vật-có-cạnh; vật-phẳng-không-cạnh-trùng-sáng vẫn lọt (hiếm).
  - **→ Nếu làm: OPT-IN + ngưỡng chặt + test recall NHIỀU vật** (đừng default-ON vì có thể bỏ vật thật).
- **Tune nhỏ thay thế**: hạ `--light-struct-ncc` 0.85→0.80 bắt thêm f5171a (NCC 0.82) — nhưng tăng rủi ro nuốt vật.
- **Khử bóng (shadow)**: chưa có; bóng tĩnh > th_diff → FP. Phương án: chroma/luma ratio. Rủi ro: bỏ nhầm vật tối.
- **Test**: video6/video7 + ≥1 video có bóng + video sạch (recall).

### P2c — `--clean-bg-image` (ảnh nền trống chụp sẵn) cho an ninh cao
- **Mục đích**: né blind-spot P3 (vật-tĩnh nhận-nhầm-người / agent baked lúc warmup). Cho phép nạp 1 ảnh nền trống chụp trước thay vì median warmup.
- **Phương án**: thêm CLI `--clean-bg-image PATH` → dùng làm clean_bg thay warmup median (vẫn cho relight cập nhật). Đơn giản, ít rủi ro.
- **Test**: 1 cảnh có người-đứng-suốt-warmup (verify hết ghost) + ABODA (không đổi).

---

## 3. ⏸️ HOÃN — cần model mạnh hơn / công lớn

### P2d — Trần PERCEPTION (leg-FP v7 · FP đám đông v11 · balo-nhầm-người)
- **Gốc**: yolo26s **sót/nhầm** người đứng yên + vật-nhỏ-mảnh ≡ người. Mọi heuristic chỉ là đánh đổi recall↔precision.
- **Đòn thật** (đã hoãn theo quyết định giữ yolo26s vì tốc độ):
  - yolo26m-seg (recall tốt hơn, chậm hơn — bù bằng OpenVINO).
  - open-vocab / fine-tune detector (nhận RA vật bỏ quên) + bắt sự kiện **"chủ rời đi"**.
- **Khi nào làm**: khi chấp nhận chậm hơn, hoặc có GPU.

### P3 — Tách god-file `run_rtsbs_aod.py` (~1200 dòng)
- **Phương án**: tách thành `cli.py` (argparse) · `semantic_engines.py` (OnlineYoloSeg/SegFormer/PSPNet) · `pipeline.py` (main loop) · `alerting.py` (dedup/owner-gate/light-struct). Engineering debt, **không chặn release**.

---

## 4. ⏳ DOCS + DỌN DẸP (làm khi chốt các fix)

- [ ] Cập nhật `solution_analysis.md`: thêm **P1a dedup-cooldown** + **P1b light-struct** + findings (light-comp usage v6/7/8; video6 corner-light; video9 dedup root; vid0103/v11 FP investigation = không phải crowd-n).
- [ ] Cập nhật `memory` (demov2-aod-fsm-findings) tương tự.
- [ ] Cập nhật `README` (mục lighting: thêm light-struct; xác nhận light-norm đã gỡ).
- [ ] **Dọn temp dirs**: `demov2/diag/`, `demov2/debug/`, `demov2/test_dedup/`, `demov2/test_lstruct/` (artifact test; giữ `results_release/` làm KQ chính).
- [ ] Cân nhắc hạ `--dedup-cooldown-s` 30→20 (churn quan sát chỉ 7–10s; cửa-sổ-che ngắn hơn) — tùy quyết định.

---

## 5. Tồn đọng nhỏ (ghi nhận, chưa ưu tiên)

- **owner-gate báo sớm (video9 alert#1)**: nổ đúng biên `owner_clear`(3s) sau khi người rời → hơi sớm. Nới owner_clear → trễ báo thật. Để nguyên (báo sớm chấp nhận được cho an ninh).
- **pybgs không-tất-định** ở vài video (v11) → FP dao động ±1 giữa các lần chchạy. dedup-cooldown làm ổn định một phần; muốn tất-định hoàn toàn phải đổi backend `controlled` (numba, chậm hơn).
- **dedup-dist 40px** (đã chốt giữ): 2 vật cách <40px → 1 báo. Muốn phân biệt vật-gần phải giảm (đánh đổi double-alert).

---

## Thứ tự đề xuất tiếp theo
1. ✅ P1a + P1b xong.
2. **Chốt B** (`--warmup-motion-mask`): chờ test video7/11/0355 → nếu không-regress thì giữ (quyết default ON hay opt-in).
3. **P2a** (vid0103 — nhanh, chỉ phân tích).
4. **Docs + dọn temp** (chốt P1a/P1b/B + findings vào solution_analysis + memory; xóa diag/debug/test_*).
5. Hỏi bạn chọn: **A** (regional-shift, floor-FP video6/7) hay **P2c** (clean-bg-image) — đều cần test kỹ recall.
6. ❌ Mục §0 (mở-cửa runtime / perception) — cần ROI/model, để sau.
