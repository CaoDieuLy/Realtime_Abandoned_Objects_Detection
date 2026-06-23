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
- `--warmup-ignore {0|1}` (+`--warmup-ignore-dilate`, `--warmup-ignore-max`) — **person-aware warmup**: loại pixel animate (dùng `--yolo-animate-classes`) khỏi median `clean_bg` để người đứng trong warmup không bị "nướng" vào nền (→ ghost FP khi họ rời). **Default 0 (TẮT) — chỉ chạy YOLO trên frame warmup khi bật, nên không ảnh hưởng tốc độ khi tắt.** Mode-agnostic (clean_bg dùng chung). Trên ABODA **không giảm FP** (người ở v11 đi-ngang nên median đã tự loại). Soi kỹ v11: net 11=11 nhưng nó **gỡ đúng 1 FP sàn rồi jitter frame/dedup** — 10/11 vùng lỗi gốc (người+lóa) y nguyên → không trị nguồn thật. Để dành cho cảnh có người đứng-yên-suốt-warmup thật.
- `--scene-memory {0|1}` (+`--scene-memory-mode {relocated|background|both}`, `-thresh`, `-source-change`, `-min-move-dist`, `-bg-sim`, `-debug`) — lớp suppress SAU matcher (`core/scene_feature_memory.py`). **`relocated`**: vật-nền bị dời chỗ (match HSV+edge + nguồn đã đổi + ở xa). **`background`**: lóa/bóng để lộ cấu trúc nền cũ. **Default OFF.** ⚠️ Test v11: `relocated` = no-op (0 vật-dời); `background` giảm FP 11→6 NHƯNG **xóa trắng khu đông (suppress 35, gọi nhầm đám đông là "nền")** → **sẽ nuốt vật bỏ quên TRONG đám đông → UNSAFE, đừng bật ở cảnh đông**. Chỉ dùng `relocated` cho camera có vật-nền hay bị xê dịch.

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
  semantic_lut.py       # tập lớp moving/object/stuff (LUT → ids)
  static_matching.py    # gom blob → theo vết → ứng viên
  dense_semantic.py     # đọc dense semantic map (mode dense)
tools/                  # sinh dense semantic (SegFormer/PSPNet), batch driver
```

## Output (mỗi `--save-masks-every`)
`events.json` · `alert_f*.jpg` · `raw_vibe_f*` (ViBE) · `rtsbs_f*` (sau feedback) · `semantic_f*.png`/`semantic_vis_f*` (SegFormer) · `statnew/newdiff/fg/staticfg/moving/keep/stuff/tight/age` · `cleanbg_f*`.

## Kết quả & giới hạn

> **`no-feedback` CHÍNH LÀ demov1** — cùng cấu trúc (pybgs ViBE + clean_bg FSM + trừ-người hậu kỳ, không feedback). Vì vậy không cần so sánh demov1 riêng: nó là **cùng một pipeline**.

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
