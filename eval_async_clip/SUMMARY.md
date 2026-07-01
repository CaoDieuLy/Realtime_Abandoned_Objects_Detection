# Test results — async-semantic (#2) + CLIP verifier (#1)

Hai cải tiến SOTA thêm vào pipeline AOD, đo trên **ABODA**, CPU, `py -3.13` (pybgs + OpenVINO + open_clip), chạy từ repo-root. Mọi `events.json` + `alert_*.jpg` trong `runs/`; log stdout đầy đủ (có dòng FPS + bảng SCORING) trong `logs/`; script tái lập trong `scripts/`.

> Chấm điểm: một event là **HIT** nếu tâm nằm trong ±45px của tâm GT ABODA; còn lại là **FP**. GT: video1 backpack (157,343); video7 handbag (156,119); video11 umbrella (309,271).

---

## #2 — Async-semantic (worker thread)

`--async-semantic {auto,on,off}` (auto = ON cho mode no-feedback). YOLO chạy thread nền, vòng BGS/FSM dùng map gần nhất (cũ vài frame = vô hại cho AOD vì vật đứng yên hàng giây). GIL nhả khi YOLO/ViBe chạy native → overlap thật trên CPU đa luồng. Chỉ bật khi **không** feedback (RT-SBS feedback cần map frame hiện tại → giữ sync); mọi `infer()` serialize qua `infer_lock`.

### FPS (video11, máy idle, xen kẽ để công bằng) — `logs/fps_bench.log`
| Run | sync | async |
|---|---|---|
| lần 1 | 5.0 | **6.1** |
| lần 2 | 5.1 | **6.0** |

→ **+20%** ổn định trên video11 (crowd — main-loop nặng nên YOLO chỉ chiếm ~12% thời gian; cảnh nhẹ lợi nhiều hơn). Async chạy **nhiều YOLO hơn** (200 vs 160 infer/800 frame) mà vẫn nhanh hơn.

### Event parity (video11, sync vs async, cùng 3000 frame) — `logs/phase2_async.log`, `runs/p2_v11_off` vs `runs/p2_v11_on`
| | sync (`p2_v11_off`) | async (`p2_v11_on`) |
|---|---|---|
| events | 10 | 11 |
| recall (umbrella) | MISS¹ | **HIT** @f2097 |
| FP | 10 | 10 |

10 event của sync **xuất hiện y hệt** trong async (lệch ±1–3 frame do map cũ hơn vài frame); FP **giống hệt**. Async bắt **thêm** umbrella (YOLO dày hơn → object-keep tươi hơn = bonus recall).

¹ Sync MISS umbrella = nhiễu pybgs (xem mục determinism dưới), không phải do mode.

**Kết luận #2:** parity hoàn hảo + +20% FPS + nhỉnh recall, 0 rủi ro → **nên default ON**.

---

## #1 — CLIP verifier (MobileCLIP2-B, per-alert)

`--clip-verify 1` (default 0). [core/clip_verifier.py](../core/clip_verifier.py) dùng **open_clip_torch** (`MobileCLIP2-B/dfndr2b`, auto-download HF; *không dùng `mobileclip2_b.ts` — đó chỉ là text-tower*). Chạy **1 lần/candidate** (cache per cand_id + TTL `--clip-recheck-s` 2s) ở khối alert, SAU owner-gate/light-struct. Zero-shot 2 nhóm prompt OBJECT (bag/suitcase/box/umbrella…) vs NOT-OBJECT (person/crowd/floor/wall/door/car/shadow/glare). **Recall-safe = abstain**: suppress CHỈ khi `P(object) < keep_conf(0.25)` **AND** top-1 (not-object) softmax `≥ suppress_conf(0.30)`; còn lại GIỮ.

### Kết quả A/B (async ON xuyên suốt) — `logs/phase1_clip.log`
| Video | clip OFF | clip ON | Hiệu ứng |
|---|---|---|---|
| **video7** (lighting) | 4 ev, HIT, **FP=3** | 1 ev, HIT, **FP=0** | cắt sạch "plain wall" (top1 0.92–0.96) + "empty floor"; giữ handbag (p_obj=0.99) |
| **video1** (sạch) | 1 ev, HIT, FP=0 | 1 ev, HIT, FP=0 | **0 regression** (backpack p_obj=1.00) |
| **video11** (crowd) | 11 ev, HIT, FP=10 | 4 ev, MISS¹, FP=4 | cắt FP đám đông/sàn/cửa |

`runs/p1_v7_off` vs `runs/p1_v7_clip`, `runs/p2_v11_on` vs `runs/p1_v11_clip`, `runs/p2_v1_on` vs `runs/p1_v1_clip`.

### ⭐ Chứng minh recall-safe — umbrella determinism — `logs/umbrella_determinism.log`
4 run cap 2250, clip off/on xen kẽ (`runs/umb_*`):
| Run | clip | Umbrella | FP |
|---|---|---|---|
| umb_off1 | 0 | MISS | 9 |
| umb_on1 | 1 | MISS | 3 |
| umb_off2 | 0 | **HIT** @f2095 | 9 |
| umb_on2 | 1 | **HIT** @f2093 | 3 |

→ HIT/MISS umbrella **uncorrelated với clip**: khi pybgs bắt được (off2/on2) thì HIT **cả off lẫn on**; khi pybgs miss (off1/on1) thì miss cả hai. **`umb_on2` HIT chứng minh CLIP giữ umbrella trong pipeline đầy đủ.** ¹ Vậy MISS ở `p1_v11_clip` là **nhiễu pybgs** (umbrella là vật borderline khó nhất, non-deterministic), KHÔNG phải CLIP. Test trực tiếp CLIP trên crop umbrella mọi frame f2050–2300 → **KEEP hết** (p_obj 0.33–0.51 ≥ 0.25, abstain kể cả khi top-1 đọc nhầm "plain wall"). Guard `p_obj` là cái cứu vật khó.

**FP cắt nhất quán:** video11 9→3 (−67%) cả hai lần off→on; video7 3→0.

### Chi phí
CLIP ~600ms/crop CPU, chạy 1 lần/candidate (cache). Ở crowd nhiều candidate re-verify → **~20–25% FPS** (video11 6.6→5.5, video7 7.7→6.7). Giảm: nâng `--clip-recheck-s` (4–5s vẫn an toàn vì vật cần 5s static mới alert) hoặc model nhỏ hơn (MobileCLIP2-S2 ~2–3× nhanh).

**Kết luận #1:** cắt FP crowd/lighting mạnh (−67% / →0), **recall-safe đã chứng minh**, đổi lấy ~20–25% FPS ở crowd → **default OFF (opt-in)**, bật cho deployment cần ít FP.

---

## Tái lập
```bash
# từ repo-root, py -3.13
# #2 async A/B:        bash scripts/phase2.sh
# #1 CLIP A/B:         bash scripts/phase1.sh
# FPS bench:           bash scripts/fps.sh
# umbrella determinism: bash scripts/umb.sh
# chấm điểm:           py -3.13 scripts/score.py <video> <outdir1> <outdir2> ...
```

## Lưu ý môi trường
- Dùng **`py -3.13`** (đủ pybgs+OpenVINO+open_clip); `python`=3.11 thiếu pybgs+OpenVINO.
- Chạy từ **repo-root** (OpenVINO model `yolo26s-seg_openvino_model` nằm ở repo-root).
- pybgs ViBe **non-deterministic** → FP/recall vật borderline dao động ±1 giữa các lần chạy (xem umbrella).
