"""Local desktop app for the realtime abandoned-object detector.

Pick a VIDEO file or a LIVE CAMERA, run the pipeline, and watch it live:
  * a red box is drawn on the alerted object and stays on every following frame while the object is
    there; it disappears automatically once the object is TAKEN AWAY (its spot returns to clean_bg);
  * top-right "Vật bỏ quên (N)" button opens a pop-up list of the currently-confirmed objects;
  * click a red box to confirm/reject it -- "Không phải" removes it from the list;
  * every confirmed object is written to its own JSON file (object_<id>.json) in the session out-dir.

For a video the processing FPS is shown (so you know the pipeline's real speed). For a camera you pick
the capture FPS up-front -- set it near that processing FPS so frames aren't buffered/lagged.

Run the GUI:        python app.py
Headless self-test: python app.py --selftest --video ABODA/video6.avi   (no display; checks the
                    pipeline hooks + writes per-object JSON -- used to validate without a screen)

Needs: tkinter (stdlib) + Pillow (GUI only). The pipeline engine is run_rtsbs_aod.main().
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rtsbs_aod as engine


# --------------------------------------------------------------------------------------------------
# Object store: mirrors the pipeline's live objects and persists one JSON file per object.
# --------------------------------------------------------------------------------------------------
class ObjectStore:
    """Tracks confirmed abandoned objects and writes object_<id>.json on every state change."""

    def __init__(self, out_dir: str, source_label: str):
        self.out_dir = out_dir
        self.source = source_label
        self.objects: dict[int, dict] = {}
        os.makedirs(out_dir, exist_ok=True)

    def update(self, idx: int, active: list[dict], rejected: set[int], fps: float) -> None:
        active_ids = set()
        for o in active:
            active_ids.add(o["id"])
            if o["id"] not in self.objects:
                rec = {
                    "id": o["id"], "status": "present", "source": self.source,
                    "frame_alert": o["frame_alert"], "t_alert_s": o["t_alert"],
                    "center": list(o["center"]), "bbox": list(o["bbox"]),
                    "frame_taken": None, "t_taken_s": None,
                }
                self.objects[o["id"]] = rec
                self._write(rec)
            else:
                self.objects[o["id"]]["bbox"] = list(o["bbox"])
        for oid, rec in self.objects.items():
            if rec["status"] == "present" and oid not in active_ids:   # left the live list
                rec["status"] = "rejected" if oid in rejected else "taken"
                rec["frame_taken"] = idx
                rec["t_taken_s"] = round(idx / max(1e-6, fps), 2)
                self._write(rec)

    def snapshot(self) -> list[dict]:
        return [dict(r) for r in self.objects.values()]

    def _write(self, rec: dict) -> None:
        path = os.path.join(self.out_dir, f"object_{rec['id']:03d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------------------------------
# Pipeline runner: drives engine.main() in a background thread, feeding frames to a queue.
# --------------------------------------------------------------------------------------------------
class PipelineRunner:
    def __init__(self, argv: list[str], out_dir: str, source_label: str, frame_q: "queue.Queue"):
        self.argv = argv
        self.store = ObjectStore(out_dir, source_label)
        self.frame_q = frame_q
        self.rejected: set[int] = set()
        self._stop = False
        self.done = False
        self.error: str | None = None
        self._n = 0
        self._t0 = time.time()
        self.proc_fps = 0.0

    def stop(self):
        self._stop = True

    def _on_frame(self, idx, display, active, fps, total):
        # measured PROCESSING fps (frames the pipeline actually pushes per real second)
        self._n += 1
        if self._n % 5 == 0:
            self.proc_fps = self._n / max(1e-6, time.time() - self._t0)
        active_copy = [{"id": o["id"], "bbox": list(o["bbox"]), "center": list(o["center"]),
                        "t_alert": o["t_alert"], "frame_alert": o["frame_alert"]} for o in active]
        self.store.update(idx, active_copy, self.rejected, fps)              # always persist
        item = (idx, display.copy(), active_copy, self.proc_fps, total, self.store.snapshot())
        try:
            self.frame_q.put_nowait(item)                                   # display: drop if behind
        except queue.Full:
            try:
                self.frame_q.get_nowait()
                self.frame_q.put_nowait(item)
            except queue.Empty:
                pass

    def run(self):
        try:
            engine.main(self.argv, on_frame=self._on_frame,
                        should_stop=lambda: self._stop, rejected_ids=self.rejected)
        except Exception as exc:                                            # surface to the GUI
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.done = True


def build_argv(source: dict, out_dir: str) -> list[str]:
    """Translate the start-screen choices into engine CLI args."""
    argv = ["--outdir", out_dir, "--gather-px", "0"]
    if source["kind"] == "video":
        argv += ["--video", source["path"], "--bg-learn-seconds", str(source.get("warmup", 20))]
    else:
        argv += ["--camera-index", str(source["index"]), "--camera-fps", str(source["fps"]),
                 "--bg-learn-seconds", str(source.get("warmup", 20))]
    return argv


# --------------------------------------------------------------------------------------------------
# Headless self-test (no display) -- validates the pipeline hooks + JSON on a video.
# --------------------------------------------------------------------------------------------------
def selftest(video: str, warmup: float, out_dir: str, max_frames: int = 0) -> int:
    print(f"[selftest] {video} -> {out_dir}")
    q: "queue.Queue" = queue.Queue(maxsize=4)
    argv = build_argv({"kind": "video", "path": video, "warmup": warmup}, out_dir)
    if max_frames:
        argv += ["--max-frames", str(max_frames)]
    runner = PipelineRunner(argv, out_dir, os.path.basename(video), q)
    th = threading.Thread(target=runner.run, daemon=True)
    th.start()
    last = 0
    while not runner.done or not q.empty():
        try:
            idx, _disp, active, pfps, _total, _snap = q.get(timeout=0.5)
            last = idx
            if active and idx % 30 == 0:
                print(f"  f{idx} ~{pfps:.1f} FPS | active: " +
                      ", ".join(f"#{o['id']}@{tuple(o['center'])}" for o in active))
        except queue.Empty:
            pass
    th.join(timeout=2)
    if runner.error:
        print(f"[selftest] ERROR: {runner.error}")
        return 1
    objs = runner.store.snapshot()
    print(f"[selftest] done at f{last}. {len(objs)} object(s) confirmed:")
    for o in sorted(objs, key=lambda r: r["id"]):
        print(f"  object_{o['id']:03d}: status={o['status']} alert@f{o['frame_alert']} "
              f"({o['t_alert_s']}s) center={o['center']} taken@{o['frame_taken']}")
    print(f"[selftest] JSON written to {out_dir}/object_*.json")
    return 0


# --------------------------------------------------------------------------------------------------
# Tkinter GUI
# --------------------------------------------------------------------------------------------------
def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("GUI needs Pillow:  python -m pip install pillow", file=sys.stderr)
        return 1

    class App:
        def __init__(self, root):
            self.root = root
            root.title("Phát hiện vật bỏ quên")
            self.frame_q: "queue.Queue" = queue.Queue(maxsize=2)
            self.runner: PipelineRunner | None = None
            self.thread: threading.Thread | None = None
            self.cur_active: list[dict] = []
            self.list_win = None
            self.photo = None
            self._build_start()

        # ---- start screen ----
        def _build_start(self):
            self.start = tk.Frame(self.root, padx=20, pady=20)
            self.start.pack()
            tk.Label(self.start, text="Chọn nguồn đầu vào", font=("", 13, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 12))
            self.kind = tk.StringVar(value="video")
            tk.Radiobutton(self.start, text="Video file", variable=self.kind, value="video", command=self._refresh).grid(row=1, column=0, sticky="w")
            tk.Radiobutton(self.start, text="Camera trực tiếp", variable=self.kind, value="camera", command=self._refresh).grid(row=1, column=1, sticky="w")

            self.path_var = tk.StringVar()
            self.row_video = tk.Frame(self.start)
            tk.Entry(self.row_video, textvariable=self.path_var, width=40).pack(side="left")
            tk.Button(self.row_video, text="Chọn…", command=self._browse).pack(side="left", padx=4)

            self.row_cam = tk.Frame(self.start)
            tk.Label(self.row_cam, text="Camera index:").grid(row=0, column=0)
            self.cam_idx = tk.IntVar(value=0)
            tk.Spinbox(self.row_cam, from_=0, to=8, width=4, textvariable=self.cam_idx).grid(row=0, column=1, padx=(2, 12))
            tk.Label(self.row_cam, text="FPS (đặt ≈ tốc độ pipeline):").grid(row=0, column=2)
            self.cam_fps = tk.DoubleVar(value=6.0)
            tk.Spinbox(self.row_cam, from_=1, to=30, increment=1, width=5, textvariable=self.cam_fps).grid(row=0, column=3, padx=2)

            tk.Label(self.start, text="Warm-up (giây học nền sạch):").grid(row=3, column=0, sticky="w", pady=(10, 0))
            self.warmup = tk.DoubleVar(value=20.0)
            tk.Spinbox(self.start, from_=1, to=60, increment=1, width=5, textvariable=self.warmup).grid(row=3, column=1, sticky="w", pady=(10, 0))

            tk.Button(self.start, text="▶ Bắt đầu", font=("", 11, "bold"), command=self._start).grid(row=4, column=0, columnspan=3, pady=14)
            self._refresh()

        def _refresh(self):
            self.row_video.grid_forget(); self.row_cam.grid_forget()
            if self.kind.get() == "video":
                self.row_video.grid(row=2, column=0, columnspan=3, sticky="w", pady=6)
            else:
                self.row_cam.grid(row=2, column=0, columnspan=3, sticky="w", pady=6)

        def _browse(self):
            p = filedialog.askopenfilename(title="Chọn video",
                                           filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
            if p:
                self.path_var.set(p)

        def _start(self):
            try:
                import pybgs  # noqa: F401  (engine default mode needs the C++ ViBe backend)
            except ImportError:
                messagebox.showerror(
                    "Thiếu pybgs",
                    f"Python đang chạy ({sys.version.split()[0]}) KHÔNG có 'pybgs'.\n\n"
                    f"Chạy bằng Python có pybgs — ví dụ:\n    py -3.13 demov2/app.py\n"
                    f"(ĐỪNG dùng 'python' nếu nó trỏ tới 3.11)\n\nHoặc cài: pip install pybgs")
                return
            if self.kind.get() == "video":
                path = self.path_var.get().strip()
                if not path or not os.path.exists(path):
                    messagebox.showerror("Lỗi", "Chọn một file video hợp lệ."); return
                source = {"kind": "video", "path": path, "warmup": self.warmup.get()}
                label = os.path.basename(path)
            else:
                source = {"kind": "camera", "index": self.cam_idx.get(), "fps": self.cam_fps.get(), "warmup": self.warmup.get()}
                label = f"camera{self.cam_idx.get()}"
            out_dir = os.path.join("aod_sessions", time.strftime("%Y%m%d_%H%M%S") + "_" + os.path.splitext(label)[0])
            argv = build_argv(source, out_dir)
            self.runner = PipelineRunner(argv, out_dir, label, self.frame_q)
            self.is_video = source["kind"] == "video"
            self.start.destroy()
            self._build_main(out_dir)
            self.thread = threading.Thread(target=self.runner.run, daemon=True)
            self.thread.start()
            self.root.after(30, self._poll)

        # ---- main screen ----
        def _build_main(self, out_dir):
            self.root.geometry("1100x760")
            self.root.minsize(640, 480)
            bar = tk.Frame(self.root, bg="#222")
            bar.pack(fill="x")
            self.status = tk.Label(bar, text="Đang học nền…", fg="white", bg="#222", font=("", 10))
            self.status.pack(side="left", padx=8, pady=4)
            self.list_btn = tk.Button(bar, text="Vật bỏ quên (0)", command=self._toggle_list)
            self.list_btn.pack(side="right", padx=8, pady=4)
            tk.Label(bar, text="F11: toàn màn hình", fg="#aaa", bg="#222").pack(side="right", padx=10)
            self.video = tk.Label(self.root, bg="black", cursor="hand2")
            self.video.pack(fill="both", expand=True)            # fill window -> video scales with it
            self.video.bind("<Button-1>", self._on_click)
            self.video.bind("<Configure>",                       # re-fit on resize even between frames
                            lambda e: self._render(self.last_disp) if self.last_disp is not None else None)
            self.last_disp = None
            self.disp_scale = 1.0
            self.disp_off = (0, 0)
            self._fs = False
            self.root.bind("<F11>", self._toggle_fs)
            self.root.bind("<Escape>", lambda e: self._toggle_fs(force_off=True))
            self.out_dir = out_dir
            self.root.protocol("WM_DELETE_WINDOW", self._close)

        def _toggle_fs(self, _e=None, force_off=False):
            self._fs = False if force_off else not self._fs
            self.root.attributes("-fullscreen", self._fs)

        def _render(self, disp):
            """Scale the BGR frame to fill the video area (keep aspect, letterbox) and show it."""
            if disp is None:
                return
            h, w = disp.shape[:2]
            aw, ah = self.video.winfo_width(), self.video.winfo_height()
            if aw < 10 or ah < 10:                               # not laid out yet -> sane default
                aw, ah = 1080, 600
            scale = min(aw / w, ah / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            rgb = cv2.cvtColor(cv2.resize(disp, (nw, nh), interpolation=interp), cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.config(image=self.photo)
            self.disp_scale = scale
            self.disp_off = ((aw - nw) // 2, (ah - nh) // 2)     # Label centers the image

        def _poll(self):
            item = None
            while True:
                try:
                    item = self.frame_q.get_nowait()
                except queue.Empty:
                    break
            if item is not None:
                idx, disp, active, pfps, total, snap = item
                self.cur_active = active
                self.last_disp = disp
                self._render(disp)
                live = [o for o in snap if o["status"] == "present"]
                fps_txt = f"{pfps:.1f} FPS" if self.is_video else f"{pfps:.1f} FPS (cam)"
                tot = f"/{total}" if total else ""
                self.status.config(text=f"frame {idx}{tot} · {fps_txt} · đang theo dõi {len(live)} vật")
                self.list_btn.config(text=f"Vật bỏ quên ({len(live)})")
                self._refresh_list(snap)
            if self.runner and self.runner.error:
                messagebox.showerror("Pipeline lỗi", self.runner.error); self.runner.error = None
            if self.runner and self.runner.done and self.frame_q.empty():
                self.status.config(text=self.status.cget("text") + "  —  XONG")
                return
            self.root.after(20, self._poll)

        # ---- click a box -> confirm / reject ----
        def _on_click(self, ev):
            ox, oy = self.disp_off                               # map click -> processing-frame coords
            px = (ev.x - ox) / max(1e-6, self.disp_scale)
            py = (ev.y - oy) / max(1e-6, self.disp_scale)
            for o in self.cur_active:
                x1, y1, x2, y2 = o["bbox"]
                if x1 <= px <= x2 and y1 <= py <= y2:
                    keep = messagebox.askyesno("Xác nhận vật bỏ quên",
                                               f"Vật #{o['id']} — đây có phải vật bị bỏ quên không?\n"
                                               f"(Chọn 'No' để loại khỏi danh sách)")
                    if not keep and self.runner:
                        self.runner.rejected.add(o["id"])
                    return

        # ---- object list pop-up (top-right) ----
        def _toggle_list(self):
            if self.list_win and self.list_win.winfo_exists():
                self.list_win.destroy(); self.list_win = None; return
            self.list_win = tk.Toplevel(self.root)
            self.list_win.title("Danh sách vật bỏ quên")
            self.list_win.geometry(f"+{self.root.winfo_rootx() + self.root.winfo_width()}+{self.root.winfo_rooty()}")
            self.tree = ttk.Treeview(self.list_win, columns=("t", "pos"), show="headings", height=12)
            self.tree.heading("t", text="Báo lúc (s)"); self.tree.heading("pos", text="Vị trí")
            self.tree.column("t", width=90); self.tree.column("pos", width=110)
            self.tree.pack(fill="both", expand=True)
            tk.Label(self.list_win, text="Vật tự rời danh sách khi được lấy đi.", fg="#555").pack(pady=2)

        def _refresh_list(self, snap):
            if not (self.list_win and self.list_win.winfo_exists()):
                return
            self.tree.delete(*self.tree.get_children())
            for o in sorted(snap, key=lambda r: r["id"]):
                if o["status"] != "present":
                    continue
                self.tree.insert("", "end", text=str(o["id"]),
                                 values=(o["t_alert_s"], f"{tuple(o['center'])}"))

        def _close(self):
            if self.runner:
                self.runner.stop()
            self.root.after(200, self.root.destroy)

    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Local GUI for the abandoned-object detector.")
    ap.add_argument("--selftest", action="store_true", help="run headless on --video, print + write JSON (no display)")
    ap.add_argument("--video", default="")
    ap.add_argument("--warmup", type=float, default=20.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.selftest:
        if not args.video:
            ap.error("--selftest needs --video")
        out = args.out or os.path.join("aod_sessions", "selftest_" + os.path.splitext(os.path.basename(args.video))[0])
        return selftest(args.video, args.warmup, out, args.max_frames)
    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
