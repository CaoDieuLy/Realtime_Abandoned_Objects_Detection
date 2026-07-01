"""Score the full-ABODA sweep. GT centers (NATIVE coords) are scaled to proc-640 (the pipeline runs
at --proc-width 640) before matching. Recall = GT objects hit / total; FP = events matching no object.
Multi-object videos (video6) scored per object. No-GT videos report event count only.
usage: score_full.py <ABODA_dir> <gt.json> <outdir_root> <proc_width>
"""
import json, sys, os, glob

TOL = 50  # px in proc coords (event center within this of a scaled GT center = hit)

def native_width(gt_v):
    return gt_v.get("resolution", [640, 480])[0]

def main():
    aboda, gtpath, root, procw = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    gt = {v["video_id"]: v for v in json.load(open(gtpath, encoding="utf-8"))["videos"]}
    rows, tot_obj, tot_hit, tot_fp = [], 0, 0, 0
    for outdir in sorted(glob.glob(os.path.join(root, "*"))):
        vid = os.path.basename(outdir).replace("clip_", "")
        ej = os.path.join(outdir, "events.json")
        if not os.path.isfile(ej):
            continue
        evs = json.load(open(ej, encoding="utf-8"))
        gv = gt.get(vid)
        if not gv or not gv.get("objects"):
            rows.append((vid, len(evs), "-", "-", f"{len(evs)} ev (no GT)"))
            continue
        scale = procw / native_width(gv)
        objs = []
        for o in gv["objects"]:
            x, y, w, h = o["bbox"]
            objs.append(((x + w / 2) * scale, (y + h / 2) * scale, o["object_type"]))
        matched_obj = set()
        fp = 0
        for e in evs:
            cx, cy = e["center"]
            hit_j = None
            for j, (ox, oy, _t) in enumerate(objs):
                if abs(cx - ox) <= TOL and abs(cy - oy) <= TOL:
                    hit_j = j; break
            if hit_j is None:
                fp += 1
            else:
                matched_obj.add(hit_j)
        nobj, nhit = len(objs), len(matched_obj)
        tot_obj += nobj; tot_hit += nhit; tot_fp += fp
        rec = f"{nhit}/{nobj}" + ("" if nhit == nobj else "  <-- MISS")
        rows.append((vid, len(evs), rec, fp, ""))
    print(f"{'video':9} {'events':>6} {'recall':>10} {'FP':>4}  note")
    print("-" * 54)
    for vid, nev, rec, fp, note in sorted(rows, key=lambda r: (r[0].startswith('vid'), r[0])):
        print(f"{vid:9} {nev:>6} {str(rec):>10} {str(fp):>4}  {note}")
    print("-" * 54)
    print(f"TOTAL GT videos: recall {tot_hit}/{tot_obj} objects, FP={tot_fp}")

if __name__ == "__main__":
    main()
