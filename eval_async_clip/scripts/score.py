"""Score a demov2 events.json against an ABODA GT center: recall (hit within tol) + FP count."""
import json, sys, os

# GT centers (proc-width 640 coords) computed from aboda_gt.json bbox [x,y,w,h]
GT = {
    "video1":  (157, 343),
    "video3":  (236, 232),
    "video7":  (156, 119),
    "video8":  (190, 227),
    "video11": (309, 271),
}
TOL = 45

def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None

def score(outdir, vid, gt_center=None, tol=TOL):
    evs = load(os.path.join(outdir, "events.json"))
    if evs is None:
        return f"{os.path.basename(outdir):28} NO events.json"
    n = len(evs)
    if gt_center is None:
        gt_center = GT.get(vid)
    if gt_center is None:
        return f"{os.path.basename(outdir):28} events={n} (no GT center -> all reported as events)"
    hit = [e for e in evs if abs(e['center'][0]-gt_center[0]) <= tol and abs(e['center'][1]-gt_center[1]) <= tol]
    fp = n - len(hit)
    rec = "HIT" if hit else "MISS"
    hf = f" @f{hit[0]['frame']}" if hit else ""
    return f"{os.path.basename(outdir):28} events={n:3}  recall={rec}{hf}  FP={fp}"

if __name__ == "__main__":
    # usage: score.py <vid> <outdir1> <outdir2> ...
    vid = sys.argv[1]
    for d in sys.argv[2:]:
        print(score(d, vid))
