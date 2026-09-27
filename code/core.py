"""Pure pandas logic for the Sapling Survival Audit app."""
import colorsys, io, math, re, zipfile
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent
PHOTO_DIR = ROOT / "data" / "photos"
OUT_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "sample_data"

CSV_COLS = ["patch_id", "status", "sapling_id", "landmark", "timestamp", "lat", "lon", "species_ai", "tree_guard", "grazing_damage",
            "physical_damage", "waterlogged", "canopy_open", "height_cm", "data_origin", "notes"]
FEATURES = ["green_cover_pct", "yellow_leaf_pct", "bare_soil_pct", "soil_color", "soil_darkness", "soil_type_ai"]
OBS_COLS = CSV_COLS + FEATURES  # only these columns are read
ALIASES = {"canopy_openness_pct": "canopy_open", "canopy_openness": "canopy_open"}  # old column names still accepted


def clean_header(c):
    """Lower-case, trimmed column name with old names mapped to the current ones."""
    c = str(c).strip().lower().replace(" ", "_")
    return ALIASES.get(c, c)
REQUIRED = ["patch_id", "sapling_id", "status"]  # photo = {patch_id}_{sapling_id}.jpg; lat/lon/timestamp from its EXIF
PATCH_COLS = ["patch_id", "patch_name", "status_for_study", "centroid_lat", "centroid_lon", "count_planted",
              "last_planting_date", "aftercare_watering", "waterlogging_history", "terrain_position", "soil_texture",
              "compaction_depth_cm", "grazing_evidence", "fencing", "standing_water_now", "foot_traffic_count_10min",
              "overhead_lines", "planned_land_use", "data_origin"]
REF_COLS = ["species_scientific", "common_name", "assamese_name", "plant_type", "native_to_assam", "soil_textures_ok",
            "waterlogging_tolerance", "shade_tolerance", "avoid_compacted", "max_height_m", "best_use_on_campus",
            "evidence_note", "source", "source_url", "verified_by", "data_origin"]
ORIGINS = ["observed", "inferred", "records", "public", "synthetic"]
ALLOWED_OBS = {"status": ["alive", "stressed", "dead", "missing"], "tree_guard": ["none", "damaged", "intact"],
               "physical_damage": ["none", "mowing", "trampling", "other"], "data_origin": ORIGINS}
ALLOWED_PATCH = {"status_for_study": ["past_plantation", "candidate"],
                 "aftercare_watering": ["none", "occasional", "scheduled", "unknown"],
                 "waterlogging_history": ["never", "after_heavy_rain", "most_of_monsoon", "unknown"],
                 "terrain_position": ["depression", "flat", "raised", "slope"],
                 "soil_texture": ["sandy", "loamy", "clayey"], "grazing_evidence": ["none", "some", "heavy"],
                 "fencing": ["none", "partial", "full"], "data_origin": ORIGINS}
OBS_BOOLS = ["grazing_damage", "waterlogged"]
PATCH_BOOLS = ["standing_water_now", "overhead_lines"]
STATUS_COLORS = {"alive": "#2E6B45", "stressed": "#8DB39A", "dead": "#6B6B6B", "missing": "#9A5B13"}
CLASS_COLORS = {"good": "#2E6B45", "risky": "#D9A441", "avoid": "#B23B3B"}
ACTIONS = {"unguarded": "install or repair tree guards", "grazing": "fence the patch or add guards",
           "physical_damage": "mark saplings and brief the mowing crew",
           "waterlogged": "improve drainage or avoid planting here", "deep_shade": "replant in open spots"}


def parse_bool(v):
    """Parse yes/no/true/false/1/0 into True/False, else NA."""
    s = str(v).strip().lower()
    return True if s in ("yes", "true", "1", "y", "t") else False if s in ("no", "false", "0", "n", "f") else pd.NA


def _true(v):
    """True only for a real True value (NA-safe)."""
    return bool(v) if pd.notna(v) else False


def _norm(df, cols, allowed, bools, nums):
    """Add missing columns, strip/lowercase categoricals, parse bools and numbers."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v).replace("", pd.NA)
    for c in allowed:
        df[c] = df[c].map(lambda v: v.lower() if isinstance(v, str) else v)
    for c in bools:
        df[c] = df[c].map(parse_bool).astype("boolean")
    for c in nums:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalise_obs(df):
    """Normalise observations, keeping only the known columns; add photo_key = {patch_id}_{sapling_id}."""
    df = df.rename(columns=clean_header)
    df = df.loc[:, ~df.columns.duplicated()]
    df = _norm(df, OBS_COLS, ALLOWED_OBS, OBS_BOOLS,
               ["lat", "lon", "canopy_open", "height_cm", "green_cover_pct", "yellow_leaf_pct", "bare_soil_pct", "soil_darkness"])
    df = df[OBS_COLS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["data_origin"] = df["data_origin"].fillna("observed")
    for c in ["sapling_id", "patch_id"]:
        df[c] = df[c].astype("string")
    df["photo_key"] = df["patch_id"] + "_" + df["sapling_id"]
    return df.reset_index(drop=True)


def normalise_patches(df):
    """Normalise patches table."""
    df = _norm(df if df is not None else pd.DataFrame(), PATCH_COLS, ALLOWED_PATCH, PATCH_BOOLS,
               ["centroid_lat", "centroid_lon", "count_planted", "compaction_depth_cm", "foot_traffic_count_10min"])
    df["patch_id"] = df["patch_id"].astype("string")
    return df


def load_ref(src=None):
    """Load species reference CSV (default: repo root file)."""
    df = pd.read_csv(src if src is not None else ROOT / "species_reference.csv", dtype=str, keep_default_na=False)
    df = _norm(df, REF_COLS, {k: 0 for k in ["plant_type", "native_to_assam", "waterlogging_tolerance",
                                             "shade_tolerance", "avoid_compacted"]}, [], ["max_height_m"])
    return df


def validate(obs, patches, ref, raw_cols=None):
    """Return a table of validation issues (flag only, never fix)."""
    rows = []
    cols = [clean_header(c) for c in raw_cols] if raw_cols is not None else None
    for c in REQUIRED:
        if cols is not None and c not in cols:
            rows.append(("observations", "missing required column", c, 1, ""))
    ignored = [c for c in (raw_cols if raw_cols is not None else []) if clean_header(c) not in OBS_COLS + ["photo_folder"]]
    if ignored:
        rows.append(("observations", "column not used (ignored)", ", ".join(map(str, ignored)), len(ignored), "see RUN.md for the column list"))
    for tbl, df, allowed in [("observations", obs, ALLOWED_OBS), ("patches", patches, ALLOWED_PATCH)]:
        for c, ok in allowed.items():
            bad = df[c].dropna()
            bad = bad[~bad.isin(ok)]
            if len(bad):
                rows.append((tbl, "value outside allowed list", c, len(bad), ", ".join(map(str, bad.unique()[:5]))))
    dup = obs.loc[obs["photo_plant"].notna() & obs["photo_plant"].duplicated(keep=False), "photo_key"]
    if len(dup):
        rows.append(("observations", "several rows matched the same photo (add _v1, _v2 ... to photo names)", "patch_id/sapling_id",
                     len(dup), ", ".join(dup.unique()[:5])))
    unknown = obs.loc[~obs["patch_id"].isin(patches["patch_id"]), "patch_id"].dropna()
    if len(patches) and len(unknown):
        rows.append(("observations", "patch_id not in patches.csv", "patch_id", len(unknown), ", ".join(unknown.unique()[:5])))
    nogps = obs[obs["lat"].isna() | obs["lon"].isna()]
    if len(nogps):
        rows.append(("observations", "missing lat/lon (not in CSV or photo EXIF)", "lat/lon", len(nogps), ", ".join(nogps["photo_key"].astype(str)[:5])))
    nots = obs[obs["timestamp"].isna()]
    if len(nots):
        rows.append(("observations", "missing or unreadable timestamp", "timestamp", len(nots), ", ".join(nots["photo_key"].astype(str)[:5])))
    nosrc = ref[ref["source"].isna()]
    if len(nosrc):
        rows.append(("species_reference", "missing source", "source", len(nosrc), ", ".join(nosrc["species_scientific"][:5])))
    return pd.DataFrame(rows, columns=["table", "issue", "column", "count", "examples"])


def summary(obs):
    """Headline counts for the loaded observations."""
    return {"rows": len(obs), "saplings": obs["sapling_uid"].nunique(), "patches": obs["patch_id"].nunique(),
            "places": obs["landmark"].nunique(), "visits": int(obs.groupby("sapling_uid").size().max()) if len(obs) else 0, "start": obs["timestamp"].min(), "end": obs["timestamp"].max(),
            "origin": obs["data_origin"].value_counts().to_dict()}


IMG_EXT = {".jpg", ".jpeg", ".png"}


def norm_place(v):
    """Normalise a place name so 'Academic Complex', 'academic_complex' and 'ACADEMIC-COMPLEX' match."""
    return re.sub(r"[^a-z0-9]", "", str(v).lower()) if isinstance(v, str) and v.strip() else ""


def index_photos(photo_dir):
    """Map lowercase file stem -> list of (path, place subfolder) for all images under a folder, including subfolders."""
    d = Path(str(photo_dir).strip().strip('"')) if str(photo_dir or "").strip() else None
    idx = {}
    if d and d.is_dir():
        for p in d.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                rel = p.relative_to(d).parts
                idx.setdefault(p.stem.lower(), []).append((p, rel[0] if len(rel) > 1 else ""))
    return idx


def visit_index(df):
    """1, 2, 3 ... for each row of the same patch_id + sapling_id, in timestamp order (then CSV order)."""
    t = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    key = df["patch_id"].astype(str).str.strip().str.lower() + "_" + df["sapling_id"].astype(str).str.strip().str.lower()
    order = pd.DataFrame({"t": t, "o": range(len(df)), "k": key.values}, index=df.index).sort_values(["t", "o"])
    return (order.groupby("k").cumcount() + 1).reindex(df.index)


def find_photo(idx, key, kind, landmark=None, visit=None):
    """key = {patch_id}_{sapling_id}. Plant: {key}_v{visit}.jpg, {key}.jpg or {key}_plant.jpg (optionally with _v{visit});
    sky/soil: {key}_v{visit}_sky.jpg or {key}_sky.jpg. With a landmark, only that place subfolder is used."""
    if not isinstance(key, str):
        return None
    k, v = key.strip().lower(), f"_v{int(visit)}" if visit and pd.notna(visit) else None
    names = ([f"{k}{v}", f"{k}{v}_plant"] if v else []) + [k, f"{k}_plant"] if kind == "plant" else \
            ([f"{k}{v}_{kind}"] if v else []) + [f"{k}_{kind}"]
    cands = next((idx[n] for n in names if n in idx), None)
    if not cands:
        return None
    place = norm_place(landmark)
    hits = [c for c in cands if norm_place(c[1]) == place] if place else cands
    return hits[0] if len(hits) == 1 or (hits and place) else None


def match_photos(obs, photo_dir=PHOTO_DIR):
    """Match each row's {patch_id}_{sapling_id} (inside its landmark subfolder) to image files; add paths and a report."""
    idx, obs = index_photos(photo_dir), obs.copy()
    obs["photo_visit"] = visit_index(obs)  # row n of a sapling -> photo {key}_v{n}.jpg
    for kind in ["plant", "sky", "soil"]:
        hits = [find_photo(idx, k, kind, lm, n) for k, lm, n in zip(obs["photo_key"], obs["landmark"], obs["photo_visit"])]
        obs[f"photo_{kind}"] = [str(h[0]) if h else None for h in hits]
        obs[f"photo_{kind}_found"] = [h is not None for h in hits]
        if kind == "plant":
            obs["photo_folder"] = [h[1] if h else None for h in hits]
    obs["photo_ground"] = obs["photo_soil"].fillna(obs["photo_plant"])  # soil photo if taken, else plant photo
    obs["landmark_origin"] = np.where(obs["landmark"].notna(), "csv", None)
    fill = obs["landmark"].isna() & obs["photo_folder"].fillna("").ne("")
    obs.loc[fill, "landmark"], obs.loc[fill, "landmark_origin"] = obs.loc[fill, "photo_folder"], "photo folder"
    obs["sapling_uid"] = obs["landmark"].map(norm_place).fillna("") + "/" + obs["photo_key"].fillna("")
    wrong = [k for k, lm, ok in zip(obs["photo_key"], obs["landmark"], obs["photo_plant_found"])
             if not ok and isinstance(k, str) and norm_place(lm) and (idx.get(k.lower()) or idx.get(k.lower() + "_plant"))]
    used = {str(p) for c in ["photo_plant", "photo_sky", "photo_soil"] for p in obs[c].dropna()}
    files = [(p, place) for lst in idx.values() for p, place in lst]
    return obs, {"n_files": len(files), "places": sorted({pl for _, pl in files if pl}),
                 "matched": sorted(obs.loc[obs["photo_plant_found"], "photo_key"].unique()),
                 "missing": sorted(obs.loc[~obs["photo_plant_found"], "photo_key"].dropna().unique()),
                 "wrong_folder": sorted(set(wrong)),
                 "unreferenced": sorted(f"{place}/{p.name}" if place else p.name for p, place in files if str(p) not in used)}


def exif_info(path):
    """(lat, lon, timestamp) from a photo's EXIF GPS/date tags, or Nones."""
    try:
        ex = Image.open(path).getexif()
        g = ex.get_ifd(0x8825)
        dms = lambda v, ref: (float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600) * (-1 if ref in ("S", "W") else 1)
        lat, lon = (dms(g[2], g.get(1)), dms(g[4], g.get(3))) if 2 in g and 4 in g else (None, None)
        ts = ex.get_ifd(0x8769).get(36867) or ex.get(306)
        return lat, lon, pd.to_datetime(ts, format="%Y:%m:%d %H:%M:%S", utc=True) if ts else None
    except Exception:
        return None, None, None


def fill_from_images(obs, patches):
    """Fill blank lat/lon/timestamp from the plant photo's EXIF."""
    obs = obs.copy()
    obs["gps_origin"] = np.where(obs["lat"].notna(), "csv", None)
    for i in obs.index[obs["photo_plant_found"] & (obs["lat"].isna() | obs["timestamp"].isna())]:
        lat, lon, ts = exif_info(obs.at[i, "photo_plant"])
        if pd.isna(obs.at[i, "lat"]) and lat is not None:
            obs.at[i, "lat"], obs.at[i, "lon"], obs.at[i, "gps_origin"] = lat, lon, "exif"
        if pd.isna(obs.at[i, "timestamp"]) and ts is not None:
            obs.at[i, "timestamp"] = ts
    return obs


def otsu_openness(path):
    """Percent sky pixels in a sky photo via Otsu threshold on the blue channel."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((512, 512))
    b = np.asarray(img)[:, :, 2].ravel()
    hist = np.bincount(b, minlength=256).astype(float)
    w = np.cumsum(hist); mu = np.cumsum(hist * np.arange(256)); tot = w[-1]
    between = (mu[-1] * w / tot - mu) ** 2 / (w * (tot - w) + 1e-9)
    return round(float((b > np.argmax(between)).mean() * 100), 1)


def fill_canopy(obs):
    """Compute canopy openness for blank rows with a matched sky photo."""
    obs = obs.copy()
    if "canopy_origin" not in obs:
        obs["canopy_origin"] = pd.NA
    for i in obs.index[obs["canopy_open"].isna() & obs["photo_sky_found"]]:
        obs.at[i, "canopy_open"] = otsu_openness(obs.at[i, "photo_sky"])
        obs.at[i, "canopy_origin"] = "inferred"
    return obs


def soil_color_class(r, g, b):
    """Name the mean soil colour; grey soil can mean waterlogging, red means iron-rich well-drained soil."""
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if v < 0.25: return "dark (organic or moist)"
    if s < 0.15: return "grey (possible waterlogging)" if v < 0.7 else "pale (sandy or dry)"
    if h * 360 < 20 or h * 360 > 340: return "red (iron-rich, well-drained)"
    return "brown" if h * 360 < 40 else "yellowish"


@lru_cache(maxsize=20000)
def image_features(path):
    """Cheap colour features from a ground/plant photo (no ML): vegetation, yellowing, bare soil, soil colour, darkness."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((256, 256))
    a = np.asarray(img).astype(float); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    hsv = np.asarray(img.convert("HSV")).astype(float) / 255; h, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    green = ((2 * g - r - b) > 20) & (g > r)  # Excess Green index; g > r keeps yellow leaves out
    yellow = (h >= 45) & (h < 70) & (sat > 0.45) & (val > 0.45) & ~green
    sky = (b > r) & (b > g) & (val > 0.5)
    soil = ~green & ~yellow & ~sky & (val > 0.08)
    foliage = green.sum() + yellow.sum()
    out = {"green_cover_pct": round(green.mean() * 100, 1), "yellow_leaf_pct": round(yellow.sum() / max(foliage, 1) * 100, 1),
           "bare_soil_pct": round(soil.mean() * 100, 1), "soil_color": None, "soil_darkness": None}
    if soil.mean() > 0.02:
        out["soil_color"] = soil_color_class(r[soil].mean(), g[soil].mean(), b[soil].mean())
        out["soil_darkness"] = round((1 - val[soil].mean()) * 100, 1)
    return out


def fill_image_features(obs):
    """Compute the pixel features for rows with a ground/plant photo where they are blank."""
    obs = obs.copy()
    for i in obs.index[obs["photo_ground"].notna() & obs["green_cover_pct"].isna()]:
        for k, v in image_features(obs.at[i, "photo_ground"]).items():
            obs.at[i, k] = v
    return obs


def fill_patch_soil(patches, obs):
    """Fill blank patch soil_texture with the most common photo soil_type_ai (sandy/loamy/clayey) in that patch."""
    p = patches.copy()
    p["soil_origin"] = np.where(p["soil_texture"].notna(), "survey", None)
    ai = obs[obs["soil_type_ai"].isin(["sandy", "loamy", "clayey"])].groupby("patch_id")["soil_type_ai"].agg(lambda x: x.mode().iloc[0])
    blank = p["soil_texture"].isna() & p["patch_id"].isin(ai.index)
    p.loc[blank, "soil_texture"] = p.loc[blank, "patch_id"].map(ai)
    p.loc[blank, "soil_origin"] = "photos (inferred)"
    return p


def latest(obs):
    """Latest record per sapling, by timestamp."""
    return obs.sort_values("timestamp", na_position="first").groupby("sapling_uid", as_index=False).tail(1)


def species_of(df):
    """Species label from the AI, else unknown."""
    return df["species_ai"].fillna("unknown")


def wilson(k, n, z=1.96):
    """95% Wilson score interval as (low, high) percent."""
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round((c - h) * 100, 1), round((c + h) * 100, 1))


def _survived(df):
    """Boolean series: alive or stressed."""
    return df["status"].isin(["alive", "stressed"])


def patch_summary(obs, patches):
    """Per-patch counts and survival with the denominator used."""
    lat = latest(obs)
    g = lat.groupby("patch_id")["status"].value_counts().unstack(fill_value=0)
    g = g.reindex(columns=ALLOWED_OBS["status"], fill_value=0)
    g["found"] = g.sum(axis=1)
    g = g.reset_index().merge(patches[["patch_id", "count_planted"]], on="patch_id", how="left")
    known = g["count_planted"].notna() & (g["count_planted"] > 0)
    g["denominator"] = np.where(known, "count_planted", "rows recorded")
    g["n"] = np.where(known, g["count_planted"], g["found"]).astype(int)
    g["survival_pct"] = ((g["alive"] + g["stressed"]) / g["n"] * 100).round(1)
    wet = ((lat["soil_type_ai"] == "waterlogged") | lat["soil_color"].fillna("").str.startswith("grey"))
    lat = lat.assign(photo_wet=wet.where(lat["soil_color"].notna() | lat["soil_type_ai"].notna()).astype(float) * 100)
    feats = lat.groupby("patch_id").agg(mean_canopy_pct=("canopy_open", "mean"), mean_green_cover_pct=("green_cover_pct", "mean"),
                                        mean_bare_soil_pct=("bare_soil_pct", "mean"), photo_wet_pct=("photo_wet", "mean")).round(1)
    return g.merge(feats, on="patch_id", how="left")


def _threats(df, patches=None):
    """Add boolean threat columns to latest-visit rows."""
    df = df.copy()
    df["unguarded"] = df["tree_guard"].isin(["none", "damaged"])
    df["grazing"] = df["grazing_damage"].fillna(False).astype(bool)
    df["physical_damage_any"] = df["physical_damage"].notna() & (df["physical_damage"] != "none")
    df["waterlogged_b"] = df["waterlogged"].fillna(False).astype(bool)
    df["deep_shade"] = df["canopy_open"] < 30
    return df


def conditions(obs, patches):
    """Survival with vs without each condition, with n and Wilson CI."""
    lat = _threats(latest(obs)).merge(patches[["patch_id", "compaction_depth_cm", "waterlogging_history"]],
                                      on="patch_id", how="left")
    s = _survived(lat)
    known = lambda cond, col: cond.where(lat[col].notna())  # rows without the measurement count on neither side
    conds = {"tree guard intact": lat["tree_guard"] == "intact", "grazing damage": lat["grazing"],
             "any physical damage": lat["physical_damage_any"], "waterlogged": lat["waterlogged_b"],
             "deep shade (canopy < 30%)": known(lat["deep_shade"], "canopy_open"),
             "hard soil (compaction < 5 cm)": known(lat["compaction_depth_cm"] < 5, "compaction_depth_cm"),
             "patch waterlogged most of monsoon": lat["waterlogging_history"] == "most_of_monsoon",
             "photo: grey soil (possible waterlogging)": known(lat["soil_color"].fillna("").str.startswith("grey"), "soil_color"),
             "photo: soil type looks waterlogged (AI)": known(lat["soil_type_ai"] == "waterlogged", "soil_type_ai"),
             "photo: yellow leaves > 20% of foliage": known(lat["yellow_leaf_pct"] > 20, "yellow_leaf_pct"),
             "photo: bare soil > 60%": known(lat["bare_soil_pct"] > 60, "bare_soil_pct"),
             "photo: green/weed cover > 50%": known(lat["green_cover_pct"] > 50, "green_cover_pct")}
    rows = []
    for name, m in conds.items():
        k = m.notna(); m = m.fillna(False).astype(bool); o = k & ~m
        kw, nw, ko, no = int(s[m].sum()), int(m.sum()), int(s[o].sum()), int(o.sum())
        pw = round(kw / nw * 100, 1) if nw else None
        po = round(ko / no * 100, 1) if no else None
        rows.append({"condition": name, "survival_with_pct": pw, "n_with": nw, "ci_with": wilson(kw, nw),
                     "survival_without_pct": po, "n_without": no, "ci_without": wilson(ko, no),
                     "gap_pts": round(pw - po, 1) if pw is not None and po is not None else None,
                     "note": "too small to conclude" if min(nw, no) < 10 else "linked with survival (not proof of cause)"})
    return pd.DataFrame(rows)


def species_survival(obs):
    """Survival by species (top 8 by n)."""
    lat = latest(obs).assign(species=lambda d: species_of(d), survived=lambda d: _survived(d))
    g = lat.groupby("species").agg(n=("survived", "size"), survived=("survived", "sum")).reset_index()
    g["survival_pct"] = (g["survived"] / g["n"] * 100).round(1)
    return g.sort_values("n", ascending=False).head(8)


def shade_band(v):
    """Shade band label from canopy openness."""
    return "unknown" if pd.isna(v) else "<30" if v < 30 else "30-60" if v <= 60 else ">60"


def classify_patches(patches, psum):
    """Rule-based good/risky/avoid class per patch with reasons."""
    p = patches.merge(psum[["patch_id", "survival_pct", "n", "mean_canopy_pct", "photo_wet_pct"]], on="patch_id", how="outer")
    out = []
    for _, r in p.iterrows():
        avoid, risky, big = [], [], pd.notna(r["n"]) and r["n"] >= 10
        if r["waterlogging_history"] == "most_of_monsoon": avoid.append("waterlogged most of monsoon")
        if r["grazing_evidence"] == "heavy" and r["fencing"] == "none": avoid.append("heavy grazing, no fencing")
        if pd.notna(r["planned_land_use"]): avoid.append(f"planned land use: {r['planned_land_use']}")
        if big and r["survival_pct"] < 30: avoid.append(f"past survival {r['survival_pct']}% (n={int(r['n'])})")
        if r["terrain_position"] == "depression": risky.append("depression")
        if _true(r["standing_water_now"]): risky.append("standing water now")
        if pd.notna(r["mean_canopy_pct"]) and r["mean_canopy_pct"] < 30: risky.append("deep shade")
        if pd.notna(r["compaction_depth_cm"]) and r["compaction_depth_cm"] < 5: risky.append("hard soil")
        if pd.notna(r["photo_wet_pct"]) and r["photo_wet_pct"] >= 30: risky.append(f"photos show wet/grey ground ({r['photo_wet_pct']}%)")
        if r["grazing_evidence"] in ("some", "heavy"): risky.append(f"grazing {r['grazing_evidence']}")
        if big and 30 <= r["survival_pct"] < 60: risky.append(f"past survival {r['survival_pct']}% (n={int(r['n'])})")
        cls = "avoid" if avoid else "risky" if risky else "good"
        out.append({"patch_id": r["patch_id"], "patch_name": r["patch_name"], "class": cls,
                    "reasons": "; ".join(avoid + risky) or "no risk flags", "color": CLASS_COLORS[cls],
                    "lat": r["centroid_lat"], "lon": r["centroid_lon"]})
    return pd.DataFrame(out)


def _ref_fit(r, p, canopy):
    """Check one reference species against one patch; return (passes, notes, avoid_reason, high_count)."""
    notes, fails, avoid, high = [], [], [], 0
    soil = r["soil_textures_ok"]
    if pd.isna(soil): notes.append("soil fit not documented")
    elif pd.notna(p["soil_texture"]) and p["soil_texture"] not in soil.split("|"): fails.append(f"soil {p['soil_texture']} not listed")
    else: notes.append("soil ok")
    wet = p["waterlogging_history"] in ("after_heavy_rain", "most_of_monsoon") or p["terrain_position"] == "depression" \
        or _true(p["standing_water_now"])
    if wet:
        need = ["high"] if p["waterlogging_history"] == "most_of_monsoon" else ["medium", "high"]
        wt = r["waterlogging_tolerance"]
        if wt in need: notes.append(f"waterlogging tolerance {wt}"); high += wt == "high"
        else:
            fails.append(f"{wt or 'unknown'} waterlogging tolerance")
            if wt in ("low", "medium"): avoid.append(f"{wt} waterlogging tolerance")  # undocumented is not 'avoid'
    if pd.notna(canopy) and canopy < 30:
        st = r["shade_tolerance"]
        if st in ("medium", "high"): notes.append(f"shade tolerance {st}"); high += st == "high"
        else: fails.append(f"{st or 'unknown'} shade tolerance")
    if pd.notna(p["compaction_depth_cm"]) and p["compaction_depth_cm"] < 5:
        if r["avoid_compacted"] == "yes": fails.append("avoid compacted soil"); avoid.append("avoid compacted soil")
        else: notes.append("compaction ok")
    if _true(p["overhead_lines"]):
        if r["plant_type"] in ("shrub", "grass") or (pd.notna(r["max_height_m"]) and r["max_height_m"] <= 8): notes.append("fits under lines")
        else: fails.append("height under lines unknown or > 8 m")
    for t, k in [("waterlogging_tolerance", "waterlogging"), ("shade_tolerance", "shade")]:
        if r[t] in (None, "unknown") or pd.isna(r[t]): notes.append(f"{k} tolerance undocumented")
    return not fails, "; ".join(notes + [f"FAIL: {f}" for f in fails]), "; ".join(avoid), high


def suggest_species(patches, classes, psum, obs, ref):
    """Proven-on-campus, reference-suitable and avoid lists for good/risky patches."""
    lat = latest(obs).assign(species=lambda d: species_of(d), survived=lambda d: _survived(d))
    info = patches.merge(psum[["patch_id", "mean_canopy_pct"]], on="patch_id", how="left")
    info["band"] = info["mean_canopy_pct"].map(shade_band)
    lat = lat.merge(info[["patch_id", "terrain_position", "band"]], on="patch_id", how="left")
    rows = []
    for _, p in info.merge(classes[["patch_id", "class"]], on="patch_id").iterrows():
        if p["class"] == "avoid":
            continue
        same = lat[(lat["terrain_position"] == p["terrain_position"]) & ((lat["band"] == p["band"]) | (p["band"] == "unknown"))]
        g = same[same["species"] != "unknown"].groupby("species")["survived"].agg(["size", "mean"])
        for sp, s in g[(g["size"] >= 5) & (g["mean"] >= 0.6)].iterrows():
            rows.append({"patch_id": p["patch_id"], "list": "proven on campus", "species": sp,
                         "fit_notes": f"survival {s['mean']*100:.0f}% (n={int(s['size'])}) on {p['terrain_position']} terrain, shade band {p['band']}"})
        fits = []
        for _, r in ref.iterrows():
            ok, notes, avoid, high = _ref_fit(r, p, p["mean_canopy_pct"])
            if ok: fits.append((r["native_to_assam"] != "yes", -high, r["species_scientific"], r, notes))
            if avoid:
                rows.append({"patch_id": p["patch_id"], "list": "avoid here", "species": r["species_scientific"], "fit_notes": avoid,
                             "source_url": r["source_url"]})
        for *_, r, notes in sorted(fits, key=lambda t: t[:3])[:5]:
            rows.append({"patch_id": p["patch_id"], "list": "suitable per reference (not yet tested here)",
                         "species": r["species_scientific"], "fit_notes": notes, "best_use_on_campus": r["best_use_on_campus"],
                         "evidence_note": r["evidence_note"], "source_url": r["source_url"], "verified_by": r["verified_by"]})
    cols = ["patch_id", "list", "species", "fit_notes", "best_use_on_campus", "evidence_note", "source_url", "verified_by"]
    return pd.DataFrame(rows).reindex(columns=cols)


def _need(d):
    """Need metrics for a group of latest-visit rows (with threat columns)."""
    threat = d["unguarded"] | d["grazing"] | d["physical_damage_any"] | d["waterlogged_b"]
    at_risk = int((_survived(d) & threat).sum())
    dead, missing, found = int((d["status"] == "dead").sum()), int((d["status"] == "missing").sum()), len(d)
    counts = {"unguarded": int(d["unguarded"].sum()), "grazing": int(d["grazing"].sum()),
              "physical_damage": int(d["physical_damage_any"].sum()), "waterlogged": int(d["waterlogged_b"].sum()),
              "deep_shade": int(d["deep_shade"].sum())}
    top = [k for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v > 0][:3]
    return {"found": found, "alive": int((d["status"] == "alive").sum()), "stressed": int((d["status"] == "stressed").sum()),
            "dead": dead, "missing": missing, "at_risk": at_risk, **counts,
            "need_score": round((dead + missing + at_risk) / max(found, 1), 2), "top_threats": ", ".join(top), "_top": top}


def _level(need, surv):
    """Gap level from need score and survival %."""
    return "endangered" if need >= 0.5 or surv < 40 else "needs attention" if need >= 0.25 else "stable"


def gap_analysis(obs, patches, psum):
    """Per past-plantation patch need score, level, threats and actions."""
    lat = _threats(latest(obs))
    past = patches.loc[patches["status_for_study"] == "past_plantation", "patch_id"]
    rows = []
    for pid in (past if len(past) else lat["patch_id"].unique()):
        d = lat[lat["patch_id"] == pid]
        if d.empty:
            continue
        m = _need(d); top = m.pop("_top")
        surv, surv_n = psum.loc[psum["patch_id"] == pid, ["survival_pct", "n"]].iloc[0]
        level = _level(m["need_score"], surv)
        rows.append({"patch_id": pid, "landmark": ", ".join(sorted(d["landmark"].dropna().unique())), **m,
                     "survival_pct": surv, "survival_n": int(surv_n), "level": level,
                     "actions": "; ".join(ACTIONS[t] for t in top) or ("less assistance needed" if level == "stable" else "")})
    return pd.DataFrame(rows).sort_values("need_score", ascending=False) if rows else pd.DataFrame()


def place_actions(obs, sug):
    """Action plan per landmark (place): survival, need level, threats, photo soil signs, what to do, species to plant."""
    lat = _threats(latest(obs)).assign(place=lambda d: d["landmark"].fillna("(no landmark)"))
    rows = []
    for place, d in lat.groupby("place"):
        m = _need(d); top = m.pop("_top")
        surv = round(_survived(d).mean() * 100, 1)
        level = _level(m["need_score"], surv)
        pids = sorted(d["patch_id"].dropna().unique())
        ok = sug[sug["patch_id"].isin(pids) & (sug["list"] != "avoid here")]["species"].drop_duplicates().head(5)
        bad = sug[sug["patch_id"].isin(pids) & (sug["list"] == "avoid here")]["species"].drop_duplicates().head(5)
        mode = lambda c: d[c].mode().iloc[0] if d[c].notna().any() else "no photo data"
        extra = []
        if (d["soil_type_ai"] == "waterlogged").mean() >= 0.3 or d["soil_color"].fillna("").str.startswith("grey").mean() >= 0.3:
            extra.append("photos show wet/grey soil: improve drainage, plant on raised mounds")
        if (d["green_cover_pct"] > 50).mean() >= 0.3: extra.append("heavy weed/grass cover: clear a 1 m ring around each sapling")
        if (d["yellow_leaf_pct"] > 20).mean() >= 0.3: extra.append("many saplings yellowing: check drainage, pests and nutrients")
        if (d["bare_soil_pct"] > 60).mean() >= 0.3: extra.append("mostly bare soil: mulch and protect from trampling/erosion")
        acts = [ACTIONS[t] for t in top] + extra
        rows.append({"landmark": place, "patches": ", ".join(pids), **m, "survival_pct": surv, "survival_n": len(d),
                     "level": level, "common_soil_color": mode("soil_color"), "common_soil_type_ai": mode("soil_type_ai"),
                     "what_to_do": "; ".join(acts) or "less assistance needed: routine checks only",
                     "species_to_plant": ", ".join(ok) or "not enough data", "species_to_avoid": ", ".join(bad)})
    return pd.DataFrame(rows).sort_values("need_score", ascending=False) if rows else pd.DataFrame()


def add_row_actions(obs):
    """Per-row 'action_needed' text from that sapling's recorded threats and photo signs."""
    t = _threats(obs)
    parts = [t["unguarded"].map({True: ACTIONS["unguarded"]}), t["grazing"].map({True: ACTIONS["grazing"]}),
             t["physical_damage_any"].map({True: ACTIONS["physical_damage"]}), t["waterlogged_b"].map({True: ACTIONS["waterlogged"]}),
             t["deep_shade"].map({True: ACTIONS["deep_shade"]}),
             (t["yellow_leaf_pct"] > 20).map({True: "check for pests, drainage and nutrients (yellowing)"}),
             (t["green_cover_pct"] > 50).map({True: "clear weeds around the sapling"}),
             t["status"].isin(["dead", "missing"]).map({True: "replant (gap-fill) before the next monsoon"})]
    obs = obs.copy()
    obs["action_needed"] = pd.concat(parts, axis=1).apply(lambda r: "; ".join(r.dropna()) or "none", axis=1)
    return obs


def run_pipeline(obs, patches, ref, photo_dir=PHOTO_DIR, log=print):
    """Match images, fill location from images, run all analyses; return dict of DataFrames."""
    log(f"Loaded {len(obs)} CSV rows; scanning image folder: {photo_dir}")
    obs, photos = match_photos(obs, photo_dir)
    log(f"Found {photos['n_files']} images in {len(photos['places'])} place folders ({', '.join(photos['places']) or 'none'}); "
        f"matched {len(photos['matched'])}/{obs['photo_key'].nunique()} saplings "
        f"({int(obs['photo_sky_found'].sum())} with sky photo); {len(photos['unreferenced'])} images not in CSV")
    obs = fill_from_images(obs, patches)
    log(f"GPS from photo EXIF: {int((obs['gps_origin'] == 'exif').sum())} rows; still no GPS: {int(obs['lat'].isna().sum())}")
    obs = fill_canopy(obs)
    log(f"Canopy openness computed from sky photos: {int((obs['canopy_origin'] == 'inferred').sum())} rows")
    obs = fill_image_features(obs)
    patches = fill_patch_soil(patches, obs)
    log(f"Image features present: {int(obs['green_cover_pct'].notna().sum())} rows; soil type (AI): "
        f"{int(obs['soil_type_ai'].notna().sum())} rows; patch soil filled from photos: {int((patches['soil_origin'] == 'photos (inferred)').sum())}")
    psum = patch_summary(obs, patches)
    classes = classify_patches(patches, psum)
    if len(photos["wrong_folder"]):
        log(f"WARNING: {len(photos['wrong_folder'])} saplings have photos only in a different place folder than their landmark")
    obs = add_row_actions(obs)
    sug = suggest_species(patches, classes, psum, obs, ref)
    places = place_actions(obs, sug)
    log(f"Analysis done: {obs['sapling_uid'].nunique()} saplings in {obs['patch_id'].nunique()} patches, {len(places)} places")
    return {"observations": obs, "patches": patches, "photos": photos, "patch_summary": psum,
            "conditions": conditions(obs, patches), "species_survival": species_survival(obs), "classes": classes,
            "species_suggestions": sug, "gap_analysis": gap_analysis(obs, patches, psum), "place_actions": places}


def gap_report_md(res):
    """Markdown gap analysis report."""
    obs, psum, gap, cond, sug = res["observations"], res["patch_summary"], res["gap_analysis"], res["conditions"], res["species_suggestions"]
    lat = latest(obs); s = summary(obs)
    n_surv = int(_survived(lat).sum())
    L = ["# Sapling Survival Gap Report — IIT Guwahati", "", "## Summary",
         f"- Saplings: {s['saplings']} in {s['patches']} patches, {s['rows']} rows, up to {s['visits']} visit(s) per sapling",
         f"- Survival (latest visit, alive+stressed): {n_surv / max(len(lat),1) * 100:.1f}% (n={len(lat)})",
         f"- Collection window: {s['start']} to {s['end']}", f"- Data origin: {s['origin']}",
         f"- Rows with AI species label: {int(obs['species_ai'].notna().sum())} (n={len(obs)}); "
         f"with image features: {int(obs['green_cover_pct'].notna().sum())}", ""]
    if (obs["data_origin"] == "synthetic").any():
        L.insert(1, "> **WARNING: contains synthetic test data — not real field results.**\n")
    for lvl in ["endangered", "needs attention", "stable"]:
        L += [f"## {lvl.title()}", ""]
        d = gap[gap["level"] == lvl] if len(gap) else gap
        L += [f"- **{r.patch_id}** ({r.landmark or 'no landmark'}) — need score {r.need_score}, survival {r.survival_pct}% (n={r.survival_n}); "
              f"threats: {r.top_threats or 'none'}; actions: {r.actions}" for r in d.itertuples()] or ["- none"]
        L.append("")
    L += ["## Action plan by place (landmark)", ""]
    for r in res["place_actions"].itertuples():
        L += [f"### {r.landmark} — {r.level}", f"- Patches: {r.patches}; survival {r.survival_pct}% (n={r.survival_n}); need score {r.need_score}",
              f"- Photo soil signs: {r.common_soil_color}; soil type (AI): {r.common_soil_type_ai}",
              f"- What to do: {r.what_to_do}", f"- Species to plant: {r.species_to_plant}"]
        L += [f"- Species to avoid: {r.species_to_avoid}"] if r.species_to_avoid else []
        L.append("")
    L += ["## Fixable causes (largest survival gaps, n ≥ 10 each side)", ""]
    big = cond[(cond["n_with"] >= 10) & (cond["n_without"] >= 10)].dropna(subset=["gap_pts"])
    big = big.reindex(big["gap_pts"].abs().sort_values(ascending=False).index)
    L += [f"- {r.condition}: {r.survival_with_pct}% with (n={r.n_with}) vs {r.survival_without_pct}% without "
          f"(n={r.n_without}) — linked with, not proven cause" for r in big.itertuples()] or ["- none with enough data"]
    L += ["", "## Species suggestions for candidate patches", ""]
    cand = res["patches"].loc[res["patches"]["status_for_study"] == "candidate", "patch_id"]
    for pid in cand:
        d = sug[sug["patch_id"] == pid]
        L.append(f"### {pid}")
        L += [f"- [{r.list}] {r.species} — {r.fit_notes}" + (f" (source: {r.source_url})" if isinstance(r.source_url, str) else "")
              for r in d.itertuples()] or ["- not enough data"]
        L.append("")
    L += ["## What this data cannot tell us", "",
          "- Which saplings died is recorded, but not when or why they died.",
          "- Only species that were actually planted can be compared on campus.",
          "- Reference suitability comes from literature and is not campus evidence.",
          "- AI species and photo soil features are unverified model/pixel estimates; staff should spot-check them.",
          "- Small groups (n < 10) are flagged and should not be used to conclude anything.",
          "- Conditions are linked with survival; they are not proven causes."]
    return "\n".join(L)


TIMELINES = ["Before planting", "0-3 months after planting", "3-12 months", "1-3 years", "3+ years"]
KNN_COLS = ["green_cover_pct", "yellow_leaf_pct", "bare_soil_pct", "soil_darkness"]


def land_status(f):
    """Current condition of a patch of land from one photo's features: list of (sign, meaning) and an overall rating."""
    signs = []
    if f.get("soil_type_ai") == "waterlogged" or str(f.get("soil_color") or "").startswith("grey"):
        signs.append(("wet / waterlogged ground", "roots may rot; needs drainage or water-tolerant species"))
    if f.get("soil_type_ai") == "cracked_clay":
        signs.append(("dry, cracked (compacted) clay", "hard for roots; loosen soil and water in dry spells"))
    if (f.get("green_cover_pct") or 0) > 50:
        signs.append(("heavy weed / grass cover", "competes with saplings for water and light"))
    if (f.get("bare_soil_pct") or 0) > 60:
        signs.append(("mostly bare soil", "erosion and trampling risk; mulch needed"))
    if (f.get("yellow_leaf_pct") or 0) > 20:
        signs.append(("yellowing foliage", "stress: check drainage, pests, nutrients"))
    rating = "good" if not signs else "fair" if len(signs) == 1 else "poor"
    return signs, rating


def predict_from_similar(f, obs, landmark=None, k=15):
    """Survival among the k most similar surveyed photos (nearest neighbours on pixel features) -> (pct, n, ci, neighbours)."""
    if obs is None:
        return None
    lat = latest(obs).dropna(subset=KNN_COLS)
    if landmark and norm_place(landmark):
        same = lat[lat["landmark"].map(norm_place) == norm_place(landmark)]
        lat = same if len(same) >= 5 else lat
    if lat.empty or any(f.get(c) is None for c in KNN_COLS):
        return None
    X = lat[KNN_COLS].to_numpy(float); mu, sd = X.mean(0), X.std(0) + 1e-9
    d = np.sqrt((((X - mu) / sd - (np.array([f[c] for c in KNN_COLS], float) - mu) / sd) ** 2).sum(1))
    nb = lat.iloc[np.argsort(d)[:min(k, len(lat))]].assign(survived=lambda x: _survived(x))
    return round(nb["survived"].mean() * 100, 1), len(nb), wilson(int(nb["survived"].sum()), len(nb)), nb


def land_species(f, ref, nb=None):
    """Reference species fit for the photographed land, plus species that survived among similar surveyed photos."""
    st = f.get("soil_type_ai")
    p = {"soil_texture": st if st in ("sandy", "loamy", "clayey") else "clayey" if st == "cracked_clay" else None,
         "waterlogging_history": "after_heavy_rain" if any("wet" in n for n, _ in land_status(f)[0]) else None,
         "terrain_position": None, "standing_water_now": False, "overhead_lines": False,
         "compaction_depth_cm": 4 if st == "cracked_clay" else None}
    fits, avoid = [], []
    for _, r in ref.iterrows():
        ok, notes, bad, high = _ref_fit(r, p, None)
        if ok: fits.append((r["native_to_assam"] != "yes", -high, r["species_scientific"], r, notes))
        if bad: avoid.append({"species": r["species_scientific"], "reason": bad})
    best = [{"species": r["species_scientific"], "common_name": r["common_name"], "why": notes,
             "best_use_on_campus": r["best_use_on_campus"], "source_url": r["source_url"]}
            for *_, r, notes in sorted(fits, key=lambda t: t[:3])[:5]]
    proven = []
    if nb is not None:
        g = nb[nb["species_ai"].notna()].groupby("species_ai")["survived"].agg(["size", "mean"])
        proven = [{"species": sp, "survived_pct": round(v["mean"] * 100), "n": int(v["size"])}
                  for sp, v in g[(g["size"] >= 3) & (g["mean"] >= 0.6)].iterrows()]
    return best, avoid, proven, p


def maintenance_plan(signs, timeline):
    """Rule-based weekly / monthly / yearly tasks for the land's condition and the saplings' age. General guidance only."""
    t = TIMELINES.index(timeline) if timeline in TIMELINES else 1
    plan = {"setup (one-time)": [], "weekly": [], "monthly": [], "yearly": []}
    if t == 0:
        plan["setup (one-time)"] += ["dig pits and add compost before the monsoon", "install tree guards / fencing before planting"]
    if t <= 1:
        plan["weekly"] += ["water if there has been no rain for 7 days", "check tree guards, stakes and grazing damage"]
        plan["monthly"] += ["clear weeds in a 1 m ring around each sapling", "top up mulch", "record status of every sapling"]
    elif t == 2:
        plan["weekly"] += ["quick walk-through for grazing, mowing or storm damage"]
        plan["monthly"] += ["weed the 1 m ring", "check leaves for yellowing or pests", "record status survey"]
        plan["yearly"] += ["replace dead saplings (gap-fill) just before the monsoon"]
    elif t == 3:
        plan["monthly"] += ["weed and check guards during the monsoon months"]
        plan["yearly"] += ["formative pruning", "gap-fill dead saplings", "repair fences and guards", "full survival survey"]
    else:
        plan["yearly"] += ["survival survey and health check", "prune dead or low branches", "remove guards once trunks are sturdy"]
    names = [n for n, _ in signs]
    if any("wet" in n for n in names):
        plan["setup (one-time)"] += ["plant on raised mounds or bunds; dig a drain to the nearest outlet"]
        plan["monthly"] += ["during the monsoon, clear drains and check for standing water"]
        plan["yearly"] += ["desilt drains before the monsoon"]
    if any("weed" in n for n in names):
        (plan["weekly"] if t <= 2 else plan["monthly"]).append("cut back weeds and grass around saplings")
    if any("bare" in n for n in names):
        plan["setup (one-time)"] += ["mulch the planting area; add a vetiver strip on slopes"]
        plan["monthly"] += ["re-mulch and repair any erosion"]
    if any("yellow" in n for n in names):
        plan["monthly"] += ["inspect yellowing saplings for pests, waterlogging or nutrient deficiency; add compost"]
    if any("cracked" in n for n in names):
        plan["setup (one-time)"] += ["loosen compacted soil and mix in compost before planting"]
        plan["weekly"] += ["water deeply during dry spells (cracked clay dries out fast)"]
    visits = 4 * bool(plan["weekly"]) + bool(plan["monthly"])
    level = "high" if len(plan["weekly"]) >= 3 or (t <= 1 and signs) else "medium" if plan["weekly"] or len(signs) >= 1 else "low"
    return plan, level, visits


def land_check_md(f, rating, signs, pred, best, avoid, proven, plan, level, visits, timeline):
    """Markdown summary of a land check."""
    L = ["# Land check", "", f"- Condition: **{rating}**; timeline: {timeline}; maintenance level: **{level}** (~{visits} visits/month)",
         "- Photo features: " + ", ".join(f"{k}={v}" for k, v in f.items() if v is not None)]
    L += [f"- Sign: {a} — {b}" for a, b in signs] or ["- No problem signs detected in the photo"]
    if pred:
        L.append(f"- Survival among the {pred[1]} most similar surveyed photos: {pred[0]}% (95% CI {pred[2][0]}–{pred[2][1]}%, n={pred[1]})")
    L += ["", "## Species"] + [f"- Proven on similar campus land: {p['species']} ({p['survived_pct']}% of n={p['n']})" for p in proven]
    L += [f"- Suitable per reference: {b['species']} ({b['common_name']}) — {b['why']} ({b['source_url']})" for b in best] or ["- not enough data"]
    L += [f"- Avoid: {a['species']} — {a['reason']}" for a in avoid]
    for k, v in plan.items():
        L += ["", f"## {k.title()}"] + [f"- {x}" for x in v] if v else []
    L += ["", "_Rule-based guidance from photo features and campus survey data; confirm with the horticulture team._"]
    return "\n".join(L)


def _arrow_safe(df):
    """Convert object columns to strings so parquet writes cleanly."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: None if v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)) else str(v))
    return df


def export(res, out_dir=OUT_DIR):
    """Write parquet outputs and return a zip (bytes) with parquet, CSV samples and schema.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["observations", "patches", "patch_summary", "conditions", "gap_analysis", "species_suggestions", "place_actions"]
    buf, schema = io.BytesIO(), ["# Schema", ""]
    with zipfile.ZipFile(buf, "w") as z:
        for n in names:
            df = _arrow_safe(res[n])
            df.to_parquet(out_dir / f"{n}.parquet", index=False)
            z.write(out_dir / f"{n}.parquet", f"{n}.parquet")
            z.writestr(f"samples/{n}_sample.csv", df.head(20).to_csv(index=False))
            schema += [f"## {n}.parquet ({len(df)} rows)", ""] + [f"- `{c}`: {t}" for c, t in df.dtypes.astype(str).items()] + [""]
        z.writestr("place_actions.csv", _arrow_safe(res["place_actions"]).to_csv(index=False))  # what to do at each place
        keep = [c for c in CSV_COLS + FEATURES + ["photo_folder", "action_needed"] if c in res["observations"]]
        z.writestr("observations_with_actions.csv", _arrow_safe(res["observations"][keep]).to_csv(index=False))
        z.writestr("gap_report.md", gap_report_md(res))
        z.writestr("schema.md", "\n".join(schema))
    return buf.getvalue()


def load_sample():
    """Load the synthetic sample (generating it if needed); its images are in sample_data/photos."""
    if not (SAMPLE_DIR / "observations.csv").exists():
        import make_sample; make_sample.main()
    raw = pd.read_csv(SAMPLE_DIR / "observations.csv", dtype=str, keep_default_na=False)
    return raw, pd.read_csv(SAMPLE_DIR / "patches.csv", dtype=str, keep_default_na=False)


if __name__ == "__main__":
    raw, pr = load_sample()
    obs, patches, ref = normalise_obs(raw), normalise_patches(pr), load_ref()
    res = run_pipeline(obs, patches, ref, SAMPLE_DIR / "photos")
    print("Validation issues:\n", validate(res["observations"], patches, ref, raw.columns).to_string())
    print("\nPatch classes:\n", res["classes"][["patch_id", "class", "reasons"]].to_string(index=False))
    print("\nSpecies suggestions:\n", res["species_suggestions"][["patch_id", "list", "species"]].to_string(index=False))
    print("\nGap levels:\n", res["gap_analysis"][["patch_id", "need_score", "survival_pct", "level"]].to_string(index=False))
    export(res); print("\nExported to", OUT_DIR)
