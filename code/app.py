"""Streamlit UI for the Sapling Survival Audit. The model runs in a separate process (ai_batch.py), never in this app."""
import importlib, os, subprocess, sys, time
import pandas as pd
import streamlit as st
import core

if getattr(core, "_mtime", None) != os.path.getmtime(core.__file__):  # pick up edits to core.py without a restart
    core = importlib.reload(core)
    core._mtime = os.path.getmtime(core.__file__)

st.set_page_config(page_title="Sapling Survival Audit — IIT Guwahati", layout="wide")
ss = st.session_state
ss.setdefault("log", [])
ss.setdefault("photo_dir", "")
AI_IN, AI_OUT = core.ROOT / "data" / "observations_input.csv", core.ROOT / "data" / "observations_ai.csv"
UPLOADS = core.ROOT / "data" / "uploads"

h1, h2, h3 = st.columns([5, 1, 1])
h1.title(":green[Sapling Survival Audit — IIT Guwahati]")
show_log = h2.toggle("Show live log", value=True)
log_w = h3.slider("Log width %", 20, 50, 28, disabled=not show_log)
if show_log:  # live log panel fixed to the right edge; scrolls on its own, main page makes room for it
    st.markdown(f"""<style>
.st-key-livelog {{position: fixed; top: 4.5rem; right: 1rem; width: {log_w}vw; height: calc(100vh - 6rem);
  overflow-y: auto; z-index: 999; background: #FFFFFF; border: 2px solid #2E6B45; border-radius: 8px; padding: 0.6rem;}}
[data-testid="stMainBlockContainer"] {{padding-right: calc({log_w}vw + 2.5rem) !important;}}
</style>""", unsafe_allow_html=True)


def read_csv(f):
    """Read an uploaded CSV as strings."""
    return pd.read_csv(f, dtype=str, keep_default_na=False)


def show():
    """Redraw the live log (newest line first)."""
    if show_log:
        log_box.code("\n".join(reversed(ss.log[-400:])) or "idle", language="text", wrap_lines=True)


def log(msg):
    """Append a timestamped line to the live log and redraw it."""
    ss.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    show()


def load_sample_cb():
    """Load the synthetic sample and point the folder box at its images."""
    ss.raw_obs, ss.raw_patches = core.load_sample()
    ss.photo_dir = str(core.SAMPLE_DIR / "photos")
    ss.log.append(f"[{time.strftime('%H:%M:%S')}] Loaded synthetic sample")


def stream(cmd):
    """Run a command, streaming its output into the live log; return (exit code, output lines)."""
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", cwd=core.ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines = []
    for line in proc.stdout:
        if line.strip() and "unauthenticated requests to the HF Hub" not in line:  # harmless download notice
            lines.append(line.rstrip().split("\r")[-1]); log(lines[-1])
    return proc.wait(), lines


def run_ai():
    """Run ai_batch.py on the CSV + folder, then load its result."""
    AI_IN.parent.mkdir(parents=True, exist_ok=True)
    ss.raw_obs.to_csv(AI_IN, index=False)
    code, _ = stream([sys.executable, "-u", str(core.ROOT / "ai_batch.py"), str(AI_IN), ss.photo_dir.strip().strip('"'), "-o", str(AI_OUT)])
    if code == 0:
        ss.raw_obs = pd.read_csv(AI_OUT, dtype=str, keep_default_na=False)
        log("AI matching finished; results loaded into the app")
    else:
        log(f"AI batch failed (exit {code}). If torch/open_clip are missing: pip install -r requirements-ai.txt")


def land_check(obs, ref):
    """Tab 8: upload one photo of a patch of land -> condition, prediction, species and maintenance plan."""
    st.caption("Upload a new photo of a patch of land. The app reads its soil and vegetation, compares it with the surveyed photos, "
               "and suggests species and a maintenance schedule. Rule-based guidance: confirm with the horticulture team.")
    c1, c2, c3 = st.columns(3)
    up = c1.file_uploader("Photo of the land (jpg/png)", type=["jpg", "jpeg", "png"], key="land_photo")
    timeline = c2.selectbox("Timeline", core.TIMELINES, index=0)
    places = sorted(obs["landmark"].dropna().unique()) if obs is not None else []
    place = c3.selectbox("Compare with place (optional)", ["all places"] + places)
    use_ai = c3.checkbox("Also run BioCLIP soil type (needs requirements-ai.txt)", value=True)
    if not up:
        return
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / up.name
    path.write_bytes(up.getbuffer())
    f = dict(core.image_features(str(path)), soil_type_ai=None)
    if use_ai:
        if ss.get("land_ai_for") != (up.file_id,):
            code, lines = stream([sys.executable, "-u", str(core.ROOT / "ai_batch.py"), "--single", str(path)])
            res = next((l[7:] for l in lines if l.startswith("RESULT ")), "")
            ss.land_ai, ss.land_ai_for = dict(kv.split("=", 1) for kv in res.split(";") if "=" in kv), (up.file_id,)
            if code: log("Soil type model not available; using pixel features only")
        f["soil_type_ai"] = ss.land_ai.get("soil_type_ai")
    signs, rating = core.land_status(f)
    pred = core.predict_from_similar(f, obs, None if place == "all places" else place)
    best, avoid, proven, _ = core.land_species(f, ref, pred[3] if pred else None)
    plan, level, visits = core.maintenance_plan(signs, timeline)
    log(f"Land check {up.name}: condition {rating}, maintenance {level}")
    a, b = st.columns([1, 2])
    a.image(str(path), width="stretch")
    b.metric("Current condition of the land", rating.upper())
    b.write("**Photo features:** " + " · ".join(f"{k}: {v}" for k, v in f.items() if v is not None))
    for s_, why in signs:
        b.warning(f"**{s_}**: {why}")
    if not signs:
        b.success("No problem signs detected in the photo.")
    if pred:
        b.info(f"Saplings on the **{pred[1]} most similar surveyed photos** survived **{pred[0]}%** of the time "
               f"(95% CI {pred[2][0]}–{pred[2][1]}%, n={pred[1]}). Based on your survey data, not a guarantee.")
    else:
        b.info("Load survey data with image features (tab 1) to predict survival from similar photos.")
    st.subheader(":green[Best suited species for this land]")
    if proven:
        st.write("**Survived well on similar campus land**"); st.dataframe(pd.DataFrame(proven), hide_index=True)
    st.write("**Suitable per reference (not yet tested here)**")
    if best:
        st.dataframe(pd.DataFrame(best), hide_index=True)
    else:
        st.write("not enough data")
    if avoid:
        st.write("**Avoid here**"); st.dataframe(pd.DataFrame(avoid), hide_index=True)
    st.subheader(f":green[Maintenance plan: {level} (~{visits} visits/month)]")
    cols = st.columns(4)
    for col, (k, v) in zip(cols, plan.items()):
        col.write(f"**{k.title()}**")
        col.markdown("\n".join(f"- {x}" for x in v) or "- nothing extra")
    st.download_button("Download land check (.md)", core.land_check_md(f, rating, signs, pred, best, avoid, proven, plan, level, visits, timeline),
                       f"land_check_{path.stem}.md", "text/markdown")


banner = st.container()
tabs = st.tabs(["1 · Upload", "2 · AI & features", "3 · Map", "4 · Survival", "5 · Patch plan", "6 · Gap report", "7 · Export", "8 · Land check"])
if show_log:
    with st.container(key="livelog"):
        st.markdown("**:green[Live log]** (newest first)")
        log_box = st.empty()
show()

with tabs[0]:
    st.caption("Upload the survey CSV and type the image folder path. Photos are found inside place subfolders (e.g. barak, kameng) "
               "and matched to each row as {patch_id}_{sapling_id}.jpg.")
    c1, c2 = st.columns(2)
    fo = c1.file_uploader("observations.csv (required)", type="csv")
    c1.text_input("Image folder path (required for photos)", key="photo_dir",
                  placeholder=r"e.g. D:\survey\photos  (contains subfolders like barak, kameng, core5, academic complex)")
    pdir = ss.photo_dir.strip().strip('"')
    if not pdir:
        c1.info("Enter the folder where the survey photos are stored.")
    elif not os.path.isdir(pdir):
        c1.error("Folder not found. Check the path.")
    c2.markdown("### OPTIONAL: patches.csv\nLeave empty if you have no patch survey. Patch-level rules then use only the photo data.")
    fp = c2.file_uploader("patches.csv (optional)", type="csv")
    c2.markdown("### OPTIONAL: species_reference.csv\nLeave empty to use the **default species list** that comes with the app.")
    fr = c2.file_uploader("species_reference.csv (optional; default list used)", type="csv")
    b1, b2 = st.columns(2)
    b1.button("Load synthetic sample", on_click=load_sample_cb)
    for key, f, name in [("raw_obs", fo, "observations"), ("raw_patches", fp, "patches")]:
        if f and ss.get(f"{key}_file") != f.file_id:
            ss[key], ss[f"{key}_file"] = read_csv(f), f.file_id
            log(f"Uploaded {name} CSV: {f.name} ({len(ss[key])} rows)")
    ss.ref = core.load_ref(fr) if fr else core.load_ref()
    if b2.button("Run AI matching (BioCLIP)", disabled="raw_obs" not in ss or not os.path.isdir(pdir or "\0")):
        run_ai()

if "raw_obs" not in ss:
    for t in tabs[:7]:
        with t:
            st.info("Upload observations.csv or click 'Load synthetic sample' in tab 1.")
    with tabs[7]:
        land_check(None, ss.ref)
    st.stop()

msgs = []
obs, patches, ref = core.normalise_obs(ss.raw_obs), core.normalise_patches(ss.get("raw_patches")), ss.ref
res = core.run_pipeline(obs, patches, ref, ss.photo_dir, log=msgs.append)
if ss.get("last_run") != (id(ss.raw_obs), id(ss.get("raw_patches")), ss.photo_dir):
    ss.last_run = (id(ss.raw_obs), id(ss.get("raw_patches")), ss.photo_dir)
    for m in msgs:
        log(m)
obs = res["observations"]
if (obs["data_origin"] == "synthetic").any():
    banner.warning("⚠️ Synthetic test data loaded — these are NOT real field results.")

with tabs[0]:
    s = core.summary(obs)
    m = st.columns(6)
    for col, (k, v) in zip(m, [("Rows", s["rows"]), ("Saplings", s["saplings"]), ("Patches", s["patches"]), ("Places", s["places"]),
                               ("Max visits per sapling", s["visits"]), ("Reference species", len(ref))]):
        col.metric(k, v)
    st.write(f"Collection window: **{s['start']}** → **{s['end']}** · data_origin split: {s['origin']}")
    st.subheader(":green[Validation]")
    v = core.validate(obs, patches, ref, ss.raw_obs.columns)
    if len(v):
        st.dataframe(v, hide_index=True)
    else:
        st.success("No validation issues found.")
    st.subheader(":green[Photo matching]")
    ph = res["photos"]
    st.write(f"{ph['n_files']} images in folder · place subfolders: {', '.join(ph['places']) or 'none'}")
    a, b, c = st.columns(3)
    a.write(f"**Matched ({len(ph['matched'])})**"); a.dataframe(pd.DataFrame({"patch_sapling": ph["matched"]}), hide_index=True)
    b.write(f"**In CSV, no photo in folder ({len(ph['missing'])})**"); b.dataframe(pd.DataFrame({"patch_sapling": ph["missing"]}), hide_index=True)
    c.write(f"**In folder, not in CSV ({len(ph['unreferenced'])})**"); c.dataframe(pd.DataFrame({"file": ph["unreferenced"]}), hide_index=True)
    if ph["wrong_folder"]:
        st.warning(f"{len(ph['wrong_folder'])} saplings have photos only in a different place folder than their landmark: "
                   + ", ".join(ph["wrong_folder"][:10]))
    st.write("**Where location and place came from**")
    st.dataframe(obs[["patch_id", "sapling_id", "landmark", "landmark_origin", "photo_folder", "lat", "lon", "gps_origin", "timestamp"]], hide_index=True)

with tabs[1]:
    st.caption("Species and soil type come from BioCLIP ('Run AI matching' in tab 1); pixel features are computed from the ground/plant photo. All are unverified estimates.")
    if obs["species_ai"].isna().all() and obs["green_cover_pct"].isna().all():
        st.info("Click 'Run AI matching (BioCLIP)' in tab 1, or run `python ai_batch.py observations.csv photos/ -o observations_ai.csv` and upload `observations_ai.csv`.")
    else:
        n = len(obs)
        m = st.columns(3)
        m[0].metric("Rows with species_ai", f"{int(obs['species_ai'].notna().sum())} / {n}")
        m[1].metric("Rows with soil_type_ai", f"{int(obs['soil_type_ai'].notna().sum())} / {n}")
        m[2].metric("Rows with pixel features", f"{int(obs['green_cover_pct'].notna().sum())} / {n}")
        c1, c2, c3 = st.columns(3)
        for col, k in zip([c1, c2, c3], ["species_ai", "soil_type_ai", "soil_color"]):
            col.write(f"**{k}**")
            vc = obs[k].value_counts()
            if len(vc):
                col.bar_chart(vc)
            else:
                col.write("no data yet")
        st.subheader(":green[Image features by patch (mean)]")
        st.dataframe(obs.groupby("patch_id")[["green_cover_pct", "yellow_leaf_pct", "bare_soil_pct", "soil_darkness", "canopy_open"]]
                     .mean().round(1).assign(n=obs.groupby("patch_id").size()), )
        st.subheader(":green[Photos with their features (first 20)]")
        for _, r in obs[obs["photo_ground"].notna()].head(20).iterrows():
            c1, c2 = st.columns([1, 5])
            c1.image(r["photo_ground"], width=90)
            c2.write(f"`{r['photo_key']}` · species: *{r['species_ai'] if pd.notna(r['species_ai']) else '—'}* · soil: {r['soil_type_ai'] if pd.notna(r['soil_type_ai']) else '—'}, "
                     f"{r['soil_color']} · green {r['green_cover_pct']}% · yellow {r['yellow_leaf_pct']}% · bare {r['bare_soil_pct']}% · darkness {r['soil_darkness']}")

with tabs[2]:
    st.caption("Latest visit per sapling, coloured by status (alive, stressed, dead, missing).")
    lat = core.latest(obs).assign(species=lambda d: core.species_of(d))
    f0, f1, f2 = st.columns(3)
    lf = f0.multiselect("Place (landmark)", sorted(lat["landmark"].dropna().unique()))
    pf = f1.multiselect("Patch", sorted(lat["patch_id"].dropna().unique()))
    sf = f2.multiselect("Species", sorted(lat["species"].unique()))
    lat = lat[lat["landmark"].isin(lf or lat["landmark"]) & lat["patch_id"].isin(pf or lat["patch_id"]) & lat["species"].isin(sf or lat["species"])].copy()
    lat["color"] = lat["status"].map(core.STATUS_COLORS).fillna("#000000")
    st.map(lat.dropna(subset=["lat", "lon"]), latitude="lat", longitude="lon", color="color")
    st.write(" · ".join(f"{k}: {c}" for k, c in core.STATUS_COLORS.items()))
    st.dataframe(lat[["landmark", "patch_id", "sapling_id", "timestamp", "status", "species", "tree_guard", "canopy_open", "action_needed"]], hide_index=True)
    sid = st.selectbox("Show photos for sapling (patch_sapling)", [""] + list(lat["photo_key"]))
    if sid:
        r = lat[lat["photo_key"] == sid].iloc[0]
        for col, k in zip(st.columns(2), ["photo_plant", "photo_sky"]):
            p = r[k]
            if isinstance(p, str):
                col.image(p, caption=os.path.basename(p), width=300)
            else:
                col.write(f"{k}: not in folder")

with tabs[3]:
    st.caption("Survival = (alive + stressed) ÷ planted if known, else ÷ rows recorded. Conditions are linked with survival, not proven causes.")
    st.dataframe(res["patch_summary"], hide_index=True)
    st.subheader(":green[Conditions (latest visit)]")
    st.dataframe(res["conditions"], hide_index=True)
    st.subheader(":green[Survival by species (top 8)]")
    sp = res["species_survival"]
    st.dataframe(sp, hide_index=True)
    st.bar_chart(sp.assign(label=sp["species"] + " (n=" + sp["n"].astype(str) + ")").set_index("label")["survival_pct"])

with tabs[4]:
    st.caption("Rule-based class per patch with reasons, plus species suggestions from campus data and the sourced reference table.")
    cl = res["classes"]
    st.dataframe(cl.drop(columns=["color", "lat", "lon"]).style.map(
        lambda v: f"background-color: {core.CLASS_COLORS[v]}; color: white" if v in core.CLASS_COLORS else "", subset=["class"]), hide_index=True)
    st.map(cl.dropna(subset=["lat", "lon"]), latitude="lat", longitude="lon", color="color", size=40)
    sug = res["species_suggestions"]
    for _, p in cl[cl["class"] != "avoid"].iterrows():
        st.markdown(f"### :green[{p['patch_id']} — {p['class']}]")
        d = sug[sug["patch_id"] == p["patch_id"]]
        if d[d["list"] != "avoid here"].empty:
            st.write("not enough data")
        for name in ["proven on campus", "suitable per reference (not yet tested here)", "avoid here"]:
            x = d[d["list"] == name].dropna(axis=1, how="all")
            if len(x):
                st.write(f"**{'Species to avoid here' if name == 'avoid here' else name.capitalize()}**")
                st.dataframe(x.drop(columns=["patch_id", "list"]), hide_index=True)

with tabs[5]:
    st.caption("Where saplings are endangered and where less help is needed: first by place (landmark), then by patch.")
    st.subheader(":green[Action plan by place]")
    pa = res["place_actions"]
    st.dataframe(pa[["landmark", "level", "survival_pct", "survival_n", "need_score", "what_to_do", "species_to_plant", "species_to_avoid",
                     "common_soil_color", "common_soil_type_ai", "patches"]], hide_index=True)
    st.download_button("Download place_actions.csv", core._arrow_safe(pa).to_csv(index=False), "place_actions.csv", "text/csv")
    st.subheader(":green[By patch]")
    st.dataframe(res["gap_analysis"], hide_index=True)
    md = core.gap_report_md(res)
    st.download_button("Download gap_report.md", md, "gap_report.md", "text/markdown")
    st.markdown(md)

with tabs[6]:
    st.caption("One zip: parquet tables, place_actions.csv (what to do at each place), observations_with_actions.csv "
               "(every sapling with its landmark and action_needed), gap_report.md, CSV samples and schema.md.")
    if st.button("Build export"):
        ss.zip = core.export(res)
        st.success(f"Wrote parquet files to {core.OUT_DIR}")
    if "zip" in ss:
        st.download_button("Download export zip", ss.zip, "sapling_audit_export.zip", "application/zip")

with tabs[7]:
    land_check(obs, ref)
