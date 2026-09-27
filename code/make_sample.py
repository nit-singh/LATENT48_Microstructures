"""Write a small SYNTHETIC sample (test data only) to sample_data/."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).parent / "sample_data"


PLACES = {"P01": "Barak", "P02": "Kameng", "P03": "Core 5", "P04": "Academic Complex"}  # synthetic place names
FOLDERS = {"P01": "barak", "P02": "kameng", "P03": "core5", "P04": "academic complex"}  # photo subfolders per place


def main():
    """Generate synthetic patches, observations and 3 tiny test images."""
    rng = np.random.default_rng(42)
    species = [s.strip() for s in (Path(__file__).parent / "species_list.txt").read_text().splitlines() if s.strip()]
    patches = pd.DataFrame([
        ["P01", "Synthetic raised lawn", "past_plantation", 26.1900, 91.6900, 20, "2025-07-01", "scheduled", "never", "raised", "loamy", 12, "none", "full", "no", 5, "no", ""],
        ["P02", "Synthetic low field", "past_plantation", 26.1920, 91.6930, "", "2025-07-05", "occasional", "after_heavy_rain", "depression", "clayey", 4, "some", "partial", "yes", 12, "no", ""],
        ["P03", "Synthetic lake edge", "past_plantation", 26.1880, 91.6950, 22, "2025-07-10", "none", "most_of_monsoon", "flat", "clayey", 8, "heavy", "none", "no", 30, "yes", ""],
        ["P04", "Synthetic candidate plot", "candidate", 26.1935, 91.6880, "", "", "unknown", "never", "raised", "loamy", 10, "none", "partial", "no", 3, "yes", ""],
    ], columns=["patch_id", "patch_name", "status_for_study", "centroid_lat", "centroid_lon", "count_planted",
                "last_planting_date", "aftercare_watering", "waterlogging_history", "terrain_position", "soil_texture",
                "compaction_depth_cm", "grazing_evidence", "fencing", "standing_water_now", "foot_traffic_count_10min",
                "overhead_lines", "planned_land_use"])
    patches["data_origin"] = "synthetic"
    probs = {"P01": [0.75, 0.1, 0.1, 0.05], "P02": [0.4, 0.2, 0.3, 0.1], "P03": [0.15, 0.1, 0.5, 0.25]}
    pool = {"P01": species[:4], "P02": species[1:5], "P03": species[:3]}
    rows = []
    for pid, (_, p) in zip(probs, patches.head(3).iterrows()):
        for i in range(1, 21):
            sp = rng.choice(pool[pid])
            for v in ([1, 2] if i <= 5 else [1]):
                status = rng.choice(["alive", "stressed", "dead", "missing"], p=probs[pid])
                ai, feat = rng.random() < 0.6, rng.random() < 0.7
                bad = status in ("stressed", "dead")
                soil = {"P01": ["brown", "red (iron-rich, well-drained)"], "P02": ["grey (possible waterlogging)", "brown"],
                        "P03": ["grey (possible waterlogging)", "dark (organic or moist)"]}[pid]
                stype = {"P01": ["loamy", "grass_covered"], "P02": ["clayey", "waterlogged"], "P03": ["waterlogged", "clayey"]}[pid]
                rows.append({
                    "patch_id": pid, "status": status, "sapling_id": f"S{i:03d}", "landmark": PLACES[pid],
                    "timestamp": f"2026-{8 + v:02d}-{10 + i % 15:02d}T09:{i:02d}:00",
                    "lat": round(p.centroid_lat + rng.normal(0, 0.0003), 6), "lon": round(p.centroid_lon + rng.normal(0, 0.0003), 6),
                    "species_ai": (sp if rng.random() < 0.8 else rng.choice(species)) if ai else "",
                    "tree_guard": rng.choice(["none", "damaged", "intact"], p=[0.05, 0.05, 0.9] if pid == "P01" else [0.4, 0.3, 0.3]),
                    "grazing_damage": rng.choice(["yes", "no"], p=[0.05, 0.95] if pid == "P01" else [0.4, 0.6]),
                    "physical_damage": rng.choice(["none", "mowing", "trampling", "other"], p=[0.95, 0.05, 0, 0] if pid == "P01" else [0.6, 0.2, 0.15, 0.05]),
                    "waterlogged": "yes" if pid != "P01" and rng.random() < 0.4 else "no",
                    "canopy_open": round(float(rng.uniform(15, 90)), 1) if rng.random() < 0.8 else "",
                    "height_cm": round(float(rng.uniform(30, 150)), 0), "data_origin": "synthetic", "notes": "synthetic test row",
                    "green_cover_pct": round(float(rng.uniform(10, 70)), 1) if feat else "",
                    "yellow_leaf_pct": round(float(rng.uniform(15, 60) if bad else rng.uniform(0, 20)), 1) if feat else "",
                    "bare_soil_pct": round(float(rng.uniform(20, 85)), 1) if feat else "",
                    "soil_color": rng.choice(soil, p=[0.7, 0.3]) if feat else "",
                    "soil_darkness": round(float(rng.uniform(30, 80)), 1) if feat else "",
                    "soil_type_ai": rng.choice(stype, p=[0.7, 0.3]) if feat else ""})
    obs = pd.DataFrame(rows)
    obs = obs.astype(object)  # row 0: app fills these from its photos (EXIF, sky photo, pixel features)
    obs.loc[0, ["canopy_open", "lat", "lon", "timestamp", "green_cover_pct", "yellow_leaf_pct", "bare_soil_pct", "soil_color", "soil_darkness"]] = ""
    patches.loc[patches["patch_id"] == "P02", "soil_texture"] = ""  # app fills it from photo soil_type_ai
    (OUT / "photos").mkdir(parents=True, exist_ok=True)
    import shutil; shutil.rmtree(OUT / "photos", ignore_errors=True)  # sample folder only holds generated test images
    for f in FOLDERS.values(): (OUT / "photos" / f).mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT / "observations.csv", index=False)
    patches.to_csv(OUT / "patches.csv", index=False)
    ex = Image.Exif()  # plant photo carries GPS + time in EXIF, like a phone camera
    ex[306] = "2026:09:10 09:01:00"
    ex[0x8825] = {1: "N", 2: (26.0, 11.0, 24.0), 3: "E", 4: (91.0, 41.0, 24.0)}
    a = np.zeros((64, 64, 3), np.uint8); a[:28] = (60, 140, 50); a[28:40] = (200, 190, 40); a[40:] = (120, 85, 55)
    key = lambda k: f"{obs.loc[k, 'patch_id']}_{obs.loc[k, 'sapling_id']}"  # photos are {patch_id}_{sapling_id}.jpg
    sub = lambda k: OUT / "photos" / FOLDERS[obs.loc[k, "patch_id"]]
    Image.fromarray(a).save(sub(0) / f"{key(0)}.jpg", exif=ex)
    for k in (0, 2):
        a = np.zeros((64, 64, 3), np.uint8); a[: 20 + 20 * k] = (140, 190, 240); a[20 + 20 * k:] = (40, 70, 30)
        Image.fromarray(a).save(sub(k) / f"{key(k)}_sky.jpg")
    print(f"Wrote {len(obs)} synthetic rows, {len(patches)} patches, 3 images to {OUT}")


if __name__ == "__main__":
    main()
