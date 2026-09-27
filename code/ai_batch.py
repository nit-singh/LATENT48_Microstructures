"""Offline batch: match {patch_id}_{sapling_id} photos (inside landmark subfolders), fill species_ai, soil_type_ai,
pixel features and canopy. GPU if available. `--single IMAGE` labels one photo (used by the app's Land check tab)."""
import argparse
import functools
from pathlib import Path
import pandas as pd
from PIL import Image
from core import FEATURES, clean_header, find_photo, image_features, index_photos, otsu_openness, visit_index

print = functools.partial(print, flush=True)  # live output when streamed to the web page
MODEL = "hf-hub:imageomics/bioclip"
MIN_CONF = 0.5  # species guesses below this are left blank
SOIL = {"a photo of dry sandy soil": "sandy", "a photo of sticky wet clay soil": "clayey",
        "a photo of dark crumbly loam soil": "loamy", "a photo of waterlogged muddy ground with standing water": "waterlogged",
        "a photo of dry cracked clay ground": "cracked_clay", "a photo of ground covered with grass and weeds": "grass_covered"}


def text_features(model, tok, prompts, device, torch):
    """Normalised text embeddings for prompts."""
    with torch.no_grad():
        f = model.encode_text(tok(prompts).to(device))
    return f / f.norm(dim=-1, keepdim=True)


def run_model(labels_out, plants, grounds):
    """BioCLIP zero-shot: species_ai from plant photos, soil_type_ai from ground photos. plants/grounds: [(key, path)]."""
    import torch, open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    model = model.to(device).eval()
    if device == "cuda": model = model.half()
    species = [s.strip() for s in (Path(__file__).parent / "species_list.txt").read_text().splitlines() if s.strip()]
    for col, todo, labels, prompts in [("species_ai", plants, species, [f"a photo of {s}." for s in species]),
                                       ("soil_type_ai", grounds, list(SOIL.values()), list(SOIL))]:
        tf = text_features(model, tok, prompts, device, torch)
        for b in range(0, len(todo), 16):
            chunk = todo[b:b + 16]
            x = torch.stack([preprocess(Image.open(p).convert("RGB")) for _, p in chunk]).to(device)
            with torch.no_grad():
                f = model.encode_image(x.half() if device == "cuda" else x)
                prob = (100 * (f / f.norm(dim=-1, keepdim=True)) @ tf.T).softmax(-1).float()
            for k, (key, _) in enumerate(chunk):
                labels_out(key, col, labels[prob[k].argmax().item()], prob[k].max().item())
            print(f"{col}: {min(b + 16, len(todo))}/{len(todo)}")


def single(path):
    """Label one uploaded land/plant photo and print a RESULT line for the app."""
    out = {}
    def put(_, col, label, conf):
        out[col] = label; print(f"  {col} = {label} ({conf:.2f})")
    print(f"Land check photo: {path}")
    run_model(put, [(0, path)], [(0, path)])
    print("RESULT " + ";".join(f"{k}={v}" for k, v in out.items()))


def main():
    """Run the batch over the observations CSV (or one photo with --single)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?"); ap.add_argument("photos", nargs="?")
    ap.add_argument("-o", "--out", default="observations_ai.csv"); ap.add_argument("--single")
    a = ap.parse_args()
    if a.single:
        return single(a.single)
    df = pd.read_csv(a.csv, dtype=str, keep_default_na=False).rename(columns=clean_header)
    df = df.loc[:, ~df.columns.duplicated()]
    for c in ["landmark", "photo_folder", "species_ai", "canopy_open"] + FEATURES:
        if c not in df: df[c] = ""
    idx = index_photos(a.photos)
    places = sorted({pl for lst in idx.values() for _, pl in lst if pl})
    keys = df["patch_id"].str.strip() + "_" + df["sapling_id"].str.strip()
    visits = visit_index(df)  # row n of a sapling -> photo {patch_id}_{sapling_id}_v{n}.jpg
    hit = lambda i, kind: find_photo(idx, keys[i], kind, df.at[i, "landmark"], visits[i])
    blank = lambda i, col: not df.at[i, col].strip() or df.at[i, col].strip().lower().startswith("unidentified")
    plants = [(i, h[0]) for i in df.index if (h := hit(i, "plant"))]
    grounds = [(i, h[0]) for i in df.index if (h := hit(i, "soil") or hit(i, "plant"))]
    print(f"Folder has {sum(map(len, idx.values()))} images in {len(places)} place folders ({', '.join(places) or 'none'})")
    print(f"Matched {len(plants)}/{len(df)} rows to a {{patch_id}}_{{sapling_id}} plant photo; {len(df) - len(plants)} rows skipped")
    for i in df.index:
        h = hit(i, "plant")
        if h:
            df.at[i, "photo_folder"] = h[1]
            if not df.at[i, "landmark"]: df.at[i, "landmark"] = h[1]  # place name from the subfolder
    for i, p in grounds:
        for k, v in image_features(str(p)).items():
            df.at[i, k] = "" if v is None else str(v)
        sky = hit(i, "sky")
        if sky and not df.at[i, "canopy_open"]:
            df.at[i, "canopy_open"] = str(otsu_openness(sky[0]))
    print(f"Pixel features (green cover, yellow leaves, bare soil, soil colour, darkness) done for {len(grounds)} photos")
    print("Loading torch + BioCLIP (first run downloads the model)...")

    def put(i, col, label, conf):
        tag = f"  [{df.at[i, 'landmark'] or '-'}] {keys[i]}_v{visits[i]}: {col}"
        if not blank(i, col):  # never overwrite a value the team entered
            return print(f"{tag} kept '{df.at[i, col]}' (model said {label}, {conf:.2f})")
        if col == "species_ai" and df.at[i, "status"].strip().lower() == "missing":
            return print(f"{tag} skipped (status missing: no sapling in photo)")
        if col == "species_ai" and conf < MIN_CONF:
            return print(f"{tag} left blank (uncertain: {label}, {conf:.2f} < {MIN_CONF})")
        df.at[i, col] = label
        print(f"  [{df.at[i, 'landmark'] or '-'}] {keys[i]}_v{visits[i]}: {col} = {label} ({conf:.2f})")
    try:
        run_model(put, plants, grounds)
    except ImportError as e:
        print(f"Model skipped ({e}); install requirements-ai.txt for species_ai and soil_type_ai")
    df.to_csv(a.out, index=False)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
