# Sapling Survival Audit: how to run it

## 1. Install
```bash
pip install -r requirements.txt          # web app (no ML)
pip install -r requirements-ai.txt       # offline AI batch only
```
**GPU:** first install the CUDA build of torch from https://pytorch.org/get-started/locally/ (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`), then install `requirements-ai.txt`. If CUDA isn't available, the batch runs on CPU instead.

## 2. Self-check with synthetic data
```bash
python make_sample.py      # writes sample_data/ (all rows data_origin=synthetic)
python core.py             # prints patch classes, species suggestions, gap levels
```

## 3. Run the app
```bash
streamlit run app.py
```
1. In tab **1 · Upload**, upload `observations.csv`, or click **Load synthetic sample**.
   - `patches.csv` and `species_reference.csv` are **optional**. If you leave the species file empty, the app uses the default list that ships with it.
2. Type the **image folder path**, for example `D:\survey\photos`. The box starts empty.
   - The app reads the photos straight from that folder, including its place subfolders; nothing is copied.
3. The app matches each CSV row to its photo `{patch_id}_{sapling_id}.jpg` inside the row's `landmark` subfolder. If the CSV leaves GPS or time blank, it reads them from the photo's EXIF.
4. Click **Run AI matching (BioCLIP)**. This runs `ai_batch.py` in a separate process.
   - Its output streams into the **Live log** panel, which is fixed to the right of the page and scrolls on its own.
   - Newest lines are at the top. Use **Show live log** and **Log width %** at the top of the page to hide the panel or resize it.
   - When the run finishes, the AI results load into the app automatically.
5. Tab **6 · Gap report** has the **action plan by place**. Tab **7 · Export** puts everything in one zip (see *Outputs* below).
6. Tab **8 · Land check**: upload one new photo of a patch of land and pick the timeline (before planting, 0–3 months, and so on). The app shows:
   - the land's current condition: wet, compacted, weedy, bare or yellowing
   - survival among the 15 most similar photos from your survey data (a nearest-neighbour comparison on the photo features)
   - species that suit this land, and species to avoid
   - a weekly / monthly / yearly maintenance plan with a maintenance level

   The plan is rule-based guidance; confirm it with the horticulture team.

The model is zero-shot, so no training happens. It picks the most likely species from `species_list.txt` for each plant photo, and the most likely soil type for each ground photo. It fills `species_ai` and `soil_type_ai`, plus the 5 pixel features (see below). If `canopy_open` is blank, it fills that from the sky photo using an Otsu threshold. It never changes `species_verified`. The first run downloads the model (about 600 MB). Exports go to `data/processed/`.

You can also run the batch from a terminal: `python ai_batch.py observations.csv photos/ -o observations_ai.csv`. Then upload `observations_ai.csv`.

## CSV columns
- **observations.csv** has exactly these 16 columns. Any other columns are ignored.
  `patch_id, status, sapling_id, landmark, timestamp, lat, lon, species_ai, tree_guard, grazing_damage, physical_damage, waterlogged, canopy_open, height_cm, data_origin, notes`
  - **Required:** `patch_id` (the site, e.g. `P07`), `sapling_id` (e.g. `S047`) and `status` (alive/stressed/dead/missing). The same sapling_id can appear in different patches: a sapling is identified by patch_id and sapling_id together.
  - **Filled in from the photos when blank:**
    - `lat`, `lon` and `timestamp` are read from the photo's EXIF.
    - `canopy_open` is worked out from the sky photo.
    - `species_ai` comes from BioCLIP.
  - **`landmark`:** the place name, e.g. Barak, Kameng, Core 5, Academic Complex. It must match the photo subfolder name; case, spaces, `_` and `-` are ignored, so `Core 5` matches `core5`. If it's blank, it's filled from the subfolder the photo was found in.
  - **Checks:** a `patch_id` that isn't in `patches.csv` is flagged in validation. The Upload tab warns when a sapling's photo exists only in a different place's folder.
  - **Allowed values:** `tree_guard` is none/damaged/intact. `physical_damage` is none/mowing/trampling/other. Yes/no columns accept yes/no/true/false/1/0.
- **Image features:** the AI run (or the app) adds these 6 columns from each ground or plant photo:

  | Column | How it's computed | Why it helps |
  |---|---|---|
  | `green_cover_pct` | % of pixels that are green (Excess Green index) | Weed and grass competition around the sapling |
  | `yellow_leaf_pct` | % of the foliage that is yellow | Early sign of stress, before the sapling dies |
  | `bare_soil_pct` | % of pixels that are bare soil | Exposed, eroding or trampled ground |
  | `soil_color` | Name of the mean soil colour: dark / grey / pale / red / brown / yellowish | Grey soil often means waterlogging; red soil is usually well-drained |
  | `soil_darkness` | 0–100; darker soil scores higher | Rough sign of moisture or organic matter |
  | `soil_type_ai` | BioCLIP picks one: sandy / clayey / loamy / waterlogged / cracked_clay / grass_covered | Stands in for a manual soil survey. If a patch's `soil_texture` is blank, it's filled from the patch's photos |

  The features appear in the Survival conditions table, and wet or grey photos count as a risk flag in the Patch plan. All six are estimates from the model or from pixel colours; staff should spot-check them.
- **patches.csv**: `patch_id, patch_name, status_for_study (past_plantation/candidate), centroid_lat, centroid_lon, count_planted, last_planting_date, aftercare_watering, waterlogging_history, terrain_position, soil_texture, compaction_depth_cm, grazing_evidence, fencing, standing_water_now, foot_traffic_count_10min, overhead_lines, planned_land_use, data_origin`. The centroids are needed to assign rows to patches by GPS.

## Photo naming
Put the photos in one folder with a subfolder for each place. Name each photo `{patch_id}_{sapling_id}`. Matching ignores case.
```
photos/
  barak/             P01_S001.jpg  P01_S001_sky.jpg  P01_S001_soil.jpg ...
  kameng/            P02_S001.jpg ...
  core5/             ...
  academic complex/  ...
```
Photos directly in the top folder also work. With a `landmark` set, only that place's subfolder is searched, so the same `P01_S001` can exist in two places.
- Plant photo: `P07_S047.jpg` (or `P07_S047_plant.jpg`)
- Sky photo (optional, used for canopy): `P07_S047_sky.jpg`
- Ground photo (optional; soil features use the plant photo if there isn't one): `P07_S047_soil.jpg`. Take it looking straight down at the soil next to the sapling.

Keep phone-camera location tagging switched on, so the photos carry GPS. The Upload tab lists matched saplings, saplings with no photo, and photos that aren't in the CSV. It also shows whether each row's GPS came from the CSV or from EXIF (`gps_origin`).

A photo name has no visit number, so a sapling surveyed twice has one photo name for both visits. Keep each survey round's photos in a separate folder, and point the app at that round's folder.

## Outputs (tab 7 · Export zip)
- `place_actions.csv` has one row per place (landmark):
  - patches, survival % with n, need score and level (endangered / needs attention / stable)
  - threat counts, and the most common photo soil colour and soil type
  - `what_to_do`: actions such as tree guards, fencing, the mowing crew, drainage, weeding, mulching or pest checks
  - `species_to_plant` and `species_to_avoid`
- `observations_with_actions.csv` has every sapling row with its `landmark`, `photo_folder`, image features and `action_needed`.
- `gap_report.md` is the full report, including an "Action plan by place" section.
- The zip also contains parquet tables, CSV samples and `schema.md`.
- The AI inference file `observations_ai.csv` keeps `landmark` and adds `photo_folder`.

## Verifying `species_reference.csv`
Horticulture or SART staff check each row against its `source_url`. When a row is confirmed, they put their name or role in `verified_by`. Traits that are blank or `unknown` stay that way until someone documents them; the code never fills them in.

## Adding a species
1. Add a new row to `species_reference.csv`, but only with a real published source (`source` and `source_url`) and `data_origin=records`.
2. Add its scientific name to `species_list.txt` so BioCLIP can pick it.
If planting records name a species that isn't in the reference file yet, add it to `species_list.txt` only.

## Deploy to Streamlit Community Cloud (web app only)
1. Push the repo to GitHub, including `app.py, core.py, make_sample.py, requirements.txt, species_reference.csv, species_list.txt`.
2. At https://share.streamlit.io, create a new app, pick the repo, and set the main file to `app.py`.
3. Don't deploy the AI part: it stays on the laptop. On Cloud, the image folder path refers to the server's disk, so run the app locally for photo matching and AI. On Cloud, upload an `observations_ai.csv` that has already been processed.
