# Sapling Survival Audit — IIT Guwahati

A web app and local AI pipeline that shows which campus saplings survived, which site conditions are linked with their death, and what to do next at each place.

A field team photographs every sapling. A free, local AI model (BioCLIP) reads the photos. A Streamlit web portal then turns the survey data into:

- a survival map
- the conditions linked with survival or death
- a plan for each patch: good, risky or avoid, and which species suit it
- an action plan for each place on campus

No paid API. The AI runs locally, on a laptop GPU (built for an RTX 3050 with 4 GB) or on the CPU if there's no GPU.

---

## Features

- **Photo matching:** point the app at a photo folder with one subfolder per place (e.g. `barak/`, `kameng/`, `umiam/`). Each CSV row is matched to its photo automatically.
- **GPS and time from the photos:** blank `lat`, `lon` and `timestamp` values are read from the photo's EXIF data.
- **AI species and soil labels:** BioCLIP labels each photo without any training (zero-shot).
  - **Species:** chosen from a list of 37 tree, bamboo and grass species found in Assam.
  - **Soil type:** sandy, clayey, loamy, waterlogged, cracked clay or grass-covered.
- **Image features (plain image maths, no model):** green / weed cover, share of yellowing leaves, bare soil, soil colour, soil darkness and canopy openness.
- **Survival analysis:**
  - survival for each patch, species and place
  - a "with vs without" table for each condition, with n and a 95% Wilson interval
  - small groups (n < 10) are flagged
- **Patch plan:** each patch is classed good / risky / avoid by explainable rules. Species are suggested from campus evidence and from a sourced reference table.
- **Gap report and place action plan:** each place is ranked endangered / needs attention / stable, with concrete actions (tree guards, fencing, drainage, weeding, mulching, briefing the mowing crew).
- **Land check:** upload one new photo of a piece of land. You get:
  - its current condition
  - survival among the most similar photos already surveyed
  - the best-suited species
  - a weekly / monthly / yearly maintenance plan
- **Live log:** a panel fixed to the right of the page streams every step, including the AI run, as it happens.
- **Exports:** a Markdown report, a CSV of what to do at each place, a CSV of every sapling with its action, and parquet tables with a schema file.

## How it works

```
Field survey (phone)  →  CSV + photo folder  →  AI batch (local GPU/CPU)  →  Web portal  →  Reports & exports
```

1. **Collect:** for each sapling the team fills in one CSV row per photo and names the photo `{patch_id}_{sapling_id}_v{n}.jpg`.
2. **Upload:** in the portal, upload the CSV and type the path of the photo folder.
3. **Match:** photos are found inside the place subfolders and matched by patch, sapling and visit number.
4. **AI:** click **Run AI matching (BioCLIP)**. The model runs in a separate process (`ai_batch.py`), and its output streams into the live log.
5. **Analyse:** survival, conditions, patch classes, species suggestions and need scores.
6. **Act:** download the gap report and the place action plan.

### What the AI does

| Step | Method |
|---|---|
| Species | BioCLIP compares each plant photo with the text "a photo of {species}" for every species in `species_list.txt`, and keeps the best match. It never overwrites a species the team entered, and skips rows marked `missing` (no sapling in the photo). Matches below 0.5 confidence are left blank. |
| Soil type | BioCLIP compares each ground photo with 6 soil descriptions. |
| Green cover | Excess Green index (2G − R − B) |
| Yellow leaves | Share of foliage pixels in the yellow hue range |
| Bare soil, soil colour, darkness | Pixels that are neither vegetation nor sky; mean colour → dark / grey / pale / red / brown / yellowish |
| Canopy openness | Otsu threshold on the sky photo's blue channel |

### Web portal tabs

| Tab | What it shows |
|---|---|
| 1 · Upload | Count tiles, validation table, photo matching lists, where each row's GPS and place came from |
| 2 · AI & features | Bar charts of species, soil type and soil colour; average features per patch; photo thumbnails |
| 3 · Map | One dot per sapling, coloured by status (alive, stressed, dead, missing); filters by place, patch and species; photo viewer |
| 4 · Survival | Survival per patch, the conditions table with confidence intervals, a bar chart of survival by species |
| 5 · Patch plan | Colour-coded good / risky / avoid table with reasons, a map of patches, species lists per patch |
| 6 · Gap report | Action plan by place, patches ranked by need, a downloadable report |
| 7 · Export | One zip with every output |
| 8 · Land check | Condition, prediction, species and maintenance plan for a single uploaded photo |

## Quick start

```bash
git clone <this-repo> && cd <this-repo>
pip install -r requirements.txt            # web app
pip install -r requirements-ai.txt         # AI step (for a GPU, install the CUDA build of torch from pytorch.org first)

python make_sample.py                      # optional: generate synthetic test data
streamlit run app.py
```

Open the app, click **Load synthetic sample** (or upload your own CSV and type your photo folder path), then click **Run AI matching (BioCLIP)**. The first run downloads the model, about 600 MB.

To run the AI step from a terminal instead:

```bash
python ai_batch.py observations.csv path/to/photos -o observations_ai.csv
```

## Input data

**`observations.csv`:** one row per photo. Only `patch_id`, `sapling_id` and `status` are required.

```
patch_id, status, sapling_id, landmark, timestamp, lat, lon, species_ai, tree_guard,
grazing_damage, physical_damage, waterlogged, canopy_open, height_cm, data_origin, notes
```

- **`status`:** alive / stressed / dead / missing
- **`tree_guard`:** none / damaged / intact
- **`physical_damage`:** none / mowing / trampling / other
- **`landmark`:** the place name, matching its photo subfolder. If blank, it's filled from the folder.
- The older column name `canopy_openness_pct` is also accepted for `canopy_open`.

**Photo folder:**

```
photos/
  barak/    P01_S001_v1.jpg  P01_S007_v1.jpg  P01_S007_v2.jpg ...
  kameng/   P03_S014_v1.jpg ...
  umiam/    P02_S011_v1.jpg ...
```

- **Numbering:** a sapling's CSV rows are numbered in timestamp order and matched to `_v1`, `_v2`, and so on. `P01_S001.jpg` without a number also works.
- **Optional photos:** `_sky.jpg` (used for canopy openness) and `_soil.jpg` (ground photo) for each sapling.

**Optional files:**
- `patches.csv`: site conditions for each patch, such as waterlogging history, terrain, soil, compaction, grazing, fencing and overhead lines.
- `species_reference.csv`: the default table of 37 species ships with the app.

See [RUN.md](RUN.md) for the full column list and details.

## Species reference

[`species_reference.csv`](species_reference.csv) lists 37 species. Each row has:

- **Traits:** soil textures, waterlogging and shade tolerance, compaction sensitivity and maximum height
- **Other details:** whether it's native to Assam, and where on campus it's best used
- **A source:** a named publication and a `source_url`, such as the World Agroforestry Agroforestree Database, eFlora of India, India Flora Online (IISc) or PFAF

Traits a source doesn't state are left as `unknown`, and the code never fills them in. Horticulture staff confirm each row and fill in `verified_by`. To add a species, add a row with a real source, then add its name to [`species_list.txt`](species_list.txt).

## Project structure

```
app.py               Streamlit web portal (UI only; never loads the model)
core.py              Data logic: matching, EXIF, image features, survival, rules, species fit, reports, export
ai_batch.py          BioCLIP inference (batch over a CSV + folder, or --single for one photo)
make_sample.py       Generates a small synthetic dataset for testing
species_reference.csv / species_list.txt    Sourced species table and the model's label list
RUN.md               Detailed run guide and column reference
.streamlit/config.toml   Theme (white background, green headings)
```

## Tech stack

- **Web:** Python, Streamlit, pandas, NumPy, Pillow, PyArrow
- **AI:** [BioCLIP](https://huggingface.co/imageomics/bioclip) via `open_clip`, running on PyTorch (CUDA GPU or CPU)
- **Deployment:** the portal can run on Streamlit Community Cloud. Photo-folder matching and the AI step need to run locally.

## Limitations

- **AI labels are estimates.** The model picks only from the listed species, so plants outside the list get the closest listed species. Staff should spot-check the labels.
- **Pixel features shift** with light and shadow.
- **The Land check prediction** compares a new photo with photos the team already surveyed. It isn't a trained model, and it improves as more surveys are added.
- **The maintenance plan** is rule-based guidance; confirm it with the horticulture team.
- **Linked, not caused:** conditions are "linked with" survival, not proven causes. Reference suitability comes from literature, not campus evidence.
- **Synthetic sample:** data from `make_sample.py` is synthetic and is labelled that way everywhere in the app.
