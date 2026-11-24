# Environments

EnvoDat covers 13 environments in three groups: indoor scenes on the Montanuniversität Leoben campus, outdoor scenes in Leoben, and subterranean tunnels. They were recorded intermittently between November 2022 and September 2024, under conditions that include dynamic entities, high illumination and shadow formation, fog and rain, partial and zero visibility, and different times of day.

The **scene id** is the name used by the toolkit and on the Hugging Face Hub (`ed.download("mu-hall")`, `load_dataset("Linpen/EnvoDat", "mu-hall")`). `envodat scenes` prints the same catalog, and {func}`envodat.info` returns it for one scene.

## Indoor

| Scene id | Name | Sequences | Description |
| -------- | ---- | --------- | ----------- |
| `mu-hall` | MU-Hall: exhibition hall | 01, 02, 03 | Mid-large exhibition hall with many dynamic and static features: moving people and exhibition furniture. |
| `mu-cor` | MU-Cor: long corridor | 01 (morning), 02 (afternoon), 03 (evening) | Highly symmetric passageway of about 6 × 120 m with tables, under artificial and natural lighting. |
| `mu-txn` | MU-TXN: glass building | 01, 02, 03 | Student study centre with large transparent and reflective surfaces and moving objects, a mixed static and dynamic environment. |
| `mu-cps` | MU-CPS: office layout | 01, 02, 03 | Typical office with multiple occlusions, furniture, daylight and artificial light, and numerous loop closures. |

## Outdoor

| Scene id | Name | Sequences | Description |
| -------- | ---- | --------- | ----------- |
| `leo-str` | Leo-Str: Leoben campus street | 01 (morning), 02 (afternoon) | Feature-sparse street with pedestrians, vehicles and several buildings. |
| `leo-for` | Leo-For: Leoben forest path | 01, 02 | Vegetated, unstructured terrain with dense canopies, varying light penetration and dynamic shadows from swaying leaves and branches. |

## Subterranean

| Scene id | Name | Sequences |
| -------- | ---- | --------- |
| `subt-zab-01` | SubT-ZAB-01 | 01, 02 |
| `subt-zab-02` | SubT-ZAB-02 | 01 |
| `subt-zab-03` | SubT-ZAB-03 | 01, 02 |
| `subt-zab-04` | SubT-ZAB-04 | 01, 02 |
| `subt-zab-05` | SubT-ZAB-05 | 01, 02 |
| `subt-sbe-01` | SubT-SBe-01 | 01 |
| `subt-sbe-02` | SubT-SBe-02 | 01 |

The tunnel annotations are released as one set for all SubT-ZAB tunnels, so on the Hub they form a single config, `subt-zab` (`ed.download("subt-zab")`).

The tunnels include poorly lit and feature-sparse sections, highly symmetric passages and narrow, sandy, feature-free stretches.

## Scene characteristics

![EnvoDat scene characteristics](scene_characteristics.gif)

Across the scenes, the recordings include, among others: feature-sparse sub-tunnels with poor visibility; open, dynamic streets; textureless spaces under artificial lighting; dense canopies over elevated, granular or sloping terrain; wet, highly symmetric floors with moving people; multiple occlusions; rain, fog and dynamic entities; and strongly varying or very high illumination. Feature-density maps, trajectory maps and SLAM-reconstructed maps for each scene are shown on the project website's [Scenes & Sequences](https://sites.google.com/view/envodat/scenes-sequences) page.
