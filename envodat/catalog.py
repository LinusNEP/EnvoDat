"""Static catalog of the EnvoDat scenes (13 environments).

Scene ids are lower-case and match the config names on the Hugging Face Hub.
"""

INDOOR, OUTDOOR, SUBTERRANEAN = "indoor", "outdoor", "subterranean"

#: Scene catalog: ``{scene_id: {name, title, environment, sequences, description}}``.
SCENES = {
    "mu-hall": dict(name="MU-Hall", title="Exhibition hall", environment=INDOOR,
                    sequences=["01", "02", "03"],
                    description="Mid-large exhibition hall with many dynamic and static features "
                                "(moving people, exhibition furniture)."),
    "mu-cor": dict(name="MU-Cor", title="Long corridor", environment=INDOOR,
                   sequences=["01", "02", "03"],
                   description="Highly symmetric passageway of roughly 6 x 120 m with tables and mixed "
                               "artificial/natural lighting; recorded morning, afternoon and evening."),
    "mu-txn": dict(name="MU-TXN", title="Glass building", environment=INDOOR,
                   sequences=["01", "02", "03"],
                   description="Student study centre with large transparent and reflective surfaces "
                               "and moving objects."),
    "mu-cps": dict(name="MU-CPS", title="Office layout", environment=INDOOR,
                   sequences=["01", "02", "03"],
                   description="Office layout with multiple occlusions, furniture, daylight and "
                               "artificial light, and numerous loop closures."),
    "leo-str": dict(name="Leo-Str", title="Leoben campus street", environment=OUTDOOR,
                    sequences=["01", "02"],
                    description="Feature-sparse street with pedestrians, vehicles and buildings; "
                                "recorded morning and afternoon."),
    "leo-for": dict(name="Leo-For", title="Leoben forest path", environment=OUTDOOR,
                    sequences=["01", "02"],
                    description="Vegetated, unstructured terrain with dense canopies, varying light "
                                "penetration and dynamic shadows."),
    "subt-zab-01": dict(name="SubT-ZAB-01", title="Underground tunnel", environment=SUBTERRANEAN,
                        sequences=["01", "02"], description="Underground tunnel."),
    "subt-zab-02": dict(name="SubT-ZAB-02", title="Underground tunnel", environment=SUBTERRANEAN,
                        sequences=["01"], description="Underground tunnel."),
    "subt-zab-03": dict(name="SubT-ZAB-03", title="Underground tunnel", environment=SUBTERRANEAN,
                        sequences=["01", "02"], description="Underground tunnel.",
                        aliases=["subt-zab-lid-03"]),
    "subt-zab-04": dict(name="SubT-ZAB-04", title="Underground tunnel", environment=SUBTERRANEAN,
                        sequences=["01", "02"], description="Underground tunnel."),
    "subt-zab-05": dict(name="SubT-ZAB-05", title="Underground tunnel", environment=SUBTERRANEAN,
                        sequences=["01", "02"], description="Underground tunnel."),
    "subt-sbe-01": dict(name="SubT-SBe-01", title="Underground tunnel", environment=SUBTERRANEAN,
                        sequences=["01"], description="Underground tunnel."),
    "subt-sbe-02": dict(name="SubT-SBe-02", title="Underground tunnel", environment=SUBTERRANEAN,
                        sequences=["01"], description="Underground tunnel."),
    # Annotation set only: the SubT-ZAB tunnels are annotated together, not per scene.
    "subt-zab": dict(name="SubT-ZAB", title="Underground tunnels (annotations)",
                     environment=SUBTERRANEAN, sequences=[],
                     description="Annotated frames of the SubT-ZAB tunnels, released as one set."),
}


def scene_ids(environment=None):
    """Return the catalog scene ids, optionally filtered by environment type."""
    return [k for k, v in SCENES.items() if environment in (None, v["environment"])]


def match_scene(text):
    """Return the catalog scene id contained in ``text`` (a path or name), or None.

    Matching ignores case and separators, so "MU-Hall", "mu_hall_yolov8" and
    "Indoors/MU-Hall/annotations" all resolve to ``"mu-hall"``.
    """
    import re
    norm = "-" + re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") + "-"
    names = [(a, sid) for sid, v in SCENES.items() for a in [sid] + v.get("aliases", [])]
    for name, sid in sorted(names, key=lambda x: len(x[0]), reverse=True):
        if f"-{name}-" in norm:
            return sid
    return None
