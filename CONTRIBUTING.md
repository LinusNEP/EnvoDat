# Contributing to EnvoDat

Thanks for your interest in improving EnvoDat! Contributions are welcome for bug
fixes, documentation, tooling, and especially new benchmark results.

## Reporting issues

Open an issue with: what you ran (command/config), what you expected, what
happened, and your environment (`python --version`, OS, GPU, and `pip freeze`
for the relevant packages). For dataset problems, include the scene id and
format.

## Development setup

See [`docs/INSTALL.md`](docs/INSTALL.md). In short:

```bash
python3 -m venv EnvoDatEnv && source EnvoDatEnv/bin/activate
pip install -r requirements.txt
```

## Submitting benchmark results (leaderboard)

The leaderboard in [`docs/BENCHMARK.md`](docs/BENCHMARK.md) is generated from
the data files in [`results/`](results/). To add a result:

1. **Reproduce it** with the provided tooling so it's verifiable:
   - *SLAM track:* produce a TUM-format estimated trajectory with the SLAM
     pipeline, then score it:
     ```bash
     python docs/scripts/evaluate_slam.py \
         --est est.tum --gt gt.tum --align sim3 \
         --algorithm "FAST-LIO2" --scene mu-hall-01 \
         --update results/slam_results.yaml
     ```
   - *Detection track:* train/evaluate with `yolov8_training.py`, then add your
     `map50`, `map5095`, `precision`, `recall` to `results/detection_results.yaml`.
2. **Regenerate** the markdown:
   ```bash
   python docs/scripts/generate_leaderboard.py
   ```
3. **Open a pull request** that includes:
   - the diff to the relevant `results/*.yaml`,
   - the regenerated `docs/BENCHMARK.md`,
   - the exact command/config used,
   - a link to logs, a short report, or a config file so the result can be
     checked.

### Result guidelines

- State the alignment used (SE(3) vs Sim(3)) for SLAM.
- Use the released scene ids exactly (`mu-hall-01`, `mu-txn-01`, `subt-zab-01`,
  `mu-cor-03`, `leo-str-01`).
- Mark a failed/lost-tracking run as `null` (renders as DNF).
- Round metrics to a sensible precision; the generator formats to 3 decimals.

## Code style

- Python 3.8+ compatible.
- Keep dependencies in `requirements.txt` and pinned to compatible ranges.
- Docstrings on scripts with at least one usage example.

## License of contributions

By contributing, you agree that your code contributions are licensed under the
repository's code license (MIT, see [`LICENSE-CODE`](LICENSE-CODE)) and any
dataset/documentation contributions under CC BY 4.0.
