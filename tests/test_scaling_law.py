import json
from dataclasses import replace

import numpy as np
import pytest

from esme_pretrain.cli import main
from esme_pretrain.study.scaling_law import (
    LossPoint,
    bootstrap_prediction,
    fit_scaling_law,
)

TRUE_E, TRUE_A, TRUE_ALPHA, TRUE_B, TRUE_BETA = 1.8, 400.0, 0.34, 410.0, 0.28
STUDY_NON_EMBEDDING_PARAMS = (37_800_000, 75_500_000, 113_300_000)
STUDY_TOKENS_PER_PARAM = (5, 10, 20, 40)
HELD_OUT_NON_EMBEDDING_PARAMS = 188_700_000
HELD_OUT_TOKENS = 10_230_000_000


def true_loss(non_embedding_params: float, tokens: float) -> float:
    return TRUE_E + TRUE_A / non_embedding_params**TRUE_ALPHA + TRUE_B / tokens**TRUE_BETA


def study_points(noise_sigma: float = 0.0, seed: int = 0) -> list[LossPoint]:
    rng = np.random.default_rng(seed)
    points = []
    for params in STUDY_NON_EMBEDDING_PARAMS:
        for ratio in STUDY_TOKENS_PER_PARAM:
            tokens = ratio * params
            val_ce = true_loss(params, tokens) + float(rng.normal(0.0, noise_sigma))
            points.append(LossPoint(f"n{params}-d{ratio}x", params, tokens, val_ce))
    return points


@pytest.fixture(scope="module")
def noiseless_fit():
    return fit_scaling_law(study_points())


def test_noiseless_fit_recovers_held_out_prediction_and_exponents(noiseless_fit) -> None:
    held_out = true_loss(HELD_OUT_NON_EMBEDDING_PARAMS, HELD_OUT_TOKENS)
    predicted = noiseless_fit.predict(HELD_OUT_NON_EMBEDDING_PARAMS, HELD_OUT_TOKENS)

    assert predicted == pytest.approx(held_out, abs=1e-9)
    assert noiseless_fit.alpha == pytest.approx(TRUE_ALPHA, abs=1e-6)
    assert noiseless_fit.beta == pytest.approx(TRUE_BETA, abs=1e-6)
    recovered = (noiseless_fit.E, noiseless_fit.A, noiseless_fit.B)
    assert recovered == pytest.approx((TRUE_E, TRUE_A, TRUE_B), rel=1e-6)


def test_bootstrap_interval_covers_true_value_under_small_noise() -> None:
    # A 90% interval misses 1 in 10 draws by design: noise seeds 0-9 cover 9 of 10
    # (seed 0 misses), so this pins seed 1 rather than claiming every seed covers.
    points = study_points(noise_sigma=0.003, seed=1)

    result = bootstrap_prediction(
        points,
        non_embedding_params=HELD_OUT_NON_EMBEDDING_PARAMS,
        tokens=HELD_OUT_TOKENS,
        resamples=100,
        seed=0,
    )

    held_out = true_loss(HELD_OUT_NON_EMBEDDING_PARAMS, HELD_OUT_TOKENS)
    assert result.resamples == 100
    assert result.lower < result.prediction < result.upper
    assert result.lower <= held_out <= result.upper
    assert result.prediction == result.fit.predict(HELD_OUT_NON_EMBEDDING_PARAMS, HELD_OUT_TOKENS)


def test_fit_is_deterministic() -> None:
    points = study_points(noise_sigma=0.003, seed=2)

    assert fit_scaling_law(points) == fit_scaling_law(points)


@pytest.mark.parametrize(
    ("field", "value"),
    [("non_embedding_params", 0), ("tokens", -1), ("val_ce", 0.0), ("val_ce", float("nan"))],
)
def test_loss_point_rejects_non_positive_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        replace(study_points()[0], **{field: value})


def test_fit_rejects_fewer_than_five_points() -> None:
    with pytest.raises(ValueError, match="at least 5 points"):
        fit_scaling_law(study_points()[:4])


def test_scaling_fit_cli_writes_output(tmp_path, capsys) -> None:
    points_path = tmp_path / "points.json"
    points_path.write_text(json.dumps([point.__dict__ for point in study_points()]))
    output_path = tmp_path / "fit.json"

    exit_code = main(
        [
            "scaling-fit",
            "--points",
            str(points_path),
            "--predict-non-embedding-params",
            str(HELD_OUT_NON_EMBEDDING_PARAMS),
            "--predict-tokens",
            str(HELD_OUT_TOKENS),
            "--resamples",
            "5",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["point_count"] == 12
    assert payload["fit"]["alpha"] == pytest.approx(TRUE_ALPHA, abs=0.05)
    assert payload["prediction"]["val_ce"] == pytest.approx(
        true_loss(HELD_OUT_NON_EMBEDDING_PARAMS, HELD_OUT_TOKENS), abs=0.005
    )
    assert payload["settings"]["huber_delta"] == 1e-3
    assert payload["settings"]["resamples"] == 5
    assert payload["settings"]["seed"] == 0
    assert payload["settings"]["initialization_grid"]["log_A"] == [0, 5, 10, 15, 20, 25]
    assert "scaling-fit" in capsys.readouterr().out


def test_scaling_fit_cli_rejects_malformed_points_file(tmp_path, capsys) -> None:
    points_path = tmp_path / "points.json"
    points_path.write_text(json.dumps([{"arm": "n37m", "tokens": 1}]))
    output_path = tmp_path / "fit.json"

    exit_code = main(
        [
            "scaling-fit",
            "--points",
            str(points_path),
            "--predict-non-embedding-params",
            str(HELD_OUT_NON_EMBEDDING_PARAMS),
            "--predict-tokens",
            str(HELD_OUT_TOKENS),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("scaling-fit failed: point 0 must be an object")
    assert captured.err.count("\n") == 1
    assert not output_path.exists()


def test_scaling_fit_reads_study_result_rows_per_arm(tmp_path) -> None:
    # The study writes <arm>/<run>.json rows with many fields; the fit must take one
    # point per run, labeled arm/run, and ignore the per-run metrics files.
    points = study_points()
    for index, point in enumerate(points):
        arm = f"L{index // 4}"
        row_dir = tmp_path / arm
        row_dir.mkdir(exist_ok=True)
        row = {
            "arm": arm,
            "run": f"run{index % 4}",
            "non_embedding_params": point.non_embedding_params,
            "tokens": point.tokens,
            "val_ce": point.val_ce,
            "steady_tokens_per_second": 1.0,
        }
        (row_dir / f"run{index % 4}.json").write_text(json.dumps(row))
        (row_dir / f"run{index % 4}-metrics.jsonl").write_text("{}\n")
    output_path = tmp_path / "fit.json"

    exit_code = main(
        [
            "scaling-fit",
            "--results-dir",
            str(tmp_path),
            "--arm",
            "L0",
            "--arm",
            "L1",
            "--arm",
            "L2",
            "--predict-non-embedding-params",
            str(HELD_OUT_NON_EMBEDDING_PARAMS),
            "--predict-tokens",
            str(HELD_OUT_TOKENS),
            "--resamples",
            "5",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["point_count"] == 12
    assert payload["prediction"]["val_ce"] == pytest.approx(
        true_loss(HELD_OUT_NON_EMBEDDING_PARAMS, HELD_OUT_TOKENS), abs=1e-6
    )
