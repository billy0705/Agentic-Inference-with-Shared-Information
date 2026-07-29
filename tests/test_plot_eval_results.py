import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_eval_results.py"


def load_plot_module():
    assert SCRIPT_PATH.exists()
    spec = importlib.util.spec_from_file_location("plot_eval_results", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_summary(path: Path, *, benchmark: str, methods: dict[str, tuple[float, float]]) -> None:
    path.mkdir(parents=True)
    payload = {
        "summary": {
            method: {
                "total": 10,
                "correct": int(accuracy * 10),
                "accuracy": accuracy,
                "avg_total_tokens": tokens,
            }
            for method, (accuracy, tokens) in methods.items()
        },
        "method_averages": {
            method: {
                "correct_avg": accuracy,
                "total_token_avg": tokens,
            }
            for method, (accuracy, tokens) in methods.items()
        },
        "results": [{"benchmark": benchmark}],
    }
    (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collect_latest_benchmark_results_uses_latest_run_per_method(tmp_path):
    module = load_plot_module()
    write_summary(
        tmp_path / "gsm8k" / "model-a" / "20260721T000000Z_old",
        benchmark="gsm8k",
        methods={"single_agent": (0.1, 100.0), "multiagent": (0.2, 200.0)},
    )
    write_summary(
        tmp_path / "gsm8k" / "model-a" / "20260728T000000Z_new",
        benchmark="gsm8k",
        methods={"multiagent": (0.8, 1200.0)},
    )
    write_summary(
        tmp_path / "gpqa" / "model-a" / "20260727T000000Z_only",
        benchmark="gpqa",
        methods={"single_agent": (0.5, 500.0)},
    )

    records = module.collect_latest_results(tmp_path)

    assert [(record.benchmark, record.method, record.accuracy, record.avg_total_tokens) for record in records] == [
        ("gpqa", "single_agent", 0.5, 500.0),
        ("gsm8k", "single_agent", 0.1, 100.0),
        ("gsm8k", "multiagent", 0.8, 1200.0),
    ]


def test_collect_configured_results_uses_explicit_runs(tmp_path):
    module = load_plot_module()
    old_run = tmp_path / "gsm8k" / "model-a" / "20260721T000000Z_old"
    new_run = tmp_path / "gsm8k" / "model-a" / "20260728T000000Z_new"
    write_summary(old_run, benchmark="gsm8k", methods={"single_agent": (0.1, 100.0), "multiagent": (0.2, 200.0)})
    write_summary(new_run, benchmark="gsm8k", methods={"single_agent": (0.7, 700.0), "multiagent": (0.8, 1200.0)})
    config_path = tmp_path / "plot_config.json"
    config_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "benchmark": "gsm8k",
                        "model": "model-a",
                        "run_id": "20260721T000000Z_old",
                        "methods": ["single_agent"],
                    },
                    {"summary_path": str(new_run / "summary.json"), "methods": ["multiagent"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = module.collect_configured_results(tmp_path, config_path)

    assert [(record.benchmark, record.method, record.accuracy, record.avg_total_tokens) for record in records] == [
        ("gsm8k", "single_agent", 0.1, 100.0),
        ("gsm8k", "multiagent", 0.8, 1200.0),
    ]


def test_configured_run_labels_allow_before_after_points(tmp_path):
    module = load_plot_module()
    old_run = tmp_path / "gsm8k" / "model-a" / "20260721T000000Z_old"
    new_run = tmp_path / "gsm8k" / "model-a" / "20260728T000000Z_new"
    write_summary(old_run, benchmark="gsm8k", methods={"multiagent": (0.2, 200.0)})
    write_summary(new_run, benchmark="gsm8k", methods={"multiagent": (0.8, 1200.0), "single_agent": (0.5, 500.0)})
    config_path = tmp_path / "plot_config.json"
    config_path.write_text(
        json.dumps(
            {
                "runs": [
                    {"summary_path": str(old_run / "summary.json"), "label": "before"},
                    {"summary_path": str(new_run / "summary.json"), "label": "after", "methods": ["multiagent"]},
                    {"summary_path": str(new_run / "summary.json"), "methods": ["single_agent"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = module.collect_configured_results(tmp_path, config_path)
    fig, ax = module.build_pareto_plot(records)
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]

    assert [record.display_name for record in records] == [
        "single_agent",
        "multiagent (before)",
        "multiagent (after)",
    ]
    assert "multiagent (before)" in legend_labels
    assert "multiagent (after)" in legend_labels
    module.plt.close(fig)


def test_before_after_points_share_method_color_but_use_different_shapes():
    module = load_plot_module()
    records = [
        module.MethodResult("bench", "multiagent", 0.2, 200.0, Path("old"), label="before"),
        module.MethodResult("bench", "multiagent", 0.8, 1200.0, Path("new"), label="after"),
        module.MethodResult("bench", "single_agent", 0.5, 500.0, Path("new")),
    ]

    fig, ax = module.build_pareto_plot(records)
    collections_by_label = {
        collection.get_label(): collection
        for collection in ax.collections
    }

    before = collections_by_label["multiagent (before)"]
    after = collections_by_label["multiagent (after)"]
    single_agent = collections_by_label["single agent"]
    assert before.get_facecolors()[0].tolist() == after.get_facecolors()[0].tolist()
    assert len(before.get_paths()[0].vertices) != len(after.get_paths()[0].vertices)
    assert len(after.get_paths()[0].vertices) == len(single_agent.get_paths()[0].vertices)
    module.plt.close(fig)


def test_accuracy_bar_plot_uses_after_without_label_and_omits_before():
    module = load_plot_module()
    records = [
        module.MethodResult("bench", "multiagent", 0.2, 200.0, Path("old"), label="before"),
        module.MethodResult("bench", "multiagent", 0.8, 1200.0, Path("new"), label="after"),
        module.MethodResult("bench", "single_agent", 0.5, 500.0, Path("new")),
    ]

    fig, ax = module.build_accuracy_bar_plot(records)
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    bar_heights = sorted(round(patch.get_height(), 2) for patch in ax.patches)

    assert legend_labels == ["single agent", "multiagent"]
    assert bar_heights == [0.5, 0.8]
    module.plt.close(fig)


def test_accuracy_bar_plot_falls_back_to_before_when_after_is_missing():
    module = load_plot_module()
    records = [
        module.MethodResult("bench", "multiagent", 0.2, 200.0, Path("old"), label="before"),
        module.MethodResult("bench", "single_agent", 0.5, 500.0, Path("new")),
    ]

    fig, ax = module.build_accuracy_bar_plot(records)
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    bar_heights = sorted(round(patch.get_height(), 2) for patch in ax.patches)

    assert legend_labels == ["single agent", "multiagent"]
    assert bar_heights == [0.2, 0.5]
    module.plt.close(fig)


def test_pareto_front_prefers_higher_accuracy_and_lower_token_usage():
    module = load_plot_module()
    records = [
        module.MethodResult("bench", "cheap_good", 0.8, 100.0, Path("a")),
        module.MethodResult("bench", "expensive_same", 0.8, 200.0, Path("b")),
        module.MethodResult("bench", "expensive_best", 0.9, 300.0, Path("c")),
        module.MethodResult("bench", "cheap_bad", 0.7, 80.0, Path("d")),
    ]

    front = module.compute_pareto_front(records)

    assert [record.method for record in front] == ["expensive_best", "cheap_good", "cheap_bad"]


def test_pareto_x_axis_uses_raw_token_usage(tmp_path):
    module = load_plot_module()
    records = [
        module.MethodResult("bench", "single_agent", 0.6, 200.0, Path("run")),
        module.MethodResult("bench", "multiagent", 0.8, 400.0, Path("run")),
    ]

    fig, ax = module.build_pareto_plot(records)

    assert ax.get_xlabel() == "Average total tokens (lower is better; axis reversed)"
    assert ax.xaxis_inverted()
    assert [point.get_offsets()[0][0] for point in ax.collections] == [200.0, 400.0]
    assert not ax.texts
    module.plt.close(fig)


def test_write_plots_defaults_to_png(tmp_path):
    module = load_plot_module()
    records = [
        module.MethodResult("bench", "single_agent", 0.6, 100.0, Path("run")),
        module.MethodResult("bench", "multiagent", 0.8, 180.0, Path("run")),
    ]

    output_paths = module.write_plots(records, tmp_path)

    assert [path.name for path in output_paths] == ["accuracy_grouped_bar.png", "pareto_bench.png"]
    for path in output_paths:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
