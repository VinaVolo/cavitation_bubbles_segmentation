from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import ray
import torch
import yaml
from clearml import Task
from ray import tune
from ray.tune import report as tune_report
from ray.tune.schedulers import ASHAScheduler
from ultralytics import YOLO
from ultralytics import settings as ultra_settings

from src.config import get_settings, load_tune_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available: %d device(s), using %s", torch.cuda.device_count(), torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS device")
    else:
        device = "cpu"
        logger.info("No GPU found, using CPU")
    return device


def _build_search_space(space_cfg: dict[str, list[float]]) -> dict[str, tune.sample.Domain]:
    result = {}
    for name, bounds in space_cfg.items():
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(f"Search space '{name}' must be [low, high] with low < high, got {bounds}")
        result[name] = tune.uniform(bounds[0], bounds[1])
    return result


def train_yolo(config: dict[str, Any], model_path: str, dataset_path: str, base_cfg: dict[str, Any]) -> None:
    """Training function executed by each Ray Tune trial."""
    import os

    gpu_ids = ray.get_gpu_ids()
    if gpu_ids:
        gpu_id = str(int(gpu_ids[0]))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        device = 0
    else:
        device = "cpu"
    model = YOLO(model_path, task="segment")

    train_kwargs = {**base_cfg, **config}

    def _report_metrics(trainer: Any) -> None:
        metrics = trainer.metrics
        tune_report(
            map50=metrics.get("metrics/mAP50(B)", 0.0),
            map50_95=metrics.get("metrics/mAP50-95(B)", 0.0),
            map50_mask=metrics.get("metrics/mAP50(M)", 0.0),
            map50_95_mask=metrics.get("metrics/mAP50-95(M)", 0.0),
            val_box_loss=metrics.get("val/box_loss", 0.0),
            val_seg_loss=metrics.get("val/seg_loss", 0.0),
            val_cls_loss=metrics.get("val/cls_loss", 0.0),
            epoch=trainer.epoch,
        )

    model.add_callback("on_fit_epoch_end", _report_metrics)

    model.train(
        data=dataset_path,
        device=device,
        project="models/tuned",
        name="trial",
        **train_kwargs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Ray Tune")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of trials (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per trial (overrides config)")
    parser.add_argument("--model_name", type=str, default=None, help="Model filename (overrides config)")
    parser.add_argument("--metric", type=str, default="map50_95_mask", help="Metric to optimize (default: map50_95_mask)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_settings = get_settings()
    device = get_device()

    Task.set_credentials(
        web_host=project_settings.clearml_web_host,
        api_host=project_settings.clearml_api_host,
        files_host=project_settings.clearml_files_host,
        key=project_settings.clearml_api_access_key,
        secret=project_settings.clearml_api_secret_key,
    )

    ray_tmp = Path(Path.home(), "ray_tmp")
    ray_tmp.mkdir(parents=True, exist_ok=True)
    ray.init(_temp_dir=str(ray_tmp))

    ultra_settings.update({"runs_dir": "models/tuned", "tensorboard": False, "clearml": False, "wandb": False})

    dataset_version = project_settings.roboflow_dataset_version
    tune_cfg = load_tune_config()

    model_name = args.model_name or tune_cfg.pop("model")
    tune_cfg.pop("model", None)
    ray_cfg = tune_cfg.pop("ray")

    if args.epochs is not None:
        tune_cfg["epochs"] = args.epochs

    num_samples = args.num_samples or ray_cfg["num_samples"]

    run_name = f"{model_name.removesuffix('.pt')}-v{dataset_version}-raytune"

    dataset_path = str(Path("data", "data.yaml").resolve())
    model_path = str(Path("models", "base", model_name).resolve())

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_path}")

    task = Task.init(
        project_name=project_settings.clearml_project,
        task_name=run_name,
        auto_connect_frameworks={"pytorch": False, "matplotlib": False},
        output_uri=False,
    )
    logger.info("ClearML Task created: %s", task.id)

    search_space = _build_search_space(ray_cfg["search_space"])

    scheduler_cfg = ray_cfg["scheduler"]
    scheduler = ASHAScheduler(
        metric=args.metric,
        mode="max",
        max_t=scheduler_cfg["max_t"],
        grace_period=scheduler_cfg["grace_period"],
        reduction_factor=scheduler_cfg["reduction_factor"],
    )

    gpu_per_trial = ray_cfg.get("gpu_per_trial", 1)
    cpu_per_trial = ray_cfg.get("cpu_per_trial", 2)
    max_concurrent = ray_cfg.get("max_concurrent_trials", 1)

    trainable = tune.with_parameters(
        train_yolo,
        model_path=model_path,
        dataset_path=dataset_path,
        base_cfg=dict(tune_cfg),
    )
    trainable_with_resources = tune.with_resources(
        trainable,
        resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=str(Path("models", "tuned", "ray_results").resolve()),
        ),
    )

    logger.info(
        "Starting Ray Tune: %d samples, metric=%s, device=%s",
        num_samples, args.metric, device,
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric=args.metric, mode="max")
    if best_result is None:
        logger.error("All trials failed — no best result available.")
        return

    best_config = best_result.config
    best_metrics = best_result.metrics

    logger.info("Best trial config: %s", best_config)
    logger.info("Best trial %s: %.4f", args.metric, best_metrics.get(args.metric, 0.0))

    clearml_logger = task.get_logger()
    for k, v in best_config.items():
        clearml_logger.report_single_value(f"best_hp/{k}", v)
    for k, v in best_metrics.items():
        if isinstance(v, (int, float)):
            clearml_logger.report_single_value(f"best_metric/{k}", v)

    best_hp_path = Path("models", "tuned") / "best_hyperparams.yaml"
    best_hp_path.parent.mkdir(parents=True, exist_ok=True)
    with best_hp_path.open("w") as f:
        yaml.dump(
            {
                "config": best_config,
                "metrics": {k: v for k, v in best_metrics.items() if isinstance(v, (int, float))},
            },
            f,
            default_flow_style=False,
        )
    logger.info("Best hyperparameters saved to %s", best_hp_path)


if __name__ == "__main__":
    main()
