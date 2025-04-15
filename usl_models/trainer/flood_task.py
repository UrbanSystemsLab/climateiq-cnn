import argparse
import logging
import os
import dataclasses

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import tensorflow as tf
from tensorflow.python.client import device_lib

import usl_models.flood_ml.dataset
import usl_models.flood_ml.model
from usl_models.flood_ml import metastore
from usl_models.flood_ml.model import FloodModel

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name", dest="model_name", type=str, help="A name for the model."
)
parser.add_argument("--epochs", dest="epochs", type=int, help="Number of epochs.")
parser.add_argument(
    "--batch-size", dest="batch_size", type=int, help="Size of a batch."
)
parser.add_argument("--sim-names", dest="sim_names", nargs="+", type=str, required=True)

parser.add_argument(
    "--model-dir",
    dest="model_dir",
    default=os.environ.get("AIP_MODEL_DIR"),
    type=str,
    help="Model dir.",
)
parser.add_argument(
    "--distribute",
    dest="distribute",
    type=str,
    default="single",
    help="distributed training strategy",
    choices=["single", "mirrored", "multiworker", "tpu"],
)

args = parser.parse_args()

logging.info("DEVICES: %s", device_lib.list_local_devices())

# Distributed strategy setup
if args.distribute == "single":
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    logging.info("Using Single device strategy.")
elif args.distribute == "mirrored":
    strategy = tf.distribute.MirroredStrategy()
    logging.info("Using Mirrored Strategy.")
elif args.distribute == "multiworker":
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    logging.info("Using Multi-worker Strategy.")
    logging.info("TF_CONFIG: %s", os.environ.get("TF_CONFIG", "Not found"))
elif args.distribute == "tpu":
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    logging.info("Using TPU Strategy.")
    logging.info("All devices: %s", tf.config.list_logical_devices("TPU"))

logging.info("num_replicas_in_sync = %d", strategy.num_replicas_in_sync)


def _is_chief(task_type, task_id):
    """Checks for primary if multiworker training."""
    return (
        (task_type == "chief")
        or (task_type == "worker" and task_id == 0)
        or task_type is None
    )


def train(
    model: FloodModel,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    firestore_client: firestore.Client,
) -> None:
    """Trains a model with the given dataset and saves the model to GCS."""
    kwargs = {}
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs

    model.fit(train_dataset, val_dataset=val_dataset, **kwargs)

    if args.distribute == "multiworker":
        task_type, task_id = (
            strategy.cluster_resolver.task_type,
            strategy.cluster_resolver.task_id,
        )
    else:
        task_type, task_id = None, None

    # Determine model saving path
    if args.distribute == "tpu":
        save_options = tf.saved_model.SaveOptions(
            experimental_io_device="/job:localhost"
        )
        model_dir = args.model_dir
        logging.info("Saving model to %s", model_dir)
        model.save_model(model_dir, options=save_options)
    elif _is_chief(task_type, task_id):
        model_dir = args.model_dir
        logging.info("Saving model to %s", model_dir)
        model.save_model(model_dir)
    else:
        model_dir = f"{args.model_dir}/workertemp_{task_id}"
        logging.info("Saving model to %s", model_dir)
        tf.io.gfile.makedirs(model_dir)
        model.save_model(model_dir)

    # ✅ Pass dict for FloodModelParams
    metastore.write_model_metadata(
        firestore_client,
        gcs_model_dir=model_dir,
        sim_names=args.sim_names,
        model_params=dataclasses.asdict(model._params),
        epochs=args.epochs,
        model_name=args.model_name,
    )


with strategy.scope():
    firestore_client = firestore.Client(project="climateiq")

    model_params = FloodModel.Params()

    model = FloodModel(params=model_params)

    logging.info(
        "Training model for %s epochs with params %s",
        args.epochs,
        dataclasses.asdict(model_params),
    )

    dataset_kwargs = {}
    if args.batch_size is not None:
        dataset_kwargs["batch_size"] = args.batch_size

    train_dataset = usl_models.flood_ml.dataset.load_dataset_windowed(
        sim_names=args.sim_names,
        dataset_split="train",
        firestore_client=firestore_client,
        storage_client=storage.Client(project="climateiq"),
        **dataset_kwargs,
    )
    val_dataset = usl_models.flood_ml.dataset.load_dataset_windowed(
        sim_names=args.sim_names,
        dataset_split="val",
        firestore_client=firestore_client,
        storage_client=storage.Client(project="climateiq"),
        **dataset_kwargs,
    )

train(model, train_dataset, val_dataset, firestore_client)
