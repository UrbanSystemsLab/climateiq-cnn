import argparse
import logging
import os

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import tensorflow as tf
from tensorflow.python.client import device_lib

import usl_models.flood_ml.dataset
import usl_models.flood_ml.model
from usl_models.flood_ml import metastore
import usl_models.flood_ml.model_params


logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
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

logging.info("DEVICES" + str(device_lib.list_local_devices()))

# Single Machine, single compute device
if args.distribute == "single":
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    logging.info("Single device training")
# Single Machine, multiple compute device
elif args.distribute == "mirrored":
    strategy = tf.distribute.MirroredStrategy()
    logging.info("Mirrored Strategy distributed training")
# Multi Machine, multiple compute device
elif args.distribute == "multiworker":
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    logging.info("Multi-worker Strategy distributed training")
    logging.info("TF_CONFIG = {}".format(os.environ.get("TF_CONFIG", "Not found")))
    # Single Machine, multiple TPU devices
elif args.distribute == "tpu":
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("All devices: ", tf.config.list_logical_devices("TPU"))

logging.info("num_replicas_in_sync = {}".format(strategy.num_replicas_in_sync))


def _is_chief(task_type, task_id):
    """Checks for primary if multiworker training."""
    return (
        (task_type == "chief")
        or (task_type == "worker" and task_id == 0)
        or task_type is None
    )


def train(
    model: usl_models.flood_ml.model.FloodModel,
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

    if args.distribute == "tpu":
        save_locally = tf.saved_model.SaveOptions(
            experimental_io_device="/job:localhost"
        )
        logging.info("Saving model to %s", args.model_dir)
        model_dir = args.model_dir
        model.save_model(model_dir, options=save_locally)
    # single, mirrored or primary for multiworker
    elif _is_chief(task_type, task_id):
        logging.info("Saving model to %s", args.model_dir)
        model_dir = args.model_dir
        model.save_model(model_dir)
    # non-primary workers for multi-workers
    else:
        # each worker saves their model instance to a unique temp location
        model_dir = args.model_dir + "/workertemp_" + str(task_id)
        logging.info("Saving model to %s", model_dir)
        tf.io.gfile.makedirs(model_dir)
        model.save_model(model_dir)

    metastore.write_model_metadata(
        firestore_client,
        gcs_model_dir=model_dir,
        sim_names=args.sim_names,
        model_params=model._model_params,
    )


with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within
    # `strategy.scope()`.
    firestore_client = firestore.Client(project="climateiq")

    model_params = usl_models.flood_ml.model_params.default_params()
    if args.batch_size is not None:
        model_params["batch_size"] = args.batch_size
    model = usl_models.flood_ml.model.FloodModel(model_params=model_params)

    kwargs = {}
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    train_dataset = usl_models.flood_ml.dataset.load_dataset_windowed(
        sim_names=args.sim_names,
        dataset_split="train",
        firestore_client=firestore.Client(project="climateiq"),
        storage_client=storage.Client(project="climateiq"),
        **kwargs,
    )
    val_dataset = usl_models.flood_ml.dataset.load_dataset_windowed(
        sim_names=args.sim_names,
        dataset_split="val",
        firestore_client=firestore_client,
        storage_client=storage.Client(project="climateiq"),
        **kwargs,
    )


train(model, train_dataset, val_dataset=val_dataset, firestore_client=firestore_client)
