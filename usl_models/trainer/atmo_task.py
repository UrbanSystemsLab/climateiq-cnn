r"""Run training on vertex AI for Atmo ML model.

Command:
cd usl_models
python trainer/atmo_task.py
"""
import argparse
import logging
import os

from google.cloud import firestore
from google.cloud import storage
import tensorflow as tf
from tensorflow.python.client import device_lib

from usl_models.atmo_ml import model as atmo_model
from usl_models.atmo_ml import metastore
from usl_models.atmo_ml import model_params as atmo_model_params
from usl_models.atmo_ml import dataset

logging.basicConfig(level=logging.INFO)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name", dest="model_name", type=str, help="A name for the model."
)
parser.add_argument("--epochs", dest="epochs", type=int, help="Number of epochs.")
parser.add_argument(
    "--batch-size", dest="batch_size", type=int, help="Size of a batch."
)
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

# Define the distribution strategy
if args.distribute == "single":
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    logging.info("Single device training")
elif args.distribute == "mirrored":
    strategy = tf.distribute.MirroredStrategy()
    logging.info("Mirrored Strategy distributed training")
elif args.distribute == "multiworker":
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    logging.info("Multi-worker Strategy distributed training")
    logging.info("TF_CONFIG = {}".format(os.environ.get("TF_CONFIG", "Not found")))
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
    model: atmo_model.AtmoModel,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    sim_names: list[str]
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

    # Save model depending on distribution strategy
    if args.distribute == "tpu":
        save_locally = tf.saved_model.SaveOptions(
            experimental_io_device="/job:localhost"
        )
        logging.info("Saving model to %s", args.model_dir)
        model_dir = args.model_dir
        model.save_model(model_dir, options=save_locally)
    elif _is_chief(task_type, task_id):
        logging.info("Saving model to %s", args.model_dir)
        model_dir = args.model_dir
        model.save_model(model_dir)
    else:
        model_dir = args.model_dir + "/workertemp_" + str(task_id)
        logging.info("Saving model to %s", model_dir)
        tf.io.gfile.makedirs(model_dir)
        model.save_model(model_dir)

    metastore.write_model_metadata(
        firestore_client,
        gcs_model_dir=model_dir,
        sim_names=sim_names,
        model_params=model._model_params,
        epochs=args.epochs,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    sim_dirs = [
        ('NYC_Heat_Test', [
            'NYC_summer_2000_01p',
            'NYC_summer_2010_25p',
            'NYC_summer_2015_50p',
            'NYC_summer_2017_75p',
            'NYC_summer_2018_99p'
        ]),
        ('PHX_Heat_Test', [
            'PHX_summer_2008_25p',
            'PHX_summer_2009_50p',
            'PHX_summer_2011_99p',
            'PHX_summer_2015_75p',
            'PHX_summer_2020_01p'
        ])
    ]

    sim_names = []
    for sim_dir, subdirs in sim_dirs:
        for subdir in subdirs:
            sim_names.append(sim_dir + '/' + subdir)

    with strategy.scope():
        data_bucket_name = "climateiq-study-area-feature-chunks"
        label_bucket_name = "climateiq-study-area-label-chunks"
        time_steps_per_day = 6
        batch_size = 4
        firestore_client = firestore.Client(project="climateiq")
        model_params = atmo_model_params.default_params()
        if args.batch_size is not None:
            model_params["batch_size"] = args.batch_size
        model = atmo_model.AtmoModel(params=model_params)
        logging.info(
            "Training model for %s epochs with params %s", args.epochs, model_params
        )
        storage_client = storage.Client(project="climateiq")
        ds = dataset.load_dataset(
            data_bucket_name=data_bucket_name,
            label_bucket_name=label_bucket_name,
            sim_names=sim_names,
            timesteps_per_day=time_steps_per_day,
            storage_client=storage_client
        ).batch(batch_size=4)
        train_dataset, val_dataset, test_dataset = dataset.split_dataset(
            ds, train_frac=0.7, val_frac=0.15, test_frac=0.15
        )

        train(model, train_dataset, val_dataset, sim_names)
