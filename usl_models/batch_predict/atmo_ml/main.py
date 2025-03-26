r"""
Local test:

python usl_models/batch_predict/atmo_ml/main.py \
    --batch_size=2 \
    --max_batches=4 \
    --model_path="gs://climateiq-vertexai/atmoml-main-20250319-204950/model" \
    --output_bucket="climateiq-predictions" \
    --output_path="atmoml-test/$(date +%Y%m%d-%H%M%S)"
"""

import argparse
import logging

import tensorflow as tf

from google.cloud import storage

from usl_models.atmo_ml.model import AtmoModel
from usl_models.atmo_ml import dataset
from usl_models.shared import env
from usl_models.shared import uploader

logging.basicConfig(level=logging.INFO)

BATCH_TASK_INDEX = env.getenv("BATCH_TASK_INDEX", int, default=0)
BATCH_TASK_COUNT = env.getenv("BATCH_TASK_COUNT", int, default=1)
SIMULATION_NAMES = [
    "NYC_Heat_Test/NYC_summer_2000_01p",
    "NYC_Heat_Test/NYC_summer_2010_99p",
]


def main(
    batch_size: int,
    output_bucket: str,
    output_path: str,
    model_path: str,
    max_batches: int,
):
    print("BATCH_TASK_INDEX:", BATCH_TASK_INDEX)
    sim_names = [
        sim_name
        for i, sim_name in enumerate(SIMULATION_NAMES)
        if i % BATCH_TASK_COUNT == BATCH_TASK_INDEX
    ]

    gpu_devices = tf.config.list_physical_devices("GPU")
    assert len(gpu_devices) > 0, "Batch prediction requires a GPU."

    client = storage.Client()
    output_bucket = client.bucket(output_bucket)

    ds = dataset.load_dataset(
        data_bucket_name=dataset.FEATURE_BUCKET_NAME,
        label_bucket_name=dataset.LABEL_BUCKET_NAME,
        sim_names=sim_names,
    ).batch(batch_size=batch_size)
    if max_batches > 0:
        ds = ds.take(max_batches)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = AtmoModel.from_checkpoint(model_path)
        for inputs, _ in ds:
            preds = model.predict(inputs)
            for b, pred in enumerate(preds):
                pred_npy = pred[b]

                sim_name = inputs["sim_name"][b].numpy().decode("utf-8")
                date = inputs["date"][b].numpy().decode("utf-8")
                path = f"{output_path}/{sim_name}/{date}.npy"
                uploader.upload_npy(
                    pred_npy,
                    bucket=output_bucket,
                    path=path,
                )
                logging.info("Uploaded file: %s", f"{output_bucket}/{path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=2, help="Size of a batch."
    )
    parser.add_argument(
        "--output_bucket",
        dest="output_bucket",
        type=str,
        default="",
        help="Output bucket.",
    )
    parser.add_argument(
        "--output_path", dest="output_path", type=str, default="", help="Output path."
    )
    parser.add_argument(
        "--model_path", dest="model_path", type=str, default="", help="Model path."
    )
    parser.add_argument(
        "--max_batches",
        dest="max_batches",
        type=int,
        default=0,
        help="Max predictions.",
    )
    main(**vars(parser.parse_args()))
