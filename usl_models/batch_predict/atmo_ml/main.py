# python usl_models/batch_predict/atmo_ml/main.py

import tensorflow as tf
from usl_models.atmo_ml.model import AtmoModel
from usl_models.atmo_ml import dataset
from usl_models.shared import env


BATCH_TASK_INDEX = env.getenv("BATCH_TASK_INDEX", int, default=0)
BATCH_TASK_COUNT = env.getenv("BATCH_TASK_COUNT", int, default=1)
SIMULATION_NAMES = [
    "NYC_Heat_Test/NYC_summer_2000_01p",
    "NYC_Heat_Test/NYC_summer_2010_99p",
]
OUTPUT_BUCKET = "climateiq-atmo-predictions"


def main():
    print("BATCH_TASK_INDEX:", BATCH_TASK_INDEX)
    sim_names = [
        sim_name
        for i, sim_name in enumerate(SIMULATION_NAMES)
        if i % BATCH_TASK_COUNT == BATCH_TASK_INDEX
    ]
    assert len(tf.config.list_physical_devices("GPU")) > 0
    model = AtmoModel.from_checkpoint(
        "gs://climateiq-vertexai/atmoml-main-20250319-204950/model"
    )
    ds = dataset.load_dataset(
        data_bucket_name=dataset.FEATURE_BUCKET_NAME,
        label_bucket_name=dataset.LABEL_BUCKET_NAME,
        sim_names=sim_names,
    ).batch(batch_size=8)
    for input, _ in ds:
        pred = model.call(input)
        print("Pred shape:", pred.shape)
        break


if __name__ == "__main__":
    main()
