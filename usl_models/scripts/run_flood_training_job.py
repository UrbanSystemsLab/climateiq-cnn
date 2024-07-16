import argparse
import distutils.core
import os
import pathlib

from google.cloud import aiplatform
from google.cloud import storage  # type:ignore[attr-defined]

IMAGE = "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest"


def main():
    cli_args = _parse_args()

    # Build a source distribution for the usl_models package.
    this_file = pathlib.Path(os.path.realpath(__file__))
    package_dir = this_file.parent.parent
    setup_path = package_dir / "setup.py"
    distutils.core.run_setup(setup_path, script_args=["sdist", "--format=gztar"])

    # Upload the usl_models package to GCS.
    source_dist_path = package_dir / "dist" / "usl_models-0.0.0.tar.gz"
    client = storage.Client(project="climateiq")
    bucket = client.bucket("climateiq-vertexai")
    bucket.blob("usl_models-0.0.0.tar.gz").upload_from_filename(str(source_dist_path))

    # Run the training script trainer/flood_task.py in VertexAI.
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name="walt-test",
        python_package_gcs_uri="gs://climateiq-vertexai/usl_models-0.0.0.tar.gz",
        python_module_name="trainer.flood_task",
        container_uri=IMAGE,
        model_serving_container_image_uri=IMAGE,
        staging_bucket="gs://climateiq-vertexai",
    )

    job_args = ["--sim-names", *cli_args.sim_names]
    if cli_args.epochs:
        job_args.extend(("--epochs", str(cli_args.epochs)))
    if cli_args.batch_size:
        job_args.extend(("--batch-size", str(cli_args.batch_size)))

    print(f"Creating training job with arguments {job_args}")
    job.run(
        model_display_name="flood-model",
        args=job_args,
        replica_count=1,
        machine_type="a2-highgpu-1g",
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
        sync=True,
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", type=int, help="Number of epochs.")
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, help="Size of a batch."
    )
    parser.add_argument(
        "--sim-names",
        dest="sim_names",
        nargs="+",
        type=str,
        required=True,
        help=(
            "Space-separated set of simulations to train the model against, e.g. "
            "--sim-names "
            "Manhattan-config_v1/Rainfall_Data_1.txt "
            "Manhattan-config_v1/Rainfall_Data_2.txt"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
