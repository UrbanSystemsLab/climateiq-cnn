import os
import subprocess
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from dynaconf import Dynaconf

import tensorflow as tf


class VertexAIHyperparameterTuning:
    def __init__(self, model_dir, package_path, config_file="config/hpt_config.json"):
        self.model_dir = model_dir
        self.package_path = package_path
        self.config = Dynaconf(
            settings_files=[config_file], environments=True, load_dotenv=True
        )
        self.project_id = self.config.get("project_id")
        self.location = self.config.get("location")
        self.staging_bucket_name = self.config.get("bucket_name")
        self.docker_base_image = self.config.get("docker_base_image")
        self.image_name = self.config.get("image_name")
        self.bucket_name = self.config.get("bucket_name")
        self.hyperparameter_tuning_job_display_name = self.config.get(
            "hyperparameter_tuning_job_display_name"
        )
        self.training_job_display_name = self.config.get("training_job_display_name")
        self.study_areas = self.config.get("study_areas")
        self.worker_pool_specs = self.config.get("worker_pool_specs")

        # Initialize the Vertex AI SDK
        aiplatform.init(
            project=self.project_id,
            location=self.location,
            staging_bucket=self.bucket_name,
        )

    def create_docker_container(docker_base_image, tag="training-job"):
        """Creates a Docker container for a training job."""

        project_dir = "~/climateiq-cnn/usl_models/usl_models"
        flood_model_dir = os.path.join(project_dir, "flood_ml")

        # Create the Dockerfile
        with open(os.path.join(project_dir, "Dockerfile"), "w") as f:
            f.write(
                f"""FROM {docker_base_image} WORKDIR /

            # Installs hypertune library
            RUN pip install cloudml-hypertune
            RUN curl -sSL https://sdk.cloud.google.com | bash

            ENV PATH $PATH:/root/google-cloud-sdk/bin

            # Copy the entire flood_model directory
            COPY {flood_model_dir} /flood_model

            # Sets up the entry point to invoke the trainer.
            # Assuming task.py is inside flood_model
            ENTRYPOINT = f"""[
                    "python",
                    "-m",
                    "flood_model.train_model",
                    "train_model",
                    "--model_dir",
                    "{self.model_dir}",
                    "--package_path",
                    "{self.package_path}",
                ]
            )

        # Build the Docker image
        try:
            output = subprocess.check_output(
                ["docker", "build", "-t", tag, "."],
                stderr=subprocess.STDOUT,  # Capture errors for better debugging
                text=True,  # Get output as text
            )
            print(output)  # Print the build output to the console
            return output  # Return output for further use or logging
        except subprocess.CalledProcessError as e:
            print(f"Error building Docker image: {e.output}")
            raise  # Reraise the exception for error handling

    def create_hyperparameter_tuning_job(self):
        """Creates a hyperparameter tuning job."""

        # Create the custom training job
        hp_job = aiplatform.CustomJob(
            display_name=self.training_job_display_name,
            worker_pool_specs=self.worker_pool_specs,
            staging_bucket=self.staging_bucket_name,
        )

        # Create the hyperparameter tuning job
        hp_tuning_job = aiplatform.HyperparameterTuningJob(
            display_name=self.hyperparameter_tuning_job_display_name,
            custom_job=hp_job,
            metric_spec={
                tf.keras.metrics.MeanAbsoluteError(): "minimize",
                tf.keras.metrics.RootMeanSquaredError(): "minimize",
            },
            parameter_spec={
                "learning_rate": hpt.DoubleParameterSpec(
                    min=1e-4, max=1e-1, scale="log"
                ),
                "batch_size": hpt.DiscreteParameterSpec(values=[16, 32, 64, 128]),
                "epochs": hpt.DiscreteParameterSpec(
                    values=[10, 20, 40, 60], scale="linear"
                ),
                "lstm_units": hpt.DiscreteParameterSpec(
                    values=[128, 256, 512], scale="linear"
                ),
                "lstm_kernel_size": hpt.DiscreteParameterSpec(
                    values=[3, 5, 7], scale="linear"
                ),
                "lstm_dropout": hpt.DoubleParameterSpec(
                    min=0.0, max=0.5, scale="linear"
                ),
                "lstm_recurrent_dropout": hpt.DoubleParameterSpec(
                    min=0.0, max=0.5, scale="linear"
                ),
            },
            max_trial_count=10,  # The total number of trials.
            parallel_trial_count=2,  # The number of parallel trials.
            worker_pool_specs=self.worker_pool_specs,  # The worker pool configuration
        )

        hp_tuning_job.run()


def __main__():
    model_dir = os.environ["AIP_MODEL_DIR"]
    package_path = os.environ["AIP_PACKAGE_PATH"]
    VertexAIHyperparameterTuning(
        model_dir, package_path
    ).create_hyperparameter_tuning_job()


if __name__ == "__main__":
    __main__()
