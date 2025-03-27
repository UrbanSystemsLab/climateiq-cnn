"""Templates for creating batch jobs on GCP."""

import time
from google.cloud import batch_v1

CLIMATEIQ_IMAGE_URI = "us-central1-docker.pkg.dev/climateiq/usl-models/atmo-ml-test:dev"


NVIDIA_L4_X2_INSTANCE = batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
    policy=batch_v1.AllocationPolicy.InstancePolicy(
        boot_disk=batch_v1.AllocationPolicy.Disk(size_gb=100),
        machine_type="g2-standard-24",
        accelerators=[
            batch_v1.AllocationPolicy.Accelerator(
                type_="nvidia-l4",
                count=2,
                driver_version="550.90.07",
            )
        ],
    ),
    install_gpu_drivers=True,
)


def await_batch_job(
    client: batch_v1.BatchServiceClient,
    job_id: str,
    job_name: str,
    poll_rate_secs: float = 10.0,
):
    """Awaits a batch job."""
    while True:
        job = client.get_job(name=job_name)
        if job.status.state in (
            batch_v1.JobStatus.State.SUCCEEDED,
            batch_v1.JobStatus.State.FAILED,
            batch_v1.JobStatus.State.CANCELLED,
        ):
            print(f"Job {job_id} completed with state: {job.status.state.name}")
            if job.status.state == batch_v1.JobStatus.State.FAILED:
                print(f"Job failed with status: {job.status}")
        else:
            print(f"Job {job_id} is in state: {job.status.state.name}. Waiting...")
            time.sleep(poll_rate_secs)
