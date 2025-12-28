Getting Started
===============

To install the package and its requirements for local development:
- pip install -r requirements.txt
- pip install -e .[dev]

To run the unit tests:
- pytest -k 'not integration'

To run the linter:
- flake8 .

To run the type checker:
- mypy --config-file ../mypy.ini .

To run the code auto-formatter:
- black .

Integration Tests
=================
The instructions above run unit tests and skip integration tests with
`pytest -k 'not integration'`.
Running the integration tests requires the Firestore emulator.
Follow
[these instructions](https://cloud.google.com/firestore/docs/emulator)
to install and run the emulator.

Start the emulator and be sure to set the `FIRESTORE_EMULATOR_HOST`
environmental variable as described in the documentation.  Once you've
done so, you may now run the full test suite, including integration
tests, with `pytest` (without the `-k 'not integration'` bit.)


GCP Training
============
To train a model in GCP using VertexAI, the script `scripts/run_flood_training_job.py` is provided.
The script does the following:
- Copies this package into Google Cloud Storage so it can be used by VertexAI.
  The local state of the repo, including any un-commited changes, will be uploaded into GCS.
  So make sure the current state of the repo represents what you want to run in GCP.
- Creates a VertexAI training job which executes the training script in `trainer/flood_task.py`.
  The script will print out links to the VertexAI jobs running the script.
  You can follow those links to monitor progress and check their logs for issues.
- Prints status about the training job's progress.
  You will see messages when the training job completes or hits an error.
  Cancelling the script will not affect the VertexAI training job running in GCP.
  You can safely close your laptop or cancel the script if you wish --
  the VertexAI training job will keep running in GCP.

Run the script with:
```bash
python run_flood_training_job.py --sim-names <simulation-1> <simulation2> ...
```
You can also state the number of epochs and batch size if you do not with to use their default values:
```bash
python run_flood_training_job.py --batch-size <batch-size> --epochs <n-epochs> --sim-names <simulation-1> <simulation2> ...
```
The available simulation names can be found in the simulations collection of
[the metastore](https://console.cloud.google.com/firestore/databases/-default-).
The names in the metastore are URL encoded, meaning a name like
`Manhattan-config_v1%2FRainfall_Data_1.txt`
is equivalent to
`Manhattan-config_v1/Rainfall_Data_1.txt`
You can supply either one as a simulation name.
e.g.
```bash
python run_flood_training_job.py --sim-names 'Manhattan-config_v1/Rainfall_Data_1.txt Manhattan-config_v1/Rainfall_Data_12.txt'
```

The script requires that the service agent IAM
`gcp-sa-aiplatform-cc.iam.gserviceaccount.com` be given the 'Cloud
Datastore User' role in order to access the Firestore metastore. This
is already configured in the climateiq project.

### Debugging GCP Training
To perform training, VertexAI will run the script in `trainer/flood_task.py`.
You can run this script yourself to debug problems:
```bash
python trainer/flood_task.py --sim-names <simulation-1> <simulation2> ...
```

If there are issues installing the usl_models package in VertexAI,
you can exactly mirror the VertexAI training environment using
[Docker](https://www.docker.com/).
You'll need to [install Docker](https://docs.docker.com/engine/install/).
You can then do:
```shell
docker pull us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310
docker run --rm -it --entrypoint /bin/bash --mount type=bind,source="$(pwd)",target=/usl_models us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310
root@eaf6dfb394d4:/# cd /usl_models/
root@eaf6dfb394d4:/usl_models# pip install -e .
```
to replicate VertexAI's installation of the usl_models package.

### Cuda installation

We currently use Cuda 12.3 for compatability with Tensorflow 2.16.1.

Tensorflow compatibiltiy table: https://www.tensorflow.org/install/source#gpu

Debian installer instructions:
https://developer.nvidia.com/cuda-12-3-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network
