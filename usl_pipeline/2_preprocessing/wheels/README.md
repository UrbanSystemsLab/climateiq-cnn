# Python wheels
This directory contains pre-generated wheels required for installing packages that may lack built distributions in PyPi that we need.
Currently, this is necessary for installing `wrf-python` on linux environments (in particular, for running within cloud functions).

## wrf-python
* PyPi:  https://pypi.org/project/wrf-python/
* Latest version: `1.3.4.1` (Released in 2022)

Currently in PyPi, the only available built distribution is for macOS - which will not be compatible in our Github CI workflow environment
or Cloud Functions environment.

We have pre-generated a linux-compatible wheel that will be used when installing requirements in a linux environment. A conditional
is used in `requirements.txt` such that when in a non-linux environment, pip will install from PyPi (using macOS wheel or build from source distribution).

### How to generate linux wheel
It's unlikely we'll get newer releases of `wrf-python` considering the last update was a few years ago, but if we do, we can easily re-generate the wheel and check the newer version in. Here are the steps to do it:

1. Prerequisites:
    * **Python 3.8+**: Ensure you have a compatible Python version installed. (wrf-python officially supports Python 3.8, 3.9, 3.10, and 3.11)
    * **pip**: The Python package installer.
    * **NumPy**: A fundamental package for numerical operations in Python. (Install using `pip install numpy`)
    * **setuptools**: A library for packaging Python projects. (Usually installed with pip)
    * **wheel**: The library for building wheels. (Install with `pip install wheel`)
    * **Fortran Compiler**: wrf-python is written in Fortran. You'll need a Fortran compiler like gfortran or ifort installed. (On Ubuntu/Debian, you can install it with `sudo apt install gfortran`)

2. Download the source distribution tar file from PyPi: https://pypi.org/project/wrf-python/#files

3. Unpack the tar.gz file
    ```bash
    tar -xvzf wrf-python-<_version_>.tar.gz
    cd wrf-python-<_version_>
    ```

4. Build the wheel
    ```bash
    export FC=gfortran #Or the compiler you have installed
    export F77=gfortran #Or the compiler you have installed
    python setup.py build #Optional, but recommended 
    python setup.py bdist_wheel
    ```
    * If you are experiencing errors with the compiler, try to run the build step in verbose mode with `python setup.py build --verbose`.
    * **Important**: The `export` commands set the environment variables `FC` and `F77` to tell the build process to use `gfortran` as the Fortran compiler. Adjust these if you are using a different Fortran compiler.

5. Check that the wheel works by installing it
    ```bash
    pip install dist/wrf_python-<_version_>-cp38-cp38-linux_x86_64.whl
    ```
    (The exact file name might vary slightly depending on your Python version and system architecture. The example above is for Python 3.8 on 64-bit Linux.)

6. Push up new version of wheel and submit it in Github

#### Troubleshooting
* **Compiler Errors**: If you encounter errors during the build process, make sure you have the correct Fortran compiler installed and the environment variables FC and F77 are set correctly.
* **NumPy Version**: `wrf-python` requires a specific version of NumPy. Refer to the wrf-python documentation to ensure compatibility.
    * For ClimateIQ - we use `numpy<1.26`