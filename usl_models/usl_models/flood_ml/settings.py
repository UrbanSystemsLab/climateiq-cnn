from decouple import config

"""
Settings for the training pipeline.

"""


class Settings:
    def __init__(self):
        self.LOCAL_NUMPY_DIR = config("LOCAL_NUMPY_DIR", default="numpy_data")
        self.DEBUG = config("DEBUG", cast=int, default=1)
        self.MAX_WORKERS = config("MAX_WORKERS", cast=int, default=4)
        # FIRESTORE_COLLECTION = config("FIRESTORE_COLLECTION")

        # set default value for all settings if not set in the environment variables
        self.LOCAL_NUMPY_DIR = self.LOCAL_NUMPY_DIR or "numpy_data"
        self.DEBUG = self.DEBUG or 1
        self.MAX_WORKERS = self.MAX_WORKERS or 4
        # FIRESTORE_COLLECTION = FIRESTORE_COLLECTION or "flood_ml"
