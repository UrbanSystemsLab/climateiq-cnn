import tensorflow as tf

from metastore import FirestoreDataHandler
from settings import Settings

"""
This class will be used to generate the geospatial tensor for the flood model.

The Geospatial Tensor represents *static* information about the Earth's surface at a specific point in time.
Typically includes features like elevation, slope, land cover, and other physical characteristics.
Useful for tasks like land use analysis, environmental monitoring, and disaster preparedness.

The size of the geospatial_tensor is typically determined by the resolution  of the underlying geospatial data.
For example, if the data has a resolution of 10 meters per pixel, then a 1000x1000 tensor would cover an area of
100,000 square meters.

"""


class GeospatialTensor:
    def __init__(
        self,
        settings: Settings = None,
        metastore: FirestoreDataHandler = (None,),
    ):
        settings: Settings = None
        metastore: FirestoreDataHandler = (None,)
        # instantiate metastore class
        self.metastore = metastore or FirestoreDataHandler(
            firestore_client=self.firestore_client, settings=self.settings
        )
        # Load settings
        self.settings = settings or Settings()

        self.study_area = None
        self.data = None

    def generate_geospatial_tensor(self, study_area, data):
        self.study_area = study_area
        self.data = data

        spatio_temporal_tensor = tf.zeros([1000, 1000, 1])
        return spatio_temporal_tensor
