import tensorflow as tf

from metastore import FirestoreDataHandler
from settings import Settings

"""
This class will be used to generate the geospatial tensor for the flood model.

The Spatiotemporal Tensor represents *dynamic* information about the Earth's surface over time.
Includes the same features as a geospatial tensor, but also incorporates temporal data like historical events,
weather patterns, and changes in land cover. Useful for tasks like flood modeling, climate change analysis, and
predicting future environmental conditions.

The size of the spatiotemporal_tensor is typically determined by the resolution  of the underlying geospatial data.
For example, if the data has a resolution of 10 meters per pixel, then a 1000x1000 tensor would cover an area of
10km x 10km.

Dimensions:

Height (H): This represents the vertical dimension of the map, typically corresponding to the latitude or y-axis.
Width (W): This represents the horizontal dimension of the map, typically corresponding to the longitude or x-axis.
Features (f): This represents the number of different geospatial features included in the tensor. In this case, the
three features: elevation, slope, and land cover.

The number of features (f) can be expanded to include additional relevant information, such as rainfall data, 
soil moisture, or historical flood events.
"""


class Spatiotemporal:
    def __init__(
        self,
        study_area,
        data,
        settings: Settings = None,
        metastore: FirestoreDataHandler = (None,),
    ):
        self.generate_geospatial_tensor()
        settings: Settings = None
        metastore: FirestoreDataHandler = (None,)
        self.study_area = study_area
        self.data = data
        # instantiate metastore class
        self.metastore = metastore or FirestoreDataHandler(
            firestore_client=self.firestore_client, settings=self.settings
        )
        # Load settings
        self.settings = settings or Settings()

    def generate_spatiotemporal_data(self):
        spatio_temporal_tensor = tf.zeros([1000, 1000, 1])
        return spatio_temporal_tensor
