import dataclasses
from typing import Optional
import urllib.parse

from google.cloud import firestore
import rasterio

from usl_lib.shared import geo_data


STUDY_AREAS = "study_areas"
STUDY_AREA_CHUNKS = "chunks"

CITY_CAT_RAINFALL_CONFIG = "city_cat_rainfall_configs"


@dataclasses.dataclass(slots=True)
class StudyArea:
    """A geography for either training or prediction.

    Attributes:
        name: A human-readable name of the study area. Must be unique.
        col_count: Number of columns in the raster of the area.
        row_count: Number of rows in the raster of the area.
        x_ll_corner: X-coordinate of the raster's origin.
        y_ll_corner: Y-coordinate of the raster's origin.
        cell_size: Size of cells in the raster.
        crs: Coordinate system connecting global geographic coordinates and local
            projected ones.
    """

    name: str
    col_count: int
    row_count: int
    x_ll_corner: float
    y_ll_corner: float
    cell_size: float
    crs: Optional[str] = None
    elevation_min: Optional[float] = None
    elevation_max: Optional[float] = None

    def as_header(self) -> geo_data.ElevationHeader:
        """Represents the study area as a header describing a raster space."""
        return geo_data.ElevationHeader(
            col_count=self.col_count,
            row_count=self.row_count,
            x_ll_corner=self.x_ll_corner,
            y_ll_corner=self.y_ll_corner,
            cell_size=self.cell_size,
            nodata_value=-9999.0,
            crs=rasterio.CRS.from_string(self.crs) if self.crs is not None else None,
        )

    def create(self, db: firestore.Client) -> None:
        """Creates a new study area in the given DB. Only writes non-None attributes.

        Args:
          db: The firestore database client to use for the read.
        """
        self.get_ref(db, self.name).create(self._as_dict())

    def set(self, db: firestore.Client) -> None:
        """Creates or updates a new study area in the given DB.

        Only writes non-None attributes.

        Args:
          db: The firestore database client to use for the read.
        """
        self.get_ref(db, self.name).set(self._as_dict())

    @classmethod
    def get(cls, db: firestore.Client, name: str) -> "StudyArea":
        """Retrieve the study area with the given name.

        Args:
          db: The firestore database client to use for the read.
          study_area_name: The study area to retrieve.

        Returns:
          A StudyArea object representing the database's contents.
        """
        ref = cls.get_ref(db, name).get()
        if not ref.exists:
            raise ValueError(f'No such study area "{name}"')

        return cls(name=name, **ref.to_dict())

    @staticmethod
    def get_ref(db: firestore.Client, name: str) -> firestore.DocumentReference:
        """Retrieve a Firestore reference to the study area with the given name."""
        return db.collection(STUDY_AREAS).document(name)

    @staticmethod
    def update_min_max_elevation(
        db: firestore.Client, study_area_name: str, min_: float, max_: float
    ) -> None:
        """Sets elevation min & max if less or greater than the current min & max.

        For the given study area, transactionally updates the elevation_min and
        elevation_max field to the given min_ and max_ values if they are less than and
        greater than the current values.

        Args:
          db: The firestore database client to use to make the update.
          study_area_name: The study area to update.
          min_: The min elevation value for the study area. Will only be set if less
                than the study area's current min.
          max_: The max elevation value for the study area. Will only be set if less
                than the study area's current max.
        """
        study_area_ref = db.collection(STUDY_AREAS).document(study_area_name)
        transaction = db.transaction()
        _update_study_area_min_max_elevation(transaction, study_area_ref, min_, max_)

    def _as_dict(self) -> dict:
        as_dict = {}
        for field in self.__dataclass_fields__:
            if field == "name":
                continue

            value = getattr(self, field)
            if value is not None:
                as_dict[field] = value

        return as_dict


@dataclasses.dataclass(slots=True)
class StudyAreaChunk:
    """A sub-area chunk of a larger study area.

    Attributes:
        id_: An ID unique within the study are for the chunk.
        archive_path: GCS location of an archive containing files describing the
            geography (e.g. tiff & shape files.)
        feature_matrix_path: GCS location of the derived feature matrix used for model
            training and prediction.
        error: Any errors encountered while processing the chunk.
    """

    id_: str
    archive_path: Optional[str] = None
    feature_matrix_path: Optional[str] = None
    error: Optional[str] = None

    def merge(self, db: firestore.Client, study_area_name: str) -> None:
        """Creates or updates an existing chunk within the given study area.

        Only attempts to write non-None attributes.
        """
        as_dict = {}
        for field in self.__dataclass_fields__:
            if field == "id_":
                continue

            value = getattr(self, field)
            if value is not None:
                as_dict[field] = value

        (
            db.collection(STUDY_AREAS)
            .document(study_area_name)
            .collection(STUDY_AREA_CHUNKS)
            .document(self.id_)
            .set(as_dict, merge=True)
        )


@firestore.transactional
def _update_study_area_min_max_elevation(
    transaction: firestore.Transaction,
    study_area_ref: firestore.DocumentReference,
    min_: float,
    max_: float,
) -> None:
    """Updates the study area's elevation min & max in a transaction."""
    snapshot = study_area_ref.get(transaction=transaction)
    update = {}

    try:
        cur_min = snapshot.get("elevation_min")
        if min_ < cur_min:
            update["elevation_min"] = min_
    except KeyError:
        update["elevation_min"] = min_

    try:
        cur_max = snapshot.get("elevation_max")
        if max_ > cur_max:
            update["elevation_max"] = max_
    except KeyError:
        update["elevation_max"] = max_

    if update:
        transaction.update(study_area_ref, update)


@dataclasses.dataclass(slots=True)
class FloodScenarioConfig:
    """A configuration file describing rainfall patterns for a CityCAT simulation.

    Attributes:
        name: A human-readable name of the study area. Must be unique.
        gcs_uri: The GCS location of the configuration file.
        as_vector_gcs_uri: The GCS location of the configuration file formatted as a
                           numpy vector for input to ML.
        parent_config_name: The grouping containing this and other configurations which
                            are run together in batches of simulations.
        rainfall_duration: The number of rainfall timesteps in the configuration.
    """

    name: str
    gcs_uri: str
    as_vector_gcs_uri: str
    parent_config_name: str
    rainfall_duration: int

    def set(self, db: firestore.Client) -> None:
        """Creates or updates an existing entry for a CityCAT configuration file."""
        self.get_ref(db, self.name).set(
            {
                "parent_config_name": self.parent_config_name,
                "gcs_uri": self.gcs_uri,
                "as_vector_gcs_uri": self.as_vector_gcs_uri,
                "rainfall_duration": self.rainfall_duration,
            }
        )

    @classmethod
    def delete(cls, db: firestore.Client, name: str) -> None:
        """Deletes the CityCAT configuration entry for the given file."""
        cls.get_ref(db, name).delete()

    @classmethod
    def get(cls, db: firestore.Client, name: str) -> "FloodScenarioConfig":
        """Retrieve the flood config with the given name."""
        ref = cls.get_ref(db, name).get()
        if not ref.exists:
            raise ValueError(f'No such flood config "{name}"')

        return cls(name=name, **ref.to_dict())

    @staticmethod
    def get_ref(db: firestore.Client, name: str) -> firestore.DocumentReference:
        """Retrieve a Firestore reference to the flood config with the given name."""
        # Escape the name to avoid characters not allowed in IDs such as slashes.
        return db.collection(CITY_CAT_RAINFALL_CONFIG).document(
            urllib.parse.quote(name, safe=())
        )
