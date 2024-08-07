import dataclasses
import datetime
import enum
import hashlib
import itertools
import math
import random
from typing import Iterable
import urllib.parse

from google.cloud import firestore
import rasterio

from usl_lib.shared import geo_data


STUDY_AREAS = "study_areas"
STUDY_AREA_CHUNKS = "chunks"

CITY_CAT_RAINFALL_CONFIG = "city_cat_rainfall_configs"
WRF_HEAT_CONFIG = "wrf_heat_configs"

SIMULATIONS = "simulations"
SIMULATION_LABEL_CHUNKS = "label_chunks"


class StudyAreaState(enum.StrEnum):
    INIT = "init"
    CHUNKS_UPLOADED = "chunks-uploaded"
    FEATURE_MATRICES_CREATED = "feature-matrices-created"
    RESCALING_DONE = "rescaling-done"


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
        state: An optional state of the area reflecting phases of processing in GCP.
    """

    name: str
    col_count: int
    row_count: int
    x_ll_corner: float
    y_ll_corner: float
    cell_size: float
    crs: str | None = None
    state: StudyAreaState | None = None
    elevation_min: float | None = None
    elevation_max: float | None = None
    chunk_size: int | None = None
    chunk_x_count: int | None = None
    chunk_y_count: int | None = None

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
          name: The study area to retrieve.

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

    @staticmethod
    def update_chunk_info(
        db: firestore.Client, study_area_name: str, chunk_size: int
    ) -> None:
        """Sets chunk-related properties of study area if they weren't set before.

        Args:
          db: The firestore database client to use to make the update.
          study_area_name: The study area to update.
          chunk_size: Size of the chunk in cells. Will only be set chunk-related fields
                if they weren't set before.
        """
        study_area = StudyArea.get(db, study_area_name)
        if study_area.chunk_size is not None:
            if study_area.chunk_size != chunk_size:
                raise ValueError(
                    "Unexpected variation detected for chunk size:"
                    + f" {study_area.chunk_size} != {chunk_size}"
                )
            return

        # Calculate number of chunks over each of X and Y axes:
        x_count = math.ceil(study_area.col_count / chunk_size)
        y_count = math.ceil(study_area.row_count / chunk_size)

        db.collection(STUDY_AREAS).document(study_area_name).update(
            {
                "chunk_size": chunk_size,
                "chunk_x_count": x_count,
                "chunk_y_count": y_count,
            },
        )

    @staticmethod
    def delete_all_chunks(
        db: firestore.Client,
        study_area_name: str,
        page_size: int | None = None,
    ) -> None:
        """Deletes all the chunks from chunk sub-collection for a given study area.

        Args:
            db: The firestore database client to use.
            study_area_name: The study area to delete.
            page_size: Optional number of documents in the processing page.
        """
        docs = (
            db.collection(STUDY_AREAS)
            .document(study_area_name)
            .collection(STUDY_AREA_CHUNKS)
            .list_documents(page_size=page_size)
        )
        for doc in docs:
            doc.delete()

    @staticmethod
    def list_all_chunk_refs(
        db: firestore.Client,
        study_area_name: str,
    ) -> list[firestore.DocumentReference]:
        """Lists all the chunks from chunk sub-collection for a given study area.

        Args:
            db: The firestore database client to use.
            study_area_name: The study area to delete.
        """
        return [
            ref
            for ref in db.collection(STUDY_AREAS)
            .document(study_area_name)
            .collection(STUDY_AREA_CHUNKS)
            .list_documents()
        ]

    @staticmethod
    def update_state(db: firestore.Client, study_area_name: str, state: StudyAreaState):
        """Updates the state for a given study area.

        Args:
            db: The firestore database client to use.
            study_area_name: The study area to delete.
            state: New state that study area should be set to.
        """
        db.collection(STUDY_AREAS).document(study_area_name).update({"state": state})

    def _as_dict(self) -> dict:
        as_dict = {}
        for field in self.__dataclass_fields__:
            if field == "name":
                continue

            value = getattr(self, field)
            if value is not None:
                as_dict[field] = value

        return as_dict


class StudyAreaChunkState(enum.StrEnum):
    FEATURE_MATRIX_PROCESSING = "feature-matrix-processing"
    FEATURE_MATRIX_READY = "feature-matrix-ready"


@dataclasses.dataclass(slots=True)
class StudyAreaChunk:
    """A chunk of a data within a larger study area.

    Attributes:
        id_: An ID unique within the study are for the chunk.
        raw_path: GCS location containing the original, raw files describing the
            geography (e.g. tiff & shape files, NetCDF files.)
        feature_matrix_path: Optional GCS location of the derived feature matrix used
            for model training and prediction
        needs_scaling: If true, means that unscaled version of feature matrix should be
            scaled before it can be used (default is false).
        error: Any errors encountered while processing the chunk.
    """

    id_: str
    state: StudyAreaChunkState | None = None
    raw_path: str | None = None
    feature_matrix_path: str | None = None
    needs_scaling: bool | None = None
    error: str | None = None

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

        self.get_ref(db, study_area_name, self.id_).set(as_dict, merge=True)

    @classmethod
    def get(
        cls, db: firestore.Client, study_area_name: str, chunk_name: str
    ) -> "StudyAreaChunk":
        """Retrieve the chunk with the given name within the given study area."""
        ref = cls.get_ref(db, study_area_name, chunk_name).get()
        if not ref.exists:
            raise ValueError(f'No such chunk {chunk_name} within {study_area_name}"')

        return cls(id_=chunk_name, **ref.to_dict())

    @classmethod
    def from_ref(cls, ref: firestore.DocumentReference) -> "StudyAreaChunk":
        """Creates an instance of the chunk class based on retrieved reference."""
        return cls(id_=ref.id, **ref.get().to_dict())

    @staticmethod
    def get_ref(
        db: firestore.Client, study_area_name: str, chunk_name: str
    ) -> firestore.DocumentReference:
        """Retrieve a reference for the chunk with the given name and study area."""
        return (
            db.collection(STUDY_AREAS)
            .document(study_area_name)
            .collection(STUDY_AREA_CHUNKS)
            .document(chunk_name)
        )

    @classmethod
    def get_if_exists(
        cls, db: firestore.Client, study_area_name: str, chunk_name: str
    ) -> "StudyAreaChunk | None":
        """Retrieve the study area chunk with the given study area name and chunk name.

        Args:
          db: The firestore database client to use for the read.
          study_area_name: The study area to retrieve the chunk metadata from.
          chunk_name: Chunk name to retrieve the metadata for.

        Returns:
          A StudyAreaChunk object representing the database's contents if it exists, or
          None otherwise.
        """
        ref = cls.get_ref(db, study_area_name, chunk_name).get()
        return None if not ref.exists else cls(id_=chunk_name, **ref.to_dict())

    @classmethod
    def update_scaling_done(
        cls,
        db: firestore.Client,
        study_area_name: str,
        chunk_name: str,
        scaled_feature_matrix_path: str,
    ) -> None:
        """Updates chunk metadata fields related to scaling.

        Args:
            db: The firestore database client to use to make the update.
            study_area_name: The study area to update.
            chunk_name: Name of chunk to look up chunk metadata.
            scaled_feature_matrix_path: New GCS path pointing to scaled feature matrix.
        """
        cls.get_ref(db, study_area_name, chunk_name).update(
            {
                "state": StudyAreaChunkState.FEATURE_MATRIX_READY,
                "needs_scaling": False,
                "feature_matrix_path": scaled_feature_matrix_path,
                "error": firestore.DELETE_FIELD,
            },
        )


@dataclasses.dataclass(slots=True)
class StudyAreaSpatialChunk(StudyAreaChunk):
    """A sub-area chunk of a larger study area.

    Attributes:
        x_index: The x index of the chunk relative to other chunks in the study area.
        y_index: The y index of the chunk relative to other chunks in the study area.
        col_count: Optional number of columns in the raster of the chunk area.
        row_count: Optional number of rows in the raster of the chunk area.
        x_ll_corner: Optional X-coordinate of the raster's origin.
        y_ll_corner: Optional Y-coordinate of the raster's origin.
    """

    x_index: int | None = None
    y_index: int | None = None
    col_count: int | None = None
    row_count: int | None = None
    x_ll_corner: float | None = None
    y_ll_corner: float | None = None


@dataclasses.dataclass(slots=True)
class StudyAreaTemporalChunk(StudyAreaChunk):
    """A sub-area chunk of a larger study area.

    Attributes:
        time: The timestep represented by the data in this chunk.
    """

    time: datetime.datetime | None = None


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


@dataclasses.dataclass(slots=True)
class HeatScenarioConfig:
    """A configuration file describing heat scenario for a WRF simulation.

    Attributes:
        name: A human-readable name of the study area. Must be unique. Must be
                in the form: <study_area>_Heat (ex: NYC_Heat, NYC_PHX, etc)
        parent_config_name: The grouping containing this and other configurations which
                    are run together in batches of simulations.
        gcs_uri: The GCS location of the configuration file.
        simulation_year: (int) The year of the source data that WRF ingested
        simulation_months: The months of the source data (must be in acronym format)
                Ex: "JJA" - denotes: June July August
        percentile: (int) The percentile of the year's heat (determined by WRF team
            when selecting yearly source data) - ex: 99
    """

    name: str
    parent_config_name: str
    gcs_uri: str
    simulation_year: int
    simulation_months: str
    percentile: int

    def set(self, db: firestore.Client) -> None:
        """Creates or updates an existing entry for a WRF configuration file."""
        self.get_ref(db, self.name).set(
            {
                "parent_config_name": self.parent_config_name,
                "gcs_uri": self.gcs_uri,
                "simulation_year": self.simulation_year,
                "simulation_months": self.simulation_months,
                "percentile": self.percentile,
            }
        )

    @classmethod
    def delete(cls, db: firestore.Client, name: str) -> None:
        """Deletes the WRF configuration entry for the given file."""
        cls.get_ref(db, name).delete()

    @classmethod
    def get(cls, db: firestore.Client, name: str) -> "HeatScenarioConfig":
        """Retrieve the heat config with the given name."""
        ref = cls.get_ref(db, name).get()
        if not ref.exists:
            raise ValueError(f'No such heat config "{name}"')

        return cls(name=name, **ref.to_dict())

    @staticmethod
    def get_ref(db: firestore.Client, name: str) -> firestore.DocumentReference:
        """Retrieve a Firestore reference to the heat config with the given name."""
        # Escape the name to avoid characters not allowed in IDs such as slashes.
        return db.collection(WRF_HEAT_CONFIG).document(
            urllib.parse.quote(name, safe=())
        )


class SimulationType(enum.StrEnum):
    CITY_CAT = "CityCAT"
    WRF = "WRF"


@dataclasses.dataclass(slots=True)
class Simulation:
    """A simulation run against a study area given some configuration.

    Attributes:
        gcs_prefix_uri: GCS URI of the prefix under which simulation results are stored.
        simulation_type: The type of simulation (e.g. CityCAT, WRF.)
        study_area: The study area the simulation ran against.
        configuration: The configuration used to run the simulation.
    """

    gcs_prefix_uri: str
    simulation_type: SimulationType
    study_area: firestore.DocumentReference
    configuration: firestore.DocumentReference

    def set(self, db: firestore.Client) -> None:
        """Creates a simulation in the given DB."""
        if not self.study_area.get().exists:
            raise ValueError(f"No such study area exists: {self.study_area.id}")

        if not self.configuration.get().exists:
            raise ValueError(f"No such configuration exists: {self.configuration.id}")

        as_dict = {
            "gcs_prefix_uri": self.gcs_prefix_uri,
            "simulation_type": str(self.simulation_type),
            "study_area": self.study_area,
            "configuration": self.configuration,
        }
        self.get_ref(
            db,
            urllib.parse.unquote(self.study_area.id),
            urllib.parse.unquote(self.configuration.id),
        ).set(as_dict)

    @classmethod
    def get(
        cls, db: firestore.Client, study_area_name: str, config_path: str
    ) -> "Simulation":
        """Retrieve the simulation for the given study area and simulation config."""
        ref = cls.get_ref(db, study_area_name, config_path).get()
        if not ref.exists:
            raise ValueError(f"No such simulation for {study_area_name} {config_path}")
        return Simulation(**ref.to_dict())

    @staticmethod
    def get_ref(
        db: firestore.Client, study_area_name: str, config_path: str
    ) -> firestore.DocumentReference:
        """Retrieves a reference for the simulation for the given study are & config."""
        return db.collection(SIMULATIONS).document(
            urllib.parse.quote(f"{study_area_name}-{config_path}", safe=())
        )


@dataclasses.dataclass(slots=True)
class SimulationLabelChunk:
    """A chunk of a simulation result formatted as tensor for ML.

    Attributes:
        gcs_uri: The full GCS uri to the location of the chunk.
    """

    gcs_uri: str

    @classmethod
    def list_chunks(
        cls, db: firestore.Client, study_area_name: str, config_path: str
    ) -> Iterable["SimulationLabelChunk"]:
        """Retrieves all label chunks for the given simulation.

        Args:
          study_area_name: The name of the study area on which the simulation was run.
          config_path: The configuration used to run the simulation.

        Yields:
          Each label chunk produced by the given simulation.
        """
        ref = Simulation.get_ref(db, study_area_name, config_path).collection(
            SIMULATION_LABEL_CHUNKS
        )
        for chunk_ref in ref.list_documents():
            yield cls(**chunk_ref.get().to_dict())


class DatasetSplit(enum.StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclasses.dataclass(slots=True)
class SimulationLabelSpatialChunk(SimulationLabelChunk):
    """A sub-area chunk of a larger study area.

    Attributes:
        x_index: The x index of the chunk relative to other chunks in the simulation.
        y_index: The y index of the chunk relative to other chunks in the simulation.
        dataset: Which data set the label belongs to: train, val, test.
    """

    x_index: int
    y_index: int
    dataset: DatasetSplit

    def set(self, db: firestore.Client, study_area_name: str, config_path: str) -> None:
        """Adds the label chunk to the given simulation."""
        id_ = f"{self.x_index}_{self.y_index}"
        Simulation.get_ref(db, study_area_name, config_path).collection(
            SIMULATION_LABEL_CHUNKS
        ).document(id_).set(dataclasses.asdict(self))

    @staticmethod
    def dataset_split(
        study_area: "StudyArea", config_name: str, x_index: int, y_index: int
    ) -> DatasetSplit:
        """Determines whether the chunk is used for training, validation, or testing.

        Given x and y indices which uniquely identify a label chunk with a given study
        area for a given simulation configuation, determines if the label chunk should
        be considered as part of the training, validation, or test set.
        Returns a consistent set membership for the same study area, config and index
        pairs while also guaranteeing a 60 train / 20 val / 20 test split for all chunks
        within the study area.

        Args:
          study_area: The study area of the label chunk being considered.
          config_name: The simulation config name of the label chunk being considered.
          x_index: The x index of the chunk relative to other chunks in the simulation.
          y_index: The y index of the chunk relative to other chunks in the simulation.

        Returns:
          A DatasetSplit indicating which set the chunk belongs to: train, val, or test.
        """
        if study_area.chunk_x_count is None or study_area.chunk_y_count is None:
            raise ValueError(
                "chunk_x_count and chunk_y_count must be set for study area "
                f"{study_area.name}"
            )

        # Calculate the indices for all chunks inside the study area. These index pairs
        # uniquely identify the chunk.
        chunk_indices = [
            (x, y)
            for x, y in itertools.product(
                range(study_area.chunk_x_count), range(study_area.chunk_y_count)
            )
        ]

        # Determine if the given x & y index should represent a member
        # of the train, val, or test set. This should be a 60/20/20 split
        # on the chunks. We do this by building a random state, using
        # the study area name and simulation config name as a seed.
        # This means we will consistently generate the same set of
        # chunks as test and training set members for a given study
        # area and simulation config.  This allows us to generate
        # consistent sets no matter how many times any individual
        # processing function may run.
        hasher = hashlib.new("sha1", usedforsecurity=False)
        hasher.update(study_area.name.encode())
        hasher.update(config_name.encode())
        seed = hasher.hexdigest()
        rand_state = random.Random(seed)
        rand_state.shuffle(chunk_indices)

        train_split = int(0.6 * len(chunk_indices))
        val_split = int(0.8 * len(chunk_indices))
        if (x_index, y_index) in set(chunk_indices[:train_split]):
            return DatasetSplit.TRAIN
        if (x_index, y_index) in set(chunk_indices[train_split:val_split]):
            return DatasetSplit.VAL
        return DatasetSplit.TEST


@dataclasses.dataclass(slots=True)
class SimulationLabelTemporalChunk(SimulationLabelChunk):
    """A sub-area chunk of a larger study area.

    Attributes:
        time: The timestep represented by the data in this chunk.
    """

    time: datetime.datetime | None = None

    def set(self, db: firestore.Client, study_area_name: str, config_path: str) -> None:
        """Adds the label chunk to the given simulation."""
        Simulation.get_ref(db, study_area_name, config_path).collection(
            SIMULATION_LABEL_CHUNKS
        ).document(str(self.time)).set(dataclasses.asdict(self))

    # TODO: Implement is_in_test_set() for temporal chunks
