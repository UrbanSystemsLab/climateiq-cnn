import dataclasses
from typing import Optional

from google.cloud import firestore


STUDY_AREAS = "study_areas"
CHUNKS = "chunks"


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
    crs: str

    def create(self, db: firestore.Client) -> None:
        """Creates a new study area in the given DB."""
        as_dict = dataclasses.asdict(self)
        del as_dict["name"]

        db.collection(STUDY_AREAS).document(self.name).create(as_dict)


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
            .collection(CHUNKS)
            .document(self.id_)
            .set(as_dict, merge=True)
        )
