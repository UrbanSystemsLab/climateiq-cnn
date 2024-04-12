from unittest import mock

from usl_lib.storage import metastore


def test_study_area_create():
    mock_db = mock.MagicMock()
    study_area = metastore.StudyArea(
        name="name",
        col_count=2,
        row_count=3,
        x_ll_corner=1.0,
        y_ll_corner=2.0,
        cell_size=3.0,
        crs="over there",
    )
    study_area.create(mock_db)
    mock_db.assert_has_calls(
        [
            mock.call.collection("study_areas"),
            mock.call.collection().document("name"),
            mock.call.collection()
            .document()
            .create(
                {
                    "col_count": 2,
                    "row_count": 3,
                    "x_ll_corner": 1.0,
                    "y_ll_corner": 2.0,
                    "cell_size": 3.0,
                    "crs": "over there",
                }
            ),
        ]
    )


def test_study_area_chunk_merge():
    mock_db = mock.MagicMock()
    chunk = metastore.StudyAreaChunk(
        id_="id",
        archive_path="archive",
        feature_matrix_path="matrix",
    )
    chunk.merge(mock_db, "study_area_name")
    mock_db.assert_has_calls(
        [
            mock.call.collection("study_areas"),
            mock.call.collection().document("study_area_name"),
            mock.call.collection().document().collection("chunks"),
            mock.call.collection().document().collection().document("id"),
            mock.call.collection()
            .document()
            .collection()
            .document()
            .set(
                {
                    "archive_path": "archive",
                    "feature_matrix_path": "matrix",
                },
                merge=True,
            ),
        ]
    )
