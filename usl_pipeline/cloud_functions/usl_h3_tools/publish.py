"""Publishes tilesets built from the H3 tiles bucket to tile servers.

<ISO3>_<STATE>_<City>/h3_<ISO3>_<STATE>_<City>.geojson is that city's tileset,
and USA_Combined/admin.geojson is one national admin tileset.
"""

import abc
import dataclasses
import hashlib
import io
import logging
import re
import uuid
from typing import Any, BinaryIO, Mapping

from google.cloud import storage
import requests

from usl_lib.shared import area_names

TILE_SOURCES_PREFIX = "tile_sources"
CITY_H3_SOURCE = "{area}/h3_{area}.geojson"
NATIONAL_ADMIN_SOURCE = "USA_Combined/admin.geojson"
CITY_H3_TILESET_PREFIX = "climateiq-h3-"
NATIONAL_ADMIN_TILESET = "climateiq-admin-national"

_CITY_H3_SOURCE_RE = re.compile(r"^(?P<area>[^/]+)/h3_(?P=area)\.geojson$")

TILE_TARGETS_ENV = "TILE_TARGETS"
MAPBOX_USERNAME_ENV = "MAPBOX_USERNAME"
MAPBOX_ACCESS_TOKEN_ENV = "MAPBOX_ACCESS_TOKEN"

MAPBOX_API_URL = "https://api.mapbox.com"
MAPBOX_MAX_ID_LENGTH = 32
MAPBOX_MAX_NAME_LENGTH = 64
MAPBOX_MAX_SOURCE_BYTES = 20 * 10**9
# (connect, read) seconds for each socket operation.
MAPBOX_REQUEST_TIMEOUT = (30, 600)

_MAPBOX_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAPBOX_ERROR_BODY_LIMIT = 500
_MAPBOX_LAYER_SIZE_KB = 2500


@dataclasses.dataclass(frozen=True)
class TileLayer:
    layer_name: str
    # admin_level of the layer's features: admin boundaries, or 100 + H3 resolution.
    admin_level: int
    minzoom: int
    maxzoom: int


CITY_H3_LAYERS = (
    TileLayer("admin-boundary-hazard-grid-h3-09", 109, 6, 13),
    TileLayer("admin-boundary-hazard-grid-h3-10", 110, 12, 13),
    TileLayer("admin-boundary-hazard-grid-h3-11", 111, 13, 14),
    TileLayer("admin-boundary-hazard-grid-h3-12", 112, 15, 15),
)

NATIONAL_ADMIN_LAYERS = (
    TileLayer("admin-boundary-hazard-6", 6, 0, 11),
    TileLayer("admin-boundary-hazard-8", 8, 0, 11),
    TileLayer("admin-boundary-hazard-10", 10, 0, 13),
)


@dataclasses.dataclass(frozen=True)
class Tileset:
    name: str  # Tileset name, without the owning account.
    description: str
    bucket: str
    source_blob: str  # GeoJSON FeatureCollection holding every layer's features.
    layers: tuple[TileLayer, ...]

    @property
    def staged_blob(self) -> str:
        """Line-delimited GeoJSON copy of source_blob."""
        return f"{TILE_SOURCES_PREFIX}/{self.name}/source.geojsonl"

    @property
    def status_blob(self) -> str:
        """Record of the latest publish."""
        return f"{TILE_SOURCES_PREFIX}/{self.name}/publish.json"


def tileset_for_object(bucket: str, object_name: str) -> Tileset | None:
    """Returns the tileset built from an object, or None if it feeds none."""
    if object_name == NATIONAL_ADMIN_SOURCE:
        return Tileset(
            name=NATIONAL_ADMIN_TILESET,
            description="ClimateIQ hazard by county, place and tract",
            bucket=bucket,
            source_blob=object_name,
            layers=NATIONAL_ADMIN_LAYERS,
        )

    match = _CITY_H3_SOURCE_RE.match(object_name)
    if match is None or not area_names.is_canonical_area_name(match["area"]):
        return None
    area_name = match["area"]
    return Tileset(
        name=f"{CITY_H3_TILESET_PREFIX}{area_name}",
        description=f"ClimateIQ H3 hazard grid for {area_name}",
        bucket=bucket,
        source_blob=object_name,
        layers=CITY_H3_LAYERS,
    )


@dataclasses.dataclass
class PublishResult:
    target: str
    tileset_id: str
    job_id: str | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)


class Tileserver(abc.ABC):
    """A tile server that hosts tilesets.

    The tileset's line-delimited GeoJSON is staged in GCS before upload_source
    is called, and publish runs after upload_source.
    """

    # Value that selects this target in TILE_TARGETS.
    name: str

    @classmethod
    @abc.abstractmethod
    def from_env(
        cls, env: Mapping[str, str], storage_client: storage.Client
    ) -> "Tileserver":
        """Creates the target from environment variables.

        Raises:
            ValueError: A setting the target needs is missing.
        """

    @abc.abstractmethod
    def upload_source(self, tileset: Tileset) -> None:
        """Sends the staged source to the tile server."""

    @abc.abstractmethod
    def publish(self, tileset: Tileset) -> PublishResult:
        """Builds and serves the tileset from its uploaded source."""


class MapboxError(RuntimeError):
    """A Mapbox API call failed. The message never includes the access token."""

    def __init__(self, method: str, path: str, status: int | None, detail: str):
        """Creates the error.

        Args:
            method: HTTP method.
            path: API path, without the query string.
            status: HTTP status, or None if no response was received.
            detail: Response body or exception type.
        """
        status_text = f"HTTP {status}" if status is not None else "no response"
        super().__init__(f"Mapbox {method} {path} failed ({status_text}): {detail}")
        self.status = status


def mapbox_id(name: str) -> str:
    """Returns the MTS tileset and tileset source id for a tileset name.

    Names over the 32 character limit keep a prefix plus a hash of the full name.

    Raises:
        ValueError: The name has characters other than letters, digits, '-'
            and '_'.
    """
    if not _MAPBOX_ID_RE.match(name):
        raise ValueError(
            f"{name!r} cannot be a Mapbox id: ids may only hold letters, digits, "
            "'-' and '_'."
        )
    if len(name) <= MAPBOX_MAX_ID_LENGTH:
        return name

    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{name[: MAPBOX_MAX_ID_LENGTH - len(digest) - 1]}-{digest}"


class _MultipartBody:
    """Streams a single-file multipart/form-data body with a known length."""

    def __init__(self, reader: BinaryIO, size: int, filename: str, boundary: str):
        preamble = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        epilogue = f"\r\n--{boundary}--\r\n".encode()
        self._parts: list[BinaryIO] = [
            io.BytesIO(preamble),
            reader,
            io.BytesIO(epilogue),
        ]
        self._length = len(preamble) + size + len(epilogue)

    def __len__(self) -> int:
        return self._length

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return b"".join(part.read() for part in self._parts)
        while self._parts:
            data = self._parts[0].read(size)
            if data:
                return data
            self._parts.pop(0)
        return b""


class Mapbox(Tileserver):
    """Publishes each tileset to the Mapbox Tiling Service (MTS).

    The tileset is <username>.<mapbox_id(name)>. The staged source is uploaded
    as one tileset source with the same id, and the recipe filters it into one
    layer per TileLayer.
    """

    name = "mapbox"

    def __init__(
        self,
        username: str,
        access_token: str,
        storage_client: storage.Client,
        session: requests.Session | None = None,
        api_url: str = MAPBOX_API_URL,
    ) -> None:
        """Creates the target.

        Args:
            username: Mapbox account that owns the tilesets.
            access_token: Secret token with tilesets:write, tilesets:read and
                tilesets:list scopes.
            storage_client: Client used to read the staged source.
            session: HTTP session, for tests.
            api_url: Mapbox API base URL.
        """
        self._username = username
        self._access_token = access_token
        self._storage_client = storage_client
        self._session = session if session is not None else requests.Session()
        self._api_url = api_url

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], storage_client: storage.Client
    ) -> "Mapbox":
        """Creates the target from MAPBOX_USERNAME and MAPBOX_ACCESS_TOKEN.

        Raises:
            ValueError: One of them is unset.
        """
        return cls(
            username=_required(env, MAPBOX_USERNAME_ENV),
            access_token=_required(env, MAPBOX_ACCESS_TOKEN_ENV),
            storage_client=storage_client,
        )

    def tileset_id(self, tileset: Tileset) -> str:
        """Returns <username>.<mapbox_id(tileset.name)>."""
        return f"{self._username}.{mapbox_id(tileset.name)}"

    def recipe(self, tileset: Tileset) -> dict[str, Any]:
        """Returns the MTS recipe for the tileset."""
        source = f"mapbox://tileset-source/{self._username}/{mapbox_id(tileset.name)}"
        return {
            "version": 1,
            "layers": {
                layer.layer_name: {
                    "source": source,
                    "minzoom": layer.minzoom,
                    "maxzoom": layer.maxzoom,
                    "tiles": {"layer_size": _MAPBOX_LAYER_SIZE_KB},
                    "features": {
                        "filter": ["==", ["get", "admin_level"], layer.admin_level],
                        "simplification": {"outward_only": True, "distance": 1},
                    },
                }
                for layer in tileset.layers
            },
        }

    def upload_source(self, tileset: Tileset) -> None:
        """Replaces the tileset source with the staged GeoJSON.

        Raises:
            ValueError: The staged source is over the MTS file size limit.
        """
        source = mapbox_id(tileset.name)
        blob = self._storage_client.bucket(tileset.bucket).blob(tileset.staged_blob)
        blob.reload()
        if blob.size > MAPBOX_MAX_SOURCE_BYTES:
            raise ValueError(
                f"gs://{tileset.bucket}/{tileset.staged_blob} is {blob.size} bytes; "
                f"Mapbox accepts at most {MAPBOX_MAX_SOURCE_BYTES} per source file."
            )
        boundary = uuid.uuid4().hex

        logging.info(
            "Uploading gs://%s/%s (%d bytes) to Mapbox source %s/%s",
            tileset.bucket,
            tileset.staged_blob,
            blob.size,
            self._username,
            source,
        )
        with blob.open("rb", if_generation_match=blob.generation) as reader:
            body = _MultipartBody(reader, blob.size, f"{source}.geojsonl", boundary)
            self._request(
                "PUT",
                f"/tilesets/v1/sources/{self._username}/{source}",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )

    def publish(self, tileset: Tileset) -> PublishResult:
        """Creates or updates the tileset's recipe and starts a publish job."""
        if not tileset.layers:
            raise ValueError(f"{tileset.name} has no layers to publish.")

        tileset_id = self.tileset_id(tileset)
        recipe = self.recipe(tileset)

        validation = self._request(
            "PUT", "/tilesets/v1/validateRecipe", json=recipe
        ).json()
        if not validation.get("valid"):
            raise MapboxError(
                "PUT",
                "/tilesets/v1/validateRecipe",
                200,
                f"recipe for {tileset_id} is invalid: {validation.get('errors')}",
            )

        existing = self._request(
            "GET", f"/tilesets/v1/{tileset_id}/recipe", allowed_statuses=(404,)
        )
        if existing.status_code == 404:
            logging.info("Creating Mapbox tileset %s", tileset_id)
            self._request(
                "POST",
                f"/tilesets/v1/{tileset_id}",
                json={
                    "recipe": recipe,
                    "name": tileset.name[:MAPBOX_MAX_NAME_LENGTH],
                    "description": tileset.description,
                },
            )
        else:
            logging.info("Updating recipe of Mapbox tileset %s", tileset_id)
            self._request("PATCH", f"/tilesets/v1/{tileset_id}/recipe", json=recipe)

        job_id = self._request("POST", f"/tilesets/v1/{tileset_id}/publish").json()[
            "jobId"
        ]
        logging.info("Started Mapbox publish job %s for %s", job_id, tileset_id)

        return PublishResult(
            target=self.name,
            tileset_id=tileset_id,
            job_id=job_id,
            details={
                "job_url": f"{self._api_url}/tilesets/v1/{tileset_id}/jobs/{job_id}",
                "source": mapbox_id(tileset.name),
                "layers": [layer.layer_name for layer in tileset.layers],
            },
        )

    def _request(
        self,
        method: str,
        path: str,
        allowed_statuses: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> requests.Response:
        try:
            response = self._session.request(
                method,
                self._api_url + path,
                params={"access_token": self._access_token},
                timeout=MAPBOX_REQUEST_TIMEOUT,
                **kwargs,
            )
        except requests.RequestException as e:
            raise MapboxError(method, path, None, type(e).__name__) from None

        if response.ok or response.status_code in allowed_statuses:
            return response
        raise MapboxError(
            method,
            path,
            response.status_code,
            response.text[:_MAPBOX_ERROR_BODY_LIMIT],
        )


TARGETS: tuple[type[Tileserver], ...] = (Mapbox,)


def make_targets(
    env: Mapping[str, str], storage_client: storage.Client
) -> list[Tileserver]:
    """Returns the targets named in TILE_TARGETS, comma separated.

    An empty or unset TILE_TARGETS disables publishing.

    Raises:
        ValueError: An unknown target name, or a missing setting it needs.
    """
    by_name = {target.name: target for target in TARGETS}
    targets: list[Tileserver] = []
    for name in env.get(TILE_TARGETS_ENV, "").split(","):
        name = name.strip()
        if not name:
            continue
        target = by_name.get(name)
        if target is None:
            raise ValueError(
                f"Unknown tile target {name!r} in {TILE_TARGETS_ENV}. "
                f"Expected one of {sorted(by_name)}."
            )
        targets.append(target.from_env(env, storage_client))
    return targets


def _required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} must be set to publish tiles.")
    return value
