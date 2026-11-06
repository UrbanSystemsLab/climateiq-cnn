"""Syncs a city's results to MongoDB."""

import logging
import os

from google.cloud import storage

MONGO_ADMIN_COLLECTION = "admin_level"


MONGO_H3_COLLECTION = "h3"


def _get_mongo_uri():
    uri = os.environ.get("MONGODB_URI")
    if uri:
        return uri
    secret_name = os.environ.get(
        "MONGODB_SECRET", "projects/climateiq-test/secrets/mongodb-uri/versions/latest"
    )
    try:
        from google.cloud import secretmanager

        sm = secretmanager.SecretManagerServiceClient()
        resp = sm.access_secret_version(request={"name": secret_name})
        return resp.payload.data.decode("UTF-8")
    except Exception as e:
        logging.warning("Could not read MongoDB secret: %s", e)
        return None


def _sync_to_mongodb(output_dir, city_name, output_bucket=None, city_output_dir=None):
    """Push combined GeoJSON to MongoDB Atlas.

    Reads the combined h3_{city_folder}.geojson → h3 collection
    Reads all admin-level files (4/6/8/10) → admin_level collection
    Upserts by (city, admin_level, boundary_id/cell_code).
    """
    import json as _json
    import glob as _glob
    import re as _re

    mongo_uri = _get_mongo_uri()
    if not mongo_uri:
        logging.info("MongoDB URI not configured — skipping sync")
        return

    try:
        from pymongo import MongoClient
    except ImportError:
        logging.warning("pymongo not installed — skipping MongoDB sync")
        return

    try:
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=30000,
            socketTimeoutMS=300000,
            connectTimeoutMS=30000,
            retryWrites=True,
        )
        db = client[os.environ.get("MONGODB_DATABASE", "climateiq_data")]
    except Exception as e:
        logging.error("MongoDB connection failed: %s", e)
        return

    h3_files = sorted(
        f
        for f in os.listdir(output_dir)
        if _re.search(r"admin_level_1\d{2}_", f) and f.endswith(".geojson")
    )

    gcs_only_blobs = []
    if output_bucket and city_output_dir:
        gcs_client = storage.Client()
        out_bkt = gcs_client.bucket(output_bucket)
        local_set = set(h3_files)
        for blob in out_bkt.list_blobs(prefix=f"{city_output_dir}/"):
            bname = os.path.basename(blob.name)
            if (
                _re.search(r"admin_level_1\d{2}_", bname)
                and bname.endswith(".geojson")
                and bname not in local_set
            ):
                gcs_only_blobs.append(blob)
                logging.info("H3 file %s missing locally — will stream from GCS", bname)

    if h3_files or gcs_only_blobs:
        removed = _mongo_batched_delete(
            db[MONGO_H3_COLLECTION], {"properties.city": city_name}
        )
        logging.info(
            "MongoDB %s: removed %d old %s docs",
            db[MONGO_H3_COLLECTION].name,
            removed,
            city_name,
        )
        for h3_fname in h3_files:
            h3_path = os.path.join(output_dir, h3_fname)
            file_size = os.path.getsize(h3_path)
            if file_size > 500 * 1024 * 1024:
                with open(h3_path) as fh:
                    _mongo_stream_insert_geojson(
                        db[MONGO_H3_COLLECTION], fh, city_name, h3_fname
                    )
            else:
                _mongo_insert_from_geojson(
                    db[MONGO_H3_COLLECTION], h3_path, city_name, _json
                )
        for blob in gcs_only_blobs:
            with blob.open("r") as fh:
                _mongo_stream_insert_geojson(
                    db[MONGO_H3_COLLECTION],
                    fh,
                    city_name,
                    os.path.basename(blob.name),
                )
    else:
        logging.info("No H3 resolution files found — skipping H3 MongoDB sync")

    admin_features = []
    for local_path in sorted(_glob.glob(os.path.join(output_dir, "*.geojson"))):
        fname = os.path.basename(local_path)
        m = _re.search(r"admin_level_(\d+)_", fname)
        if not m:
            continue
        level = int(m.group(1))
        if level > 100:
            continue
        try:
            with open(local_path) as f:
                fc = _json.load(f)
            feats = fc.get("features", [])
            if level == 8:
                feats = [f for f in feats if f.get("properties", {}).get("is_city")]
            admin_features.extend(feats)
        except Exception as e:
            logging.error("Failed to read %s: %s", fname, e)

    if admin_features:
        docs = [
            {
                "type": "Feature",
                "properties": f.get("properties", {}),
                "geometry": f.get("geometry"),
            }
            for f in admin_features
        ]
        _mongo_batched_delete(
            db[MONGO_ADMIN_COLLECTION], {"properties.city": city_name}
        )
        _mongo_batch_insert(db[MONGO_ADMIN_COLLECTION], docs, city_name)

    try:
        client.close()
    except Exception:
        pass


def _mongo_batched_delete(collection, filter_dict, batch_size=50000):
    """Delete matching docs in batches to avoid socket timeouts on large cities."""
    total = 0
    while True:
        ids = [
            doc["_id"]
            for doc in collection.find(filter_dict, {"_id": 1}).limit(batch_size)
        ]
        if not ids:
            break
        collection.delete_many({"_id": {"$in": ids}})
        total += len(ids)
        if total % 100000 < batch_size:
            logging.info(
                "MongoDB %s: deleted %d docs so far",
                collection.name,
                total,
            )
    return total


def _mongo_insert_from_geojson(collection, file_path, city_name, _json):
    """Insert docs from a single GeoJSON file (no delete)."""
    try:
        with open(file_path) as f:
            fc = _json.load(f)
    except Exception as e:
        logging.error("Failed to read %s: %s", file_path, e)
        return
    features = fc.get("features", [])
    if not features:
        return
    docs = [
        {
            "type": "Feature",
            "properties": feat.get("properties", {}),
            "geometry": feat.get("geometry"),
        }
        for feat in features
    ]
    del fc, features
    _mongo_batch_insert(collection, docs, city_name)


def _mongo_stream_insert_geojson(collection, readable, city_name, source_name):
    """Stream-insert GeoJSON features without full load."""
    import json as _json

    BATCH = 5000
    READ_SIZE = 2 * 1024 * 1024
    decoder = _json.JSONDecoder()
    docs = []
    total = 0

    buf = ""
    while True:
        chunk = readable.read(READ_SIZE)
        if not chunk:
            return
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        buf += chunk
        idx = buf.find("[")
        if idx != -1:
            buf = buf[idx + 1 :]
            break

    WS = frozenset(" ,\n\r\t")
    eof = False
    while True:
        pos = 0
        buf_len = len(buf)
        while pos < buf_len:
            while pos < buf_len and buf[pos] in WS:
                pos += 1
            if pos >= buf_len:
                break
            if buf[pos] == "]":
                eof = True
                break
            try:
                feat, end_idx = decoder.raw_decode(buf, pos)
                pos = end_idx
                docs.append(
                    {
                        "type": "Feature",
                        "properties": feat.get("properties", {}),
                        "geometry": feat.get("geometry"),
                    }
                )
                del feat
                if len(docs) >= BATCH:
                    _mongo_batch_insert(collection, docs, city_name)
                    total += len(docs)
                    docs = []
            except _json.JSONDecodeError:
                break

        if eof:
            break

        buf = buf[pos:]
        chunk = readable.read(READ_SIZE)
        if not chunk:
            break
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        buf += chunk

    if docs:
        _mongo_batch_insert(collection, docs, city_name)
        total += len(docs)

    logging.info(
        "MongoDB: streamed %d features from %s",
        total,
        source_name,
    )


def _mongo_replace_city_file(collection, file_path, city_name, _json):
    """Replace a city's docs from a GeoJSON file."""
    try:
        with open(file_path) as f:
            fc = _json.load(f)
    except Exception as e:
        logging.error("Failed to read %s: %s", file_path, e)
        return
    features = fc.get("features", [])
    if not features:
        return
    docs = [
        {
            "type": "Feature",
            "properties": feat.get("properties", {}),
            "geometry": feat.get("geometry"),
        }
        for feat in features
    ]
    del fc, features
    removed = _mongo_batched_delete(collection, {"properties.city": city_name})
    logging.info(
        "MongoDB %s: removed %d old %s docs",
        collection.name,
        removed,
        city_name,
    )
    _mongo_batch_insert(collection, docs, city_name)


def _mongo_batch_insert(collection, docs, city_name, batch_size=25000):
    """Insert docs in batches with retry."""
    import time as _time

    total = len(docs)
    done = 0
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        for attempt in range(5):
            try:
                collection.insert_many(batch, ordered=False)
                break
            except Exception as e:
                if attempt >= 4:
                    logging.error(
                        "MongoDB insert failed at %d/%d: %s",
                        done,
                        total,
                        e,
                    )
                    raise
                _time.sleep(min(30, 5 * (attempt + 1)))
        done += len(batch)
        if done % 50000 < batch_size:
            logging.info(
                "MongoDB %s %s: %d/%d",
                collection.name,
                city_name,
                done,
                total,
            )
    logging.info("MongoDB %s %s: DONE %d docs", collection.name, city_name, total)
