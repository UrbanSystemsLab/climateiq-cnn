"""Merges per-city results and publishes the combined outputs."""

import logging
import os

from google.cloud import storage

from postprocessing.gcs_io import _upload_to_gcs


def _merge_and_upload_outputs(
    bucket_name,
    output_prefix,
    output_dir,
    cities_in_run,
    output_bucket=None,
    city_output_dir=None,
):
    """Upload per-city files, then compose all_cities via GCS server-side compose.

    For each admin level:
    1. Upload per-city GeoJSON.
    2. Create a features-only blob (raw features, no FeatureCollection wrapper).
    3. Bootstrap features-only blobs for other cities from their per-city files.
    4. GCS compose: header + city1 + comma + city2 + … + footer → all_cities.

    The compose is server-side — no data streams through the function, so it
    works for files of any size in seconds.
    """
    import glob as _glob
    import re as _re

    output_files = _glob.glob(os.path.join(output_dir, "*.geojson"))
    if not output_files:
        logging.info("No output files to upload")
        return

    def _level_key(path):
        m = _re.search(r"admin_level_(\d+)_", os.path.basename(path))
        return int(m.group(1)) if m else 9999

    output_files = sorted(output_files, key=_level_key)

    client = storage.Client()
    bucket_obj = client.bucket(bucket_name)
    city_name = next(iter(cities_in_run))
    FEATURES_RAW_PREFIX = f"{output_prefix}features_raw"
    CHUNK = 256 * 1024 * 1024

    header_blob = bucket_obj.blob(f"{FEATURES_RAW_PREFIX}/_header.bin")
    header_blob.upload_from_string(b'{"type":"FeatureCollection","features":[')
    comma_blob = bucket_obj.blob(f"{FEATURES_RAW_PREFIX}/_comma.bin")
    comma_blob.upload_from_string(b",")
    footer_blob = bucket_obj.blob(f"{FEATURES_RAW_PREFIX}/_footer.bin")
    footer_blob.upload_from_string(b"]}")

    def _strip_fc_wrapper(data):
        """Return the content between [ and ] in a FeatureCollection."""
        start = data.find(b"[")
        end = data.rfind(b"]")
        if start == -1 or end == -1 or end <= start:
            return b""
        return data[start + 1 : end]

    def _ensure_features_raw(city, level_str, bucket_o):
        """Create features_raw blob from per-city file if it doesn't exist."""
        raw_path = f"{FEATURES_RAW_PREFIX}/{city}/level_{level_str}.bin"
        raw_blob = bucket_o.blob(raw_path)
        if raw_blob.exists():
            return raw_blob

        pc_path = (
            f"{output_prefix}by_city/{city}/admin_level_{level_str}_all_cities.geojson"
        )
        pc_blob = bucket_o.blob(pc_path)
        if not pc_blob.exists():
            return None

        pc_blob.reload()
        pc_size = pc_blob.size or 0

        if pc_size < 500 * 1024 * 1024:
            content = pc_blob.download_as_bytes()
            features_raw = _strip_fc_wrapper(content)
        else:
            raw_blob.chunk_size = CHUNK
            pc_blob.chunk_size = CHUNK
            with pc_blob.open("rb") as inp, raw_blob.open("wb") as out:
                header_done = False
                leftover = b""
                while True:
                    block = inp.read(CHUNK)
                    if not block:
                        if leftover:
                            end = leftover.rfind(b"]")
                            if end > 0:
                                out.write(leftover[:end])
                        break
                    if not header_done:
                        combined = leftover + block
                        bracket = combined.find(b"[")
                        if bracket == -1:
                            leftover = combined[-100:]
                            continue
                        header_done = True
                        leftover = combined[bracket + 1 :]
                        continue
                    if leftover:
                        out.write(leftover)
                    leftover = block
            return raw_blob

        if features_raw:
            raw_blob.upload_from_string(features_raw)
            return raw_blob
        return None

    for local_path in output_files:
        fname = os.path.basename(local_path)
        level_match = _re.search(r"admin_level_(\d+)_", fname)
        if not level_match:
            continue
        level_str = level_match.group(1)
        all_cities_path = f"{output_prefix}{fname}"

        try:
            per_city_dest = f"{output_prefix}by_city/{city_name}/{fname}"
            _upload_to_gcs(bucket_name, local_path, per_city_dest)
            logging.info("Uploaded per-city: %s", per_city_dest)

            raw_path = f"{FEATURES_RAW_PREFIX}/{city_name}/level_{level_str}.bin"
            raw_blob = bucket_obj.blob(raw_path)
            file_size = os.path.getsize(local_path)

            if file_size < 500 * 1024 * 1024:
                with open(local_path, "rb") as f:
                    content = f.read()
                features_raw = _strip_fc_wrapper(content)
                raw_blob.upload_from_string(features_raw)
                raw_size = len(features_raw)
                del content, features_raw
            else:
                raw_blob.chunk_size = CHUNK
                with open(local_path, "rb") as inp, raw_blob.open("wb") as out:
                    header_done = False
                    leftover = b""
                    while True:
                        block = inp.read(CHUNK)
                        if not block:
                            if leftover:
                                end = leftover.rfind(b"]")
                                if end > 0:
                                    out.write(leftover[:end])
                            break
                        if not header_done:
                            combined = leftover + block
                            bracket = combined.find(b"[")
                            if bracket == -1:
                                leftover = combined[-100:]
                                continue
                            header_done = True
                            leftover = combined[bracket + 1 :]
                            continue
                        if leftover:
                            out.write(leftover)
                        leftover = block
                raw_size = raw_blob.size or file_size

            if file_size > 500 * 1024 * 1024:
                if output_bucket and city_output_dir and int(level_str) > 100:
                    _pub_bkt = client.bucket(output_bucket)
                    _pub_dest = f"{city_output_dir}/{fname}"
                    _pub_bkt.blob(_pub_dest).upload_from_filename(local_path)
                    logging.info(
                        "Published H3 (pre-delete): gs://%s/%s (%.0f MB)",
                        output_bucket,
                        _pub_dest,
                        file_size / 1e6,
                    )
                os.remove(local_path)
                logging.info("Freed %.0f MB tmpfs: %s", file_size / 1e6, fname)

            logging.info(
                "Uploaded features_raw for %s level %s (%d bytes)",
                city_name,
                level_str,
                raw_size,
            )

            by_city_prefix = f"{output_prefix}by_city/"
            city_dirs = set()
            blobs_iter = client.list_blobs(
                bucket_name, prefix=by_city_prefix, delimiter="/"
            )
            for _ in blobs_iter:
                pass
            for prefix_str in blobs_iter.prefixes:
                c = prefix_str.replace(by_city_prefix, "").rstrip("/")
                if c:
                    city_dirs.add(c)

            logging.info(
                "Found %d cities for level %s compose: %s",
                len(city_dirs),
                level_str,
                sorted(city_dirs),
            )

            city_raw_blobs = []
            for c in sorted(city_dirs):
                if c == city_name:
                    city_raw_blobs.append((c, raw_blob))
                    continue
                rb = _ensure_features_raw(c, level_str, bucket_obj)
                if rb is not None:
                    city_raw_blobs.append((c, rb))

            if not city_raw_blobs:
                logging.warning("No features_raw blobs for level %s", level_str)
                continue

            sources = [header_blob]
            for i, (c, rb) in enumerate(city_raw_blobs):
                if i > 0:
                    sources.append(comma_blob)
                sources.append(rb)
            sources.append(footer_blob)

            logging.info(
                "Composing %s from %d sources (%d cities)",
                fname,
                len(sources),
                len(city_raw_blobs),
            )

            dest_blob = bucket_obj.blob(all_cities_path)
            dest_blob.compose(sources)

            logging.info("  Composed %s successfully", fname)

        except Exception as e:
            logging.error("Failed to compose %s: %s", fname, e)


def _combine_geojson_files(file_paths):
    """Merge multiple GeoJSON FeatureCollections into one."""
    import json as _json

    all_features = []
    for path in file_paths:
        try:
            with open(path) as f:
                fc = _json.load(f)
            all_features.extend(fc.get("features", []))
        except Exception as e:
            logging.error("Failed to read %s for combining: %s", path, e)
    return {"type": "FeatureCollection", "features": all_features}


def _publish_combined_outputs(
    output_dir,
    city_folder,
    output_bucket,
    pipeline_bucket,
    state_fips=None,
):
    """Publish combined GeoJSON files to the output bucket.

    Per-city:  {city_folder}/h3_{city_folder}.geojson
    National:  USA_Combined/admin.geojson
    """
    import glob as _glob
    import re as _re

    output_files = _glob.glob(os.path.join(output_dir, "*.geojson"))
    if not output_files:
        return

    h3_files = []
    admin_files = []
    for local_path in sorted(output_files):
        fname = os.path.basename(local_path)
        m = _re.search(r"admin_level_(\d+)_", fname)
        if not m:
            continue
        level = int(m.group(1))
        if level > 100:
            h3_files.append(local_path)
        else:
            admin_files.append(local_path)

    client = storage.Client()
    out_bucket = client.bucket(output_bucket)

    for local_path in h3_files:
        fname = os.path.basename(local_path)
        dest = f"{city_folder}/{fname}"
        out_bucket.blob(dest).upload_from_filename(local_path)
        file_mb = os.path.getsize(local_path) / (1024 * 1024)
        logging.info(
            "Published H3: gs://%s/%s (%.1f MB)",
            output_bucket,
            dest,
            file_mb,
        )

    _create_combined_h3(h3_files, city_folder, output_bucket, client)

    _publish_national_admin(admin_files, output_bucket)


def _copy_features_raw(readable, writer, prepend_comma, chunk_size=4 * 1024 * 1024):
    """Copy feature JSON from a GeoJSON FeatureCollection to *writer*.

    Streams the text between the ``[`` and ``]`` of the features array
    without parsing individual features, keeping memory constant.
    Returns True if any content was written.
    """
    buf = ""
    while True:
        chunk = readable.read(chunk_size)
        if not chunk:
            readable.close()
            return False
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        buf += chunk
        bracket = buf.find("[")
        if bracket != -1:
            pending = buf[bracket + 1 :]
            break

    wrote = False
    while True:
        chunk = readable.read(chunk_size)
        is_last = not chunk

        if is_last:
            close = pending.rfind("]")
            if close >= 0:
                pending = pending[:close]

        text = pending

        if not wrote:
            text = text.lstrip()
            if text.startswith(","):
                text = text[1:].lstrip()

        if is_last:
            text = text.rstrip()
            if text.endswith(","):
                text = text[:-1]

        if text and not text.isspace():
            if not wrote and prepend_comma:
                writer.write(",\n")
            writer.write(text)
            wrote = True

        if is_last:
            break

        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        pending = chunk

    readable.close()
    return wrote


def _create_combined_h3(h3_local, city_folder, output_bucket, client):
    """Merge per-resolution H3 files into {city_folder}/h3_{city_folder}.geojson.

    The combined file triggers the Mapbox tile-publish workflow via Eventarc.
    Streams directly to GCS so the full combined file never sits in RAM.
    """
    import re as _re

    out_bkt = client.bucket(output_bucket)
    dest = f"{city_folder}/h3_{city_folder}.geojson"

    local_names = {os.path.basename(p) for p in h3_local}
    gcs_only = []
    for blob in out_bkt.list_blobs(prefix=f"{city_folder}/admin_level_1"):
        bname = os.path.basename(blob.name)
        if bname.endswith(".geojson") and bname not in local_names:
            gcs_only.append(blob)

    if not h3_local and not gcs_only:
        return

    def _level_key(name):
        m = _re.search(r"admin_level_(\d+)", name)
        return int(m.group(1)) if m else 0

    sources = []
    for p in sorted(h3_local, key=lambda x: _level_key(os.path.basename(x))):
        sources.append(("local", p))
    for b in sorted(gcs_only, key=lambda x: _level_key(os.path.basename(x.name))):
        sources.append(("gcs", b))

    combined_blob = out_bkt.blob(dest)
    combined_blob.chunk_size = 10 * 1024 * 1024
    need_sep = False

    with combined_blob.open("w") as out:
        out.write('{"type":"FeatureCollection","features":[\n')

        for src_type, src in sources:
            if src_type == "local":
                fh = open(src)
                label = os.path.basename(src)
            else:
                fh = src.open("r")
                label = os.path.basename(src.name)

            if _copy_features_raw(fh, out, need_sep):
                need_sep = True
            logging.info("Combined H3: streamed %s", label)

        out.write("\n]}")

    combined_blob.reload()
    sz = (combined_blob.size or 0) / (1024 * 1024)
    logging.info(
        "Published combined H3: gs://%s/%s (%.1f MB)",
        output_bucket,
        dest,
        sz,
    )

    for src_type, src in sources:
        if src_type == "local":
            bname = os.path.basename(src)
        else:
            bname = os.path.basename(src.name)
        out_bkt.blob(f"{city_folder}/{bname}").delete()
    logging.info(
        "Cleaned up %d per-resolution H3 files",
        len(sources),
    )


def _publish_national_admin(local_admin_files, output_bucket):
    """Merge this city's admin files into USA_Combined/admin.geojson.

    Reads existing admin.geojson from GCS (other cities), adds this city's
    admin features, and re-uploads the combined file.
    """
    import json as _json

    if not local_admin_files:
        return

    client = storage.Client()
    out_bucket = client.bucket(output_bucket)

    all_features = []
    existing_blob = out_bucket.blob("USA_Combined/admin.geojson")
    if existing_blob.exists():
        try:
            existing = _json.loads(existing_blob.download_as_text())
            all_features.extend(existing.get("features", []))
            logging.info("Read existing national admin: %d features", len(all_features))
        except Exception as e:
            logging.error("Failed to read existing admin.geojson: %s", e)

    new_features = _combine_geojson_files(local_admin_files).get("features", [])

    new_cities = {
        f.get("properties", {}).get("city") for f in new_features if f.get("properties")
    }
    if new_cities:
        all_features = [
            f
            for f in all_features
            if f.get("properties", {}).get("city") not in new_cities
        ]

    all_features.extend(new_features)

    if not all_features:
        logging.info("No admin features for national merge")
        return

    combined = {"type": "FeatureCollection", "features": all_features}
    existing_blob.upload_from_string(
        _json.dumps(combined), content_type="application/json"
    )
    logging.info(
        "Published national admin: gs://%s/USA_Combined/admin.geojson (%d features)",
        output_bucket,
        len(all_features),
    )
