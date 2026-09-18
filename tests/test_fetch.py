"""segment.fetch / make_session: getting a photo from the open-data bucket onto
disk and into memory, and what happens when that goes wrong.

Tests marked `network` make real requests to inaturalist-open-data.s3.amazonaws.com
(anonymous, a handful of small GETs). The rest use a fake session.
"""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
import requests
from PIL import Image

from conftest import POPPY, POPPY_TAXON


def _jpeg_bytes(size=(40, 30)):
    buf = BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return self.response


ROW = {"photo_id": 1, "photo_url": "https://example.invalid/photos/1/original.jpg"}


# -- offline -------------------------------------------------------------------

def test_download_writes_file_and_returns_image(segment, tmp_path):
    data = _jpeg_bytes()
    session = FakeSession(FakeResponse(200, data))
    target = tmp_path / "images" / "batch_00001" / "x.jpg"

    row, image, error, info = segment.fetch(session, ROW, target)

    assert error is None and image.size == (40, 30)
    assert target.read_bytes() == data
    assert not list(target.parent.glob("*.part")), "temp file left behind"
    assert info["reused_image"] == 0 and info["download_bytes"] == len(data)
    assert info["download_s"] >= 0
    assert session.calls == [(ROW["photo_url"], segment.HTTP_TIMEOUT)]


def test_http_error_is_reported_not_raised(segment, tmp_path):
    target = tmp_path / "x.jpg"
    row, image, error, info = segment.fetch(FakeSession(FakeResponse(404)), ROW, target)
    assert image is None and "404" in error
    assert not target.exists()
    assert "download_s" in info


def test_non_image_body_is_an_error_but_leaves_no_bad_cache(segment, tmp_path):
    """An HTML error page served with 200 must not be cached as a good image...
    """
    target = tmp_path / "x.jpg"
    _, image, error, _ = segment.fetch(FakeSession(FakeResponse(200, b"<html>")), ROW, target)
    assert image is None and error
    # ...and if it was written, a rerun must not trust it: the reuse path
    # re-downloads anything PIL cannot open.
    data = _jpeg_bytes()
    _, image, error, info = segment.fetch(FakeSession(FakeResponse(200, data)), ROW, target)
    assert error is None and info["reused_image"] == 0
    assert target.read_bytes() == data


def test_existing_image_is_reused_without_a_request(segment, tmp_path):
    target = tmp_path / "x.jpg"
    target.write_bytes(_jpeg_bytes())
    session = FakeSession(FakeResponse(500))
    _, image, error, info = segment.fetch(session, ROW, target)
    assert error is None and image is not None
    assert info["reused_image"] == 1 and session.calls == []


@pytest.mark.parametrize("junk", [b"", b"\xff\xd8\xff\xe0 truncated"])
def test_empty_or_truncated_image_is_downloaded_again(segment, tmp_path, junk):
    target = tmp_path / "x.jpg"
    target.write_bytes(junk)
    data = _jpeg_bytes()
    session = FakeSession(FakeResponse(200, data))
    _, image, error, info = segment.fetch(session, ROW, target)
    assert error is None and info["reused_image"] == 0 and len(session.calls) == 1
    assert target.read_bytes() == data


def test_session_retries_throttling_and_server_errors(segment):
    session = segment.make_session(16)
    adapter = session.get_adapter("https://inaturalist-open-data.s3.amazonaws.com/")
    retry = adapter.max_retries
    assert retry.total == 8
    assert {429, 500, 502, 503, 504} <= set(retry.status_forcelist)
    assert 404 not in retry.status_forcelist, "a deleted photo must fail fast"
    assert adapter._pool_maxsize == 16


# -- live S3 -------------------------------------------------------------------

@pytest.fixture(scope="module")
def session(segment):
    return segment.make_session(8)


@pytest.mark.network
@pytest.mark.parametrize("photo_id,ext", POPPY)
def test_real_photo_downloads(segment, session, tmp_path, photo_id, ext):
    url = segment.PHOTO_URL.format(photo_id=photo_id, ext=ext)
    _, image, error, info = segment.fetch(session, {"photo_id": photo_id, "photo_url": url},
                                          tmp_path / f"{photo_id}.{ext}")
    assert error is None, error
    assert min(image.size) > 100 and image.mode == "RGB"
    assert info["download_bytes"] > 10_000
    assert info["download_mbps"] > 0


@pytest.mark.network
def test_wrong_extension_is_a_fast_404(segment, session, tmp_path):
    """47848057 is stored as .jpeg; asking for .jpg must 404 without the eight
    backoff retries (which would take ~6 min per missing photo)."""
    url = segment.PHOTO_URL.format(photo_id=47848057, ext="jpg")
    _, image, error, info = segment.fetch(session, {"photo_id": 1, "photo_url": url},
                                          tmp_path / "x.jpg")
    assert image is None and ("404" in error or "403" in error)
    assert info["download_s"] < 15


@pytest.mark.network
def test_parquet_to_images_end_to_end(segment, index_parquet, tmp_path):
    """What segment.main does for one shard, minus the GPU: read the shard's rows
    out of the index, fetch them in parallel, land them in batch folders."""
    rows = segment.load_photos(index_parquet, POPPY_TAXON, None, 1, 2).to_dict("records")
    session = segment.make_session(4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda r: segment.fetch(session, r, segment._image_path(tmp_path, r)), rows))

    assert len(results) == 3
    for row, image, error, info in results:
        assert error is None, (row["photo_url"], error)
        path = segment._image_path(tmp_path, row)
        assert path.parent.name == "batch_00001"
        assert path.name == f"{row['stem']}.{row['extension']}"
        assert Image.open(path).size == image.size

    # A second pass is a rerun: every image is reused, nothing is requested.
    rerun = [segment.fetch(session, r, segment._image_path(tmp_path, r)) for r in rows]
    assert all(info["reused_image"] == 1 for *_, info in rerun)


# -- against the real index (opt-in) --------------------------------------------

@pytest.mark.network
@pytest.mark.real_index
def test_real_index_first_photos_download(segment, real_parquet, tmp_path):
    """The built index, the default taxon, first 20 photos across 2 shards: every
    URL it produces must resolve. Catches a broken or stale index build."""
    rows = segment.load_photos(real_parquet, 160559, 20, 0, 2).to_dict("records")
    session = segment.make_session(8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda r: segment.fetch(session, r, segment._image_path(tmp_path, r)), rows))
    failed = [(r["photo_url"], e) for r, _, e, _ in results if e]
    # iNat users delete photos and the dump lags, so allow a stray miss.
    assert len(failed) <= 1, failed
