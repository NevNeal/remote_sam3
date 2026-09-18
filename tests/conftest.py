"""
Shared fixtures for the download-path tests: reading photo rows out of the
parquet index (segment.load_photos) and fetching them from the iNat open-data
S3 bucket (segment.fetch / make_session).

These stages need no GPU, so torch and transformers are stubbed when they are
not installed. That lets the tests run on a laptop or a login node; in the real
sam3 env the genuine packages are imported instead.

    pytest tests/                    # everything, including live S3 requests
    pytest tests/ -m "not network"   # offline only
    INAT_PARQUET=data/inat_photos.parquet pytest tests/ -m real_index
"""

import importlib
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_missing_gpu_stack():
    try:
        import torch  # noqa: F401
        from transformers import Sam3Model, Sam3Processor  # noqa: F401
        return
    except ImportError:
        pass
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None)
    torch.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch
    transformers = types.ModuleType("transformers")
    transformers.Sam3Model = transformers.Sam3Processor = object
    sys.modules["transformers"] = transformers


_stub_missing_gpu_stack()


@pytest.fixture(scope="session")
def segment():
    return importlib.import_module("segment")


# Real research-grade California poppy photos (taxon 48225), checked to exist in
# the bucket. Two of them are .jpeg, not .jpg - the index's extension column is
# what makes their URLs resolve.
POPPY = [
    (9211, "jpg"), (580121, "jpg"), (31164303, "jpg"),
    (47848057, "jpeg"), (181513663, "jpg"), (255584384, "jpeg"),
]
POPPY_TAXON = 48225


def _index_table(rows):
    """Rows in exactly the schema build_index.py writes."""
    schema = pa.schema([
        ("taxon_id", pa.int32()), ("photo_id", pa.int64()), ("extension", pa.string()),
        ("taxon_name", pa.string()), ("quality_grade", pa.string()),
        ("latitude", pa.float32()), ("longitude", pa.float32()),
    ])
    return pa.Table.from_pandas(pd.DataFrame(rows), schema=schema, preserve_index=False)


@pytest.fixture(scope="session")
def index_parquet(tmp_path_factory):
    """A small index shaped like data/inat_photos.parquet:

    - the six real poppy photos, written out of photo_id order
    - one of them duplicated byte-for-byte, as iNat's dumps do ~0.1% of the time
    - 2,500 synthetic photos of a second taxon, to cross batch_NNNNN boundaries
    - a few rows of an unrelated taxon, which the filter must exclude
    """
    rows = []
    for photo_id, ext in reversed(POPPY):
        rows.append(dict(taxon_id=POPPY_TAXON, photo_id=photo_id, extension=ext,
                         taxon_name="Eschscholzia californica", quality_grade="research",
                         latitude=37.5, longitude=-122.1))
    rows.append(dict(rows[2]))
    for i in range(2500):
        rows.append(dict(taxon_id=999, photo_id=10_000_000 + 7 * i, extension="jpg",
                         taxon_name="Symphyotrichum novae-angliae", quality_grade="research",
                         latitude=40.0, longitude=-80.0))
    for i in range(5):
        rows.append(dict(taxon_id=52899, photo_id=9208 + i, extension="png",
                         taxon_name="Solanum carolinense", quality_grade="research",
                         latitude=None, longitude=None))
    path = tmp_path_factory.mktemp("index") / "inat_photos.parquet"
    # Several row groups, so the taxon filter is actually pushed down across them.
    pq.write_table(_index_table(rows), path, row_group_size=700)
    return path


@pytest.fixture(scope="session")
def real_parquet():
    path = os.environ.get("INAT_PARQUET")
    if not path or not Path(path).exists():
        pytest.skip("set INAT_PARQUET to the real index to run this")
    return Path(path)
