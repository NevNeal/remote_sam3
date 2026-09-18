"""segment.load_photos: which rows of the parquet index a shard gets, and the
URLs, filenames and batch folders derived from them."""

import pandas as pd
import pytest

from conftest import POPPY, POPPY_TAXON


def test_filters_to_taxon_and_sorts(segment, index_parquet):
    df = segment.load_photos(index_parquet, POPPY_TAXON, None, 0, 1)
    assert set(df["taxon_id"]) == {POPPY_TAXON}
    assert df["photo_id"].tolist() == sorted(p for p, _ in POPPY)


def test_drops_duplicate_photo_rows(segment, index_parquet, capsys):
    df = segment.load_photos(index_parquet, POPPY_TAXON, None, 0, 1)
    assert df["photo_id"].is_unique
    assert len(df) == len(POPPY)
    assert "dropped 1 duplicate" in capsys.readouterr().out


def test_urls_use_the_index_extension(segment, index_parquet):
    df = segment.load_photos(index_parquet, POPPY_TAXON, None, 0, 1)
    expected = {
        p: f"https://inaturalist-open-data.s3.amazonaws.com/photos/{p}/original.{e}"
        for p, e in POPPY
    }
    assert dict(zip(df["photo_id"], df["photo_url"])) == expected


def test_stem_and_metadata(segment, index_parquet):
    df = segment.load_photos(index_parquet, 999, 3, 0, 1)
    assert df["stem"].iloc[0] == "Symphyotrichum_novaeangliae_10000000"
    row = df.iloc[0]
    assert row["quality_grade"] == "research"
    assert row["latitude"] == pytest.approx(40.0)


def test_row_index_is_contiguous_and_batches_hold_1000(segment, index_parquet):
    df = segment.load_photos(index_parquet, 999, None, 0, 1)
    assert df["row_index"].tolist() == list(range(2500))
    counts = df["batch"].value_counts().sort_index()
    assert counts.to_dict() == {"batch_00001": 1000, "batch_00002": 1000, "batch_00003": 500}
    assert df.loc[df["row_index"] == 999, "batch"].item() == "batch_00001"
    assert df.loc[df["row_index"] == 1000, "batch"].item() == "batch_00002"


def test_limit_applies_before_sharding(segment, index_parquet):
    """LIMIT=200 across 2 shards gives each shard 100 - the README contract."""
    shards = [segment.load_photos(index_parquet, 999, 200, s, 2) for s in range(2)]
    assert [len(s) for s in shards] == [100, 100]
    assert sorted(pd.concat(shards)["row_index"]) == list(range(200))


@pytest.mark.parametrize("num_shards", [2, 3, 7, 60])
def test_shards_partition_the_taxon(segment, index_parquet, num_shards):
    whole = segment.load_photos(index_parquet, 999, None, 0, 1)
    parts = [segment.load_photos(index_parquet, 999, None, s, num_shards)
             for s in range(num_shards)]
    merged = pd.concat(parts)
    assert merged["photo_id"].is_unique, "a photo landed in two shards"
    assert len(merged) == len(whole), "a photo landed in no shard"
    for s, part in enumerate(parts):
        assert (part["row_index"] % num_shards == s).all()


def test_names_do_not_depend_on_shard_count(segment, index_parquet):
    """A rerun with a different NUM_SHARDS must write the same filenames into the
    same batch folders, or resume and dedup both break."""
    cols = ["photo_id", "row_index", "stem", "batch", "photo_url"]
    whole = segment.load_photos(index_parquet, 999, None, 0, 1)[cols]
    sharded = pd.concat(segment.load_photos(index_parquet, 999, None, s, 5)
                        for s in range(5))[cols]
    merged = whole.merge(sharded, on="photo_id", suffixes=("", "_sharded"))
    for c in cols[1:]:
        assert (merged[c] == merged[f"{c}_sharded"]).all(), c


def test_unknown_taxon_exits(segment, index_parquet):
    with pytest.raises(SystemExit, match="No research-grade photos"):
        segment.load_photos(index_parquet, 123456789, None, 0, 1)


def test_more_shards_than_photos_exits(segment, index_parquet):
    with pytest.raises(SystemExit, match="got no rows"):
        segment.load_photos(index_parquet, POPPY_TAXON, None, 10, 20)
