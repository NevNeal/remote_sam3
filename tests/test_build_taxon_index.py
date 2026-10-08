"""
build_taxon_index.py: taxon photos (inat_photos.parquet) x flower-detected
annotation rows x image files on disk -> a --local-index parquet.
"""

import csv
import sys

import pandas as pd

import build_taxon_index
import segment

HEADER = ("dataSource,scientificName,trait,family,year,dayOfYear,latitude,longitude,"
          "observedMetadataUrl,occurrenceID,genus,date,recordedBy,"
          "coordinateUncertaintyInMeters,modelUri,accuracyExcludingUncertainFamily,"
          "observedImageUrl,predictionClass,countImages,countFamily,certainty,"
          "predictionProbability,proportionCertaintyFamily,accuracyFamily,"
          "observedImageGuid,basisOfRecord,annotationID,verbatimTrait,"
          "annotationMethod").split(",")


def _row(photo_id, trait="flower present", cls="Detected", name="Rudbeckia hirta", p=0.99):
    values = dict.fromkeys(HEADER, "")
    values.update(dataSource="iNaturalist", scientificName=name, trait=trait,
                  occurrenceID=f"obs-{photo_id}", predictionClass=cls, countImages="2",
                  certainty="High", predictionProbability=str(p),
                  observedImageGuid=str(photo_id),
                  observedImageUrl=f"https://www.inaturalist.org/photos/{photo_id}")
    return values


def test_build_taxon_index(tmp_path, monkeypatch):
    root = tmp_path / "share"
    medium = root / "phenobase_inat_data" / "images" / "medium"
    for batch, pid, ext in [(1, 101, "webp"), (7, 102, "webp"), (3, 104, "jpg"),
                            (2, 999, "webp")]:
        (medium / f"batch_{batch}").mkdir(parents=True, exist_ok=True)
        (medium / f"batch_{batch}" / f"{pid}.{ext}").write_bytes(b"x")

    inat = tmp_path / "inat_photos.parquet"
    pd.DataFrame({"photo_id": [101, 102, 103, 104, 105, 106, 999],
                  "taxon_id": [62741] * 6 + [1],
                  "taxon_name": ["Rudbeckia hirta"] * 6 + ["Other"]}).to_parquet(inat)

    # 101 in the path index; 102 only findable by listing; 103 nowhere on disk;
    # 104 indexed as .webp but stored as .jpg; 105 Not Detected; 106 another
    # trait; 999 another taxon.
    annotations = tmp_path / "annotations.csv"
    with open(annotations, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for r in [_row(101), _row(101, p=0.5), _row(102), _row(103), _row(104),
                  _row(105, cls="Not Detected"), _row(106, trait="fruit present"),
                  _row(999, name="Other")]:
            writer.writerow(r)

    paths = tmp_path / "local_image_paths.parquet"
    pd.DataFrame({"photo_id": [101, 104],
                  "file_name": ["data/phenobase_inat_data/images/medium/batch_1/101.webp",
                                "data/phenobase_inat_data/images/medium/batch_3/104.webp"]}
                 ).to_parquet(paths)

    out = tmp_path / "data" / "62741_flower_index.parquet"
    monkeypatch.setattr(sys, "argv", [
        "build_taxon_index.py", "--taxon-id", "62741", "--parquet", str(inat),
        "--annotations", str(annotations), "--image-root", str(root),
        "--path-index", str(paths), "--out", str(out), "--chunksize", "3"])
    build_taxon_index.main()

    index = pd.read_parquet(out)
    assert list(index["photo_id"]) == [101, 102, 104]
    assert set(index["taxon_id"]) == {62741}
    by_id = dict(zip(index["photo_id"], index["file_name"]))
    assert by_id[102] == "data/phenobase_inat_data/images/medium/batch_7/102.webp"
    assert by_id[104] == "data/phenobase_inat_data/images/medium/batch_3/104.jpg"
    assert index.loc[index.photo_id == 101, "prediction_probability"].item() == 0.99

    missing = pd.read_csv(out.with_name("62741_flower_index_missing.csv"))
    assert list(missing["photo_id"]) == [103]
    assert len(pd.read_parquet(out.with_name("62741_photos.parquet"))) == 6

    # It is a valid --local-index: segment.py resolves every image.
    photos = segment.load_local_photos(out, root, None, 0, 1)
    assert all(segment.resolve_local(p) for p in photos["local_path"])
