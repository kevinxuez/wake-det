"""Tests for YOLO-OBB → DOTA label format conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.train.train_oriented_rcnn import convert_yolo_obb_to_dota


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a minimal YOLO-OBB dataset for testing."""
    for split in ("train", "val", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)

    # Positive sample: two wakes
    (tmp_path / "train" / "labels" / "1.txt").write_text(
        "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8\n"
        "0 0.0 0.0 0.5 0.0 0.5 1.0 0.0 1.0\n"
    )
    # Negative sample: empty file
    (tmp_path / "train" / "labels" / "2.txt").write_text("")
    # Single wake
    (tmp_path / "val" / "labels" / "3.txt").write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
    )
    return tmp_path


class TestConvertYoloObbToDota:
    def test_creates_output_dirs(self, sample_dataset: Path):
        out = convert_yolo_obb_to_dota(sample_dataset, image_size=1024)
        assert (out / "train").is_dir()
        assert (out / "val").is_dir()

    def test_denormalises_coordinates(self, sample_dataset: Path):
        out = convert_yolo_obb_to_dota(sample_dataset, image_size=1024)
        content = (out / "train" / "1.txt").read_text().strip()
        lines = content.splitlines()
        assert len(lines) == 2

        # First line: 0.1*1024=102.4, 0.2*1024=204.8, ...
        parts = lines[0].split()
        assert float(parts[0]) == pytest.approx(102.4, abs=0.1)
        assert float(parts[1]) == pytest.approx(204.8, abs=0.1)
        assert parts[8] == "ship_wake"
        assert parts[9] == "0"  # difficulty

    def test_negative_sample_empty(self, sample_dataset: Path):
        out = convert_yolo_obb_to_dota(sample_dataset, image_size=1024)
        content = (out / "train" / "2.txt").read_text()
        assert content == ""

    def test_preserves_file_count(self, sample_dataset: Path):
        out = convert_yolo_obb_to_dota(sample_dataset, image_size=1024)
        assert len(list((out / "train").glob("*.txt"))) == 2
        assert len(list((out / "val").glob("*.txt"))) == 1

    def test_custom_output_root(self, sample_dataset: Path, tmp_path: Path):
        custom_out = tmp_path / "custom_dota"
        out = convert_yolo_obb_to_dota(
            sample_dataset, output_root=custom_out, image_size=512
        )
        assert out == custom_out
        content = (out / "val" / "3.txt").read_text().strip()
        parts = content.split()
        # 0.25 * 512 = 128.0
        assert float(parts[0]) == pytest.approx(128.0, abs=0.1)
