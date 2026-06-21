from kanomori.ingest.artifacts import frame_dir_for, frame_path_for


def test_frame_artifact_paths_are_content_hash_keyed() -> None:
    assert frame_dir_for("abc123").parts[-2:] == ("abc123", "frames")
    assert frame_path_for("abc123", 12.345).name == "frame_000012_345.jpg"
