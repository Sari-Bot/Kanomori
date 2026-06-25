from kanomori.ingest.job_time_costs import merge_time_costs


def test_merge_time_costs_preserves_pipeline_order():
    merged = merge_time_costs(
        [{"stage": "ocr", "seconds": 3.0}, {"stage": "register", "seconds": 1.0}],
        "parse_transcript",
        2.0,
    )

    assert merged == [
        {"stage": "register", "seconds": 1.0},
        {"stage": "parse_transcript", "seconds": 2.0},
        {"stage": "ocr", "seconds": 3.0},
    ]


def test_merge_time_costs_replaces_existing_stage_entry():
    merged = merge_time_costs(
        [{"stage": "register", "seconds": 1.0}, {"stage": "frames", "seconds": 2.0}],
        "register",
        9.999,
    )

    assert merged == [
        {"stage": "register", "seconds": 9.999},
        {"stage": "frames", "seconds": 2.0},
    ]
