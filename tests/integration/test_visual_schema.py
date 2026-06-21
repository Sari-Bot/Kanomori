import pytest

pytestmark = pytest.mark.requires_db


def test_visual_tables_exist(db_conn) -> None:
    tables = {
        row[0]
        for row in db_conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('frames', 'ocr_segments', 'scene_segments')
            """
        ).fetchall()
    }

    assert tables == {"frames", "ocr_segments", "scene_segments"}


def test_phash_hamming_sql_uses_bit_cast(db_conn) -> None:
    distance = db_conn.execute(
        "SELECT bit_count((%s::bigint # %s::bigint)::bit(64))",
        (-9223372036854775808, -9223372036854775807),
    ).fetchone()[0]

    assert distance == 1
