from __future__ import annotations


def merge_time_costs(
    existing: list[dict] | None, stage_name: str, seconds: float
) -> list[dict[str, float | str]]:
    entries = {
        item["stage"]: {"stage": item["stage"], "seconds": item["seconds"]}
        for item in (existing or [])
        if isinstance(item, dict) and "stage" in item and "seconds" in item
    }
    entries[stage_name] = {"stage": stage_name, "seconds": round(float(seconds), 3)}
    return sorted(entries.values(), key=_stage_index)


def _stage_index(entry: dict[str, float | str]) -> tuple[int, str]:
    from kanomori.ingest.pipeline import STAGES

    order = {name: idx for idx, (name, _mod) in enumerate(STAGES)}
    stage = str(entry["stage"])
    return (order.get(stage, len(order)), stage)
