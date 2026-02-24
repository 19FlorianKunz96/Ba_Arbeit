#!/usr/bin/env python3
"""
evaluate_bag_runs_v4.py

- Liest eine rosbag2 (sqlite3) aus (BAG_DIR)
- Segmentiert Runs über /perf/segment_event (RUN_START / RUN_END / COLLISION)
- Extrahiert /odom und /rl/target_path
- Berechnet pro Run:
    - time_s
    - planned_len_m
    - driven_len_m
    - planned_over_driven (planned / driven)
    - driven_over_time (driven / time -> m/s)
- Schreibt runs.csv und summary.json in BAG_DIR/offline_eval
- Reportet totals + Mittelwerte über alle Successes
"""

import json
import math
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# ----------------- EDITIEREN -----------------
BAG_DIR = "/home/verwalter/rosbags/Stage4_AS5_RewardRobotis_AlleKomponenten_20260213_144126"
STORAGE_ID = "sqlite3"
# --------------------------------------------

EVENT_TOPIC = "/perf/segment_event"
ODOM_TOPIC = "/odom"
PLAN_TOPIC = "/rl/target_path"   # ggf. "/plan"
STATUS_SUCCEEDED = 4  # action_msgs/GoalStatus.STATUS_SUCCEEDED

@dataclass
class Event:
    event: str
    run_id: int
    bag_time_ns: int
    fields: dict

@dataclass
class Run:
    run_id: int
    start_bag_ns: int
    end_bag_ns: int
    status: Optional[int]

def path_length_xy(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    dist = 0.0
    x0, y0 = points[0]
    for x1, y1 in points[1:]:
        dist += math.hypot(x1 - x0, y1 - y0)
        x0, y0 = x1, y1
    return dist

def mean(vals: List[float]) -> Optional[float]:
    return (sum(vals) / len(vals)) if vals else None

def safe_div(a: float, b: float) -> Optional[float]:
    try:
        if b == 0:
            return None
        return a / b
    except Exception:
        return None

def main():
    bag_path = str(Path(BAG_DIR))
    out_dir = Path(BAG_DIR) / "offline_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=STORAGE_ID)
    converter_options = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics_and_types}

    for t in (EVENT_TOPIC, ODOM_TOPIC, PLAN_TOPIC):
        if t not in type_map:
            print(f"ERROR: Topic missing in bag: {t}")
            return

    EventMsg = get_message(type_map[EVENT_TOPIC])  # std_msgs/msg/String
    OdomMsg  = get_message(type_map[ODOM_TOPIC])   # nav_msgs/msg/Odometry
    PathMsg  = get_message(type_map[PLAN_TOPIC])   # nav_msgs/msg/Path

    # --- load data from bag into memory (bag timestamps) ---
    events: List[Event] = []
    collision_count = 0

    odom_samples: List[Tuple[int, float, float]] = []            # (bag_t_ns, x, y)
    plan_paths: List[Tuple[int, List[Tuple[float, float]]]] = [] # (bag_t_ns, [(x,y),...])

    print("Reading bag... (this may take a moment)")
    while reader.has_next():
        topic, data, bag_t_ns = reader.read_next()
        bag_t_ns = int(bag_t_ns)

        if topic == EVENT_TOPIC:
            msg = deserialize_message(data, EventMsg)
            payload = json.loads(msg.data)
            ev = str(payload.get("event"))
            if ev == "COLLISION":
                collision_count += 1
            events.append(Event(
                event=ev,
                run_id=int(payload.get("run_id")),
                bag_time_ns=bag_t_ns,
                fields={k: v for k, v in payload.items() if k not in ("event", "run_id", "ros_time_ns")}
            ))

        elif topic == ODOM_TOPIC:
            msg = deserialize_message(data, OdomMsg)
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            odom_samples.append((bag_t_ns, x, y))

        elif topic == PLAN_TOPIC:
            msg = deserialize_message(data, PathMsg)
            pts = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]
            plan_paths.append((bag_t_ns, pts))

    if not odom_samples:
        print("\nERROR: Keine /odom Samples in Bag.")
        return

    # --- build runs from RUN_START / RUN_END using bag time ---
    starts: Dict[int, int] = {}
    ends: Dict[int, Tuple[int, Optional[int]]] = {}

    for e in events:
        if e.event == "RUN_START":
            starts[e.run_id] = e.bag_time_ns
        elif e.event == "RUN_END":
            status = e.fields.get("status", None)
            ends[e.run_id] = (e.bag_time_ns, int(status) if status is not None else None)

    runs: List[Run] = []
    for run_id, start_ns in sorted(starts.items()):
        if run_id not in ends:
            continue
        end_ns, status = ends[run_id]
        if end_ns < start_ns:
            start_ns, end_ns = end_ns, start_ns
        runs.append(Run(run_id=run_id, start_bag_ns=start_ns, end_bag_ns=end_ns, status=status))

    total_runs = len(runs)
    successes = sum(1 for r in runs if r.status == STATUS_SUCCEEDED)

    # --- compute per-run metrics and aggregate for averages over successes ---
    rows = []
    times_success: List[float] = []
    planned_success: List[float] = []
    driven_success: List[float] = []
    planned_over_driven_success: List[float] = []
    driven_over_time_success: List[float] = []

    for r in runs:
        driven_pts = [(x, y) for (t, x, y) in odom_samples if r.start_bag_ns <= t <= r.end_bag_ns]
        driven_len = path_length_xy(driven_pts)

        plan_in_window = [pts for (t, pts) in plan_paths if r.start_bag_ns <= t <= r.end_bag_ns]
        planned_pts = plan_in_window[-1] if plan_in_window else []
        planned_len = path_length_xy(planned_pts)

        time_s = (r.end_bag_ns - r.start_bag_ns) / 1e9

        planned_over_driven = safe_div(planned_len, driven_len)  # planned / driven
        driven_over_time = safe_div(driven_len, time_s)         # driven / time (m/s)

        if r.status == STATUS_SUCCEEDED:
            # collect for averages across successes
            times_success.append(time_s)
            planned_success.append(planned_len)
            driven_success.append(driven_len)
            if planned_over_driven is not None:
                planned_over_driven_success.append(planned_over_driven)
            if driven_over_time is not None:
                driven_over_time_success.append(driven_over_time)

        rows.append({
            "run_id": r.run_id,
            "status": r.status if r.status is not None else "",
            "time_s": time_s,
            "planned_len_m": planned_len,
            "driven_len_m": driven_len,
            "planned_over_driven": (planned_over_driven if planned_over_driven is not None else ""),
            "driven_over_time_m_per_s": (driven_over_time if driven_over_time is not None else ""),
            "n_odom_points": len(driven_pts),
            "n_plan_points_last": len(planned_pts),
        })

    # write per-run CSV into bag folder
    out_csv = out_dir / "runs.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "bag_dir": BAG_DIR,
        "total_runs": total_runs,
        "successes": successes,
        "collisions": collision_count,

        # averages over ALL successes
        "avg_time_s_over_successes": mean(times_success),
        "avg_planned_len_m_over_successes": mean(planned_success),
        "avg_driven_len_m_over_successes": mean(driven_success),
        "avg_planned_over_driven_over_successes": mean(planned_over_driven_success),
        "avg_driven_over_time_m_per_s_over_successes": mean(driven_over_time_success),

        "output_runs_csv": str(out_csv),
    }

    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Console output (kurz)
    print("\n=== RESULTS ===")
    print(f"Total runs: {total_runs}")
    print(f"Collisions (events): {collision_count}")
    print(f"Successes:  {successes}")
    print("\nAverages over ALL successes:")
    print(f"  avg time [s]:                     {summary['avg_time_s_over_successes']}")
    print(f"  avg planned length [m]:           {summary['avg_planned_len_m_over_successes']}")
    print(f"  avg driven length [m]:            {summary['avg_driven_len_m_over_successes']}")
    print(f"  avg planned/driven [-]:            {summary['avg_planned_over_driven_over_successes']}")
    print(f"  avg driven/time [m/s]:            {summary['avg_driven_over_time_m_per_s_over_successes']}")
    print(f"\nSaved: {out_json}")
    print(f"CSV:   {out_csv}")

if __name__ == "__main__":
    main()
