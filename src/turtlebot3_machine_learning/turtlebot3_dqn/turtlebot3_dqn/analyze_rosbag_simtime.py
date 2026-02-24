#!/usr/bin/env python3
import json, math, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG_DIR = "/home/verwalter/rosbags/Stage4_AS5_RewardRobotis_AlleKomponenten_20260213_144126"
STORAGE_ID = "sqlite3"

EVENT_TOPIC = "/perf/segment_event"
ODOM_TOPIC = "/odom"
PLAN_TOPIC = "/rl/target_path"
STATUS_SUCCEEDED = 4

@dataclass
class Event:
    event: str
    run_id: int
    ros_time_ns: int
    bag_time_ns: int
    fields: dict

@dataclass
class Run:
    run_id: int
    start_ros_ns: int
    end_ros_ns: int
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
    if b is None or b == 0.0:
        return None
    return a / b

def main():
    out_dir = Path(BAG_DIR) / "offline_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(Path(BAG_DIR)), storage_id=STORAGE_ID),
        rosbag2_py.ConverterOptions("cdr", "cdr")
    )

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for t in (EVENT_TOPIC, ODOM_TOPIC, PLAN_TOPIC):
        if t not in type_map:
            print(f"ERROR: missing topic {t}")
            return

    EventMsg = get_message(type_map[EVENT_TOPIC])
    OdomMsg  = get_message(type_map[ODOM_TOPIC])
    PathMsg  = get_message(type_map[PLAN_TOPIC])

    events: List[Event] = []
    collision_count = 0

    odom_samples_bag: List[Tuple[int, float, float]] = []   # (bag_t_ns, x, y)
    plan_paths_bag: List[Tuple[int, List[Tuple[float, float]]]] = []  # (bag_t_ns, pts)

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
                ros_time_ns=int(payload.get("ros_time_ns")),
                bag_time_ns=bag_t_ns,
                fields={k: v for k, v in payload.items() if k not in ("event","run_id","ros_time_ns")}
            ))

        elif topic == ODOM_TOPIC:
            msg = deserialize_message(data, OdomMsg)
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            odom_samples_bag.append((bag_t_ns, x, y))

        elif topic == PLAN_TOPIC:
            msg = deserialize_message(data, PathMsg)
            pts = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]
            plan_paths_bag.append((bag_t_ns, pts))

    events.sort(key=lambda e: e.ros_time_ns)
    odom_samples_bag.sort(key=lambda it: it[0])
    plan_paths_bag.sort(key=lambda it: it[0])

    starts: Dict[int, Tuple[int,int]] = {}
    ends: Dict[int, Tuple[int,int,Optional[int]]] = {}

    for e in events:
        if e.event == "RUN_START":
            starts[e.run_id] = (e.ros_time_ns, e.bag_time_ns)
        elif e.event == "RUN_END":
            status = e.fields.get("status", None)
            ends[e.run_id] = (e.ros_time_ns, e.bag_time_ns, int(status) if status is not None else None)

    runs: List[Run] = []
    for run_id, (s_ros, s_bag) in sorted(starts.items()):
        if run_id not in ends:
            continue
        e_ros, e_bag, status = ends[run_id]
        if e_ros < s_ros: s_ros, e_ros = e_ros, s_ros
        if e_bag < s_bag: s_bag, e_bag = e_bag, s_bag
        runs.append(Run(run_id, s_ros, e_ros, s_bag, e_bag, status))

    total_runs = len(runs)
    successes = sum(1 for r in runs if r.status == STATUS_SUCCEEDED)

    # averages over successes
    times_s_s, planned_m_s, driven_m_s = [], [], []
    planned_over_driven_s, driven_per_time_s = [], []

    rows = []
    for r in runs:
        # driven path by BAG window
        driven_pts = [(x,y) for (t,x,y) in odom_samples_bag if r.start_bag_ns <= t <= r.end_bag_ns]
        driven_len = path_length_xy(driven_pts)

        # planned path by BAG window (last path)
        plan_in_window = [pts for (t,pts) in plan_paths_bag if r.start_bag_ns <= t <= r.end_bag_ns]
        planned_pts = plan_in_window[-1] if plan_in_window else []
        planned_len = path_length_xy(planned_pts)

        # time by SIM (event ros_time_ns)
        time_s = (r.end_ros_ns - r.start_ros_ns) / 1e9

        planned_over_driven = safe_div(planned_len, driven_len)
        driven_per_time = safe_div(driven_len, time_s)

        if r.status == STATUS_SUCCEEDED:
            times_s_s.append(time_s)
            planned_m_s.append(planned_len)
            driven_m_s.append(driven_len)
            if planned_over_driven is not None: planned_over_driven_s.append(planned_over_driven)
            if driven_per_time is not None: driven_per_time_s.append(driven_per_time)

        rows.append({
            "run_id": r.run_id,
            "status": r.status if r.status is not None else "",
            "time_s": time_s,
            "planned_len_m": planned_len,
            "driven_len_m": driven_len,
            "planned_over_driven": planned_over_driven if planned_over_driven is not None else "",
            "driven_per_time_m_per_s": driven_per_time if driven_per_time is not None else "",
            "n_odom_points": len(driven_pts),
            "n_plan_points_last": len(planned_pts),
        })

    out_csv = out_dir / "runs.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "bag_dir": BAG_DIR,
        "total_runs": total_runs,
        "collisions": collision_count,
        "successes": successes,
        "avg_time_s_over_successes": mean(times_s_s),
        "avg_planned_len_m_over_successes": mean(planned_m_s),
        "avg_driven_len_m_over_successes": mean(driven_m_s),
        "avg_planned_over_driven_over_successes": mean(planned_over_driven_s),
        "avg_driven_per_time_m_per_s_over_successes": mean(driven_per_time_s),
        "output_runs_csv": str(out_csv),
    }

    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("\n=== RESULTS (time=SIM via events, paths=BAG windows) ===")
    print(f"Total runs: {total_runs}")
    print(f"Collisions (events): {collision_count}")
    print(f"Successes: {successes}")
    print("\nAverages over ALL successes:")
    print(f"  avg time [s]:                     {summary['avg_time_s_over_successes']}")
    print(f"  avg planned length [m]:           {summary['avg_planned_len_m_over_successes']}")
    print(f"  avg driven length [m]:            {summary['avg_driven_len_m_over_successes']}")
    print(f"  avg planned/driven [-]:            {summary['avg_planned_over_driven_over_successes']}")
    print(f"  avg driven/time [m/s]:            {summary['avg_driven_per_time_m_per_s_over_successes']}")
    print(f"\nSaved: {out_json}")

if __name__ == "__main__":
    main()
