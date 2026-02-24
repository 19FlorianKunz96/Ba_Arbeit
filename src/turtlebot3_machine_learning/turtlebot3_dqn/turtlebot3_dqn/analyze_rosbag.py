#!/usr/bin/env python3
import json
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG_DIR = "/home/verwalter/rosbags/Stage4_AS5_RewardRobotis_AlleKomponenten_20260213_144126"
STORAGE_ID = "sqlite3"

def main():

    #öffnen
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(Path(BAG_DIR)), storage_id=STORAGE_ID)
    converter_options = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader.open(storage_options, converter_options)

    #was ist drin?
    topics_and_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics_and_types}

    print("Topics in bag:")
    for name, typ in sorted(type_map.items()):
        print(f"  {name}: {typ}")

    if "/perf/segment_event" not in type_map:
        print("\nERROR: /perf/segment_event not found in bag.")
        return

    #auslesen einer bestimmten Message
    msg_type = get_message(type_map["/perf/segment_event"])
    print("\n--- /perf/segment_event payloads ---")
    n = 0
    
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != "/perf/segment_event":
            continue
        msg = deserialize_message(data, msg_type)  # std_msgs/String
        payload = json.loads(msg.data)
        print(payload)
        n += 1

    print(f"\nDone. Read {n} segment events.")

if __name__ == "__main__":
    main()
