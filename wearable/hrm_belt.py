import asyncio
import threading
import time
import csv
import json
import os
from datetime import datetime
from bleak import BleakClient, BleakScanner

HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
DEVICE_NAME_HINT = "HR"

latest_hr = None
latest_rr_intervals = []
connected = False
last_update_ts = None
stop_flag = False
monitor_thread = None

csv_file = None
csv_writer = None
current_session_id = None
current_label = None

# Added: control console print rate and show elapsed session time
last_print_ts = 0
session_start_ts = None

DATA_DIR = "data/hr_sessions"
os.makedirs(DATA_DIR, exist_ok=True)


def parse_heart_rate_measurement(data: bytearray):
    if len(data) < 2:
        return None

    flags = data[0]
    hr_16bit = flags & 0x01
    energy_present = (flags >> 3) & 0x01
    rr_present = (flags >> 4) & 0x01

    index = 1

    if hr_16bit:
        if len(data) < 3:
            return None
        hr = int.from_bytes(data[index:index + 2], byteorder="little")
        index += 2
    else:
        hr = data[index]
        index += 1

    energy = None
    if energy_present and len(data) >= index + 2:
        energy = int.from_bytes(data[index:index + 2], byteorder="little")
        index += 2

    rr_intervals = []
    if rr_present:
        while len(data) >= index + 2:
            rr_raw = int.from_bytes(data[index:index + 2], byteorder="little")
            rr_intervals.append(rr_raw / 1024.0)
            index += 2

    return {"hr": hr, "energy": energy, "rr_intervals": rr_intervals}


def start_hr_recording(session_id: str, label: str):
    global csv_file, csv_writer, current_session_id, current_label
    global session_start_ts, last_print_ts

    filepath = os.path.join(DATA_DIR, f"{session_id}.csv")
    csv_file = open(filepath, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "timestamp",
            "unix_time",
            "session_id",
            "label",
            "hr",
            "rr_intervals",
            "energy",
        ],
    )
    csv_writer.writeheader()

    current_session_id = session_id
    current_label = label
    session_start_ts = time.time()
    last_print_ts = 0

    print(f"[HRM] Recording started: {filepath}")


def stop_hr_recording():
    global csv_file, csv_writer, current_session_id, current_label
    global session_start_ts

    if csv_file is not None:
        csv_file.close()
        print("[HRM] Recording stopped and file saved.")

    csv_file = None
    csv_writer = None
    current_session_id = None
    current_label = None
    session_start_ts = None


async def find_hr_device():
    print("[HRM] Scanning for HR belt...")
    devices = await BleakScanner.discover(timeout=8.0)

    for d in devices:
        name = d.name or ""
        print(f"[HRM] Found: {name} [{d.address}]")
        if DEVICE_NAME_HINT.lower() in name.lower():
            print(f"[HRM] Using device: {name} [{d.address}]")
            return d

    return None


async def hr_monitor_loop():
    global latest_hr, latest_rr_intervals, connected, last_update_ts, stop_flag

    device = await find_hr_device()
    if not device:
        print("[HRM] No HR device found.")
        connected = False
        return

    def notification_handler(sender, data):
        global latest_hr, latest_rr_intervals, last_update_ts
        global csv_writer, csv_file, last_print_ts

        parsed = parse_heart_rate_measurement(bytearray(data))
        if not parsed:
            return

        latest_hr = parsed["hr"]
        latest_rr_intervals = parsed["rr_intervals"]
        now = time.time()
        last_update_ts = now

        # Print at most once every 1 second
        if now - last_print_ts >= 1:
            elapsed = int(now - session_start_ts) if session_start_ts else 0
            mins = elapsed // 60
            secs = elapsed % 60
            print(f"[{mins:02d}:{secs:02d}] HR: {parsed['hr']}")
            last_print_ts = now

        if csv_writer is not None:
            csv_writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "unix_time": now,
                "session_id": current_session_id,
                "label": current_label,
                "hr": parsed["hr"],
                "rr_intervals": json.dumps(parsed["rr_intervals"]),
                "energy": parsed["energy"],
            })
            csv_file.flush()

    try:
        async with BleakClient(device) as client:
            connected = client.is_connected
            print("[HRM] Connected:", connected)

            if not connected:
                print("[HRM] Failed to connect.")
                return

            await client.start_notify(HR_CHAR_UUID, notification_handler)
            print("[HRM] Receiving HR notifications... Press Ctrl+C to stop.")

            try:
                while not stop_flag:
                    await asyncio.sleep(1)
            finally:
                await client.stop_notify(HR_CHAR_UUID)
                connected = False
                print("[HRM] Notifications stopped.")

    except Exception as e:
        connected = False
        print(f"[HRM] Connection error: {e}")


def _run_async_loop():
    try:
        asyncio.run(hr_monitor_loop())
    except Exception as e:
        print(f"[HRM] Async loop error: {e}")


def start_hr_monitor():
    global monitor_thread, stop_flag

    if monitor_thread is not None and monitor_thread.is_alive():
        return

    stop_flag = False
    monitor_thread = threading.Thread(target=_run_async_loop, daemon=True)
    monitor_thread.start()


def stop_hr_monitor():
    global stop_flag
    stop_flag = True
    stop_hr_recording()


def get_latest_hr():
    return latest_hr


def get_latest_rr_intervals():
    return latest_rr_intervals


def is_connected():
    return connected


def has_fresh_data(max_age_sec: int = 5):
    if last_update_ts is None:
        return False
    return (time.time() - last_update_ts) <= max_age_sec


if __name__ == "__main__":
    start_hr_recording("session_001", "calm")
    start_hr_monitor()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[HRM] Stopped by user.")
        stop_hr_monitor()
