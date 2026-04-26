import asyncio
import csv
import os
import argparse
import serial
import threading
import time
from datetime import datetime
from bleak import BleakScanner, BleakClient

# ================= HRM BELT CONFIG =================
HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
DEVICE_NAME_HINT = "HR"   # change to "Decathlon" if needed

latest_hr = None
latest_rr = []
hr_connected = False

# ================= ARDUINO CONFIG =================
SERIAL_PORT = "COM5"
BAUD_RATE = 9600

latest_temperature = None
latest_accel = None
latest_gyro = None
latest_movement_score = None
arduino_connected = False


# ================= HRM FUNCTIONS =================
def parse_hr(data):
    flags = data[0]
    hr_16bit = flags & 0x01
    rr_present = flags & 0x10
    index = 1

    if hr_16bit:
        hr = int.from_bytes(data[index:index + 2], "little")
        index += 2
    else:
        hr = data[index]
        index += 1

    rr = []
    if rr_present:
        while index + 1 < len(data):
            rr_raw = int.from_bytes(data[index:index + 2], "little")
            rr.append(rr_raw / 1024.0)
            index += 2

    return hr, rr


def hr_callback(sender, data):
    global latest_hr, latest_rr

    hr, rr = parse_hr(data)
    latest_hr = hr

    if rr:
        latest_rr.extend(rr)
        latest_rr = latest_rr[-30:]  # rolling RR history for RMSSD


async def start_hrm():
    global hr_connected

    print("Scanning for HRM belt...")
    devices = await BleakScanner.discover(timeout=8)

    target = None
    for d in devices:
        if d.name and DEVICE_NAME_HINT.lower() in d.name.lower():
            target = d
            break

    if target is None:
        print("HRM belt not found.")
        return

    print(f"Connected to HRM: {target.name}")

    async with BleakClient(target.address) as client:
        hr_connected = True
        await client.start_notify(HR_CHAR_UUID, hr_callback)

        while True:
            await asyncio.sleep(1)


# ================= ARDUINO FUNCTIONS =================
def parse_arduino_line(line):
    """
    Arduino sends:
    accelX,accelY,accelZ,movement,temp

    Example:
    0.012,-0.030,0.998,0.044,32.50
    """
    line = line.strip()

    if line.startswith("STATUS"):
        print("Arduino status:", line)
        return None, None, None, None

    try:
        parts = line.split(",")

        if len(parts) != 5:
            return None, None, None, None

        ax = float(parts[0])
        ay = float(parts[1])
        az = float(parts[2])
        movement_score = float(parts[3])
        temp = float(parts[4])

        if temp == -1.0:
            temp = None

        accel = {
            "x": ax,
            "y": ay,
            "z": az,
        }

        gyro = None

        return temp, accel, gyro, movement_score

    except ValueError:
        return None, None, None, None


def arduino_reader():
    global latest_temperature
    global latest_accel
    global latest_gyro
    global latest_movement_score
    global arduino_connected

    while True:
        try:
            print(f"Connecting to Arduino on {SERIAL_PORT}...")
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            arduino_connected = True
            print("Arduino connected.")

            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                temp, accel, gyro, movement = parse_arduino_line(line)

                if temp is not None:
                    latest_temperature = temp

                if accel is not None:
                    latest_accel = accel

                if gyro is not None:
                    latest_gyro = gyro

                if movement is not None:
                    latest_movement_score = movement

        except Exception as e:
            arduino_connected = False
            print(f"Arduino error: {e}")
            print("Retrying Arduino connection in 3 seconds...")
            time.sleep(3)


# ================= FEATURES =================
def compute_rmssd(rr_intervals):
    if rr_intervals is None or len(rr_intervals) < 2:
        return None

    diffs = [
        rr_intervals[i] - rr_intervals[i - 1]
        for i in range(1, len(rr_intervals))
    ]

    mean_squared = sum(d ** 2 for d in diffs) / len(diffs)

    return mean_squared ** 0.5


def classify_sensor_state(hr, movement, temperature=None):
    if hr is None:
        return "calm"

    movement = movement if movement is not None else 0

    if hr >= 95 and movement >= 0.60:
        return "stress"

    if hr >= 105:
        return "stress"

    if movement >= 0.85:
        return "active"

    if hr <= 82 and movement <= 0.52:
        return "calm"

    return "neutral"


def get_sensor_data():
    rr_mean = None
    rmssd = None

    if latest_rr:
        rr_mean = sum(latest_rr) / len(latest_rr)
        rmssd = compute_rmssd(latest_rr)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hr_connected": hr_connected,
        "arduino_connected": arduino_connected,
        "heart_rate": latest_hr,
        "rr_intervals": latest_rr,
        "rr_mean": rr_mean,
        "rmssd": rmssd,
        "temperature": latest_temperature,
        "accel": latest_accel,
        "gyro": latest_gyro,
        "movement_score": latest_movement_score,
    }


def get_sensor_state():
    data = get_sensor_data()

    data["sensor_state"] = classify_sensor_state(
        data["heart_rate"],
        data["movement_score"],
        data["temperature"],
    )

    return data


# ================= CSV RECORDING =================
def ensure_csv_header(output_file):
    if os.path.exists(output_file):
        return

    folder = os.path.dirname(output_file)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "heart_rate",
            "rr_mean",
            "rmssd",
            "temperature",
            "movement_score",
            "sensor_state",
            "label",
        ])


def save_row(output_file, data, label):
    ensure_csv_header(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            data["timestamp"],
            data["heart_rate"],
            data["rr_mean"],
            data["rmssd"],
            data["temperature"],
            data["movement_score"],
            data["sensor_state"],
            label,
        ])


# ================= MAIN =================
async def main(label=None, output=None):
    arduino_thread = threading.Thread(target=arduino_reader, daemon=True)
    arduino_thread.start()

    asyncio.create_task(start_hrm())

    if output:
        ensure_csv_header(output)
        print(f"Recording to: {output}")
        print(f"Label: {label}")
        print("Waiting for Arduino + HRM belt before recording...")

    while output and label:
        data = get_sensor_state()

        if data["arduino_connected"] and data["hr_connected"] and data["heart_rate"] is not None:
            print("Both Arduino and HRM are connected. Recording started.")
            break

        print(
            f"Waiting... Arduino={data['arduino_connected']} | "
            f"HRM={data['hr_connected']} | HR={data['heart_rate']}"
        )
        await asyncio.sleep(1)

    while True:
        data = get_sensor_state()

        print("\n====== REAL-TIME SENSOR DATA ======")
        print(f"HR Connected: {data['hr_connected']}")
        print(f"Arduino Connected: {data['arduino_connected']}")
        print(f"Heart Rate: {data['heart_rate']} bpm")
        print(f"RR Intervals: {data['rr_intervals']}")
        print(f"RR Mean: {data['rr_mean']}")
        print(f"RMSSD: {data['rmssd']}")
        print(f"Temperature: {data['temperature']} °C")
        print(f"Accel: {data['accel']}")
        print(f"Gyro: {data['gyro']}")
        print(f"Movement Score: {data['movement_score']}")
        print(f"Sensor State: {data['sensor_state']}")
        print("===================================")

        if output and label:
            save_row(output, data, label)

        await asyncio.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    asyncio.run(main(label=args.label, output=args.output))