import asyncio
import glob
import time
import sys
import smbus2
from bleak import BleakScanner, BleakClient

sys.modules["smbus"] = smbus2
from mpu6050 import mpu6050


HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
DEVICE_NAME_HINT = "HR"

latest_hr = None
latest_rr = []
hr_connected = False

mpu = mpu6050(0x68)


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
    latest_hr, latest_rr = parse_hr(data)


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


def read_mpu():
    try:
        accel = mpu.get_accel_data()
        gyro = mpu.get_gyro_data()

        movement_score = (
            abs(accel["x"]) +
            abs(accel["y"]) +
            abs(accel["z"] - 9.81)
        )

        return accel, gyro, movement_score
    except Exception:
        return None, None, None


def read_temperature():
    try:
        folders = glob.glob("/sys/bus/w1/devices/28-*")

        if not folders:
            return None

        device_file = folders[0] + "/w1_slave"

        with open(device_file, "r") as f:
            lines = f.readlines()

        if lines[0].strip()[-3:] != "YES":
            return None

        temp_pos = lines[1].find("t=")
        if temp_pos != -1:
            return float(lines[1][temp_pos + 2:]) / 1000.0

        return None

    except Exception:
        return None


def get_sensor_data():
    accel, gyro, movement = read_mpu()
    temp = read_temperature()

    return {
        "hr_connected": hr_connected,
        "heart_rate": latest_hr,
        "rr_intervals": latest_rr,
        "temperature": temp,
        "accel": accel,
        "gyro": gyro,
        "movement_score": movement,
    }


async def main():
    asyncio.create_task(start_hrm())

    while True:
        data = get_sensor_data()

        print("\n====== REAL-TIME SENSOR DATA ======")
        print(f"HR Connected: {data['hr_connected']}")
        print(f"Heart Rate: {data['heart_rate']} bpm")
        print(f"RR Intervals: {data['rr_intervals']}")
        print(f"Temperature: {data['temperature']} °C")
        print(f"Accel: {data['accel']}")
        print(f"Gyro: {data['gyro']}")
        print(f"Movement Score: {data['movement_score']}")
        print("===================================")

        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())