# coding=utf-8
from __future__ import absolute_import

import octoprint.plugin
import threading
import time
import os
import csv
import json
import requests
import shutil
import serial
import glob

class SerialSensorReader(threading.Thread):
    def __init__(self, port, shared_data, lock, logger):
        super().__init__(daemon=True)
        self._port = port
        self._shared_data = shared_data
        self._lock = lock
        self._logger = logger
        self._running = True

    def run(self):
        self._logger.info(f"Checking port {self._port}...")
        ser = None

        while self._running:
            try:
                # connect if not connected
                if ser is None:
                    try:
                        ser = serial.Serial(self._port, 115200, timeout=1)
                        time.sleep(2)
                    except:
                        time.sleep(5)
                        continue

                # read parse
                if ser.in_waiting > 0:
                    try:
                        raw_data = ser.read_all().decode('utf-8', errors='ignore')
                        lines = raw_data.strip().split('\n')
                        if not lines: continue
                        
                        last_line = lines[-1]

                        if last_line.startswith('{') and last_line.endswith('}'):
                            data = json.loads(last_line)
                            
                            # sort id sensors and update mailbox
                            sensor_id = data.get("id")

                            x = float(data.get("x", 0.0))
                            y = float(data.get("y", 0.0))
                            z = float(data.get("z", 0.0))

                            with self._lock:
                                if sensor_id == "adxl1":
                                    self._shared_data["adxl1"] = {"x": x, "y": y, "z": z}
                                elif sensor_id == "adxl2":
                                    self._shared_data["adxl2"] = {"x": x, "y": y, "z": z}  
                                elif sensor_id == "load":
                                    self._shared_data["load_cell"] = float(data.get("val", 0.0))
                                    
                    except Exception:
                        pass
                
                time.sleep(0.01) # important 

            except Exception:
                if ser: ser.close()
                ser = None
                time.sleep(1)

    def stop(self):
        self._running = False

class DataCollectorPlugin(
    octoprint.plugin.StartupPlugin,
    octoprint.plugin.ShutdownPlugin,
    octoprint.plugin.SettingsPlugin,
    octoprint.plugin.TemplatePlugin
):

    def __init__(self):
        self._running = False
        self._readers = []
        self._data_lock = threading.Lock()
              
        self._latest_data = { 
            "adxl1": 0.0,
            "adxl2": 0.0,
            "load_cell": 0.0
        }
        
        #paths
        self._base_dir = None
        self._image_dir = None
        self._csv_path = None
        self._snapshot_url = "http://127.0.0.1:8080/?action=snapshot"

    def on_after_startup(self):
        # files
        self._base_dir = os.path.join(self.get_plugin_data_folder(), "data")
        self._image_dir = os.path.join(self._base_dir, "images")
        self._csv_path = os.path.join(self._base_dir, "log.csv")
        
        if not os.path.exists(self._image_dir): os.makedirs(self._image_dir)
        
        # csv header
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                "frame_id", "timestamp",
                "adxl1_x", "adxl1_y", "adxl1_z", 
                "adxl2_x", "adxl2_y", "adxl2_z",
                "load_cell",
                "filename"
                ])

        self._running = True

        # auto find ports
        potential_ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
        
        self._logger.info(f"Found USB ports: {potential_ports}")

        for port in potential_ports:
            reader = SerialSensorReader(
                port=port,
                shared_data=self._latest_data,
                lock=self._data_lock,
                logger=self._logger
            )
            reader.start()
            self._readers.append(reader)

        # camera thread
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()

    def on_shutdown(self):
        self._running = False
        for reader in self._readers:
            reader.stop()

    def _camera_loop(self):
        while self._running:
            start_time = time.time()
            
            # data snapshot
            current_values = {}
            with self._data_lock:
                current_values = self._latest_data.copy()

            self._check_and_capture(current_values)

            # calculate sleep needed for 2 second interval
            elapsed = time.time() - start_time
            sleep_time = 2.0 - elapsed
            if sleep_time > 0: time.sleep(sleep_time)

    def _check_and_capture(self, values):
        # check printer state
        printer_data = self._printer.get_current_data()
        state = "Printing"#printer_data["state"]["text"]
        if state in ["Printing"]:
            self._save_snapshot(values)

    def _save_snapshot(self, values):
        try:
            resp = requests.get(self._snapshot_url, timeout=2.0, stream=True)
            if resp.status_code != 200: return
        except: return

        timestamp = time.time()
        frame_id = int(timestamp * 1000)
        filename = f"{frame_id}.jpg"
        full_path = os.path.join(self._image_dir, filename)

        with open(full_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4096):
                f.write(chunk)

        try:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                
                a1 = values.get("adxl1", {"x": 0, "y": 0, "z": 0}) 
                a2 = values.get("adxl2", {"x": 0, "y": 0, "z": 0})
                row = [
                    frame_id,
                    "{:.6f}".format(timestamp),
                    a1["x"], a1["y"], a1["z"],  
                    a2["x"], a2["y"], a2["z"],
                    values.get("load_cell", 0),
                    filename
                ]

                writer.writerow(row)
                self._logger.info(f"output: {row}")

        except Exception:
            pass

__plugin_name__ = "Data Collector"
__plugin_pythoncompat__ = ">=3.7,<4"
__plugin_implementation__ = DataCollectorPlugin()