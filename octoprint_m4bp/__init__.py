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
import numpy as np
import traceback

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import collections
except ImportError:
    pass

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class SerialSensorReader(threading.Thread):
    def __init__(self, port, shared_data, lock, logger):
        super().__init__(daemon=True)
        self._port = port
        self._shared_data = shared_data
        self._lock = lock
        self._logger = logger
        self._running = True

    def run(self):
        self._logger.debug(f"Checking port {self._port}...")
        ser = None

        try:
            ser = serial.Serial(self._port, 115200, timeout=1)
            time.sleep(2) 

            is_sensor = False
            handshake_start = time.time()
            
            while time.time() - handshake_start < 3.0:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('{') and '"id":' in line:
                        is_sensor = True
                        self._logger.debug(f"Sensor verified on {self._port}!")
                        break
                time.sleep(0.05)
            if not is_sensor:

                self._logger.debug(f"Port {self._port} is not a sensor. Releasing it for OctoPrint.")
                ser.close()
                return 
                
        except Exception as e:
            self._logger.warning(f"Could not open {self._port}: {e}")
            if ser: ser.close()
            return

        while self._running:
            try:
                # read parse
                if ser.in_waiting > 0:
                    try:
                        raw_data = ser.read_all().decode('utf-8', errors='ignore')
                        lines = raw_data.strip().split('\n')
                        if not lines: continue
                        
                        last_line = lines[-1]

                        if last_line.startswith('{') and last_line.endswith('}'):
                            data = json.loads(last_line)

                            current_time = time.time()
                            sensor_id = data.get("id")

                            with self._lock:
                                if sensor_id == "adxl1":
                                    self._shared_data["adxl1"].append({
                                        "ts": current_time, 
                                        "x": float(data.get("x", 0.0)), 
                                        "y": float(data.get("y", 0.0)), 
                                        "z": float(data.get("z", 0.0))
                                    })                                    
                                elif sensor_id == "adxl2":
                                    self._shared_data["adxl2"].append({
                                        "ts": current_time, 
                                        "x": float(data.get("x", 0.0)), 
                                        "y": float(data.get("y", 0.0)), 
                                        "z": float(data.get("z", 0.0))
                                    })                                    
                                elif sensor_id == "load":
                                    self._shared_data["load_cell"].append({
                                        "ts": current_time, 
                                        "val": float(data.get("val", 0.0))
                                    })                                    
                    except Exception as e:
                        self._logger.warning(f"Error processing sensor data: {e}")
                
                time.sleep(0.01) # important 

            except Exception as e:
                self._logger.warning(f"Lost connection to sensor on {self._port}: {e}")
                break 

        if ser: ser.close()

    def stop(self):
        self._running = False

class DataCollectorPlugin(
    octoprint.plugin.StartupPlugin,
    octoprint.plugin.ShutdownPlugin,
    octoprint.plugin.SettingsPlugin,
    octoprint.plugin.TemplatePlugin,
    octoprint.plugin.EventHandlerPlugin,
    octoprint.plugin.AssetPlugin
):

    def __init__(self):
        self._running = False
        self._readers = []
        self._data_lock = threading.Lock()

        self._latest_data = { 
            "adxl1": [],
            "adxl2": [],
            "load_cell": []
        }

        self._last_load_avg = 0.0
        self._last_capture_time = time.time()

        self._latest_correction = "0"
        
        #paths
        self._base_dir = None
        self._image_dir = None
        self._csv_path = None
        self._snapshot_url = "http://127.0.0.1:8080/?action=snapshot"

        self._yolo_model = None
        self._cat_model = None

        # Sliding window consensus: store last 5 raw predictions
        self._prediction_window = collections.deque(maxlen=5)
        self._last_confirmed_prediction = "None"

    def get_settings_defaults(self):
        return {
            "yolo_model_path": "",
            "xgb_model_path": ""
        }

    def on_settings_save(self, data):
        octoprint.plugin.SettingsPlugin.on_settings_save(self, data)
        self._load_models()

    def get_template_configs(self):
        return [
            dict(type="tab", name="AI m4bp", custom_bindings=True)
        ]

    def get_assets(self):
        return {
            "js": ["js/m4bp.js"],
            "css": ["css/m4bp.css"]
        }

    def _load_models(self):
        yolo_path = self._settings.get(["yolo_model_path"])
        xgb_path = self._settings.get(["xgb_model_path"])

        if yolo_path and os.path.exists(yolo_path) and YOLO is not None:
            self._logger.info(f"Loading YOLO model from {yolo_path}")
            try:
                self._yolo_model = YOLO(yolo_path)
            except Exception as e:
                self._logger.error(f"Failed to load YOLO model: {e}")
        
        if xgb_path and os.path.exists(xgb_path) and xgb is not None:
            self._logger.info(f"Loading XGBoost model from {xgb_path}")
            try:
                self._cat_model = xgb.XGBClassifier()
                self._cat_model.load_model(xgb_path)
                self._logger.info("XGBoost model loaded successfully.")
            except Exception as e:
                self._logger.error(f"Failed to load XGBoost model: {e}")

    def _get_consensus_prediction(self, raw_pred):
        """Sliding Window Consensus: only confirm prediction if 4/5 recent ticks agree."""
        self._prediction_window.append(str(raw_pred))

        if len(self._prediction_window) < 5:
            return "Warming up..."

        # Count how many of the last 5 ticks agree on any one class
        from collections import Counter
        counts = Counter(self._prediction_window)
        most_common_pred, count = counts.most_common(1)[0]

        if count >= 4:
            self._last_confirmed_prediction = most_common_pred

        return self._last_confirmed_prediction


    def on_after_startup(self):
        # files
        self._base_dir = os.path.join(self.get_plugin_data_folder(), "data")
        if not os.path.exists(self._base_dir): 
            os.makedirs(self._base_dir)

        self._load_models()

        self._running = True

        # auto find ports
        potential_ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')

        self._logger.debug(f"Found USB ports: {potential_ports}")

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
    
    def on_event(self, event, payload):
        if payload is None:
            return
        if event == "PrintStarted":
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            raw_name = payload.get("name", "unknown_print")            
            clean_name, _ = os.path.splitext(raw_name)
            print_name = clean_name.replace(" ", "_")
            
            current_print_dir = os.path.join(self._base_dir, f"{timestamp}_{print_name}")
            self._image_dir = os.path.join(current_print_dir, "images")
            self._csv_path = os.path.join(current_print_dir, "log.csv")
            
            os.makedirs(self._image_dir, exist_ok=True)

            with self._data_lock:
                self._latest_data["adxl1"].clear()
                self._latest_data["adxl2"].clear()
                self._latest_data["load_cell"].clear()
                self._latest_correction = "0"

                self._last_load_avg = None 
                self._last_capture_time = time.time()
            
            self._logger.debug("Buffers flushed for new print.")
            
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["timestamp", "relative_img_path"]
                for s in ["a1", "a2"]:
                    for axis in ["x", "y", "z"]:
                        headers += [f"{s}_{axis}_rms", f"{s}_{axis}_p2p", f"{s}_{axis}_std"]
                headers += ["load_avg", "load_slope", "oextrusion_conf", "uextrusion_conf", "string_conf", "spag_conf", "prediction", "correction"]
                writer.writerow(headers)
                
            self._logger.debug(f"--- NEW PRINT STARTED: Saving data to {current_print_dir} ---")

        elif event in ["PrintDone", "PrintFailed", "PrintCancelled"]:
            self._logger.debug(f"--- {event}: Stopped logging data ---")

    def on_shutdown(self):
        self._running = False
        for reader in self._readers:
            reader.stop()

    def _camera_loop(self):
        while self._running:
            start_time = time.time()
            
            self._check_and_capture()

            # calculate sleep needed for 2 second interval
            elapsed = time.time() - start_time
            sleep_time = 2.0 - elapsed
            if sleep_time > 0: time.sleep(sleep_time)

    def _check_and_capture(self):
        # check printer state
        printer_data = self._printer.get_current_data()
        state = printer_data["state"]["text"]
        if state in ["Printing"] and self._csv_path is not None:
            self._save_snapshot()

    def _save_snapshot(self):
        now_ms = int(time.time() * 1000)
        now_sec = now_ms / 1000.0

        time_diff = now_sec - self._last_capture_time

        img_filename = "{}.jpg".format(now_ms)
        relative_img_path = os.path.join("images", img_filename)

        #get image snapshot
        try:
            resp = requests.get(self._snapshot_url, timeout=2.0)
            if resp.status_code == 200:
                full_path = os.path.join(self._image_dir, img_filename)
                with open(full_path, "wb") as f:
                    f.write(resp.content)
            else:
                self._logger.error("Camera error: {}".format(resp.status_code))
        except Exception as e:
            self._logger.error("Camera capture failed: {}".format(e))
            full_path = None

        oextrusion_conf = 0.0
        uextrusion_conf = 0.0
        string_conf = 0.0
        spag_conf = 0.0

        if full_path and getattr(self, "_yolo_model", None) is not None:
            try:
                results = self._yolo_model(full_path, verbose=False)
                if len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        classes = boxes.cls.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        names = results[0].names 
                        
                        for cls_id, conf in zip(classes, confs):
                            cls_name = names[int(cls_id)].lower()
                            if "over" in cls_name or "oextrusion" in cls_name:
                                oextrusion_conf = max(oextrusion_conf, float(conf))
                            elif "under" in cls_name or "uextrusion" in cls_name:
                                uextrusion_conf = max(uextrusion_conf, float(conf))
                            elif "string" in cls_name:
                                string_conf = max(string_conf, float(conf))
                            elif "spagh" in cls_name:
                                spag_conf = max(spag_conf, float(conf))
            except Exception as e:
                self._logger.error("YOLO inference failed: {}".format(e))

        row_features = []
        with self._data_lock:
            current_adxl1 = self._latest_data["adxl1"][:]
            current_adxl2 = self._latest_data["adxl2"][:]
            current_load = self._latest_data["load_cell"][:]

            current_correction = self._latest_correction
            
            self._latest_data["adxl1"].clear()
            self._latest_data["adxl2"].clear()
            self._latest_data["load_cell"].clear()

        adxl_data_map = {"adxl1": current_adxl1, "adxl2": current_adxl2}
        
        for s_id in ["adxl1", "adxl2"]:
            data_points = adxl_data_map[s_id]
            if len(data_points) > 50:
                for axis in ['x', 'y', 'z']:
                    arr = np.array([d[axis] for d in data_points])
                    
                    # calculate adxl features
                    rms = round(np.sqrt(np.mean(arr**2)), 4)
                    p2p = np.ptp(arr) # Peak-to-Peak
                    std = round(np.std(arr), 4)
                    row_features += [rms, p2p, std]
            else:
                row_features += [np.nan] * 9 # default to Nan if not enough data
        
        #load cell features
        current_load_avg = np.nan
        load_slope = np.nan
        
        if len(current_load) > 2:
            vals = np.array([d['val'] for d in current_load])
            current_load_avg = np.mean(vals)
            if self._last_load_avg is not None and not np.isnan(self._last_load_avg) and time_diff > 0:
                load_slope = (current_load_avg - self._last_load_avg) / time_diff
            else:
                load_slope = np.nan

        else:
            current_load_avg = np.nan
            load_slope = np.nan
            
        row_features += [round(current_load_avg, 4), round(load_slope, 4)]
        self._last_load_avg = current_load_avg

        yolo_features = [oextrusion_conf, uextrusion_conf, string_conf, spag_conf]
        
        raw_prediction = "None"
        if getattr(self, "_cat_model", None) is not None:
            try:
                # XGBoost input: adxl (18) + load (2) + yolo (4) = 24 features
                # Matches training data: drops timestamp, relative_img_path, correction
                features_array = np.array([row_features + yolo_features])
                preds = self._cat_model.predict(features_array)
                raw_prediction = str(preds[0])
            except Exception as e:
                self._logger.error("XGBoost inference failed: {}".format(traceback.format_exc()))

        # Apply sliding window consensus to eliminate single-tick false alarms
        final_prediction = self._get_consensus_prediction(raw_prediction)

        row_features += yolo_features + [final_prediction, current_correction]

        # Send websocket update for UI visualization
        try:
            self._plugin_manager.send_plugin_message(self._identifier, dict(
                type="live_data",
                timestamp=now_ms,
                prediction=final_prediction,
                raw_prediction=raw_prediction,
                yolo=dict(
                    oextrusion=oextrusion_conf,
                    uextrusion=uextrusion_conf,
                    stringing=string_conf,
                    spaghetti=spag_conf
                ),
                load_avg=current_load_avg
            ))
        except Exception as e:
            self._logger.debug("Failed to send websocket message: {}".format(e))

        try:
            if self._csv_path:
                with open(self._csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    row = [now_ms, relative_img_path] + row_features
                    writer.writerow(row)
                    self._logger.debug("Row saved:[ {} ]".format(row))
                    self._logger.debug("number of data points used: a1: {}, a2: {}, load: {}".format(len(current_adxl1), len(current_adxl2), len(current_load)))
        except Exception as e:
            self._logger.error("CSV Write Failed: {}".format(e))
        
        self._last_capture_time = now_sec

    def hook_gcode_received(self, comm_instance, line, *args, **kwargs):
        if "M998" in line and "S" in line:
            _, _, after_s = line.partition("S")
        
            extracted_number = after_s.strip()[:1]
            
            with self._data_lock:
                self._latest_correction = extracted_number
        
        return line

__plugin_name__ = "m4bp"
__plugin_pythoncompat__ = ">=3.7,<4"
__plugin_implementation__ = DataCollectorPlugin()

__plugin_hooks__ = {
    "octoprint.comm.protocol.gcode.received": __plugin_implementation__.hook_gcode_received
}