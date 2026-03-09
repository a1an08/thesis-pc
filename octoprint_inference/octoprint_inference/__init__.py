# coding=utf-8
from __future__ import absolute_import

import octoprint.plugin
import threading
import time
import os
import json
import glob
import serial
import numpy as np
from collections import deque
from flask import Flask, jsonify, render_template_string, Response

# ─────────────────────────────────────────────
#  Class label map (raw model output → label)
# ─────────────────────────────────────────────
CLASS_LABELS = {
    0: "Normal Printing",
    1: "Low Flowrate",
    2: "High Flowrate",
    3: "Firmware Retraction",
    5: "Temp Too High",
}

# GCode corrections per error class
CORRECTIONS = {
    1: ("M221 S110",        "Increased Flow to 110%"),
    2: ("M221 S90",         "Decreased Flow to 90%"),
    3: ("M207 S2 F2400",    "Reset Retraction Settings"),
    5: ("M104 S200",        "Reduced Hotend Temp to 200°C"),
}

# How many consecutive matching predictions before triggering a correction
PERSISTENCE_THRESHOLD = 3

# Exact order of base features as produced by the training script
BASE_FEATURE_NAMES = [
    "a1_x_rms", "a1_x_p2p", "a1_x_std",
    "a1_y_rms", "a1_y_p2p", "a1_y_std",
    "a1_z_rms", "a1_z_p2p", "a1_z_std",
    "a2_x_rms", "a2_x_p2p", "a2_x_std",
    "a2_y_rms", "a2_y_p2p", "a2_y_std",
    "a2_z_rms", "a2_z_p2p", "a2_z_std",
    "load_avg", "load_slope"
]

ALL_FEATURE_NAMES = []
ALL_FEATURE_NAMES.extend(BASE_FEATURE_NAMES)
for k in BASE_FEATURE_NAMES:
    ALL_FEATURE_NAMES.append(f"{k}_lag1")
    ALL_FEATURE_NAMES.append(f"{k}_roll_mean")

# ─────────────────────────────────────────────
#  Serial Sensor Reader Thread
# ─────────────────────────────────────────────
class SerialSensorReader(threading.Thread):
    def __init__(self, port, shared_data, lock, logger):
        super().__init__(daemon=True)
        self._port = port
        self._shared_data = shared_data
        self._lock = lock
        self._logger = logger
        self._running = True

    def run(self):
        self._logger.debug(f"[SENSOR DETECT] Checking port {self._port} ...")
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
                        self._logger.debug(f"[SENSOR DETECT] ✓ Sensor verified on {self._port}!")
                        break
                time.sleep(0.05)

            if not is_sensor:
                self._logger.debug(f"[SENSOR DETECT] ✗ {self._port} is NOT a sensor → releasing for OctoPrint/printer.")
                ser.close()
                return

        except Exception as e:
            self._logger.warning(f"[SENSOR DETECT] Could not open {self._port}: {e}")
            if ser:
                ser.close()
            return

        self._logger.debug(f"[SENSOR DETECT] Starting data read loop on {self._port}")
        while self._running:
            try:
                if ser.in_waiting > 0:
                    try:
                        raw_data = ser.read_all().decode('utf-8', errors='ignore')
                        lines = raw_data.strip().split('\n')
                        if not lines:
                            continue
                        last_line = lines[-1].strip()
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
                                        "z": float(data.get("z", 0.0)),
                                    })
                                elif sensor_id == "adxl2":
                                    self._shared_data["adxl2"].append({
                                        "ts": current_time,
                                        "x": float(data.get("x", 0.0)),
                                        "y": float(data.get("y", 0.0)),
                                        "z": float(data.get("z", 0.0)),
                                    })
                                elif sensor_id == "load":
                                    self._shared_data["load_cell"].append({
                                        "ts": current_time,
                                        "val": float(data.get("val", 0.0)),
                                    })
                    except Exception as e:
                        self._logger.warning(f"[SENSOR DATA] Parse error on {self._port}: {e}")
                time.sleep(0.01)
            except Exception as e:
                self._logger.warning(f"[SENSOR DATA] Lost connection on {self._port}: {e}")
                break

        if ser:
            ser.close()

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────
#  Feature Extraction  (mirrors training script)
# ─────────────────────────────────────────────
class FeatureBuffer:
    """
    Keeps a history of base-feature rows so we can compute lag-1 and
    rolling-mean-5 exactly as the training script does.
    """
    def __init__(self, window=5):
        self._window = window
        self._history = deque(maxlen=window + 1)  # need at least 2 for lag

    def push(self, row: dict):
        self._history.append(row)

    def get_engineered(self):
        """Return a flat feature vector with lag1 + roll_mean interleaved, or None if not ready."""
        if len(self._history) < 2:
            return None
        
        current = self._history[-1]
        prev    = self._history[-2]
        history_list = list(self._history)

        feature_vec = []
        
        # 1. Base features (20)
        for k in BASE_FEATURE_NAMES:
            feature_vec.append(current.get(k, 0.0))

        # 2. Engineered features (40): for each base column, add lag1 then roll_mean
        for k in BASE_FEATURE_NAMES:
            # lag1
            feature_vec.append(prev.get(k, 0.0))
            # roll_mean (window 5)
            # history_list has up to window+1 items. We need the last window items.
            vals = [h.get(k, 0.0) for h in history_list]
            effective_window = vals[-self._window:]
            feature_vec.append(np.mean(effective_window))

        return feature_vec


def extract_base_features(adxl1_buf, adxl2_buf, load_buf, last_load_avg, time_diff):
    """Compute the 20 base features from raw sensor buffers."""
    row = {}
    for s_id, buf in [("a1", adxl1_buf), ("a2", adxl2_buf)]:
        if len(buf) > 50:
            for axis in ['x', 'y', 'z']:
                arr = np.array([d[axis] for d in buf])
                row[f"{s_id}_{axis}_rms"]  = float(np.sqrt(np.mean(arr ** 2)))
                row[f"{s_id}_{axis}_p2p"]  = float(np.ptp(arr))
                row[f"{s_id}_{axis}_std"]  = float(np.std(arr))
        else:
            for axis in ['x', 'y', 'z']:
                row[f"{s_id}_{axis}_rms"]  = np.nan
                row[f"{s_id}_{axis}_p2p"]  = np.nan
                row[f"{s_id}_{axis}_std"]  = np.nan

    if len(load_buf) > 2:
        vals = np.array([d['val'] for d in load_buf])
        current_avg = float(np.mean(vals))
        row["load_avg"] = current_avg
        if last_load_avg is not None and not np.isnan(last_load_avg) and time_diff > 0:
            row["load_slope"] = float((current_avg - last_load_avg) / time_diff)
        else:
            row["load_slope"] = np.nan
    else:
        row["load_avg"]   = np.nan
        row["load_slope"] = np.nan

    return row


# ─────────────────────────────────────────────
#  The OctoPrint Plugin
# ─────────────────────────────────────────────
class PrintInferencePlugin(
    octoprint.plugin.StartupPlugin,
    octoprint.plugin.ShutdownPlugin,
    octoprint.plugin.SettingsPlugin,
    octoprint.plugin.TemplatePlugin,
    octoprint.plugin.BlueprintPlugin,
    octoprint.plugin.EventHandlerPlugin,
    octoprint.plugin.AssetPlugin,
):

    def is_blueprint_csrf_protected(self):
        return False

    def __init__(self):
        self._running = False
        self._readers = []
        self._data_lock = threading.Lock()

        # Sensor ring-buffers
        self._latest_data = {"adxl1": [], "adxl2": [], "load_cell": []}

        # For load slope calculation
        self._last_load_avg   = None
        self._last_capture_ts = time.time()

        # Model & inference
        self._model         = None
        self._feature_buf   = FeatureBuffer(window=5)

        # Live state (exposed to UI and Flask)
        self._current_inference    = -1   # -1 = Idle/No model
        self._inference_label      = "Idle"
        self._current_adxl1        = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._current_adxl2        = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._current_load         = 0.0

        # Correction engine
        self._correction_enabled   = False
        self._persistence_counter  = 0
        self._last_correction_text = "None"
        self._last_correction_gcode= "None"
        self._total_corrections    = 0
        self._corrections_log      = []   # list of {time, label, gcode}

        # Flask web server refs
        self._flask_app      = None
        self._flask_thread   = None
        self._api_thread     = None

    # ── Settings ────────────────────────────────
    def get_settings_defaults(self):
        return {
            "correction_enabled":   False,
            "persistence_threshold": PERSISTENCE_THRESHOLD,
        }

    def on_settings_save(self, data):
        octoprint.plugin.SettingsPlugin.on_settings_save(self, data)
        self._correction_enabled = self._settings.get_boolean(["correction_enabled"])

    # ── Template ────────────────────────────────
    def get_template_configs(self):
        return [
            {
                "type": "tab",
                "name": "Live Inference",
                "template": "octoprint_inference_tab.jinja2",
                "custom_bindings": True,
            }
        ]

    # ── Assets ──────────────────────────────────
    def get_assets(self):
        return {
            "js": ["js/octoprint_inference.js"],
            "css": ["css/octoprint_inference.css"],
        }

    # ── Startup ─────────────────────────────────
    def on_after_startup(self):
        self._running = True
        self._correction_enabled = self._settings.get_boolean(["correction_enabled"])

        # Load XGBoost model
        model_path = os.path.join(self.get_plugin_data_folder(), "xgboost_vibration_model.json")
        self._logger.debug(f"[INFERENCE] Looking for model at: {model_path}")
        if os.path.exists(model_path):
            try:
                import xgboost as xgb
                self._model = xgb.XGBClassifier()
                self._model.load_model(model_path)
                self._logger.debug("[INFERENCE] ✓ XGBoost model loaded successfully!")
            except Exception as e:
                self._logger.error(f"[INFERENCE] ✗ Failed to load model: {e}")
        else:
            self._logger.warning(f"[INFERENCE] ✗ Model file not found at {model_path}. Running without inference.")

        # Discover and initialize sensor ports
        potential_ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
        self._logger.debug(f"[SENSOR DETECT] Found USB ports: {potential_ports}")
        for port in potential_ports:
            reader = SerialSensorReader(
                port=port,
                shared_data=self._latest_data,
                lock=self._data_lock,
                logger=self._logger,
            )
            reader.start()
            self._readers.append(reader)

        # Inference thread
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

        # Flask dashboard (port 6969) + API (port 6767) in separate threads
        self._start_web_servers()

    def on_shutdown(self):
        self._running = False
        for r in self._readers:
            r.stop()

    # ── Event Handler ───────────────────────────
    def on_event(self, event, payload):
        if event in ["PrintStarted", "PrintResumed"]:
            self._logger.debug(f"[INFERENCE] Print started/resumed — inference active.")
            with self._data_lock:
                self._latest_data["adxl1"].clear()
                self._latest_data["adxl2"].clear()
                self._latest_data["load_cell"].clear()
            self._feature_buf   = FeatureBuffer(window=5)
            self._persistence_counter = 0
            self._last_load_avg = None
            self._last_capture_ts = time.time()

        elif event in ["PrintDone", "PrintFailed", "PrintCancelled", "PrintPaused"]:
            self._logger.debug(f"[INFERENCE] Print ended/paused — inference paused.")
            self._inference_label   = "Idle"
            self._current_inference = -1
            self._persistence_counter = 0

    # ── Inference Loop ───────────────────────────
    def _inference_loop(self):
        while self._running:
            try:
                state_data = self._printer.get_current_data()
                state = state_data.get("state", {}).get("text", "")

                if state == "Printing" and self._model is not None:
                    self._run_inference()
                else:
                    # Still update current sensor displayables even when not printing
                    self._update_display_values()

            except Exception as e:
                self._logger.warning(f"[INFERENCE] Loop error: {e}")

            time.sleep(2.0)

    def _update_display_values(self):
        """Snapshot the latest sensor reading for display (no inference)."""
        with self._data_lock:
            a1 = list(self._latest_data["adxl1"])
            a2 = list(self._latest_data["adxl2"])
            lc = list(self._latest_data["load_cell"])

        if a1:
            last = a1[-1]
            self._current_adxl1 = {"x": last["x"], "y": last["y"], "z": last["z"]}
        if a2:
            last = a2[-1]
            self._current_adxl2 = {"x": last["x"], "y": last["y"], "z": last["z"]}
        if lc:
            self._current_load = lc[-1]["val"]

    def _run_inference(self):
        now = time.time()

        with self._data_lock:
            adxl1_buf = list(self._latest_data["adxl1"])
            adxl2_buf = list(self._latest_data["adxl2"])
            load_buf  = list(self._latest_data["load_cell"])
            self._latest_data["adxl1"].clear()
            self._latest_data["adxl2"].clear()
            self._latest_data["load_cell"].clear()

        time_diff = now - self._last_capture_ts
        self._last_capture_ts = now

        # Update display values from captured buffers
        if adxl1_buf:
            last = adxl1_buf[-1]
            self._current_adxl1 = {"x": last["x"], "y": last["y"], "z": last["z"]}
        if adxl2_buf:
            last = adxl2_buf[-1]
            self._current_adxl2 = {"x": last["x"], "y": last["y"], "z": last["z"]}
        if load_buf:
            self._current_load = np.mean([d["val"] for d in load_buf])

        # Extract base features
        base_row = extract_base_features(adxl1_buf, adxl2_buf, load_buf,
                                         self._last_load_avg, time_diff)

        if "load_avg" in base_row and not np.isnan(base_row["load_avg"]):
            self._last_load_avg = base_row["load_avg"]

        # Check if we have enough valid features
        if all(np.isnan(v) for v in base_row.values()):
            self._logger.debug("[INFERENCE] Insufficient sensor data — skipping inference.")
            return

        self._feature_buf.push(base_row)
        feat_vec = self._feature_buf.get_engineered()
        if feat_vec is None:
            self._logger.debug("[INFERENCE] Feature buffer warming up — need 2+ windows.")
            return

        # Fill NaN with 0 (same bfill/ffill effect at inference time)
        feat_arr = np.array(feat_vec, dtype=float)
        feat_arr = np.nan_to_num(feat_arr, nan=0.0)

        try:
            import xgboost as xgb
            # Provide feature names to DMatrix so XGBoost doesn't complain about mismatch
            dmatrix = xgb.DMatrix(feat_arr.reshape(1, -1), feature_names=ALL_FEATURE_NAMES)
            proba   = self._model.get_booster().predict(dmatrix)
            # proba shape: (1, num_classes) with softprob
            pred_idx = int(np.argmax(proba, axis=1)[0])

            # Map encoded index back to raw class (model was label-encoded 0,1,2,3,5 → 0,1,2,3,4)
            # LabelEncoder sorts numerically: [0,1,2,3,5] → internal [0,1,2,3,4]
            raw_class_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}
            raw_class = raw_class_map.get(pred_idx, pred_idx)

            self._current_inference = raw_class
            self._inference_label   = CLASS_LABELS.get(raw_class, f"Class {raw_class}")

            self._logger.debug(
                f"[INFERENCE] Prediction: {self._inference_label} "
                f"(encoded={pred_idx}, raw_class={raw_class})"
            )

            # Correction engine
            self._check_and_correct(raw_class)

        except Exception as e:
            self._logger.error(f"[INFERENCE] Prediction error: {e}")

    def _check_and_correct(self, raw_class):
        if raw_class == 0:
            # Normal — reset counter
            self._persistence_counter = 0
            return

        # Error class detected
        self._persistence_counter += 1
        threshold = self._settings.get_int(["persistence_threshold"]) or PERSISTENCE_THRESHOLD

        self._logger.debug(
            f"[CORRECTION] Error '{CLASS_LABELS.get(raw_class)}' "
            f"count {self._persistence_counter}/{threshold}"
        )

        if self._persistence_counter >= threshold and self._correction_enabled:
            self._send_correction(raw_class)
            self._persistence_counter = 0  # reset after firing

    def _send_correction(self, raw_class):
        if raw_class not in CORRECTIONS:
            return
        gcode, desc = CORRECTIONS[raw_class]
        try:
            self._printer.commands([gcode])
            self._last_correction_text  = desc
            self._last_correction_gcode = gcode
            self._total_corrections    += 1
            entry = {
                "time":  time.strftime("%H:%M:%S"),
                "label": CLASS_LABELS.get(raw_class, str(raw_class)),
                "desc":  desc,
                "gcode": gcode,
            }
            self._corrections_log.append(entry)
            if len(self._corrections_log) > 50:
                self._corrections_log = self._corrections_log[-50:]
            self._logger.debug(f"[CORRECTION] ✓ Sent GCode: {gcode} ({desc})")
        except Exception as e:
            self._logger.error(f"[CORRECTION] Failed to send GCode {gcode}: {e}")

    # ── OctoPrint API Blueprint ──────────────────
    @octoprint.plugin.BlueprintPlugin.route("/data", methods=["GET"])
    def api_data(self):
        return jsonify(self._build_state_dict())

    @octoprint.plugin.BlueprintPlugin.route("/settings", methods=["POST"])
    def api_settings(self):
        from flask import request as flask_request
        data = flask_request.get_json(silent=True) or {}
        return jsonify(self._handle_settings_update(data))

    def _handle_settings_update(self, data):
        if "correction_enabled" in data:
            enabled = bool(data["correction_enabled"])
            self._correction_enabled = enabled
            self._settings.set_boolean(["correction_enabled"], enabled)
            self._settings.save()
            self._logger.debug(f"[CORRECTION] Correction {'ENABLED' if enabled else 'DISABLED'} via API")
        return {"correction_enabled": self._correction_enabled}

    def _build_state_dict(self):
        threshold = self._settings.get_int(["persistence_threshold"]) or PERSISTENCE_THRESHOLD
        return {
            "adxl1":              dict(self._current_adxl1),
            "adxl2":              dict(self._current_adxl2),
            "load_cell":          round(float(self._current_load), 4),
            "inference_class":    self._current_inference,
            "inference_label":    self._inference_label,
            "correction_enabled": self._correction_enabled,
            "persistence_counter":self._persistence_counter,
            "persistence_threshold": threshold,
            "last_correction":    self._last_correction_text,
            "last_correction_gcode": self._last_correction_gcode,
            "total_corrections":  self._total_corrections,
            "corrections_log":    self._corrections_log[-10:],
            "timestamp":          time.time(),
        }

    # ── Flask Web Servers ────────────────────────
    def _start_web_servers(self):
        plugin_ref = self

        # ── Dashboard on 6969 ──
        dashboard_app = Flask("octoprint_inference_dashboard")

        @dashboard_app.route("/")
        def dashboard_index():
            return render_template_string(DASHBOARD_HTML)

        @dashboard_app.route("/data")
        def dashboard_data():
            return jsonify(plugin_ref._build_state_dict())

        @dashboard_app.route("/settings", methods=["POST"])
        def dashboard_settings():
            from flask import request as flask_request
            data = flask_request.get_json(silent=True) or {}
            return jsonify(plugin_ref._handle_settings_update(data))

        def run_dashboard():
            self._logger.debug("[WEB] Starting live dashboard on port 6969")
            dashboard_app.run(host="0.0.0.0", port=6969, debug=False, use_reloader=False)

        self._flask_thread = threading.Thread(target=run_dashboard, daemon=True)
        self._flask_thread.start()

        # ── JSON API on 6767 ──
        api_app = Flask("octoprint_inference_api")

        @api_app.route("/data")
        def api_json():
            return jsonify(plugin_ref._build_state_dict())

        @api_app.route("/settings", methods=["POST"])
        def api_settings_internal():
            from flask import request as flask_request
            data = flask_request.get_json(silent=True) or {}
            return jsonify(plugin_ref._handle_settings_update(data))

        def run_api():
            self._logger.debug("[WEB] Starting JSON API on port 6767")
            api_app.run(host="0.0.0.0", port=6767, debug=False, use_reloader=False)

        self._api_thread = threading.Thread(target=run_api, daemon=True)
        self._api_thread.start()


# ─────────────────────────────────────────────
#  Dashboard HTML (embedded, no external files needed)
# ─────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3D Print Live Inference Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:      #0a0e1a;
    --surface: #111827;
    --card:    #1c2336;
    --border:  #2a3550;
    --accent:  #3b82f6;
    --green:   #22c55e;
    --yellow:  #f59e0b;
    --red:     #ef4444;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --font:    'Inter', sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); min-height: 100vh; }
  header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1e1b4b 100%);
    padding: 1.2rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 24px #0006;
    border-bottom: 1px solid #3b82f640;
  }
  header h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -.5px; }
  header h1 span { color: #60a5fa; }
  #conn-status { font-size: .8rem; padding: .3rem .8rem; border-radius: 99px;
    background: #22c55e22; color: var(--green); border: 1px solid var(--green); }
  .main { max-width: 1400px; margin: 0 auto; padding: 2rem 1.5rem; display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1.5rem; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.5rem;
    box-shadow: 0 4px 20px #0004;
    transition: box-shadow .2s;
  }
  .card:hover { box-shadow: 0 8px 32px #0008; }
  .card-title { font-size: .75rem; letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--muted); margin-bottom: 1rem; }
  canvas { max-height: 180px; }
  .inference-badge {
    display: inline-block; padding: .5rem 1.2rem; border-radius: 99px;
    font-size: 1.1rem; font-weight: 700; letter-spacing: .5px;
    margin-top: .5rem;
    transition: background .3s, color .3s;
  }
  .badge-idle    { background:#1e293b; color:#94a3b8; border:1px solid #334155; }
  .badge-normal  { background:#052e16; color:var(--green); border:1px solid var(--green); }
  .badge-warning { background:#451a03; color:var(--yellow); border:1px solid var(--yellow); }
  .badge-error   { background:#450a0a; color:var(--red); border:1px solid var(--red); }
  .kv-row { display:flex; justify-content:space-between; align-items:center;
    padding:.45rem 0; border-bottom:1px solid var(--border); }
  .kv-row:last-child { border-bottom:none; }
  .kv-key { color:var(--muted); font-size:.85rem; }
  .kv-val { font-weight:600; font-size:.95rem; }
  .toggle-row { display:flex; align-items:center; gap:1rem; margin-top:.5rem; }
  .toggle { position:relative; display:inline-block; width:52px; height:28px; }
  .toggle input { opacity:0; width:0; height:0; }
  .slider {
    position:absolute; cursor:pointer; inset:0;
    background:#334155; border-radius:28px; transition:.3s;
  }
  .slider::before {
    content:""; position:absolute;
    height:20px; width:20px; left:4px; bottom:4px;
    background:white; border-radius:50%; transition:.3s;
  }
  input:checked + .slider { background: var(--accent); }
  input:checked + .slider::before { transform: translateX(24px); }
  .corrections-list { max-height:160px; overflow-y:auto; }
  .correction-item {
    background:#1e293b; border-radius:8px; padding:.5rem .75rem;
    margin-bottom:.4rem; font-size:.8rem;
    border-left: 3px solid var(--accent);
  }
  .correction-item .time { color:var(--muted); font-size:.7rem; }
  .correction-item .desc { color: var(--yellow); font-weight:600; }
  .correction-item .gcode { color:#60a5fa; font-family:monospace; }
  .counter-big { font-size:2.5rem; font-weight:900; color: var(--accent); }
  .counter-label { color:var(--muted); font-size:.85rem; }
  .full-width { grid-column: 1 / -1; }
  ::-webkit-scrollbar { width:4px; }
  ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
</style>
</head>
<body>
<header>
  <h1>🖨️ 3D Print <span>Live Inference</span> Dashboard</h1>
  <span id="conn-status">● LIVE</span>
</header>

<div class="main">

  <!-- Inference Status -->
  <div class="card">
    <div class="card-title">Current Inference</div>
    <div id="inf-badge" class="inference-badge badge-idle">Idle</div>
    <div style="margin-top:1.2rem;">
      <div class="kv-row">
        <span class="kv-key">Class Code</span>
        <span class="kv-val" id="inf-class">—</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Error Count</span>
        <span class="kv-val" id="persist-counter">0</span>
        <span class="kv-val" style="color:var(--muted)"> / <span id="persist-thresh">3</span></span>
      </div>
    </div>
  </div>

  <!-- Load Cell -->
  <div class="card">
    <div class="card-title">Load Cell</div>
    <canvas id="chartLoad"></canvas>
  </div>

  <!-- Correction Controls -->
  <div class="card">
    <div class="card-title">Auto-Correction</div>
    <div class="toggle-row">
      <label class="toggle">
        <input type="checkbox" id="correction-toggle" onchange="toggleCorrection(this)">
        <span class="slider"></span>
      </label>
      <span style="font-weight:600;" id="toggle-label">Disabled</span>
    </div>
    <div style="margin-top:1rem;" class="kv-row">
      <span class="kv-key">Last Correction</span>
      <span class="kv-val" id="last-correction" style="color:var(--yellow);">None</span>
    </div>
    <div class="kv-row">
      <span class="kv-key">GCode Sent</span>
      <span class="kv-val" id="last-gcode" style="font-family:monospace;color:#60a5fa;">—</span>
    </div>
    <div class="kv-row">
      <span class="kv-key">Total Corrections</span>
      <span class="kv-val" id="total-corrections">0</span>
    </div>
  </div>

  <!-- ADXL1 -->
  <div class="card">
    <div class="card-title">ADXL1 — Accelerometer 1</div>
    <canvas id="chartAdxl1"></canvas>
    <div style="margin-top:.75rem;">
      <div class="kv-row"><span class="kv-key">X</span><span class="kv-val" id="a1x">—</span></div>
      <div class="kv-row"><span class="kv-key">Y</span><span class="kv-val" id="a1y">—</span></div>
      <div class="kv-row"><span class="kv-key">Z</span><span class="kv-val" id="a1z">—</span></div>
    </div>
  </div>

  <!-- ADXL2 -->
  <div class="card">
    <div class="card-title">ADXL2 — Accelerometer 2</div>
    <canvas id="chartAdxl2"></canvas>
    <div style="margin-top:.75rem;">
      <div class="kv-row"><span class="kv-key">X</span><span class="kv-val" id="a2x">—</span></div>
      <div class="kv-row"><span class="kv-key">Y</span><span class="kv-val" id="a2y">—</span></div>
      <div class="kv-row"><span class="kv-key">Z</span><span class="kv-val" id="a2z">—</span></div>
    </div>
  </div>

  <!-- Correction Log -->
  <div class="card">
    <div class="card-title">Correction History</div>
    <div class="corrections-list" id="correction-log">
      <div style="color:var(--muted);font-size:.85rem;">No corrections yet.</div>
    </div>
  </div>

</div>

<script>
const MAX_POINTS = 60;
const labels = [];

function makeChart(id, datasets) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      animation: false,
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#94a3b8', boxWidth: 10, font: { size: 11 } } } },
      scales: {
        x: { display: false },
        y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#2a3550' } }
      }
    }
  });
}

const chartAdxl1 = makeChart('chartAdxl1', [
  { label: 'X', data: [], borderColor: '#f87171', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
  { label: 'Y', data: [], borderColor: '#4ade80', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
  { label: 'Z', data: [], borderColor: '#60a5fa', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
]);
const chartAdxl2 = makeChart('chartAdxl2', [
  { label: 'X', data: [], borderColor: '#fb923c', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
  { label: 'Y', data: [], borderColor: '#a78bfa', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
  { label: 'Z', data: [], borderColor: '#34d399', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
]);
const chartLoad = makeChart('chartLoad', [
  { label: 'Load', data: [], borderColor: '#fbbf24', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: true,
    backgroundColor: 'rgba(251,191,36,0.08)' },
]);

function pushData(chart, ...vals) {
  const ts = new Date().toLocaleTimeString();
  if (labels.length > MAX_POINTS) { labels.shift(); }
  else { labels.push(ts); }
  vals.forEach((v, i) => {
    if (chart.data.datasets[i].data.length > MAX_POINTS)
      chart.data.datasets[i].data.shift();
    chart.data.datasets[i].data.push(v);
  });
  chart.update('none');
}

function setBadge(label, cls) {
  const el = document.getElementById('inf-badge');
  el.textContent = label;
  el.className = 'inference-badge ' + cls;
}

function getBadgeClass(inf_class) {
  if (inf_class === -1) return 'badge-idle';
  if (inf_class === 0)  return 'badge-normal';
  if (inf_class === 3)  return 'badge-warning';
  return 'badge-error';
}

async function fetchData() {
  try {
    const r = await fetch('/data');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();

    const a1 = d.adxl1, a2 = d.adxl2;
    pushData(chartAdxl1, a1.x, a1.y, a1.z);
    pushData(chartAdxl2, a2.x, a2.y, a2.z);
    pushData(chartLoad, d.load_cell);

    document.getElementById('a1x').textContent = a1.x.toFixed(4);
    document.getElementById('a1y').textContent = a1.y.toFixed(4);
    document.getElementById('a1z').textContent = a1.z.toFixed(4);
    document.getElementById('a2x').textContent = a2.x.toFixed(4);
    document.getElementById('a2y').textContent = a2.y.toFixed(4);
    document.getElementById('a2z').textContent = a2.z.toFixed(4);

    setBadge(d.inference_label, getBadgeClass(d.inference_class));
    document.getElementById('inf-class').textContent = d.inference_class >= 0 ? d.inference_class : '—';
    document.getElementById('persist-counter').textContent = d.persistence_counter;
    document.getElementById('persist-thresh').textContent  = d.persistence_threshold;

    document.getElementById('last-correction').textContent = d.last_correction;
    document.getElementById('last-gcode').textContent = d.last_correction_gcode;
    document.getElementById('total-corrections').textContent = d.total_corrections;

    const toggle = document.getElementById('correction-toggle');
    toggle.checked = d.correction_enabled;
    document.getElementById('toggle-label').textContent = d.correction_enabled ? 'Enabled' : 'Disabled';

    // Correction log
    if (d.corrections_log && d.corrections_log.length > 0) {
      document.getElementById('correction-log').innerHTML = d.corrections_log.slice().reverse().map(c =>
        `<div class="correction-item">
          <div class="time">${c.time}</div>
          <div class="desc">${c.label} → ${c.desc}</div>
          <div class="gcode">${c.gcode}</div>
        </div>`
      ).join('');
    }

  } catch(e) {
    document.getElementById('conn-status').textContent = '● DISCONNECTED';
    document.getElementById('conn-status').style.color = '#ef4444';
  }
}

async function toggleCorrection(el) {
  await fetch('/data', { method: 'GET' }); // dummy — handled by OctoPrint API
  // Actually POST to OctoPrint API
  try {
    await fetch('/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ correction_enabled: el.checked })
    });
  } catch(e) { console.warn('Could not toggle correction via internal API', e); }
}

setInterval(fetchData, 1500);
fetchData();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
#  Plugin metadata
# ─────────────────────────────────────────────
__plugin_name__          = "Print Inference"
__plugin_pythoncompat__  = ">=3.7,<4"
__plugin_implementation__ = PrintInferencePlugin()

__plugin_hooks__ = {}
