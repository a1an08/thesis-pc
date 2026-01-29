import octoprint.plugin
import time
import threading
import os
import csv
import requests

class DataCollectorPlugin(
    octoprint.plugin.StartupPlugin,
    octoprint.plugin.ShutdownPlugin,
    octoprint.plugin.SettingsPlugin,
    octoprint.plugin.TemplatePlugin,
    octoprint.plugin.WebcamProviderPlugin
):

    def __init__(self):
        self._capture_interval = 2.0  # seconds
        self._timer = None
        self._running = False

        self._base_dir = None
        self._image_dir = None
        self._csv_path = None

    # ─────────────────────────────
    # Startup / Shutdown
    # ─────────────────────────────
    def on_after_startup(self):
        self._base_dir = os.path.join(self.get_plugin_data_folder(), "data")
        self._image_dir = os.path.join(self._base_dir, "images")
        self._csv_path = os.path.join(self._base_dir, "log.csv")

        os.makedirs(self._image_dir, exist_ok=True)

        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_id",
                    "request_ts",
                    "capture_ts",
                    "sensor_ts",
                    "image_path",
                    "x",
                    "y",
                    "z",
                    "nozzle_temp",
                    "bed_temp",
                    "vibration"
                ])

        self._logger.info("DataCollectorPlugin initialized")
        self._start_capture_loop()  # start polling immediately

    def on_shutdown(self):
        self._stop_capture_loop()

    # ─────────────────────────────
    # Timer Loop for Polling
    # ─────────────────────────────
    def _start_capture_loop(self):
        if self._running:
            return
        self._running = True
        self._schedule_next()

    def _stop_capture_loop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()

    def _schedule_next(self):
        if not self._running:
            return
        self._timer = threading.Timer(self._capture_interval, self._poll_and_capture)
        self._timer.start()

    def _poll_and_capture(self):
        try:
            printer_data = self._printer.get_current_data()
            state = printer_data["state"]["text"].lower()

            if state == true:#"printing":
                self._capture_snapshot()
        except Exception as e:
            self._logger.error(f"Polling/capture error: {e}")
        finally:
            self._schedule_next()

    # ─────────────────────────────
    # Snapshot + Sensor Logging
    # ─────────────────────────────
    def _capture_snapshot(self):
        request_ts = time.time()  # When snapshot is requested

        snapshot_url = self.get_webcam_snapshot_url()
        if not snapshot_url:
            self._logger.warning("No webcam snapshot URL available")
            return

        try:
            response = requests.get(snapshot_url, timeout=5)
        except Exception as e:
            self._logger.warning(f"Failed to fetch snapshot: {e}")
            return

        response_ts = time.time()  # When snapshot received

        # Attempt to get capture time from headers (if supported)
        capture_ts = response.headers.get("X-Timestamp")
        if capture_ts:
            capture_ts = float(capture_ts)
        else:
            capture_ts = (request_ts + response_ts) / 2  # best estimate

        # Save image
        frame_id = int(capture_ts * 1000)
        filename = f"{frame_id}.jpg"
        image_path = os.path.join(self._image_dir, filename)

        with open(image_path, "wb") as f:
            f.write(response.content)

        # ───────── Printer position ─────────
        coords = self._printer.get_current_position()
        x = coords.get("x") if coords else None
        y = coords.get("y") if coords else None
        z = coords.get("z") if coords else None

        # ───────── Temperatures ─────────
        temps = self._printer.get_current_temperatures()
        nozzle_temp = temps["tool0"]["actual"] if "tool0" in temps else None
        bed_temp = temps["bed"]["actual"] if "bed" in temps else None

        # ───────── Sensor placeholder ─────────
        sensor_ts = time.time()
        vibration = self._read_vibration_sensor()

        # ───────── Log CSV ─────────
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                request_ts,
                capture_ts,
                sensor_ts,
                image_path,
                x,
                y,
                z,
                nozzle_temp,
                bed_temp,
                vibration
            ])

    # ─────────────────────────────
    # External Sensor Stub
    # ─────────────────────────────
    def _read_vibration_sensor(self):
        return 0.0  # placeholder
    

# ─────────────────────────────
# Plugin Metadata
# ─────────────────────────────

__plugin_pythoncompat__ = ">=3.7,<4"
__plugin_implementation__ = DataCollectorPlugin()
