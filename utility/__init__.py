import octoprint.plugin
import time
import threading
import os
import csv
import requests


class DataCollectorPlugin(
    octoprint.plugin.StartupPlugin,
    octoprint.plugin.ShutdownPlugin,
    octoprint.plugin.EventHandlerPlugin,
    octoprint.plugin.WebcamPluginMixin
):

    def __init__(self):
        self._capture_interval = 2.0  # seconds (N)
        self._timer = None
        self._running = False

        self._base_dir = None
        self._image_dir = None
        self._csv_path = None

    # ─────────────────────────────
    # Startup / Shutdown
    # ─────────────────────────────

    def on_after_startup(self):
        self._base_dir = os.path.join(
            self.get_plugin_data_folder(), "data"
        )
        self._image_dir = os.path.join(self._base_dir, "images")
        self._csv_path = os.path.join(self._base_dir, "log.csv")

        os.makedirs(self._image_dir, exist_ok=True)

        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "image_path",
                    "x",
                    "y",
                    "z",
                    "nozzle_temp",
                    "bed_temp",
                    "vibration"
                ])

        self._logger.info("DataCollectorPlugin initialized")

    def on_shutdown(self):
        self._stop_capture()

    # ─────────────────────────────
    # Event Handling
    # ─────────────────────────────

    def on_event(self, event, payload):
        if event == octoprint.events.Events.PRINT_STARTED:
            self._start_capture()

        elif event in (
            octoprint.events.Events.PRINT_DONE,
            octoprint.events.Events.PRINT_FAILED,
            octoprint.events.Events.PRINT_CANCELLED
        ):
            self._stop_capture()

    # ─────────────────────────────
    # Capture Loop
    # ─────────────────────────────

    def _start_capture(self):
        if self._running:
            return

        self._running = True
        self._schedule_next()
        self._logger.info("Started data capture")

    def _stop_capture(self):
        self._running = False
        if self._timer:
            self._timer.cancel()
        self._logger.info("Stopped data capture")

    def _schedule_next(self):
        if not self._running:
            return

        self._timer = threading.Timer(
            self._capture_interval, self._capture_step
        )
        self._timer.start()

    def _capture_step(self):
        try:
            self._capture_snapshot()
        except Exception as e:
            self._logger.error(f"Capture error: {e}")

        self._schedule_next()

    # ─────────────────────────────
    # Snapshot + Logging
    # ─────────────────────────────

    def _capture_snapshot(self):
        timestamp = time.time()  # Unix epoch

        snapshot_url = self.get_webcam_snapshot_url()
        if not snapshot_url:
            self._logger.warning("No webcam snapshot URL")
            return

        response = requests.get(snapshot_url, timeout=5)
        if response.status_code != 200:
            self._logger.warning("Failed to fetch snapshot")
            return

        filename = f"{int(timestamp * 1000)}.jpg"
        image_path = os.path.join(self._image_dir, filename)

        with open(image_path, "wb") as f:
            f.write(response.content)

        # ───────── Printer position ─────────
        data = self._printer.get_current_data()
        pos = data["currentZ"] if data else None
        coords = self._printer.get_current_position()

        x = coords["x"] if coords else None
        y = coords["y"] if coords else None
        z = coords["z"] if coords else None

        # ───────── Temperatures ─────────
        temps = self._printer.get_current_temperatures()
        nozzle_temp = temps["tool0"]["actual"] if "tool0" in temps else None
        bed_temp = temps["bed"]["actual"] if "bed" in temps else None

        # ───────── Sensor placeholder ─────────
        vibration = self._read_vibration_sensor()

        # ───────── Log CSV ─────────
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
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
        """
        Replace this with:
        - serial read
        - socket
        - shared memory
        - file read
        """
        return 0.0  # placeholder


# ─────────────────────────────
# Plugin Metadata
# ─────────────────────────────

__plugin_pythoncompat__ = ">=3.7,<4"
