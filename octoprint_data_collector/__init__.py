# coding=utf-8
from __future__ import absolute_import

import octoprint.plugin
import time
import threading
import os
import csv
import requests
import shutil

class DataCollectorPlugin(
    octoprint.plugin.StartupPlugin,
    octoprint.plugin.ShutdownPlugin,
    octoprint.plugin.SettingsPlugin,
    octoprint.plugin.TemplatePlugin
):

    def __init__(self):
        self._capture_interval = 2.0  # Seconds between captures
        self._running = False
        self._worker_thread = None
        
        # Internal URL
        self._snapshot_url = "http://127.0.0.1:8080/?action=snapshot"
        
        self._base_dir = None
        self._image_dir = None
        self._csv_path = None

    # ─────────────────────────────
    # Startup / Shutdown
    # ─────────────────────────────
    def on_after_startup(self):
        # Set up directories in the standard Plugin Data folder
        self._base_dir = os.path.join(self.get_plugin_data_folder(), "data")
        self._image_dir = os.path.join(self._base_dir, "images")
        self._csv_path = os.path.join(self._base_dir, "log.csv")

        if not os.path.exists(self._image_dir):
            os.makedirs(self._image_dir)

        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_id", "request_ts", "capture_ts", "filename"])

        self._logger.info(f"DataCollector initialized. Logging to: {self._base_dir}")
        self._start_worker()

    def on_shutdown(self):
        self._logger.info("DataCollector shutting down...")
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    # ─────────────────────────────
    # Worker Thread
    # ─────────────────────────────
    def _start_worker(self):
        if self._running:
            return

        self._running = True
        # jic OctoPrint crashes, daemon stops
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        self._logger.info("DataCollector worker thread started.")
        
        while self._running:
            start_time = time.time()

            try:
                self._check_and_capture()
            except Exception as e:
                self._logger.error(f"Error in capture loop: {e}")

            # Calculate how long the work took
            elapsed = time.time() - start_time
            
            # Calculate exact sleep needed to maintain the 2.0s interval
            sleep_time = self._capture_interval - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self._logger.warning(f"Capture took too long ({elapsed:.2f}s)! Skipping sleep to catch up.")

    # ─────────────────────────────
    # Logic & Safety Checks
    # ─────────────────────────────
    def _check_and_capture(self):
        # Stop if disk is full (currently set to < 500MB free)
        total, used, free = shutil.disk_usage(self._base_dir)
        if free < (500 * 1024 * 1024): 
            if self._running:
                self._logger.critical("DISK ALMOST FULL (<500MB). Stopping DataCollector.")
                self._running = False
            return

        #Check Printer State
        printer_data = self._printer.get_current_data()
        state = "Offline"#printer_data["state"]["text"]
        
        valid_states = ["Printing", "Offline"] 
        if state in valid_states:
            self._capture_snapshot()

    # ─────────────────────────────
    # Capture & Log (Epoch Time)
    # ─────────────────────────────
    def _capture_snapshot(self):
        request_ts = time.time() 

        try:
            # stream=True reduces memory spike
            response = requests.get(self._snapshot_url, timeout=2.0, stream=True)
            if response.status_code != 200:
                self._logger.warning(f"Snapshot failed: HTTP {response.status_code}")
                return
        except Exception as e:
            return 

        response_ts = time.time()
        
        # Estimate the actual hardware capture time (midpoint of request)
        capture_ts = (request_ts + response_ts) / 2
        
        # Generate id based on timestamp
        frame_id = int(capture_ts * 1000)
        
        filename = f"{frame_id}.jpg"
        image_path = os.path.join(self._image_dir, filename)

        # Save Image
        with open(image_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)

        # Log csv
        try:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    frame_id,
                    "{:.6f}".format(request_ts),
                    "{:.6f}".format(capture_ts),
                    filename
                ])
        except Exception as e:
            self._logger.error(f"CSV Write Failed: {e}")

# ─────────────────────────────
# Plugin Metadata
# ─────────────────────────────

__plugin_name__ = "Data Collector"
__plugin_pythoncompat__ = ">=3.7,<4"
__plugin_implementation__ = DataCollectorPlugin()