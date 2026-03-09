$(function () {
    "use strict";

    var POLL_INTERVAL = 1500; // ms
    var API_URL = "/plugin/octoprint_inference/data";
    var SETTINGS_URL = "/plugin/octoprint_inference/settings";

    // ── KnockoutJS View Model ──────────────────────────
    function InferenceViewModel(parameters) {
        var self = this;

        // OctoPrint provided view models
        self.settingsViewModel = parameters[0];

        // ── Observables ────────────────────────────
        self.adxl1x = ko.observable("—");
        self.adxl1y = ko.observable("—");
        self.adxl1z = ko.observable("—");

        self.adxl2x = ko.observable("—");
        self.adxl2y = ko.observable("—");
        self.adxl2z = ko.observable("—");

        self.loadCell = ko.observable("—");

        self.inferenceClass  = ko.observable("—");
        self.inferenceLabel  = ko.observable("Idle");
        self.inferenceBadgeClass = ko.observable("badge-idle");

        self.persistenceCounter   = ko.observable(0);
        self.persistenceThreshold = ko.observable(3);

        self.correctionEnabled = ko.observable(false);
        self.lastCorrection    = ko.observable("None");
        self.lastGCode         = ko.observable("—");
        self.totalCorrections  = ko.observable(0);
        self.correctionsLog    = ko.observableArray([]);

        // ── Badge class helper ─────────────────────
        function getBadgeClass(cls) {
            if (cls === -1) return "badge-idle";
            if (cls === 0)  return "badge-normal";
            if (cls === 3)  return "badge-warning";
            return "badge-error";
        }

        // ── Poll API ───────────────────────────────
        function poll() {
            $.ajax({
                url: API_URL,
                type: "GET",
                dataType: "json",
                success: function (data) {
                    // Accelerometers
                    self.adxl1x(data.adxl1.x.toFixed(4));
                    self.adxl1y(data.adxl1.y.toFixed(4));
                    self.adxl1z(data.adxl1.z.toFixed(4));

                    self.adxl2x(data.adxl2.x.toFixed(4));
                    self.adxl2y(data.adxl2.y.toFixed(4));
                    self.adxl2z(data.adxl2.z.toFixed(4));

                    // Load cell
                    self.loadCell(parseFloat(data.load_cell).toFixed(4));

                    // Inference
                    self.inferenceClass(data.inference_class >= 0 ? data.inference_class : "—");
                    self.inferenceLabel(data.inference_label);
                    self.inferenceBadgeClass(getBadgeClass(data.inference_class));

                    // Persistence
                    self.persistenceCounter(data.persistence_counter);
                    self.persistenceThreshold(data.persistence_threshold);

                    // Corrections
                    self.correctionEnabled(data.correction_enabled);
                    self.lastCorrection(data.last_correction);
                    self.lastGCode(data.last_correction_gcode);
                    self.totalCorrections(data.total_corrections);

                    var log = (data.corrections_log || []).slice().reverse();
                    self.correctionsLog(log);
                },
                error: function () {
                    console.warn("[InferencePlugin] Failed to fetch plugin data.");
                }
            });
        }

        // ── Toggle correction ──────────────────────
        self.toggleCorrection = function () {
            var enabled = self.correctionEnabled();
            $.ajax({
                url: SETTINGS_URL,
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({ correction_enabled: enabled }),
                success: function (resp) {
                    self.correctionEnabled(resp.correction_enabled);
                    var msg = resp.correction_enabled
                        ? "Auto-correction ENABLED."
                        : "Auto-correction DISABLED.";
                    new PNotify({ title: "Inference Plugin", text: msg, type: "info" });
                }
            });
            return true; // allow checkbox to proceed
        };

        // ── Lifecycle ──────────────────────────────
        self.onBeforeBinding = function () {
            // nothing yet
        };

        self.onAfterBinding = function () {
            poll();
            setInterval(poll, POLL_INTERVAL);
        };
    }

    // ── Register View Model ────────────────────────
    OCTOPRINT_VIEWMODELS.push({
        construct: InferenceViewModel,
        dependencies: ["settingsViewModel"],
        elements: ["#plugin-inference-content"]
    });
});
