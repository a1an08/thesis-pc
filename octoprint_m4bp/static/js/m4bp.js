$(function() {
    function DataCollectorViewModel(parameters) {
        var self = this;
        self.settings = parameters[0];

        self.currentPrediction = ko.observable("Waiting for data...");
        self.yolo_oextrusion = ko.observable(0.0);
        self.yolo_uextrusion = ko.observable(0.0);
        self.yolo_stringing = ko.observable(0.0);
        self.yolo_spaghetti = ko.observable(0.0);
        
        self.load_avg = ko.observable(0.0);
        self.last_timestamp = ko.observable(0);

        self.predictionClass = ko.computed(function() {
            var pred = self.currentPrediction();
            if (!pred || pred === "None" || pred === "Waiting for data...") return "muted";
            
            // Highlight prediction red if it's an error class, green if normal
            var lowerPred = pred.toLowerCase();
            if (lowerPred.includes("normal") || pred === "0" || pred === "0.0") return "text-success";
            return "text-error"; 
        });

        self.onDataUpdaterPluginMessage = function(plugin, data) {
            if (plugin != "m4bp") {
                return;
            }

            if (data.type == "live_data") {
                self.currentPrediction(data.prediction);
                self.last_timestamp(data.timestamp);
                
                if (data.yolo) {
                    self.yolo_oextrusion(data.yolo.oextrusion);
                    self.yolo_uextrusion(data.yolo.uextrusion);
                    self.yolo_stringing(data.yolo.stringing);
                    self.yolo_spaghetti(data.yolo.spaghetti);
                }
                
                self.load_avg(data.load_avg);
            }
        };
    }

    OCTOPRINT_VIEWMODELS.push({
        construct: DataCollectorViewModel,
        dependencies: ["settingsViewModel"],
        elements: ["#tab_plugin_m4bp"]
    });
});
