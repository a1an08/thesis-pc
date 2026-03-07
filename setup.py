# coding=utf-8
from setuptools import setup

plugin_identifier = "m4bp"
plugin_package = "octoprint_m4bp"
plugin_name = "m4bp"
plugin_version = "0.1.0"
plugin_description = """A plugin to collect and visualize sensor data and YOLO/CatBoost inference scores."""
plugin_author = "Alan Ocho"
plugin_author_email = ""
plugin_url = ""
plugin_license = "AGPLv3"

plugin_requires = [
    "pyserial",
    "numpy",
    "requests",
    "ultralytics",
    "onnxruntime",
    "pandas"
]

plugin_additional_data = []
plugin_additional_packages = []
plugin_ignored_packages = []
additional_setup_parameters = {}

setup(
    name="OctoPrint-DataCollector",
    version=plugin_version,
    description=plugin_description,
    author=plugin_author,
    author_email=plugin_author_email,
    url=plugin_url,
    license=plugin_license,
    requires=[],
    install_requires=plugin_requires,
    packages=[plugin_package] + plugin_additional_packages,
    package_data={plugin_package: ["static/js/*", "static/css/*", "templates/*"]},
    zip_safe=False,
    entry_points={
        "octoprint.plugin": [
            f"{plugin_identifier} = {plugin_package}"
        ]
    },
    **additional_setup_parameters
)
