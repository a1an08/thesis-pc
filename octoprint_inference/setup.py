from setuptools import setup, find_packages

plugin_identifier  = "octoprint_inference"
plugin_package     = "octoprint_inference"
plugin_name        = "OctoPrint-Inference"
plugin_version     = "1.0.0"
plugin_description = "XGBoost live inference for 3D print error detection with sensor data and auto-correction."
plugin_author      = "mem4"
plugin_author_email = ""
plugin_url         = ""
plugin_license     = "MIT"
plugin_requires    = [
    "OctoPrint",
    "xgboost>=1.7.0",
    "numpy>=1.21.0",
    "pyserial>=3.5",
    "flask>=2.0.0",
]

setup(
    name=plugin_package,
    version=plugin_version,
    description=plugin_description,
    author=plugin_author,
    author_email=plugin_author_email,
    url=plugin_url,
    license=plugin_license,
    packages=find_packages(),
    include_package_data=True,
    package_data={
        plugin_package: [
            "templates/**",
            "static/**/*",
        ]
    },
    install_requires=plugin_requires,
    entry_points={
        "octoprint.plugin": [
            f"{plugin_identifier} = {plugin_package}",
        ]
    },
)
