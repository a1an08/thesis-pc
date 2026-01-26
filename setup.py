from setuptools import setup

plugin_identifier = "data_collector"
plugin_package = "octoprint_data_collector"
plugin_name = "OctoPrint Data Collector"
plugin_version = "0.1.0"
plugin_description = "Captures webcam snapshots and logs printer data for ML dataset collection"
plugin_author = "Your Name"
plugin_author_email = "your@email.com"
plugin_url = "https://github.com/yourname/octoprint-data-collector"
plugin_license = "MIT"

plugin_requires = [
    "requests"
]

setup(
    name=f"OctoPrint-{plugin_name}",
    version=plugin_version,
    description=plugin_description,
    author=plugin_author,
    author_email=plugin_author_email,
    url=plugin_url,
    license=plugin_license,
    packages=[plugin_package],
    include_package_data=True,
    install_requires=plugin_requires,
    entry_points={
        "octoprint.plugin": [
            f"{plugin_identifier} = {plugin_package}"
        ]
    },
    python_requires=">=3.7,<4",
)
