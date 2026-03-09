import importlib.metadata
import os

print("--- Distributions with None origin ---")
for dist in importlib.metadata.distributions():
    origin = getattr(dist, 'origin', None)
    if origin is None:
        try:
            # Check for entry points in octoprint.plugin
            eps = [ep for ep in dist.entry_points if ep.group == 'octoprint.plugin']
            if eps:
                print(f"Name: {dist.metadata.get('Name')}")
                print(f"Location: {getattr(dist, '_path', 'N/A')}")
                print(f"Entry Points: {eps}")
                print("-" * 20)
        except:
            pass
