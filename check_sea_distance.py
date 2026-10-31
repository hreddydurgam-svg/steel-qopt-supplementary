"""
Check of the sea distance Hay Point -> Karaikal (routes.H1.sea_distance_km) with searoute.

Needs: pip install searoute==1.6.0   (adds searoute, geojson, networkx; no other package is changed)
Port points (lon, lat) from OpenStreetMap: Hay Point Coal Terminal and Karaikal Port.
Usage:  python check_sea_distance.py
"""
import json
from importlib.metadata import version
from pathlib import Path

import searoute as sr

HAY_POINT = [149.2930, -21.2768]
KARAIKAL = [79.8457, 10.8388]

used = json.loads(Path(__file__).with_name("testbed_params_v18.json").read_text())["routes"]["H1"]["sea_distance_km"]["value"]
km = sr.searoute(HAY_POINT, KARAIKAL, units="km").properties["length"]
print(f"searoute {version('searoute')}: Hay Point -> Karaikal = {km:.1f} km; parameter file = {used:.1f} km; "
      f"{'MATCH' if abs(km - used) < 0.05 else 'DIFFERENT'}")
