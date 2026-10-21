import os
import sys

# Let tests import the ME193Sumo modules (config, xbox, ...) directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
