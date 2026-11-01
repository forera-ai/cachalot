"""pareto.py with the 0.72.2 stale-record check disabled: ResidentExpertStore.holds always True, so a stale prefetched record is used as before."""
import runpy
import sys

from cachalot.cache.resident_store import ResidentExpertStore

ResidentExpertStore.holds = lambda self, resident: True
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
