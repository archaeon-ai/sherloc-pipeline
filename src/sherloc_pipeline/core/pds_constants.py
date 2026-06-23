"""PDS archive constants shared across modules.

Constants live here (rather than in ``pds_client``) so that importers
can read defaults without forcing a lazy ``httpx`` import. The actual
HTTP client (``PDSDownloader``) is in ``pds_client``; importing it
requires the ``[pds]`` extra to be installed.
"""

from pathlib import Path

# PDS Geosciences Node base URL for the SHERLOC bundle (WUSTL archive).
PDS_BASE_URL = (
    "https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_sherloc"
)

# PDS Search API base URL (used for resolving ACI image download URLs).
PDS_SEARCH_API_BASE = "https://pds.mcp.nasa.gov/api/search/1"

# Default local cache directory for downloaded PDS data.
PDS_DEFAULT_CACHE_DIR = Path("./pds")
