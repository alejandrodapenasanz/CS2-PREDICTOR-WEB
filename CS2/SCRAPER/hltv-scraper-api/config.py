import datetime
import os
from pathlib import Path

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

BASE_DIR = "./hltv_scraper"
DATA_DIR = str(Path(__file__).resolve().parents[3] / "VAULT" / "CS2" / "SCRAPER" / "hltv-scraper-api" / "data")

TODAY = datetime.date.today()
