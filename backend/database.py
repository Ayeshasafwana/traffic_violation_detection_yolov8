import json
from datetime import datetime
import os

def save_log(data):
    log = {
        "time": str(datetime.now()),
        "data": data
    }

    # Ensure folder exists
    os.makedirs("../data", exist_ok=True)

    try:
        with open("../data/logs.json", "r") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(log)

    with open("../data/logs.json", "w") as f:
        json.dump(logs, f, indent=4)