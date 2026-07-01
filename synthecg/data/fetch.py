import os

import wfdb


def fetch_ptbxl_record(record_name: str, database: str = "ptb-xl/1.0.3"):
    """Fetch a 12-lead ECG record from PTB-XL on PhysioNet."""
    print(f"Fetching record {record_name} ...")
    record_dir, record_file = os.path.split(record_name)
    full_pn_dir = f"{database}/{record_dir}"
    return wfdb.rdrecord(record_file, pn_dir=full_pn_dir)
