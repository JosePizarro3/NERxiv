from pathlib import Path

import h5py

for i, path in enumerate(Path("./data").glob("*.hdf5")):
    # if i > 0:
    #     break
    with h5py.File(path, "r+") as f:
        if "raw_llm_answers" not in f:
            continue
        raw = f["raw_llm_answers"]
        old = raw.require_group("20251028_OLD_raw_llm_answers")
        for run in list(raw.keys()):
            if not run.startswith("run_"):
                continue
            old.copy(raw[run], run)
            del raw[run]
        print(f"Processed file: {path}")
