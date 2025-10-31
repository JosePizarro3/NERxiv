from pathlib import Path

import h5py

for i, path in enumerate(Path("./data").glob("*.hdf5")):
    # if i > 0:
    #     break
    with h5py.File(path, "r+") as f:
        rag_group = f.require_group("rag_extraction")

        # Groups to move
        to_move = ["chunks_cache", "raw_llm_answers", "retrieval_cache"]
        for name in to_move:
            if name in f:
                f.copy(f[name], rag_group, name)
                del f[name]
        print(f"Processed file: {path}")
