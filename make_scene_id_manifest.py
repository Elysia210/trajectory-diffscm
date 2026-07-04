"""
Convert a build_collision_labels.py manifest into one keyed by a globally
unique scene_id.

Why: the raw manifest is keyed by bare `scene_key`, which is NOT unique across
the 189 HDF5 files (same scene-####_ctrl_[n] reused across cases/policies).
TrajectoryDataset.resolve_label() first looks up a label under
`f"{file_path}:{scene_key}"` (its `full_scene_id`). So if we emit a `scene_id`
column equal to `source_file:scene_key`, the loader matches it exactly and every
scene gets its own correct label -- no loader change required.

Usage (run from repo root):
    python make_scene_id_manifest.py labels/apr11_collision_labels.csv labels/apr11_collision_labels_scene_id.csv
"""

import csv
import sys


def main():
    if len(sys.argv) != 3:
        print("usage: python make_scene_id_manifest.py <in_csv> <out_csv>")
        sys.exit(1)
    in_csv, out_csv = sys.argv[1], sys.argv[2]

    rows_out = []
    seen = set()
    collisions = 0
    with open(in_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = f"{row['source_file']}:{row['scene_key']}"
            if scene_id in seen:
                # Should never happen: scene keys are unique *within* one h5 file.
                raise ValueError(f"Unexpected duplicate scene_id: {scene_id}")
            seen.add(scene_id)
            label = int(row["label"])
            collisions += label
            rows_out.append({"scene_id": scene_id, "label": label})

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "label"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"wrote {len(rows_out)} rows ({len(seen)} unique scene_id) -> {out_csv}")
    print(f"collision: {collisions}  non-collision: {len(rows_out) - collisions}")


if __name__ == "__main__":
    main()
