from pud.algos.cbfs.cbfs_eval import catalog_precompiled_paths
import argparse
from pathlib import Path

if __name__ == "__main__":
    """
    demo:
    python pud/algos/cbfs_catalog.py --cbfs_dir pud/envs/precompiles/mptest --output pud/envs/precompiles/mptest_catalog
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cbfs_dir",
        type=str,
        help="the directory that contains the CBFS precompilation output",
    )

    parser.add_argument(
        "--output", type=str, help="the FULL path to save the output CBFS catalog file"
    )
    args = parser.parse_args()

    cbfs_dir = Path(args.cbfs_dir)

    assert (
        cbfs_dir.exists() and cbfs_dir.is_dir()
    ), "cbfs_dir needs to be an existing directory"
    output_path = Path(args.output)

    catalog_precompiled_paths(cbfs_dir.as_posix(), output_path.as_posix())
