import yaml
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, help="cfg python file path")
    args = parser.parse_args()

    cfg_file = Path(args.cfg_path)
    cfg = eval(open(cfg_file, "r").read())

    target_fname = cfg_file.name.replace(cfg_file.suffix, ".yaml")
    target_path = cfg_file.parent.joinpath(target_fname)

    with open(target_path, "w") as f:
        yaml.safe_dump(data=cfg, stream=f, allow_unicode=True, indent=4)
