import argparse

from pud.envs.safe_habitatenv.unit_tests.test_safe_habitatenv import cost_contour


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, help="")
    parser.add_argument("--normalize", action="store_true", help="")

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose printing/logging"
    )
    args = parser.parse_args()

    cost_contour(args.scene_name, normalize=args.normalize)
