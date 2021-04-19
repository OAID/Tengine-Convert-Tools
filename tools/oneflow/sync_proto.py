import shutil
import argparse
from pathlib import Path


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--source", type=str, required=True)
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_path = Path(args.source) / "oneflow"
    src_path_str = str(src_path.absolute())
    dst_path_str = str((Path(".") / "oneflow").absolute())
    for proto_path in src_path.glob("**/*.proto"):
        proto_path_str = str(proto_path)
        dst_proto_path_str = proto_path_str.replace(src_path_str, dst_path_str)
        dst_proto_path = Path(dst_proto_path_str)
        if not dst_proto_path.parent.exists():
            dst_proto_path.parent.mkdir(parents=True)
        shutil.copy2(proto_path_str, dst_proto_path_str)
