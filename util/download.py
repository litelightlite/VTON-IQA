import os
import tempfile
import argparse
import subprocess

ROOT = "https://research.zozo.com/data_release/vtoniqa"


def aria2c(urls: list[str], out_dir: str, max_connections: int = 16, parallel: int = 3):

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("\n".join(urls) + "\n")
        url_file = f.name

    try:
        cmd = [
            "aria2c",
            "-c",
            "-x", str(max_connections),
            "-s", str(max_connections),
            "-j", str(parallel),
            "-i", url_file,
            "-d", out_dir,
        ]

        subprocess.run(cmd, check=True)

    finally:
        if os.path.exists(url_file):
            os.remove(url_file)


def download_vtoniqa(outdir: str):

    filenames = [
        "config.json",
        "model.safetensors",
        "target_scaler.pkl",
    ]

    urls = [os.path.join(ROOT, "vtoniqa", fname) for fname in filenames]

    for url in urls:
        print(f"Queued: {url}")

    aria2c(urls, outdir)


def download_vtonqbench(outdir: str):

    filenames = [
        "synth_vton.zip",
        "vtonqbench.zip",
    ]
    urls = [os.path.join(ROOT, fname) for fname in filenames]
    for url in urls:
        print(f"Queued: {url}")

    aria2c(urls, outdir)


def download_test(outdir: str):

    filenames = [
        "test.zip",
    ]
    urls = [os.path.join(ROOT, fname) for fname in filenames]
    for url in urls:
        print(f"Queued: {url}")

    aria2c(urls, outdir)


DOWNLOAD_FUNCS = {
    "vtoniqa": download_vtoniqa,
    "vtonqbench": download_vtonqbench,
    "test": download_test,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=list(DOWNLOAD_FUNCS.keys()),
        required=True,
        help="Download target. Example: --target vtoniqa",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for downloaded files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    DOWNLOAD_FUNCS[args.target](args.outdir)


if __name__ == "__main__":
    main()
