import argparse
import os
import tempfile

from plumbum import local, path, FG


def main(args):
    with tempfile.TemporaryDirectory() as tmp:
        local.cwd.chdir(tmp)
        local.cmd.pip['wheel', '--no-deps', f'https://github.com/bennofs/compy-learn/archive/{args.commit}.tar.gz'].run_fg()
        files = list(local.cwd // "*.whl")
        assert len(files) == 1, "pip wheel should produce exactly one wheel, got: " + ", ".join(str(f) for f in files)
        whl = files[0]

        parts = list(whl.name.split("-"))
        assert len(parts) == 5, 'wheel file name expected to have 5 components: ' + whl.stem

        parts[1] = args.commit
        out_name = '-'.join(parts)
        os.makedirs(args.outdir, exist_ok=True)
        path.utils.move(whl, args.outdir + "/" + out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a wheel for a specific compy-learn version")
    parser.add_argument('commit', metavar='COMMIT', help='Version of compy-learn to build wheel for')
    parser.add_argument('outdir', metavar='DIR', help='Directory in which the built wheel is placed')
    main(parser.parse_args())
