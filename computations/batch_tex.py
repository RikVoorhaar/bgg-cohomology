"""Script to call PDFLaTeX on all files in a folder.

Puts the resulting PDF's in a new folder, and also produces a file with all the PDF's merged"""

import argparse
import fnmatch
import glob
import os
import re
import shutil
import subprocess
import sys

from tqdm import tqdm


def main(args):
    parser = argparse.ArgumentParser(
        description="Turn a list of .tex files into a list of .pdf files, as well as one large pdf file."
    )
    parser.add_argument("input", help="Name of input folder")
    parser.add_argument(
        "-o", default="", type=str, help="Name of output folder"
    )
    parser.add_argument(
        "-g",
        default="*",
        type=str,
        help="Glob pattern (default: '*'). Only matching .tex files are used, and recursive search is applied.",
    )
    parser.add_argument(
        "-m",
        default="merge",
        type=str,
        help="Name of merged pdf/tex (without file extension) (default: 'merge')",
    )

    args = parser.parse_args()

    input_folder = args.input
    if args.o == "":
        output_folder = input_folder
    else:
        output_folder = args.o

    glob_pattern = args.g
    merge_name = args.m

    temp_dir = os.path.join(output_folder, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    if len(os.listdir(temp_dir)):
        raise IOError(f"Working directory '{temp_dir}' not empty")

    tex_files = os.listdir(input_folder)
    tex_files = [f for f in tex_files if fnmatch.fnmatch(f, glob_pattern)]
    # tex_files = glob.glob(os.path.join(input_folder, glob_pattern))
    tex_files = [f for f in tex_files if re.match(r".*\.tex$", f)]
    print(f"Found {len(tex_files)} .tex files.")

    sort_keys = []
    for f in tex_files:
        try:
            diag, s, subset = f.split(".")[0].split("-")
            s = int(s[1:])
            subset = subset[1:-1]
            if len(subset) > 0:
                try:
                    subset_tuple = (int(subset),)
                except ValueError:
                    subset_tuple = tuple(int(i) for i in subset.split(","))
            else:
                subset_tuple = tuple()
            sort_keys.append((diag, subset_tuple, s))
        except ValueError:
            sort_keys.append(f)
    tex_files = [t[1] for t in sorted(zip(sort_keys, tex_files))]

    print("Compiling pdfs")
    pbar = tqdm(tex_files)
    for file in pbar:
        pbar.set_description(file)
        old_path = os.path.join(input_folder, file)
        new_path = os.path.join(temp_dir, file)
        shutil.copyfile(old_path, new_path)

        tex_log = subprocess.run(
            [
                "pdflatex",
                "-output-directory",
                output_folder,
                new_path,
            ],
            capture_output=True,
        )

    print("Merging files")
    merge_tex = ""
    preamble = [
        r"\documentclass[multi=page,crop]{standalone}",
        r"\usepackage{graphics}",
        r"\begin{document}",
    ]
    merge_tex += "\n".join(preamble)

    merge_tex += r"\begin{page}"+"\n"
    merge_tex += r"\begin{tabular}{l}"+"\n"
    for i, (diag, subset_tuple, s) in enumerate(sorted(sort_keys)):
        merge_tex += f"Diagram {diag}, subset ${subset_tuple}$, $s={s}$: page {i+2}"+r"\\ " + "\n"
    merge_tex += r"\end{tabular}"+"\n"
    merge_tex += r"\end{page}" + "\n"

    for i, f in enumerate(tex_files):
        f = f[:-4]
        merge_tex += "\n"
        merge_tex += r"\begin{page} \includegraphics{"
        merge_tex += f
        merge_tex += r"}\end{page}"
    merge_tex += "\n"
    merge_tex += r"\end{document}"

    merge_path = os.path.join(temp_dir, merge_name + ".tex")
    with open(merge_path, "w") as tex_file:
        tex_file.write(merge_tex)

    tex_log = subprocess.run(
        [
            "pdflatex",
            "-output-dir",
            output_folder,
            merge_path,
        ],
        capture_output=True,
    )


    print("Cleaning up")
    shutil.rmtree(temp_dir, ignore_errors=False)
    for file in glob.glob("pdfs/*.log")+glob.glob("pdfs/*.aux"):
        os.remove(file)
    print("Done!")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
