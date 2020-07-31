"""Script to call PDFLaTeX on all files in a folder.

Puts the resulting PDF's in a new folder, and also produces a file with all the PDF's merged"""

import os
import sys
import re
import shutil
import subprocess
from tqdm.auto import tqdm


def main(args):
    if len(args) == 2:
        input_folder, output_folder = args
    elif len(args) == 1:
        input_folder = args[0]
        output_folder = os.path.join(input_folder, "pdfs")
    else:
        raise ValueError(
            "Wrong arguments. Format: `batch_tex.py input_folder (output_folder)`"
        )

    temp_dir = os.path.join(output_folder, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    if len(os.listdir(temp_dir)):
        raise IOError(f"Working directory '{temp_dir}' not empty")

    tex_files = os.listdir(input_folder)
    tex_files = [f for f in tex_files if re.match(r".*\.tex$", f)]
    print(f"Found {len(tex_files)} .tex files.")

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
                "-aux-directory",
                temp_dir,
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
    for f in tex_files:
        f = f[:-4]
        merge_tex += "\n"
        merge_tex += r"\begin{page}\includegraphics{"
        merge_tex += f
        merge_tex += r"}\end{page}"
    merge_tex += "\n"
    merge_tex += r"\end{document}"

    merge_path = os.path.join(temp_dir, "merge.tex")
    with open(merge_path, "w") as tex_file:
        tex_file.write(merge_tex)

    tex_log = subprocess.run(
        [
            "pdflatex",
            "-aux-directory",
            temp_dir,
            "-output-directory",
            output_folder,
            merge_path,
        ],
        capture_output=True,
    )

    print("Cleaning up")
    shutil.rmtree(temp_dir, ignore_errors=False)

    print("Done!")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
