"""Script to load a .pkl file containing partial computations of a bigraded table
and just make a LaTeX file containing whatever we know, inserting `???` into unknown
entries."""
import sage.all

import argparse
import os
import pickle
import sys

from bggcohomology.bggcomplex import BGGComplex
from bggcohomology.la_modules import BGGCohomology
from bggcohomology.quantum_center import (
    all_abijk,
    display_bigraded_table,
    display_cohomology_stats,
    extend_from_symmetry,
    prepare_texfile,
)


def main():
    parser = argparse.ArgumentParser(description="Produce table from .pkl file")
    parser.add_argument("diagram")
    parser.add_argument("--compact", default=True)
    parser.add_argument("-s", default=0, type=int)
    parser.add_argument("--subset", default="", type=str)
    args = parser.parse_args()

    print(vars(args))

    os.makedirs("pickles", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

    diagram = args.diagram
    s = args.s
    compact = args.compact
    if len(args.subset) == 0:
        subset = []
    else:
        try:
            subset = [int(x) for x in args.subset.strip().split(",")]
        except:
            raise ValueError(f"argument --subset inproperly formated: {args.subset}")

    # the parameters we actually want to change
    BGG = BGGComplex(diagram)
    compact = True

    picklefile = os.path.join("pickles", f"{diagram}-s{s}-{subset}.pkl")
    cohom_dic = pickle.load(open(picklefile, "rb"))
    cohom = BGGCohomology(BGG)
    abijk = all_abijk(BGG, s=s, subset=subset, half_only=False)
    max_a = max(x[0] for x in abijk)
    cohom_dic = extend_from_symmetry(cohom_dic, max_a=max_a)
    for a, b, i, j, k in all_abijk(BGG, s=s, subset=subset, half_only=False):
        if (a, b) not in cohom_dic:
            cohom_dic[(a, b)] = None
    latex_dic = {k: cohom.cohom_to_latex(c, compact=compact) for k, c in cohom_dic.items()}
    betti_dic = {k: cohom.betti_number(c) for k, c in cohom_dic.items()}
    tab1 = display_bigraded_table(latex_dic, text_only=True)
    tab2 = display_bigraded_table(betti_dic, text_only=True)
    tab3 = display_cohomology_stats(cohom_dic, BGG, text_only=True)
    texfile = os.path.join("tables", f"{diagram}-s{s}-{subset}.tex")
    with open(texfile, "w") as f:
        f.write(
            prepare_texfile(
                [tab1, tab2, tab3], title=f"type {diagram}, s={s}, subset={subset}"
            )
        )

if __name__ == '__main__':
    main()