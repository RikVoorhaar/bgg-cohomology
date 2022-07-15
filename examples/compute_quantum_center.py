import pickle
import os
import itertools
from time import perf_counter
from datetime import timedelta
from tqdm.auto import tqdm
from bggcohomology.la_modules import BGGCohomology
from bggcohomology.bggcomplex import BGGComplex
from bggcohomology.quantum_center import *
from bggcohomology.weight_set import WeightSet
from bggcohomology.cohomology import compute_diff
import logging
import traceback
import sys

LOG_FILE = "quantum_center_log.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(filename)s / %(funcName)s:\n\t %(message)s",
)


def log_except_hook(*exc_info):
    text = "".join(traceback.format_exception(*exc_info))
    logging.error("Unhandled exception: %s", text)


sys.excepthook = log_except_hook

os.makedirs("pickles", exist_ok=True)
os.makedirs("tables", exist_ok=True)
logging.info("")
logging.info("=" * 80)
logging.info("script started")
logging.info("=" * 80)
logging.info("")

# Only compute cohomology for particular highest weight module
mu = (0, 0, 0, 0)
# mu=(0,0)
logging.info(f"{mu=}")

# the parameters we actually want to change
diagram = "B4"
logging.info(f"{diagram=}")
BGG = BGGComplex(diagram)
subset = []
logging.info(f"{subset=}")

# compute only half of the table, extend by symmetry
half_only = True
extend_half = half_only


# Exclude the top-left to bottom-right diagonal. If s=0, these should all be the trivial rep.
exclude_diagonal = True
logging.info(f"{half_only=}, {extend_half=}, {exclude_diagonal=}")

# Display in full form
compact = True

# Load results if already computed
load_pickle = True
logging.info(f"{load_pickle=}")

s = 0
logging.info(f"{s=}")
# for method in [0,1]:
# for s in itertools.count():
method = 0
for s in [0]:
    picklefile = os.path.join("pickles", f"{diagram}-s{s}-{subset}.pkl")
    if load_pickle and os.path.isfile(picklefile):
        logging.info("Loading pickle file")
        previous_cohom = pickle.load(open(picklefile, "rb"))
    else:
        previous_cohom = None
    texfile = os.path.join("tables", f"{diagram}-s{s}-{subset}.tex")
    cohom_dic = dict()
    with tqdm(
        all_abijk(BGG, s=s, subset=subset, half_only=half_only)
    ) as inner_pbar:
        with tqdm(leave=None) as outer_pbar:
            map_pbar = tqdm()
            for a, b, i, j, k in inner_pbar:
                if previous_cohom is not None and (a, b) in previous_cohom:
                    logging.info(f"Loading {(a,b)} from pickle file")
                    cohom_dic[(a, b)] = previous_cohom[(a, b)]
                    inner_pbar.update()
                    continue
                if exclude_diagonal and s == 0 and (a == b):
                    logging.info(f"Skipping {(a,b)} on diagonal")
                    cohom_dic[(a, b)] = [((0,) * BGG.rank, 1)]
                    inner_pbar.update()
                    continue
                inner_pbar.set_description("i+j= %d, j-i = %d" % (a, b))
                logging.info("-" * 80)
                logging.info(f"Computing cohomology for i+j= {a}, j-i = {b}")
                current_time = perf_counter()
                mjk = Mjk(BGG, j, k, subset=subset)
                time_taken = timedelta(seconds=perf_counter() - current_time)
                logging.info(f"Computed Mjk in {time_taken}")
                current_time = perf_counter()
                outer_pbar.set_description("Initializing cohomology")
                coker = Eijk_basis(BGG, j, k, subset=subset, sparse=True)
                time_taken = timedelta(seconds=perf_counter() - current_time)
                logging.info(f"Computed Eijk in {time_taken}")
                current_time = perf_counter()
                cohom = BGGCohomology(
                    BGG, mjk, coker=coker, pbars=[outer_pbar, map_pbar]
                )
                time_taken = timedelta(seconds=perf_counter() - current_time)
                logging.info(f"Cohomology initialized in {time_taken}")
                current_time = perf_counter()
                outer_pbar.set_description("Computing cohomology")
                time_taken = timedelta(seconds=perf_counter() - current_time)
                logging.info(f"Cohomology computed in {time_taken}")
                cohom_list = cohom.cohomology(i, mu=mu)
                cohom_dic[(a, b)] = cohom_list
                with open(picklefile, "wb") as f:
                    pickle.dump(cohom_dic, f)
                    logging.info("Pickle file saved")
    logging.info("=" * 80)
    logging.info("Computiation finished")

    cohom = BGGCohomology(BGG)
    logging.info("Computing information about cohomology")
    cohom_dic = extend_from_symmetry(cohom_dic)
    latex_dic = {
        k: cohom.cohom_to_latex(c, compact=compact)
        for k, c in cohom_dic.items()
    }
    betti_dic = {k: cohom.betti_number(c) for k, c in cohom_dic.items()}
    tab1 = display_bigraded_table(latex_dic)
    tab2 = display_bigraded_table(betti_dic)
    tab3 = display_cohomology_stats(cohom_dic, BGG)
    with open(texfile, "w") as f:
        f.write(
            prepare_texfile(
                [tab1, tab2, tab3],
                title=f"type {diagram}, s={s}, subset={subset}",
            )
        )
    logging.info("Table written to %s" % texfile)
