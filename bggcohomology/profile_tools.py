import pickle
import time
import os
import datetime


def load_profile(filename: str) -> list:
    try:
        return pickle.load(open(filename, "rb"))
    except FileNotFoundError:
        return []


def compute_integer_dense_rank(matrix, profile_file=None):
    if profile_file is not None:
        profile_list = load_profile(profile_file)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        key = -1
        profile_list.append({})
        profile_list[key] = {"time": now_str}
        mat_numpy = matrix.numpy()
        profile_list[key]["matrix"] = mat_numpy
        profile_list[key]["start_time"] = time.perf_counter()
        profile_list[key]["shape"] = mat_numpy.shape
        pickle.dump(profile_list, open(profile_file, "wb"))

    rank = matrix.rank()
    if profile_file is not None:
        profile_list[key]["rank"] = rank
        profile_list[key]["time_taken"] = (
            time.perf_counter() - profile_list[key]["start_time"]
        )
        pickle.dump(profile_list, open(profile_file, "wb"))

    return rank
