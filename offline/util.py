from threading import Thread
import numpy as np
from time import perf_counter

def load_xyidst(directory, verbose=False, load_bad_ch=False):
    x = np.load(directory + "x.npy")
    y = np.load(directory + "y.npy")
    ids = np.load(directory + "ids.npy")
    erp_t = np.load(directory + "erp_t.npy")
    speed = np.load(directory + "speed.npy")
    bad_ch = None
    if load_bad_ch:
        bad_ch = np.load(directory + "bad_ch.npy")

    if verbose:
        print(f"Finished loading data from {directory}:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"ids shape: {ids.shape}")
        print(f"erp_t shape: {erp_t.shape}")
        print(f"speed shape: {speed.shape}")
        if load_bad_ch:
            print(f"bad_ch shape: {bad_ch.shape}")

        print("\n")

    return x, y, ids, erp_t, speed, bad_ch

def load_xyidst_threaded(directory, verbose=False, load_bad_ch=False):
    d = Data()

    tx = Thread(target=d.load_x, args=[directory])
    tx.start()

    tyidst = Thread(target=d.load_yidst, args=[directory, load_bad_ch])
    tyidst.start()

    tyidst.join()
    tx.join()
    x = d.x
    y = d.y
    ids = d.ids
    erp_t = d.erp_t
    speed = d.speed
    bad_ch = None
    if load_bad_ch:
        bad_ch = d.bad_ch

    if verbose:
        print(f"Finished loading data from {directory}:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"ids shape: {ids.shape}")
        print(f"erp_t shape: {erp_t.shape}")
        print(f"speed shape: {speed.shape}")
        if load_bad_ch:
            print(f"bad_ch shape: {bad_ch.shape}")

        print("\n")

    del d

    return x, y, ids, erp_t, speed, bad_ch

class Data:
    def __init__(self) -> None:
        self.x = None
        self.y = None
        self.ids = None
        self.erp_t = None
        self.speed = None
        self.bad_ch = None

    def load_x(self, directory):
        self.x = np.load(directory + "x.npy")

    def load_yidst(self, directory, load_bad_ch):
        self.y = np.load(directory + "y.npy")
        self.ids = np.load(directory + "ids.npy")
        self.erp_t = np.load(directory + "erp_t.npy")
        self.speed = np.load(directory + "speed.npy")
        if load_bad_ch:
            self.bad_ch = np.load(directory + "bad_ch.npy")

def save_xyidst(x, y, ids, erp_t, speed, folder, bad_ch=None, verbose=False):
    np.save(folder + "x.npy", x)
    np.save(folder + "y.npy", y)
    np.save(folder + "ids.npy", ids)
    np.save(folder + "erp_t.npy", erp_t)
    np.save(folder + "speed.npy", speed)
    if bad_ch is not None:
        np.save(folder + "bad_ch.npy", bad_ch)

    if verbose:
        print(f"Finished saving data to {folder}:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"ids shape: {ids.shape}")
        print(f"erp_t shape: {erp_t.shape}")
        print(f"speed shape: {speed.shape}")
        if bad_ch is not None:
            print(f"bad_ch shape: {bad_ch.shape}")

        print("\n")

def save_xyidst_threaded(x, y, ids, erp_t, speed, folder, bad_ch=None, verbose=False):
    tx = Thread(target=np.save, args=[folder + "x.npy", x])
    tx.start()

    tyidst = Thread(target=save_yidst, args=[y, ids, erp_t, speed, folder, bad_ch])
    tyidst.start()

    tyidst.join()
    tx.join()

    if verbose:
        print(f"Finished saving data to {folder}:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"ids shape: {ids.shape}")
        print(f"erp_t shape: {erp_t.shape}")
        print(f"speed shape: {speed.shape}")
        if bad_ch is not None:
            print(f"bad_ch shape: {bad_ch.shape}")

        print("\n")

def save_yidst(y, ids, erp_t, speed, folder, bad_ch):
    np.save(folder + "y.npy", y)
    np.save(folder + "ids.npy", ids)
    np.save(folder + "erp_t.npy", erp_t)
    np.save(folder + "speed.npy", speed)
    if bad_ch is not None:
        np.save(folder + "bad_ch.npy", bad_ch)

def data_split_load_save(source_folder, target_folder, load_bad_ch, verbose=False):
    x, y, ids, oz_t, speed, bad_ch = load_xyidst_threaded(source_folder, load_bad_ch=load_bad_ch, verbose=verbose)

    data_split_save(x, y, ids, oz_t, speed, target_folder, bad_ch, verbose)

def data_split_save(x, y, ids, oz_t, speed, target_folder, bad_ch = None, verbose=False):
    id_set = np.asarray(list(set(ids)))
    K = id_set.shape[0]
    groups = np.random.permutation(K)
    print(id_set)
    print(groups)
    train_mask = np.zeros(len(ids), dtype=bool)
    id_mask_train = np.logical_and(groups >= 0, groups <= int(0.8*K) - 1)
    print(f"ID MASK: {id_mask_train}")
    train_ids = id_set[id_mask_train]
    print(f"Train ids: {train_ids}")
    for id in train_ids:
        train_tmp = ids == id
        train_mask = np.logical_or(train_mask, train_tmp)

    print("Train: ", np.sum(id_mask_train), "(", np.sum(train_mask), ")")

    val_mask = np.zeros(len(ids), dtype=bool)
    id_mask_val = np.logical_and(groups >= int(0.8*K) - 1, groups <= int(0.9*K) - 1)
    val_ids = id_set[id_mask_val]
    print(f"Val ids: {val_ids}")
    for id in val_ids:
        val_tmp = ids == id
        val_mask = np.logical_or(val_mask, val_tmp)

    print("Val: ", np.sum(id_mask_val), "(", np.sum(val_mask), ")")

    test_mask = np.zeros(len(ids), dtype=bool)
    id_mask_test = np.logical_and(groups >=int(0.9*K) - 1, groups <= K - 1)
    test_ids = id_set[id_mask_test]
    print(f"Test ids: {test_ids}")
    for id in test_ids:
        test_tmp = ids == id
        test_mask = np.logical_or(test_mask, test_tmp)

    print("Test: ", np.sum(id_mask_test), "(", np.sum(test_mask), ")")

    masks = [train_mask, val_mask, test_mask]
    paths = [target_folder + "train/", target_folder + "val/", target_folder + "test/"]
    if bad_ch is not None:
        for i, m in enumerate(masks):
            path = paths[i]
            save_xyidst(x[m], y[m], ids[m], oz_t[m], speed[m], path, bad_ch=bad_ch[m], verbose=verbose)

    else:
        for i, m in enumerate(masks):
            path = paths[i]
            save_xyidst(x[m], y[m], ids[m], oz_t[m], speed[m], path, verbose=verbose)


if __name__ == "__main__":
    folder = "data/lessthan7/npy/"
    N = 5
    start_t = perf_counter()
    for i in range(N):
        x, y, ids, erp_t, speed, _  = load_xyidst(folder)

    end_t = perf_counter()

    del x
    del y
    del ids
    del erp_t
    del speed

    print(f"{N} iterations of load_xyidst took {end_t - start_t} seconds")


    start_t = perf_counter()
    for i in range(N):
        x, y, ids, erp_t, speed, _ = load_xyidst_threaded(folder)

    end_t = perf_counter()

    print(f"{N} iterations of load_xyidst_threaded took {end_t - start_t} seconds")


    folder = "data/greaterthan7/dummy/"

    start_t = perf_counter()
    for i in range(N):
        save_xyidst(x, y, ids, erp_t, speed, folder)

    end_t = perf_counter()

    print(f"{N} iterations of save_xyidst took {end_t - start_t} seconds")

    folder = "data/greaterthan7/dummythread/"

    start_t = perf_counter()
    for i in range(N):
        save_xyidst_threaded(x, y, ids, erp_t, speed, folder)

    end_t = perf_counter()

    print(f"{N} iterations of save_xyidst_threaded took {end_t - start_t} seconds")
