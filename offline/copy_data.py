import os
import shutil
import numpy as np
from tqdm import tqdm
from threading import Thread
import mne

error_codes = {
    0: "Couldn't read stm- events from event file",
    1: "Not valid due to no Oz comments, check upper loweer case!!!!",
    2: "Not valid due to error reading .raw file",
    3: "Not valid due to wrong sampling frequency",
    4: "Automatically approved due to same filename",
    5: "Not valid due to timing differences",
    6: "Validated by comparing timing",
}

def get_timestamps_evt(fname, sfreq, event="Oz"):
    """get timestamps as sample number from .evt file"""
    """
        only used by copy_data
    """
    inputfile = open(fname, "r")
    timestamps = []

    for index, line in enumerate(inputfile):
        if index == 0:
            continue
        if event.lower() in line.lower():
            chunks = line.split(" ")
            try:
                tmu = int(chunks[0])
            except ValueError:
                chunks = line.split("\t")
                try:
                    tmu = int(chunks[0])
                except ValueError:
                    print(f"{fname}: valueerror: {chunks[0]} can not be converted to int")
                    continue

            t = int(tmu * sfreq / 1e6)

            timestamps.append(t)

    if not timestamps:
        return None

    return np.asarray(timestamps)


def get_timestamps_raw(fname, event="stm-"):
    """get timestamps as sample number from .raw file, returned as numpy array"""
    """
        only used by copy_data
    """
    triggers = ["stm+"]

    try:
        egi = mne.io.read_raw_egi(fname, exclude=triggers, verbose="WARNING")
    except:
        print("Read raw EGI failed when getting timestamps from raw file.")
        return None, 0

    ch_names_idx = {}
    for i, ch_name in enumerate(egi.ch_names):
        ch_names_idx[ch_name] = i

    _coll_events = egi.get_data(picks=[ch_names_idx[event]])
    coll_events = _coll_events.astype(int)
    coll_mask = coll_events != 0
    coll_sample = np.where(coll_mask)[1]
    sfreq = egi.info["sfreq"]
    return coll_sample, sfreq



def validate_files(raw_f, raw_e, raw_dict, parent_folder):
    raw_file = parent_folder + raw_f + ".raw"
    evt_file = parent_folder + raw_e + ".evt"
    raw_t, sfreq = None, 0
    if raw_f in raw_dict.keys():
        raw_t, sfreq = raw_dict[raw_f]
    else:
        raw_t, sfreq = get_timestamps_raw(raw_file, event="stm-")

    if raw_t is None or sfreq == 0:
        print(f"{raw_file}: raw_t is None or sfreq == 0")
        return False, raw_dict
    else:
        raw_dict[raw_f] = raw_t, sfreq

    """ evt_t = get_timestamps_evt(evt_file, sfreq, event="stm-")
    if evt_t is None:
        print(f"{raw_file}: Couldn't read stm- events from event file")
        return False, raw_dict """

    oz_t = get_timestamps_evt(evt_file, sfreq, event="Oz")
    if oz_t is None:
        print(f"{raw_file}: Not valid due to no Oz comments")
        return False, raw_dict

    if sfreq > 500.1 or sfreq < 499.9:
        print(f"{raw_file}: Not valid due to sampling frequency: {sfreq}")
        return False, raw_dict

    """ timing_good = False
    raw_l = raw_t.shape[0]
    evt_l = evt_t.shape[0]

    if raw_l == evt_l:
        diff = np.absolute(raw_t - evt_t)
        equal = diff == 0
        if np.all(equal):
            timing_good = True

    if not timing_good:
        print(f"{raw_file}: discrepency in timing")
        return False, raw_dict """

    return True, raw_dict


def copy_files(parent_folder, destination_folder, fnames_include):
    raw = []
    evt = []
    raw_dict = {}
    print(f"Beginning read of files in {parent_folder}\n")
    for d in os.listdir(parent_folder):
        tmp_raw = []
        tmp_evt = []
        if not os.path.isdir(parent_folder + d) or "preterm" in d.lower():
            continue
        for f in os.listdir(parent_folder + d):
            file_name, file_ext = os.path.splitext(f)

            if file_ext == ".raw":
                if file_name in fnames_include:
                    tmp_raw.append(d + "/" + file_name)
                    print(f"Found file to copy: {file_name}")

            elif file_ext == ".evt":
                if file_name in fnames_include:
                    tmp_evt.append(d + "/" + file_name)
                    print(f"Found file to copy: {file_name}")

        hope = True
        l_raw = len(tmp_raw)
        l_evt = len(tmp_evt)
        if l_evt == 0:
            hope = False
        if l_raw == 0:
            hope = False
        file_count = 0
        while hope:
            l_raw = len(tmp_raw)
            l_evt = len(tmp_evt)
            if l_raw == 0 or l_evt == 0:
                hope = False

            else:
                file_added = False
                for i in range(l_raw):
                    for j in range(l_evt):
                        b, raw_dict = validate_files(
                            tmp_raw[i], tmp_evt[j], raw_dict, parent_folder
                        )
                        if b:
                            raw.append(tmp_raw.pop(i))
                            evt.append(tmp_evt.pop(j))
                            file_count += 1
                            file_added = True
                            break

                    if file_added:
                        break

                if not file_added:
                    hope = False

    print(f"Starting copying of {len(raw)} files")
    for i, r in enumerate(raw):
        rs = r.split("/")
        shutil.copyfile(parent_folder + r + ".raw", destination_folder + rs[1] + ".raw")
        shutil.copyfile(
            parent_folder + evt[i] + ".evt", destination_folder + rs[1] + ".evt"
        )

    print(f"Successfully copied {len(raw)} files")


if __name__ == "__main__":
    ages = ["lessthan7", "greaterthan7"]
    for age in ages:
        sort_key = "younger than" if age == "lessthan7" else "older than"

        babies_file = (os.path.dirname(__file__) or '.') + "/data/" + age + "/baby_files.txt"
        fnames_include = []
        with open(babies_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                for fname in line.split(","):
                    fnames_include.append(fname.strip())

        print(fnames_include)

        if os.name == 'posix':
            root = '/run/user/1000/gvfs/smb-share:server=felles.ansatt.ntnu.no,share=ntnu/su/ips/nullab/'
        else:
            root = 'T:/'
        root += "Analysis/EEG/Looming/Silje-Adelen/3. BCI Annotation (Silje-Adelen)/1. Infant VEP Annotations/1 )Annotated_Silje"
        target_folder = (os.path.dirname(__file__) or '.') + "/data/" + age + "/raw/"
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)
        threads = []
        for d1 in os.listdir(root):
            root1 = os.path.join(root, d1)
            if not os.path.isdir(root1):
                continue

            for d2 in os.listdir(root1):
                root2 = root1 + "/" + d2 + "/"
                if not sort_key in d2.lower() or not os.path.isdir(root2):
                    continue

                t = Thread(target=copy_files, args=(root2, target_folder, fnames_include))
                threads.append(t)
                t.start()
                print(f"Thread {len(threads)} started")

        for i, t in enumerate(threads):
            t.join()
            print(f"Thread {i+1} joined")
