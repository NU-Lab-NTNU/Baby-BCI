import os
import shutil
import numpy as np
from data_finder import get_timestamps_evt, get_timestamps_raw
from tqdm import tqdm
from threading import Thread

error_codes = {
    0: "Couldn't read stm- events from event file",
    1: "Not valid due to no Oz comments, check upper loweer case!!!!",
    2: "Not valid due to error reading .raw file",
    3: "Not valid due to wrong sampling frequency",
    4: "Automatically approved due to same filename",
    5: "Not valid due to timing differences",
    6: "Validated by comparing timing"
}

def validate_files(raw_f, raw_e, raw_dict, parent_folder):
    raw_file = parent_folder + raw_f + '.raw'
    evt_file = parent_folder + raw_e + '.evt'
    raw_t, sfreq = None, 0
    if raw_f in raw_dict.keys():
        raw_t, sfreq = raw_dict[raw_f]
    else:
        raw_t, sfreq = get_timestamps_raw(raw_file, event="stm-")

    if raw_t is None or sfreq == 0:
        return False, raw_dict
    else:
        raw_dict[raw_f] = raw_t, sfreq

    evt_t = get_timestamps_evt(evt_file, sfreq, event="stm-")
    if evt_t is None:
        #print("Couldn't read stm- events from event file")
        return False, raw_dict

    oz_t = get_timestamps_evt(evt_file, sfreq, event="Oz")
    if oz_t is None:
        #print("Not valid due to no Oz comments")
        return False, raw_dict

    if sfreq > 500.1 or sfreq < 499.9:
        #print("Not valid due to sampling frequency: ", sfreq)
        if sfreq < 249.9 or sfreq > 250.1:
            return False, raw_dict


    timing_good = False
    raw_l = raw_t.shape[0]
    evt_l = evt_t.shape[0]

    if raw_l == evt_l:
        diff = np.absolute(raw_t - evt_t)
        equal = diff == 0
        if np.all(equal):
            timing_good = True

    if not timing_good:
        return False, raw_dict

    return True, raw_dict


def copy_files(parent_folder, destination_folder):
    raw = []
    evt = []
    raw_dict = {}
    print("Beginning read of files in ", parent_folder)
    for d in os.listdir(parent_folder):
        tmp_raw = []
        tmp_evt = []
        if not os.path.isdir(parent_folder+d) or "preterm" in d.lower():
            continue
        for f in os.listdir(parent_folder + d):
            file_name, file_ext = os.path.splitext(f)
            if file_ext == '.raw':
                tmp_raw.append(d+'/'+file_name)
            if file_ext == '.evt':
                tmp_evt.append(d+'/'+file_name)


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
                        b, raw_dict = validate_files(tmp_raw[i], tmp_evt[j], raw_dict, parent_folder)
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




    print("\nStarting copying of ", len(raw), " files")
    for i, r in enumerate(raw):
        rs = r.split('/')
        shutil.copyfile(parent_folder+r+'.raw', destination_folder+rs[1]+'.raw')
        shutil.copyfile(parent_folder+evt[i]+'.evt', destination_folder+rs[1]+'.evt')

    print("Successfully copied ", len(raw), " files")

if __name__ == "__main__":
    print("young")
    root = "T:/su/ips/Nullab/Analysis/EEG/looming/Silje-Adelen/Infant VEP Annotations/1 )Annotated_Silje"
    target_folder = "C:/Users/vegardkb/NU-BCI/offline/data/lessthan7/raw/"
    threads = []
    for d1 in os.listdir(root):
        root1 = os.path.join(root, d1)
        if not os.path.isdir(root1):
            continue

        for d2 in os.listdir(root1):
            root2 = root1 + '/' + d2 + '/'
            if not "younger than" in d2.lower() or not os.path.isdir(root2):
                continue

            t = Thread(target=copy_files, args=(root2, target_folder))
            threads.append(t)
            t.start()
            print(f"Thread {len(threads)} started")

    for i, t in enumerate(threads):
        t.join()
        print(f"Thread {i+1} joined")
