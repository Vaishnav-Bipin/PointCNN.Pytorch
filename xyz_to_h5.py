import h5py
import numpy as np
import os
import json
import sys


def main(old_pc_base, new_h5_base):
    # ********
    F_PER_H5 = 2048
    N_POINTS = 2048

    categories = "3d-printing,aerospace,agriculture,architecture,automotive,aviation,components,computer,construction,educational,electrical,energy-and-power,fixtures,furniture,hobby,household,industrial-design,interior-design,jewellery,just-for-fun,machine-design,marine,medical,military,miscellaneous,nature,piping,robotics,speedrun,sport,tech,tools,toys"
    cat_arr = categories.split(",")

    np.random.seed(50)

    base = old_pc_base
    # ********

    n_files = 0
    rdf_all = sorted(list(os.walk(base)))
    f_all = []

    for root, dirs, files in rdf_all:
        pth_arr = root.split("/")
        t = pth_arr[-1]
        if t != 'train' and t != 'test':
            continue
        
        sfs = sorted(files)
        n_files += len(files) // 2
        c = cat_arr.index(pth_arr[-2])

        f_names = [(os.path.join(root, f), os.path.join(root, f[:-6] + "_n.xyz"), t, c) for f in sfs if f[-6:] == "_p.xyz"]
        f_all += f_names


    def create_h5s(base, f_all):
        if not os.path.exists(base):
            os.mkdir(base)
        # global F_PER_H5
        # global N_POINTS
        f_train =  [(f_p, f_n, t, c) for (f_p, f_n, t, c) in f_all if t == 'train']
        f_test = [(f_p, f_n, t, c) for (f_p, f_n, t, c) in f_all if t == 'test']

        n_train = int(len(f_train))
        perm = np.random.permutation(n_train)
        n_left = n_train
        i = -1
        idx_f = 0
        out_fs_train = []
        while n_left != 0:
            print(n_left)
            n_h5 = min(F_PER_H5, n_left)

            all_matp = np.zeros((n_h5, N_POINTS, 3))
            all_matn = np.zeros((n_h5, N_POINTS, 3))
            all_matl = np.zeros((n_h5, 1))
            id2_fs = []
            for j in range(n_h5):
                i += 1
                idx = perm[i]
                f_p, f_n, t, c = f_train[idx]

                matp = np.loadtxt(f_p)
                max_norm = -np.inf
                for k in range(N_POINTS):
                    q = matp[k, :]
                    nrm = np.linalg.norm(q)
                    if nrm > max_norm:
                        max_norm = nrm
                matp = matp / max_norm

                matn = np.loadtxt(f_n)

                all_matp[j, :, :] = matp
                all_matn[j, :, :] = matn
                all_matl[j] = c
                
                f_arr = f_p.split("/")
                l = 1 + len(f_arr[1]) + 1 + len(f_arr[2]) + 1 + len(f_arr[3]) + 1
                id2_fs.append(f_p[l:])

            out_f = 'ply_data_train' + str(idx_f) + '.h5'
            out_f_pth = os.path.join(base, out_f)
            out_fs_train.append(out_f_pth)
            
            h5 = h5py.File(out_f_pth, 'w')
            h5.create_dataset('data', data=all_matp)
            h5.create_dataset('normal', data=all_matn)
            h5.create_dataset('label', data=all_matl)
            h5.close()

            id2_f = os.path.join(base, 'ply_data_train_' + str(idx_f) + '_id2file.json')
            with open(id2_f, 'w') as f:
                pass
            with open(id2_f, 'a') as f:
                json.dump(id2_fs, f)
                f.write("\n")

            idx_f += 1
            n_left -= n_h5




        n_test = int(len(f_test))
        perm = np.random.permutation(n_test)
        n_left = n_test
        i = -1
        idx_f = 0
        out_fs_test = []
        while n_left != 0:
            print(n_left)
            n_h5 = min(F_PER_H5, n_left)

            all_matp = np.zeros((n_h5, N_POINTS, 3))
            all_matn = np.zeros((n_h5, N_POINTS, 3))
            all_matl = np.zeros((n_h5, 1))
            id2_fs = []
            for j in range(n_h5):
                i += 1
                idx = perm[i]
                f_p, f_n, t, c = f_test[idx]

                matp = np.loadtxt(f_p)
                max_norm = -np.inf
                for k in range(N_POINTS):
                    q = matp[k, :]
                    nrm = np.linalg.norm(q)
                    if nrm > max_norm:
                        max_norm = nrm
                matp = matp / max_norm

                matn = np.loadtxt(f_n)

                all_matp[j, :, :] = matp
                all_matn[j, :, :] = matn
                all_matl[j] = c

                f_arr = f_p.split("/")
                l = 1 + len(f_arr[1]) + 1 + len(f_arr[2]) + 1 + len(f_arr[3]) + 1
                id2_fs.append(f_p[l:])

            out_f = 'ply_data_test' + str(idx_f) + '.h5'
            out_f_pth = os.path.join(base, out_f)
            out_fs_test.append(out_f_pth)
            
            h5 = h5py.File(out_f_pth, 'w')
            h5.create_dataset('data', data=all_matp)
            h5.create_dataset('normal', data=all_matn)
            h5.create_dataset('label', data=all_matl)
            h5.close()
            
            id2_f = os.path.join(base, 'ply_data_test_' + str(idx_f) + '_id2file.json')
            with open(id2_f, 'w') as f:
                pass
            with open(id2_f, 'a') as f:
                json.dump(id2_fs, f)
                f.write("\n")


            idx_f += 1
            n_left -= n_h5

        
        with open(os.path.join(base, 'train_files.txt'), 'w') as f:
            for line in out_fs_train:
                f.write(f"{line}\n")
        with open(os.path.join(base, 'test_files.txt'), 'w') as f:
            for line in out_fs_test:
                f.write(f"{line}\n")

    create_h5s(new_h5_base, f_all)
        

if __name__ == "__main__":
    if sys.argv[1] == "100":
        old_pc_base = "/home/vaishnav/PC_files"
        new_h5_base = '/home/vaishnav/Benchmarks/PointCNN.Pytorch/data/GrabCad100K_hdf5_2048'
        main(old_pc_base, new_h5_base)
    if sys.argv[1] == "50":
        old_pc_base = "/home/vaishnav/PC_files_50"
        new_h5_base = '/home/vaishnav/Benchmarks/PointCNN.Pytorch/data/GrabCad50K_hdf5_2048'
        main(old_pc_base, new_h5_base)

