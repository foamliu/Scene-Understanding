import hdf5storage
import numpy as np

if __name__ == '__main__':
    filename = 'data/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat'
    SUNRGBD2Dseg = hdf5storage.loadmat(filename)
    num_samples = len(SUNRGBD2Dseg['SUNRGBD2Dseg'][0])

    lbl_counts = {}

    for i in range(num_samples):
        img = SUNRGBD2Dseg[0][i][0]
        id, counts = np.unique(img, return_counts=True)
        # normalize on image
        counts = counts / float(sum(counts))
        for j in range(len(id)):
            if id[j] in lbl_counts.keys():
                lbl_counts[id[j]] += counts[j]
            else:
                lbl_counts[id[j]] = counts[j]

    # normalize on training set
    for k in lbl_counts:
        lbl_counts[k] /= num_samples

    print("##########################")
    print("class probability:")
    for k in lbl_counts:
        print("%i: %f" % (k, lbl_counts[k]))
    print("##########################")

    # normalize on median freuqncy
    med_frequ = np.median(lbl_counts.values())
    lbl_weights = {}
    for k in lbl_counts:
        lbl_weights[k] = med_frequ / lbl_counts[k]

    print("##########################")
    print("median frequency balancing:")
    for k in lbl_counts:
        print("%i: %f" % (k, lbl_weights[k]))
    print("##########################")

    # class weight for classes that are not present in labeled image
    missing_class_weight = 100000

    max_class_id = np.max(lbl_weights.keys()) + 1

    # print formated output for caffe prototxt
    print("########################################################")
    print("### caffe SoftmaxWithLoss format #######################")
    print("########################################################")
    print("  loss_param: {\n"
          "    weight_by_label_freqs: true")
    # "\n    ignore_label: 0"
    for k in range(max_class_id):
        if k in lbl_weights:
            print("    class_weighting:", lbl_weights[k])
        else:
            print("    class_weighting:", missing_class_weight)
    print("  }")
    print("########################################################")
