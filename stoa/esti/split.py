import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def split(split_time, predict_result, dataset_name,logger):

    data = np.array([row[2] for row in predict_result])
    hist, bin_edges = np.histogram(data, bins=100)
    # # visualization
    # benign_scores = [row[2] for row in predict_result if "trainClean" in row[3]]
    # poison_scores = [row[2] for row in predict_result if "trainClean" not in row[3]]
    #
    # benign_scores = np.array(benign_scores)
    # poison_scores = np.array(poison_scores)
    # import matplotlib.pyplot as plt
    # # 1. Compute common min/max over both
    # all_min = min(benign_scores.min(), poison_scores.min())
    # all_max = max(benign_scores.max(), poison_scores.max())
    #
    # # 2. Build shared bin edges (100 bins → 101 edges)
    # bins = np.linspace(all_min, all_max, 101)
    # plt.figure()
    # plt.hist(benign_scores, bins=bins, alpha=0.5, label="benign")
    # plt.hist(poison_scores, bins=bins, alpha=0.5, label="poison")
    # plt.xlabel("score (row[2])")
    # plt.ylabel("count")
    # plt.title("Histogram of scores for benign vs poison")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    if (dataset_name == 'gtsrb'):
        valid_bins = hist > len(predict_result) * 0.00005
    elif (dataset_name in ['cifar10', 'imagenette']):
        valid_bins = hist > len(predict_result) * 0.001
    if(split_time == 'TrapModelvEnd'):
        valid_bins = hist > len(predict_result) * 0.00005
    bin_indices = np.digitize(data, bin_edges) - 1
    bin_indices[bin_indices >= len(valid_bins)] = len(valid_bins) - 1
    filtered_data = data[valid_bins[bin_indices]]

    kde = gaussian_kde(filtered_data)
    x = np.linspace(filtered_data.min(), filtered_data.max(), 1000)
    pdf = kde(x)
    peaks_max, _ = find_peaks(pdf)
    peaks_min, _ = find_peaks(-pdf)
    logger.info(f"peaks_max\n{x[peaks_max]}")
    logger.info(f"peaks_min\n{x[peaks_min]}")

    if 'CTMv1' == split_time:
        threshold = x[peaks_max[-1]]
    elif 'CTM' in split_time:
        threshold = x[peaks_min[-1]]
    elif 'PTMv1' in split_time or 'Split_Clean' == split_time :
        threshold = x[peaks_min[0]]
    elif 'PTM' in split_time or 'TrapPre' in split_time:
        # threshold = np.max(filtered_data[filtered_data < 5])
        # assume filtered_data is a 1D numpy array
        threshold = x[peaks_min[0]]
    elif 'TrapModel' in split_time:
        # threshold = []
        # threshold.append(np.max(filtered_data[filtered_data < 5]))
        # threshold.append(np.min(filtered_data[filtered_data > 5]))
        threshold = x[peaks_min[0]]
    logger.info(f"threshold is:{threshold}")

    if 'CTM' in split_time:
        clean_pool_image_paths = [row[3] for row in predict_result if row[2] < threshold]
        poison_pool_image_paths = [row[3] for row in predict_result if row[2] > threshold]
    elif 'PTM' in split_time or 'Split_Clean' == split_time or 'TrapPre' in split_time:
        clean_pool_image_paths = [row[3] for row in predict_result if row[2] > threshold]
        poison_pool_image_paths = [row[3] for row in predict_result if row[2] < threshold]
    elif 'TrapModel' in split_time:
        clean_pool_image_paths = [row[3] for row in predict_result if row[2] < threshold]
        poison_pool_image_paths = [row[3] for row in predict_result if row[2] > threshold]

    logger.info(f"splited clean_pool_image_paths len:{len(clean_pool_image_paths)}")
    logger.info(f"splited poison_pool_image_paths len:{len(poison_pool_image_paths)}")
    return clean_pool_image_paths, poison_pool_image_paths, threshold