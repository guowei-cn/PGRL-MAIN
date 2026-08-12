import numpy as np
from matplotlib import pyplot as plt
from utils import args

global arg
arg = args.get_args()

f_clean = arg.checkpoint_load
f_clean = f_clean[:-4] + '_clean.txt'
f_poison = arg.checkpoint_load
f_poison = f_poison[:-4] + '_poison.txt'

clean = np.loadtxt(f_clean)
poison = np.loadtxt(f_poison)

# calculate the auc where the gt label of clean and poison are 0 and 1
from sklearn.metrics import roc_auc_score

labels = np.concatenate((np.zeros_like(clean), np.ones_like(poison)))
scores = np.concatenate((clean, poison))
auc = roc_auc_score(labels, scores)
print(f"AUC: {auc}")

all = np.hstack((clean,poison))
# normalize to [0,1]
# all = (all - np.min(all)) / (np.max(all) - np.min(all))

num_bar = 100
x_axis = np.linspace(np.min(all), np.max(all), num=num_bar+1)
count_clean, count_poison = [], []
for i in range(x_axis.shape[0]-1):
    left = x_axis[i]
    right = x_axis[i+1]
    if i != x_axis.shape[0]-2:
        count_clean.append(np.sum(((clean >= left) & (clean < right))))
        count_poison.append(np.sum(((poison >= left) & (poison < right))))
    else:
        count_clean.append(np.sum(((clean >= left) & (clean <= right))))
        count_poison.append(np.sum(((poison >= left) & (poison <= right))))
count_clean, count_poison = np.array(count_clean), np.array(count_poison)
step = (np.max(all) - np.min(all)) / num_bar
x_axis = x_axis[:-1] + step/2
print(np.sum(count_poison) + np.sum(count_clean))

plt.figure()
plt.bar(x=x_axis, height=count_clean, width=step, color='g', alpha=0.5, label = 'clean samples')
plt.bar(x=x_axis, height=count_poison, width=step, color='r', alpha=0.5, label= 'poisoning samples')

plt.xlabel(r'$\Delta_{trans}(x;\tau,f)$')#, fontsize=20)
plt.ylabel('Num of Samples in Log Scale')#, fontsize=20)
plt.title(f'AUC: {auc:.4f}')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# y axis should be log scale
plt.yscale('log')
plt.legend(fontsize=20)
save_path = arg.checkpoint_load
save_path = save_path[:-4] + '.jpg'
print(save_path)
plt.savefig(save_path)
# plt.show()
