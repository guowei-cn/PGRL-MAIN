import matplotlib.pyplot as plt
from adjustText import adjust_text

# Data (same as provided)
method_name = ['ABL', 'PIPD', 'ASD', 'CBD', 'ESTI', 'DBD', 'EBD',  'VaB', 'LCV', 'FDE', 'PGRL'] #
# Predefined colors and markers
colors  = ['grey', 'grey', 'grey', 'grey', 'grey', 'blue', 'blue', 'blue', 'red', 'red', 'red']
markers = ['o',  # circle
           's',  # square
           '^',  # triangle_up
           'v',  # triangle_down
           '<',  # triangle_left
           '>',  # triangle_right
           'D',  # diamond
           'p',  # pentagon
           'P',  # hexagon1
           'X',  # x (cross)
           '*']  # star

# pattern_003_ACC_ASR = [(0.87, 1.00), (0.88, 0.99), (0.91, 0.99), (0.90, 0.016), (0.90, 0.093),
#                        (0.86, 0.11), (0.77, 0.63), (0.92, 0.77), (0.92, 0.97), (0.92, 0.01), (0.91, 0.004)] #
# pattern_05_ACC_ASR = [(0.86, 0.00), (0.88, 0.00), (0.93, 0.006), (0.91, 0.038), (0.93, 0.00),
#                       (0.88, 0.09), (0.85, 0.09), (0.92, 0.04), (0.92, 0.99), (0.91, 0.01), (0.92, 0.006)] #
# adapblend_003_ACC_ASR = [(0.91, 0.82), (0.93, 0.87), (0.86, 0.77), (0.90, 0.01), (0.93, 0.58),
#                          (0.68, 0.64), (0.74, 0.65), (0.91, 0.84), (0.92, 0.05), (0.94, 0.82), (0.91, 0.006)] #
# adapblend_05_ACC_ASR = [(0.85, 0.93), (0.93, 0.84), (0.93, 0.72), (0.90, 0.08), (0.92, 0.69),
#                         (0.78, 0.95), (0.73, 0.82), (0.91, 0.86), (0.92, 0.08), (0.92, 0.01), (0.93, 0.009)] #
# #ultrasonic_003_ACC_ASR = [(0.93, 0.002), (0.97, 0.99), (0.97, 1.00), (0.97, 0.95), (0.89, 0.10), (0.96, 1.00),
# #                          (0.86, 0.10), (0.94, 0.91), (0.95, 0.99), (0.94, 0.07)] #  (),
# #ultrasonic_05_ACC_ASR = [(0.93, 0.012), (0.96, 0.18), (0.90, 0.12), (0.93, 0.10), (0.86, 0.11), (0.95, 1.00),
# #                         (0.84, 0.09), (0.92, 0.13), (0.95, 0.99), (0.95, 0.01)] #  (),
# # ultrasonic_05_ACC_ASR = [(0.93, 0.012), (0.97, 1.00), (0.96, 0.99), (0.96, 1.00), (0.86, 0.11), (0.95, 1.00),
# #                          (0.84, 0.09), (0.93, 1.00), (), (), ()]
# freq_500_003_ACC_ASR = [(0.87, 0.82), (0.87, 0.91), (0.92, 0.75), (0.91, 0.89), (0.92, 0.75),
#                         (0.79, 0.10), (0.87, 0.96), (0.92, 0.98), (0.93, 0.70), (0.92, 0.09), (0.92, 0.007)] #
# freq_500_05_ACC_ASR = [(0.88, 0.04), (0.87, 0.08), (0.88, 0.00), (0.90, 0.95), (0.92, 0.95),
#                        (0.84, 0.10), (0.86, 0.11), (0.92, 0.97), (0.92, 0.91), (0.91, 0.01), (0.92, 0.006)] #

# pattern in cifar10
# pattern_003_ACC_ASR = [
#     (0.87, 1.00), (0.88, 0.99), (0.91, 0.99),
#     (0.77, 0.63),  # moved from 5th-from-last
#     (0.90, 0.016), (0.90, 0.093), (0.86, 0.11),
#     (0.92, 0.77), (0.92, 0.97), (0.92, 0.01), (0.91, 0.004)
# ]
#
# pattern_05_ACC_ASR = [
#     (0.86, 0.00), (0.88, 0.00), (0.93, 0.006),
#     (0.85, 0.09),  # moved
#     (0.91, 0.038), (0.93, 0.00), (0.88, 0.09),
#     (0.92, 0.04), (0.92, 0.99), (0.91, 0.01), (0.92, 0.006)
# ]
# pattern in imagenette
pattern_003_ACC_ASR = [
    (0.889, 0.910), (0.91, 0.990), (0.876,0.972),
    (0.82, 0.83),  (0.908, 1.0),# moved from 5th-from-last
    (0.905, 0.018), (0.8611, 0.070), #CT(0.75, 0.02),
    (0.87, 0.69), (0.910, 1.00), (0.90, 0.024), (0.910, 0.004) # ESTI (0.908, 1.0)
]

pattern_05_ACC_ASR = [
    (0.895,0.053), (0.88,0.040), (0.8858, 0.099),
    (0.811, 0,.087), (0.909, 0.0), # moved
    (0.893, 0.027), (0.86, 0.065), # CT0.80, 0.05),
    (0.88, 0.08),  (0.910, 1.00),  (0.905, 0.021), (0.910,0.001) # ESTI (0.909, 0.0)
]

adapblend_003_ACC_ASR = [
    (0.83, 0.92), (0.88, 0.933), (0.86, 0.71),
    (0.80, 0.945), (0.904, 0.34),
    (0.85, 0.03), (0.80, 0.63), #CT(0.68, 0.61),
    (0.85, 0.61), (0.91, 0.03), (0.88, 0.54), (0.90, 0.04)
] # ESTI (0.904, 0.34)

adapblend_05_ACC_ASR = [
    (0.77, 0.91), (0.85, 0.941), (0.77, 0.90),
    (0.77, 0.68), (0.908, 0.928),
    (0.88, 0.05), (0.85, 0.82), # CT(0.72, 0.91),
    (0.834, 0.65), (0.90, 0.05), (0.90, 0.84), (0.90, 0.03)
] # ESTI (0.908, 0.928)

freq_500_003_ACC_ASR = [
    (0.79, 0.68), (0.83, 0.832), (0.84, 0.0),
    (0.79, 0.38), (0.905, 0.0),
    (0.87, 0.72), (0.78, 0.42), #CT (0.71, 0.82),
    (0.81, 0.73), (0.90, 0.58), (0.89, 0.002), (0.90, 0.01)
] # ESTI (0.905, 0.0)

freq_500_05_ACC_ASR = [
    (0.88, 0.001), (0.885, 0.02), (0.86, 0.003),
    (0.78, 0.10), (0.905, 0.0),
    (0.87, 0.82), (0.82, 0.85), #CT (0.79, 0.08),
    (0.80, 0.57), (0.89, 0.91), (0.90, 0.003), (0.90, 0.01)
] # ESTI (0.905, 0.0)

# # Predefined colors and markers for the points in each plot
# colors = ['red', 'grey', 'grey', 'grey', 'green', 'orange', 'blue', 'grey', 'purple', 'red', 'red']
# markers = ['o', 'p', 'p', 'p', 'D', '^', 'v', 'p', '*', 'o', 'o'] # ['o', 'p', 'P', 's', 'D', '^', '+', 'v', '*', 'o', 'o']
#
# # Create subplots
# fig, axes = plt.subplots(2, 3, figsize=(12, 5))
#
# # Reorder data to swap second and third columns (i.e., first row, second and third columns; second row, second and third columns)
# data = [
#     (pattern_05_ACC_ASR, r"Pattern $\alpha=$0.05 "),
#     #(ultrasonic_05_ACC_ASR, r"Ultrasonic $\alpha=$0.05"),  # Swap these
#     (adapblend_05_ACC_ASR, r"AdapBlend $\alpha=$0.05"),    # Swap these
#     (freq_500_05_ACC_ASR, r"Freq $\alpha=$0.05"),
#     (pattern_003_ACC_ASR, r"Pattern $\alpha=$0.003"),
#     #(ultrasonic_003_ACC_ASR, r"Ultrasonic $\alpha=$0.003"),  # Swap these
#     (adapblend_003_ACC_ASR, r"AdapBlend $\alpha=$0.003"),    # Swap these
#     (freq_500_003_ACC_ASR, r"Freq $\alpha=$0.003")
# ]
#
# # Loop through the data and axes to plot each
# for i, (acc_asr_data, title) in enumerate(data):
#     ax = axes[i // 3, i % 3]
#     acc, asr = zip(*acc_asr_data)
#
#     texts = []
#     for j in range(len(acc_asr_data)):
#         ax.scatter(asr[j], acc[j], color=colors[j], marker=markers[j])
#
#         # Slight offset helps reduce initial overlaps
#         text = ax.text(asr[j] + 0.01, acc[j] + 0.01, method_name[j], fontsize=12)
#         texts.append(text)
#
#     ax.set_title(title)
#     ax.set_ylim(0.66, 1.05)
#     ax.set_xlim(-0.02, 1.15)
#
#     adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', lw=1.0))
#
# plt.subplots_adjust(wspace=0.05, hspace=0.2)
# plt.tight_layout()
# plt.savefig('tpr_fpr.png', dpi=800)
# plt.close()




# Prepare figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 5))

data = [
    (pattern_05_ACC_ASR,   r"Pattern $\alpha=0.05$"),
    (adapblend_05_ACC_ASR, r"AdapBlend $\alpha=0.05$"),
    (freq_500_05_ACC_ASR,  r"Freq $\alpha=0.05$"),
    (pattern_003_ACC_ASR,  r"Pattern $\alpha=0.003$"),
    (adapblend_003_ACC_ASR,r"AdapBlend $\alpha=0.003$"),
    (freq_500_003_ACC_ASR, r"Freq $\alpha=0.003$"),
]

# Build one set of legend handles
legend_handles = [
    plt.Line2D([0], [0],
               marker=markers[j],
               color='w',
               markerfacecolor=colors[j],
               label=label,
               markersize=8)
    for j, label in enumerate(method_name)
]

# Plot each subplot without text annotations
for i, (acc_asr_data, title) in enumerate(data):
    ax = axes[i // 3, i % 3]
    acc, asr = zip(*acc_asr_data)

    for j in range(len(acc_asr_data)):
        ax.scatter(asr[j], acc[j],
                   color=colors[j],
                   marker=markers[j])

    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.66, 1.02)

# Add a single legend below all subplots
fig.legend(handles=legend_handles,
           loc='lower center',
           ncol=11,
           fontsize=10)

# Tidy up spacing (reserve space for legend)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.subplots_adjust(wspace=0.1, hspace=0.3)

plt.savefig('imagenette_asr_acc.png', dpi=800)
plt.close()






# import matplotlib.pyplot as plt
# from adjustText import adjust_text
#
# def plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name):
#     # Define distinct predefined colors for each method
#     colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'grey', 'cyan']  # Distinct predefined colors
#     markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'P', 'h']
#
#     # Prepare data for TPR vs FPR plot
#     FPR = []  # X-axis values (FPR)
#     TPR = []  # Y-axis values (TPR)
#     valid_methods_tpr_fpr = []
#     valid_markers_tpr_fpr = []
#
#     # Filter out 'NA' entries and extract FPR, TPR values for the plot
#     for i, point in enumerate(TPR_FPR):
#         if point != 'NA':
#             TPR.append(point[0])  # First element is TPR (y-axis)
#             FPR.append(point[1])  # Second element is FPR (x-axis)
#             valid_methods_tpr_fpr.append(method_name[i])
#             valid_markers_tpr_fpr.append(markers[i])
#
#     # Prepare data for ASR vs ACC plot
#     ASR = [point[1] for point in ACC_ASR]  # X-axis values (ASR)
#     ACC = [point[0] for point in ACC_ASR]  # Y-axis values (ACC)
#
#     # Create the plots
#     plt.figure(figsize=(12, 6))  # Width to height ratio adjusted for better layout
#
#     # First subplot: TPR vs FPR
#     ax1 = plt.subplot(1, 2, 1)
#     texts = []  # List to store text annotations for adjustText
#     for i, method in enumerate(valid_methods_tpr_fpr):
#         plt.scatter(FPR[i], TPR[i], marker=markers[i], color=colors[i],  # Fill the markers
#                     edgecolor='black', label=method)
#         texts.append(plt.text(FPR[i], TPR[i], method, fontsize=9, ha='left', va='bottom'))  # Add to list for adjustment
#     plt.title('TPR vs FPR')
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')
#     plt.xlim(-0.1, 1.1)
#     plt.ylim(-0.1, 1.1)
#     adjust_text(
#         texts,
#         expand_text=(2.2, 1.4),  # Push the text further away from each other
#         expand_points=(1.5, 1.7),  # Push the text further away from points
#         force_points=(0.3, 0.5),  # Force text not to overlap markers
#         force_text=(0.3, 0.5),  # Force text not to overlap other text
#         arrowprops=dict(arrowstyle='-', lw=0.5)  # Add arrows for clarity
#     )
#     ax1.set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
#
#     # Second subplot: ASR vs ACC
#     ax2 = plt.subplot(1, 2, 2)
#     texts = []  # Reset text annotations list for second plot
#     for i, method in enumerate(method_name):
#         plt.scatter(ASR[i], ACC[i], marker=markers[i], color=colors[i],  # Fill the markers
#                     edgecolor='black', label=method)
#         texts.append(plt.text(ASR[i], ACC[i], method, fontsize=9, ha='left', va='bottom'))  # Add to list for adjustment
#     plt.title('ASR vs ACC')
#     plt.xlabel('ASR')
#     plt.ylabel('ACC')
#     plt.xlim(-0.02, 1.0)
#
#     # Dynamically adjust the minimum y-axis limit for ACC
#     min_y_limit = max(0, min(ACC) - 0.02)  # Ensure y-axis limit does not go below 0
#     plt.ylim(min_y_limit, 1.0)
#
#     adjust_text(
#         texts,
#         expand_text=(2.2, 1.4),  # Push the text further away from each other
#         expand_points=(1.5, 1.7),  # Push the text further away from points
#         force_points=(0.3, 0.5),  # Force text not to overlap markers
#         force_text=(0.3, 0.5),  # Force text not to overlap other text
#         arrowprops=dict(arrowstyle='-', lw=0.5)  # Add arrows for clarity
#     )
#     ax2.set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
#
#     # Add a legend outside the left center of the second plot
#     plt.legend(loc='center right', bbox_to_anchor=(-0.15, 0.5), ncol=1, frameon=False)
#
#     plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to leave space for the legend
#     plt.savefig(save_name, dpi=600)
#     plt.close()
#
#
# # pattern 0.003
# method_name = ['OUR', 'ABL', 'PIPD', 'ASD', 'DBD', 'EBD', 'FPP', 'CBD']
# # TPR_FPR = [(1., 0.03), (0.02, 0.05), (0, 0.02), (0.12, 0.50), (1.00, 0.49), (1.00, 0.79), (1.00, 0.41), 'NA']
# pattern_003_ACC_ASR = [(0.91, 0.004), (0.87, 1.00), (0.88, 0.99), (0.91, 0.99), (0.90, 0.016), (0.90, 0.093), (0.86, 0.11), (0.77, 0.63)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_pattern_0.003.png')
#
# # pattern 0.05
# # TPR_FPR = [(0.99, 0.02), (0.95, 0.01), (0.95, 0.00), (1.00, 0.47), (0.99, 0.47), (1.00, 0.082), (1.00, 0.23), 'NA']
# pattern_05_ACC_ASR = [(0.92, 0.006), (0.84, 0.00), (0.85, 0.00), (0.93, 0.006), (0.91, 0.038), (0.93, 0.00), (0.88, 0.09), (0.83, 0.09)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_pattern_0.05.png')
#
# # adaptivecifar10 0.003
# # TPR_FPR = [(0.86, 0.02), (0.00, 0.05), (0., 0.00), (0.33, 0.50), (1.00, 0.50), (0.21, 0.19), (0.64, 0.69), 'NA']
# adaptivecifar10_003_ACC_ASR = [(0.91, 0.006), (0.91, 0.82), (0.93, 0.87), (0.86, 0.77), (0.90, 0.01), (0.93, 0.58), (0.68, 0.64), (0.74, 0.65)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_adaptivecifar10_0.003.png')
#
# # adaptivecifar10 0.05
# # TPR_FPR = [(0.96, 0.03), (0.00, 0.05), (0., 0.00), (0.39, 0.51), (0.97, 0.47), (0.29, 0.27), (0.91, 0.62), 'NA']
# adaptivecifar10_05_ACC_ASR = [(0.93, 0.009), (0.85, 0.93), (0.93, 0.84), (0.93, 0.72), (0.90, 0.08), (0.92, 0.69), (0.78, 0.95), (0.73, 0.82)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_adaptivecifar10_0.05.png')
#
# # ultrasonic 0.003
# # TPR_FPR = [(1.00, 0.082), (0.00, 0.05), (0., 0.00), (0.48, 0.50), (1.00, 0.49), (0.00, 0.00), (0.34, 0.15), 'NA']
# ultrasonic_003_ACC_ASR = [(0.93, 0.002), (0.97, 0.99), (0.97, 1.00), (0.97, 0.95), (0.89, 0.10), (0.96, 1.00), (0.86, 0.10), (0.94, 0.91)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_ultrasonic_0.003.png')
#
# # ultrasonic 0.05
# # TPR_FPR = [(0.99, 0.06), (0.06, 0.047), (0., 0.00), (0.01, 0.53), (0.95, 0.47), (0.00, 0.00), (0.67, 0.17), 'NA']
# ultrasonic_05_ACC_ASR = [(0.93, 0.012), (0.97, 1.00), (0.96, 0.99), (0.96, 1.00), (0.86, 0.11), (0.95, 1.00), (0.84, 0.09), (0.93, 1.00)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_ultrasonic_0.05.png')
#
# # freq_meg_500 0.003
# # TPR_FPR = [(0.93, 0.03), (0.006, 0.05), (0., 0.019), (1.00, 0.49), (0.64, 0.49), (0.91, 0.80), (0.73, 0.67), 'NA']
# freq_meg_500_003_ACC_ASR = [(0.92, 0.007), (0.87, 0.82), (0.87, 0.91), (0.92, 0.75), (0.91, 0.89), (0.92, 0.75), (0.79, 0.10), (0.87, 0.96)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_freq_meg_500_0.003.png')
#
# # freq_meg_500 0.05
# # TPR_FPR = [(0.93, 0.03), (0.93, 0.003), (0.91, 0.00), (0.91, 0.49), (0.30, 0.51), (0.30, 0.18), (1.00, 0.56), 'NA']
# freq_meg_500_05_ACC_ASR = [(0.92, 0.006), (0.88, 0.04), (0.87, 0.08), (0.88, 0.00), (0.90, 0.95), (0.92, 0.95), (0.84, 0.10), (0.86, 0.11)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_freq_meg_500_0.05.png')
#
#
#
# # # pattern 0.003
# # method_name = ['OUR', 'ABL', 'PIPD', 'ASD', 'CBD', 'DBD', 'EBD', 'FPP', 'NONE']
# # TPR_FPR = [(1., 0.03), (0.02, 0.05), (0, 0.02), (0.12, 0.50), 'NA', (1.00, 0.49), (1.00, 0.79), (1.00, 0.41), (1.00, 0.)]
# # ACC_ASR = [(0.91, 0.06), (0.87, 1.00), (0.88, 0.99), (0.91, 0.99), (0.77, 0.63), (0.90, 0.016), (0.90, 0.093), (0.86, 0.11), (0.92, 0.08)]
# #
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_pattern_0.003.png')
# #
# # # pattern 0.05
# # TPR_FPR = [(0.99, 0.02), (0.95, 0.01), (0.95, 0.00), (1.00, 0.47), 'NA', (0.99, 0.47), (1.00, 0.082), (1.00, 0.23), (1.00, 0.)]
# # ACC_ASR = [(0.92, 0.10), (0.84, 0.00), (0.85, 0.00), (0.93, 0.006), (0.83, 0.09), (0.91, 0.038), (0.93, 0.00), (0.88, 0.09),(0.92,0.09)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_pattern_0.05.png')
# #
# # # # wanet 0.003
# # # TPR_FPR = [(0.95, 0.04), (0.00, 0.05), (0., 0.02), (0.006, 0.50), 'NA', (1., 0.49), (0.73, 0.80), (0., 0.45)]
# # # ACC_ASR = [(0.90, 0.10), (0.85, 0.12), (0.85, 0.14), (0.68, 0.37), (0.69, 0.17), (0.87, 0.004), (0.90, 0.03), (0.10, 1.00)]
# # # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_wanet_0.003.png')
# #
# # # # wanet 0.05
# # # TPR_FPR = [(0.96, 0.04), (0.00, 0.05), (0., 0.02), (0.83, 0.48), 'NA', (0.97, 0.47), (0.13, 0.11), (0.99, 0.39)]
# # # ACC_ASR = [(0.90, 0.10), (0.90, 0.99), (0.23, 0.83), (0.68, 0.13), (0.54, 0.52), (0.88, 0.042), (0.63, 0.19), (0.14, 0.89)]
# # # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_wanet_0.05.png')
# #
# # # adaptivecifar10 0.003
# # TPR_FPR = [(0.86, 0.02), (0.00, 0.05), (0., 0.00), (0.33, 0.50), 'NA', (1.00, 0.50), (0.21, 0.19), (0.64, 0.69), (0.99,0.98)]
# # ACC_ASR = [(0.91, 0.06), (0.91, 0.82), (0.93, 0.87), (0.86, 0.77), (0.74, 0.65), (0.90, 0.01), (0.93, 0.58), (0.68, 0.64), (0.84,0.87)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_adaptivecifar10_0.003.png')
# #
# # # adaptivecifar10 0.05
# # method_name = ['OUR', 'ABL', 'PIPD', 'ASD', 'CBD', 'DBD', 'EBD', 'FPP', 'NONE']
# # TPR_FPR = [(0.96, 0.03), (0.00, 0.05), (0., 0.00), (0.39, 0.51), 'NA', (0.97,0.47), (0.29, 0.27), (0.91, 0.62), (0.78,0.69)]
# # ACC_ASR = [(0.93, 0.09), (0.85, 0.93), (0.93, 0.84), (0.93,0.72), (0.73, 0.82), (0.90, 0.08), (0.92, 0.69), (0.78, 0.95), (0.89,0.80)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_adaptivecifar10_0.05.png')
# #
# # # ultrasonic 0.003
# # TPR_FPR = [(1.00, 0.082), (0.00, 0.05), (0., 0.00), (0.48, 0.50), 'NA', (1.00, 0.49), (0.00, 0.00), (0.34, 0.15), (0.77,0.59)]
# # ACC_ASR = [(0.93, 0.02), (0.97, 0.99), (0.97, 1.00), (0.97, 0.95), (0.94, 0.91), (0.89, 0.10), (0.96, 1.00), (0.86, 0.10), (0.95,0.91)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_ultrasonic_0.003.png')
# #
# # # ultrasonic 0.05
# # TPR_FPR = [(0.99, 0.06), (0.06, 0.047), (0., 0.00), (0.01, 0.53), 'NA', (0.95, 0.47), (0.00, 0.00), (0.67, 0.17), (0.38, 0.37)]
# # ACC_ASR = [(0.93, 0.12), (0.97, 1.00), (0.96, 0.99), (0.96,1.00), (0.93, 1.00), (0.86, 0.11), (0.95, 1.00), (0.84, 0.09), (0.96,1.00)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_ultrasonic_0.05.png')
# #
# # # # freq 0.003
# # # TPR_FPR = [(0.033, 0.022), (0.0, 0.05), (0., 0.005), (0.41,0.50), 'NA', (0.92, 0.50), (0.91, 0.80), (0.73, 0.67)]
# # # ACC_ASR = [(0.92, 0.09), (0.92,0.12), (0.92,0.14), (0.90,0.15), (0.76, 0.10), (0.89, 0.62), (0.90, 0.35), (0.79, 0.10)]
# # # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_freq_0.003.png')
# # # freq 0.01
# # # TPR_FPR = [(), (0.0, 0.05), (0., 0.00), (0.61,0.49), 'NA', (0.07, 0.50), (0.84, 0.80), ()]
# # # ACC_ASR = [(), (0.90,0.55), (0.92, 0.37), (0.91, 0.52), (0.78, 0.11), (0.89, 0.87), (0.90, 0.53), ()]
# # # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_freq_0.01.png')
# # #
# # # # freq 0.05
# # # TPR_FPR = [(0.93, 0.03), (0.0, 0.05), (0., 0.005), (0.91,0.95), 'NA', (0.30, 0.51), (0.30, 0.18), (1.00, 0.56)]
# # # ACC_ASR = [(0.92, 0.06), (0.92, 0.78), (0.91, 0.71), (0.91,0.95), (0.75, 0.42), (0.90, 0.95), (0.92, 0.91), (0.84, 0.10)]
# # # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_freq_0.05.png')
# #
# # # freq_meg_500 0.003
# # method_name = ['OUR', 'ABL', 'PIPD', 'ASD', 'CBD', 'DBD', 'EBD', 'FPP', 'NONE']
# # TPR_FPR = [(0.93, 0.03), (0.006, 0.05), (0., 0.019), (1.00,0.49), 'NA', (0.64, 0.49), (0.91, 0.80), (0.73, 0.67), (1.00,0.99)]
# # ACC_ASR = [(0.92, 0.07), (0.87, 0.82), (0.87, 0.91), (0.92,0.75), (0.87, 0.96), (0.91, 0.89), (0.92, 0.75), (0.79, 0.10), (0.82,0.98)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_freq_meg_500_0.003.png')
# #
# # # freq_meg_500 0.05
# # method_name = ['OUR', 'ABL', 'PIPD', 'ASD', 'CBD', 'DBD', 'EBD', 'FPP', 'NONE']
# # TPR_FPR = [(0.93, 0.03), (0.93, 0.003), (0.91, 0.00), (0.91,0.49), 'NA', (0.30, 0.51), (0.30, 0.18), (1.00, 0.56), (0.86, 0.38)]
# # ACC_ASR = [(0.92, 0.06), (0.88, 0.04), (0.87, 0.08), (0.88,0.00), (0.86, 0.11), (0.90, 0.95), (0.92, 0.95), (0.84, 0.10), (0.90, 0.99)]
# # plot_scatter(method_name, TPR_FPR, ACC_ASR, save_name='results_freq_meg_500_0.05.png')