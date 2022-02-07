import pickle
import numpy as np
from utils.helpers import *

history = pickle.load(open("saved_histories/18 no norm/" + "model_0", 'rb'))
print(history['val_sensitivity'][-10])
print(history['val_specificity'][-10])
print(history['val_accuracy'][-10])
print(history['val_mcc'][-10])
print(history['val_auc'])

plot_specificity(history, 1)

# # no_norm400 = [0.945, 0.869, 0.881, 0.68, 0.971]
# # norm400 = [0.945, 0.941, 0.942, 0.811, 0.974]
# # norm200 = [0.972, 0.859, 0.877, 0.683, 0.968]
# # norm300 = [0.972, 0.892, 0.905, 0.763, 0.973]
# # norm500 = [0.918, 0.875, 0.881, 0.669, 0.963]
# # norm600 = [0.959, 0.872, 0.886, 0.693, 0.968]
# #
# vote = [0.71, 0.6486, 0.6887, 0.347, 0.7356]
# ave = [0.855, 0.6486, 0.783, 0.514, 0.763]
# med = [0.811, 0.675, 0.764, 0.484, 0.762]
# #
# # first = [0.608, 0.989, 0.929, 0.712, 0.854]
# # best = [0.743,	0.982,	0.944,	0.78,	0.922]
# # up = [0.864,	0.992,	0.976,	0.909,	0.953]
# # down = [0.945,	0.423,	0.506,	0.28,	0.946]
# # smote = [0.837,	0.984,	0.961,	0.851,	0.951]
# #
# X = np.arange(5) * 2
# width = 0.25
# #
# #
# fig, ax = plt.subplots()
# # compare length
# # rects1 = ax.bar(X - 0.6875, no_norm400, color = 'blue', width = width, label='Length 400, no normalize')
# # rects2 = ax.bar(X - 0.4125, norm400, color = 'orange', width = width, label='Length 400, normalized')
# # rects3 = ax.bar(X - 0.1375, norm200, color = 'green', width = width, label='Length 200, normalized')
# # rects4 = ax.bar(X + 0.1375, norm300, color = 'red', width = width, label='Length 300, normalized')
# # rects5 = ax.bar(X + 0.4125, norm500, color = 'purple', width = width, label='Length 500, normalized')
# # rects6 = ax.bar(X + 0.6875, norm600, color = 'teal', width = width, label='Length 600, normalized')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_title('Compare difference method of length and normalization', fontsize=25)
# # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6])
# # ax.set_xticklabels(['a' ,'Sensitivity', 'Specificity', 'Accuracy', 'MCC', 'AUC'], fontsize=20)
# # ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1, ' ', ' ', ' '],fontsize=20)
# # ax.legend(fontsize=20)
#
#
# # # compare vote
# rects2 = ax.bar(X - 0.3, vote, color = 'orange', width = width, label='Paper')
# rects3 = ax.bar(X , ave, color = 'green', width = width, label='G-gap = 1')
# rects4 = ax.bar(X + 0.3, med, color = 'red', width = width, label='G-gap = 2')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_title('Compare difference method', fontsize=25)
# ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# ax.set_xticklabels(['a' ,'Sensitivity', 'Specificity', 'Accuracy', 'MCC', 'AUC'], fontsize=20)
# ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1],fontsize=20)
# ax.legend(fontsize=20)
#
#
# # # compare RF
# # rects1 = ax.bar(X - 0.55, first, color = 'blue', width = width, label='First feature')
# # rects2 = ax.bar(X - 0.275, best, color = 'orange', width = width, label='Best feature')
# # rects3 = ax.bar(X, up, color = 'green', width = width, label='Up resample')
# # rects5 = ax.bar(X + 0.275, smote, color = 'purple', width = width, label='SMOTE')
# # rects4 = ax.bar(X + 0.55, down, color = 'red', width = width, label='Down resample')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_title('Compare difference method of Ranfom Forest model', fontsize=25)
# # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6])
# # ax.set_xticklabels(['a' ,'Sensitivity', 'Specificity', 'Accuracy', 'MCC', 'AUC'], fontsize=20)
# # ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1, ' ', ' ', ' '],fontsize=20)
# # ax.legend(fontsize=20)
# #
# plt.show()

