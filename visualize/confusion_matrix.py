import itertools
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, name='conf'):
    """
,  This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }

    # 设置输出的图片大小
    figsize = 8, 6
    figure, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.ylabel('True label', font)
    plt.xlabel('Predicted label', font)
    plt.savefig(name + '.png', dpi=300)
    plt.show()


# cnf_matrix_m2r = np.array([[ 15, 1, 0, 0, 0, 0, 1, 1, 4, 4],
#                            [0, 33, 1, 0, 10, 1, 4, 2, 7, 27],
#                            [0, 9, 52, 0, 21, 18, 3, 14, 9, 20],
#                            [1, 6, 20, 3, 9, 10, 48, 17, 10, 25],
#                            [1, 50, 2, 0, 595, 22, 52, 36, 12, 31],
#                            [0, 3, 0, 0, 6, 18, 7, 6, 0, 1],
#                            [0, 0, 1, 0, 0, 3, 47, 6, 1, 3],
#                            [0, 0, 0, 0, 1, 0, 0, 24, 0, 0],
#                            [1, 13, 0, 0, 32, 2, 2, 11, 61, 12],
#                            [0, 25, 5, 2, 13, 8, 0, 18, 7, 223]])
#
# cnf_matrix_s2r = np.array([[15, 0, 0, 0, 0, 0, 6, 0, 1, 4],
#                             [0, 0, 0, 0,  26, 5, 5, 0, 5,  44],
#                             [4, 1,  28, 4,  23,  29, 8, 3, 2,  44],
#                             [2, 0, 3, 0, 9,  27,  64, 0, 0,  44],
#                             [6, 2, 0, 0, 625,  20,  54, 3, 1,  90],
#                             [0, 0, 0, 1,  15,  10, 2, 1, 0,  12],
#                             [0, 0, 0, 0, 5, 3,  34, 1, 0,  18],
#                             [0, 0, 0, 0,  12, 3, 0, 9, 0, 1],
#                             [3, 0, 0, 0,  47, 9, 3, 1,  54,  17],
#                             [2, 1, 1, 0,  24,  33, 2, 3, 2, 233]])
#
# cnf_matrix_unsu_m2r = np.array([[16, 0, 0, 0, 0, 1, 3, 1, 1, 4],
#                              [1, 27, 4, 0, 3, 2, 5, 24, 4, 15],
#                              [1, 5, 48, 0, 3, 20, 13, 37, 6, 13],
#                              [0, 0, 18, 0, 2, 11, 51, 48, 0, 19],
#                              [3, 37, 16, 0, 399, 123, 96, 100, 4, 23],
#                              [0, 0, 4, 0, 0, 15, 4, 17, 0, 1],
#                              [0, 0, 16, 0, 0, 2, 33, 8, 0, 2],
#                              [0, 0, 0, 0, 0, 0, 1, 24, 0, 0],
#                              [3, 10, 6, 1, 29, 4, 4, 32, 41, 4],
#                              [5, 16, 18, 2, 1, 14, 22, 64, 16, 143]])
#
# cnf_matrix_unsu_s2r = np.array([[14, 0, 0, 0, 3, 2, 3, 0, 2, 2],
#                              [3, 1, 0, 0, 26, 9, 4, 0, 11, 31],
#                              [5, 1, 7, 7, 18, 57, 14, 9, 10, 18],
#                              [1, 0, 0, 1, 16, 52, 48, 1, 2, 28],
#                              [39, 1, 0, 4, 508, 95, 83, 9, 18, 44],
#                              [0, 0, 0, 2, 1, 25, 3, 3, 0, 7],
#                              [2, 0, 1, 1, 0, 16, 30, 3, 2, 6],
#                              [3, 0, 0, 0, 2, 14, 1, 3, 0, 2],
#                              [11, 1, 0, 0, 58, 9, 3, 4, 33, 15],
#                              [12, 1, 0, 0, 22, 119, 2, 5, 20, 120]])


# attack_types = ['Bathtub', 'Bed', 'Bookshelf', 'Cabinet', 'Chair', 'Lamp', 'Monitor', 'Plant', 'Sofa', 'Table',]
#
# plot_confusion_matrix(cnf_matrix_m2r, classes=attack_types, normalize=True, title='Normalized confusion matrix', name='confusion_matrix_m2r')
# plot_confusion_matrix(cnf_matrix_s2r, classes=attack_types, normalize=True, title='Normalized confusion matrix', name='confusion_matrix_s2r')
# plot_confusion_matrix(cnf_matrix_unsu_m2r, classes=attack_types, normalize=True, title='Normalized confusion matrix', name='confusion_matrix_unsu_m2r')
# plot_confusion_matrix(cnf_matrix_unsu_s2r, classes=attack_types, normalize=True, title='Normalized confusion matrix', name='confusion_matrix_unsu_s2r')

cnf_matrix_s2r_9 = np.array([
    [5, 0, 1, 7, 1, 1, 0, 0, 2],
    [0, 1, 0, 2, 0, 1, 0, 3, 15],
    [1, 0, 22, 1, 29, 0, 1, 0, 21],
    [0, 0, 0, 78, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 40, 0, 0, 0, 0],
    [0, 0, 1, 2, 3, 2, 0, 1, 12],
    [0, 1, 12, 7, 6, 1, 14, 0, 8],
    [0, 0, 0, 5, 0, 0, 0, 33, 4],
    [0, 0, 1, 0, 2, 0, 0, 0, 51]])

attack_types = ['Bag', 'Bed', 'Cabinet', 'Chair', 'Display', 'Pillow', 'Shelf', 'Sofa', 'Table']

plot_confusion_matrix(cnf_matrix_s2r_9, classes=attack_types, normalize=True, title='Normalized confusion matrix',
                      name='confusion_matrix_s2r_9')

cnf_matrix_s9_sup = np.array([
    [0, 0, 0, 8, 2, 0, 0, 1, 6],
    [0, 0, 0, 1, 0, 0, 0, 4, 17],
    [0, 0, 5, 4, 29, 0, 0, 1, 36],
    [0, 0, 0, 78, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 38, 0, 0, 0, 1],
    [0, 0, 1, 2, 2, 0, 0, 0, 16],
    [0, 0, 4, 6, 5, 0, 18, 0, 16],
    [0, 0, 0, 4, 0, 0, 0, 29, 9],
    [0, 0, 0, 0, 1, 0, 0, 0, 53]])
plot_confusion_matrix(cnf_matrix_s9_sup, classes=attack_types, normalize=True, title='Normalized confusion matrix',
                      name='cnf_matrix_s9_sup')

cnf_matrix_m2r_10 = np.array([
    [15, 1, 0, 0, 1, 0, 2, 0, 4, 3],
    [0, 31, 0, 0, 3, 2, 5, 4, 8, 32],
    [0, 6, 68, 0, 6, 18, 4, 13, 8, 23],
    [1, 2, 21, 4, 11, 4, 56, 16, 4, 30],
    [0, 50, 9, 0, 545, 33, 71, 31, 24, 38],
    [0, 0, 1, 2, 0, 23, 2, 11, 0, 2],
    [0, 0, 1, 0, 0, 4, 47, 4, 0, 5],
    [0, 0, 0, 0, 1, 2, 0, 21, 1, 0],
    [0, 6, 0, 0, 16, 7, 3, 16, 75, 11],
    [0, 37, 8, 2, 3, 18, 1, 20, 6, 206]])

attack_types = ['Bathtub', 'Bed', 'Bookshelf', 'Cabinet', 'Chair', 'Lamp', 'Monitor', 'Plant', 'Sofa', 'Table']

plot_confusion_matrix(cnf_matrix_m2r_10, classes=attack_types, normalize=True, title='Normalized confusion matrix',
                      name='confusion_matrix_m2r_10')

cnf_matrix_m10_sup = np.array([
    [16, 0, 0, 0, 0, 1, 3, 1, 1, 4],
    [1, 27, 4, 0, 3, 2, 5, 24, 4, 15],
    [1, 5, 48, 0, 3, 20, 13, 37, 6, 13],
    [0, 0, 18, 0, 2, 11, 51, 48, 0, 19],
    [3, 37, 16, 0, 399, 123, 96, 100, 4, 23],
    [0, 0, 4, 0, 0, 15, 4, 17, 0, 1],
    [0, 0, 16, 0, 0, 2, 33, 8, 0, 2],
    [0, 0, 0, 0, 0, 0, 1, 24, 0, 0],
    [3, 10, 6, 1, 29, 4, 4, 32, 41, 4],
    [5, 16, 18, 2, 1, 14, 22, 64, 16, 143]])
plot_confusion_matrix(cnf_matrix_m10_sup, classes=attack_types, normalize=True, title='Normalized confusion matrix',
                      name='cnf_matrix_m10_sup')

cnf_matrix_s2r_10 = np.array([
    [20, 0, 0, 0, 0, 1, 2, 0, 2, 1],
    [1, 0, 0, 0, 9, 6, 5, 0, 18, 46],
    [2, 0, 22, 16, 22, 39, 9, 2, 11, 23],
    [0, 1, 0, 16, 6, 16, 56, 3, 13, 38],
    [17, 3, 1, 9, 552, 81, 53, 1, 38, 46],
    [0, 0, 0, 0, 1, 34, 2, 2, 0, 2],
    [0, 0, 0, 3, 0, 10, 42, 0, 1, 5],
    [0, 0, 0, 0, 1, 11, 0, 13, 0, 0],
    [1, 0, 0, 0, 21, 15, 3, 0, 79, 15],
    [14, 0, 2, 1, 9, 47, 2, 3, 16, 207]])

plot_confusion_matrix(cnf_matrix_s2r_10, classes=attack_types, normalize=True, title='Normalized confusion matrix',
                      name='confusion_matrix_s2r_10')

cnf_matrix_s10_sup = np.array([
    [18, 0, 0, 0, 0, 2, 2, 0, 1, 3],
    [4, 1, 0, 0, 12, 11, 5, 0, 10, 42],
    [7, 0, 29, 7, 14, 53, 10, 1, 6, 19],
    [1, 0, 3, 8, 3, 36, 60, 1, 3, 34],
    [43, 1, 2, 2, 473, 117, 116, 4, 15, 28],
    [0, 0, 0, 0, 0, 31, 2, 1, 0, 7],
    [0, 0, 2, 7, 0, 7, 38, 1, 0, 6],
    [0, 0, 0, 0, 2, 12, 0, 11, 0, 0],
    [10, 0, 0, 0, 26, 22, 2, 3, 61, 10],
    [117, 0, 0, 1, 2, 81, 4, 2, 4, 90]])

plot_confusion_matrix(cnf_matrix_s10_sup, classes=attack_types, normalize=True, title='Normalized confusion matrix',
                      name='cnf_matrix_s10_sup')

cnf_matrix_m2r_11 = np.array([
    [18, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
    [0, 13, 7, 6, 27, 11, 3, 1, 0, 7, 0],
    [1, 0, 77, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 5, 12, 1, 0, 2, 2, 0, 6, 0],
    [0, 0, 3, 0, 38, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0],
    [1, 2, 7, 1, 1, 2, 35, 0, 0, 0, 0],
    [3, 0, 4, 0, 1, 0, 0, 13, 0, 2, 1],
    [2, 0, 3, 1, 0, 0, 0, 3, 31, 1, 1],
    [1, 1, 3, 1, 0, 1, 0, 0, 0, 47, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 14]])

attack_types = ['Bed', 'Cabinet', 'Chair', 'Desk', 'Display', 'Door', 'Shelf', 'Sink', 'Sofa', 'Table', 'Toilet']
plot_confusion_matrix(cnf_matrix_m2r_11, classes=attack_types, normalize=True, title='Normalized confusion matrix',
                      name='confusion_matrix_m2r_11')

cnf_matrix_m11_sup = np.array([
    [10, 0, 0, 4, 0, 0, 0, 0, 1, 7, 0],
    [0, 12, 5, 8, 26, 15, 2, 2, 0, 5, 0],
    [0, 0, 77, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 9, 1, 0, 0, 3, 0, 10, 0],
    [0, 1, 0, 0, 38, 1, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0],
    [1, 2, 7, 2, 3, 2, 31, 0, 0, 1, 0],
    [1, 1, 9, 0, 1, 0, 0, 10, 0, 2, 0],
    [1, 0, 6, 8, 0, 0, 0, 3, 18, 4, 2],
    [0, 0, 3, 2, 1, 1, 0, 0, 0, 47, 0],
    [0, 0, 3, 0, 1, 0, 0, 1, 0, 2, 10]])

plot_confusion_matrix(cnf_matrix_m11_sup, classes=attack_types, normalize=True, title='Normalized confusion matrix',
                      name='cnf_matrix_m11_sup')
