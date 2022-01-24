import matplotlib as mpl
import matplotlib.pyplot as plt
import brewer2mpl
import numpy as np
import pandas as pd

bmap = brewer2mpl.get_map('Set2', 'Qualitative', 8)
colors = bmap.mpl_colors
# mpl.rcParams['axes.prop_cycle'] = colors
# plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 13


def plt_multi_bars(tick_step=1, group_gap=0.2, bar_gap = 0):
    # filename = 'results/ablation_study_aug_loss_amazon'
    filename = 'results/ablation_study_aug_loss_yelp'
    f1 = pd.read_table(filename + '.txt', sep=' ')
    x_labels = f1['METHODS'].tolist()
    datas = [ f1['AP'].tolist(), f1['RECALL'].tolist(),  f1['AUC'].tolist(), f1['ACC'].tolist()]
    legend_labels = ['AP', 'Recall', 'AUC', 'Acc.']
    datas = np.array(datas)
    fig = plt.figure()
    x = np.arange(len(x_labels)) * tick_step
    group_num = datas.shape[1]
    group_width = tick_step - group_gap
    bar_span = group_width/group_num
    bar_width = bar_span - bar_gap
    for index, y in enumerate(datas):
        plt.bar(x + index*bar_span, y, bar_width, color=colors[index])
        for k, v in enumerate(y):
            plt.text((x + index*bar_span)[k] - 0.05 , v-6.0, str(v), color='black', rotation='vertical') #  v-6.0 for amazon, v-8.0 for yelp
    plt.ylabel('Percent')
    plt.ylim(65,103) #65,103 for amazon, 40,95 for yelp
    ticks = x + (group_width - bar_gap) /2
    plt.xticks(ticks, x_labels)
    plt.legend(legend_labels, loc='upper center', ncol=len(legend_labels))
    plt.savefig(filename + '.pdf')
    plt.show()

# plt_multi_bars()

def plt_multi_lines():
    # filename = 'results/ablation_study_encoder_amazon'
    filename = 'results/ablation_study_encoder_yelp'
    f1 = pd.read_table(filename + '.txt', sep='\t')

    x_labels = f1['METHODS'].tolist()
    datas = [ f1['AP'].tolist(), f1['RECALL'].tolist(),  f1['AUC'].tolist(), f1['ACC'].tolist()]
    legend_labels = ['AP', 'Recall', 'AUC', 'Acc.']

    datas = np.array(datas)
    marks = ['-*', '-^', '-o', '-+']

    x = np.arange(len(x_labels))
    for index, y in enumerate(datas):
        plt.plot(x, y, marks[index], color=colors[index], linewidth=2)

    plt.ylabel('Percent')
    min_v_1 = datas.min(0).min(0) - 10
    plt.ylim(min_v_1, 90) #100 for amazon, 90 for yelp
    plt.xticks(x, x_labels, rotation=20)
    plt.legend(legend_labels)
    plt.savefig(filename + '.png', format='png', dpi=300)
    plt.show()

# plt_multi_lines()


def plt_hyper_param():
    fig, ax1 = plt.subplots()

    f1 = 'results/hyper_param_hidden_amazon.txt'
    f2 = 'results/hyper_param_hidden_yelp.txt'
    data1 = pd.read_table(f1, sep='\t')
    data2 = pd.read_table(f2, sep='\t')
    xlabels = ['AP', 'Recall', 'AUC', 'Acc.']
    box_1 = [data1['AP'].tolist(), data1['Recall'].tolist(), data1['AUC'].tolist(), data1['ACC'].tolist()]
    box_2 = [data2['AP'].tolist(), data2['Recall'].tolist(), data2['AUC'].tolist(), data2['ACC'].tolist()]

    b1 = ax1.boxplot(box_1, labels=xlabels, showmeans=True, patch_artist=True,
                boxprops={'facecolor': colors[2]})
    ax1.set_ylim(75, 100)
    ax1.set_ylabel('Amazon (%)')
    ax1.yaxis.label.set_color(colors[2])
    ax1.tick_params(axis='y', colors=colors[2])
    ax2 = ax1.twinx()
    b2 = ax2.boxplot(box_2, labels=xlabels, showmeans=True, patch_artist=True,
                boxprops={'facecolor': colors[3]})
    ax2.set_ylim(50,100)
    ax2.set_ylabel('YelpChi (%)')
    ax2.yaxis.label.set_color(colors[3])
    ax2.tick_params(axis='y', colors=colors[3])
    plt.savefig('results/hyper_param_hidden' + '.pdf')
    plt.show()

# plt_hyper_param()

def plt_hyper_param_hidden():
    fig = plt.figure()

    f1 = 'results/hyper_param_hidden_amazon.txt'
    f2 = 'results/hyper_param_hidden_yelp.txt'
    data1 = pd.read_table(f1, sep='\t')
    data2 = pd.read_table(f2, sep='\t')
    x_labels = data1['hidden'].tolist()
    legend_labels = ['AP', 'Recall', 'AUC']
    lines_1 = [data1['AP'].tolist(), data1['Recall'].tolist(), data1['AUC'].tolist()]
    lines_2 = [data2['AP'].tolist(), data2['Recall'].tolist(), data2['AUC'].tolist()]
    marks = ['-*', '-^', '-o']

    x = np.arange(len(x_labels))
    for index, y in enumerate(lines_2):
        plt.plot(x, y, marks[index], color=colors[index], linewidth=2)


    plt.ylim(50, 90)  # yelp 50, 90, amazon 75, 100
    plt.ylabel('Percent')
    plt.xlabel('Hidden dimension')
    plt.xticks(x, x_labels)
    # b2 = ax2.boxplot(box_2, labels=xlabels, showmeans=True, patch_artist=True,
    #             boxprops={'facecolor': colors[3]})
    # ax2.set_ylim(50,100)
    # ax2.set_ylabel('YelpChi (%)')
    # ax2.yaxis.label.set_color(colors[3])
    # ax2.tick_params(axis='y', colors=colors[3])
    plt.legend(legend_labels)
    plt.savefig('results/hyper_param_hidden_yelp' + '.pdf')
    plt.show()

plt_hyper_param_hidden()