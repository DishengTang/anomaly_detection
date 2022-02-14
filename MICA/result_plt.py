# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import brewer2mpl
import numpy as np
import pandas as pd
import time
import seaborn as sns

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

# plt_hyper_param_hidden()
def plot_similarity(net, dataset, scale='area'):
    start_time = time.time()
    metric_name = 'Cosine similarity'
    print('Loading data...')
    Distance = pd.read_pickle('./{}_{}_cosine_similarity.pkl'.format(net, dataset))
    print('Plotting...')
    fig = plt.figure()          # scale = {'area', 'count', 'width'}
    ax = sns.violinplot(x='type', scale=scale, y=metric_name, data=Distance) # , color=sns.color_palette("Set2")[0]
    ax.set(xlabel=None)
    # ax.set(ylabel=None)
    # plt.yticks(fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.ylim(-10, 60)
    # plt.gca().set_title('Euclidean distance', rotation=0)
    # plt.legend()
    plt.tight_layout()
    plt.savefig('{}_{}_similarity_distri.pdf'.format(net, dataset), transparent=True)
    # plt.show()
    print("--- %s minutes for plotting" % ((time.time() - start_time)/60))

def plot_gate(net, dataset):
    start_time = time.time()
    print('Loading data...')
    gates = pd.read_pickle('./{}_{}_gates.pkl'.format(net, dataset))
    print('Plotting...')
    c_f = pd.DataFrame(np.vstack(((gates['context']-gates['feature']).mean(1), [r'$mean(\alpha^{c}-\alpha^{f})$']*gates['context'].shape[0])).T, columns=['data', 'type'])
    c_t = pd.DataFrame(np.vstack(((gates['context']-gates['topology']).mean(1), [r'$mean(\alpha^{c}-\alpha^{t})$']*gates['context'].shape[0])).T, columns=['data', 'type'])
    f_t = pd.DataFrame(np.vstack(((gates['feature']-gates['topology']).mean(1), [r'$mean(\alpha^{f}-\alpha^{t})$']*gates['context'].shape[0])).T, columns=['data', 'type'])
    gate_diff = pd.concat([c_f, c_t, f_t], ignore_index=True)
    gate_diff['data'] = pd.to_numeric(gate_diff['data'])
    fig = plt.figure()
    ax = sns.histplot(data=gate_diff, x='data', hue='type', element='step')
    ax.set(xlabel=None)
    ax.get_legend().set_title(None)
    # ax.set(ylabel=None)
    # plt.yticks(fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.ylim(-10, 60)
    # plt.gca().set_title('Euclidean distance', rotation=0)
    # plt.legend()
    plt.tight_layout()
    plt.savefig('{}_{}_gates.pdf'.format(net, dataset), transparent=True)
    # plt.show()
    print("--- %s minutes for plotting" % ((time.time() - start_time)/60))
# %%
plot_similarity('SupCL', 'Amazon', 'width')
# %%
plot_similarity('SupCL', 'YelpChi', 'area')
# %%
plot_similarity('ICA', 'Amazon', 'width')
# %%
plot_similarity('ICA', 'YelpChi', 'width')
# %%
plot_gate('SupCL', 'Amazon')
# %%
plot_gate('SupCL', 'YelpChi')
# %%
