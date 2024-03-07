
from conf import conf  # 全局conf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
from collections import Counter
import time
import os

def Draw_Statistic(SCENARIO):
    '''
    画统计图
    '''
    mpl.rcParams["font.sans-serif"] = ["SimHei"] # 解决中文乱码
    mpl.rcParams["axes.unicode_minus"] = False

    # 读取数据
    data_path = os.path.join(conf.SUMMARY_PATH, 'SCENARIO_' + str(SCENARIO))
    StartTime = np.load(data_path + '_StartTime.npy', allow_pickle=True).item()
    EndTime = np.load(data_path + '_EndTime.npy', allow_pickle=True).item()

    UseTime = dict()
    Rate = dict()
    cellTexts = []

    for id in range(0, len(conf.STEPS)-1):
        if (id in StartTime.keys()) and (id in EndTime.keys()):
            # 处理用时
            time_cost = EndTime[id] - StartTime[id]
            UseTime[id] = time_cost

            # 处理评分
            if time_cost >= 2 * float(conf.EXPECTED_TIME[id]):
                Rate[id] = '差'
            elif (time_cost >= float(conf.EXPECTED_TIME[id])) and (time_cost < 2 * float(conf.EXPECTED_TIME[id])):
                Rate[id] = '良'
            else:
                Rate[id] = '优'

            cellTexts.append([
                # time.strftime("%Y%m%d %H:%M:%S", time.localtime(StartTime[id])),
                # time.strftime("%Y%m%d %H:%M:%S", time.localtime(EndTime[id])),
                time.strftime("%H:%M:%S", time.localtime(StartTime[id])),
                time.strftime("%H:%M:%S", time.localtime(EndTime[id])),
                "%.2f s" % (time_cost),
                Rate[id]
            ])
        else:
            # 处理用时
            cellTexts.append(['00:00:00', '00:00:00', '00.00 s', '差'])

            # 处理评分
            UseTime[id] = -1
            Rate[id] = '差'

    # 可视化饼图
    fig1 = plt.figure(1, dpi=100,
                     constrained_layout=True,
                     )
    Rate_list = list(Rate.values())  # 把评分转为list
    most_rate = Counter(Rate_list).most_common(1)
    rate_count = dict(Counter(Rate_list))  # 得到评分的数量统计，{id: count}
    kinds = list(rate_count.keys())
    soldNums = [count/len(Rate_list) for rate, count in rate_count.items()]
    plt.pie(soldNums,
            labels=kinds,
            autopct="%3.1f%%",
            startangle=60)
    plt.title("评分统计图")
    plt.legend()
    plt.savefig(os.path.join(conf.SUMMARY_PATH, 'SCENARIO_' + str(SCENARIO) +'_pie.png'), format='png', bbox_inches='tight', dpi=150, transparent=True,pad_inches=0)
    plt.savefig(os.path.join(conf.SUMMARY_PATH, 'SCENARIO_' + str(SCENARIO) + '_pie.png'), format='png', bbox_inches='tight',
                dpi=150, transparent=True, pad_inches=0)
    plt.close(fig1)  # 一定要关闭，不然会重复绘图

    # 可视化表格
    fig2 = plt.figure(2, dpi=100,
                     constrained_layout=True,
                     )
    colLabels = ['开始时刻', '结束时刻', '总用时', '评分']
    rowLabels = conf.STEPS[:-1]
    plt.table(cellText=cellTexts,  # 简单理解为表示表格里的数据
              colWidths=[0.3,0.3,0.2,0.1],  # 每个小格子的宽度 * 个数，要对应相应个数
              colLabels=colLabels,  # 每列的名称
              rowLabels=rowLabels,  # 每行的名称（从列名称的下一行开始）
              rowLoc="center",  # 行名称的对齐方式
              loc="center",  # 表格所在位置
              cellLoc='center'
              )
    plt.axis('off')
    plt.savefig(os.path.join(conf.SUMMARY_PATH, 'SCENARIO_' + str(SCENARIO) + '_table.png'), format='png', bbox_inches='tight', dpi=150, transparent=True)
    plt.close(fig2)  # 一定要关闭，不然会重复绘图

    # 可视化每个动作的用时,柱形图 x:动作名称 y: 用时
    fig3 = plt.figure(3, dpi=100,
                     constrained_layout=True,
                     )
    actions = [id_name.split()[0] for id_name in rowLabels]
    use_time = np.array([float(line[2].split()[0]) if line[2]!="00:00:00" else 0 for line in cellTexts])
    expected_time = np.array([float(t) for t in conf.EXPECTED_TIME])
    rate = use_time / expected_time
    # 柱状图
    plt.bar(x=actions, height=use_time, label='实际用时', color='Coral', alpha=0.8)
    plt.bar(x=actions, height=expected_time, label='建议用时', color='LemonChiffon', alpha=0.8)
    # 在左侧显示图例
    plt.legend(loc="upper left")
    # 设置标题
    plt.title("用时统计")
    # 为两条坐标轴设置名称
    plt.xlabel("动作序号")
    plt.ylabel("检测用时")
    plt.savefig(os.path.join(conf.SUMMARY_PATH, 'SCENARIO_' + str(SCENARIO) + '_line.png'), format='png', bbox_inches='tight',
                dpi=150, transparent=True)
    plt.close(fig3)  # 一定要关闭，不然会重复绘图

    # plt.show()


    return cellTexts,rowLabels,most_rate

# 表格：https://blog.csdn.net/weixin_43842649/article/details/118456837
# 柱形图+折线图：https://blog.csdn.net/candice5566/article/details/121444185
# 子图布局：https://mp.weixin.qq.com/s/SeqyfzNBO_mcGJbPLHbjlQ
