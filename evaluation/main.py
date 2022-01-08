import pandas as pd

# input box shape
# box => (x1,y1, x2,y2)
def cal_iou(gt_box, pred_box):

    gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)

    # intersection
    x1 = max(gt_box[0], pred_box[0])
    y1 = max(gt_box[1], pred_box[1])
    x2 = min(gt_box[2], pred_box[2])
    y2 = min(gt_box[3], pred_box[3])
    # intersection width, height
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    intersection = w * h
    iou = intersection / (gt_area + pred_area - intersection)
    return iou


# input result shape
# result[i] => (index, confidence, iou)


example_result = [[1, 0.43, 0.7],
[2, 0.25, 0.7],
[3, 0.95, 0.7],
[4, 0.88, 0.7],
[5, 0.10, 0.3],
[6, 0.55, 0.3],
[7, 0.78, 0.7],
[8, 0.66, 0.7],
[9, 0.75, 0.7],
[10, 0.32, 0.3]]

import matplotlib.pyplot as plt
from sklearn.metrics import auc

def pr_curve(result, iou_thres):
    result = pd.DataFrame(result)
    result['tpfp']=[0 if t else 1 for t in list(result[2]<iou_thres)]  # 1: tp 0: fp
    result = result.sort_values(by=1, ascending=False, axis=0)
    result['cumul_tp'] = result['tpfp'].cumsum()
    result = result.reset_index(drop=True)
    result['cumul_fp'] = result.index-result['cumul_tp']+1
    result['precision']= result['cumul_tp']/(result.index+1)
    result['recall']= result['cumul_tp']/15         ## all ground truth = 15
    result['adj_precision']=0
    result['area']=0


    max=0
    for i in range(len(result)):
        a = result['precision'][len(result)-i-1]
        if a>max:
            max=a
        result.loc[len(result)-i-1,'adj_precision']=max

    plt.plot(result['recall'], result['precision'])
    plt.plot(result['recall'], result['adj_precision'])
    plt.show()

    # PR curve AUC score
    print(auc(result['recall'],result['adj_precision']))

    return result


print(pr_curve(example_result, 0.5))
