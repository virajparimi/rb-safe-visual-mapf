import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pud.algos.data_struct import arg_group_vals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_file',
        type=str,
        default="runs/results/job_local_cfg_blend_reset_0.5/2024-04-04-23-51-09/2024-04-04-23-51-09.pickle",
        help='')

    args = parser.parse_args()
    tbdata = None
    with open(args.pickle_file, 'rb') as f:
        tbdata = pickle.load(f)
    
    """this is regrouping the means, not individual eval data points"""

    #tbdata["scalars"]["Eval_D=02 C=0.00/c_pred_mean"].keys() # pickle path -> steps, value
    #k0 = list(tbdata["scalars"]["Eval_D=02 C=0.00/c_pred_mean"].keys())[0]
    #tbdata["scalars"]["Eval_D=02 C=0.00/c_pred_mean"][k0]["step"]
    #tbdata["scalars"]["Eval_D=02 C=0.00/c_pred_mean"][k0]["value"]

    """parse all data points of the same steps, and run argsort
    c_pred_mean
    c_true_mean
    """
    paired_keys = {}
    for k in tbdata["scalars"]:
        if k.startswith("Eval_"):
            if "c_pred_mean" in k:
                group_tag, val_tag = k.split("/")
                paired_keys[group_tag] = {}

    pred_steps = []
    pred_values = []
    true_steps = []
    true_values = []

    for group in paired_keys:
        # parse pred means
        field = group + "/" + "c_pred_mean"
        filename = list(tbdata["scalars"][field])[0]

        pred_steps.extend(tbdata["scalars"][field][filename]["step"])
        pred_values.extend(tbdata["scalars"][field][filename]["value"])

        field = group + "/" + "c_true_mean"
        true_steps.extend(tbdata["scalars"][field][filename]["step"])
        true_values.extend(tbdata["scalars"][field][filename]["value"])

    pred_steps = np.array(pred_steps)
    pred_values = np.array(pred_values)
    true_steps = np.array(true_steps)
    true_values = np.array(true_values)

    up_inds = pred_steps.argsort()

    sort_pred_steps = pred_steps[up_inds]
    assert np.all(np.diff(sort_pred_steps) >= 0)
    sort_pred_values = pred_values[up_inds]

    sort_true_steps = true_steps[up_inds]
    assert np.allclose(sort_pred_steps, sort_true_steps)
    sort_true_values = true_values[up_inds]

    cur_step = sort_true_steps[0]
    divs = [0,]
    for i in range(len(sort_true_steps)):
        if sort_pred_steps[i] > cur_step:
            divs.append(i)
            cur_step = sort_pred_steps[i]
    divs.append(len(sort_pred_steps))

    
    cost_groups = np.linspace(0, 1, 4)
    vis_data = {}
    for i in range(len(cost_groups)-1):
        vis_data[i] = {"pred": [], "true": [], "steps": [],}
    
    for i_d in range(len(divs)-1):
        start = divs[i_d]
        end = divs[i_d+1]
        step_i = sort_pred_steps[start:end]
        pred_val_i = sort_pred_values[start:end]
        true_val_i = sort_true_values[start:end]

        groups = arg_group_vals(vals=true_val_i, divs=cost_groups)
        for gi in groups:
            if len(groups[gi]["inds"]) > 0:
                vis_data[gi]["steps"].append(step_i[0])
                vis_data[gi]["true"].append(np.mean(groups[gi]["vals"]))
                tmp_inds = np.array(groups[gi]["inds"], dtype=int)
                vis_data[gi]["pred"].append(
                    np.mean(pred_val_i[tmp_inds])
                    )

    fig, axes = plt.subplots(3, 1, sharex=True)

    for ig in vis_data.keys():
        axes[ig].plot(
            vis_data[ig]["steps"], 
            vis_data[ig]["pred"], 
            marker='o', color='r', linestyle='-', linewidth=2, markersize=2, label='pred')
        axes[ig].plot(
            vis_data[ig]["steps"], 
            vis_data[ig]["true"], 
            marker='o', color='g', linestyle='-', linewidth=2, markersize=2, label='true')
        axes[ig].set_ylim([0, 1])
        axes[ig].set_title('compare')
        axes[ig].set_xlabel('steps')
        axes[ig].set_ylabel('group_{}'.format(ig))
        axes[ig].legend()
    fig.tight_layout()
    fig.savefig(fname="temp/group_training_stats.jpg", dpi=320)
    plt.close(fig)