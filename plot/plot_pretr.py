from collections import OrderedDict
from datetime import datetime
import argparse
import json
from plot_tb import draw_plots, check_mean_best_perf
import wandb
import git
import os
import sys

to_bool = lambda s: (s.lower() == "true")
parser = argparse.ArgumentParser()
parser.add_argument("config_file")
parser.add_argument("--debug", default="False", type=to_bool)
args = parser.parse_args()


# Plotting parameters
output_folder_prefix = "img/"
output_folder_suffix = ""

input_json_path = args.config_file

with open(input_json_path, 'r') as in_file:
    input_json = json.load(in_file)

api = wandb.Api(timeout=60)
run_prefix = ""

def save_config(dir_path):
    os.makedirs(dir_path, exist_ok=True)

    # Save metadata
    ctn = dict()
    ctn["commit"] = git.Repo(search_parent_directories=True).head.object.hexsha
    ctn["argv"] = sys.argv
    ctn["input_json_path"] = input_json_path
    log_ctn = json.dumps(ctn, indent=2)
    log_path = os.path.join(dir_path, "meta.json")
    with open(log_path, "w") as out_file:
        out_file.write(log_ctn)

    # Save the entire input config
    log_ctn = json.dumps(input_json, indent=2)
    log_path = os.path.join(dir_path, "config.json")
    with open(log_path, "w") as out_file:
        out_file.write(log_ctn)

    # Convert each run id to run name
    log_ctn = ""
    for task_label, task in input_json["tasks"].items():
        log_ctn += "----------\n"
        log_ctn += f"Task: {task_label}\n"
        for run_label, run_id_list in task["run_ids"].items():
            log_ctn += f"  Label: {run_label}\n"
            for run_id in run_id_list:
                run_name = api.run(run_prefix + run_id).name
                log_ctn += f"    {run_id} -> {run_name}\n"
    log_path = os.path.join(dir_path, "run_names.log")
    with open(log_path, "w") as out_file:
        out_file.write(log_ctn)

def provide_run_ids(task):
    run_ids = OrderedDict(input_json["tasks"][task]["run_ids"])
    model_order = input_json["tasks"][task]["order"]
    for model_name in model_order:
        run_ids.move_to_end(model_name)
    return run_ids

def parse_func(run_id, y_metric):
    global api, run_prefix
    history = api.run(run_prefix + run_id).history(pandas=False)
    x_list, y_list = [], []
    for s in history:
        if y_metric in s.keys() and s[y_metric] is not None:
            x_list.append(s["_step"])
            y_list.append(s[y_metric])
    return x_list, y_list


if __name__ == "__main__":
    output_folder = output_folder_prefix
    output_folder += datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder += output_folder_suffix
    if args.debug is True:
        output_folder += "_dbg"

    import time; start_time = time.time()
    save_config(output_folder)
    print(f"Save config time: {time.time() - start_time}")

    exp_note = input_json["exp_note"]

    for task, config in input_json["tasks"].items():
        print("task: {}".format(task))
        result_png_path = "./{}/{}/result.png".format(output_folder, task)
        run_ids = provide_run_ids(task)

        _parse_func = lambda rid: parse_func(rid, config["y_metric"])
        draw_plots(run_ids, result_png_path, config, _parse_func)

        print("Plot saved to {}".format(result_png_path))

        log_dir = f"./{output_folder}/{task}"
        check_mean_best_perf(run_ids, log_dir, config, _parse_func)

    # # Below code is used for measuring sample efficiency, but
    # # I haven't maintained this code, so it might be outdated.
    # task = 'R1'
    # print('=====\nsample efficiency check, task {}'.format(task))
    # log_path = './{}/sample_eff.log'.format(output_folder)
    # with open(log_path, 'w') as _:
    #     pass
    # file_names = provide_file_names(task)
    # xlim = input_json['tasks'][task]['xaxis_bound']
    # check_sample_eff(file_names, 1, min_updates, max_updates, xlim, 'SAC', log_path)

    # # Below was used for main performance measure.
    # # Measures/prints the training step with the best performance, as well as
    # # the value of best performance.
    # # Comment out below if you only want the learning curves.
    # print('=====\nbest performance check')
    # log_path = './{}/best_ur5.log'.format(output_folder)
    # with open(log_path, 'w') as _:
    #     pass
    # for task in input_json['tasks'].keys():
    #     msg = 'task: {}'.format(task)
    #     with open(log_path, 'a') as out_file:
    #         out_file.write(msg + '\n')
    #     print(msg)
    #     file_names = provide_file_names(task)
    #     xmax = input_json['tasks'][task]['xaxis_bound'][1]
    #     min_updates = input_json['min_updates']
    #     check_best_perf(file_names, 1, min_updates, xmax, log_path)

