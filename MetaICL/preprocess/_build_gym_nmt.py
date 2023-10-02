# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import hashlib
import os
import subprocess

import multiprocessing as mp
from multiprocessing import Process, Manager
from _all_tasks import ALL_TASKS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="../../data", type=str)
    # parser.add_argument("--task_dir", default="./", type=str)
    parser.add_argument("--n_proc", default=1, type=int)
    parser.add_argument('--debug', action='store_true',
                        help="Run 2 tasks per process to test the code")

    parser.add_argument('--inst', action='store_true',
                        help="Construct data from hg datasets.")
    parser.add_argument('--do_train', action='store_true',
                        help="Verify the datafiles with pre-computed MD5")
    parser.add_argument('--do_test', action='store_true',
                        help="Run 2 tasks per process to test the code")

    parser.add_argument('--train_k', type=int, default=16384, help="k for meta-training tasks")
    parser.add_argument('--test_k', type=int, default=16, help="k for target tasks")

    args = parser.parse_args()

    if args.do_train and args.do_test:
        raise NotImplementedError("You should specify one of `--do_train` and `--do_test`, not both")
    if not args.do_train and not args.do_test:
        raise NotImplementedError("You should specify one of `--do_train` and `--do_test`")

    return args

def process_tasks(idx, task_list, args, fail_dict):

    # debug mode, process 2 tasks in each process
    if args.debug:
        task_list = task_list[:2]

    print("Process {} is handling the following tasks: {}".format(idx, task_list))

    failed_tasks = []
    for task in task_list:
        print("Process {}: Processing {} ...".format(idx, task["hf_identifier"]))
        ## Added extra args for hf identifier and prompt template
        command = "python3 nmt.py --hf_identifier %s%s%s%s --train_k %d --test_k %d" % (
            task["hf_identifier"],
            " --inst" if args.inst else "",
            " --do_train" if args.do_train else "",
            " --do_test" if args.do_test else "",
            args.train_k,
            args.test_k)
        print(command)
        ret_code = subprocess.run([command], shell=True) # stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
        if ret_code.returncode != 0:
            print("Process {}: Processing {} ... [Failed]".format(idx, task))
            failed_tasks.append(command)
        else:
            print("Process {}: Processing {} ... [Success]".format(idx, task))
    fail_dict[idx] = failed_tasks

def build_gym(args):
    successful = []
    failed = []
    all_tasks = ALL_TASKS
    # for filename in sorted(os.listdir(args.task_dir)):
    #     if filename.endswith(".py") and not filename.startswith("0") and not filename.startswith("_") and \
    #             filename!="utils.py" and "unifiedqa" not in filename:
    #         all_tasks.append(filename)

    assert all_tasks == ALL_TASKS
    print("Passing file checks ...")

    manager = Manager()
    fail_dict = manager.dict()

    if args.n_proc > 1:
        tasks_per_proc = int(len(all_tasks) / args.n_proc)
        tasks_split = [all_tasks[i * tasks_per_proc: (i+1) * tasks_per_proc] for i in range(args.n_proc - 1)]
        tasks_split.append(all_tasks[(args.n_proc-1) * tasks_per_proc:])

        processes = []
        for i in range(args.n_proc):
            p = mp.Process(target=process_tasks, args=(i+1, tasks_split[i], args, fail_dict))
            p.start()
            processes.append(p)

        for proc in processes:
            proc.join()
    else:
        process_tasks(0, all_tasks, args, fail_dict)

    all_failed_tasks = []
    for item in fail_dict.values():
        all_failed_tasks += item
    if len(all_failed_tasks) > 0:
        print("Please try the following tasks later by running {}".format(all_failed_tasks))
    else:
        print("Processing finished successfully.")

def main():
    args = parse_args()
    build_gym(args)

if __name__ == "__main__":
    main()
