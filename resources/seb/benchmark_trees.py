#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

import fastinference.Loader
import pandas as pd
import treelite
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def eval_treelite(model, out_path, name, benchmark_file, n_repeat=5):
    treelite_model = treelite.sklearn.import_model(model)
    if hasattr(model, "n_estimators"):
        treelite_model.compile(dirpath=out_path, params={"parallel_comp": model.n_estimators}, verbose=True)
    else:
        treelite_model.compile(dirpath=out_path, verbose=True)

    prepare_and_compile = """
    cp {test_file} {outpath}/ && 
    cp {outpath}/main.c {outpath}/model.cpp &&
    for f in {outpath}/*.c; do mv -- "$f" "${f%.c}.cpp"; done &&
    cp ./CMakeLists.txt {outpath} &&
    cp ./main.cpp {outpath} && 
    cd {outpath} &&
    cmake . -DMODELNAME={name} -DFEATURE_TYPE={feature_type} -DTREELITE=ON &&
    make""".replace("{outpath}", out_path).replace("{name}", name).replace("{feature_type}", "double").replace("{test_file}", benchmark_file)

    # command.split()
    subprocess.call(prepare_and_compile, shell=True)
    output = subprocess.check_output([
        "/{outpath}/testCode".replace("{outpath}", out_path),
        "/{outpath}/{test_file}".replace("{outpath}", out_path).replace("{test_file}", benchmark_file),
        str(model.n_classes_),
        str(n_repeat)
    ]).decode(sys.stdout.encoding).strip()

    accuracy = output.split("\n")[-1].split(",")[0]
    latency = output.split("\n")[-1].split(",")[3]

    return {
        "implementation": "treelite",
        "accuracy": accuracy,
        "latency": latency
    }


def eval_fastinference(model, out_path, name, benchmark_file, n_repeat=5, implementation_type="ifelse", implementation_args={}, optimizer="", optimizer_args=[]):
    print("Loading testing data")
    df = pd.read_csv(benchmark_file)
    y_test = df["label"].to_numpy()
    x_test = df.drop(columns=["label"]).to_numpy()
    print("")

    accuracy = accuracy_score(y_test, model.predict(x_test)) * 100.0

    print("Exporting {}".format(name))
    fi_model = fastinference.Loader.model_from_sklearn(model, name, accuracy=accuracy)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print("Exporting {} to {}".format(name, out_path))

    if isinstance(model, RandomForestClassifier):
        fi_model.optimize(optimizers="", args="", base_optimizers=optimizer, base_args=optimizer_args)
        fi_model.implement(out_path, "model", "cpp", "cpp.{}".format(implementation_type), **implementation_args)
    else:
        fi_model.optimize(optimizer, optimizer, optimizer_args)
        fi_model.implement(out_path, "model", "cpp.{}".format(implementation_type), **implementation_args)

    prepare_and_compile = """
    cp ./main.cpp {outpath} && 
    cp {test_file} {outpath}/ && 
    cp ./CMakeLists.txt {outpath} &&
    cd {outpath} &&
    cmake . -DMODELNAME={name} -DFEATURE_TYPE={feature_type} &&
    make""".replace("{outpath}", out_path).replace("{name}", name).replace("{feature_type}", "double").replace("{test_file}", benchmark_file)

    # command.split()
    subprocess.call(prepare_and_compile, shell=True)
    output = subprocess.check_output([
        "/{outpath}/testCode".replace("{outpath}", out_path),
        "/{outpath}/{test_file}".replace("{outpath}", out_path).replace("{test_file}", benchmark_file),
        str(model.n_classes_),
        str(n_repeat)
    ]).decode(sys.stdout.encoding).strip()

    accuracy = output.split("\n")[-1].split(",")[0]
    latency = output.split("\n")[-1].split(",")[3]

    return {
        "implementation": name,
        "accuracy": accuracy,
        "latency": latency
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train linear models on supplied data. This script assumes that each supplied training / testing CSV has a unique column called `label` which contains the labels.')
    parser.add_argument('--training', required=True, help='Filename of training data CSV-file')
    parser.add_argument('--testing', required=True, help='Filename of testing data CSV-file')
    parser.add_argument('--out', required=True, help='Folder where data should written to.')
    parser.add_argument('--name', required=True, help='Modelname')
    parser.add_argument('--nestimators', required=False, type=int, default=8, help='Number of trees in a random forest.')
    parser.add_argument('--maxdepth', required=False, type=int, default=20, help='Maximum tree-depth for decision trees and random forest.')
    args = parser.parse_args()

    print("Loading training data")
    df = pd.read_csv(args.training)
    y_train = df["label"].to_numpy()
    x_train = df.drop(columns=["label"]).to_numpy()

    performance = []

    if args.nestimators <= 1:
        model = DecisionTreeClassifier(max_depth=args.maxdepth)
    else:
        model = RandomForestClassifier(n_estimators=args.nestimators, max_depth=args.maxdepth)

    model.fit(x_train, y_train)

    # performance.append(eval_fastinference(model,os.path.join(args.out, "ifelse"), "ifelse", args.testing, 5, "ifelse"))
    # performance.append(eval_fastinference(model,os.path.join(args.out, "ifelse_swap"), "ifelse_swap", args.testing, 5, "ifelse", optimizer="swap"))
    # performance.append(eval_fastinference(model,os.path.join(args.out, "native"), "native", args.testing, 5, "native"))
    # for s in [2,4,6,8,10]:
    #     performance.append(eval_fastinference(model,os.path.join(args.out, "native_optimized_{}".format(s)), "native_optimized_{}".format(s), args.testing, 5, implementation_type="native", implementation_args= {"set_size":s}))
    #     performance.append(eval_fastinference(model,os.path.join(args.out, "native_optimized_full_{}".format(s)), "native_optimized_full_{}".format(s), args.testing, 5, implementation_type="native", implementation_args= {"set_size":s, "force_cacheline":True}))
    performance.append(eval_treelite(model, os.path.join(args.out, "treelite"), args.name, args.testing, 5))

    df = pd.DataFrame(performance)
    print(df)


if __name__ == '__main__':
    # print(sys.argv)
    # sys.argv = ['benchmark_trees.py', '--training', 'training.csv', '--testing', 'testing.csv', '--out', 'tmp/', '--name', 'model']
    main()
