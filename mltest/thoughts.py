import os
import pickle
import shutil
import subprocess
import sys
import time
import uuid
from collections import OrderedDict
from shutil import copyfile

import fastinference.Loader
import numpy as np
import pandas as pd
import treelite
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, \
    mean_absolute_percentage_error, r2_score

from mltest.random_forest.base import RFTest, InferenceTimeTest, check_status_code
# Group Test?
# Hierarchical
# we have groups of tests
# - Each group can summarize its results to an artifact??
from mltest.random_forest.skl2onnx import Skl2OnnxInferenceRFTimeTest


# TODO: Cleanup...

class TreeLiteInferenceTimeTest(InferenceTimeTest):

    def __init__(self, n_repeats=5, max_samples=5000, tmp_dir='/tmp'):
        super().__init__(n_repeats, max_samples)
        self.tmp_dir = tmp_dir

        self.work_dir = os.path.join(self.tmp_dir, f'cl-treelite-{uuid.uuid4()}')

        # print(self.work_dir)

        os.makedirs(self.work_dir)

    def _build_binary(self, sk_model_obj, x_test, y_test):
        os.makedirs(self.work_dir, exist_ok=True)

        treelite_model = treelite.sklearn.import_model(sk_model_obj)
        if hasattr(sk_model_obj, "n_estimators"):
            treelite_model.compile(dirpath=self.work_dir, params={"parallel_comp": sk_model_obj.n_estimators}, verbose=True)
        else:
            treelite_model.compile(dirpath=self.work_dir, verbose=True)

        os.rename(os.path.join(self.work_dir, 'main.c'), os.path.join(self.work_dir, 'model.cpp'))  # cp {outpath}/main.c {outpath}/model.cpp

        for file in os.listdir(self.work_dir):
            if file.endswith('.c'):
                abs_in = os.path.join(self.work_dir, file)
                abs_out = abs_in.replace('.c', '.cpp')
                os.rename(abs_in, abs_out)

        # Copy CMAKE Template to work directory
        copyfile("/home/lukas/workspace/carelabel-single/resources/seb/CMakeLists.txt", os.path.join(self.work_dir, 'CMakeLists.txt'))  # cp ./main.cpp {outpath}
        copyfile("/home/lukas/workspace/carelabel-single/resources/seb/main.cpp", os.path.join(self.work_dir, 'main.cpp'))  #

        build_dir = os.path.join(self.work_dir, 'build')

        os.makedirs(build_dir)
        subprocess.run(['cmake', '..', '-DMODELNAME=treelite', '-DFEATURE_TYPE=double', '-DTREELITE=ON'], cwd=build_dir)  # , capture_output=True

        subprocess.run(['make'], cwd=build_dir)

    def run_test(self, sk_model_obj, x_test, y_test):
        # (x_train, x_test, y_train, y_test) = dataset

        # Limit data to max samples
        x_test_prep, y_test_prep = self._cut_test_data(x_test, y_test)

        dataset_uuid = uuid.uuid4()

        test_file_path = os.path.join(self.work_dir, f'test_{dataset_uuid}.csv')  # cp {test_file} {outpath}/

        np.savetxt(test_file_path, x_test_prep, delimiter=',', fmt='%f')
        data = pd.read_csv(test_file_path, header=None)
        data['label'] = y_test_prep

        data.to_csv(test_file_path, index=False)

        self._build_binary(sk_model_obj, x_test, y_test)

        build_dir = os.path.join(self.work_dir, 'build')

        if not os.path.exists(os.path.join(build_dir, 'testCode')):
            raise RuntimeError('Binary not located at expected path.')

        # Now execute binary and parse output...

        output = subprocess.check_output([
            os.path.join(build_dir, "testCode"),
            test_file_path,
            str(sk_model_obj.n_classes_),
            str(self.n_repeats)
        ]).decode(sys.stdout.encoding).strip()

        # Cleanup
        shutil.rmtree(self.work_dir)

        print(output)

        accuracy = output.split("\n")[-1].split(",")[0]
        latency = output.split("\n")[-1].split(",")[3]

        result = {
            'implementation': 'treelite',
            'accuracy': accuracy,
            'latency': latency
        }
        print(result)
        return result


class FastInferenceTimeTest(InferenceTimeTest):

    def __init__(self, name,
                 n_repeats=5,
                 max_samples=5000,
                 implementation_type="ifelse",
                 implementation_args=None,
                 optimizer="",
                 optimizer_args=None):
        super().__init__(n_repeats, max_samples)

        self.name = name

        if optimizer_args is None:
            optimizer_args = []
        if implementation_args is None:
            implementation_args = {}
        self.implementation_type = implementation_type
        self.implementation_args = implementation_args
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

        self.work_dir = os.path.join('/tmp', f'cl-fastinference-{uuid.uuid4()}')

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # print(self.work_dir)

    def run_test(self, sk_model_obj, x_test, y_test):

        fi_model = fastinference.Loader.model_from_sklearn(sk_model_obj, self.name, accuracy=None)

        if isinstance(sk_model_obj, RandomForestClassifier):
            fi_model.optimize(optimizers="", args="", base_optimizers=self.optimizer, base_args=self.optimizer_args)
            fi_model.implement(self.work_dir, "model", "cpp", "cpp.{}".format(self.implementation_type), **self.implementation_args)
        else:
            fi_model.optimize(self.optimizer, self.optimizer, self.optimizer_args)
            fi_model.implement(self.work_dir, "model", "cpp.{}".format(self.implementation_type), **self.implementation_args)

        # (x_train, x_test, y_train, y_test) = dataset

        test_file_path = os.path.join(self.work_dir, f'test_{uuid.uuid4()}.csv')

        # Limit data to max samples
        x_test_prep, y_test_prep = self._cut_test_data(x_test, y_test)

        np.savetxt(test_file_path, x_test_prep, delimiter=',', fmt='%f')
        data = pd.read_csv(test_file_path, header=None)
        data['label'] = y_test_prep

        data.to_csv(test_file_path, index=False)

        self._prepare_and_compile()

        output = subprocess.check_output([
            os.path.join(self.work_dir, 'build', "testCode"),
            test_file_path,
            str(sk_model_obj.n_classes_),
            str(5)
        ]).decode(sys.stdout.encoding).strip()

        accuracy = output.split("\n")[-1].split(",")[0]
        latency = output.split("\n")[-1].split(",")[3]

        return {
            "implementation": self.name,
            "accuracy": accuracy,
            "latency": latency
        }

    def _prepare_and_compile(self):
        # Copy CMAKE Template to work directory
        copyfile("/home/lukas/workspace/carelabel-single/resources/seb/CMakeLists.txt", os.path.join(self.work_dir, 'CMakeLists.txt'))  # cp ./main.cpp {outpath}
        copyfile("/home/lukas/workspace/carelabel-single/resources/seb/main.cpp", os.path.join(self.work_dir, 'main.cpp'))  #

        # TODO: Generate test file

        build_dir = os.path.join(self.work_dir, 'build')

        os.makedirs(build_dir)
        ret = subprocess.run(['cmake', '..', f'-DMODELNAME={self.name}', '-DFEATURE_TYPE=double'], cwd=build_dir)  # , capture_output=True
        check_status_code(ret)

        ret = subprocess.run(['make'], cwd=build_dir)
        check_status_code(ret)


class RFSklearnRuntimeTest(InferenceTimeTest):

    def __init__(self, n_repeats=5, max_samples=5000):
        super().__init__(n_repeats, max_samples)

    def run_test(self, sk_model_obj, x_test, y_test):
        # (x_train, x_test, y_train, y_test) = dataset

        # Limit data to max samples
        x_test_prep, y_test_prep = self._cut_test_data(x_test, y_test)

        now = time.time()

        for r in range(self.n_repeats):
            y_hat = sk_model_obj.predict(x_test_prep)

        end = time.time()

        return {
            'mean_runtime': (end - now) / self.n_repeats / x_test_prep.shape[0],
            'test_samples': x_test_prep.shape[0]
        }


all_tests = [{
    'test_name': 'RandomForest',
    'test_class': 'RegressionMetricTest',
    'test_clz': 'mltest.thoughts',
    'task': ['regression'],
    'requirements': [],
    'constraints': [],
}]

# test = find_matching_tests(all_tests)

# Dynamisch
# pipeline = TestPipeline.from(test)

# Ausführen


class RegressionMetricTest(RFTest):
    regression_metrics = {
        'eva': explained_variance_score,
        'max_error': max_error,
        'mean_abolsute_error': mean_absolute_error,
        'mean_squared_error': mean_squared_error,
        # 'mean_squared_log_error': mean_squared_log_error,
        'median_absolute_error': median_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error,
        'r2_score': r2_score
    }

    def run_test(self, sk_model_obj, x_test, y_test):
        # (x_train, x_test, y_train, y_test) = dataset
        y_hat = sk_model_obj.predict(x_test)

        result = {}
        for metric_name, metric_fn in self.regression_metrics.items():
            result[metric_name] = metric_fn(y_hat, y_test)
        return result


class RFBiasVarTest(RFTest):

    def run_test(self, sk_model_obj, x_test, y_test):
        # https://www.jmlr.org/papers/volume6/brown05a/brown05a.pdf

        # https://docs.google.com/document/d/1rdwzcBU63aHjID0iFsXDYZjWY4BORK8mW1v8Y7yVNRQ/edit

        preds_array = np.empty((sk_model_obj.n_estimators, x_test.shape[0]))
        for idx, estimator in enumerate(sk_model_obj.estimators_):
            preds_array[idx, :] = estimator.predict(x_test)

        mean_ensemble_prediction = np.mean(preds_array)

        predictions = sk_model_obj.predict(x_test)

        bias_bar = np.mean(predictions - y_test)

        learner_prediction_mean = preds_array.mean(axis=1)

        bias = np.mean(mean_ensemble_prediction - y_test)

        label_mean = y_test.mean()
        # What is t??
        t = label_mean
        bias_bar = np.mean(learner_prediction_mean - t)

        var_bar = []
        for i in range(sk_model_obj.n_estimators):
            inner = np.mean((preds_array[i] - learner_prediction_mean[i]) ** 2)
            var_bar.append(inner)
        var_bar = np.mean(var_bar)

        M = sk_model_obj.n_estimators
        scaling_factor = (1 / ((M - 1) * M))

        inner_val = 0
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                inner_val += ((preds_array[i] - learner_prediction_mean[i]) * (preds_array[j] - learner_prediction_mean[j])).mean()

        covbar = scaling_factor * inner_val

        mse = mean_squared_error(y_true=y_test, y_pred=predictions)

        print('Bias bar:', bias_bar)
        print('Var bar:', var_bar)
        print('cov bar:', covbar)

        return {
            'bias_bar': bias_bar,
            'var_bar': var_bar,
            'covar_bar': covbar,
            'rescaled_covarbar': (1 - (1 / M)) * covbar,
            'success': mse < (1 - (1 / M)) * covbar
        }


class TestPipeline(object):

    def __init__(self):
        self.tests = OrderedDict()

    def register(self, test, name):
        self.tests[name] = test

    def run(self, sk_model_obj, x_test, y_test):
        results = {}
        for test_name, test in self.tests.items():
            test_result = test.run_test(sk_model_obj, x_test, y_test)
            results[test_name] = test_result
        return results


class SKLearnModelSizeTest(RFTest):

    def __init__(self):
        self.work_dir = '/tmp'

    def run_test(self, sk_model_obj, x_test, y_test):
        # save the classifier
        with open(os.path.join(self.work_dir, 'my_dumped_classifier.pkl'), 'wb') as fid:
            pickle.dump(sk_model_obj, fid)

        # Get file size
        pickle_size = os.path.getsize(os.path.join(self.work_dir, 'my_dumped_classifier.pkl'))

        # Remove file
        os.remove(os.path.join(self.work_dir, 'my_dumped_classifier.pkl'))

        return {
            'pickle_size': pickle_size
        }


def generate_test_pipeline():
    # Create test pipeline as meta (json)

    #

    test_pipeline = TestPipeline()
    test_pipeline.register(TreeLiteInferenceTimeTest(), 'tree_lite_inference')
    test_pipeline.register(Skl2OnnxInferenceRFTimeTest(), 'skl2onx_inference_time')
    test_pipeline.register(
        FastInferenceTimeTest(
            name='ifelse', implementation_type='ifelse'), 'fast_inference_ifelse')
    test_pipeline.register(
        FastInferenceTimeTest(
            name='ifelse_swap', implementation_type='ifelse', optimizer='swap'), 'fast_inference_ifelse_swap')

    # TODO: Optimize w.r.t name
    # TODO: OnnxToSKlearn

    for s in [1024, 2048, 4096]:
        test_pipeline.register(
            FastInferenceTimeTest(name=f'ifelse_swap_path_{s}', implementation_type='ifelse', optimizer='swap',
                                  implementation_args={"kernel_type": "path", "kernel_budget": s}), name=f'ifelse_swap_path_{s}')

        test_pipeline.register(
            FastInferenceTimeTest(name=f'ifelse_swap_node_{s}', implementation_type='ifelse', optimizer='swap',
                                  implementation_args={"kernel_type": "node", "kernel_budget": s}), name=f'ifelse_swap_node_{s}')

    test_pipeline.register(
        FastInferenceTimeTest(
            name='native', implementation_type='native'), 'fast_inference_native')

    for s in [2, 4, 6, 8, 10]:
        test_pipeline.register(
            FastInferenceTimeTest(
                name=f'native_optimized_{s}', implementation_type='native', implementation_args={'set_size': s}), f'fast_inference_optimized_native_{s}')
        test_pipeline.register(
            FastInferenceTimeTest(
                name=f'native_optimized_full_{s}', implementation_type='native', implementation_args={'set_size': s, 'force_cacheline': True}),
            f'fast_inference_native_optimized_full_{s}')

    test_pipeline.register(SKLearnModelSizeTest(), 'sklearn_pickle_size_test')
    return test_pipeline
