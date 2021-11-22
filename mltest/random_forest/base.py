import subprocess


def check_status_code(completed_process: subprocess.CompletedProcess):
    if completed_process.returncode != 0:
        raise RuntimeError('Error in child process')


class RFTest(object):

    def run_test(self, sk_model_obj, x_test, y_test):
        raise NotImplementedError


class InferenceTimeTest(RFTest):

    def __init__(self, n_repeats=5, max_samples=5000):
        self.n_repeats = n_repeats
        self.max_samples = max_samples

    def _cut_test_data(self, x_test, y_test):
        test_sample_size = min(y_test.shape[0], self.max_samples)

        return x_test[:test_sample_size], y_test[:test_sample_size]
