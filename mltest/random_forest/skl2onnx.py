import os
import time
import uuid

from mltest.thoughts import InferenceTimeTest


class Skl2OnnxInferenceRFTimeTest(InferenceTimeTest):

    def __init__(self, n_repeats=5, max_samples=5000):
        super().__init__(n_repeats, max_samples)
        self.work_dir = os.path.join('/tmp', f'sk2onnx-infernece-{uuid.uuid4()}')

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def run_test(self, sk_model_obj, x_test, y_test):
        # Convert into ONNX format
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, x_test.shape[1]]))]
        onx = convert_sklearn(sk_model_obj, initial_types=initial_type)
        with open(os.path.join(self.work_dir, "rf_iris.onnx"), "wb") as f:
            f.write(onx.SerializeToString())

        # Compute the prediction with ONNX Runtime
        import onnxruntime as rt
        import numpy
        sess = rt.InferenceSession(os.path.join(self.work_dir, "rf_iris.onnx"))
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        start = time.time()
        for repeat in range(self.n_repeats):
            pred_onx = sess.run([label_name], {input_name: x_test.astype(numpy.float32)})[0]
        end = time.time()

        return {
            'latency': (end - start) / self.n_repeats / x_test.shape[0]
        }

