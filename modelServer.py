"""
====================================================================
 JIRA: PLATFORM-2930 — Fix TensorFlow Model Serving OOM
====================================================================
 Priority: P0 | Points: 2 | Labels: ml, python, production, memory
 
 Model server runs out of memory after ~200 requests. No batch limit
 on inference requests and GPU memory isn't configured.
 
 PRODUCTION LOG:
 [ERROR] OOM: allocator ran out of memory trying to allocate 2.4GiB
 [ERROR] Pod killed: OOMKilled — memory limit 4Gi exceeded
 
 ACCEPTANCE CRITERIA:
 - [ ] Batch size limited to configurable max (default 32)
 - [ ] GPU memory growth enabled (not pre-allocated)
 - [ ] Requests exceeding max batch size return 413
 - [ ] Model loaded once, reused across requests
====================================================================
"""

import numpy as np

class ModelServer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        # This leaks memory as old model references aren't freed
        print(f"Loading model from {self.model_path}")
        self.model = {"weights": np.random.randn(1000, 1000)}  # Simulate

    def predict(self, inputs):
        """Run inference on input batch."""
        self.load_model()

        # Should reject if len(inputs) > MAX_BATCH_SIZE
        batch = np.array(inputs)

        # Should enable memory growth: tf.config.experimental.set_memory_growth

        # Simulate inference
        results = []
        for inp in batch:
            score = float(np.dot(inp[:len(self.model['weights'])],
                                 self.model['weights'][:len(inp), 0]))
            results.append({'score': score, 'label': 'positive' if score > 0 else 'negative'})

        return results


# Tests
if __name__ == '__main__':
    server = ModelServer("model_v3.h5")
    result = server.predict([[1.0, 2.0, 3.0]])
    assert len(result) == 1, "FAIL: Should return one result"

    # Large batch should be rejected
    huge_batch = [[float(i)] * 100 for i in range(10000)]
    try:
        server.predict(huge_batch)
        assert False, "FAIL: Should reject batch > 32"
    except Exception:
        pass
    print("Model server tests complete")
