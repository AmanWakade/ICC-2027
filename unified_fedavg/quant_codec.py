"""Quantise client->server model parameters; FP32 server->client remains unchanged."""
import numpy as np

def encode(weights, precision):
    if precision == 'fp32':
        return [np.asarray(w, dtype=np.float32) for w in weights]
    if precision == 'fp16':
        return [np.asarray(w, dtype=np.float16) for w in weights]
    if precision == 'int8':
        encoded = []
        for w in weights:
            a = np.asarray(w, dtype=np.float32)
            maxabs = float(np.max(np.abs(a))) if a.size else 0.0
            scale = np.float32(maxabs / 127.0 if maxabs > 0 else 1.0)
            q = np.clip(np.rint(a / scale), -127, 127).astype(np.int8)
            encoded.extend([q, np.array([scale], dtype=np.float32)])
        return encoded
    raise ValueError(precision)

def decode(arrays, precision, reference_shapes):
    if precision == 'fp32':
        if len(arrays) != len(reference_shapes):
            raise ValueError('FP32 tensor count mismatch')
        result = [np.asarray(a, dtype=np.float32) for a in arrays]
    elif precision == 'fp16':
        if len(arrays) != len(reference_shapes):
            raise ValueError('FP16 tensor count mismatch')
        result = [np.asarray(a, dtype=np.float32) for a in arrays]
    elif precision == 'int8':
        if len(arrays) != 2 * len(reference_shapes):
            raise ValueError('INT8 tensor/scale count mismatch')
        result = []
        for i in range(0, len(arrays), 2):
            q, scale = arrays[i], arrays[i + 1]
            if q.dtype != np.int8 or scale.size != 1:
                raise ValueError('Invalid INT8 payload')
            result.append(q.astype(np.float32) * float(scale.reshape(-1)[0]))
    else:
        raise ValueError(precision)
    for a, shape in zip(result, reference_shapes):
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError('Invalid decoded tensor')
    return result
