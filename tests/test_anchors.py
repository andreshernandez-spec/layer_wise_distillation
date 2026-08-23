import numpy as np
import torch

from lwd.harvest.anchors import (channel_relative_error, dequantize, load, pack,
                                 quantize, relative_error, rotation)


def _stream(n=512, d=256, outlier=200.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    x[:, 7] *= outlier  # massive-activation channel
    return x


def test_rotation_is_orthogonal():
    Q = rotation(96, seed=3)
    np.testing.assert_allclose((Q @ Q.T).numpy(), np.eye(96), atol=2e-5)
    assert torch.equal(Q, rotation(96, seed=3))


def test_int8_channel_normalized_is_the_best_store():
    x = _stream()
    sd = x.std(0)
    err = {}
    err["int8 chan"] = channel_relative_error(x, dequantize(*quantize(x, sd, "int8"), sd))
    err["int8 plain"] = channel_relative_error(x, dequantize(*quantize(x, None, "int8")))
    err["fp8 plain"] = channel_relative_error(x, dequantize(*quantize(x, None, "fp8")))
    Q = rotation(256)
    err["fp8 rot"] = channel_relative_error(x, dequantize(*quantize(x, None, "fp8", Q), None, Q))
    assert err["int8 chan"] < 0.01, err
    assert err["fp8 plain"] < 0.04, err
    assert err["int8 plain"] > 0.2 and err["fp8 rot"] > 0.2, err  # the two traps
    # the raw-norm metric cannot see the int8-plain failure: the outlier channel is exact
    assert relative_error(x, dequantize(*quantize(x, None, "int8"))) < err["int8 plain"] / 5


def test_pack_and_load_roundtrip(tmp_path):
    x = _stream(n=24).reshape(6, 4, 256)
    sd = x.reshape(-1, 256).std(0)
    np.save(tmp_path / "ref_000000.npy", x[:3].half().numpy())
    np.save(tmp_path / "ref_000001.npy", x[3:].half().numpy())
    pack(sorted(tmp_path.glob("ref_*.npy")), tmp_path / "store", sd)
    y = load(tmp_path / "store")
    assert y.shape == x.shape
    assert channel_relative_error(x, y) < 0.01
    assert len(list((tmp_path / "store").glob("shard_*.npz"))) == 2
