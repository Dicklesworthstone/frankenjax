#!/usr/bin/env python3
"""Smoke test for frankenjax Python bindings.

Run after building with:
    maturin develop -m crates/fj-py/Cargo.toml
    python crates/fj-py/tests/smoke_test.py
"""

import frankenjax as fj
import numpy as np
import operator
import struct


def test_version_metadata():
    """Test package version metadata exports."""
    assert fj.__version__ == "0.1.0"
    assert fj.__version_info__ == (0, 1, 0)
    assert fj.__array_api_version__ == "2024.12"
    print("✓ __version__/__version_info__ metadata")


def test_float0_sentinel():
    """Test zero-tangent dtype sentinel metadata."""
    assert str(fj.float0) == "float0"
    assert repr(fj.float0) == "float0"
    assert fj.float0.name == "float0"
    print("✓ float0 sentinel metadata")


def test_environment_info():
    """Test environment info helper return and print modes."""
    info = fj.print_environment_info(return_string=True)
    assert "jax:    0.1.0" in info
    assert "device info: cpu-1, 1 local devices" in info
    assert "process_count: 1" in info
    assert fj.print_environment_info() is None
    print("✓ print_environment_info returns and prints CPU-local diagnostics")


def test_value_scalar():
    """Test scalar value creation and retrieval."""
    assert fj.Array is fj.PyValue

    v = fj.PyValue.scalar_f64(42.0)
    assert isinstance(v, fj.Array)
    assert v.shape == ()
    assert v.T.shape == ()
    assert abs(v.T.item() - 42.0) < 1e-12
    assert v.__array_namespace__() is fj
    assert v.__array_namespace__(api_version="2024.12") is fj
    try:
        v.__array_namespace__(api_version="2023.12")
    except ValueError as exc:
        assert "available versions" in str(exc)
        assert "2024.12" in str(exc)
    else:
        raise AssertionError("__array_namespace__ should reject unsupported versions")
    assert v.dtype == "F64"
    assert isinstance(v.aval, fj.ShapeDtypeStruct)
    assert v.aval.shape == ()
    assert v.aval.dtype == "F64"
    assert not v.aval.weak_type
    assert v.__numpy_dtype__ == np.dtype("float64")
    try:
        hash(v)
    except TypeError as exc:
        assert "unhashable type" in str(exc)
    else:
        raise AssertionError("Array should be unhashable")
    assert v.ndim == 0
    assert v.size == 1
    assert v.itemsize == 8
    assert v.nbytes == 8
    assert v.weak_type is False
    assert v.committed is False
    assert isinstance(v.device, fj.Device)
    assert v.device.id == 0
    assert v.device.process_index == 0
    assert v.device.platform == "cpu"
    assert v.device.device_kind == "cpu"
    moved = v.to_device(v.device)
    assert isinstance(moved, fj.Array)
    assert abs(moved.as_f64() - 42.0) < 1e-12
    streamed = v.to_device(fj.local_devices()[0], stream=None)
    assert abs(streamed.as_f64() - 42.0) < 1e-12
    assert v.platform() == "cpu"
    assert {device.platform for device in v.devices()} == {"cpu"}
    assert str(v) == "42.0"
    assert format(v, ".1f") == "42.0"
    assert v.astype("int64").as_i64() == 42
    assert abs(v.astype(None).as_f64() - 42.0) < 1e-12
    assert abs(v.reshape(()).as_f64() - 42.0) < 1e-12
    assert v.reshape(1).shape == (1,)
    assert v.reshape(1).as_f64_list() == [42.0]
    assert v.flatten().shape == (1,)
    assert v.flatten().as_f64_list() == [42.0]
    assert v.ravel().as_f64_list() == [42.0]
    try:
        v.astype("float64", device="gpu")
    except ValueError as exc:
        assert "device" in str(exc)
    else:
        raise AssertionError("Array.astype should reject non-CPU devices")
    assert abs(v.round().as_f64() - 42.0) < 1e-12
    assert round(fj.PyValue.scalar_f64(10.5)).as_i64() == 10
    assert round(fj.PyValue.scalar_f64(21.5)).as_i64() == 22
    assert abs(round(fj.PyValue.scalar_f64(1.25), 1).as_f64() - 1.2) < 1e-12
    try:
        v.round(out=object())
    except NotImplementedError as exc:
        assert "out" in str(exc)
    else:
        raise AssertionError("Array.round should reject non-None out")
    array = np.asarray(v)
    assert array.shape == ()
    assert abs(array.item() - 42.0) < 1e-12
    assert np.asarray(v, dtype=np.float32).dtype == np.dtype("float32")
    dl_device_type, dl_device_id = v.__dlpack_device__()
    assert int(dl_device_type) == 1
    assert dl_device_id == 0
    try:
        v.device_buffer
    except AttributeError as exc:
        assert "Use arr.addressable_data(0)" in str(exc)
    else:
        raise AssertionError("device_buffer should raise the upstream deprecation error")
    try:
        v.device_buffers
    except AttributeError as exc:
        assert "Use [x.data for x in arr.addressable_shards]" in str(exc)
    else:
        raise AssertionError("device_buffers should raise the upstream deprecation error")
    assert abs(v.addressable_data(0).as_f64() - 42.0) < 1e-12
    try:
        v.addressable_data(1)
    except IndexError as exc:
        assert "out of bounds" in str(exc)
    else:
        raise AssertionError("addressable_data should reject nonzero shard indexes")
    addressable_shards = v.addressable_shards
    assert len(addressable_shards) == 1
    shard = addressable_shards[0]
    assert isinstance(shard, fj.Shard)
    assert shard.device.platform == "cpu"
    assert shard.index == ()
    assert shard.replica_id == 0
    assert abs(shard.data.as_f64() - 42.0) < 1e-12
    global_shards = v.global_shards
    assert len(global_shards) == 1
    assert abs(global_shards[0].data.as_f64() - 42.0) < 1e-12
    assert v.on_device_size_in_bytes() == v.nbytes
    assert v.is_fully_addressable is True
    assert v.is_fully_replicated is True
    try:
        len(v)
    except TypeError as exc:
        assert str(exc) == "len() of unsized object"
    else:
        raise AssertionError("len(scalar) should raise TypeError")
    try:
        iter(v)
    except TypeError as exc:
        assert str(exc) == "iteration over a 0-d array"
    else:
        raise AssertionError("iter(scalar) should raise TypeError")
    try:
        v[0]
    except IndexError as exc:
        assert "Too many indices" in str(exc)
    else:
        raise AssertionError("scalar indexing should raise IndexError")
    ready = v.block_until_ready()
    assert isinstance(ready, fj.Array)
    assert abs(ready.as_f64() - 42.0) < 1e-12
    assert v.is_ready()
    assert v.copy_to_host_async() is None
    copied = v.copy()
    assert isinstance(copied, fj.Array)
    assert abs(copied.as_f64() - 42.0) < 1e-12
    deleted = v.copy()
    assert deleted.is_deleted() is False
    assert deleted.delete() is None
    assert deleted.is_deleted() is True

    def assign_first(array):
        array[0] = 9

    try:
        deleted.is_ready()
    except RuntimeError as exc:
        assert "Array has been deleted" in str(exc)
    else:
        raise AssertionError("is_ready should reject deleted arrays")
    try:
        deleted.addressable_data(0)
    except RuntimeError as exc:
        assert "Array has been deleted" in str(exc)
    else:
        raise AssertionError("addressable_data should reject deleted arrays")
    deleted_accessors = [
        ("block_until_ready", deleted.block_until_ready),
        ("copy_to_host_async", deleted.copy_to_host_async),
        ("devices", deleted.devices),
        ("tolist", deleted.tolist),
        ("tobytes", deleted.tobytes),
        ("device_get", lambda: fj.device_get(deleted)),
        ("module block_until_ready", lambda: fj.block_until_ready(deleted)),
        ("module copy_to_host_async", lambda: fj.copy_to_host_async(deleted)),
        ("astype", lambda: deleted.astype("float64")),
        ("reshape", lambda: deleted.reshape(1)),
        ("flatten", deleted.flatten),
        ("ravel", deleted.ravel),
        ("squeeze", deleted.squeeze),
        ("item", deleted.item),
        ("indexing", lambda: deleted[0]),
        ("item assignment", lambda: assign_first(deleted)),
        ("T", lambda: deleted.T),
        ("mT", lambda: deleted.mT),
        ("transpose", deleted.transpose),
        ("real", lambda: deleted.real),
        ("imag", lambda: deleted.imag),
        ("conj", deleted.conj),
        ("conjugate", deleted.conjugate),
        ("round", deleted.round),
        ("dunder round", lambda: round(deleted)),
        ("reversed", lambda: list(reversed(deleted))),
        ("array namespace", deleted.__array_namespace__),
        ("array protocol", lambda: np.asarray(deleted)),
        ("DLPack device protocol", deleted.__dlpack_device__),
        ("addressable_shards", lambda: deleted.addressable_shards),
        ("global_shards", lambda: deleted.global_shards),
        ("to_device", lambda: deleted.to_device(v.device)),
    ]
    for name, accessor in deleted_accessors:
        try:
            accessor()
        except RuntimeError as exc:
            assert "Array has been deleted" in str(exc)
        else:
            raise AssertionError(f"{name} should reject deleted arrays")
    assert v.is_deleted() is False
    assert v.item() == 42.0
    assert v.tolist() == 42.0
    assert abs(v.real.tolist() - 42.0) < 1e-12
    assert abs(v.imag.tolist()) < 1e-12
    assert abs(v.conj().tolist() - 42.0) < 1e-12
    assert abs(v.conjugate().tolist() - 42.0) < 1e-12
    assert v.tobytes() == struct.pack("@d", 42.0)
    assert v.tobytes(order="F") == struct.pack("@d", 42.0)
    try:
        v.tobytes(order="bad")
    except ValueError as exc:
        assert "order must be one of" in str(exc)
    else:
        raise AssertionError("tobytes should reject invalid order")
    assert float(v) == 42.0
    assert int(v) == 42
    assert complex(v) == complex(42.0, 0.0)
    assert bool(v) is True
    try:
        operator.index(v)
    except TypeError as exc:
        assert "integer scalar arrays" in str(exc)
    else:
        raise AssertionError("operator.index(float scalar) should raise TypeError")
    try:
        list(reversed(v))
    except TypeError as exc:
        assert "iteration over a 0-d array" in str(exc)
    else:
        raise AssertionError("reversed(scalar) should raise TypeError")
    try:
        v.__hex__()
    except TypeError as exc:
        assert "integer scalar arrays" in str(exc)
    else:
        raise AssertionError("__hex__(float scalar) should raise TypeError")
    assert abs(v.as_f64() - 42.0) < 1e-12
    print("✓ scalar_f64 roundtrip")

    v2 = fj.PyValue.scalar_i64(123)
    assert isinstance(v2, fj.Array)
    assert v2.shape == ()
    assert v2.dtype == "I64"
    assert v2.__numpy_dtype__ == np.dtype("int64")
    assert v2.ndim == 0
    assert v2.size == 1
    assert v2.itemsize == 8
    assert v2.nbytes == 8
    assert v2.weak_type is False
    assert v2.committed is False
    assert v2.is_fully_addressable is True
    assert v2.is_fully_replicated is True
    assert int(v2) == 123
    assert complex(v2) == complex(123.0, 0.0)
    assert operator.index(v2) == 123
    assert v2.__hex__() == "0x7b"
    assert v2.__oct__() == "0o173"
    assert v2.real.as_i64() == 123
    assert v2.imag.as_i64() == 0
    assert abs(v2.astype("float64").as_f64() - 123.0) < 1e-12
    assert v2.round().as_i64() == 123
    try:
        v2.round(decimals=-1)
    except NotImplementedError as exc:
        assert "decimals < 0" in str(exc)
    else:
        raise AssertionError("integer Array.round should reject decimals < 0")
    assert v2.tobytes(order="K") == struct.pack("@q", 123)
    assert bool(v2) is True
    assert bool(fj.PyValue.scalar_i64(0)) is False
    assert v2.as_i64() == 123
    print("✓ scalar_i64 roundtrip")

    vec = fj.PyValue.vector_i64([1, 2, 3])
    assert isinstance(vec, fj.Array)
    assert vec.shape == (3,)
    assert vec.dtype == "I64"
    assert vec.aval.shape == (3,)
    assert vec.aval.dtype == "I64"
    assert vec.__numpy_dtype__ == np.dtype("int64")
    assert vec.ndim == 1
    assert vec.size == 3
    assert vec.itemsize == 8
    assert vec.nbytes == 24
    assert vec.weak_type is False
    assert vec.committed is False
    assert vec.device.platform == "cpu"
    assert vec.is_fully_addressable is True
    assert vec.is_fully_replicated is True
    assert len(vec) == 3
    vec_t = vec.T
    assert vec_t.shape == (3,)
    assert vec_t.tolist() == [1, 2, 3]
    assert vec.real.as_i64_list() == [1, 2, 3]
    assert vec.imag.as_i64_list() == [0, 0, 0]
    assert vec.transpose().tolist() == [1, 2, 3]
    assert vec.transpose(None).tolist() == [1, 2, 3]
    assert vec.transpose(0).tolist() == [1, 2, 3]
    assert vec.transpose((0,)).tolist() == [1, 2, 3]
    try:
        vec.mT
    except ValueError as exc:
        assert "two-dimensional" in str(exc)
    else:
        raise AssertionError("Array.mT should reject vectors")
    assert vec.item(0) == 1
    assert vec.item(-1) == 3
    try:
        vec.item()
    except ValueError as exc:
        assert "size 1" in str(exc)
    else:
        raise AssertionError("vector item() should require size 1")
    try:
        vec.item(3)
    except IndexError as exc:
        assert "out of bounds" in str(exc)
    else:
        raise AssertionError("vector item(index) should reject out-of-bounds indexes")
    try:
        vec[0] = 9
    except TypeError as exc:
        assert "immutable" in str(exc)
        assert "x.at" in str(exc)
    else:
        raise AssertionError("Array item assignment should be immutable")
    iterated = list(vec)
    assert [item.as_i64() for item in iterated] == [1, 2, 3]
    assert [item.shape for item in iterated] == [(), (), ()]
    assert all(isinstance(item, fj.Array) for item in iterated)
    reversed_items = list(reversed(vec))
    assert [item.as_i64() for item in reversed_items] == [3, 2, 1]
    assert [item.shape for item in reversed_items] == [(), (), ()]
    assert all(isinstance(item, fj.Array) for item in reversed_items)
    vec_shard = vec.addressable_shards[0]
    assert vec_shard.index == (slice(None),)
    assert vec_shard.replica_id == 0
    assert vec_shard.data.as_i64_list() == [1, 2, 3]
    assert vec[0].as_i64() == 1
    assert vec[-1].as_i64() == 3
    assert vec[1:].shape == (2,)
    assert vec[1:].as_i64_list() == [2, 3]
    assert vec[::-1].shape == (3,)
    assert vec[::-1].as_i64_list() == [3, 2, 1]
    assert vec[3:].shape == (0,)
    assert vec[3:].as_i64_list() == []
    try:
        vec[3]
    except IndexError as exc:
        assert "index 3 is out of bounds for axis 0 with size 3" in str(exc)
    else:
        raise AssertionError("out-of-bounds vector indexing should raise IndexError")
    assert vec.block_until_ready().as_i64_list() == [1, 2, 3]
    assert vec.copy_to_host_async() is None
    assert vec.copy().as_i64_list() == [1, 2, 3]
    assert vec.tolist() == [1, 2, 3]
    assert vec.astype("float64").as_f64_list() == [1.0, 2.0, 3.0]
    assert vec.reshape(3, 1).shape == (3, 1)
    assert vec.reshape(3, 1).as_i64_list() == [1, 2, 3]
    assert vec.reshape((-1,)).shape == (3,)
    expanded = vec.reshape(1, 3, 1)
    assert expanded.squeeze().shape == (3,)
    assert expanded.squeeze().as_i64_list() == [1, 2, 3]
    assert expanded.squeeze(axis=0).shape == (3, 1)
    assert expanded.squeeze(axis=-1).shape == (1, 3)
    assert expanded.squeeze(axis=(0, 2)).shape == (3,)
    try:
        expanded.squeeze(axis=1)
    except ValueError as exc:
        assert "size not equal to one" in str(exc)
    else:
        raise AssertionError("Array.squeeze should reject non-size-one axes")
    assert vec.flatten().as_i64_list() == [1, 2, 3]
    assert vec.ravel().as_i64_list() == [1, 2, 3]
    assert vec.round().as_i64_list() == [1, 2, 3]
    rounded = fj.PyValue.vector_f64([10.5, 21.5, 12.5, 31.5])
    assert rounded.round().as_f64_list() == [10.0, 22.0, 12.0, 32.0]
    assert round(rounded).as_i64_list() == [10, 22, 12, 32]
    assert vec.tobytes(order="A") == struct.pack("@qqq", 1, 2, 3)
    try:
        float(vec)
    except TypeError as exc:
        assert "only scalar arrays" in str(exc)
    else:
        raise AssertionError("float(vector) should raise TypeError")
    try:
        int(vec)
    except TypeError as exc:
        assert "only scalar arrays" in str(exc)
    else:
        raise AssertionError("int(vector) should raise TypeError")
    try:
        complex(vec)
    except TypeError as exc:
        assert "only scalar arrays" in str(exc)
    else:
        raise AssertionError("complex(vector) should raise TypeError")
    try:
        operator.index(vec)
    except TypeError as exc:
        assert "integer scalar arrays" in str(exc)
    else:
        raise AssertionError("operator.index(vector) should raise TypeError")
    try:
        bool(vec)
    except ValueError as exc:
        assert "more than one element" in str(exc)
    else:
        raise AssertionError("bool(vector) should raise ValueError")
    assert bool(fj.PyValue.vector_i64([0])) is False
    try:
        bool(fj.PyValue.vector_i64([]))
    except ValueError as exc:
        assert "empty array is ambiguous" in str(exc)
    else:
        raise AssertionError("bool(empty vector) should raise ValueError")
    assert vec.as_i64_list() == [1, 2, 3]
    assert vec.as_f64_list() == [1.0, 2.0, 3.0]
    print("✓ vector_i64 roundtrip")


def test_jit_add():
    """Test JIT compilation of add2."""
    jaxpr = fj.make_jaxpr_add2()
    result = fj.jit(jaxpr, [fj.PyValue.scalar_i64(3), fj.PyValue.scalar_i64(4)])
    assert len(result) == 1
    assert result[0].as_i64() == 7
    print("✓ jit(add2)(3, 4) = 7")


def test_make_jaxpr_generic():
    """Test generic ProgramSpec-backed make_jaxpr entrypoint."""
    square = fj.make_jaxpr("square")
    result = fj.jit(square, [fj.PyValue.scalar_f64(3.0)])
    assert len(result) == 1
    assert abs(result[0].as_f64() - 9.0) < 1e-12

    add2 = fj.make_jaxpr("add2")
    result = fj.jit(add2, [fj.PyValue.scalar_i64(3), fj.PyValue.scalar_i64(4)])
    assert len(result) == 1
    assert result[0].as_i64() == 7

    reshape = fj.make_jaxpr("lax_reshape_6_to_2x3")
    matrix = fj.jit(reshape, [fj.PyValue.vector_i64([1, 2, 3, 4, 5, 6])])[0]
    assert matrix.shape == (2, 3)
    assert matrix.real.tolist() == [1, 2, 3, 4, 5, 6]
    assert matrix.imag.tolist() == [0, 0, 0, 0, 0, 0]
    matrix_t = matrix.T
    assert matrix_t.shape == (3, 2)
    assert matrix_t.tolist() == [1, 4, 2, 5, 3, 6]
    matrix_call_t = matrix.transpose()
    assert matrix_call_t.shape == (3, 2)
    assert matrix_call_t.tolist() == [1, 4, 2, 5, 3, 6]
    assert matrix.transpose(1, 0).tolist() == [1, 4, 2, 5, 3, 6]
    assert matrix.transpose((1, 0)).tolist() == [1, 4, 2, 5, 3, 6]
    assert matrix.transpose([1, 0]).tolist() == [1, 4, 2, 5, 3, 6]
    assert matrix.transpose(-1, -2).tolist() == [1, 4, 2, 5, 3, 6]
    assert matrix.reshape(6).shape == (6,)
    assert matrix.reshape(6).tolist() == [1, 2, 3, 4, 5, 6]
    assert matrix.reshape((3, 2), order="F").shape == (3, 2)
    assert matrix.reshape((3, 2), order="F").tolist() == [1, 5, 4, 3, 2, 6]
    assert matrix.flatten().tolist() == [1, 2, 3, 4, 5, 6]
    assert matrix.flatten(order="F").tolist() == [1, 4, 2, 5, 3, 6]
    assert matrix.ravel(order="F").tolist() == [1, 4, 2, 5, 3, 6]
    try:
        matrix.reshape(4)
    except ValueError as exc:
        assert "cannot reshape" in str(exc)
    else:
        raise AssertionError("Array.reshape should reject incompatible shapes")
    try:
        matrix.reshape((3, 2), order="A")
    except NotImplementedError as exc:
        assert "order=A" in str(exc)
    else:
        raise AssertionError("Array.reshape should reject order=A")
    try:
        matrix.reshape((3, 2), order="K")
    except ValueError as exc:
        assert "order" in str(exc)
    else:
        raise AssertionError("Array.reshape should reject unknown order")
    try:
        matrix.transpose(0)
    except ValueError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("Array.transpose should reject too few axes")
    try:
        matrix.transpose(0, 0)
    except ValueError as exc:
        assert "repeated" in str(exc)
    else:
        raise AssertionError("Array.transpose should reject repeated axes")
    try:
        matrix.transpose(2, 0)
    except ValueError as exc:
        assert "out of bounds" in str(exc)
    else:
        raise AssertionError("Array.transpose should reject out-of-bounds axes")
    matrix_mt = matrix.mT
    assert matrix_mt.shape == (3, 2)
    assert matrix_mt.tolist() == [1, 4, 2, 5, 3, 6]

    complex_value = fj.jit(
        fj.make_jaxpr("lax_complex"),
        [fj.PyValue.scalar_f64(3.0), fj.PyValue.scalar_f64(-2.0)],
    )[0]
    assert abs(complex_value.real.as_f64() - 3.0) < 1e-12
    assert abs(complex_value.imag.as_f64() + 2.0) < 1e-12
    assert abs(complex_value.conj().real.as_f64() - 3.0) < 1e-12
    assert abs(complex_value.conj().imag.as_f64() - 2.0) < 1e-12
    assert abs(complex_value.conjugate().imag.as_f64() - 2.0) < 1e-12

    try:
        fj.make_jaxpr("missing_program")
    except ValueError as exc:
        assert "unknown ProgramSpec" in str(exc)
        assert "square" in str(exc)
    else:
        raise AssertionError("make_jaxpr should reject unknown ProgramSpec names")

    print("✓ make_jaxpr dispatches built-in ProgramSpec names")


def test_grad_square():
    """Test gradient of x^2."""
    jaxpr = fj.make_jaxpr_square()
    grads = fj.grad(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert len(grads) == 1
    # d/dx(x^2) = 2x, so at x=3, gradient = 6
    assert abs(grads[0].as_f64() - 6.0) < 1e-6
    print("✓ grad(square)(3.0) = 6.0")


def test_jvp_square():
    """Test forward-mode JVP of x^2."""
    jaxpr = fj.make_jaxpr_square()
    primals, tangents = fj.jvp(
        jaxpr,
        [fj.PyValue.scalar_f64(3.0)],
        [fj.PyValue.scalar_f64(1.0)],
    )
    assert len(primals) == 1
    assert len(tangents) == 1
    assert abs(primals[0].as_f64() - 9.0) < 1e-12
    assert abs(tangents[0].as_f64() - 6.0) < 1e-6
    print("✓ jvp(square)(3.0, 1.0) = (9.0, 6.0)")


def test_vjp_square():
    """Test reverse-mode VJP of x^2."""
    jaxpr = fj.make_jaxpr_square()
    values, pullback = fj.vjp(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert len(values) == 1
    assert abs(values[0].as_f64() - 9.0) < 1e-12

    grads = pullback.call([fj.PyValue.scalar_f64(1.0)])
    assert len(grads) == 1
    assert abs(grads[0].as_f64() - 6.0) < 1e-6
    print("✓ vjp(square)(3.0)(1.0) = (9.0, 6.0)")


def test_linear_transpose_square():
    """Test linear_transpose returns a callable cotangent pullback."""
    jaxpr = fj.make_jaxpr_square()
    transpose = fj.linear_transpose(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    grads = transpose([fj.PyValue.scalar_f64(1.0)])
    assert len(grads) == 1
    assert abs(grads[0].as_f64() - 6.0) < 1e-6
    print("✓ linear_transpose(square)(3.0)(1.0) = 6.0")


def test_linearize_square():
    """Test reusable linearized JVP of x^2."""
    jaxpr = fj.make_jaxpr_square()
    values, linearized = fj.linearize(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert len(values) == 1
    assert abs(values[0].as_f64() - 9.0) < 1e-12

    tangents = linearized.call([fj.PyValue.scalar_f64(1.0)])
    assert len(tangents) == 1
    assert abs(tangents[0].as_f64() - 6.0) < 1e-6

    scaled_tangents = linearized.call([fj.PyValue.scalar_f64(2.0)])
    assert len(scaled_tangents) == 1
    assert abs(scaled_tangents[0].as_f64() - 12.0) < 1e-6
    print("✓ linearize(square)(3.0) reuses pushforward for tangents 1.0 and 2.0")


def test_fwd_and_bwd_square():
    """Test reusable forward/backward pair of x^2."""
    jaxpr = fj.make_jaxpr_square()
    forward, backward = fj.fwd_and_bwd(jaxpr)
    values, residuals = forward([fj.PyValue.scalar_f64(3.0)])
    assert len(values) == 1
    assert abs(values[0].as_f64() - 9.0) < 1e-12

    grads = backward(residuals, [fj.PyValue.scalar_f64(1.0)])
    assert len(grads) == 1
    assert abs(grads[0].as_f64() - 6.0) < 1e-6

    scaled_grads = backward(residuals, [fj.PyValue.scalar_f64(2.0)])
    assert len(scaled_grads) == 1
    assert abs(scaled_grads[0].as_f64() - 12.0) < 1e-6
    print("✓ fwd_and_bwd(square) reuses backward pass for cotangents 1.0 and 2.0")


def test_eval_shape():
    """Test eval_shape metadata for scalar and vector outputs."""
    scalar_meta = fj.eval_shape(fj.make_jaxpr_square(), [fj.PyValue.scalar_f64(3.0)])
    assert len(scalar_meta) == 1
    assert isinstance(scalar_meta[0], fj.ShapeDtypeStruct)
    assert scalar_meta[0].shape == ()
    assert scalar_meta[0].dtype == "F64"

    vector_meta = fj.eval_shape(
        fj.make_jaxpr_add_one(),
        [fj.PyValue.vector_f64([1.0, 2.0, 3.0])],
    )
    assert len(vector_meta) == 1
    assert isinstance(vector_meta[0], fj.ShapeDtypeStruct)
    assert vector_meta[0].shape == (3,)
    assert vector_meta[0].dtype == "F64"
    print("✓ eval_shape returns shape/dtype metadata for scalar and vector outputs")


def test_shape_dtype_struct_constructor():
    """Test public ShapeDtypeStruct constructor metadata."""
    meta = fj.ShapeDtypeStruct([2, 3], "F64", sharding=None, vma=None)
    assert meta.shape == (2, 3)
    assert meta.dtype == "F64"
    assert meta.sharding is None
    assert meta.vma is None
    assert meta.ndim == 2
    assert meta.size == 6
    assert len(meta) == 2
    assert meta.weak_type is False
    assert meta.is_ref is False
    assert repr(meta) == "ShapeDtypeStruct(shape=(2, 3), dtype=F64)"
    assert str(meta) == repr(meta)
    weak_meta = fj.ShapeDtypeStruct([], "F64", weak_type=True, is_ref=True)
    assert weak_meta.shape == ()
    assert weak_meta.dtype == "F64"
    assert weak_meta.sharding is None
    assert weak_meta.vma is None
    assert weak_meta.ndim == 0
    assert weak_meta.size == 1
    try:
        len(weak_meta)
    except TypeError as exc:
        assert str(exc) == "len() of unsized object"
    else:
        raise AssertionError("len(scalar ShapeDtypeStruct) should raise TypeError")
    assert weak_meta.weak_type is True
    assert weak_meta.is_ref is True
    assert repr(weak_meta) == (
        "ShapeDtypeStruct(shape=(), dtype=F64, weak_type=True, is_ref=True)"
    )
    assert str(weak_meta) == repr(weak_meta)
    updated = meta.update(
        shape=[4],
        dtype="I64",
        sharding=None,
        weak_type=True,
        vma=None,
        is_ref=True,
    )
    assert updated.shape == (4,)
    assert updated.dtype == "I64"
    assert updated.sharding is None
    assert updated.vma is None
    assert updated.ndim == 1
    assert updated.size == 4
    assert len(updated) == 4
    assert updated.weak_type is True
    assert updated.is_ref is True
    assert repr(updated) == (
        "ShapeDtypeStruct(shape=(4,), dtype=I64, weak_type=True, is_ref=True)"
    )
    assert str(updated) == repr(updated)
    assert meta.shape == (2, 3)
    assert meta.dtype == "F64"
    assert meta.weak_type is False
    assert meta.is_ref is False
    same_meta = fj.ShapeDtypeStruct([2, 3], "F64")
    assert meta == same_meta
    assert hash(meta) == hash(same_meta)
    assert meta != weak_meta
    assert meta != updated

    vma_meta = fj.ShapeDtypeStruct([2], "F64", vma={"data"})
    assert vma_meta.vma == frozenset({"data"})
    assert "vma=frozenset(" in repr(vma_meta)
    same_vma_meta = fj.ShapeDtypeStruct([2], "F64", vma=frozenset({"data"}))
    assert vma_meta == same_vma_meta
    assert hash(vma_meta) == hash(same_vma_meta)
    preserved_vma_meta = vma_meta.update()
    assert preserved_vma_meta.vma == frozenset({"data"})
    updated_vma_meta = vma_meta.update(vma={"batch"})
    assert updated_vma_meta.vma == frozenset({"batch"})
    cleared_vma_meta = vma_meta.update(vma=None)
    assert cleared_vma_meta.vma is None
    try:
        fj.ShapeDtypeStruct([2], "F64", vma=["data"])
    except TypeError as exc:
        assert "set` or `frozenset`" in str(exc)
    else:
        raise AssertionError("ShapeDtypeStruct list vma should raise TypeError")
    try:
        fj.ShapeDtypeStruct([2], None)
    except ValueError as exc:
        assert str(exc) == "ShapeDtypeStruct: dtype must be specified."
    else:
        raise AssertionError("ShapeDtypeStruct dtype=None should raise ValueError")

    print("✓ ShapeDtypeStruct constructor preserves metadata")


def test_typeof():
    """Test typeof metadata for scalar and vector values."""
    scalar_meta = fj.typeof(fj.PyValue.scalar_i64(7))
    assert isinstance(scalar_meta, fj.ShapeDtypeStruct)
    assert scalar_meta.shape == ()
    assert scalar_meta.dtype == "I64"
    assert scalar_meta.ndim == 0
    assert scalar_meta.size == 1
    try:
        len(scalar_meta)
    except TypeError as exc:
        assert str(exc) == "len() of unsized object"
    else:
        raise AssertionError("len(scalar typeof metadata) should raise TypeError")
    assert scalar_meta.weak_type is False
    assert scalar_meta.is_ref is False

    vector_meta = fj.typeof(fj.PyValue.vector_f64([1.0, 2.0, 3.0]))
    assert isinstance(vector_meta, fj.ShapeDtypeStruct)
    assert vector_meta.shape == (3,)
    assert vector_meta.dtype == "F64"
    assert vector_meta.ndim == 1
    assert vector_meta.size == 3
    assert len(vector_meta) == 3
    assert vector_meta.weak_type is False
    assert vector_meta.is_ref is False
    print("✓ typeof returns ShapeDtypeStruct metadata for scalar and vector values")


def test_value_and_grad():
    """Test value_and_grad of x^2."""
    jaxpr = fj.make_jaxpr_square()
    values, grads = fj.value_and_grad(jaxpr, [fj.PyValue.scalar_f64(4.0)])
    assert abs(values[0].as_f64() - 16.0) < 1e-6
    assert abs(grads[0].as_f64() - 8.0) < 1e-6
    print("✓ value_and_grad(square)(4.0) = (16.0, 8.0)")


def test_device_helpers():
    """Test CPU-local host/device helper behavior."""
    scalar = fj.PyValue.scalar_f64(3.5)
    put_scalar = fj.device_put(scalar)
    assert abs(put_scalar.as_f64() - 3.5) < 1e-12
    replicated_scalar = fj.device_put_replicated(scalar, fj.local_devices())
    assert abs(replicated_scalar.as_f64() - 3.5) < 1e-12
    ready_scalar = fj.block_until_ready(put_scalar)
    assert abs(ready_scalar.as_f64() - 3.5) < 1e-12
    host_scalar = fj.device_get(ready_scalar)
    assert abs(host_scalar.as_f64() - 3.5) < 1e-12

    vector = fj.PyValue.vector_i64([1, 2, 3])
    sharded_vector = fj.device_put_sharded([vector], fj.local_devices())
    assert sharded_vector.as_i64_list() == [1, 2, 3]
    host_vector = fj.device_get(fj.block_until_ready(fj.device_put(vector)))
    assert host_vector.shape == (3,)
    assert host_vector.dtype() == "I64"
    assert host_vector.as_i64_list() == [1, 2, 3]

    copied_vector = fj.copy_to_host_async(host_vector)
    assert copied_vector.shape == (3,)
    assert copied_vector.dtype() == "I64"
    assert copied_vector.as_i64_list() == [1, 2, 3]
    assert fj.effects_barrier() is None
    assert fj.clear_caches() is None
    try:
        fj.device_put_replicated(scalar, [])
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("device_put_replicated should reject empty devices")

    try:
        fj.device_put_sharded([vector], [])
    except ValueError as exc:
        assert "len(shards)" in str(exc)
    else:
        raise AssertionError("device_put_sharded should reject length mismatch")

    print(
        "✓ device_put/device_get/block_until_ready/copy_to_host_async preserve CPU-local values"
    )


def test_backend_topology_helpers():
    """Test CPU-only backend and process topology helpers."""
    assert fj.default_backend() == "cpu"
    assert fj.device_count() == 1
    assert fj.device_count("cpu") == 1
    assert fj.local_device_count() == 1
    assert fj.process_index() == 0
    assert fj.process_count() == 1
    assert fj.process_indices() == [0]
    assert fj.host_id() == 0
    assert fj.host_count() == 1
    assert fj.host_ids() == [0]
    assert fj.host_id("cpu") == 0
    assert fj.host_count("cpu") == 1
    assert fj.host_ids("cpu") == [0]

    devices = fj.devices()
    assert len(devices) == 1
    device = devices[0]
    assert isinstance(device, fj.Device)
    assert device.id == 0
    assert device.process_index == 0
    assert device.platform == "cpu"
    assert device.device_kind == "cpu"
    local_devices = fj.local_devices()
    assert len(local_devices) == 1
    assert local_devices[0].id == 0
    assert fj.local_devices(0)[0].id == 0
    assert fj.local_devices(host_id=0)[0].id == 0
    assert fj.default_device().name == "default_device(None)"
    assert fj.default_device(None).name == "default_device(None)"
    assert fj.default_device("cpu").name == 'default_device("cpu")'
    assert (
        fj.default_device(device).name
        == "default_device(Device(id=0, process_index=0))"
    )

    with fj.default_device(device):
        assert fj.default_backend() == "cpu"

    try:
        fj.local_devices(1)
    except ValueError as exc:
        assert "process_index" in str(exc)
    else:
        raise AssertionError("local_devices should reject unknown process_index")

    try:
        fj.local_devices(host_id=1)
    except ValueError as exc:
        assert "process_index" in str(exc)
    else:
        raise AssertionError("local_devices should reject unknown host_id")

    try:
        fj.device_count("gpu")
    except ValueError as exc:
        assert "unsupported backend" in str(exc)
    else:
        raise AssertionError("device_count should reject unsupported backend")

    try:
        fj.default_device("gpu")
    except ValueError as exc:
        assert "unsupported backend" in str(exc)
    else:
        raise AssertionError("default_device should reject unsupported backend")

    try:
        fj.default_device(object())
    except ValueError as exc:
        assert "default_device expects" in str(exc)
    else:
        raise AssertionError("default_device should reject non-device objects")

    try:
        fj.host_count("gpu")
    except ValueError as exc:
        assert "unsupported backend" in str(exc)
    else:
        raise AssertionError("host_count should reject unsupported backend")

    print("✓ backend topology helpers expose one local CPU device")


def test_backend_cleanup_helpers():
    """Test CPU-local backend cleanup and live-array introspection helpers."""
    assert fj.clear_backends() is None
    assert fj.clean_up() is None
    assert fj.live_arrays() == []
    assert fj.live_arrays("cpu") == []

    try:
        fj.live_arrays("gpu")
    except ValueError as exc:
        assert "unsupported backend" in str(exc)
    else:
        raise AssertionError("live_arrays should reject unsupported backend")

    print("✓ clear_backends/clean_up/live_arrays expose CPU-local backend helpers")


def test_named_helpers():
    """Test no-op named_call and named_scope helper behavior."""

    def add_one(x):
        return x + 1

    wrapped = fj.named_call(add_one, name="add_one")
    assert wrapped is add_one
    assert wrapped(2) == 3

    with fj.named_scope("outer"):
        assert add_one(3) == 4

    decorated = fj.named_scope("decorated")(add_one)
    assert decorated is add_one
    assert decorated(4) == 5

    if fj.named_scope("").name != "":
        raise AssertionError("named_scope should preserve an empty name")
    if fj.named_call(add_one, name="") is not add_one:
        raise AssertionError("named_call should preserve callable for an empty name")

    print("✓ named_call/named_scope preserve Python callables")


def test_local_context_helpers():
    """Test local no-op context helpers."""

    def add_one(x):
        return x + 1

    with fj.disable_jit():
        assert add_one(2) == 3

    with fj.disable_jit(False):
        assert add_one(3) == 4

    with fj.enable_checks():
        assert add_one(4) == 5

    with fj.enable_checks(False):
        assert add_one(5) == 6

    with fj.debug_key_reuse():
        assert add_one(5) == 6

    with fj.debug_key_reuse(False):
        assert add_one(6) == 7

    with fj.enable_x64():
        assert add_one(5) == 6

    with fj.enable_x64(False):
        assert add_one(6) == 7

    with fj.enable_custom_prng():
        assert add_one(6) == 7

    with fj.enable_custom_prng(False):
        assert add_one(7) == 8

    with fj.softmax_custom_jvp():
        assert add_one(7) == 8

    with fj.softmax_custom_jvp(False):
        assert add_one(8) == 9

    with fj.enable_custom_vjp_by_custom_transpose():
        assert add_one(8) == 9

    with fj.enable_custom_vjp_by_custom_transpose(False):
        assert add_one(9) == 10

    with fj.check_tracer_leaks():
        assert add_one(6) == 7

    with fj.check_tracer_leaks(False):
        assert add_one(7) == 8

    with fj.checking_leaks():
        assert add_one(8) == 9

    with fj.debug_nans():
        assert add_one(9) == 10

    with fj.debug_nans(False):
        assert add_one(10) == 11

    with fj.debug_infs():
        assert add_one(11) == 12

    with fj.debug_infs(False):
        assert add_one(12) == 13

    with fj.log_compiles():
        assert add_one(13) == 14

    with fj.log_compiles(False):
        assert add_one(14) == 15

    with fj.explain_cache_misses():
        assert add_one(15) == 16

    with fj.explain_cache_misses(False):
        assert add_one(16) == 17

    with fj.no_tracing():
        assert add_one(16) == 17

    with fj.no_tracing(False):
        assert add_one(17) == 18

    with fj.no_execution():
        assert add_one(17) == 18

    with fj.no_execution(False):
        assert add_one(18) == 19

    with fj.default_matmul_precision():
        assert add_one(18) == 19

    with fj.default_matmul_precision("high"):
        assert add_one(19) == 20

    with fj.default_prng_impl("threefry2x32"):
        assert add_one(20) == 21

    with fj.numpy_dtype_promotion("strict"):
        assert add_one(21) == 22

    with fj.numpy_rank_promotion("warn"):
        assert add_one(22) == 23

    with fj.allow_f16_reductions():
        assert add_one(22) == 23

    with fj.allow_f16_reductions(False):
        assert add_one(23) == 24

    with fj.jax2tf_associative_scan_reductions():
        assert add_one(23) == 24

    with fj.jax2tf_associative_scan_reductions(False):
        assert add_one(24) == 25

    with fj.legacy_prng_key("warn"):
        assert add_one(24) == 25

    with fj.threefry_partitionable():
        assert add_one(25) == 26

    with fj.threefry_partitionable(False):
        assert add_one(26) == 27

    with fj.array_garbage_collection_guard():
        assert add_one(26) == 27

    with fj.array_garbage_collection_guard("fatal"):
        assert add_one(27) == 28

    with fj.remove_size_one_mesh_axis_from_type():
        assert add_one(27) == 28

    with fj.remove_size_one_mesh_axis_from_type(False):
        assert add_one(28) == 29

    with fj.thread_guard():
        assert add_one(28) == 29

    with fj.thread_guard(False):
        assert add_one(29) == 30

    user_context = fj.make_user_context()
    assert isinstance(user_context, fj.PyUserContext)
    assert user_context.value == "None"
    with user_context(7):
        assert add_one(30) == 31

    named_user_context = fj.make_user_context("seed")
    assert named_user_context.value == "'seed'"
    with named_user_context("active"):
        assert add_one(31) == 32

    with fj.transfer_guard("allow"):
        assert add_one(23) == 24

    with fj.transfer_guard_host_to_device("log"):
        assert add_one(24) == 25

    with fj.transfer_guard_device_to_device("disallow"):
        assert add_one(25) == 26

    with fj.transfer_guard_device_to_host("log_explicit"):
        assert add_one(26) == 27

    with fj.ensure_compile_time_eval():
        assert add_one(17) == 18

    assert fj.enable_checks().name == "enable_checks(true)"
    assert fj.enable_checks(False).name == "enable_checks(false)"
    assert fj.debug_key_reuse().name == "debug_key_reuse(true)"
    assert fj.debug_key_reuse(False).name == "debug_key_reuse(false)"
    assert fj.enable_x64().name == "enable_x64(true)"
    assert fj.enable_x64(False).name == "enable_x64(false)"
    assert fj.enable_custom_prng().name == "enable_custom_prng(true)"
    assert fj.enable_custom_prng(False).name == "enable_custom_prng(false)"
    assert fj.softmax_custom_jvp().name == "softmax_custom_jvp(true)"
    assert fj.softmax_custom_jvp(False).name == "softmax_custom_jvp(false)"
    assert (
        fj.enable_custom_vjp_by_custom_transpose().name
        == "enable_custom_vjp_by_custom_transpose(true)"
    )
    assert (
        fj.enable_custom_vjp_by_custom_transpose(False).name
        == "enable_custom_vjp_by_custom_transpose(false)"
    )
    assert fj.check_tracer_leaks().name == "check_tracer_leaks(true)"
    assert fj.check_tracer_leaks(False).name == "check_tracer_leaks(false)"
    assert fj.checking_leaks().name == "checking_leaks"
    assert fj.debug_nans().name == "debug_nans(true)"
    assert fj.debug_nans(False).name == "debug_nans(false)"
    assert fj.debug_infs().name == "debug_infs(true)"
    assert fj.debug_infs(False).name == "debug_infs(false)"
    assert fj.log_compiles().name == "log_compiles(true)"
    assert fj.log_compiles(False).name == "log_compiles(false)"
    assert fj.explain_cache_misses().name == "explain_cache_misses(true)"
    assert fj.explain_cache_misses(False).name == "explain_cache_misses(false)"
    assert fj.no_tracing().name == "no_tracing(true)"
    assert fj.no_tracing(False).name == "no_tracing(false)"
    assert fj.no_execution().name == "no_execution(true)"
    assert fj.no_execution(False).name == "no_execution(false)"
    assert fj.default_matmul_precision().name == "default_matmul_precision(None)"
    assert (
        fj.default_matmul_precision("TF32_TF32_F32").name
        == 'default_matmul_precision("TF32_TF32_F32")'
    )
    assert fj.default_prng_impl("rbg").name == 'default_prng_impl("rbg")'
    assert (
        fj.numpy_dtype_promotion("standard").name
        == 'numpy_dtype_promotion("standard")'
    )
    assert (
        fj.numpy_rank_promotion("raise").name
        == 'numpy_rank_promotion("raise")'
    )
    assert fj.allow_f16_reductions().name == "allow_f16_reductions(true)"
    assert fj.allow_f16_reductions(False).name == "allow_f16_reductions(false)"
    if (
        fj.jax2tf_associative_scan_reductions().name
        != "jax2tf_associative_scan_reductions(true)"
    ):
        raise AssertionError("jax2tf_associative_scan_reductions default name")
    if (
        fj.jax2tf_associative_scan_reductions(False).name
        != "jax2tf_associative_scan_reductions(false)"
    ):
        raise AssertionError("jax2tf_associative_scan_reductions false name")
    assert fj.legacy_prng_key("allow").name == 'legacy_prng_key("allow")'
    assert fj.threefry_partitionable().name == "threefry_partitionable(true)"
    assert fj.threefry_partitionable(False).name == "threefry_partitionable(false)"
    assert (
        fj.array_garbage_collection_guard().name
        == "array_garbage_collection_guard(None)"
    )
    assert (
        fj.array_garbage_collection_guard("log").name
        == 'array_garbage_collection_guard("log")'
    )
    assert (
        fj.remove_size_one_mesh_axis_from_type().name
        == "remove_size_one_mesh_axis_from_type(true)"
    )
    assert (
        fj.remove_size_one_mesh_axis_from_type(False).name
        == "remove_size_one_mesh_axis_from_type(false)"
    )
    assert fj.thread_guard().name == "thread_guard(true)"
    assert fj.thread_guard(False).name == "thread_guard(false)"
    assert user_context(5).name == "user_context(5)"
    assert named_user_context(None).name == "user_context(None)"
    assert fj.transfer_guard("allow").name == 'transfer_guard("allow")'
    assert (
        fj.transfer_guard_host_to_device("log").name
        == 'transfer_guard_host_to_device("log")'
    )
    assert (
        fj.transfer_guard_device_to_device("disallow").name
        == 'transfer_guard_device_to_device("disallow")'
    )
    assert (
        fj.transfer_guard_device_to_host("disallow_explicit").name
        == 'transfer_guard_device_to_host("disallow_explicit")'
    )
    assert fj.disable_jit().name == "disable_jit(true)"
    assert fj.disable_jit(False).name == "disable_jit(false)"
    assert fj.ensure_compile_time_eval().name == "ensure_compile_time_eval"
    assert fj.enable_checks()(add_one) is add_one
    assert fj.debug_key_reuse()(add_one) is add_one
    assert fj.enable_x64()(add_one) is add_one
    assert fj.enable_custom_prng()(add_one) is add_one
    assert fj.softmax_custom_jvp()(add_one) is add_one
    assert fj.enable_custom_vjp_by_custom_transpose()(add_one) is add_one
    assert fj.check_tracer_leaks()(add_one) is add_one
    assert fj.checking_leaks()(add_one) is add_one
    assert fj.debug_nans()(add_one) is add_one
    assert fj.debug_infs()(add_one) is add_one
    assert fj.log_compiles()(add_one) is add_one
    assert fj.explain_cache_misses()(add_one) is add_one
    assert fj.no_tracing()(add_one) is add_one
    assert fj.no_execution()(add_one) is add_one
    assert fj.default_matmul_precision("highest")(add_one) is add_one
    assert fj.default_prng_impl("unsafe_rbg")(add_one) is add_one
    assert fj.numpy_dtype_promotion("strict")(add_one) is add_one
    assert fj.numpy_rank_promotion("allow")(add_one) is add_one
    assert fj.allow_f16_reductions()(add_one) is add_one
    if fj.jax2tf_associative_scan_reductions()(add_one) is not add_one:
        raise AssertionError("jax2tf_associative_scan_reductions decorator")
    assert fj.legacy_prng_key("error")(add_one) is add_one
    assert fj.threefry_partitionable()(add_one) is add_one
    assert fj.array_garbage_collection_guard("allow")(add_one) is add_one
    assert fj.remove_size_one_mesh_axis_from_type()(add_one) is add_one
    assert fj.thread_guard()(add_one) is add_one
    assert user_context(1)(add_one) is add_one
    assert fj.transfer_guard("log_explicit")(add_one) is add_one
    assert fj.transfer_guard_host_to_device("allow")(add_one) is add_one
    assert fj.transfer_guard_device_to_device("allow")(add_one) is add_one
    assert fj.transfer_guard_device_to_host("allow")(add_one) is add_one
    assert fj.disable_jit()(add_one) is add_one
    assert fj.ensure_compile_time_eval()(add_one) is add_one

    invalid_enum_calls = [
        (fj.default_matmul_precision, "ultra"),
        (fj.default_prng_impl, "mt19937"),
        (fj.numpy_dtype_promotion, "loose"),
        (fj.numpy_rank_promotion, "error"),
        (fj.legacy_prng_key, "panic"),
        (fj.array_garbage_collection_guard, "warn"),
        (fj.transfer_guard, "block"),
        (fj.transfer_guard_host_to_device, "block"),
        (fj.transfer_guard_device_to_device, "block"),
        (fj.transfer_guard_device_to_host, "block"),
    ]
    for function, value in invalid_enum_calls:
        try:
            function(value)
        except ValueError as exc:
            assert "unsupported" in str(exc)
        else:
            raise AssertionError(f"{function.__name__} should reject {value!r}")

    print("✓ config-style local context helpers preserve Python callables")


def test_vmap():
    """Test vmap of add_one."""
    jaxpr = fj.make_jaxpr_add_one()
    batch = fj.PyValue.vector_f64([1.0, 2.0, 3.0])
    result = fj.vmap(jaxpr, [batch])
    assert len(result) == 1
    assert result[0].shape == (3,)
    assert result[0].as_f64_list() == [2.0, 3.0, 4.0]
    print("✓ vmap(add_one)([1,2,3]) = [2,3,4]")


def test_pmap_fails_closed():
    """Test pmap reports the V1 multi-device unsupported contract."""
    jaxpr = fj.make_jaxpr_add_one()
    batch = fj.PyValue.vector_f64([1.0, 2.0])
    try:
        fj.pmap(jaxpr, [batch])
    except RuntimeError as exc:
        message = str(exc).lower()
        assert "pmap unavailable" in message
        assert "multi-device" in message
    else:
        raise AssertionError("pmap should fail closed without multi-device context")
    print("✓ pmap fails closed without multi-device context")


def test_jacobian_and_hessian():
    """Test Jacobian and Hessian transform wrappers."""
    jaxpr = fj.make_jaxpr_square()

    jac = fj.jacobian(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert jac.shape == (1, 1)
    assert abs(jac.as_f64_list()[0] - 6.0) < 1e-6

    jac_rev = fj.jacrev(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert jac_rev.shape == (1, 1)
    assert abs(jac_rev.as_f64_list()[0] - 6.0) < 1e-6

    jac_fwd = fj.jacfwd(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert jac_fwd.shape == (1, 1)
    assert abs(jac_fwd.as_f64_list()[0] - 6.0) < 1e-6

    hess = fj.hessian(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert hess.shape == (1, 1)
    assert abs(hess.as_f64_list()[0] - 2.0) < 1e-6
    print("✓ jacobian/jacrev/jacfwd/hessian(square)(3.0) = (6.0, 6.0, 6.0, 2.0)")


def test_checkpoint():
    """Test checkpoint wrapper."""
    jaxpr = fj.make_jaxpr_square()
    checkpointed = fj.checkpoint(jaxpr)
    assert checkpointed.memory_savings_entries() > 0

    values = checkpointed.call([fj.PyValue.scalar_f64(3.0)])
    assert abs(values[0].as_f64() - 9.0) < 1e-12

    grads = checkpointed.grad([fj.PyValue.scalar_f64(3.0)])
    assert abs(grads[0].as_f64() - 6.0) < 1e-6

    values_again, grads_again = checkpointed.value_and_grad([fj.PyValue.scalar_f64(4.0)])
    assert abs(values_again[0].as_f64() - 16.0) < 1e-12
    assert abs(grads_again[0].as_f64() - 8.0) < 1e-6
    print("✓ checkpoint(square) exposes call/grad/value_and_grad")


def test_remat_alias():
    """Test remat alias for checkpoint wrapper."""
    jaxpr = fj.make_jaxpr_square()
    rematted = fj.remat(jaxpr)
    assert rematted.memory_savings_entries() > 0

    values = rematted.call([fj.PyValue.scalar_f64(3.0)])
    assert abs(values[0].as_f64() - 9.0) < 1e-12

    grads = rematted.grad([fj.PyValue.scalar_f64(3.0)])
    assert abs(grads[0].as_f64() - 6.0) < 1e-6
    print("✓ remat(square) aliases checkpoint call/grad behavior")


if __name__ == "__main__":
    test_version_metadata()
    test_float0_sentinel()
    test_environment_info()
    test_value_scalar()
    test_jit_add()
    test_make_jaxpr_generic()
    test_grad_square()
    test_jvp_square()
    test_vjp_square()
    test_linear_transpose_square()
    test_linearize_square()
    test_fwd_and_bwd_square()
    test_eval_shape()
    test_shape_dtype_struct_constructor()
    test_typeof()
    test_value_and_grad()
    test_device_helpers()
    test_backend_topology_helpers()
    test_backend_cleanup_helpers()
    test_named_helpers()
    test_local_context_helpers()
    test_vmap()
    test_pmap_fails_closed()
    test_jacobian_and_hessian()
    test_checkpoint()
    test_remat_alias()
    print("\n✅ All smoke tests passed!")
