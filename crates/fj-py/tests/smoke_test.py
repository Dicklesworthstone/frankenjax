#!/usr/bin/env python3
"""Smoke test for frankenjax Python bindings.

Run after building with:
    maturin develop -m crates/fj-py/Cargo.toml
    python crates/fj-py/tests/smoke_test.py
"""

import frankenjax as fj


def test_version_metadata():
    """Test package version metadata exports."""
    assert fj.__version__ == "0.1.0"
    assert fj.__version_info__ == (0, 1, 0)
    print("✓ __version__/__version_info__ metadata")


def test_value_scalar():
    """Test scalar value creation and retrieval."""
    v = fj.PyValue.scalar_f64(42.0)
    assert abs(v.as_f64() - 42.0) < 1e-12
    print("✓ scalar_f64 roundtrip")

    v2 = fj.PyValue.scalar_i64(123)
    assert v2.as_i64() == 123
    print("✓ scalar_i64 roundtrip")

    vec = fj.PyValue.vector_i64([1, 2, 3])
    assert vec.shape() == [3]
    assert vec.dtype() == "I64"
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


def test_eval_shape():
    """Test eval_shape metadata for scalar and vector outputs."""
    scalar_meta = fj.eval_shape(fj.make_jaxpr_square(), [fj.PyValue.scalar_f64(3.0)])
    assert len(scalar_meta) == 1
    assert scalar_meta[0].shape() == []
    assert scalar_meta[0].dtype() == "F64"

    vector_meta = fj.eval_shape(
        fj.make_jaxpr_add_one(),
        [fj.PyValue.vector_f64([1.0, 2.0, 3.0])],
    )
    assert len(vector_meta) == 1
    assert vector_meta[0].shape() == [3]
    assert vector_meta[0].dtype() == "F64"
    print("✓ eval_shape returns shape/dtype metadata for scalar and vector outputs")


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
    ready_scalar = fj.block_until_ready(put_scalar)
    assert abs(ready_scalar.as_f64() - 3.5) < 1e-12
    host_scalar = fj.device_get(ready_scalar)
    assert abs(host_scalar.as_f64() - 3.5) < 1e-12

    vector = fj.PyValue.vector_i64([1, 2, 3])
    host_vector = fj.device_get(fj.block_until_ready(fj.device_put(vector)))
    assert host_vector.shape() == [3]
    assert host_vector.dtype() == "I64"
    assert host_vector.as_i64_list() == [1, 2, 3]

    copied_vector = fj.copy_to_host_async(host_vector)
    assert copied_vector.shape() == [3]
    assert copied_vector.dtype() == "I64"
    assert copied_vector.as_i64_list() == [1, 2, 3]
    assert fj.effects_barrier() is None
    assert fj.clear_caches() is None
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

    devices = fj.devices()
    assert len(devices) == 1
    device = devices[0]
    assert isinstance(device, fj.Device)
    assert device.id == 0
    assert device.process_index == 0
    assert device.platform == "cpu"
    local_devices = fj.local_devices()
    assert len(local_devices) == 1
    assert local_devices[0].id == 0
    assert fj.local_devices(0)[0].id == 0

    try:
        fj.local_devices(1)
    except ValueError as exc:
        assert "process_index" in str(exc)
    else:
        raise AssertionError("local_devices should reject unknown process_index")

    try:
        fj.device_count("gpu")
    except ValueError as exc:
        assert "unsupported backend" in str(exc)
    else:
        raise AssertionError("device_count should reject unsupported backend")

    print("✓ backend topology helpers expose one local CPU device")


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

    with fj.ensure_compile_time_eval():
        assert add_one(4) == 5

    assert fj.disable_jit().name == "disable_jit(true)"
    assert fj.disable_jit(False).name == "disable_jit(false)"
    assert fj.ensure_compile_time_eval().name == "ensure_compile_time_eval"
    assert fj.disable_jit()(add_one) is add_one
    assert fj.ensure_compile_time_eval()(add_one) is add_one
    print("✓ disable_jit/ensure_compile_time_eval expose local no-op contexts")


def test_vmap():
    """Test vmap of add_one."""
    jaxpr = fj.make_jaxpr_add_one()
    batch = fj.PyValue.vector_f64([1.0, 2.0, 3.0])
    result = fj.vmap(jaxpr, [batch])
    assert len(result) == 1
    assert result[0].shape() == [3]
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
    assert jac.shape() == [1, 1]
    assert abs(jac.as_f64_list()[0] - 6.0) < 1e-6

    jac_rev = fj.jacrev(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert jac_rev.shape() == [1, 1]
    assert abs(jac_rev.as_f64_list()[0] - 6.0) < 1e-6

    jac_fwd = fj.jacfwd(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert jac_fwd.shape() == [1, 1]
    assert abs(jac_fwd.as_f64_list()[0] - 6.0) < 1e-6

    hess = fj.hessian(jaxpr, [fj.PyValue.scalar_f64(3.0)])
    assert hess.shape() == [1, 1]
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
    test_value_scalar()
    test_jit_add()
    test_grad_square()
    test_jvp_square()
    test_vjp_square()
    test_linearize_square()
    test_eval_shape()
    test_value_and_grad()
    test_device_helpers()
    test_backend_topology_helpers()
    test_named_helpers()
    test_local_context_helpers()
    test_vmap()
    test_pmap_fails_closed()
    test_jacobian_and_hessian()
    test_checkpoint()
    test_remat_alias()
    print("\n✅ All smoke tests passed!")
