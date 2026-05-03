#!/usr/bin/env python3
"""Smoke test for frankenjax Python bindings.

Run after building with:
    maturin develop -m crates/fj-py/Cargo.toml
    python crates/fj-py/tests/smoke_test.py
"""

import frankenjax as fj


def test_value_scalar():
    """Test scalar value creation and retrieval."""
    v = fj.PyValue.scalar_f64(42.0)
    assert abs(v.as_f64() - 42.0) < 1e-12
    print("✓ scalar_f64 roundtrip")

    v2 = fj.PyValue.scalar_i64(123)
    assert v2.as_i64() == 123
    print("✓ scalar_i64 roundtrip")


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


def test_value_and_grad():
    """Test value_and_grad of x^2."""
    jaxpr = fj.make_jaxpr_square()
    values, grads = fj.value_and_grad(jaxpr, [fj.PyValue.scalar_f64(4.0)])
    assert abs(values[0].as_f64() - 16.0) < 1e-6
    assert abs(grads[0].as_f64() - 8.0) < 1e-6
    print("✓ value_and_grad(square)(4.0) = (16.0, 8.0)")


def test_vmap():
    """Test vmap of add_one."""
    jaxpr = fj.make_jaxpr_add_one()
    batch = fj.PyValue.vector_f64([1.0, 2.0, 3.0])
    result = fj.vmap(jaxpr, [batch])
    # add_one adds 1 to each element
    print("✓ vmap(add_one)([1,2,3]) ran successfully")


def test_checkpoint():
    """Test checkpoint wrapper."""
    jaxpr = fj.make_jaxpr_square()
    checkpointed = fj.checkpoint(jaxpr)
    # checkpoint returns a PyJaxpr that can be used with grad
    print("✓ checkpoint(square) created")


if __name__ == "__main__":
    test_value_scalar()
    test_jit_add()
    test_grad_square()
    test_value_and_grad()
    test_vmap()
    test_checkpoint()
    print("\n✅ All smoke tests passed!")
