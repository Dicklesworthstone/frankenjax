//! CPU backend implementation for FrankenJAX.
//!
//! Provides a dependency-wave CPU scheduler over Jaxpr equations with a
//! parallel fast path for independent operations.
//! CPU is the baseline backend — always available, no external dependencies.
//!
//! Legacy anchor: P2C006-A17 (CpuBackend), P2C006-A02 (_discover_backends).

#![forbid(unsafe_code)]

mod executor;

pub use executor::CpuBackend;
