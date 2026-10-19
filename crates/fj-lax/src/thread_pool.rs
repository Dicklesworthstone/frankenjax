//! A process-lifetime persistent worker pool, so hot threaded ops don't pay the
//! per-call OS-thread spawn cost (`std::thread::scope` spawns each worker
//! sequentially — ~tens of µs apiece, ~ms for all cores).
//!
//! Safe Rust, no `unsafe`, no external deps. The borrow wall that forces `rayon`
//! to use `unsafe` (handing persistent workers a `&mut` slice of a caller-local
//! output) is sidestepped: each worker computes its chunk into an OWNED `Vec`,
//! ships it back over a channel, and the caller concatenates the chunks in order.
//! That concatenation is one extra `O(n)` copy — worth it only when the spawn it
//! replaces costs more than the copy (compute-bound, or many small calls).
//!
//! Distribution is round-robin over one MPSC channel per worker (no shared
//! `Mutex<Receiver>`, which would serialize `recv` and deadlock-starve workers).

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Pool {
    txs: Vec<Sender<Job>>,
    next: AtomicUsize,
}

impl Pool {
    fn submit(&self, job: Job) {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.txs.len();
        // A worker only disconnects if its thread panicked; run inline as a
        // last resort so the caller never hangs waiting for a lost result.
        if let Err(returned) = self.txs[i].send(job) {
            (returned.0)();
        }
    }
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let mut txs = Vec::with_capacity(n);
        for _ in 0..n {
            let (tx, rx) = channel::<Job>();
            if std::thread::Builder::new()
                .name("fj-lax-pool".into())
                .spawn(move || {
                    // Exits when the matching Sender is dropped (process teardown).
                    while let Ok(job) = rx.recv() {
                        job();
                    }
                })
                .is_ok()
            {
                txs.push(tx);
            }
        }
        Pool {
            txs,
            next: AtomicUsize::new(0),
        }
    })
}

/// Number of worker threads in the pool (= available parallelism).
#[must_use]
pub(crate) fn pool_size() -> usize {
    pool().txs.len()
}

/// Produce a length-`len` `Vec<f64>` by filling `threads` contiguous chunks on the
/// persistent pool, then concatenating them in order. `fill(chunk_start, &mut
/// chunk)` writes `chunk` (the output slice `[chunk_start, chunk_start+chunk.len())`);
/// it is `Fn + Send + Sync + 'static` and shared across chunks via `Arc`.
///
/// Each element is produced by exactly one worker at its global index, and chunks
/// are reassembled at their offsets, so the result is bit-for-bit identical to a
/// serial `fill(0, &mut whole)` regardless of `threads`. Falls back to an inline
/// fill for `threads <= 1`.
pub(crate) fn parallel_fill_f64<F>(len: usize, threads: usize, fill: F) -> Vec<f64>
where
    F: Fn(usize, &mut [f64]) + Send + Sync + 'static,
{
    let p = pool();
    if threads <= 1 || len == 0 || p.txs.is_empty() {
        let mut out = vec![0.0f64; len];
        fill(0, &mut out);
        return out;
    }
    let fill = Arc::new(fill);
    let chunk = len.div_ceil(threads);
    let (tx, rx) = channel::<(usize, Vec<f64>)>();
    let mut njobs = 0usize;
    let mut start = 0usize;
    while start < len {
        let end = (start + chunk).min(len);
        let f = Arc::clone(&fill);
        let txc = tx.clone();
        p.submit(Box::new(move || {
            let mut buf = vec![0.0f64; end - start];
            f(start, &mut buf);
            // Receiver lives until all chunks are collected; ignore send error
            // only if the caller already tore down (cannot happen here).
            let _ = txc.send((start, buf));
        }));
        start = end;
        njobs += 1;
    }
    drop(tx);

    let mut out = vec![0.0f64; len];
    for _ in 0..njobs {
        if let Ok((s, buf)) = rx.recv() {
            out[s..s + buf.len()].copy_from_slice(&buf);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_fill_matches_serial_across_thread_counts() {
        let len = 100_003usize; // prime-ish: exercises uneven last chunk
        let serial: Vec<f64> = (0..len).map(|i| (i as f64 * 1.5).sin()).collect();
        for threads in [1usize, 2, 3, 7, 8, 64] {
            let got = parallel_fill_f64(len, threads, |start, chunk| {
                for (k, o) in chunk.iter_mut().enumerate() {
                    *o = ((start + k) as f64 * 1.5).sin();
                }
            });
            assert_eq!(got.len(), serial.len());
            for (a, b) in got.iter().zip(serial.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "threads={threads}");
            }
        }
    }

    #[test]
    fn pool_size_is_positive() {
        assert!(pool_size() >= 1);
    }
}
