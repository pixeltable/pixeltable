//! Randomness for `Query.sample()` and for ids that must be unique (relative path roots).

use std::sync::atomic::{AtomicU64, Ordering};

/// splitmix64: a fast, well-distributed 64-bit mixer (also used as a seeded PRNG step).
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A random u64 (time- and counter-seeded; not cryptographic).
pub fn random_u64() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    splitmix64(t ^ splitmix64(c ^ (std::process::id() as u64) << 32))
}

/// The sampling hash of the row with key `(rowid, pos)` for `seed` (rows are ranked by it; a fractional sample keeps
/// the rows whose hash is below `fraction_threshold()`).
pub fn key_hash(seed: u64, rowid: i64, pos: &[i64]) -> u64 {
    let mut h = splitmix64(rowid as u64);
    for p in pos {
        h = splitmix64(h ^ *p as u64);
    }
    splitmix64(seed ^ h)
}

/// The hash below which a fractional sample keeps a row.
pub fn fraction_threshold(f: f64) -> u64 {
    if f >= 1.0 { u64::MAX } else { (f * u64::MAX as f64) as u64 }
}

/// True if a fractional sample with `fraction` keeps a row with hash `h`.
pub fn fraction_keeps(h: u64, fraction: f64) -> bool {
    fraction >= 1.0 || h < fraction_threshold(fraction)
}

/// A small seeded PRNG.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(splitmix64(seed))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        splitmix64(self.0)
    }
    /// uniform in [0, n)
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }
}
