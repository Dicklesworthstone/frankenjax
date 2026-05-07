//! Buffer types for device-side memory management.
//!
//! V1 scope: host-resident buffers backed by `Vec<u8>`. Buffers represent
//! owned, contiguous memory regions. The CPU backend allocates buffers
//! via the Rust global allocator.
//!
//! Legacy anchor: P2C006-A10 (Buffer: device-resident array),
//! P2C006-A23 (transfer_to_device).

use crate::device::DeviceId;

/// A contiguous memory region on a specific device.
///
/// V1: always host-resident (CPU backend). The `device` field tracks
/// ownership for future multi-device support.
///
/// Invariant: `data.len() == size` at all times.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    /// Raw buffer data.
    data: Vec<u8>,
    /// Device that owns this buffer.
    device: DeviceId,
}

impl Buffer {
    /// Create a new buffer with the given data on the specified device.
    #[must_use]
    pub fn new(data: Vec<u8>, device: DeviceId) -> Self {
        Self { data, device }
    }

    /// Create a zero-initialized buffer of the given size.
    #[must_use]
    pub fn zeroed(size: usize, device: DeviceId) -> Self {
        Self {
            data: vec![0u8; size],
            device,
        }
    }

    /// The device that owns this buffer.
    #[must_use]
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Buffer size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Read-only view of the buffer data.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Mutable view of the buffer data.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Consume the buffer and return the underlying data.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

/// Read-only view into a sub-region of a Buffer.
///
/// V1: simple slice reference. Future: could support zero-copy views
/// with reference counting for shared device memory.
#[derive(Debug, Clone, Copy)]
pub struct BufferView<'a> {
    data: &'a [u8],
    device: DeviceId,
}

impl<'a> BufferView<'a> {
    /// Create a view of the entire buffer.
    #[must_use]
    pub fn from_buffer(buffer: &'a Buffer) -> Self {
        Self {
            data: buffer.as_bytes(),
            device: buffer.device(),
        }
    }

    /// Create a view of a sub-region.
    ///
    /// Returns None if the range is out of bounds.
    #[must_use]
    pub fn slice(&self, offset: usize, len: usize) -> Option<Self> {
        let end = offset.checked_add(len)?;
        if end > self.data.len() {
            return None;
        }
        Some(Self {
            data: &self.data[offset..end],
            device: self.device,
        })
    }

    /// The device that owns the underlying buffer.
    #[must_use]
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// View size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Read-only access to the viewed data.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_new_and_accessors() {
        let buf = Buffer::new(vec![1, 2, 3, 4], DeviceId(0));
        assert_eq!(buf.device(), DeviceId(0));
        assert_eq!(buf.size(), 4);
        assert_eq!(buf.as_bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn buffer_zeroed() {
        let buf = Buffer::zeroed(8, DeviceId(0));
        assert_eq!(buf.size(), 8);
        assert!(buf.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn buffer_into_bytes() {
        let buf = Buffer::new(vec![10, 20], DeviceId(0));
        let data = buf.into_bytes();
        assert_eq!(data, vec![10, 20]);
    }

    #[test]
    fn buffer_view_from_buffer() {
        let buf = Buffer::new(vec![1, 2, 3, 4, 5], DeviceId(0));
        let view = BufferView::from_buffer(&buf);
        assert_eq!(view.size(), 5);
        assert_eq!(view.device(), DeviceId(0));
        assert_eq!(view.as_bytes(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn buffer_view_slice() {
        let buf = Buffer::new(vec![10, 20, 30, 40, 50], DeviceId(0));
        let view = BufferView::from_buffer(&buf);

        let sub = view.slice(1, 3).expect("valid slice");
        assert_eq!(sub.as_bytes(), &[20, 30, 40]);
        assert_eq!(sub.size(), 3);

        // Out of bounds
        assert!(view.slice(3, 5).is_none());
    }

    #[test]
    fn buffer_roundtrip_preserves_data() {
        // Contract p2c006.strict.inv003: device_put/device_get round-trip
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let buf = Buffer::new(original.clone(), DeviceId(0));
        let recovered = buf.into_bytes();
        assert_eq!(original, recovered);
    }

    #[test]
    fn buffer_empty() {
        let buf = Buffer::new(vec![], DeviceId(0));
        assert_eq!(buf.size(), 0);
        assert!(buf.as_bytes().is_empty());
    }

    #[test]
    fn buffer_zeroed_empty() {
        let buf = Buffer::zeroed(0, DeviceId(0));
        assert_eq!(buf.size(), 0);
    }

    #[test]
    fn buffer_mutable_write() {
        let mut buf = Buffer::zeroed(4, DeviceId(0));
        buf.as_bytes_mut().copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(buf.as_bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn buffer_clone_is_independent() {
        let buf = Buffer::new(vec![1, 2, 3], DeviceId(0));
        let mut cloned = buf.clone();
        cloned.as_bytes_mut()[0] = 99;
        assert_eq!(buf.as_bytes()[0], 1, "original should be unchanged");
        assert_eq!(cloned.as_bytes()[0], 99);
    }

    #[test]
    fn buffer_equality() {
        let a = Buffer::new(vec![1, 2, 3], DeviceId(0));
        let b = Buffer::new(vec![1, 2, 3], DeviceId(0));
        let c = Buffer::new(vec![1, 2, 3], DeviceId(1));
        assert_eq!(a, b);
        assert_ne!(a, c, "different device should be unequal");
    }

    #[test]
    fn buffer_view_empty_buffer() {
        let buf = Buffer::new(vec![], DeviceId(0));
        let view = BufferView::from_buffer(&buf);
        assert_eq!(view.size(), 0);
        assert!(view.slice(0, 1).is_none());
    }

    #[test]
    fn buffer_view_full_slice() {
        let buf = Buffer::new(vec![1, 2, 3], DeviceId(0));
        let view = BufferView::from_buffer(&buf);
        let full = view.slice(0, 3).expect("full slice should work");
        assert_eq!(full.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn buffer_view_boundary_slice() {
        let buf = Buffer::new(vec![10, 20, 30], DeviceId(0));
        let view = BufferView::from_buffer(&buf);
        // Exact boundary: offset + len == size
        let last = view.slice(2, 1).expect("boundary slice");
        assert_eq!(last.as_bytes(), &[30]);
        // Just past boundary
        assert!(view.slice(2, 2).is_none());
    }

    #[test]
    fn buffer_view_slice_overflow_returns_none() {
        let buf = Buffer::new(vec![10, 20, 30], DeviceId(0));
        let view = BufferView::from_buffer(&buf);

        assert!(view.slice(usize::MAX, 1).is_none());
        assert!(view.slice(1, usize::MAX).is_none());
    }

    #[test]
    fn buffer_view_nested_slice() {
        let buf = Buffer::new(vec![1, 2, 3, 4, 5, 6], DeviceId(0));
        let view = BufferView::from_buffer(&buf);
        let sub = view.slice(1, 4).expect("first slice");
        let nested = sub.slice(1, 2).expect("nested slice");
        assert_eq!(nested.as_bytes(), &[3, 4]);
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(
            fj_test_utils::property_test_case_count()
        ))]

        #[test]
        fn metamorphic_buffer_roundtrip(data in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..256)) {
            let buf = Buffer::new(data.clone(), DeviceId(0));
            proptest::prop_assert_eq!(buf.as_bytes(), &data[..]);
            proptest::prop_assert_eq!(buf.into_bytes(), data);
        }

        #[test]
        fn metamorphic_buffer_zeroed_all_zeros(size in 0_usize..512) {
            let buf = Buffer::zeroed(size, DeviceId(0));
            proptest::prop_assert_eq!(buf.size(), size);
            proptest::prop_assert!(buf.as_bytes().iter().all(|&b| b == 0));
        }

        #[test]
        fn metamorphic_slice_preserves_content(
            data in proptest::collection::vec(proptest::prelude::any::<u8>(), 1..64),
            offset in 0_usize..32,
            len in 0_usize..32
        ) {
            let buf = Buffer::new(data.clone(), DeviceId(0));
            let view = BufferView::from_buffer(&buf);
            if let Some(end) = offset.checked_add(len) {
                if end <= data.len() {
                    let slice = view.slice(offset, len).expect("valid slice");
                    proptest::prop_assert_eq!(slice.as_bytes(), &data[offset..end]);
                } else {
                    proptest::prop_assert!(view.slice(offset, len).is_none());
                }
            } else {
                proptest::prop_assert!(view.slice(offset, len).is_none());
            }
        }

        #[test]
        fn metamorphic_slice_overflow_safety(offset in 0_usize..=usize::MAX, len in 0_usize..=usize::MAX) {
            let buf = Buffer::new(vec![1, 2, 3, 4], DeviceId(0));
            let view = BufferView::from_buffer(&buf);
            let result = view.slice(offset, len);
            if let Some(end) = offset.checked_add(len) {
                if end <= 4 {
                    proptest::prop_assert!(result.is_some());
                } else {
                    proptest::prop_assert!(result.is_none());
                }
            } else {
                proptest::prop_assert!(result.is_none());
            }
        }
    }
}
