use std::{mem::size_of, ptr::null_mut};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct AssemblerLabel {
    pub offset: u32,
}
impl AssemblerLabel {
    pub fn set_offset(&mut self, offset: u32) {
        self.offset = offset;
    }

    pub const fn offset(&self) -> u32 {
        self.offset
    }

    pub const fn is_set(&self) -> bool {
        self.offset() != u32::MAX
    }

    pub const fn label_at_offset(&self, offset: u32) -> AssemblerLabel {
        AssemblerLabel {
            offset: offset + self.offset,
        }
    }

    pub const fn new(offset: u32) -> Self {
        Self { offset }
    }
}

pub struct AssemblerData {
    inline_buffer: [u8; 128],
    buffer: *mut u8,
    capacity: usize,
}

impl AssemblerData {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer, self.capacity) }
    }

    pub fn as_slcie_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer, self.capacity) }
    }
    pub fn buffer(&self) -> *mut u8 {
        self.buffer
    }
    pub fn new() -> Box<Self> {
        let mut this = Box::new(Self {
            buffer: null_mut(),
            capacity: 128,
            inline_buffer: [0; 128],
        });

        this.buffer = this.inline_buffer.as_mut_ptr();
        this
    }

    pub fn with_capacity(capacity: usize) -> Box<Self> {
        if capacity <= 128 {
            Self::new()
        } else {
            let this = Box::new(Self {
                buffer: unsafe { libc::malloc(capacity).cast() },
                capacity,
                inline_buffer: [0; 128],
            });
            this
        }
    }
    pub fn is_inline(&self) -> bool {
        self.buffer as usize == self.inline_buffer.as_ptr() as usize
    }
    pub fn grow(&mut self, extra_capacity: usize) {
        self.capacity = self.capacity + self.capacity / 2 + extra_capacity;
        if self.is_inline() {
            unsafe {
                self.buffer = libc::malloc(self.capacity).cast();
                core::ptr::copy_nonoverlapping(
                    self.inline_buffer.as_ptr(),
                    self.buffer,
                    self.capacity,
                );
            }
        } else {
            unsafe {
                self.buffer = libc::realloc(self.buffer.cast(), self.capacity).cast();
            }
        }
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        if !self.is_inline() {
            unsafe {
                libc::free(self.buffer.cast());
            }
            self.buffer = self.inline_buffer.as_mut_ptr();
            self.capacity = 128;
        }
    }

    pub fn take_buffer_if_larger(&mut self, other: &mut Self) {
        if other.is_inline() {
            return;
        }

        if self.capacity >= other.capacity {
            return;
        }

        if !self.buffer.is_null() && !self.is_inline() {
            unsafe {
                libc::free(self.buffer.cast());
            }
        }

        self.buffer = other.buffer;
        self.capacity = other.capacity;
        other.buffer = other.inline_buffer.as_mut_ptr();
        other.capacity = 128;
    }
}

impl Drop for AssemblerData {
    fn drop(&mut self) {
        self.clear();
    }
}

pub struct AssemblerBuffer {
    storage: Box<AssemblerData>,
    index: usize,
}

impl AssemblerBuffer {
    pub fn data(&self) -> &AssemblerData {
        &self.storage
    }
    pub fn data_mut(&mut self) -> &mut AssemblerData {
        &mut self.storage
    }
    pub fn new() -> Self {
        Self {
            storage: AssemblerData::new(),
            index: 0,
        }
    }
    pub fn is_available(&self, space: usize) -> bool {
        self.index + space >= self.storage.capacity
    }

    /// Ensures that there is enough memory for 'space' bytes
    pub fn ensure_space(&mut self, space: usize) {
        while !self.is_available(space) {
            self.storage.grow(space);
        }
    }
    /// Returns true if the buffer is aligned to 'alignment'
    pub fn is_aligned(&self, alignment: usize) -> bool {
        (self.index & (alignment - 1)) == 0
    }

    pub fn out_of_line_grow(&mut self) {
        self.storage.grow(0);
    }

    pub fn put_integral<T>(&mut self, value: T) {
        if self.index + size_of::<T>() > self.storage.capacity() {
            self.out_of_line_grow();
        }
        self.put_integral_unchecked(value);
    }

    pub fn put_integral_unchecked<T>(&mut self, value: T) {
        unsafe {
            core::ptr::write_unaligned(self.storage.buffer().add(self.index).cast(), value);
        }
        self.index += size_of::<T>();
    }

    pub fn label(&self) -> AssemblerLabel {
        AssemblerLabel {
            offset: self.index as _,
        }
    }

    pub fn debug_offset(&self) -> usize {
        self.index
    }

    pub fn release_assembler_data(&mut self) -> Box<AssemblerData> {
        core::mem::replace(&mut self.storage, AssemblerData::new())
    }

    pub fn set_code_size(&mut self, size: usize) {
        self.index = size;
    }

    pub fn code_size(&self) -> usize {
        self.index
    }

    pub fn put_int(&mut self, value: i32) {
        self.put_integral(value);
    }
    pub fn put_int_unchecked(&mut self, value: i32) {
        self.put_integral_unchecked(value);
    }

    pub fn put_byte_unchecked(&mut self, value: i8) {
        self.put_integral_unchecked(value);
    }

    pub fn put_byte(&mut self, value: i8) {
        self.put_integral(value);
    }

    pub fn put_short_unchecked(&mut self, value: i16) {
        self.put_integral_unchecked(value);
    }

    pub fn put_short(&mut self, value: i16) {
        self.put_integral(value);
    }

    pub fn put_int64_unchecked(&mut self, value: i64) {
        self.put_integral_unchecked(value);
    }

    pub fn put_int64(&mut self, value: i64) {
        self.put_integral(value);
    }
}

pub struct LocalWriter<'a> {
    buffer: &'a mut AssemblerBuffer,
    storage_buffer: *mut u8,
    index: usize,
}

impl<'a> LocalWriter<'a> {
    pub fn new(buffer: &'a mut AssemblerBuffer, required_space: usize) -> Self {
        let mut this = Self {
            index: buffer.index,
            buffer,
            storage_buffer: null_mut(),
        };

        this.buffer.ensure_space(required_space);
        this.storage_buffer = this.buffer.storage.buffer;
        this
    }

    fn put_integral_unchecked<T>(&mut self, value: T) {
        unsafe {
            core::ptr::write_unaligned(self.storage_buffer.add(self.index).cast(), value);
            self.index += size_of::<T>();
        }
    }

    pub fn put_byte_unchecked(&mut self, value: i8) {
        self.put_integral_unchecked(value)
    }

    pub fn put_short_unchecked(&mut self, value: i16) {
        self.put_integral_unchecked(value)
    }

    pub fn put_int_unchecked(&mut self, value: i32) {
        self.put_integral_unchecked(value)
    }

    pub fn put_int64_unchecked(&mut self, value: i64) {
        self.put_integral_unchecked(value)
    }
}

impl<'a> Drop for LocalWriter<'a> {
    fn drop(&mut self) {
        self.buffer.index = self.index;
    }
}
