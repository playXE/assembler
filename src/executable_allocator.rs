pub const CODE_ALIGNMENT: usize = 16;
const CHUNK_SIZE: usize = 128 * 1024;
use std::collections::LinkedList;

use crate::os;
use crate::os::*;
use parking_lot::Mutex;

struct AllocData {
    top: Address,
    limit: Address,
    freelist: LinkedList<(Address, usize)>,
}

pub struct CodeSpace {
    total: Region,
    mutex: Mutex<AllocData>,

    chunk_size: usize,
}
impl CodeSpace {
    pub fn new(limit: usize) -> CodeSpace {
        let reservation = os::reserve_align(limit, 0, true);
        let space_start = reservation.start;
        let space_end = space_start.offset(limit);

        let alloc_data = AllocData {
            top: space_start,
            limit: space_start,
            freelist: Default::default(),
        };

        CodeSpace {
            total: Region::new(space_start, space_end),
            mutex: Mutex::new(alloc_data),
            chunk_size: os::page_align(CHUNK_SIZE),
        }
    }

    fn try_freelist(data: &mut AllocData, size: usize) -> Option<(Address, usize)> {
        let mut cursor = data.freelist.cursor_back_mut();
        while let Some(node) = cursor.peek_next() {
            if node.1 <= size {
                return cursor.remove_current();
            }
            cursor.move_next();
        }
        None
    }

    pub fn alloc(&self, size: usize) -> (Address, usize) {
        debug_assert!(size > 0);

        let mut data = self.mutex.lock();
        let aligned_size = align_usize(size, CODE_ALIGNMENT);
        if let Some((addr, size)) = Self::try_freelist(&mut data, aligned_size) {
            let free = aligned_size - size;
            if free >= CODE_ALIGNMENT {
                data.freelist.push_back((addr.offset(aligned_size), free));
            }
            return (addr, aligned_size);
        }
        if data.top.offset(aligned_size) > data.limit {
            let size = align_usize(
                aligned_size - data.limit.offset_from(data.top),
                self.chunk_size,
            );
            let new_limit = data.limit.offset(size);

            if new_limit > self.total.end {
                panic!("OOM in code space");
            }

            os::protect(data.limit, size, MemoryPermission::ReadWriteExecute);
            data.limit = new_limit;
        }

        debug_assert!(data.top.offset(aligned_size) <= data.limit);
        let object_address = data.top;
        data.top = data.top.offset(aligned_size);
        (object_address, aligned_size)
    }

    pub fn allocated_region(&self) -> Region {
        let start = self.total.start;
        let end = self.mutex.lock().top;
        Region::new(start, end)
    }
}

impl Drop for CodeSpace {
    fn drop(&mut self) {
        os::free(self.total.start, self.total.size());
    }
}
