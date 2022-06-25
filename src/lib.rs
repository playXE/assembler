#![feature(associated_type_defaults, linked_list_cursors)]
#[macro_use]
pub mod assembler_common;

#[cfg(target_arch = "aarch64")]
pub mod arm64;
pub mod assembler_buffer;
pub mod executable_allocator;
pub mod link_buffer;
pub mod macro_assembler;
pub mod os;

#[cfg(target_arch = "aarch64")]
pub use arm64::{registers::*, *};
