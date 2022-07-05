use capstone::{arch::arm64::ArchMode, prelude::*};
use masm::{
    arm64::*,
    macro_assembler::{Address, BaseIndex, MacroAssemblerARM64, Scale, GR},
};

fn foo() {}

fn main() {
    let mut asm = MacroAssemblerARM64::new();
    asm.store64_imm(1, Address::new(x0, 42));
    let mut cs = Capstone::new().arm64().mode(ArchMode::Arm).build().unwrap();
    cs.set_skipdata(true).unwrap();

    let insns = cs
        .disasm_all(
            &asm.buffer().data().as_slice()[0..asm.buffer().code_size()],
            0x0,
        )
        .unwrap();
    for ins in insns.iter() {
        println!("{}", ins);
    }
}
