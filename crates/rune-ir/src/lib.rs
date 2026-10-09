pub mod arch;
pub mod incremental;
pub mod kernels;
pub mod types;
pub mod verify;
pub mod version;

pub use arch::{arch_family, is_resnet_arch, required_ops_for_arch};
pub use kernels::{kernel_for, shape_key};
pub use types::{BufferPlace, FusionGroup, IrModel, IrOp, IrTarget, IrTensor, KernelEntry, MemoryPlan, RuneIr};
pub use verify::{parse_bytes, valid_isa, verify};
pub use version::{ACCEPTED_IR_VERSIONS, ACCEPTED_SPEC_VERSIONS, CLASSIC_REQUIRED_OPS, COMPILER_VERSION, IR_VERSION, PACKING_VERSION, RESNET_OP_KINDS, RESNET_OPTIONAL_OPS, RESNET_REQUIRED_OPS, SPEC_VERSION};
