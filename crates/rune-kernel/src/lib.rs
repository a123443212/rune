pub mod activations;
pub mod arena;
pub mod conv;
pub mod dense;
pub mod fused;
pub mod policy;
pub mod quant;
pub mod scale;
pub mod simd;
pub mod specialized;
pub mod util;

pub use activations::{clamp_delta, clipped_relu, hard_sigmoid, relu, relu_inplace, screlu, tokens_clip, tokens_dequant_clip, Gate};
pub use conv::{conv2d_nchw, global_avg_pool, residual_add, residual_add_relu};
pub use dense::{mat_mul, mat_mul_tt, mat_vec, mat_vec_clipped, mat_vec_scalar};
pub use policy::{normalize_policy, policy_entropy, softmax};
pub use scale::{dequantize, quantize_half_away, symmetric_scale};
pub use simd::{active_path, active_path_name, clear_path_for_test, set_path_for_test, uses_simd, KernelPath};
pub use util::{max_abs_diff, routing_refine, spec_check};
