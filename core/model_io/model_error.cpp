#include "core/model_io/model_error.h"
namespace rune {
const char* modelErrorName(ModelError e) {
  switch (e) {
    case ModelError::Ok: return "ok";
    case ModelError::CannotOpen: return "cannot-open";
    case ModelError::BadMagic: return "bad-magic";
    case ModelError::HeaderTooLarge: return "header-too-large";
    case ModelError::Truncated: return "truncated";
    case ModelError::BadHeader: return "bad-header";
    case ModelError::UnsupportedFormat: return "unsupported-format";
    case ModelError::UnsupportedArch: return "unsupported-architecture";
    case ModelError::UnsupportedQuant: return "unsupported-quantization";
    case ModelError::FeatureMismatch: return "feature-version-mismatch";
    case ModelError::TensorMismatch: return "tensor-mismatch";
    case ModelError::ShapeMismatch: return "shape-mismatch";
    case ModelError::ChecksumMismatch: return "checksum-mismatch";
    case ModelError::Oversized: return "oversized-allocation";
    case ModelError::InvalidDim: return "invalid-dim";
  }
  return "unknown";
}
}
