#pragma once
#include <string>
namespace rune {
enum class ModelError {
  Ok = 0,
  CannotOpen,
  BadMagic,
  HeaderTooLarge,
  Truncated,
  BadHeader,
  UnsupportedFormat,
  UnsupportedArch,
  UnsupportedQuant,
  FeatureMismatch,
  TensorMismatch,
  ShapeMismatch,
  ChecksumMismatch,
  Oversized,
  InvalidDim
};
struct LoadStatus {
  ModelError code = ModelError::Ok;
  std::string message;
  bool ok() const { return code == ModelError::Ok; }
};
const char* modelErrorName(ModelError e);
}
