use super::games::VOCAB_SIZES;

pub fn vocab_size(group: usize) -> usize {
    VOCAB_SIZES[group]
}
