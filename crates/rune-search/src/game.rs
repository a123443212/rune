use crate::contract::EvalOutput;

pub trait GameState: Clone {
    type Move: Clone + ToString;
    fn legal_moves(&self) -> Vec<Self::Move>;
    fn apply(&self, m: &Self::Move) -> Self;
    fn is_terminal(&self) -> bool;
    fn terminal_value(&self) -> f32;
    fn key(&self) -> u64;
}

pub trait PolicyNet {
    fn evaluate(&mut self, key: u64) -> EvalOutput;
}

pub trait StateEncoder<S: GameState> {
    fn encode(&self, s: &S) -> u64;
}
