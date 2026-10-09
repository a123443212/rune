use rune_ir::RuneIr;
use rune_model::RuneModel;
use super::parser_classic::build_classic;
use super::parser_resnet::build_resnet;

pub fn build_from_model(m: &RuneModel, isa: &str, cpu: &str) -> Result<RuneIr, String> {
    if rune_ir::is_resnet_arch(&m.header.architecture_id) {
        build_resnet(m, isa, cpu)
    } else {
        build_classic(m, isa, cpu)
    }
}
