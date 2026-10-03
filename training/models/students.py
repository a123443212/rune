STUDENT_BUDGETS = {
    "S1": {"token_dim": 24, "cheap_hidden": 24, "ref_h1": 96, "ref_h2": 24,
            "head_h1": 96, "head_h2": 24},
    "S2": {"token_dim": 16, "cheap_hidden": 16, "ref_h1": 64, "ref_h2": 16,
            "head_h1": 64, "head_h2": 16},
    "S3": {"token_dim": 12, "cheap_hidden": 12, "ref_h1": 48, "ref_h2": 12,
            "head_h1": 48, "head_h2": 12},
    "S4": {"token_dim": 8, "cheap_hidden": 8, "ref_h1": 32, "ref_h2": 8,
            "head_h1": 32, "head_h2": 8},
}


def describe_budget(name):
    if name not in STUDENT_BUDGETS:
        raise ValueError(f"unknown student budget {name}")
    return dict(STUDENT_BUDGETS[name])


def build_dense_student(budget="S2", variant="A", pooling="none", gate_on=False):
    from training.models.dense import build_dense_model

    spec = describe_budget(budget)
    dims = [spec["token_dim"]] * 8
    return build_dense_model(variant=variant, token_dims=dims, pooling=pooling,
                             gate_on=gate_on, head_h1=spec["head_h1"],
                             head_h2=spec["head_h2"])


def build_adaptive_student(budget="S2", cheap_pooling="none", threshold=0.5,
                           uncertainty_on=False):
    from training.models.uncertainty import build_search_model

    spec = describe_budget(budget)
    return build_search_model(dim=spec["token_dim"], cheap_pooling=cheap_pooling,
                              threshold=threshold, uncertainty_on=uncertainty_on,
                              cheap_hidden=spec["cheap_hidden"],
                              ref_h1=spec["ref_h1"], ref_h2=spec["ref_h2"])


def student_report(model):
    total = model.parameter_count()
    cheap = model.cheap_parameter_count() if hasattr(model, "cheap_parameter_count") else total
    return {
        "params": total,
        "param_bytes": model.model_size_bytes(),
        "cheap_params": cheap,
        "spec": model.model_spec(),
    }
