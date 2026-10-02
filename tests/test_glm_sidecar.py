from cachalot.glm import model as glm_model


def test_sidecar_key_flattens_forget_gate_and_names_the_fused_conv():
    key = glm_model._sidecar_key
    assert key("language_model.model.layers.3.self_attn.forget_gate.f_a_proj.weight") == "language_model.model.layers.3.self_attn.f_a_proj.weight"
    assert key("language_model.model.layers.3.self_attn.forget_gate.A_log") == "language_model.model.layers.3.self_attn.A_log"
    assert key("language_model.model.layers.3.self_attn.conv1d.weight") == "language_model.model.layers.3.self_attn.qkv_conv.conv.weight"


def test_sidecar_key_leaves_other_names_alone():
    for name in (
        "language_model.model.layers.3.self_attn.o_proj.weight",
        "language_model.model.layers.3.attn_hc.base",
        "language_model.lm_head.scales",
    ):
        assert glm_model._sidecar_key(name) == name


def test_sidecar_file_name_is_stable():
    assert glm_model.NONEXPERT_SIDECAR == "nonexpert-sanitized.safetensors"
