"""Auto-dispatch single-slot bank patch based on model class."""
def install_bank_patch_for_model(model):
    cls = type(model).__name__.lower()
    if "qwen3" in cls:
        from .qwen3_lpl_patch import install_lpl_patch, LPLState, lpl_state_scope
        install_lpl_patch(model)
    elif "gemma2" in cls:
        from .gemma2_bank_patch import install_lpl_patch, LPLState, lpl_state_scope
        install_lpl_patch(model)
    elif "qwen2" in cls:
        from .vanilla_bank_patch import install_qwen2_patch as install_lpl_patch, LPLState, lpl_state_scope
        install_lpl_patch(model)
    elif "llama" in cls:
        from .vanilla_bank_patch import install_llama_patch as install_lpl_patch, LPLState, lpl_state_scope
        install_lpl_patch(model)
    else:
        raise NotImplementedError(f"no bank patch for {cls}")
    return LPLState, lpl_state_scope
