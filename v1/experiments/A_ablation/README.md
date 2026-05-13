# A: v0.5 Component Ablation Matrix

This experiment dissects the v0.4 memory injection stack into its seven architectural
components and runs a per-component ablation matrix to establish which components are
**necessary** for the measured NLL shift on counterfactual prompts. Each ablation
switches off exactly one component while leaving the other six active; the control arm
has all components enabled. A component is deemed necessary if ablating it significantly
degrades performance (paired Wilcoxon p < 0.01 after Holm correction, 95% CI excludes
0, expected sign).

See `PREREG.md` for the full pre-registration, including hypotheses, grid parameters,
statistical procedures, red-lines, and deliverables.
