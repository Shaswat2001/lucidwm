"""LucidWM: LeWM (LeWorldModel) on PushT / Reacher / OGBench

Paper:     "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
            Architecture from Pixels" (Maes et al., 2026)
Reference: https://github.com/lucas-maes/le-wm

Core idea:
  The simplest possible JEPA world model. Learn an encoder and predictor
  end-to-end from raw pixels with ONLY TWO loss terms: MSE prediction loss
  + SIGReg (a regularizer that prevents collapse by enforcing a Gaussian
  latent distribution). No stop-gradient, no EMA, no frozen backbone,
  no reconstruction loss, no reward model. Just predict next latent and
  regularize. That's it.

Architecture:
  Encoder:    ViT-Tiny (~5M params). Image (3,96,96) -> CLS token -> z in R^192.
  Projectors: TWO separate MLP+BatchNorm heads (matching repo):
              - Encoder projector: projects encoder CLS token into embedding space
              - Predictor projector: projects predictor output into same space
              MSE loss is computed between these projected representations.
  Predictor:  Transformer (~10M params). (z_t, a_t) -> z_hat_{t+1}.
              Action is projected to latent_dim and added as a token.
              Dropout 0.1 in predictor (critical for stability).
  Planning:   CEM in latent space. Cost = ||z_pred - z_goal||^2.
              Each frame is a single 192-dim token (vs 256 tokens for DINO-WM).
              Planning completes in ~1 second (48x faster than DINO-WM).

Training objective (the whole thing):
  L = L_pred + lambda * SIGReg(Z_all)
  L_pred = MSE(pred_projector(predictor(z_t, a_t)), enc_projector(encoder(o_{t+1})))
  SIGReg = Epps-Pulley normality test on ALL encoder embeddings (z_t AND z_{t+1})
  Note: SIGReg is applied to the FULL sequence of embeddings, not just current frame.

Key hyperparameters:
  - Encoder: ViT-Tiny (patch_size=8, embed_dim=192, depth=12, heads=3)
  - Latent dim: 192 (CLS token from ViT-Tiny)
  - Predictor: depth=6, heads=8, dim=512
  - Predictor dropout: 0.1 (critical)
  - SIGReg lambda: 0.1
  - SIGReg projections M: 512
  - Total params: ~15M
  - Training: single GPU, few hours

Components from Layer 1: CEMPlanner (adapted for goal-reaching)
Env:     PushT, Reacher, Two-Room, OGBench-Cube
Target:  Competitive with DINO-WM without pre-trained backbone
Compute: ~2-4 hours on single GPU
"""
