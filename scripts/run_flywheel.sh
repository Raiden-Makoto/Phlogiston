set -euo pipefail
cd /workspace/Phlogiston
echo ===FINETUNE===
python -m phlogiston.cli train --stage 1 --init-ckpt data/runs/predictor_stage1_best.pt --feedback-root data/runs/feedback --feedback-weight 30 --lr 1e-4 --epochs 4 --warmup-epochs 1 --patience 0 --num-workers 0 --out-dir data/runs/ft_stage1
echo ===HEAD===
python -m phlogiston.cli fit-latent-head --generator data/runs/cdvae_long/cdvae_best.pt --feedback-root data/runs/feedback --feedback-weight 30 --epochs 150 --num-workers 0 --out data/runs/latent_head_ft.pt
echo ===DISCOVER===
mkdir -p data/runs/candidates_ft
cp -rn data/runs/candidates_synth/verify_cache data/runs/candidates_ft/ 2>/dev/null || true
python -m phlogiston.cli discover --generator data/runs/cdvae_long/cdvae_best.pt --predictor data/runs/property/predictor_stage2_best.pt --stability-ckpt data/runs/ft_stage1/predictor_stage1_last.pt --latent-head data/runs/latent_head_ft.pt --synth-ckpt data/runs/synth_best.pt --synth-min 0.3 --n-samples 384 --steps-per-level 8 --cond-steps 100 --cond-trust-radius 4.0 --e-hull-max 0.1 --top-k 12 --save-dir data/runs/candidates_ft
echo ===VERIFY===
python -m phlogiston.cli verify --save-dir data/runs/candidates_ft --backend chgnet --cross-backend mattersim --ehull-cutoff 0.05 --relax-steps 300 --competitor-relax-steps 150 --phonon-mesh 8
echo ===DONE===
