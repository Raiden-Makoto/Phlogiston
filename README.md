# Phlogiston

A crystal-structure ML framework for materials discovery, pairing the **GNoME**
stable-material candidates with structural records from the **Materials Project**
and training established structural architectures (CGCNN) to screen for new
stable materials.

## Running on the gbt350 box (8x AMD Instinct MI350X)

The target compute is the `gbt` box (ROCm 7.2, gfx950). Local Docker is not
available, so we build the image **on the box** (the ROCm PyTorch base image is
already cached there) and keep `.git` in the image for an in-container update
loop.

Prerequisites: an `ssh gbt` entry (already configured in `~/.ssh/config`).

```bash
# From the repo root on your workstation:
./docker/deploy_gbt.sh sync    # rsync repo -> gbt:~/Phlogiston (keeps .git)
./docker/deploy_gbt.sh build   # docker build -t phlogiston:rocm on the box
./docker/deploy_gbt.sh run     # interactive container with all 8 GPUs

# Inside the container, per your workflow:
git pull                       # get the most recent code
python -c "import torch; print(torch.cuda.device_count())"   # -> 8
```

`docker/deploy_gbt.sh all` runs `sync` + `build` in one shot.

### Notes on the ROCm image
- Base: `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0`
  (torch 2.8 + ROCm 7.2). We **do not** reinstall torch from PyPI, which would
  clobber the ROCm build.
- Do **not** set `HSA_OVERRIDE_GFX_VERSION`; even an empty value breaks HSA
  device enumeration on gfx950 (`device_count()` returns 0).
- The container is launched with the host's numeric `render`/`video` group GIDs
  plus `--device=/dev/kfd --device=/dev/dri` so it can reach the GPUs.

## Project layout

```
phlogiston/
  config.py                 # dataclass configs (YAML-serializable)
  data/                     # GNoME + Materials Project loaders, crystal graphs
  models/                   # CGCNN architecture
  train.py / discover.py    # train a property model, then screen candidates
Dockerfile                  # ROCm image for the gbt box
docker/deploy_gbt.sh        # sync / build / run helper
```
