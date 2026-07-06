# Phlogiston runtime image for the gbt350 box (8x AMD Instinct MI350X, gfx950).
#
# Base is a ROCm PyTorch image that already ships a ROCm build of torch, so we
# must NOT reinstall torch from PyPI (that would clobber the ROCm build with a
# CPU/CUDA wheel). The default tag below is already cached on the gbt box:
#   rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0
ARG ROCM_PYTORCH_TAG=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0
FROM ${ROCM_PYTORCH_TAG}

# gfx950 = MI350X. Set the build arch so ROCm libs target it directly.
# NOTE: do NOT set HSA_OVERRIDE_GFX_VERSION -- even an empty value breaks HSA
# device enumeration (torch.cuda.device_count() -> 0). gfx950 is natively
# supported by ROCm 7.2, so no override is needed.
ENV PYTORCH_ROCM_ARCH=gfx950 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1

# Bypass corporate TLS-inspection cert verification for package hosts, matching
# the local dev setup (avoids SSL: CERTIFICATE_VERIFY_FAILED behind the proxy).
RUN printf '[global]\ntrusted-host =\n    pypi.org\n    files.pythonhosted.org\n' \
    > /etc/pip.conf
# Same for git so the in-container `git pull` works behind the proxy.
RUN git config --system http.sslVerify false
# Authenticate git over HTTPS using a PAT supplied at runtime via $GITHUB_PAT
# (passed with `docker run --env-file`). No secret is baked into the image.
RUN git config --system credential.helper \
    '!f() { echo "username=x-access-token"; echo "password=${GITHUB_PAT}"; }; f'

WORKDIR /workspace/Phlogiston

# Copy the repo (including .git, so `git pull` works inside the container).
COPY . /workspace/Phlogiston

# Install everything EXCEPT torch (provided by the ROCm base image).
# Exclude the bare `torch` requirement (provided by the ROCm base image) while
# keeping siblings like torch-geometric.
RUN grep -viE '^\s*torch\s*[><=~!]' requirements.txt > /tmp/requirements.notorch.txt \
    && pip install --no-cache-dir -r /tmp/requirements.notorch.txt \
    && pip install --no-cache-dir -e . --no-deps

CMD ["/bin/bash"]
