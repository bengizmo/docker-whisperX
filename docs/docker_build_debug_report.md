# Docker Build Debugging Report: Resolving WhisperX Installation Issues

**Date:** 2025-04-12

## Objective

To diagnose and fix the persistent `ModuleNotFoundError: No module named 'whisperx'` error encountered during the test phase of a multi-stage Docker build for a WhisperX application.

## Initial State

The Docker build process was failing consistently at the final test step:
```dockerfile
RUN /venv/bin/python3 -c 'import whisperx; print("whisperX import successful")' && \
    /venv/bin/whisperx -h
```
The error indicated that the `whisperx` module, despite various installation attempts, could not be found by the Python interpreter within the `/venv` virtual environment in the final image stage. Initial attempts involved installing `whisperx` from a local Git submodule.

## Debugging Steps & Findings

The debugging process involved several approaches and iterations:

1.  **Submodule Investigation:**
    *   **Attempt:** Update the local `whisperX` Git submodule to the latest `main` branch commit (`git submodule update --init --recursive --remote`).
    *   **Finding:** The `main` branch commit (`172d9492`) contained code inconsistencies:
        *   Incorrect import source for `Segment` in `whisperx/vads/pyannote.py`.
        *   Missing `whisperx/vads/vad.py` file required by an import.
    *   **Attempt:** Check out a known stable tag (`v3.1.1`).
    *   **Finding:** Failed, as `git fetch --tags` and `git ls-remote --tags` revealed no tags were available on the submodule's remote repository.
    *   **Conclusion:** Installing from the submodule was deemed unreliable due to the instability and lack of tagged releases in the tracked commit/branch.

2.  **Switch to PyPI Installation (Build Stage):**
    *   **Decision:** Modify the Dockerfile to remove submodule steps and install the stable `whisperx` package from PyPI using `uv` in the `prepare_build_amd64` stage.
    *   **Attempt 1:** Added `whisperx` to the `uv pip install --no-deps` command.
        *   *Failure:* `ModuleNotFoundError` persisted. The `--no-deps` flag likely prevented necessary dependencies from being installed.
    *   **Attempt 2:** Moved `whisperx` installation to a separate `uv pip install whisperx` command *without* `--no-deps`.
        *   *Failure:* `ModuleNotFoundError` persisted in the final stage.
    *   **Attempt 3:** Verified package existence (`whisperx==3.3.2`) and explicitly set the version: `uv pip install whisperx==3.3.2`.
        *   *Failure:* `ModuleNotFoundError` persisted in the final stage.

3.  **Investigating Installation Environment (Build Stage):**
    *   **Attempt 4:** Added `uv pip list` after the `whisperx` installation in the build stage (`prepare_build_amd64`) to verify installation.
        *   *Failure:* Build failed due to a stray `pyannote.audio==3.3.2` argument mistakenly appended to the `RUN` command during previous edits. Corrected syntax.
        *   *Failure:* Build failed again due to an incorrect line continuation (`\`) merging the `RUN` command with the subsequent one. Corrected syntax.
    *   **Finding:** After fixing syntax, `uv pip list` *confirmed* that `whisperx==3.3.2` and its dependencies *were* successfully installed by `uv` within the `/venv` environment during the build stage. However, the final stage test still failed with `ModuleNotFoundError`.

4.  **Alternative Installation Strategy (Final Stage):**
    *   **Hypothesis:** The issue might be related to how the `/venv` directory was copied or activated in the final `no_model` stage.
    *   **Decision:** Move the `whisperx` installation entirely to the final stage, installing it just before the test command.
    *   **Attempt 5:** Used `/venv/bin/pip install whisperx==3.3.2` in the final stage `RUN` command.
        *   *Failure:* Build failed with `/venv/bin/pip: not found`. Realized `uv venv` doesn't install `pip` into the venv by default.
    *   **Attempt 6:** Tried installing `pip` into the venv during the build stage (`uv pip install pip`).
        *   *Failure:* `/venv/bin/pip: not found` persisted in the final stage. Installing `pip` in the build stage venv didn't make it available after the venv was copied.
    *   **Attempt 7:** Installed `python3-pip` via `apt-get` in the `prepare_base_amd64` stage (the base for the final stage).
        *   *Failure:* `/venv/bin/pip: not found` persisted. `apt-get` installs to the system, not `/venv/bin`.
    *   **Attempt 8:** Used the system `pip3` in the final stage: `pip3 install whisperx==3.3.2`.
        *   *Partial Success:* `pip3` was found, and installation *succeeded*.
        *   *Failure:* `ModuleNotFoundError` persisted! `pip3` installed `whisperx` into the *system* site-packages, not the `/venv` site-packages used by `/venv/bin/python3`.
    *   **Attempt 9 (Success):** Used the virtual environment's Python interpreter to run the `pip` module: `RUN /venv/bin/python3 -m pip install whisperx==3.3.2 && ...`
        *   *Success:* This command correctly used the `pip` module associated with the `/venv` Python interpreter, installing `whisperx` directly into the `/venv/lib/python3.11/site-packages` directory. The subsequent test command `/venv/bin/python3 -c 'import whisperx; ...'` found the module, and the build completed successfully.

## Root Cause Analysis

The core issue stemmed from the interaction between the multi-stage Docker build, the use of `uv` for virtual environment creation (`--system-site-packages`), and the need to install packages into that specific virtual environment in the final stage.

- Installing `whisperx` in the build stage *appeared* successful according to `uv pip list`, but the module wasn't accessible after `/venv` was copied to the final stage. The exact reason for this remains slightly unclear but might relate to how `uv` handles installations or how the environment is copied/activated.
- Installing `whisperx` in the final stage required `pip`.
    - `pip` wasn't included by `uv venv`.
    - Installing `pip` via `uv` in the build stage didn't make it available in the final stage venv copy.
    - Installing `python3-pip` via `apt-get` made `pip3` available system-wide, but using `pip3 install` installed `whisperx` to the system site-packages, not the virtual environment's site-packages, even though the venv used `--system-site-packages`. The venv's Python (`/venv/bin/python3`) still couldn't find the module.
- The final solution forced the installation directly into the virtual environment's context by invoking its specific Python interpreter with the `-m pip` flag.

## Final Solution

1.  Ensure `python3-pip` is installed in the base image stage (`prepare_base_amd64`) using `apt-get install python3-pip`.
2.  Remove any `whisperx` installation steps from the build stage (`prepare_build_amd64`). Keep core dependencies like `torch`, `torchaudio`, `pyannote.audio` installed here using `uv`.
3.  In the final image stage (`no_model`), add a `RUN` command *before* the test step to install `whisperx` directly into the virtual environment:
    ```dockerfile
    # Install and test whisperx in the final stage
    RUN /venv/bin/python3 -m pip install whisperx==3.3.2 && \
        /venv/bin/python3 -c 'import whisperx; print("whisperX import successful")' && \
        /venv/bin/whisperx -h
    ```

## Troubleshooting Tips for Similar Issues

*   **Verify Installation Location:** When a module isn't found, use commands like `which python`, `which pip`, `pip show <package>`, `python -c 'import sys; print(sys.path)'` *within the specific execution context (stage and interpreter)* to confirm where packages are expected and where they actually are.
*   **Multi-Stage Context:** Be explicit about which tools and environments are available in each Docker stage. Don't assume tools installed in a build stage are present later unless explicitly copied.
*   **Virtual Environment Specificity:** When installing into a specific virtual environment, especially one copied between stages or using `--system-site-packages`, prefer using the venv's own interpreter to run installation tools (`/path/to/venv/bin/python -m pip install ...`) to avoid ambiguity with system-level tools.
*   **Syntax Matters:** Double-check Dockerfile syntax, especially line continuations (`\`) and argument separation (`&&`), as errors can silently merge or break commands.
*   **Isolate Changes:** When debugging, make one change at a time and rebuild to clearly identify the impact of each modification. Use `--no-cache` periodically to ensure changes aren't masked by cached layers.

## Future Guide & Recommendations

*   **Prefer Stable Sources:** Use stable package versions from PyPI whenever possible, rather than relying on potentially unstable Git branches or submodules, unless active development requires it.
*   **Explicit Installation Context:** When installing Python packages in Docker, especially in multi-stage builds or with virtual environments, be explicit about the target environment. Using `/path/to/venv/bin/python -m pip install ...` is often safer than relying on system `pip` or `pip3`.
*   **Minimize Final Stage Installations:** While installing in the final stage fixed this issue, it generally increases image size and build time compared to installing in a build stage and copying the result. If the build stage installation *can* be made to work reliably (perhaps by avoiding `--system-site-packages` or ensuring `uv` installs correctly for copying), that might be preferable. However, the final stage installation proved necessary here.
*   **Simplify Venv:** Consider if `--system-site-packages` is truly necessary. It can sometimes complicate dependency management compared to a fully isolated virtual environment.