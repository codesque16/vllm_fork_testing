# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM CLI entrypoint: ``python -m vllm.entrypoints.dflash_draft_server``."""

from vllm.v1.spec_decode.disagg_dflash.draft_server import main

if __name__ == "__main__":
    main()
