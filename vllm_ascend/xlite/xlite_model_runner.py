#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
# isort: skip_file
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.xlite.xlite import XliteWrapper


class XliteModelRunner(NPUModelRunner):
    runner_cls: type[XliteWrapper] = XliteWrapper
    _runner_enabled: bool = False  # the current model forward backend is not yet the xlite runner

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8:
            logger.error("xlite does not currently support Sparse SFA/LI C8.")

    def get_model(self) -> nn.Module:
        """See :meth:`NPUModelRunner.get_model` and :meth:`XliteWrapper.unwrap` for details."""
        if not self.runner_enabled:
            return super().get_model()
        with self._bypass_xlite_wrapper():
            return super().get_model()

    def load_model(self) -> None:
        super().load_model()
        self.fallback_model = self.model
        # NOTE: this will create a circular reference between XliteModelRunner and XliteWrapper instances,
        # but this should be fine since they are both long-lived objects
        self.model = self.runner_model = self.runner_cls(self, self.vllm_config, device=self.device)  # type: ignore[assignment]

    @contextmanager
    def _bypass_xlite_wrapper(self):
        """Temporarily route ``self.model`` to the native runnable.

        Profile runs and dummy runs (including ``execute_dummy_batch`` and cudagraph capture warmups) should not enter
        the xlite forward path.
        """
        self.model = self.fallback_model
        try:
            yield
        finally:
            self.model = self.runner_model  # type: ignore[assignment]

    def _dummy_run(
        self, *args: Any, is_profile: bool = False, is_graph_capturing: bool = False, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route dummy/profile runs to the native runnable, bypassing the xlite wrapper.

        See :meth:`_bypass_xlite_wrapper` for why dummy runs must not trigger the xlite forward. Delegates every
        argument to the base implementation unchanged.
        """
        if not is_profile and not is_graph_capturing:
            # DP `excute_dummy_batch` must be routed to xlite forward path to avoid out of sync issues
            return super()._dummy_run(*args, is_profile=False, is_graph_capturing=False, **kwargs)  # type: ignore[return-value]

        with self._bypass_xlite_wrapper():
            return super()._dummy_run(*args, is_profile=False, is_graph_capturing=False, **kwargs)  # type: ignore[return-value]

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_kv_cache(kv_cache_config)
        self.runner_model.register_kv_caches(self.kv_caches)  # type: ignore[arg-type]

        # check attention metadata backend compatibility
        ascend_metadata_builder = self.attn_groups[0][-1].get_metadata_builder(0)
        ascend_metadata_cls = getattr(ascend_metadata_builder, "metadata_cls", ascend_metadata_builder)
        xlite_expects = self.runner_model.adapter_xlite_model._attn_metadata_type
        if ascend_metadata_cls != xlite_expects and (
            not isinstance(xlite_expects, tuple) or ascend_metadata_cls not in xlite_expects
        ):
            logger.error(
                "Attention metadata mismatch: xlite expects (one of) %s, but got %s. Be aware of runtime issues.",
                xlite_expects,
                ascend_metadata_cls,
            )

    def _should_build_dummy_attn_metadata(
        self, force_attention: bool = False, is_profile: bool = False, *args: Any, **kwargs: Any
    ) -> bool:
        base_condition = super()._should_build_dummy_attn_metadata(force_attention, is_profile, *args, **kwargs)
        if base_condition or not self.runner_enabled:
            return base_condition
        return self.ascend_config.xlite_graph_config.enabled and not is_profile

    @property
    def model(self) -> nn.Module:
        """The current model forward backend."""
        return self._model

    @model.setter
    def model(self, value: nn.Module) -> None:
        self._model = value
        self._runner_enabled = isinstance(value, self.runner_cls)

    @property
    def runner_enabled(self) -> bool:
        """If the current model forward backend is the xlite runner."""
        return self._runner_enabled

    @runner_enabled.setter
    def runner_enabled(self, value: bool) -> None:
        self._runner_enabled = value
