import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tokenspeed.runtime.pd.base.mooncake_engine import MooncakeTransferEngine


def _install_fake_modules(monkeypatch, *, is_npu: bool, initialize_result: int = 0):
    engine = Mock()
    engine.initialize.return_value = initialize_result
    engine.get_rpc_port.return_value = 9000

    mooncake = ModuleType("mooncake")
    mooncake.__path__ = []
    mooncake_engine = ModuleType("mooncake.engine")
    mooncake_engine.TransferEngine = Mock(return_value=engine)

    kernel = ModuleType("tokenspeed_kernel")
    kernel.__path__ = []
    platform = ModuleType("tokenspeed_kernel.platform")
    platform.current_platform = lambda: SimpleNamespace(is_npu=is_npu)

    monkeypatch.setitem(sys.modules, "mooncake", mooncake)
    monkeypatch.setitem(sys.modules, "mooncake.engine", mooncake_engine)
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel", kernel)
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel.platform", platform)
    return engine


@pytest.mark.parametrize(
    ("is_npu", "protocol"),
    ((True, "ascend_direct"), (False, "rdma")),
)
def test_transfer_engine_selects_the_platform_protocol(monkeypatch, is_npu, protocol):
    engine = _install_fake_modules(monkeypatch, is_npu=is_npu)

    wrapper = MooncakeTransferEngine("127.0.0.1", 0, "device0")

    engine.initialize.assert_called_once_with(
        "127.0.0.1", "P2PHANDSHAKE", protocol, "device0"
    )
    assert wrapper.get_session_id() == "127.0.0.1:9000"


def test_transfer_engine_rejects_failed_initialization(monkeypatch):
    _install_fake_modules(monkeypatch, is_npu=True, initialize_result=1)

    with pytest.raises(RuntimeError, match="initialization failed"):
        MooncakeTransferEngine("127.0.0.1", 0)
