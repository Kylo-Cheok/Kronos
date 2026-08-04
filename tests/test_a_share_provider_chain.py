from types import SimpleNamespace

import pandas as pd

from data.prepare_a_share_finetune import (
    Candidate,
    SelectedSymbol,
    fetch_current_eligibility,
    download_history,
    normalize_and_validate_history,
)


def _adata_frame(rows: int = 520) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=rows)
    close = pd.Series(range(100, 100 + rows), dtype=float)
    return pd.DataFrame(
        {
            "trade_time": dates.astype(str),
            "open": close,
            "close": close + 1,
            "high": close + 2,
            "low": close - 1,
            "volume": 1000,
            "amount": 100000,
        }
    )


def test_adata_eligibility_uses_stock_code_aliases():
    info = SimpleNamespace(
        index_constituent=lambda index_code: pd.DataFrame(
            {"index_code": [index_code], "stock_code": ["300750"]}
        ),
        all_code=lambda: pd.DataFrame(
            {
                "stock_code": ["300750", "688169"],
                "short_name": ["CATL", "Roborock"],
            }
        ),
    )
    provider = SimpleNamespace(stock=SimpleNamespace(info=info))

    universe = fetch_current_eligibility(provider, provider_name="adata")

    assert universe.csi300_codes == {"300750"}
    assert universe.listed_names["688169"] == "Roborock"


def test_adata_history_columns_are_normalized():
    normalized = normalize_and_validate_history(
        _adata_frame(), "300750", "20220103", "20240101", 512
    )

    assert list(normalized.columns) == [
        "timestamps",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert len(normalized) == 520
    assert normalized["timestamps"].iloc[0] == "2022-01-03"


def test_adata_history_route_calls_market_api():
    raw = _adata_frame()

    class Market:
        def get_market(self, **kwargs):
            assert kwargs["stock_code"] == "300750"
            assert kwargs["k_type"] == 1
            assert kwargs["adjust_type"] == 1
            return raw

    selected = SelectedSymbol(
        candidate=Candidate("new_energy", "300750", "test"),
        name="CATL",
        eligible_via=("ChiNext",),
    )
    provider = SimpleNamespace(stock=SimpleNamespace(market=Market()))

    normalized = download_history(
        provider,
        selected,
        "20220103",
        "20240101",
        "qfq",
        20.0,
        512,
        provider_name="adata",
    )

    assert len(normalized) == 520


def test_akshare_csindex_constituents_are_accepted_as_a_sina_fallback():
    provider = SimpleNamespace(
        index_stock_cons_csindex=lambda symbol: pd.DataFrame(
            {"成分券代码": ["600519", "300750"]}
        ),
        stock_info_a_code_name=lambda: pd.DataFrame(
            {"code": ["600519", "300750", "688169"], "name": ["Moutai", "CATL", "Roborock"]}
        ),
    )

    universe = fetch_current_eligibility(provider, provider_name="akshare")

    assert universe.csi300_codes == {"600519", "300750"}
    assert universe.listed_names["688169"] == "Roborock"


def test_akshare_tencent_history_route_uses_exchange_prefixed_symbol():
    raw = _adata_frame()

    class Provider:
        def stock_zh_a_hist_tx(self, **kwargs):
            assert kwargs["symbol"] == "sh600519"
            assert kwargs["adjust"] == ""
            return raw

    selected = SelectedSymbol(
        candidate=Candidate("consumer", "600519", "test"),
        name="Moutai",
        eligible_via=("CSI300",),
    )
    normalized = download_history(
        Provider(),
        selected,
        "20220103",
        "20240101",
        "",
        20.0,
        512,
        provider_name="akshare_tencent",
    )

    assert len(normalized) == 520
