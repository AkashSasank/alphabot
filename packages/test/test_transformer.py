import pytest

torch = pytest.importorskip("torch")

from tradingbot.models import CandleGPT, build_transformer


def test_build_transformer_creates_classification_decoder():
    model = build_transformer(
        input_dim=12,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        output_classes=3,
        output_type="classification",
    )

    x = torch.randn(5, 10, 12)
    logits = model(x)
    probabilities = model.predict_proba(x)

    assert logits.shape == (5, 3)
    assert probabilities.shape == (5, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_build_transformer_creates_regression_decoder():
    model = build_transformer(
        input_dim=12,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        output_dim=2,
        output_type="regression",
    )

    prediction = model(torch.randn(5, 10, 12))

    assert prediction.shape == (5, 2)
    with pytest.raises(RuntimeError, match="classification"):
        model.predict_proba(torch.randn(5, 10, 12))


def test_build_transformer_rejects_encoder_architecture():
    with pytest.raises(ValueError, match="decoder_only"):
        build_transformer(
            input_dim=12,
            model_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            output_classes=2,
            output_type="classification",
            architecture="encoder_decoder",
        )


def test_candle_gpt_wrapper_matches_notebook_constructor():
    model = CandleGPT(
        input_dim=12,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        output_classes=2,
    )

    assert model(torch.randn(5, 10, 12)).shape == (5, 2)
    assert model.output_type == "classification"
    assert model.output_classes == 2
