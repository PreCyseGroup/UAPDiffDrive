import torch

from uap_il.dataset import POLICY_INPUT_DIM
from uap_il.model import ThetaTrigPreprocessor, TrajectoryMLP, WheelMLP


def test_theta_trig_preprocessor_appends_two_trig_channels():
    x_mean = torch.zeros(POLICY_INPUT_DIM)
    x_std = torch.ones(POLICY_INPUT_DIM)
    preprocessor = ThetaTrigPreprocessor(POLICY_INPUT_DIM, x_mean=x_mean, x_std=x_std)
    x = torch.zeros(2, POLICY_INPUT_DIM)
    x[0, 2] = 0.0
    x[1, 2] = torch.pi / 2.0

    out = preprocessor(x)

    assert out.shape == (2, POLICY_INPUT_DIM + 2)
    torch.testing.assert_close(out[:, :POLICY_INPUT_DIM], x)
    torch.testing.assert_close(out[0, -2:], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(out[1, -2:], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=1e-6)


def test_theta_trig_preprocessor_uses_raw_theta_from_normalization_stats():
    x_mean = torch.zeros(POLICY_INPUT_DIM)
    x_std = torch.ones(POLICY_INPUT_DIM)
    x_mean[2] = 10.0
    x_std[2] = 2.0
    preprocessor = ThetaTrigPreprocessor(POLICY_INPUT_DIM, x_mean=x_mean, x_std=x_std)
    x_norm = torch.zeros(1, POLICY_INPUT_DIM)
    x_norm[0, 2] = (torch.pi / 2.0 - x_mean[2]) / x_std[2]

    out = preprocessor(x_norm)

    torch.testing.assert_close(out[0, -2:], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=1e-6)


def test_trajectory_mlp_accepts_preprocessed_ten_dimensional_features():
    model = TrajectoryMLP(n_inputs=10, n_outputs=2, dropout=0.0)

    assert model(torch.zeros(3, 10)).shape == (3, 2)


def test_wheel_mlp_keeps_external_input_dim_and_expands_internal_dim():
    model = WheelMLP(input_dim=POLICY_INPUT_DIM, hidden_dim=8)

    assert model.input_dim == POLICY_INPUT_DIM
    assert model.preprocess.output_dim == POLICY_INPUT_DIM + 2
    assert model.net.stem[0].in_features == POLICY_INPUT_DIM + 2
    assert model(torch.zeros(3, POLICY_INPUT_DIM)).shape == (3, 2)
