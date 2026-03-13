import numpy as np
import pytest

from hyper_surrogate.reporting.reporter import Reporter


@pytest.fixture
def c_batch():
    """Batch of SPD (3,3) tensors mimicking right Cauchy-Green."""
    rng = np.random.default_rng(42)
    n = 20
    # F close to identity + small perturbation → C = F^T F is SPD
    F = np.eye(3) + 0.1 * rng.standard_normal((n, 3, 3))
    return np.einsum("nji,njk->nik", F, F)


@pytest.fixture
def reporter(c_batch):
    return Reporter(c_batch)


def test_init_validates_shape():
    with pytest.raises(ValueError, match="Expected tensor of shape"):
        Reporter(np.zeros((5, 2, 2)))


def test_init_validates_tensor_type():
    with pytest.raises(ValueError, match="tensor_type must be"):
        Reporter(np.zeros((5, 3, 3)), tensor_type="X")


def test_init_from_F():
    rng = np.random.default_rng(0)
    F = np.eye(3) + 0.1 * rng.standard_normal((5, 3, 3))
    r = Reporter(F, tensor_type="F")
    assert r.F is not None
    assert r.C.shape == (5, 3, 3)


def test_n_samples(reporter, c_batch):
    assert reporter.n_samples == c_batch.shape[0]


def test_basic_statistics(reporter):
    stats = reporter.basic_statistics()
    assert set(stats.keys()) == {"det(C)", "I1_bar", "I2_bar", "J"}
    for _name, s in stats.items():
        assert set(s.keys()) == {"mean", "std", "min", "max"}
        assert s["min"] <= s["mean"] <= s["max"]


def test_fig_eigenvalues(reporter):
    figs = reporter.fig_eigenvalues()
    assert len(figs) == 1
    assert len(figs[0].axes) == 3  # one per eigenvalue


def test_fig_determinants(reporter):
    figs = reporter.fig_determinants()
    assert len(figs) == 1


def test_fig_invariants(reporter):
    figs = reporter.fig_invariants()
    assert len(figs) == 1
    assert len(figs[0].axes) == 3  # I1_bar, I2_bar, J


def test_fig_principal_stretches(reporter):
    figs = reporter.fig_principal_stretches()
    assert len(figs) == 1
    assert len(figs[0].axes) == 3


def test_fig_volume_change(reporter):
    figs = reporter.fig_volume_change()
    assert len(figs) == 1


def test_generate_figures(reporter):
    figs = reporter.generate_figures()
    assert len(figs) == len(Reporter.REPORT_FIGURES)


def test_generate_report(tmp_path, reporter):
    reporter.generate_report(tmp_path)
    assert (tmp_path / "report.pdf").exists()


def test_generate_report_standalone(tmp_path, reporter):
    reporter.generate_report(tmp_path, layout="standalone")
    pdfs = list(tmp_path.glob("*.pdf"))
    assert len(pdfs) == len(Reporter.REPORT_FIGURES)


def test_create_report_alias(tmp_path, reporter):
    """create_report is a backward-compat alias for generate_report."""
    reporter.create_report(tmp_path)
    assert (tmp_path / "report.pdf").exists()
