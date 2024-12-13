import numpy as np
import pytest

from hyper_surrogate.reporter import Reporter


@pytest.fixture
def tensor():
    return np.random.rand(2, 3, 3)


@pytest.fixture
def reporter(tensor):
    return Reporter(tensor)


# The class can be instantiated with a tensor and a save directory.
def test_statistics(reporter):
    # basic_statistics returns a tuple with mean, median, std_dev, and value_range
    assert len(reporter.basic_statistics()) == 4


def test_fig_eigenvalues(reporter):
    # visualize_eigenvalues returns a list of figures
    assert len(reporter.visualize_eigenvalues()) == 1


def test_fig_determinants(reporter):
    # visualize_determinants returns a list of figures
    assert len(reporter.visualize_determinants()) == 1


def test_generate_figures(reporter):
    # generate_figures returns a list of figures
    assert len(reporter.generate_figures()) == 2


def test_generate_report(tmp_path, reporter):
    # create_report saves a pdf file
    reporter.create_report(tmp_path)
    assert (tmp_path / "report.pdf").exists()
