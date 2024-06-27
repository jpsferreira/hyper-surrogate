from datetime import datetime
from pathlib import Path
from typing import ClassVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

matplotlib.use("svg")
plt.set_loglevel("error")
plt.rcParams["figure.max_open_warning"] = -1


class Reporter:
    """
    Generate a PDF report from batches of tensors.

    It provides various methods to create different types of visualizations and tables based on the parsed data.

    """

    LAYOUT: ClassVar[list[str]] = ["standalone", "combined"]

    FIG_SIZE = (8.27, 11.69)  # A4 paper size in inches

    REPORT_FIGURES: ClassVar[list[str]] = [
        "visualize_eigenvalues",
        "visualize_determinants",
    ]

    def __init__(self, tensor: np.ndarray, save_dir: Path):
        self.tensor = tensor  # tensor is a np.array (N,3,3)
        self.save_dir = save_dir
        # Path.mkdir(Path(self.save_dir), parents=True, exist_ok=True)
        self.pdf_metadata = {
            "Title": "Generation Session Report",
            "Author": "João Ferreira",
            "Subject": "Tensor Batch Session Summary",
            "CreationDate": datetime.today(),
        }

    def basic_statistics(self) -> tuple[float, float, float, float]:
        # Calculate mean, median, standard deviation, and range for the entire tensor
        mean = np.mean(self.tensor)
        median = np.median(self.tensor)
        std_dev = np.std(self.tensor)
        value_range = np.ptp(self.tensor)  # Peak-to-peak (max - min)
        return (mean, median, std_dev, value_range)

    def visualize_eigenvalues(self) -> list[matplotlib.figure.Figure]:
        # Plot histograms of the eigenvalues
        eigenvalues = np.linalg.eigvals(self.tensor)
        fig, ax = plt.subplots(1, 1, figsize=self.FIG_SIZE)
        sns.histplot(eigenvalues.flatten(), kde=True, ax=ax, alpha=0.75)
        plt.title("Histogram of Eigenvalues")
        plt.xlabel("Eigenvalue")
        plt.ylabel("Frequency")
        return [fig]

    def visualize_determinants(self) -> list[matplotlib.figure.Figure]:
        # Plot a histogram of the determinants of the 3x3 matrices
        determinants = np.linalg.det(self.tensor)
        fig, ax = plt.subplots(1, 1, figsize=self.FIG_SIZE)
        sns.histplot(determinants.flatten(), kde=True, ax=ax)
        # plt.hist(determinants, alpha=0.75)
        plt.title("Histogram of Determinants")
        plt.xlabel("Determinant")
        plt.ylabel("Frequency")
        return [fig]

    @staticmethod
    def tablefig(
        df: pd.DataFrame,
        figsize: tuple = (8.27, 11.69),
        location: str = "center left",
        title: str = "Title",
    ) -> list[matplotlib.figure.Figure]:
        """Create a table figure using the provided DataFrame and displays it using Matplotlib."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc=location,
            rowLabels=df.index,
        )

        ax.axis("off")
        ax.set_title(title)
        return [fig]

    def generate_figures(self) -> list[matplotlib.figure.Figure]:
        """Generate figures for report."""
        fig_list = []
        report_figures_pbar = tqdm(self.REPORT_FIGURES, total=len(self.REPORT_FIGURES), leave=False)
        for report_figure in report_figures_pbar:
            report_figures_pbar.set_description(f"Generating {report_figure}")
            func = getattr(self, report_figure)
            for fig_item in func():
                if isinstance(fig_item, list):
                    fig_list.extend(fig_item)
                else:
                    fig_list.append(fig_item)
        return fig_list

    def create_report(self, layout: str = "combined") -> None:
        """Create final pdf report."""
        fig_list = self.generate_figures()
        fig_list_pbar = tqdm(fig_list, total=len(fig_list), leave=False)
        fig_list_pbar.set_description(f"Creating {layout} pdf report.")
        if layout == "combined":
            with PdfPages(Path(self.save_dir, "report.pdf"), metadata=self.pdf_metadata) as pp_combined:
                for fig in fig_list_pbar:
                    fig_list_pbar.update(1)
                    title = fig.axes[0].get_title()
                    fig.savefig(pp_combined, format="pdf", bbox_inches="tight")
        else:
            for fig in fig_list_pbar:
                title = fig.axes[0].get_title()
                file_name = Path(self.save_dir, f"{title}.pdf")
                with PdfPages(file_name, metadata=self.pdf_metadata) as pp_single:
                    fig.savefig(pp_single, format="pdf", bbox_inches="tight")
                fig_list_pbar.update(1)
        plt.close("all")
