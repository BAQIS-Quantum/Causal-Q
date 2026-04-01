import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from scipy.stats import spearmanr

from .utils import load_module


class CausalQPipeline:
    """
    Orchestrates the copied scripts in CausCal-Q/scripts:
      - unified_calibration_model_v4_GPU.py (data generation)
      - pc_analysis_with_unified_model-v5.py (PC)
      - load_directed_plot.py (MI on edges)
      - plot_bipart_graph.py (bipartite plot)
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.scripts_dir = self.project_root / "scripts"
        self._mods: Dict[str, Any] = {}

    def _mod(self, key: str, filename: str):
        if key not in self._mods:
            self._mods[key] = load_module(key, self.scripts_dir / filename)
        return self._mods[key]

    # -------- data generation --------
    def generate_dataset(
        self,
        width: int,
        length: int,
        samples: int,
        coupling_form: str = "dense",
        flux_loc: float = 0.0,
        flux_scale: float = 0.25,
        Omega_rng: Tuple[float, float] = (0.06, 0.08),
        Omega_error_std: float = 0.005,
        phi_rule: str = "zeros",
        Tg_rng: Tuple[float, float] = (120.0, 120.0),
        drive_noise_std: float = 0.2,
        freq_noise_std: float = 0.0,
        use_torch: bool = True,
        compute_F: bool = False,
        seed: Optional[int] = 43,
        out: Path = Path("dataset.npz"),
    ) -> Path:
        um = self._mod("unified_calib", "unified_calibration_model_v4_GPU.py")
        model = um.UnifiedCalibrationModel(
            column=width,
            row=length,
            coupling_form=coupling_form,
            seed=seed,
        )
        data = model.generate_dataset(
            samples=samples,
            flux_dist=("normal", {"loc": flux_loc, "scale": flux_scale}),
            Omega_rng=Omega_rng,
            Omega_error_std=Omega_error_std,
            phi_rule=phi_rule,
            Tg_rng=Tg_rng,
            drive_noise_std=drive_noise_std,
            freq_noise_std=freq_noise_std,
            use_torch=use_torch,
            compute_F=compute_F,
        )
        model.save_dataset(data, str(out))
        return Path(out)

    # -------- PC (Layer 1 by default) --------
    def run_pc_layer1(
        self,
        npz_path: Path,
        alpha1: float = 0.0,
        max_cond_set: int = 2,
        use_perm: bool = False,
        merge_mode: str = "directed",
        out_json: Path = Path("P1_graph.json"),
        fig_path: Optional[Path] = None,
    ):
        pc = self._mod("pc_v5", "pc_analysis_with_unified_model-v5.py")
        with np.load(npz_path, allow_pickle=True) as f:
            data = {k: f[k] for k in f.files}

        P_total, _ = pc.run_pc_pipeline(
            data,
            run_layers=("P1",),
            merge_mode=merge_mode,
            alpha1=alpha1,
            max_cond_set=max_cond_set,
            use_perm=use_perm,
            verbose=False,
        )
        pc.save_P_json(P_total, str(out_json))

        if fig_path:
            fig = pc.plot_layered_causal_graph(P_total)
            fig.savefig(fig_path, bbox_inches="tight")
        return P_total

    # -------- MI on edges --------
    def compute_mi_on_graph(
        self,
        npz_path: Path,
        p_json: Path,
        knn: int = 3,
        mi_out: Path = Path("mi_results.npz"),
    ):
        ldp = self._mod("ldp", "load_directed_plot.py")
        with np.load(npz_path, allow_pickle=True) as f:
            data = {k: f[k] for k in f.files}
        P_total = ldp.load_P_json(str(p_json))
        var_map = {
            "Phi": "flux",
            "omega": "omega_q_clean",
            "Omega": "Omega_real",
            "omega_d": "omega_d_noisy",
            "F": "F",
        }
        mi_results = ldp.compute_flux_f_mi_on_edges(P_total, data, var_map, knn=knn)
        meta = {"npz_path": str(npz_path), "p_json": str(p_json), "knn": knn}
        ldp.save_mi_results_npz(mi_results, path=str(mi_out), meta=meta)
        return mi_results

    # -------- Bipartite plot --------
    def plot_bipartite(self, mi_file: Path, out_pdf: Path = Path("bipartite_mi.pdf"), threshold: float = 0.01):
        bp = self._mod("bipart", "plot_bipart_graph.py")
        mi_results, _meta = bp.load_mi_results(str(mi_file))
        fig = bp.fig
        bp.plot_bipartite(mi_results, threshold=threshold)
        fig.savefig(out_pdf, bbox_inches="tight")
        return out_pdf

    # -------- hij heatmap --------
    def plot_hij(self, npz_path: Path, out_pdf: Path = Path("hij_heatmap.pdf")):
        import matplotlib.pyplot as plt

        with np.load(npz_path, allow_pickle=True) as f:
            hij = np.asarray(f["hij"])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.abs(hij), origin="lower", cmap="magma")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(out_pdf, bbox_inches="tight")
        return out_pdf

    # -------- scatter hij vs MI --------
    def scatter_hij_mi(self, npz_path: Path, mi_file: Path, out_pdf: Path = Path("hij_vs_mi.pdf")):
        import matplotlib.pyplot as plt

        with np.load(npz_path, allow_pickle=True) as f:
            hij = np.abs(np.asarray(f["hij"]))
        mi_results, _meta = self._mod("bipart", "plot_bipart_graph.py").load_mi_results(str(mi_file))

        hij_vals = []
        mi_vals = []
        for item in mi_results:
            i = int(item["src_idx"])
            j = int(item["dst_idx"])
            if hij[i, j] == 0:
                continue
            hij_vals.append(hij[i, j])
            mi_vals.append(float(item["MI"]))

        hij_vals = np.asarray(hij_vals)
        mi_vals = np.asarray(mi_vals)
        rho, p = spearmanr(hij_vals, mi_vals)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(hij_vals * 1e3, mi_vals, s=18)
        ax.set_xlabel("|h_ij| (MHz)")
        ax.set_ylabel("MI (nats)")
        ax.set_title(f"Spearman rho={rho:.3f}, p={p:.3g}")
        fig.savefig(out_pdf, bbox_inches="tight")
        return rho, p
