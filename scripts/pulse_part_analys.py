import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import colors
from npeet import entropy_estimators as ee
import ipywidgets as widgets
from IPython.display import display, clear_output


# ========== Compute CMI and plot ==========
def compute_mi_omega_d_noisy_vs_f_matrix(data, k):
    Omega = data["omega_d_noisy"]   # (S, N)
    F     = data["F"]            # (S, N)
    S_num, N = Omega.shape
    S_mat = np.zeros((N, N))
    for i in range(N):
        Omega_i = Omega[:, [i]]
        for j in range(N):
            if i !=j :
                F_j = F[:, [j]]
                val = ee.mi(Omega_i, F_j, k=k)
                S_mat[i, j] = val
            # S_mat[j, i] = val
    return S_mat

def compute_pair_score_Omega_noisy_matrix(data, k):
    Omega = data["Omega_real"]   # (S, N)
    F     = data["F"]            # (S, N)
    S_num, N = Omega.shape
    S_mat = np.zeros((N, N))
    for i in range(N):
        Omega_i    = Omega[:, [i]]
        F_i = F[:, [i]]
        for j in range(N):
            if i != j:
                Omega_j = Omega[:, [j]]
                F_j = F[:, [j]]
                s1 = ee.cmi(Omega_i, Omega_j, F_i, k=k)
                s2 = ee.cmi(Omega_i, Omega_j, F_j, k=k)
                S_mat[i, j] = s1+s2
    return S_mat

# def compute_cmi_omega_real_pair_given_f_matrix(data, k):
#     Omega = data["Omega_real"]   # (S, N)
#     F     = data["F"]            # (S, N)
#     S_num, N = Omega.shape
#     S_mat = np.zeros((N, N))
#     for i in range(N):
#         Omega_i    = Omega[:, [i]]
#         for j in range(i,N):
#             Omega_j = Omega[:, [j]]
#             S_mat[i, j] = ee.cmi(Omega_i, Omega_j, F, k=k)
#     return S_mat

# # ========== Compute CMI and plot ==========
# def compute_mi_omega_real_vs_f_matrix(data, k):
#     Omega = data["Omega_real"]   # (S, N)
#     F     = data["F"]            # (S, N)
#     S_num, N = Omega.shape
#     S_mat = np.zeros((N, N))
#     for i in range(N):
#         Omega_i    = Omega[:, [i]]
#         for j in range(N):
#             if i !=j:
#                 F_j = F[:, [j]]
#                 S_mat[i, j] = ee.mi(Omega_i, F_j, k=k)
#     return S_mat


def build_remapped_cmap(base_cmap, c1, c2, c3, c4, n=256, name_prefix="remap"):
    xs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ys = np.array([0.0, c1, c2, c3, c4, 1.0])
    t = np.linspace(0.0, 1.0, n)
    mapped = np.interp(t, xs, ys)
    colors_list = base_cmap(mapped)
    name = f"{name_prefix}-{base_cmap.name}"
    return colors.LinearSegmentedColormap.from_list(name, colors_list, N=n)


def plot_results(
    S_mat,
    data,
    s_norm=None,
    h_norm=None,
    s_ticks=None,
    h_ticks=None,
    s_cmap=None,
    h_cmap=None,
    font_size=10,
):
    N = S_mat.shape[0]
    # 1) heatmap of CMI
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(S_mat, origin='lower', cmap=s_cmap or 'viridis', norm=s_norm)
    cb1 = plt.colorbar(
        label=r"CMI $S_{ij}\times 10^{-2}$",
        ticks=s_ticks,
        fraction=0.046,
        pad=0.04,
        shrink=0.9,
        location="left",
    )
    cb1.ax.tick_params(labelsize=font_size)
    cb1.set_label(r"CMI $S_{ij}\times 10^{-2}$", fontsize=font_size)
    if s_ticks is not None:
        cb1.set_ticklabels([f"{t*1e2:.2f}" for t in s_ticks])
    plt.xlabel("", fontsize=font_size)
    plt.ylabel("", fontsize=font_size)
    # plt.title("S_ij = I(Omega_i ; Omega_j | F)", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # 2) heatmap of |h_ij| — directly using data["hij"]
    plt.subplot(1,3,2)
    h_eff = data["hij"]  # expected shape (N, N)
    if h_eff.shape != (N, N):
        raise ValueError(f"hij shape {h_eff.shape} and S_mat shape {(N, N)} must match")

    h_eff_abs = np.abs(h_eff)
    plt.imshow(h_eff_abs, origin='lower', cmap=h_cmap or 'magma', norm=h_norm)
    cb2 = plt.colorbar(
        label=r"$|h_{ij}|/\mathrm{GHz}\times 10^{3}$",
        ticks=h_ticks,
        fraction=0.046,
        pad=0.04,
        shrink=0.9,
        location="left",
    )
    cb2.ax.tick_params(labelsize=font_size)
    cb2.set_label(r"$h_{ij}/\mathrm{GHz}\times 10^{3}$", fontsize=font_size)
    if h_ticks is not None:
        cb2.set_ticklabels([f"{t*1e3:.2f}" for t in h_ticks])
    # plt.title("|h_ij|", fontsize=font_size)
    plt.xlabel("", fontsize=font_size)
    plt.ylabel("", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # 3) scatter: |h_ij| vs S_ij
    plt.subplot(1,3,3)
    hij_list = []
    sij_list = []

    for i in range(N):
        for j in range(N):
            if i != j:
                hij_list.append(h_eff_abs[i, j])
                sij_list.append(S_mat[i, j])

    hij_list = np.array(hij_list)
    sij_list = np.array(sij_list)

    plt.scatter(hij_list, sij_list)
    plt.xlabel("|h_ij|", fontsize=font_size)
    plt.ylabel("S_ij", fontsize=font_size)
    plt.title("CMI vs coupling", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.tight_layout()
    plt.show()


def show_interactive_plots(S_mat, data):
    s_min = float(np.min(S_mat))
    s_max = float(np.max(S_mat))
    h_eff_abs = np.abs(data["hij"])
    h_min = float(np.min(h_eff_abs))
    h_max = float(np.max(h_eff_abs))
    s_ticks = [
        s_min,
        s_min + 0.2 * (s_max - s_min),
        s_min + 0.4 * (s_max - s_min),
        s_min + 0.6 * (s_max - s_min),
        s_min + 0.8 * (s_max - s_min),
        s_max,
    ]
    h_ticks = [
        h_min,
        h_min + 0.2 * (h_max - h_min),
        h_min + 0.4 * (h_max - h_min),
        h_min + 0.6 * (h_max - h_min),
        h_min + 0.8 * (h_max - h_min),
        h_max,
    ]

    s_c1 = widgets.FloatSlider(
        value=0.2,
        min=0.0,
        max=1.0,
        step=0.01,
        description="S_ij color stop1",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    s_c2 = widgets.FloatSlider(
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.01,
        description="S_ij color stop2",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    s_c3 = widgets.FloatSlider(
        value=0.6,
        min=0.0,
        max=1.0,
        step=0.01,
        description="S_ij color stop3",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    s_c4 = widgets.FloatSlider(
        value=0.8,
        min=0.0,
        max=1.0,
        step=0.01,
        description="S_ij color stop4",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )

    h_c1 = widgets.FloatSlider(
        value=0.2,
        min=0.0,
        max=1.0,
        step=0.01,
        description="|h_ij| color stop1",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    h_c2 = widgets.FloatSlider(
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.01,
        description="|h_ij| color stop2",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    h_c3 = widgets.FloatSlider(
        value=0.6,
        min=0.0,
        max=1.0,
        step=0.01,
        description="|h_ij| color stop3",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    h_c4 = widgets.FloatSlider(
        value=0.8,
        min=0.0,
        max=1.0,
        step=0.01,
        description="|h_ij| color stop4",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    save_font = widgets.IntText(
        value=10,
        description="save font size",
        layout=widgets.Layout(width="95%"),
    )
    save_name = widgets.Text(
        value="full_figure.png",
        description="output filename",
        layout=widgets.Layout(width="95%"),
    )
    save_btn = widgets.Button(description="Save figure", button_style="success")

    def _update(
        s_c1_val,
        s_c2_val,
        s_c3_val,
        s_c4_val,
        h_c1_val,
        h_c2_val,
        h_c3_val,
        h_c4_val,
    ):
        s_norm = colors.Normalize(vmin=s_min, vmax=s_max)
        h_norm = colors.Normalize(vmin=h_min, vmax=h_max)
        s_cmap = build_remapped_cmap(
            plt.get_cmap("viridis"),
            s_c1_val,
            s_c2_val,
            s_c3_val,
            s_c4_val,
            name_prefix="sij",
        )
        h_cmap = build_remapped_cmap(
            plt.get_cmap("viridis"),
            h_c1_val,
            h_c2_val,
            h_c3_val,
            h_c4_val,
            name_prefix="hij",
        )
        plot_results(
            S_mat,
            data,
            s_norm=s_norm,
            h_norm=h_norm,
            s_ticks=s_ticks,
            h_ticks=h_ticks,
            s_cmap=s_cmap,
            h_cmap=h_cmap,
            font_size=16,
        )

    def _save_all(_):
        s_norm = colors.Normalize(vmin=s_min, vmax=s_max)
        h_norm = colors.Normalize(vmin=h_min, vmax=h_max)
        s_cmap = build_remapped_cmap(
            plt.get_cmap("viridis"),
            s_c1.value,
            s_c2.value,
            s_c3.value,
            s_c4.value,
            name_prefix="sij",
        )
        h_cmap = build_remapped_cmap(
            plt.get_cmap("viridis"),
            h_c1.value,
            h_c2.value,
            h_c3.value,
            h_c4.value,
            name_prefix="hij",
        )

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 3, 1)
        im1 = ax1.imshow(S_mat, origin="lower", cmap=s_cmap, norm=s_norm)
        font = int(save_font.value)
        cb1 = fig.colorbar(
            im1,
            ax=ax1,
            label=r"CMI $S_{ij}\times 10^{-2}$",
            ticks=s_ticks,
            fraction=0.046,
            pad=0.04,
            shrink=0.9,
            location="left",
        )
        cb1.ax.tick_params(labelsize=font)
        cb1.set_label(r"CMI $S_{ij}\times 10^{-2}$", fontsize=font)
        if s_ticks is not None:
            cb1.set_ticklabels([f"{t*1e2:.2f}" for t in s_ticks])
        ax1.set_xlabel("", fontsize=font)
        ax1.set_ylabel("", fontsize=font)
        # ax1.set_title("S_ij = I(Omega_i ; Omega_j | F)", fontsize=16)
        ax1.tick_params(labelsize=font)

        ax2 = fig.add_subplot(1, 3, 2)
        im2 = ax2.imshow(h_eff_abs, origin="lower", cmap=h_cmap, norm=h_norm)
        cb2 = fig.colorbar(
            im2,
            ax=ax2,
            label=r"$h_{ij}\ \mathrm{(MHz)}$",
            ticks=h_ticks,
            fraction=0.046,
            pad=0.04,
            shrink=0.9,
            location="left",
        )
        cb2.ax.tick_params(labelsize=font)
        cb2.set_label(r"$h_{ij}\ \mathrm{(MHz)}$", fontsize=font)
        if h_ticks is not None:
            cb2.set_ticklabels([f"{t*1e3:.2f}" for t in h_ticks])
        # ax2.set_title("|h_ij|", fontsize=16)
        ax2.set_xlabel("", fontsize=font)
        ax2.set_ylabel("", fontsize=font)
        ax2.tick_params(labelsize=font)

        ax3 = fig.add_subplot(1, 3, 3)
        hij_list = []
        sij_list = []
        for i in range(S_mat.shape[0]):
            for j in range(S_mat.shape[1]):
                if i != j:
                    hij_list.append(h_eff_abs[i, j])
                    sij_list.append(S_mat[i, j])
        ax3.scatter(hij_list, sij_list)
        ax3.set_xlabel("|h_ij|", fontsize=font)
        ax3.set_ylabel("S_ij", fontsize=font)
        ax3.set_title("CMI vs coupling", fontsize=font)
        ax3.tick_params(labelsize=font)

        fig.tight_layout()
        fig.savefig(save_name.value, dpi=200)
        plt.close(fig)

    save_btn.on_click(_save_all)

    out = widgets.interactive_output(
        _update,
        {
            "s_c1_val": s_c1,
            "s_c2_val": s_c2,
            "s_c3_val": s_c3,
            "s_c4_val": s_c4,
            "h_c1_val": h_c1,
            "h_c2_val": h_c2,
            "h_c3_val": h_c3,
            "h_c4_val": h_c4,
        },
    )
    display(
        widgets.VBox(
            [
                s_c1,
                s_c2,
                s_c3,
                s_c4,
                h_c1,
                h_c2,
                h_c3,
                h_c4,
                save_font,
                save_name,
                save_btn,
                out,
            ]
        )
    )


# with np.load("test_dataset_gpu_1_8_chain_withF.npz",allow_pickle=True) as f:
# with np.load("dataset2_gpu_1_8_chain_withF.npz", allow_pickle=True) as f:
# with np.load("dataset_gpu_1_8_chain_withF.npz", allow_pickle=True) as f:

# Chain heatmap for the paper
# with np.load("interac_dataset_gpu_1_8_chain_withF-v2.npz",allow_pickle=True) as f: 

# Uncoupled chain for the paper
# with np.load("dataset_nointerac_gpu_1_8_chain_withF.npz",allow_pickle=True) as f:      

# Fully coupled case for the paper



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pulse part analysis")
    parser.add_argument("--npz", required=True, help="dataset npz with F/hij")
    parser.add_argument("--knn", type=int, default=6)
    parser.add_argument("--out1", default="pulse_mi_heatmap.pdf", help="MI omega_d vs F heatmap")
    parser.add_argument("--out2", default="pulse_cmi_heatmap.pdf", help="CMI Omega vs F heatmap")
    parser.add_argument("--scatter", default="pulse_h_vs_cmi.pdf")
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=True) as f:
        data = {k: f[k] for k in f.files}

    num_qubits = data['flux'][0,:].shape[0]
    knn = args.knn
    corre_array = []
    h_array = []
    h = data['hij']
    for k in range(num_qubits):
        flux_k = data['flux'][:,k]
        for l in range(k+1, num_qubits):
            flux_l = data['flux'][:,l]
            romegaq_data = np.stack([data['omega_q_clean'][:,k], data['omega_q_clean'][:,l]], axis=1)
            corre = ee.cmi(flux_k, flux_l, romegaq_data, k=knn)
            corre_array.append(corre)
            h_array.append(h[k, l])

    # scatter
    plt.figure()
    plt.scatter(np.abs(h_array), corre_array, s=40)
    plt.xlabel("|h_ij|")
    plt.ylabel("CMI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.scatter)

    # heatmaps
    S_mat = compute_mi_omega_d_noisy_vs_f_matrix(data, k=knn)
    S2_mat = compute_pair_score_Omega_noisy_matrix(data, k=knn)
    plot_results(S_mat, data)
    plt.savefig(args.out1)
    plt.close()
    plot_results(S2_mat, data)
    plt.savefig(args.out2)
    plt.close()
    print(f"[SAVE] scatter->{args.scatter}, mi_heatmap->{args.out1}, cmi_heatmap->{args.out2}")
