#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import kron
from scipy.linalg import expm, sqrtm  # Keep CPU version for comparison / fallback
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# ----------------- Global Torch device & dtype -----------------
if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")  # on A100 this is cuda:0
else:
    TORCH_DEVICE = torch.device("cpu")

TORCH_DTYPE = torch.complex128  # aligned with numpy complex128


# Optional: MI / CMI
try:
    from npeet import entropy_estimators as ee
    _HAVE_NPEET = True
except Exception:
    _HAVE_NPEET = False

# ================== Global Pauli operators (numpy + torch) ==================
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# torch versions (on GPU)
I2_t = torch.eye(2, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
X_t = torch.tensor([[0, 1], [1, 0]], dtype=TORCH_DTYPE, device=TORCH_DEVICE)
Y_t = torch.tensor([[0, -1j], [1j, 0]], dtype=TORCH_DTYPE, device=TORCH_DEVICE)
Z_t = torch.tensor([[1, 0], [0, -1]], dtype=TORCH_DTYPE, device=TORCH_DEVICE)


class UnifiedCalibrationModel:
    """
    Unified model:
      flux -> f0(flux) -> omega_d_clean
           -> f_dyson(flux) -> omega_q
      plus drive noise and coupling to generate RB-style single-qubit fidelity data.

    Added Torch+GPU evolution/fidelity:
      - build_real_H_torch / build_ideal_H_for_qubit_torch
      - simulate_F_list_torch (compute F_fidelity on GPU)
    """

    def __init__(self, column, row, coupling_form="dense", seed=None):
        self.column = column
        self.row = row
        self.N = column * row
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # toy transmon params
        self.E_JS = np.full(self.N, 7.0)
        self.E_C = np.full(self.N, 0.22)
        self.d = np.full(self.N, 0.35)

        # resonator-like ref (mostly unused)
        signs = self.rng.choice([-1, 1], size=self.N)
        self.wr = (self.E_JS - self.E_C) + signs * self.rng.uniform(0.05, 0.1, self.N)

        # coupling matrix hij
        self.hij = self._generate_hij(mode=coupling_form, seed=seed)
        self.coupling_form = coupling_form

        # last computed Dyson spectrum
        self.spectrum = None

    # ================== 工具函数（numpy 版） ==================
    @staticmethod
    def n_qubit_op(single_op, target, N):
        ops = [single_op if i == target else I2 for i in range(N)]
        M = ops[0]
        for i in range(1, N):
            M = kron(M, ops[i])
        return M

    @staticmethod
    def partial_trace_keep_one_qubit(statevec, keep, N):
        psi = statevec.reshape([2] * N)
        axes = [keep] + [i for i in range(N) if i != keep]
        psi_perm = np.transpose(psi, axes).reshape(2, -1)
        rho = psi_perm @ psi_perm.conj().T
        return rho

    @staticmethod
    def state_fidelity(rho, sigma):
        s_rho = sqrtm(rho)
        inner = s_rho @ sigma @ s_rho
        eigvals = np.linalg.eigvalsh(inner)
        eigvals = np.clip(eigvals, 0, None)
        return float((np.sum(np.sqrt(eigvals))) ** 2)

    # ================== 工具函数（Torch+GPU 版） ==================
    @staticmethod
    def n_qubit_op_torch(single_op_t, target, N):
        """
        single_op_t: 2x2 torch 矩阵（在 GPU 上）
        返回: 2^N x 2^N torch 矩阵
        """
        ops = [single_op_t if i == target else I2_t for i in range(N)]
        M = ops[0]
        for i in range(1, N):
            M = torch.kron(M, ops[i])
        return M

    @staticmethod
    def partial_trace_keep_one_qubit_torch(statevec_t, keep, N):
        """
        statevec_t: (2^N,) torch 复数向量（GPU）
        返回: rho (2x2) torch 矩阵
        """
        psi = statevec_t.view(*([2] * N))
        axes = [keep] + [i for i in range(N) if i != keep]
        psi_perm = psi.permute(axes).reshape(2, -1)
        rho = psi_perm @ psi_perm.conj().T
        return rho

    @staticmethod
    def state_fidelity_torch(rho_t, sigma_t):
        """
        Uhlmann fidelity, GPU 版本
        """
        evals, evecs = torch.linalg.eigh(rho_t)
        evals_clamped = torch.clamp(evals, min=0.0)
        sqrt_evals = torch.sqrt(evals_clamped)
        sqrt_rho = (evecs * sqrt_evals.unsqueeze(-2)) @ evecs.conj().T

        inner = sqrt_rho @ sigma_t @ sqrt_rho
        evals_inner, _ = torch.linalg.eigh(inner)
        evals_inner_clamped = torch.clamp(evals_inner, min=0.0)
        fidelity = torch.sum(torch.sqrt(evals_inner_clamped)) ** 2
        return fidelity.real.item()

    def avg_f_i_for_qubit_torch(self, i_target, U_real_t, U_ideal_i_t):
        """
        GPU 版：对 { |0>, |1>, |+>, |+i> } 四个输入态做平均，得到单比特平均保真度。
        """
        N = self.N
        b0 = torch.tensor([1, 0], dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        b1 = torch.tensor([0, 1], dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        norm2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=TORCH_DEVICE))
        bp = (b0 + b1) / norm2
        bpi = (b0 + 1j * b1) / norm2
        test_states = [b0, b1, bp, bpi]

        fids = []
        for psi_i in test_states:
            kets = [(psi_i if q == i_target else b0) for q in range(N)]
            init = kets[0]
            for q in range(1, N):
                init = torch.kron(init, kets[q])

            out_real = U_real_t @ init
            out_ideal = U_ideal_i_t @ init

            rho_real = self.partial_trace_keep_one_qubit_torch(out_real, i_target, N)
            rho_ideal = self.partial_trace_keep_one_qubit_torch(out_ideal, i_target, N)

            fids.append(self.state_fidelity_torch(rho_real, rho_ideal))

        return float(np.mean(fids))

    # ================== hij 生成 & 频率 ==================
    def _generate_hij(self, mode="dense", seed=None, scale=0.03, diag_scale=0.03, periodic=False):
        rng = np.random.default_rng(seed)
        N = self.N
        H = np.zeros((N, N), dtype=float)

        # ---------- 1D 链式耦合 ----------
        if mode == "nointeraction":
            return H
        if mode == "chain":
            for i in range(N - 1):
                w = rng.uniform(0.0, scale)
                H[i,i+1] = 0 # WARNING
            #    H[i, i + 1] = H[i + 1, i] = w
            # H[2, 3] = 0  # 你之前手动断一条边
            np.fill_diagonal(H, 0.0)
            # WARNING
            
            return H

        # ---------- 全连接耦合 ----------
        if mode == "dense":
            scale_dense = 0.02
            A = rng.uniform(0.0, scale_dense, size=(N, N))
            H = 0.5 * (A + A.T)
            np.fill_diagonal(H, 0.0)
            # FIXME 
            # H[0,1] = 0.001
            # H[1,0] = 0.001
            # H[0,2] = 0.02
            # H[2,0] = 0.02
            # H[0,3] = 0.014
            # H[3,0] = 0.014
            # H[1,2] = 0.016
            # H[2,1] = 0.016
            # H[1,3] = 0.008
            # H[3,1] = 0.008
            # H[2,3] = 0.018
            # H[3,2] = 0.018
            return H

        # ---------- 2D 格点耦合 ----------
        def rc(i):
            return divmod(i, self.column)

        def neighbor_index(r, c):
            if periodic:
                r = (r + self.row) % self.row
                c = (c + self.column) % self.column
                return r * self.column + c
            else:
                if 0 <= r < self.row and 0 <= c < self.column:
                    return r * self.column + c
                return None

        scale_grid = 0.02
        diag_scale_grid = 0.02

        for i in range(N):
            r, c = rc(i)
            axial = [neighbor_index(r - 1, c), neighbor_index(r + 1, c),
                     neighbor_index(r, c - 1), neighbor_index(r, c + 1)]
            axial = [j for j in axial if j is not None]

            if mode == "nn4":
                for j in axial:
                    if H[i, j] == 0.0:
                        w = rng.uniform(0.0, scale_grid)
                        H[i, j] = H[j, i] = w
                continue

            if mode == "nn8":
                for j in axial:
                    if H[i, j] == 0.0:
                        w = rng.uniform(0.0, scale_grid)
                        H[i, j] = H[j, i] = w

                diagonals = [neighbor_index(r - 1, c - 1), neighbor_index(r - 1, c + 1),
                             neighbor_index(r + 1, c - 1), neighbor_index(r + 1, c + 1)]
                diagonals = [j for j in diagonals if j is not None]
                for j in diagonals:
                    if H[i, j] == 0.0:
                        w = rng.uniform(0.0, diag_scale_grid)
                        H[i, j] = H[j, i] = w
                continue

        H = 0.5 * (H + H.T)
        np.fill_diagonal(H, 0.0)
        return H

    def _compute_f0(self, fluxes):
        """
        裸频率 f0(flux)（实验上拟合到的比特频率，不含残留耦合），单位 ~ GHz。
        """
        fluxes = np.asarray(fluxes, float)
        return self.E_JS * np.sqrt(np.cos(fluxes) ** 2 + self.d ** 2 * np.sin(fluxes) ** 2) - self.E_C

    def calculate_wi_dyson(self, fluxes, max_iter=50, tol=1e-10, damping=0.7):
        """
        Dyson-like 自洽频率 f_i（真实频率，含残留耦合修正），单位同上 ~ GHz：

            f_i = w0_i + Σ_j h_ij^2 / (f_i - w0_j)
        """
        fluxes = np.asarray(fluxes, float)
        w0 = self._compute_f0(fluxes)
        N = self.N

        H2 = self.hij ** 2
        mask_offdiag = ~np.eye(N, dtype=bool)

        denom0 = w0[:, None] - w0[None, :]
        frac0 = np.zeros_like(denom0)
        frac0[mask_offdiag] = H2[mask_offdiag] / np.where(
            denom0[mask_offdiag] == 0, np.inf, denom0[mask_offdiag]
        )
        f = w0 + np.sum(frac0, axis=1)

        for _ in range(max_iter):
            denom = f[:, None] - w0[None, :]
            denom = np.where(mask_offdiag, denom, np.inf)
            Sigma = np.sum(H2 / denom, axis=1)
            dSigma = -np.sum(H2 / (denom ** 2), axis=1)
            residual = f - w0 - Sigma
            step = residual / (1.0 - dSigma)
            f_new = f - damping * step
            if np.max(np.abs(f_new - f)) < tol:
                f = f_new
                break
            f = f_new

        self.spectrum = f
        return f

    def frequencies_from_flux(self, flux):
        """
        给定 flux（长度 N），返回：
          - omega_q:  真实角频率（Dyson），这里先返回 f_dyson（你原代码 2π 被注释）
          - omega_d:  clean 驱动角频率（bare），同样返回 f0

        注意：这里 **不加噪声**，噪声只在构造真实哈密顿量时加入。
        """
        flux = np.asarray(flux, float)
        f0 = self._compute_f0(flux)
        f_dyson = self.calculate_wi_dyson(flux)

        omega_d_clean = f0
        omega_q_clean = f_dyson
        return omega_q_clean, omega_d_clean

    # ================== RB-style 哈密顿量（numpy 版，保留） ==================
    def build_real_H(self, omega_q_noisy, omega_d_noisy, Omega_d, phi):
        N = self.N
        H = np.zeros((2 ** N, 2 ** N), dtype=complex)
        for i in range(N):
            Delta_i = omega_q_noisy[i] - omega_d_noisy[i]
            H += 0.5 * Delta_i * self.n_qubit_op(Z, i, N)
            H += 0.5 * Omega_d[i] * (
                np.cos(phi[i]) * self.n_qubit_op(X, i, N)
                + np.sin(phi[i]) * self.n_qubit_op(Y, i, N)
            )

        for i in range(N):
            for j in range(i + 1, N):
                h = self.hij[i, j]
                if abs(h) > 0:
                    H += h * (
                        self.n_qubit_op(X, i, N) @ self.n_qubit_op(X, j, N)
                        + self.n_qubit_op(Y, i, N) @ self.n_qubit_op(Y, j, N)
                    )
        return H

    def build_ideal_H_for_qubit(self, i_target, omega_q_clean, omega_d_clean, Omega_d, phi):
        N = self.N
        Delta = omega_q_clean[i_target] - omega_d_clean[i_target]
        H = np.zeros((2 ** N, 2 ** N), dtype=complex)
        H += 0.5 * Delta * self.n_qubit_op(Z, i_target, N)
        H += 0.5 * Omega_d[i_target] * (
            np.cos(phi[i_target]) * self.n_qubit_op(X, i_target, N)
            + np.sin(phi[i_target]) * self.n_qubit_op(Y, i_target, N)
        )
        return H

    def avg_f_i_for_qubit(self, i_target, U_real, U_ideal_i):
        N = self.N
        b0 = np.array([1, 0], dtype=complex)
        b1 = np.array([0, 1], dtype=complex)
        bp = (b0 + b1) / np.sqrt(2)
        bpi = (b0 + 1j * b1) / np.sqrt(2)
        test_states = [b0, b1, bp, bpi]

        fids = []
        for psi_i in test_states:
            kets = [(psi_i if q == i_target else b0) for q in range(N)]
            init = kets[0]
            for q in range(1, N):
                init = kron(init, kets[q])
            out_real = U_real @ init
            out_ideal = U_ideal_i @ init
            rho_real = self.partial_trace_keep_one_qubit(out_real, i_target, N)
            rho_ideal = self.partial_trace_keep_one_qubit(out_ideal, i_target, N)
            fids.append(self.state_fidelity(rho_real, rho_ideal))
        return float(np.mean(fids).real)

    def simulate_F_list(
        self,
        omega_q_clean,
        omega_q_noisy,
        omega_d_clean,
        omega_d_noisy,
        Omega_real,
        Omega_ideal,
        Tg,
        phi=None,
    ):
        """
        CPU+numpy 版本，暂时保留。
        """
        N = self.N
        if phi is None:
            phi = np.zeros(N)
        H_real = self.build_real_H(omega_q_noisy, omega_d_noisy, Omega_real, phi)
        U_real = expm(-1j * H_real * Tg)

        F_list = []
        for i in range(N):
            H_ideal_i = self.build_ideal_H_for_qubit(i, omega_q_clean, omega_d_clean, Omega_ideal, phi)
            U_ideal_i = expm(-1j * H_ideal_i * Tg)
            Fi = self.avg_f_i_for_qubit(i, U_real, U_ideal_i)
            F_list.append(Fi)
        return np.array(F_list)

    # ================== RB-style 哈密顿量 & 保真度（Torch+GPU 版） ==================
    def build_real_H_torch(self, omega_q, omega_d_noisy, Omega_d, phi):
        """
        真实哈密顿量，GPU 版本：
          H_real = Σ_i Δ_i/2 Z_i
                   + Σ_i Ω_i/2 (cosφ_i X_i + sinφ_i Y_i)
                   + Σ_{i<j} h_ij (X_i X_j + Y_i Y_j)
        """
        N = self.N
        dim = 2 ** N

        omega_q_t       = torch.tensor(omega_q,       dtype=torch.float64, device=TORCH_DEVICE)
        omega_d_noisy_t = torch.tensor(omega_d_noisy, dtype=torch.float64, device=TORCH_DEVICE)
        Omega_d_t       = torch.tensor(Omega_d,       dtype=torch.float64, device=TORCH_DEVICE)
        phi_t           = torch.tensor(phi,           dtype=torch.float64, device=TORCH_DEVICE)

        H = torch.zeros((dim, dim), dtype=TORCH_DTYPE, device=TORCH_DEVICE)

        for i in range(N):
            Delta_i = omega_q_t[i] - omega_d_noisy_t[i]
            Zi = self.n_qubit_op_torch(Z_t, i, N)
            Xi = self.n_qubit_op_torch(X_t, i, N)
            Yi = self.n_qubit_op_torch(Y_t, i, N)

            H = H + 0.5 * Delta_i * Zi
            H = H + 0.5 * Omega_d_t[i] * (
                torch.cos(phi_t[i]) * Xi + torch.sin(phi_t[i]) * Yi
            )

        hij_t = torch.tensor(self.hij, dtype=torch.float64, device=TORCH_DEVICE)
        for i in range(N):
            for j in range(i + 1, N):
                h_ij = hij_t[i, j]
                if torch.abs(h_ij) > 0:
                    Xi = self.n_qubit_op_torch(X_t, i, N)
                    Xj = self.n_qubit_op_torch(X_t, j, N)
                    Yi = self.n_qubit_op_torch(Y_t, i, N)
                    Yj = self.n_qubit_op_torch(Y_t, j, N)
                    H = H + h_ij * (Xi @ Xj + Yi @ Yj)

        return H

    def build_ideal_H_for_qubit_torch(self, i_target, omega_q, omega_d_clean, Omega_d, phi):
        """
        理想单比特哈密顿量，GPU 版本。
        """
        N = self.N
        dim = 2 ** N

        omega_q_t        = torch.tensor(omega_q,        dtype=torch.float64, device=TORCH_DEVICE)
        omega_d_clean_t  = torch.tensor(omega_d_clean,  dtype=torch.float64, device=TORCH_DEVICE)
        Omega_d_t        = torch.tensor(Omega_d,        dtype=torch.float64, device=TORCH_DEVICE)
        phi_t            = torch.tensor(phi,            dtype=torch.float64, device=TORCH_DEVICE)

        Delta = omega_q_t[i_target] - omega_d_clean_t[i_target]

        H = torch.zeros((dim, dim), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        Zi = self.n_qubit_op_torch(Z_t, i_target, N)
        Xi = self.n_qubit_op_torch(X_t, i_target, N)
        Yi = self.n_qubit_op_torch(Y_t, i_target, N)

        H = H + 0.5 * Delta * Zi
        H = H + 0.5 * Omega_d_t[i_target] * (
            torch.cos(phi_t[i_target]) * Xi + torch.sin(phi_t[i_target]) * Yi
        )
        return H

    def simulate_F_list_torch(
        self,
        omega_q_clean,
        omega_q_noisy,
        omega_d_clean,
        omega_d_noisy,
        Omega_real,
        Omega_ideal,
        Tg,
        phi=None,
    ):
        """
        在 GPU 上计算所有比特的单比特平均保真度 F_i：

          - 真实哈密顿量：omega_d_noisy = omega_d_clean + N(0, drive_noise_std)
          - 理想哈密顿量：用 omega_d_clean（不加噪声）
          - U_real, U_ideal_i, partial trace, fidelity 全在 A100 上跑
        """
        N = self.N
        if phi is None:
            phi = np.zeros(N)
        # 真实演化
        H_real_t = self.build_real_H_torch(omega_q_noisy, omega_d_noisy, Omega_real, phi)
        U_real_t = torch.linalg.matrix_exp(-1j * H_real_t * Tg)

        F_list = []
        for i in range(N):
            H_ideal_i_t = self.build_ideal_H_for_qubit_torch(i, omega_q_clean, omega_d_clean, Omega_ideal, phi)
            U_ideal_i_t = torch.linalg.matrix_exp(-1j * H_ideal_i_t * Tg)
            Fi = self.avg_f_i_for_qubit_torch(i, U_real_t, U_ideal_i_t)
            F_list.append(Fi)

        return np.array(F_list, dtype=float)

    # ================== 数据集生成 ==================
    def _sample_flux(self, dist):
        kind, params = dist
        if kind == "normal":
            loc = params.get("loc", 0.0)
            scale = params.get("scale", 0.2)
            return self.rng.normal(loc, scale, size=self.N)
        elif kind == "uniform":
            a = params.get("low", -0.4)
            b = params.get("high", 0.4)
            return self.rng.uniform(a, b, size=self.N)
        else:
            raise ValueError(f"Unsupported flux_dist kind: {kind}")

    def generate_dataset(
        self,
        samples=128,
        flux_dist=("normal", {"loc": 0.0, "scale": 0.20}),
        Omega_rng=(0.02, 0.08),
        Omega_error_std=0.008,
        phi_rule="zeros",
        Tg_rng=(120.0, 120.0),
        drive_noise_std=0.0,
        freq_noise_std=0.0,
        use_torch=True,
        compute_F = True,
    ):
        
        """
        生成数据集：

        对每个样本 s：
          1) 采样 flux_s
          2) 由 frequencies_from_flux 得到：
             - omega_d_clean
             - omega_q
          3) 采样 Ω_ideal, Ω_real, φ, Tg
          4) 根据 use_torch:
             - True:  调用 simulate_F_list_torch（A100 上算 F）
             - False: 调用 simulate_F_list（CPU+numpy）
        """
        N = self.N

        Omega_all_real = np.zeros((samples, N))
        Omega_all_ideal = np.zeros((samples, N))
        phi_all = np.zeros((samples, N))
        omegad_noisy_all = np.zeros((samples, N))  # Noisy omega_d
        omegad_clean_all = np.zeros((samples, N))  # Clean omega_d
        omegaq_noisy_all = np.zeros((samples, N))  # Real omega_q
        omegaq_clean_all = np.zeros((samples, N))  # Noisy omega_q
        flux_all = np.zeros((samples, N))
        Tg_all = np.zeros((samples,))
        # F_all = np.zeros((samples, N))
        F_all = np.zeros((samples, N)) if compute_F else None

        for s in tqdm(range(samples), desc="Generating dataset"):
            flux = self._sample_flux(flux_dist)
            omega_q_clean, omega_d_clean = self.frequencies_from_flux(flux)

            mu_Omega = (Omega_rng[0] + Omega_rng[1]) / 2
            sigma_Omega = (Omega_rng[1] - Omega_rng[0]) / 6
            Omega_ideal = self.rng.normal(mu_Omega, sigma_Omega, size=N)
            Omega_real = Omega_ideal + self.rng.normal(0.0, Omega_error_std, size=N)

            # 真实驱动频率：加入噪声
            if drive_noise_std > 0.0:
                omega_d_noisy = omega_d_clean + self.rng.normal(0.0, drive_noise_std, size=N)
            else:
                omega_d_noisy = omega_d_clean

            if freq_noise_std > 0.0:
                omega_q_noisy = omega_q_clean + self.rng.normal(0.0, freq_noise_std, size=N)
            else:
                omega_q_noisy = omega_q_clean
            if phi_rule == "zeros":
                PHI = np.zeros(N)
            elif phi_rule == "uniform":
                PHI = self.rng.uniform(0.0, 2 * np.pi, size=N)
            else:
                raise ValueError(f"Unsupported phi_rule: {phi_rule}")

            Tg = self.rng.uniform(*Tg_rng)
            if compute_F:
                if use_torch:
                    F = self.simulate_F_list_torch(
                        omega_q_clean,
                        omega_q_noisy,
                        omega_d_clean,
                        omega_d_noisy,
                        Omega_real,
                        Omega_ideal,
                        Tg,
                        PHI
                    )
                else:
                    F = self.simulate_F_list(
                        omega_q_clean,
                        omega_q_noisy,
                        omega_d_clean,
                        omega_d_noisy,
                        Omega_real,
                        Omega_ideal,
                        Tg,
                        PHI
                    )
            # 记录本次采样的驱动幅度与相位
            Omega_all_real[s] = Omega_real
            Omega_all_ideal[s] = Omega_ideal
            flux_all[s] = flux
            omegad_noisy_all[s] = omega_d_noisy
            omegad_clean_all[s] = omega_d_clean
            omegaq_noisy_all[s] = omega_q_noisy
            omegaq_clean_all[s] = omega_q_clean
            phi_all[s] = PHI
            Tg_all[s] = Tg
            if compute_F:
                F_all[s] = F

        data = dict(
            flux=flux_all,
            omega_d_noisy=omegad_noisy_all,               # Now storing noisy omega_d here
            omega_d_clean=omegad_clean_all,         # Added clean omega_d here
            omega_q_noisy=omegaq_noisy_all,           # Added real omega_q here
            omega_q_clean=omegaq_clean_all,         # Added noisy omega_q here
            Omega_real=Omega_all_real,
            Omega_ideal=Omega_all_ideal,
            phi=phi_all,
            Tg=Tg_all,
            hij=self.hij.copy(),
            grid=(self.column, self.row),
            coupling_form=self.coupling_form,
            meta=dict(
                Omega_rng=Omega_rng,
                Omega_error_std=Omega_error_std,
                phi_rule=phi_rule,
                Tg_rng=Tg_rng,
                flux_dist=flux_dist,
                drive_noise_std=drive_noise_std,
                seed=self.seed,
                use_torch=use_torch,
            ),
        )
        if compute_F:
            data["F"] = F_all
        return data

    def save_dataset(self, data, filepath, compressed=True):
        """
        保存 generate_dataset 返回的字典到 .npz 文件。
        """
        saver = np.savez_compressed if compressed else np.savez
        payload = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                payload[key] = value
            else:
                payload[key] = np.array(value, dtype=object)
        saver(filepath, **payload)


if __name__ == "__main__":
    # 小测试：在 A100 上计算 F_fidelity
    print("Using torch device:", TORCH_DEVICE)

    model = UnifiedCalibrationModel(
        column=4,
        row=4,
        # coupling_form="nn8",   # 触发 chain 模式
        coupling_form='dense',
        seed=43,
    )

    data = model.generate_dataset(
        samples=40000,
        flux_dist=("normal", {"loc": 0.0, "scale": 0.25}),
        Omega_rng=(0.06, 0.08),
        Omega_error_std=0.005,   # 可设置为 0
        phi_rule="zeros",
        Tg_rng=(120, 120.0),
        drive_noise_std=0.2, #可设置为0
        freq_noise_std=0.000,
        # drive_noise_std=0,
        use_torch=True,# <== 开启 GPU 版 F_fidelity 计算
        compute_F = False
    )
    
    model.save_dataset(data, "fullinterac_dataset_gpu_4_4_dense_withoutF.npz")
    print("Saved dataset to demo_dataset_gpu.npz")
