#!/usr/bin/env python3
import argparse
from pathlib import Path

from causal_q.pipeline import CausalQPipeline


def str2bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes", "y"}


def cmd_gen(args):
    pipe = CausalQPipeline()
    pipe.generate_dataset(
        width=args.width,
        length=args.length,
        samples=args.samples,
        coupling_form=args.coupling,
        flux_loc=args.flux_loc,
        flux_scale=args.flux_scale,
        Omega_rng=(args.Omega_min, args.Omega_max),
        Omega_error_std=args.Omega_err,
        phi_rule=args.phi_rule,
        Tg_rng=(args.Tg_min, args.Tg_max),
        drive_noise_std=args.drive_noise,
        freq_noise_std=args.freq_noise,
        use_torch=not args.cpu,
        compute_F=args.compute_F,
        seed=args.seed,
        out=Path(args.out),
    )


def cmd_pc(args):
    pipe = CausalQPipeline()
    pipe.run_pc_layer1(
        npz_path=Path(args.npz),
        alpha1=args.alpha1,
        max_cond_set=args.max_cond_set,
        use_perm=args.use_perm,
        merge_mode=args.merge_mode,
        out_json=Path(args.out),
        fig_path=Path(args.fig) if args.fig else None,
    )


def cmd_mi(args):
    pipe = CausalQPipeline()
    pipe.compute_mi_on_graph(
        npz_path=Path(args.npz),
        p_json=Path(args.graph),
        knn=args.knn,
        mi_out=Path(args.out),
    )


def cmd_bipart(args):
    pipe = CausalQPipeline()
    pipe.plot_bipartite(
        mi_file=Path(args.mi),
        out_pdf=Path(args.out),
        threshold=args.threshold,
    )


def cmd_hij(args):
    pipe = CausalQPipeline()
    pipe.plot_hij(
        npz_path=Path(args.npz),
        out_pdf=Path(args.out),
    )


def cmd_scatter(args):
    pipe = CausalQPipeline()
    pipe.scatter_hij_mi(
        npz_path=Path(args.npz),
        mi_file=Path(args.mi),
        out_pdf=Path(args.out),
    )


def build_parser():
    p = argparse.ArgumentParser(description="Causal-Q pipeline (self-contained)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-data", help="Generate dataset")
    g.add_argument("--width", type=int, required=True)
    g.add_argument("--length", type=int, required=True)
    g.add_argument("--samples", type=int, default=40000)
    g.add_argument("--coupling", type=str, default="dense")
    g.add_argument("--flux-loc", type=float, default=0.0)
    g.add_argument("--flux-scale", type=float, default=0.25)
    g.add_argument("--Omega-min", type=float, default=0.06)
    g.add_argument("--Omega-max", type=float, default=0.08)
    g.add_argument("--Omega-err", type=float, default=0.005)
    g.add_argument("--phi-rule", type=str, default="zeros")
    g.add_argument("--Tg-min", type=float, default=120.0)
    g.add_argument("--Tg-max", type=float, default=120.0)
    g.add_argument("--drive-noise", type=float, default=0.2)
    g.add_argument("--freq-noise", type=float, default=0.0)
    g.add_argument("--cpu", action="store_true")
    g.add_argument("--compute-F", action="store_true")
    g.add_argument("--seed", type=int, default=43)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_gen)

    pc = sub.add_parser("pc", help="Run layer-1 PC")
    pc.add_argument("--npz", required=True)
    pc.add_argument("--alpha1", type=float, default=0.0)
    pc.add_argument("--max-cond-set", type=int, default=2)
    pc.add_argument("--use-perm", action="store_true")
    pc.add_argument("--merge-mode", choices=["directed", "skeleton"], default="directed")
    pc.add_argument("--out", default="P1_graph.json")
    pc.add_argument("--fig", help="optional layered graph figure")
    pc.set_defaults(func=cmd_pc)

    mi = sub.add_parser("mi", help="Compute MI on P1 edges")
    mi.add_argument("--npz", required=True)
    mi.add_argument("--graph", required=True, help="P1_graph.json")
    mi.add_argument("--knn", type=int, default=3)
    mi.add_argument("--out", default="mi_results.npz")
    mi.set_defaults(func=cmd_mi)

    bi = sub.add_parser("bipart", help="Plot bipartite MI graph")
    bi.add_argument("--mi", required=True)
    bi.add_argument("--threshold", type=float, default=0.01)
    bi.add_argument("--out", default="bipartite_mi.pdf")
    bi.set_defaults(func=cmd_bipart)

    hj = sub.add_parser("hij", help="Plot hij heatmap from dataset")
    hj.add_argument("--npz", required=True)
    hj.add_argument("--out", default="hij_heatmap.pdf")
    hj.set_defaults(func=cmd_hij)

    sc = sub.add_parser("scatter", help="Scatter hij vs MI")
    sc.add_argument("--npz", required=True)
    sc.add_argument("--mi", required=True)
    sc.add_argument("--out", default="hij_vs_mi.pdf")
    sc.set_defaults(func=cmd_scatter)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
