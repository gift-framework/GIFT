import json

with open('K7_GIFT_v1_3b_rigorous.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Current cells: {len(nb["cells"])}')

part3 = [
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 12. Fixed-Structure RG Flow']},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
        'class RGFlowFixed:\n',
        '    """RG Flow with FIXED structural coefficients."""\n',
        '    def __init__(self, sc, zpp):\n',
        '        self.A = -sc.p2 * sc.dim_G2  # -28\n',
        '        self.B = 0.0\n',
        '        self.C = 2 * (sc.H_star // 11)  # 18\n',
        '        self.D = sc.N_gen / sc.p2  # 1.5\n',
        '    \n',
        '    def compute_beta(self, kappa_T, det_g, div_T, fract):\n',
        '        return self.A * div_T + self.C * (det_g - 1.0) + self.D * fract\n',
        '\n',
        'RG_FIXED = RGFlowFixed(SC, ZPP)\n',
        'print(f"Fixed RG: A={RG_FIXED.A}, C={RG_FIXED.C}, D={RG_FIXED.D}")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 13. Loss Functions']},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
        'def compute_losses(net, geom, coords, zpp, phase):\n',
        '    g, info = geom.compute_metric(net, coords)\n',
        '    phi, det_g, eigvals = info["phi"], info["det"], info["eigvals"]\n',
        '    dphi = exterior_derivative_phi(phi, coords, net)\n',
        '    star_phi = hodge_dual_phi(phi, g)\n',
        '    d_star = exterior_derivative_star_phi(star_phi, coords)\n',
        '    T = torsion_norm_full(dphi, d_star)\n',
        '    T_mean = T.mean()\n',
        '    losses = {}\n',
        '    losses["kappa"] = (T_mean - zpp.kappa_T) ** 2\n',
        '    if T_mean > 0.04:\n',
        '        losses["kappa"] = losses["kappa"] + 5.0 * (T_mean - 0.04) ** 2\n',
        '    losses["closure"] = (dphi ** 2).sum(dim=(1,2,3,4)).mean()\n',
        '    losses["coclosure"] = (d_star ** 2).sum(dim=(1,2,3,4,5)).mean()\n',
        '    losses["det"] = (det_g.mean() - 1.0) ** 2\n',
        '    losses["pos"] = torch.relu(-eigvals.min(dim=-1)[0] + 0.01).mean()\n',
        '    losses["ortho"] = ((phi ** 2).sum(dim=(1,2,3)).mean() - 7.0) ** 2 * 0.01\n',
        '    w = {1:[1,0.5,0.5,0.5,1,0], 2:[1,1,1,0.8,1.5,0], 3:[2,1,1,0.5,1,1], 4:[3,0.5,0.5,1,1,2]}[min(phase,4)]\n',
        '    losses["total"] = w[0]*losses["kappa"] + w[1]*losses["closure"] + w[2]*losses["coclosure"] + w[3]*losses["det"] + w[4]*losses["pos"] + w[5]*losses["ortho"]\n',
        '    losses["T_val"] = T_mean.item()\n',
        '    losses["det_val"] = det_g.mean().item()\n',
        '    losses["kappa_dev"] = abs(T_mean.item() - zpp.kappa_T) / zpp.kappa_T * 100\n',
        '    return losses\n',
        '\n',
        'print("Losses ready")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 14. Checkpointing']},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
        'def save_ckpt(net, opt, phase, epoch, hist, best, cfg, name="ckpt.pt"):\n',
        '    d = Path(cfg["output_dir"]) / "checkpoints"\n',
        '    d.mkdir(exist_ok=True)\n',
        '    torch.save({"phase": phase, "epoch": epoch, "net": net.state_dict(), "opt": opt.state_dict(), "hist": hist, "best": best}, d / name)\n',
        '\n',
        'def load_ckpt(net, opt, cfg):\n',
        '    p = Path(cfg["output_dir"]) / "checkpoints" / "ckpt.pt"\n',
        '    if p.exists():\n',
        '        c = torch.load(p, map_location=device)\n',
        '        net.load_state_dict(c["net"])\n',
        '        opt.load_state_dict(c["opt"])\n',
        '        return c["phase"], c["epoch"], c["hist"], c.get("best", {})\n',
        '    return 1, 0, [], {}\n',
        '\n',
        'def get_lr(epoch, cfg):\n',
        '    w = cfg["warmup_epochs"]\n',
        '    if epoch < w: return cfg["lr_initial"] * (epoch + 1) / w\n',
        '    return max(cfg["lr_min"], cfg["lr_initial"] * 0.95 ** ((epoch - w) // 500))\n',
        '\n',
        'print("Checkpointing ready")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 15. Training Loop']},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
        'def train(net, geom, cfg, zpp):\n',
        '    opt = optim.Adam(net.parameters(), lr=cfg["lr_initial"])\n',
        '    phase0, epoch0, hist, best = load_ckpt(net, opt, cfg)\n',
        '    print("Training v1.3b...")\n',
        '    print("Phase | Epoch | T | kappa_dev% | det | Loss")\n',
        '    for phase in range(phase0, 5):\n',
        '        e0 = epoch0 if phase == phase0 else 0\n',
        '        for epoch in range(e0, cfg["epochs_per_phase"]):\n',
        '            for pg in opt.param_groups: pg["lr"] = get_lr(epoch, cfg)\n',
        '            coords = sample_coords(cfg["batch_size"])\n',
        '            losses = compute_losses(net, geom, coords, zpp, phase)\n',
        '            opt.zero_grad()\n',
        '            losses["total"].backward()\n',
        '            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)\n',
        '            opt.step()\n',
        '            if epoch % cfg["print_every"] == 0:\n',
        '                print(f"  {phase} | {epoch:5d} | {losses[\'T_val\']:.4f} | {losses[\'kappa_dev\']:5.1f}% | {losses[\'det_val\']:.4f} | {losses[\'total\'].item():.2e}")\n',
        '            if epoch % cfg["checkpoint_every"] == 0 and epoch > 0:\n',
        '                save_ckpt(net, opt, phase, epoch, hist, best, cfg)\n',
        '            if losses["kappa_dev"] < best.get("kappa_dev", 999):\n',
        '                best = {"kappa_dev": losses["kappa_dev"], "T": losses["T_val"], "phase": phase, "epoch": epoch}\n',
        '            hist.append({"phase": phase, "epoch": epoch, "loss": losses["total"].item(), "T": losses["T_val"], "det": losses["det_val"], "kappa_dev": losses["kappa_dev"]})\n',
        '        save_ckpt(net, opt, phase+1, 0, hist, best, cfg)\n',
        '    return pd.DataFrame(hist), best\n',
        '\n',
        'print("Training ready")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 16. Initialize and Train']},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
        'phi_net = PhiNet(CONFIG).to(device)\n',
        'geometry = GeometryG2(CONFIG, SC)\n',
        'print(f"Parameters: {sum(p.numel() for p in phi_net.parameters()):,}")\n',
        'history_df, best = train(phi_net, geometry, CONFIG, ZPP)\n',
        'print("Training complete!")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 17. Harmonic Extraction']},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
        'betti = extract_betti_numbers(phi_net, geometry)\n',
        'print("Betti numbers:")\n',
        'print(f"  b2 = {betti[\'b2_eff\']} (target: 21)")\n',
        'print(f"  b3 = {betti[\'b3_eff\']} (target: 77)")\n',
        'print(f"  det(Gram_2) = {betti[\'det_Gram_2\']:.4f}")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 18. Yukawa Predictions']},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
        'bridge = TauScaleBridge(SC, ZPP)\n',
        'bridge.print_predictions()\n',
        'print(f"tau = {ZPP.tau_num}/{ZPP.tau_den} = {ZPP.tau:.6f}")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 19. Export']},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
        'out = Path(CONFIG["output_dir"]) / "exports"\n',
        'out.mkdir(parents=True, exist_ok=True)\n',
        'torch.save({"net": phi_net.state_dict()}, out / "models.pt")\n',
        'with torch.no_grad():\n',
        '    c = sample_coords(1000)\n',
        '    g, info = geometry.compute_metric(phi_net, c)\n',
        'np.save(out / "coords.npy", c.cpu().numpy())\n',
        'np.save(out / "metric.npy", g.cpu().numpy())\n',
        'meta = {"kappa_T": ZPP.kappa_T, "tau": ZPP.tau, "b2": SC.b2_target, "b3": SC.b3_target}\n',
        'with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)\n',
        'history_df.to_csv(out / "history.csv", index=False)\n',
        'print(f"Exports saved to {out}")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 20. Summary']},
    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
        'print("=" * 60)\n',
        'print("K7 GIFT v1.3b RIGOROUS + ZERO-PARAM SUMMARY")\n',
        'print("=" * 60)\n',
        'print(f"kappa_T = 1/61 = {ZPP.kappa_T:.6f}")\n',
        'print(f"tau = {ZPP.tau_num}/{ZPP.tau_den} = {ZPP.tau:.6f}")\n',
        'print(f"sin2_theta_W = {ZPP.sin2_theta_W:.6f}")\n',
        'print(f"alpha_s = {ZPP.alpha_s_MZ:.6f}")\n',
        'print(f"Best kappa deviation: {best.get(\'kappa_dev\', 0):.2f}%")\n',
        'print("=" * 60)'
    ]}
]

nb['cells'].extend(part3)

with open('K7_GIFT_v1_3b_rigorous.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Part 3 added. Total cells: {len(nb["cells"])}')
