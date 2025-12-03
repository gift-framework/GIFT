(** * Neutrino Sector Relations

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith QArith Lia.
From GIFT.Algebra Require Import E8RootSystem.
From GIFT.Geometry Require Import G2Group TwistedConnectedSum.
From GIFT.Topology Require Import BettiNumbers.
From GIFT.Relations Require Import Constants.

(** ** CP Violation Phase δ_CP = 197° *)

Theorem delta_CP_formula : 7 * dim_G2 + H_star = delta_CP.
Proof.
  unfold dim_G2, H_star, K7_b2, K7_b3, quintic_b2, CI222_b2, quintic_b3, CI222_b3,
         delta_CP.
  reflexivity.
Qed.

Theorem delta_CP_decomposition : 98 + 99 = delta_CP.
Proof. unfold delta_CP. reflexivity. Qed.

Theorem delta_CP_value : delta_CP = 197.
Proof. reflexivity. Qed.

(** ** Reactor Angle θ₁₃ Structure *)

Theorem theta_13_denominator : K7_b2 = 21.
Proof.
  unfold K7_b2, quintic_b2, CI222_b2. reflexivity.
Qed.

(** ** Atmospheric Angle θ₂₃ Structure *)

Open Scope Q_scope.

Theorem theta_23_fraction :
  inject_Z (Z.of_nat (rank_E8 + K7_b3)) / inject_Z (Z.of_nat H_star) == 85 # 99.
Proof.
  unfold rank_E8, K7_b3, H_star, quintic_b3, CI222_b3, K7_b2, quintic_b2, CI222_b2.
  reflexivity.
Qed.

Theorem theta_23_numerator : 8 + 77 = 85.
Proof. reflexivity. Qed.

Close Scope Q_scope.

(** ** Solar Angle Structure *)

Theorem gamma_GIFT_numerator : 2 * rank_E8 + 5 * H_star = 511.
Proof.
  unfold rank_E8, H_star, K7_b2, K7_b3, quintic_b2, CI222_b2, quintic_b3, CI222_b3.
  reflexivity.
Qed.

Theorem gamma_GIFT_denominator : 10 * dim_G2 + 3 * dim_E8 = 884.
Proof. unfold dim_G2, dim_E8. reflexivity. Qed.

Theorem gamma_denom_factor : 884 = 4 * 221.
Proof. reflexivity. Qed.
