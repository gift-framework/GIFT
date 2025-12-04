(** * Gauge Sector Relations

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith QArith Lia.
From GIFT.Algebra Require Import E8RootSystem E8WeylGroup.
From GIFT.Geometry Require Import G2Group TwistedConnectedSum.
From GIFT.Topology Require Import BettiNumbers.
From GIFT.Relations Require Import Constants.

Open Scope Q_scope.

(** ** Weinberg Angle *)

Theorem sin2_theta_W_from_topology :
  inject_Z (Z.of_nat K7_b2) / inject_Z (Z.of_nat (K7_b3 + dim_G2)) == 3 # 13.
Proof.
  unfold K7_b2, K7_b3, dim_G2, quintic_b2, CI222_b2, quintic_b3, CI222_b3.
  reflexivity.
Qed.

Theorem sin2_theta_W_simplified : 21 # 91 == 3 # 13.
Proof. reflexivity. Qed.

Theorem denominator_91 : 77 + 14 = 91.
Proof. reflexivity. Qed.

Theorem factor_91 : 91 = 7 * 13.
Proof. reflexivity. Qed.

(** ** Strong Coupling Structure *)

Theorem alpha_s_denominator : dim_G2 - p2 = 12.
Proof. unfold dim_G2, p2. reflexivity. Qed.

Theorem twelve_structure : 12 = 8 + 3 + 1.
Proof. reflexivity. Qed.

(** ** Fine Structure Constant Components *)

Theorem alpha_inv_algebraic : (dim_E8 + rank_E8) / 2 = 128.
Proof. unfold dim_E8, rank_E8. reflexivity. Qed.

Theorem alpha_inv_bulk : H_star / 11 = 9.
Proof. unfold H_star, K7_b2, K7_b3, quintic_b2, CI222_b2, quintic_b3, CI222_b3.
       reflexivity. Qed.

Close Scope Q_scope.
