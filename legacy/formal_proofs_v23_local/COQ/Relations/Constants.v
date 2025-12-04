(** * GIFT Framework Constants

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith ZArith QArith Reals Lia.
From GIFT.Algebra Require Import E8RootSystem E8WeylGroup E8Representations ExceptionalJordan.
From GIFT.Geometry Require Import G2Group G2Structure TwistedConnectedSum.
From GIFT.Topology Require Import BettiNumbers.

(** ** Derived Structural Constants *)

(** Binary duality p₂ *)
Definition p2 : nat := 2.

Theorem p2_from_G2_K7 : dim_G2 / dim_K7 = p2.
Proof. unfold dim_G2, dim_K7, p2. reflexivity. Qed.

Theorem p2_from_E8xE8 : dim_E8xE8 / dim_E8 = p2.
Proof. unfold dim_E8xE8, dim_E8, p2. reflexivity. Qed.

(** Generation number *)
Definition N_gen : nat := 3.

Theorem N_gen_value : N_gen = 3.
Proof. reflexivity. Qed.

(** ** Rational Constants *)

Open Scope Q_scope.

(** Weinberg angle *)
Definition sin2_theta_W : Q := 3 # 13.

(** Hierarchy parameter τ *)
Definition tau : Q := 3472 # 891.

(** Metric determinant *)
Definition det_g : Q := 65 # 32.

(** Torsion magnitude *)
Definition kappa_T : Q := 1 # 61.

(** Koide parameter *)
Definition Q_Koide : Q := 2 # 3.

Close Scope Q_scope.

(** ** Integer Constants *)

(** CP violation phase *)
Definition delta_CP : nat := 197.

(** Tau-electron mass ratio *)
Definition m_tau_m_e : nat := 3477.

(** Strange-down mass ratio *)
Definition m_s_m_d : nat := 20.

(** Higgs coupling numerator *)
Definition lambda_H_num : nat := 17.
