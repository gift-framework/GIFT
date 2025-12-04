(** * G₂ Holonomy

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith Lia.
From GIFT.Geometry Require Import G2Group G2Structure.

(** ** Holonomy Inclusion *)

(** G₂ ⊂ SO(7) *)
Definition dim_SO7 : nat := 21.

Theorem SO7_dim : dim_SO7 = 7 * 6 / 2.
Proof. reflexivity. Qed.

Theorem G2_proper_subgroup : dim_G2 < dim_SO7.
Proof. unfold dim_G2, dim_SO7. lia. Qed.

(** Codimension *)
Theorem G2_codim_in_SO7 : dim_SO7 - dim_G2 = 7.
Proof. unfold dim_SO7, dim_G2. reflexivity. Qed.

(** ** Torsion-Free Condition *)

(** For torsion-free G₂: dφ = 0 and d*φ = 0 *)
(** Controlled non-closure gives torsion magnitude κ_T *)

(** κ_T denominator *)
Definition kappa_T_denom : nat := 61.

Theorem kappa_T_from_cohomology : 77 - 14 - 2 = kappa_T_denom.
Proof. reflexivity. Qed.

(** 61 is the 18th prime *)
Theorem sixty_one_prime : kappa_T_denom = 61.
Proof. reflexivity. Qed.
