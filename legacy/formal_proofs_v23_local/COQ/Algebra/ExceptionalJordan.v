(** * Exceptional Jordan Algebra J₃(𝕆)

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith Lia.
From GIFT.Algebra Require Import E8RootSystem.

(** ** Dimension *)

Definition dim_J3O : nat := 27.

(** 3×3 Hermitian matrices over octonions:
    3 diagonal (real) + 3×8 off-diagonal (octonion) = 3 + 24 = 27 *)

Theorem J3O_dim_decomposition : 3 + 3 * 8 = dim_J3O.
Proof. reflexivity. Qed.

(** ** Relation to E₈ *)

Theorem E8_minus_J3O : dim_E8 - dim_J3O = 221.
Proof. unfold dim_E8, dim_J3O. reflexivity. Qed.

Theorem E8_minus_J3O_factors : dim_E8 - dim_J3O = 13 * 17.
Proof. unfold dim_E8, dim_J3O. reflexivity. Qed.

(** ** Role in tau parameter *)

(** τ denominator involves dim(J₃(𝕆)) × H* = 27 × 99 = 2673 *)
Theorem tau_denominator_component : dim_J3O * 99 = 2673.
Proof. reflexivity. Qed.
