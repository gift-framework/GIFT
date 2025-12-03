(** * E₈ Representations

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith ZArith Lia.
From GIFT.Algebra Require Import E8RootSystem E8WeylGroup.

(** ** E₈×E₈ Product Structure *)

Definition dim_E8xE8 : nat := 496.

Theorem E8xE8_is_double : dim_E8xE8 = 2 * dim_E8.
Proof. unfold dim_E8xE8, dim_E8. reflexivity. Qed.

Theorem E8xE8_sum : dim_E8 + dim_E8 = dim_E8xE8.
Proof. unfold dim_E8xE8, dim_E8. reflexivity. Qed.

(** Rank of product *)
Definition rank_E8xE8 : nat := 16.

Theorem rank_E8xE8_is_double : rank_E8xE8 = 2 * rank_E8.
Proof. unfold rank_E8xE8, rank_E8. reflexivity. Qed.

(** Root count of product *)
Definition roots_E8xE8 : nat := 480.

Theorem roots_E8xE8_is_double : roots_E8xE8 = 2 * E8_roots_count.
Proof. unfold roots_E8xE8, E8_roots_count. reflexivity. Qed.

(** ** Adjoint Representation *)

Theorem adjoint_dim : dim_E8 = 248.
Proof. reflexivity. Qed.

(** ** Structural number 221 *)

Definition structural_221 : nat := 221.

Theorem structural_221_factorization : structural_221 = 13 * 17.
Proof. reflexivity. Qed.
