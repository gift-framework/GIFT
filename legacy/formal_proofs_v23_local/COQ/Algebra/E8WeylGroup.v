(** * E₈ Weyl Group

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith ZArith Lia.
From GIFT.Algebra Require Import E8RootSystem.

(** ** Weyl Group Order *)

Definition E8_Weyl_order : nat := 696729600.

(** Prime factorization: 2^14 × 3^5 × 5^2 × 7 *)
Definition factor_2_exp : nat := 14.
Definition factor_3_exp : nat := 5.
Definition factor_5_exp : nat := 2.
Definition factor_7_exp : nat := 1.

(** Individual prime powers *)
Definition pow_2_14 : nat := 16384.
Definition pow_3_5 : nat := 243.
Definition pow_5_2 : nat := 25.
Definition pow_7_1 : nat := 7.

(** ** Theorems *)

Theorem pow_2_14_correct : Nat.pow 2 14 = pow_2_14.
Proof. reflexivity. Qed.

Theorem pow_3_5_correct : Nat.pow 3 5 = pow_3_5.
Proof. reflexivity. Qed.

Theorem pow_5_2_correct : Nat.pow 5 2 = pow_5_2.
Proof. reflexivity. Qed.

Theorem Weyl_order_factorization :
  pow_2_14 * pow_3_5 * pow_5_2 * pow_7_1 = E8_Weyl_order.
Proof. reflexivity. Qed.

(** Weyl factor = 5 (base of unique non-trivial perfect square) *)
Definition Weyl_factor : nat := 5.

Theorem Weyl_factor_squared : Weyl_factor * Weyl_factor = pow_5_2.
Proof. reflexivity. Qed.

(** Coxeter number *)
Definition E8_Coxeter : nat := 30.

Theorem E8_Coxeter_value : E8_Coxeter = 30.
Proof. reflexivity. Qed.
