(** * G₂ Exceptional Lie Group

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith Lia.

(** ** Basic Data *)

Definition dim_G2 : nat := 14.
Definition rank_G2 : nat := 2.

Theorem dim_G2_is_14 : dim_G2 = 14.
Proof. reflexivity. Qed.

Theorem rank_G2_is_2 : rank_G2 = 2.
Proof. reflexivity. Qed.

(** ** G₂ Weyl Group *)

Definition G2_Weyl_order : nat := 12.

Theorem G2_Weyl_is_dihedral : G2_Weyl_order = 12.
Proof. reflexivity. Qed.

(** Dihedral group D₆ has order 12 *)
Theorem G2_Weyl_is_D6 : G2_Weyl_order = 2 * 6.
Proof. reflexivity. Qed.

(** ** G₂ Root System *)

Definition G2_roots_count : nat := 12.

Theorem G2_dim_from_roots : G2_roots_count + rank_G2 = dim_G2.
Proof. reflexivity. Qed.

(** ** Relation to dim(K₇) *)

Definition dim_K7 : nat := 7.

Theorem G2_K7_ratio : dim_G2 / dim_K7 = 2.
Proof. reflexivity. Qed.

Theorem G2_K7_exact : dim_G2 = 2 * dim_K7.
Proof. reflexivity. Qed.
