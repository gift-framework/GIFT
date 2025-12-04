(** * Quark Sector Relations

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith Lia.
From GIFT.Algebra Require Import E8WeylGroup.
From GIFT.Geometry Require Import TwistedConnectedSum.
From GIFT.Relations Require Import Constants.

(** ** Strange/Down Mass Ratio m_s/m_d = 20 *)

Theorem ms_md_from_Weyl : 4 * Weyl_factor = m_s_m_d.
Proof. unfold Weyl_factor, m_s_m_d. reflexivity. Qed.

Theorem ms_md_from_b2 : K7_b2 - 1 = m_s_m_d.
Proof.
  unfold K7_b2, quintic_b2, CI222_b2, m_s_m_d.
  reflexivity.
Qed.

Theorem ms_md_value : m_s_m_d = 20.
Proof. reflexivity. Qed.

(** ** Structural Relations *)

Theorem p2_squared_times_Weyl : p2 * p2 * Weyl_factor = 20.
Proof. unfold p2, Weyl_factor. reflexivity. Qed.

Theorem twenty_factorization : 20 = 4 * 5.
Proof. reflexivity. Qed.

Theorem twenty_alt : 20 = 2 * 2 * 5.
Proof. reflexivity. Qed.
