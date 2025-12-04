(** * Zero-Parameter Paradigm

    Part of the Coq formalization of Geometric Information Field Theory.
    Version: 2.3.0
*)

From Coq Require Import Arith QArith Lia.
From GIFT.Algebra Require Export E8RootSystem E8WeylGroup E8Representations ExceptionalJordan.
From GIFT.Geometry Require Export G2Group G2Structure G2Holonomy TwistedConnectedSum.
From GIFT.Topology Require Export BettiNumbers CohomologyStructure EulerCharacteristic.
From GIFT.Relations Require Export Constants.

(** ** GIFT Structure Record *)

Record GIFTStructure : Type := mkGIFT {
  (* E₈ data *)
  gs_dim_E8 : nat;
  gs_rank_E8 : nat;
  gs_dim_E8xE8 : nat;
  gs_Weyl_factor : nat;
  gs_dim_J3O : nat;

  (* K₇ data *)
  gs_dim_K7 : nat;
  gs_b2 : nat;
  gs_b3 : nat;

  (* G₂ data *)
  gs_dim_G2 : nat;

  (* Derived *)
  gs_H_star : nat;
  gs_p2 : nat;
  gs_N_gen : nat
}.

(** ** Default GIFT Structure *)

Definition GIFT_default : GIFTStructure := {|
  gs_dim_E8 := 248;
  gs_rank_E8 := 8;
  gs_dim_E8xE8 := 496;
  gs_Weyl_factor := 5;
  gs_dim_J3O := 27;
  gs_dim_K7 := 7;
  gs_b2 := 21;
  gs_b3 := 77;
  gs_dim_G2 := 14;
  gs_H_star := 99;
  gs_p2 := 2;
  gs_N_gen := 3
|}.

(** ** Zero-Parameter Predicate *)

Definition is_zero_parameter (G : GIFTStructure) : Prop :=
  G.(gs_dim_E8) = 248 /\
  G.(gs_rank_E8) = 8 /\
  G.(gs_dim_E8xE8) = 496 /\
  G.(gs_Weyl_factor) = 5 /\
  G.(gs_dim_J3O) = 27 /\
  G.(gs_dim_K7) = 7 /\
  G.(gs_b2) = 21 /\
  G.(gs_b3) = 77 /\
  G.(gs_dim_G2) = 14 /\
  G.(gs_H_star) = 99 /\
  G.(gs_p2) = 2 /\
  G.(gs_N_gen) = 3.

(** ** Verification *)

Theorem GIFT_is_zero_parameter : is_zero_parameter GIFT_default.
Proof.
  unfold is_zero_parameter, GIFT_default.
  repeat split; reflexivity.
Qed.

(** ** Derived Quantities Are Fixed *)

Theorem H_star_derived (G : GIFTStructure) :
  is_zero_parameter G -> G.(gs_H_star) = G.(gs_b2) + G.(gs_b3) + 1.
Proof.
  intros [_ [_ [_ [_ [_ [_ [Hb2 [Hb3 [_ [HH [_ _]]]]]]]]]]].
  rewrite Hb2, Hb3, HH. reflexivity.
Qed.

Theorem p2_derived (G : GIFTStructure) :
  is_zero_parameter G -> G.(gs_p2) = G.(gs_dim_G2) / G.(gs_dim_K7).
Proof.
  intros [_ [_ [_ [_ [_ [HK7 [_ [_ [HG2 [_ [Hp2 _]]]]]]]]]]].
  rewrite HK7, HG2, Hp2. reflexivity.
Qed.
