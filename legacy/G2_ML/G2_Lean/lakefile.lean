import Lake
open Lake DSL

package gift_banach where
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.14.0"

@[default_target]
lean_lib GIFTBanach where
  globs := #[.submodules `GIFT]
