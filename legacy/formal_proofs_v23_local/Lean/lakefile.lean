import Lake
open Lake DSL

package gift where
  version := v!"2.3.0"
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]
  -- Enable Mathlib cache downloads
  moreLinkArgs := #["-L./.lake/packages/mathlib/.lake/build/lib"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.14.0"

-- Main GIFT library
@[default_target]
lean_lib GIFT where
  globs := #[.submodules `GIFT]

-- Individual module targets for faster partial builds
lean_lib «GIFT.Algebra» where
  srcDir := "GIFT"
  roots := #[`Algebra.E8RootSystem, `Algebra.E8WeylGroup,
             `Algebra.E8Representations, `Algebra.ExceptionalJordan]

lean_lib «GIFT.Geometry» where
  srcDir := "GIFT"
  roots := #[`Geometry.G2Group, `Geometry.G2Structure,
             `Geometry.G2Holonomy, `Geometry.TwistedConnectedSum]

lean_lib «GIFT.Topology» where
  srcDir := "GIFT"
  roots := #[`Topology.BettiNumbers, `Topology.CohomologyStructure,
             `Topology.EulerCharacteristic]

lean_lib «GIFT.Relations» where
  srcDir := "GIFT"
  roots := #[`Relations.Constants, `Relations.GaugeSector,
             `Relations.NeutrinoSector, `Relations.QuarkSector,
             `Relations.LeptonSector, `Relations.HiggsSector,
             `Relations.CosmologySector]

lean_lib «GIFT.Certificate» where
  srcDir := "GIFT"
  roots := #[`Certificate.ZeroParameter, `Certificate.MainTheorem,
             `Certificate.Summary]
