/-
# GIFT Framework - Lean 4 Formalization

Formal verification of the Geometric Information Field Theory framework,
proving that all 39 observables derive from fixed mathematical structures
with zero continuous adjustable parameters.

## Structure

- **Algebra**: E₈ root system, Weyl group, exceptional Jordan algebra
- **Geometry**: G₂ group, holonomy structures, twisted connected sum
- **Topology**: Betti numbers, cohomology structure
- **Relations**: Physical observables by sector
- **Certificate**: Main theorem and zero-parameter paradigm

## Version: 2.3.0
-/

import GIFT.Algebra.E8RootSystem
import GIFT.Algebra.E8RootsExplicit
import GIFT.Algebra.E8WeylGroup
import GIFT.Algebra.E8Representations
import GIFT.Algebra.ExceptionalJordan

import GIFT.Geometry.G2Group
import GIFT.Geometry.G2Structure
import GIFT.Geometry.G2Holonomy
import GIFT.Geometry.TwistedConnectedSum

import GIFT.Topology.BettiNumbers
import GIFT.Topology.CohomologyStructure
import GIFT.Topology.EulerCharacteristic

import GIFT.Relations.Constants
import GIFT.Relations.GaugeSector
import GIFT.Relations.NeutrinoSector
import GIFT.Relations.QuarkSector
import GIFT.Relations.LeptonSector
import GIFT.Relations.HiggsSector
import GIFT.Relations.CosmologySector

import GIFT.Certificate.ZeroParameter
import GIFT.Certificate.MainTheorem
import GIFT.Certificate.Summary
