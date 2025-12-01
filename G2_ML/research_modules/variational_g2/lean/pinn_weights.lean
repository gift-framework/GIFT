/-
  GIFT PINN Weights - Auto-generated

  Total parameters: 568105
  Input dim: 7
  Output dim: 35

  This file provides Lean definitions for the frozen neural network.
-/

namespace GIFT.PINN

-- Network architecture constants
def input_dim : ℕ := 7
def output_dim : ℕ := 35
def num_frequencies : ℕ := 64
def num_layers : ℕ := 4

-- Layer dimensions
def layer_dims : List (ℕ × ℕ) := [
  (128, 256),
  (256, 512),
  (512, 512),
  (512, 256),
]

-- Weights are stored in external JSON file
-- Load with: `json_weights := load_json "weights.json"`

-- Standard G2 form bias (output initialization)
def standard_g2_indices : List (ℕ × ℕ × ℕ) := [
  (0, 1, 2),  -- e^{123}
  (0, 3, 4),  -- e^{145}
  (0, 5, 6),  -- e^{167}
  (1, 3, 5),  -- e^{246}
  (1, 4, 6),  -- e^{257} sign -1
  (2, 3, 6),  -- e^{347} sign -1
  (2, 4, 5),  -- e^{356} sign -1
]

def standard_g2_signs : List ℤ := [1, 1, 1, 1, -1, -1, -1]

end GIFT.PINN
