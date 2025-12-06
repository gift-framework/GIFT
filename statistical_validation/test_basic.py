#!/usr/bin/env python3
"""
Basic test script for GIFT statistical validation components.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from advanced_validation import G2Configuration, GIFTPredictionEngine, ExperimentalData

    print("Testing basic components...")

    # Test configuration creation
    config = G2Configuration(name="test", b2=21, b3=77)
    print(f"✓ Configuration created: b2={config.b2}, b3={config.b3}, H*={config.h_star}")

    # Test prediction engine
    engine = GIFTPredictionEngine(config)
    predictions = engine.predict_observables()
    print(f"✓ Predictions computed: {len(predictions)} observables")

    # Test some key predictions
    print(".6f")
    print(".6f")
    print(".6f")
    print(".6f")

    # Test experimental data
    exp_data = ExperimentalData()
    all_obs = exp_data.get_all_observables()
    print(f"✓ Experimental data loaded: {len(all_obs)} observables")

    # Test deviation calculation
    deviations = {}
    for obs_name, pred_value in predictions.items():
        try:
            exp_value, _ = exp_data.get_observable(obs_name)
            rel_dev = abs(pred_value - exp_value) / exp_value * 100
            deviations[obs_name] = rel_dev
        except:
            continue

    mean_dev = sum(deviations.values()) / len(deviations) if deviations else float('inf')
    print(".4f")

    print("\n✅ All basic tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
