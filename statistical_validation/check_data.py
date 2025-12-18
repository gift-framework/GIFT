#!/usr/bin/env python3
import json
with open('statistical_validation/results/summary.json', 'r') as f:
    data = json.load(f)

print('Direct JSON values:')
print(f'Sigma separation: {data["statistical_significance"]["sigma_separation"]}')
print(f'Z-score: {data["statistical_significance"]["z_score"]}')
print(f'P-value: {data["statistical_significance"]["p_value"]}')
print(f'GIFT deviation: {data["reference_config"]["mean_deviation_percent"]}')
print(f'Alt mean: {data["alternative_configs"]["mean_deviation_percent"]}')
print(f'Alt std: {data["alternative_configs"]["std_deviation_percent"]}')
print(f'Count: {data["alternative_configs"]["count"]}')



