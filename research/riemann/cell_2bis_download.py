# ============================================================
# CELLULE 2bis : TÉLÉCHARGEMENT DIRECT LMFDB
# ============================================================
# Coller cette cellule après la cellule 2 (upload) dans le notebook
# ============================================================

import urllib.request
import json
import os

os.makedirs('data', exist_ok=True)

def download_lmfdb_zeros(url: str, filename: str) -> bool:
    """Télécharge les zéros depuis LMFDB."""
    filepath = f'data/{filename}'
    if os.path.exists(filepath):
        print(f"   ✓ {filename} existe déjà")
        return True

    try:
        print(f"   Téléchargement {filename}...")
        # Add headers to avoid 403
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8')

        with open(filepath, 'w') as f:
            f.write(content)

        print(f"   ✓ {filename} téléchargé")
        return True
    except Exception as e:
        print(f"   ✗ Erreur {filename}: {e}")
        return False

print("="*60)
print("TÉLÉCHARGEMENT DIRECT LMFDB")
print("="*60)

# URLs LMFDB pour les zéros
# Format: https://www.lmfdb.org/L/download_zeros/[label]

LMFDB_ZEROS = {
    # Dirichlet L-functions (degré 1)
    'L_q5.json': 'https://www.lmfdb.org/L/1/5/5.4/r0/0/0/download/zeros',
    'L_q7.json': 'https://www.lmfdb.org/L/1/7/7.6/r1/0/0/download/zeros',
    'L_q11.json': 'https://www.lmfdb.org/L/1/11/11.10/r1/0/0/download/zeros',
    'L_q21.json': 'https://www.lmfdb.org/L/1/21/21.20/r0/0/0/download/zeros',
    'L_q77.json': 'https://www.lmfdb.org/L/1/77/77.76/r0/0/0/download/zeros',
    'L_q248.json': 'https://www.lmfdb.org/L/1/248/248.123/r0/0/0/download/zeros',

    # Ramanujan Delta (forme modulaire poids 12, niveau 1)
    'ramanujan_delta.json': 'https://www.lmfdb.org/L/2/1/1.1/c11/0/0/download/zeros',
}

# Alternative URLs si les premières ne marchent pas
LMFDB_ALT = {
    'L_q5.json': 'https://www.lmfdb.org/L/download_zeros/1-5-5.4-r0-0-0',
    'L_q7.json': 'https://www.lmfdb.org/L/download_zeros/1-7-7.6-r1-0-0',
    'L_q11.json': 'https://www.lmfdb.org/L/download_zeros/1-11-11.10-r1-0-0',
    'L_q77.json': 'https://www.lmfdb.org/L/download_zeros/1-77-77.76-r0-0-0',
    'ramanujan_delta.json': 'https://www.lmfdb.org/L/download_zeros/2-1-1.1-c11-0-0',
}

print("\n--- Dirichlet L-functions ---")
for filename, url in LMFDB_ZEROS.items():
    if not download_lmfdb_zeros(url, filename):
        # Try alternative URL
        if filename in LMFDB_ALT:
            download_lmfdb_zeros(LMFDB_ALT[filename], filename)

# Vérifier ce qui a été téléchargé
print("\n" + "="*60)
print("FICHIERS DISPONIBLES")
print("="*60)

for f in sorted(os.listdir('data')):
    size = os.path.getsize(f'data/{f}')
    print(f"   {f}: {size:,} bytes")

# Charger dans DATA
print("\n--- Chargement dans DATA ---")
for f in os.listdir('data'):
    filepath = f'data/{f}'
    try:
        zeros = load_zeros(filepath)
        if len(zeros) > 0:
            # Extraire le nom
            if 'ramanujan' in f.lower():
                key = 'ramanujan_delta'
            elif f.startswith('L_q'):
                q = f.replace('L_q', '').replace('.json', '').replace('.txt', '')
                key = f'L_q{q}'
            elif f.startswith('1-'):
                parts = f.split('-')
                q = parts[1]
                key = f'L_q{q}'
            elif f.startswith('2-'):
                key = 'ramanujan_delta'
            else:
                key = f.replace('.json', '').replace('.txt', '')

            DATA[key] = zeros
            print(f"   ✓ {key}: {len(zeros)} zéros")
    except Exception as e:
        print(f"   ✗ {f}: {e}")

print(f"\n=== {len(DATA)} datasets chargés ===")
