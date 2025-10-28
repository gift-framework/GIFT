# GIFT Twitter Bot - Documentation Complète

## 🎯 Vue d'ensemble

Le GIFT Twitter Bot est un système automatisé qui poste quotidiennement du contenu scientifique sur le framework GIFT (Geometric Information Field Theory). Il génère automatiquement des tweets éducatifs, des résumés hebdomadaires et des highlights mensuels.

## 📁 Structure du projet

```
twitter_bot/
├── config.py                          # Configuration avec clés API (NE PAS COMMITER)
├── config_template.py                 # Template de configuration
├── content_generator_windows.py       # Générateur de contenu (version Windows)
├── content_generator.py               # Générateur de contenu (version complète)
├── twitter_bot.py                     # Script principal du bot
├── scheduler.py                       # Scheduler automatique
├── test_content.py                   # Test du générateur de contenu
├── requirements.txt                   # Dépendances Python
├── api_application_description.txt   # Description pour l'API Twitter
└── README.md                          # Cette documentation
```

## 🔧 Installation et Configuration

### 1. Prérequis

- Python 3.11+
- Compte développeur Twitter/X avec accès Elevated
- Clés API Twitter/X

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 3. Configuration des clés API

1. **Copiez le template de configuration :**
   ```bash
   cp config_template.py config.py
   ```

2. **Éditez `config.py` avec vos clés API :**
   ```python
   # Clés API Twitter/X
   TWITTER_API_KEY = "votre_api_key"
   TWITTER_API_SECRET = "votre_api_secret"
   TWITTER_ACCESS_TOKEN = "votre_access_token"
   TWITTER_ACCESS_TOKEN_SECRET = "votre_access_token_secret"
   TWITTER_BEARER_TOKEN = "votre_bearer_token"
   
   # Clés OAuth 2.0
   TWITTER_CLIENT_ID = "votre_client_id"
   TWITTER_CLIENT_SECRET = "votre_client_secret"
   
   # Configuration du bot
   BOT_USERNAME = "@votre_compte_twitter"
   POSTING_SCHEDULE = "daily"  # daily, weekly, monthly
   MAX_TWEETS_PER_DAY = 2
   DRY_RUN = False  # True pour tester sans poster
   ```

3. **Ajoutez `config.py` à votre `.gitignore` :**
   ```bash
   echo "twitter_bot/config.py" >> .gitignore
   ```

## 🚀 Utilisation

### Test du générateur de contenu

```bash
cd twitter_bot
python content_generator_windows.py
```

### Test du bot (mode dry run)

```bash
cd twitter_bot
# Modifiez DRY_RUN = True dans config.py
python twitter_bot.py
```

### Lancement du scheduler automatique

```bash
cd twitter_bot
python scheduler.py
```

## 📅 Planning des posts

### Configuration par défaut

- **Post quotidien** : 9h00 (contenu aléatoire)
- **Résumé hebdomadaire** : Lundi 10h00
- **Highlight mensuel** : 1er du mois 11h00
- **Limite quotidienne** : 2 tweets maximum

### Types de contenu

1. **Précision** : Statistiques de performance du framework
2. **Beauté mathématique** : Formules et relations topologiques
3. **Prédictions expérimentales** : Tests à venir (DUNE, Euclid, LHC)
4. **Insights théoriques** : Implications philosophiques
5. **Comparaisons** : GIFT vs autres approches
6. **Détails techniques** : Aspects mathématiques avancés
7. **Appels à l'action** : Liens vers documentation et ressources

## 🔑 Clés API requises

### OAuth 1.0a (pour poster des tweets)
- **API Key** (Consumer Key)
- **API Secret Key** (Consumer Secret)
- **Access Token**
- **Access Token Secret**

### OAuth 2.0 (pour l'API v2)
- **Client ID**
- **Client Secret**
- **Bearer Token**

### Comment obtenir les clés

1. Allez sur [developer.twitter.com](https://developer.twitter.com)
2. Connectez-vous avec votre compte
3. Sélectionnez votre projet/app
4. Allez dans l'onglet **"Keys and tokens"**
5. Copiez les clés dans les sections appropriées

## ⚠️ Permissions requises

**Important** : Votre compte développeur doit avoir l'accès **Elevated** pour pouvoir poster des tweets. L'accès Basic ne permet que la lecture.

### Demande d'accès Elevated

1. Allez dans les paramètres de votre app sur developer.twitter.com
2. Demandez l'accès Elevated
3. Justification suggérée :

```
We need Elevated access to post tweets programmatically for our GIFT (Geometric Information Field Theory) research framework bot. The bot will post daily educational content about theoretical physics discoveries, share scientific updates about experimental validation (DUNE, Euclid, LHC), and provide educational threads explaining topological unification. This is purely for scientific education and research dissemination. We will post 1-2 tweets daily with original, factually accurate content about our open-source physics framework.
```

## 🛠️ Personnalisation

### Modifier le contenu

Éditez `content_generator_windows.py` pour :
- Ajouter de nouveaux templates de contenu
- Modifier les faits de la base de données
- Ajuster les hashtags
- Personnaliser les messages

### Modifier le planning

Éditez `config.py` pour :
- Changer la fréquence des posts (`POSTING_SCHEDULE`)
- Ajuster les heures de publication
- Modifier la limite quotidienne (`MAX_TWEETS_PER_DAY`)

### Ajouter de nouveaux types de contenu

1. Ajoutez une nouvelle catégorie dans `_load_content_templates()`
2. Créez une méthode `_generate_[nouveau_type]_post()`
3. Ajoutez la logique dans `generate_daily_content()`

## 📊 Monitoring et logs

### Fichiers de logs

- `twitter_bot.log` : Logs détaillés du bot
- Console : Affichage en temps réel

### Niveaux de log

- **INFO** : Informations générales
- **WARNING** : Avertissements (tweets trop longs, etc.)
- **ERROR** : Erreurs (problèmes d'API, etc.)

## 🔒 Sécurité

### Bonnes pratiques

1. **Ne jamais commiter** le fichier `config.py` avec les vraies clés
2. **Utiliser des variables d'environnement** en production
3. **Limiter les permissions** de l'app Twitter au minimum nécessaire
4. **Surveiller les logs** pour détecter des activités suspectes

### Variables d'environnement (recommandé)

```bash
export TWITTER_API_KEY="votre_api_key"
export TWITTER_API_SECRET="votre_api_secret"
export TWITTER_ACCESS_TOKEN="votre_access_token"
export TWITTER_ACCESS_TOKEN_SECRET="votre_access_token_secret"
```

## 🚨 Dépannage

### Erreurs courantes

1. **401 Unauthorized** : Vérifiez vos clés API
2. **403 Forbidden** : Demandez l'accès Elevated
3. **Rate limit exceeded** : Réduisez la fréquence des posts
4. **Tweet too long** : Le bot tronque automatiquement à 280 caractères

### Tests de diagnostic

```bash
# Test de connexion
python twitter_bot.py

# Test de génération de contenu
python test_content.py

# Test du scheduler
python scheduler.py
```

## 📈 Déploiement en production

### Options de déploiement

1. **Serveur dédié** : VPS ou serveur cloud
2. **GitHub Actions** : Automatisation via CI/CD
3. **Heroku** : Plateforme cloud simple
4. **Docker** : Conteneurisation

### Exemple de déploiement avec GitHub Actions

```yaml
name: GIFT Twitter Bot
on:
  schedule:
    - cron: '0 9 * * *'  # Tous les jours à 9h00 UTC
  workflow_dispatch:

jobs:
  tweet:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          cd twitter_bot
          pip install -r requirements.txt
      - name: Run bot
        env:
          TWITTER_API_KEY: ${{ secrets.TWITTER_API_KEY }}
          TWITTER_API_SECRET: ${{ secrets.TWITTER_API_SECRET }}
          TWITTER_ACCESS_TOKEN: ${{ secrets.TWITTER_ACCESS_TOKEN }}
          TWITTER_ACCESS_TOKEN_SECRET: ${{ secrets.TWITTER_ACCESS_TOKEN_SECRET }}
        run: |
          cd twitter_bot
          python twitter_bot.py
```

## 📞 Support

### Ressources utiles

- [Documentation Twitter API](https://developer.twitter.com/en/docs)
- [Documentation Tweepy](https://docs.tweepy.org/)
- [GIFT Framework Repository](https://github.com/gift-framework/GIFT)

### Contact

Pour des questions ou contributions :
- Ouvrez une issue sur le repository GIFT
- Contactez l'équipe de développement

## 📝 Changelog

### Version 1.0.0 (2025-10-28)
- ✅ Générateur de contenu automatisé
- ✅ Intégration API Twitter/X
- ✅ Scheduler quotidien
- ✅ Support Windows (sans emojis)
- ✅ Documentation complète
- ✅ Tests et validation

---

**Note** : Ce bot est conçu pour promouvoir la recherche scientifique et l'éducation. Respectez les conditions d'utilisation de Twitter/X et les bonnes pratiques de communication scientifique.
