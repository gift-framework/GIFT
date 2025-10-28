# Configuration Twitter Bot - GIFT Framework
# ⚠️ NE JAMAIS COMMITER CE FICHIER AVEC LES VRAIES CLÉS !

# Instructions pour obtenir les clés :
# 1. Allez sur https://developer.twitter.com
# 2. Connectez-vous avec votre compte
# 3. Sélectionnez votre projet/app
# 4. Allez dans "Keys and tokens"
# 5. Copiez les valeurs ci-dessous

# Clés API Twitter/X
TWITTER_API_KEY = "votre_api_key_ici"
TWITTER_API_SECRET = "votre_api_secret_ici"
TWITTER_ACCESS_TOKEN = "votre_access_token_ici"
TWITTER_ACCESS_TOKEN_SECRET = "votre_access_token_secret_ici"
TWITTER_BEARER_TOKEN = "votre_bearer_token_ici"  # Optionnel

# Configuration du bot
BOT_USERNAME = "@votre_compte_twitter"  # Ex: @GIFT_Framework
POSTING_SCHEDULE = "daily"  # daily, weekly, monthly
MAX_TWEETS_PER_DAY = 2
HASHTAGS = ["#GIFT", "#Physics", "#Topology", "#Mathematics"]

# Paramètres de sécurité
RATE_LIMIT_DELAY = 1  # secondes entre les tweets
MAX_RETRIES = 3
DRY_RUN = False  # True pour tester sans poster

# Contenu
CONTENT_TYPES = [
    "precision_achievements",
    "mathematical_beauty", 
    "experimental_predictions",
    "theoretical_insights",
    "comparisons",
    "philosophical",
    "technical_details",
    "call_to_action"
]

# Logs
LOG_LEVEL = "INFO"
LOG_FILE = "twitter_bot.log"
