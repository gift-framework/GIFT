# Configuration Twitter Bot - GIFT Framework
# ⚠️ NE JAMAIS COMMITER CE FICHIER AVEC LES VRAIES CLÉS !

# Clés API Twitter/X
TWITTER_API_KEY = "819ya6jHwi50q1jnv5kvEldXD"
TWITTER_API_SECRET = "qWRL2PEukI5RtXX0Udzc1GeEyIrNehpU6LfkaQG6rkgQZ0reVw"
TWITTER_ACCESS_TOKEN = "1974863469351784449-2Vp8qICVfBeTg6ICfLjIkhYd8gIvvo"
TWITTER_ACCESS_TOKEN_SECRET = "5zEFdOiqUM1GuSTCDBPLyIWMjozjFwuMVOUV9hv5awsmI"
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIhS5AEAAAAATEC6Nxfdm5mhi3OAfaVLozI%2FgGo%3Dt8e9ZtsKqcJom8poQa0saVNKnswh79i1WwLSsO98tisN92Wgbz"

# Clés OAuth 2.0 (pour l'API v2)
TWITTER_CLIENT_ID = "TElsaFFneFYyNDgyTUpXaGEyWEg6MTpjaQ"
TWITTER_CLIENT_SECRET = "eO2WFmgMjkt_WmUUyH3B00CXSQKO6hOqDhJWY3k2PhFuVUbS5q"

# Configuration du bot
BOT_USERNAME = "@GIFT_Framework"  # Changez selon votre nom d'utilisateur Twitter
POSTING_SCHEDULE = "weekly"  # daily, weekly, monthly (RECOMMANDE: weekly)
MAX_TWEETS_PER_DAY = 1  # Maximum 1 tweet par jour pour éviter le spam
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
LOG_FILE = "twitter_bot.log"  # Sera créé dans le dossier du bot
