#!/usr/bin/env python3
"""
GIFT Twitter Bot - Main Bot Script
Bot automatisé pour poster du contenu GIFT sur Twitter/X
"""

import tweepy
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Optional
import json

from content_generator import GIFTContentGenerator
from config_template import *

class GIFTTwitterBot:
    def __init__(self, config_file: str = "config.py"):
        """Initialise le bot Twitter avec les clés API"""
        self.logger = self._setup_logging()
        self.content_generator = GIFTContentGenerator()
        self.api = self._setup_twitter_api(config_file)
        self.last_post_time = None
        
    def _setup_logging(self) -> logging.Logger:
        """Configure le système de logs"""
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('GIFTTwitterBot')
    
    def _setup_twitter_api(self, config_file: str) -> tweepy.API:
        """Configure l'API Twitter"""
        try:
            # Charger la configuration
            if os.path.exists(config_file):
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", config_file)
                config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config)
                
                # Configuration OAuth 1.0a (pour poster des tweets)
                auth = tweepy.OAuth1UserHandler(
                    config.TWITTER_API_KEY,
                    config.TWITTER_API_SECRET,
                    config.TWITTER_ACCESS_TOKEN,
                    config.TWITTER_ACCESS_TOKEN_SECRET
                )
                
                # Créer l'API
                api = tweepy.API(auth, wait_on_rate_limit=True)
                
                # Vérifier les credentials
                try:
                    user = api.verify_credentials()
                    self.logger.info(f"✅ Connexion réussie avec @{user.screen_name}")
                    return api
                except tweepy.Unauthorized:
                    self.logger.error("❌ Erreur d'authentification Twitter")
                    raise
                except tweepy.Forbidden:
                    self.logger.error("❌ Accès interdit - vérifiez les permissions")
                    raise
                    
            else:
                self.logger.error(f"❌ Fichier de configuration {config_file} non trouvé")
                raise FileNotFoundError(f"Configuration file {config_file} not found")
                
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la configuration de l'API: {e}")
            raise
    
    def post_tweet(self, content: str, dry_run: bool = False) -> bool:
        """Poste un tweet"""
        try:
            if dry_run:
                self.logger.info(f"🧪 DRY RUN - Tweet qui serait posté:\n{content}")
                return True
            
            # Vérifier la longueur
            if len(content) > 280:
                self.logger.warning(f"⚠️ Tweet trop long ({len(content)} caractères), tronqué")
                content = content[:277] + "..."
            
            # Poster le tweet
            tweet = self.api.update_status(content)
            self.logger.info(f"✅ Tweet posté avec succès: {tweet.id}")
            self.logger.info(f"📝 Contenu: {content[:100]}...")
            
            # Attendre pour respecter les rate limits
            time.sleep(RATE_LIMIT_DELAY)
            
            return True
            
        except tweepy.TooManyRequests:
            self.logger.error("❌ Trop de requêtes - rate limit atteint")
            return False
        except tweepy.Unauthorized:
            self.logger.error("❌ Erreur d'authentification")
            return False
        except tweepy.Forbidden:
            self.logger.error("❌ Tweet interdit - contenu problématique")
            return False
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du post: {e}")
            return False
    
    def post_daily_content(self, dry_run: bool = False) -> bool:
        """Poste le contenu quotidien"""
        try:
            content = self.content_generator.generate_daily_content()
            success = self.post_tweet(content, dry_run)
            
            if success:
                self.last_post_time = datetime.now()
                self.logger.info("✅ Contenu quotidien posté avec succès")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du post quotidien: {e}")
            return False
    
    def post_weekly_summary(self, dry_run: bool = False) -> bool:
        """Poste le résumé hebdomadaire"""
        try:
            content = self.content_generator.generate_weekly_summary()
            success = self.post_tweet(content, dry_run)
            
            if success:
                self.logger.info("✅ Résumé hebdomadaire posté avec succès")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du post hebdomadaire: {e}")
            return False
    
    def post_monthly_highlight(self, dry_run: bool = False) -> bool:
        """Poste le highlight mensuel"""
        try:
            content = self.content_generator.generate_monthly_highlight()
            success = self.post_tweet(content, dry_run)
            
            if success:
                self.logger.info("✅ Highlight mensuel posté avec succès")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du post mensuel: {e}")
            return False
    
    def run_scheduled_post(self, dry_run: bool = False) -> bool:
        """Exécute le post selon le planning configuré"""
        now = datetime.now()
        
        if POSTING_SCHEDULE == "daily":
            return self.post_daily_content(dry_run)
        elif POSTING_SCHEDULE == "weekly" and now.weekday() == 0:  # Lundi
            return self.post_weekly_summary(dry_run)
        elif POSTING_SCHEDULE == "monthly" and now.day == 1:  # 1er du mois
            return self.post_monthly_highlight(dry_run)
        else:
            self.logger.info("ℹ️ Pas de post prévu pour aujourd'hui")
            return True
    
    def test_connection(self) -> bool:
        """Teste la connexion à l'API"""
        try:
            user = self.api.verify_credentials()
            self.logger.info(f"✅ Connexion testée avec succès: @{user.screen_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Test de connexion échoué: {e}")
            return False

def main():
    """Fonction principale"""
    print("GIFT Twitter Bot - Demarrage")
    
    # Mode dry run pour les tests
    dry_run = DRY_RUN
    
    try:
        # Initialiser le bot
        bot = GIFTTwitterBot()
        
        # Tester la connexion
        if not bot.test_connection():
            print("Impossible de se connecter a Twitter")
            return
        
        # Exécuter le post programmé
        success = bot.run_scheduled_post(dry_run)
        
        if success:
            print("Bot execute avec succes")
        else:
            print("Erreur lors de l'execution du bot")
            
    except Exception as e:
        print(f"Erreur fatale: {e}")

if __name__ == "__main__":
    main()
