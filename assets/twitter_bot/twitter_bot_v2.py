#!/usr/bin/env python3
"""
GIFT Twitter Bot - Version API v2 (OAuth 2.0)
Version alternative utilisant l'API v2 de Twitter/X
"""

import tweepy
import logging
import time
import os
from datetime import datetime
from typing import Optional
import json

from content_generator_en import GIFTContentGenerator
from config import *

class GIFTTwitterBotV2:
    def __init__(self, config_file: str = "config.py"):
        """Initialise le bot Twitter avec l'API v2"""
        self.logger = self._setup_logging()
        self.content_generator = GIFTContentGenerator()
        self.client = self._setup_twitter_client(config_file)
        
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
        return logging.getLogger('GIFTTwitterBotV2')
    
    def _setup_twitter_client(self, config_file: str) -> tweepy.Client:
        """Configure le client Twitter API v2"""
        try:
            # Charger la configuration
            if os.path.exists(config_file):
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", config_file)
                config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config)
                
                # Configuration OAuth 2.0 (pour l'API v2)
                client = tweepy.Client(
                    bearer_token=config.TWITTER_BEARER_TOKEN,
                    consumer_key=config.TWITTER_API_KEY,
                    consumer_secret=config.TWITTER_API_SECRET,
                    access_token=config.TWITTER_ACCESS_TOKEN,
                    access_token_secret=config.TWITTER_ACCESS_TOKEN_SECRET,
                    wait_on_rate_limit=True
                )
                
                # Vérifier les credentials
                try:
                    user = client.get_me()
                    self.logger.info(f"Connexion reussie avec @{user.data.username}")
                    return client
                except tweepy.Unauthorized:
                    self.logger.error("Erreur d'authentification Twitter")
                    raise
                except tweepy.Forbidden:
                    self.logger.error("Acces interdit - verifiez les permissions")
                    raise
                    
            else:
                self.logger.error(f"Fichier de configuration {config_file} non trouve")
                raise FileNotFoundError(f"Configuration file {config_file} not found")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration de l'API: {e}")
            raise
    
    def post_tweet(self, content: str, dry_run: bool = False) -> bool:
        """Poste un tweet avec l'API v2"""
        try:
            if dry_run:
                self.logger.info(f"DRY RUN - Tweet qui serait poste:\n{content}")
                return True
            
            # Vérifier la longueur
            if len(content) > 280:
                self.logger.warning(f"Tweet trop long ({len(content)} caracteres), tronque")
                content = content[:277] + "..."
            
            # Poster le tweet avec l'API v2
            tweet = self.client.create_tweet(text=content)
            self.logger.info(f"Tweet poste avec succes: {tweet.data['id']}")
            self.logger.info(f"Contenu: {content[:100]}...")
            
            # Attendre pour respecter les rate limits
            time.sleep(RATE_LIMIT_DELAY)
            
            return True
            
        except tweepy.TooManyRequests:
            self.logger.error("Trop de requetes - rate limit atteint")
            return False
        except tweepy.Unauthorized:
            self.logger.error("Erreur d'authentification")
            return False
        except tweepy.Forbidden:
            self.logger.error("Tweet interdit - contenu problematique")
            return False
        except Exception as e:
            self.logger.error(f"Erreur lors du post: {e}")
            return False
    
    def post_daily_content(self, dry_run: bool = False) -> bool:
        """Poste le contenu quotidien"""
        try:
            content = self.content_generator.generate_daily_content()
            success = self.post_tweet(content, dry_run)
            
            if success:
                self.logger.info("Contenu quotidien poste avec succes")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erreur lors du post quotidien: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Teste la connexion à l'API"""
        try:
            user = self.client.get_me()
            self.logger.info(f"Connexion testee avec succes: @{user.data.username}")
            return True
        except Exception as e:
            self.logger.error(f"Test de connexion echoue: {e}")
            return False

def main():
    """Fonction principale"""
    print("GIFT Twitter Bot V2 - Demarrage")
    
    # Mode dry run pour les tests
    dry_run = DRY_RUN
    
    try:
        # Initialiser le bot
        bot = GIFTTwitterBotV2()
        
        # Tester la connexion
        if not bot.test_connection():
            print("Impossible de se connecter a Twitter")
            return
        
        # Exécuter le post programmé
        success = bot.post_daily_content(dry_run)
        
        if success:
            print("Bot execute avec succes")
        else:
            print("Erreur lors de l'execution du bot")
            
    except Exception as e:
        print(f"Erreur fatale: {e}")

if __name__ == "__main__":
    main()
