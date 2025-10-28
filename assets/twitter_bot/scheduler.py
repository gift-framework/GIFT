#!/usr/bin/env python3
"""
GIFT Twitter Bot - Scheduler Automatique
Planifie et exécute les posts Twitter automatiquement
"""

import schedule
import time
import logging
from datetime import datetime
import os
import sys

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from content_generator_en import GIFTContentGenerator
from twitter_bot_v2 import GIFTTwitterBotV2
from config import *

class GIFTScheduler:
    def __init__(self):
        self.logger = self._setup_logging()
        self.content_generator = GIFTContentGenerator()
        self.post_count_today = 0
        self.last_post_date = None
        
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
        return logging.getLogger('GIFTScheduler')
    
    def reset_daily_counter(self):
        """Remet à zéro le compteur quotidien"""
        today = datetime.now().date()
        if self.last_post_date != today:
            self.post_count_today = 0
            self.last_post_date = today
            self.logger.info(f"Compteur quotidien remis à zéro pour {today}")
    
    def can_post_today(self) -> bool:
        """Vérifie si on peut encore poster aujourd'hui"""
        self.reset_daily_counter()
        return self.post_count_today < MAX_TWEETS_PER_DAY
    
    def post_daily_content(self):
        """Poste le contenu quotidien"""
        if not self.can_post_today():
            self.logger.info("Limite quotidienne de tweets atteinte")
            return
        
        try:
            # Utiliser le bot API v2
            bot = GIFTTwitterBotV2()
            success = bot.post_daily_content(dry_run=DRY_RUN)
            
            if success:
                self.post_count_today += 1
                self.logger.info("Tweet poste avec succes")
            else:
                self.logger.error("Erreur lors du post")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la generation du contenu: {e}")
    
    def post_weekly_summary(self):
        """Poste le résumé hebdomadaire"""
        try:
            content = self.content_generator.generate_weekly_summary()
            self.logger.info(f"Résumé hebdomadaire généré: {content[:100]}...")
            
            # Simulation du post
            self.logger.info("SIMULATION: Résumé hebdomadaire posté")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du résumé: {e}")
    
    def post_monthly_highlight(self):
        """Poste le highlight mensuel"""
        try:
            content = self.content_generator.generate_monthly_highlight()
            self.logger.info(f"Highlight mensuel généré: {content[:100]}...")
            
            # Simulation du post
            self.logger.info("SIMULATION: Highlight mensuel posté")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du highlight: {e}")
    
    def _check_monthly_post(self):
        """Vérifie si c'est le 1er du mois pour poster le highlight"""
        today = datetime.now()
        if today.day == 1:
            self.post_monthly_highlight()
    
    def setup_schedule(self):
        """Configure le planning des posts"""
        
        if POSTING_SCHEDULE == "daily":
            # Post quotidien à 9h00
            schedule.every().day.at("09:00").do(self.post_daily_content)
            self.logger.info("Planning quotidien configuré: 9h00")
            
        elif POSTING_SCHEDULE == "weekly":
            # Résumé hebdomadaire le lundi à 10h00
            schedule.every().monday.at("10:00").do(self.post_weekly_summary)
            self.logger.info("Planning hebdomadaire configuré: Lundi 10h00")
            
        elif POSTING_SCHEDULE == "monthly":
            # Highlight mensuel le 1er du mois à 11h00
            schedule.every().day.at("11:00").do(self._check_monthly_post)
            self.logger.info("Planning mensuel configuré: 1er du mois 11h00")
        
        # Toujours programmer le résumé hebdomadaire (en plus du quotidien)
        if POSTING_SCHEDULE != "weekly":
            schedule.every().monday.at("10:00").do(self.post_weekly_summary)
            self.logger.info("Résumé hebdomadaire ajouté: Lundi 10h00")
        
        # Toujours programmer le highlight mensuel
        if POSTING_SCHEDULE != "monthly":
            schedule.every().day.at("11:00").do(self._check_monthly_post)
            self.logger.info("Highlight mensuel ajouté: 1er du mois 11h00")
    
    def run(self):
        """Lance le scheduler"""
        self.logger.info("Démarrage du scheduler GIFT Twitter Bot")
        self.setup_schedule()
        
        self.logger.info("Scheduler démarré. En attente des tâches programmées...")
        self.logger.info(f"Planning configuré: {POSTING_SCHEDULE}")
        self.logger.info(f"Limite quotidienne: {MAX_TWEETS_PER_DAY} tweets")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Vérifier toutes les minutes
                
        except KeyboardInterrupt:
            self.logger.info("Arrêt du scheduler demandé par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur dans le scheduler: {e}")

def main():
    """Fonction principale"""
    print("GIFT Twitter Bot - Scheduler Automatique")
    print("=" * 50)
    
    scheduler = GIFTScheduler()
    
    # Test de génération de contenu
    print("\nTest de génération de contenu:")
    print("-" * 30)
    
    content = scheduler.content_generator.generate_daily_content()
    print(f"Contenu quotidien: {content[:100]}...")
    print(f"Longueur: {len(content)} caractères")
    
    print(f"\nPlanning configuré: {POSTING_SCHEDULE}")
    print(f"Limite quotidienne: {MAX_TWEETS_PER_DAY} tweets")
    
    # Démarrer le scheduler
    print("\nDémarrage du scheduler...")
    print("Appuyez sur Ctrl+C pour arrêter")
    
    scheduler.run()

if __name__ == "__main__":
    main()
