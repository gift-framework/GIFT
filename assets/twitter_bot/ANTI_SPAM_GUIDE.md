# GIFT Twitter Bot - Configuration Recommandée Anti-Spam

## ⚠️ **IMPORTANT : Éviter le bannissement Twitter**

### **Fréquences recommandées :**

#### **Option 1 : Conservatrice (RECOMMANDÉE)**
```python
POSTING_SCHEDULE = "weekly"  # 1 tweet par semaine
MAX_TWEETS_PER_DAY = 1       # Maximum 1 tweet par jour
```
- **Planning** : Lundi 10h00 (résumé hebdomadaire)
- **Avantage** : Très sûr, pas de risque de spam
- **Inconvénient** : Moins de visibilité

#### **Option 2 : Modérée**
```python
POSTING_SCHEDULE = "daily"   # 1 tweet par jour
MAX_TWEETS_PER_DAY = 1       # Maximum 1 tweet par jour
```
- **Planning** : 9h00 (contenu quotidien)
- **Avantage** : Bonne visibilité
- **Risque** : Peut être considéré comme spam si contenu répétitif

#### **Option 3 : Agressive (DÉCONSEILLÉE)**
```python
POSTING_SCHEDULE = "daily"   # 1 tweet par jour
MAX_TWEETS_PER_DAY = 2       # Maximum 2 tweets par jour
```
- **Risque** : Fort risque de bannissement
- **Déconseillé** : Peut être considéré comme spam

### **Bonnes pratiques anti-spam :**

1. **Variété du contenu** : 8 types différents de contenu
2. **Hashtags limités** : Maximum 4-5 hashtags par tweet
3. **Horaires variés** : Éviter les heures de pointe
4. **Contenu de qualité** : Toujours scientifique et éducatif
5. **Pas de répétition** : Base de données de 10 faits uniques

### **Signaux d'alerte Twitter :**

- **Trop de tweets identiques** → Bannissement
- **Trop de hashtags** → Shadowban
- **Horaires trop réguliers** → Détection bot
- **Contenu non-engagé** → Réduction de portée

### **Recommandation finale :**

**Utilisez l'Option 1 (Conservatrice)** pour commencer :
- 1 tweet par semaine le lundi
- Contenu de haute qualité
- Pas de risque de bannissement
- Possibilité d'augmenter la fréquence plus tard

### **Monitoring :**

Surveillez :
- **Engagement** : Likes, retweets, réponses
- **Portée** : Nombre d'impressions
- **Avertissements** : Notifications Twitter
- **Shadowban** : Diminution soudaine de portée
