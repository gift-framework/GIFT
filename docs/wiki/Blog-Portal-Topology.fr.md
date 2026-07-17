---
title: "Blog : Le jour où Portal t'a appris que l'espace peut avoir des trous"
layout: default
---

> Version originale en anglais sur [arithmon.substack.com](https://arithmon.substack.com/p/episode-9-the-day-portal-taught-you)

# Épisode 9 : Le jour où Portal t'a appris que l'espace peut avoir des trous

*Portal, la topologie multi-connectée, et le drôle de problème de l'espace lui-même.*

---

Tu es dans une salle blanche stérile d'Aperture Science. Devant toi, deux murs. Sur le mur de gauche, un portail orange rond. Sur le mur d'en face, un portail bleu rond. Tu marches vers le portail orange. Tu entres. Tu sors immédiatement du portail bleu, de l'autre côté de la salle.

Tu as parcouru zéro mètre. Tu es à trois mètres de ton point de départ.

Ce moment-là, c'est probablement le tutoriel le plus efficace d'une idée que les mathématiciens ont mis deux cents ans à formuler et que les physiciens du XXe siècle ont recommencé à prendre au sérieux : **l'espace peut ne pas être ce que tu crois qu'il est**.

Ton intuition quotidienne est celle d'un espace à trois dimensions, homogène, où pour aller d'un point à un autre il faut *parcourir la distance qui les sépare*. Deux points éloignés d'un kilomètre exigent qu'on marche un kilomètre. C'est cette évidence qui structure ta perception, tes déplacements, tes cartes.

Portal casse cette évidence en une seconde. Le jeu te force à admettre, avec ton corps de joueur, que la topologie de l'espace peut être *autre*. Que deux points visuellement éloignés peuvent être, dans un certain sens, *voisins*. Et que la géométrie n'est pas la seule chose qui compte quand on parle d'espace.

C'est l'épisode où on parle de ça.

---

## Ce que Portal fait sans le dire

Ce que le jeu fait techniquement est simple à décrire, même si c'est spectaculaire à voir. Quand tu tires ton portal gun sur deux surfaces, tu crées une paire de disques circulaires qui sont **identifiés** l'un à l'autre. Ça veut dire que du point de vue de tout ce qui passe (toi, un cube, une balle d'energy pellet), les deux disques *sont le même disque*. Rentrer par l'un, c'est sortir par l'autre. Instantanément, sans transition, sans distance parcourue.

Le moteur graphique fait un rendu propre de cette identification : quand tu regardes dans le portail orange, tu vois ce qu'on voit par le portail bleu, et vice-versa. Si les deux portails sont dans la même salle et se font face, tu peux littéralement te voir toi-même en train de te regarder dans le portail en face de toi. C'est vertigineux visuellement. C'est aussi, mathématiquement, une manière très propre de rendre visible une propriété topologique fondamentale.

L'espace de Portal n'est pas ce qu'on appelle **simplement connexe**.

Un espace simplement connexe, c'est un espace où si tu traces une boucle fermée (tu pars d'un point, tu marches, tu reviens au même point sans lever le pied), tu peux toujours *rétrécir cette boucle jusqu'à un point* sans la sortir de l'espace, sans la casser. C'est le cas d'une feuille de papier, d'une balle pleine, du volume de ta chambre. Toute boucle peut être ramenée à un point.

Un espace multi-connexe, c'est un espace où *il existe des boucles qu'on ne peut pas rétrécir à un point*. Ces boucles butent sur quelque chose : un trou, une poignée, une identification. Un donut (un tore) est l'exemple classique : une boucle qui fait le tour du trou central ne peut pas être ramenée à un point sans passer par l'extérieur du donut. Le trou fait obstacle.

Portal transforme des salles ordinaires en espaces multi-connexes en y insérant des portails. Une pièce avec deux portails identifiés est topologiquement différente d'une pièce sans portails. Et cette différence topologique se traduit par des expériences physiques différentes que ton corps de joueur ressent immédiatement.

---

## Trois expériences que Portal te force à vivre

**La première :** *la distance parcourue et la position atteinte peuvent être décorrélées*. Tu peux traverser toute une salle en un pas si les portails sont bien placés. À l'inverse, tu peux marcher pendant des minutes dans certaines chambres et ne jamais atteindre l'endroit visé si la topologie ne coopère pas. La *distance métrique* (combien de mètres tu marches) et la *distance topologique* (combien de portails il faut franchir) sont deux choses différentes. Ton intuition quotidienne suppose qu'elles coïncident. Portal te force à les distinguer.

**La deuxième :** *la conservation de la vitesse à travers les portails*. Si tu tombes de haut, tu accélères sous l'effet de la gravité. Si tu tombes dans un portail au sol et que l'autre portail est sur un mur, tu ressors à l'horizontale, à la vitesse à laquelle tu tombais. Tu conserves ta *quantité de mouvement*, mais sa direction est réorientée par la géométrie du portail de sortie. La chute infinie du puzzle classique de Portal (*"speedy thing goes in, speedy thing comes out"*) rend cette conservation viscérale.

Ce n'est pas juste une astuce de gameplay. C'est l'expression concrète d'une idée profonde : *les lois de conservation ne dépendent pas de la topologie de l'espace*. L'énergie, l'impulsion, le moment angulaire sont conservés même si l'espace lui-même est multi-connecté. Cette invariance est un des principes les plus profonds de la physique moderne, et Portal te le fait sentir avec un cube et une pente.

**La troisième :** *l'orientation peut être conservée ou renversée à travers un portail*. Dans Portal, les deux portails d'une paire sont orientés de la même manière : ta main droite reste ta main droite en sortant. Mais mathématiquement, on peut imaginer des identifications qui renversent l'orientation. Une **bouteille de Klein**, par exemple, est une surface non-orientable : ta main droite y devient ta main gauche en la parcourant. Portal ne pousse pas jusque-là (les développeurs ont voulu que le jeu reste jouable), mais l'idée que la topologie *peut* faire ça est là, en filigrane. Tu la sens, sans que le jeu la formule.

---

## Où Portal cesse d'être l'image de la physique

Comme on l'a fait pour tous les épisodes, il faut poser la limite honnête de l'analogie.

Portal est un jeu qui *utilise* la topologie multi-connectée comme mécanique de gameplay. Le jeu suppose que l'espace de base est l'espace euclidien classique, et il y greffe des identifications ponctuelles (les portails) pour créer localement de la multi-connexité. Ce n'est pas un modèle sérieux de l'espace physique. C'est un dispositif ludique inspiré par une idée mathématique.

Les cosmologistes et les physiciens théoriciens qui prennent au sérieux la question topologique de l'espace-temps se posent des questions plus subtiles. Trois territoires principaux :

**Un.** L'*univers global* pourrait-il avoir une topologie non-triviale ? Notre univers observable a un rayon d'environ 46 milliards d'années-lumière, mais l'univers total pourrait être plus grand, et il pourrait, à très grande échelle, avoir une topologie de type tore, sphère, ou plus exotique. On chercherait alors des *échos topologiques* : voir la même galaxie deux fois dans deux directions du ciel, par exemple, parce que la lumière aurait fait le tour de l'univers par des chemins différents. Aucun écho de ce type n'a été détecté à ce jour, ce qui contraint fortement la topologie possible de l'univers, sans la déterminer.

**Deux.** La *relativité générale* d'Einstein permet, en principe, l'existence de **trous de ver** : des connexions topologiques entre deux régions éloignées de l'espace-temps, où le tissu de l'espace-temps se plie et se raccorde à lui-même. Mathématiquement, un trou de ver est possible dans les équations. Physiquement, personne n'en a jamais observé, et les conditions énergétiques nécessaires pour en stabiliser un exigent des formes de matière (dites "exotiques") qu'on n'a jamais rencontrées. Les portails de Portal sont, en un certain sens, des trous de ver domestiqués.

**Trois.** La **gravité quantique** (on en avait parlé avec Minecraft, sous l'angle discrétisation) considère souvent que la topologie de l'espace-temps elle-même pourrait *fluctuer* à très petite échelle. À l'échelle de Planck, on ne parlerait plus d'un espace lisse mais d'une *mousse spatio-temporelle* où des micro-trous-de-ver apparaîtraient et disparaîtraient en permanence. C'est spéculatif, personne ne sait vraiment, mais l'idée est prise au sérieux dans plusieurs programmes de recherche.

Donc Portal te donne *l'intuition* qu'un espace peut avoir une topologie non-triviale. C'est déjà énorme. Mais les portails du jeu sont des *identifications ponctuelles pré-programmées par les développeurs*. La topologie réelle de l'univers, si elle est non-triviale, est *une propriété intrinsèque* du tissu de l'espace-temps, pas quelque chose qu'un être conscient a bricolé avec un pistolet. C'est la même différence structurel/contingent qu'on a notée pour les autres épisodes.

---

## Le pistolet et la question du "qui a le droit"

Il y a un point subtil et fascinant dans Portal qui prépare une des questions les plus profondes de la physique théorique. Le portail gun d'Aperture peut créer des portails à volonté, sur commande, sur des surfaces "portable". Aucun coût énergétique n'est représenté dans le jeu. Aucune conservation n'est mise en péril.

En physique réelle, si on prend au sérieux l'idée qu'on pourrait *créer* une identification topologique entre deux régions de l'espace-temps (un trou de ver artificiel, par exemple), on tombe sur un obstacle immédiat : *ça coûterait des quantités absurdes d'énergie*. Pour maintenir un trou de ver traversable, il faudrait de la matière à énergie négative, et personne ne sait si ça existe. Portal escamote ce coût. C'est un jeu, pas une simulation.

Mais Portal soulève, sans jamais la nommer, une question philosophique majeure : *dans quelle mesure la topologie de l'espace est-elle une propriété donnée du monde, et dans quelle mesure peut-elle être manipulée par un agent* ? Dans le jeu, c'est manipulable. Dans notre univers, ça ne l'est pas, autant qu'on sache. Mais cette possibilité *théorique* de manipuler la topologie est au cœur de nombreux travaux en physique fondamentale, depuis les trous de ver de Kip Thorne jusqu'à certaines spéculations sur les moteurs à distorsion (Alcubierre) qui permettraient de "raccourcir" les distances en pliant l'espace.

Portal ne te fait pas comprendre ces théories, mais il te fait *poser la question*. Ce qui, pour un jeu vidéo, est déjà remarquable.

---

## Ce que tu sais maintenant sans avoir vu d'équations

Si tu as suivi jusqu'ici, tu as compris trois choses que la plupart des gens n'ont jamais formulées :

**Un.** L'espace n'est pas seulement caractérisé par ses distances (sa *géométrie*), mais aussi par sa forme globale (sa *topologie*). Deux points peuvent être proches en distance et éloignés en topologie, ou l'inverse. Portal te force à distinguer ces deux notions parce que ses portails créent des cas où elles se dissocient brutalement.

**Deux.** Un espace peut être **multi-connexe** : il peut contenir des boucles qu'on ne peut pas rétrécir à un point. Un donut, une salle avec deux portails identifiés, éventuellement notre univers à très grande échelle. C'est une catégorie mathématique qui a mis longtemps à être bien formalisée (au XIXe siècle, par Riemann puis Poincaré) et qui reste un territoire de recherche actif en cosmologie et en gravité quantique.

**Trois.** Les lois physiques de conservation (énergie, impulsion, moment angulaire) ne dépendent pas de la topologie de l'espace où elles s'appliquent. Portal te le montre en respectant scrupuleusement la conservation de la vitesse à travers les portails, ce qui rend certains puzzles solvables uniquement par cette conservation. C'est une propriété profonde des lois physiques : *elles sont plus fondamentales que la topologie qui les héberge*.

Dans le prochain épisode, on va boucler la série d'une manière particulière. On reviendra à Dota, avec un long *deep dive* sur Artifact et l'univers cosmologique de Valve, pour tirer tous les fils qu'on a posés au fil des épisodes précédents. Faux vide cosmologique, brisure spontanée de symétrie, invariants topologiques, multivers d'Everett : tout cela nous attend, condensé dans un jeu de cartes que peu de gens ont pris au sérieux à sa sortie.

Pour l'instant, retiens juste ça : la prochaine fois que tu tires un portail bleu sur un mur d'Aperture Science, tu ne joues pas seulement à un puzzle. Tu réactives une des questions les plus profondes de la physique moderne : *quelle est vraiment la forme de l'espace ?*

Poincaré a fait des surfaces. Toi tu as fait des portails. Le geste profond est le même.

---

*Cette série explore une douzaine de jeux et leurs concepts physiques associés. Pour ne rien rater, abonne-toi.*

🌀
