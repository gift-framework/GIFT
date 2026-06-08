---
title: "Blog : Le jour où Bordeciel ne respirait pas sans toi"
layout: default
---

> Version originale en anglais sur [giftheory.substack.com](https://giftheory.substack.com/p/episode-5-the-day-skyrim-wasnt-breathing)

# Épisode 5 — Le jour où Bordeciel ne respirait pas sans toi

*Skyrim, le monde en attente d'observateur.*

---

Tu marches sur une route déserte de Bordeciel, vers le nord, depuis Blancherive. Devant toi, à un kilomètre peut-être, un fort en ruines occupé par des bandits. Tu le sais parce que tu y es déjà allé deux fois, et la deuxième fois ils étaient en train de cuisiner un cerf.

Pose-toi une question simple. Là, maintenant, pendant que tu chemines, à quoi font les bandits ? Est-ce qu'ils discutent ? Est-ce que l'un d'eux affûte sa hache ? Est-ce que le feu de camp brûle ? Est-ce qu'il y a une marmite qui mijote ?

La réponse, si tu as déjà ouvert un fichier de sauvegarde Skyrim ou si tu connais un peu la machinerie des jeux à monde ouvert, est très brutale : *non*. Les bandits ne font rien. Ils n'existent pas. Aucune ligne de code ne calcule ce qu'ils sont en train de faire. Il y a, dans la mémoire de ton ordinateur, une description vide : *bandits, fort de [coordonnées], status : non chargés*. C'est tout. Pas de pensée, pas de respiration, pas de geste. Ils sont en attente d'observateur.

Tu t'approches. Tu passes une crête. À une distance précise — quelques centaines de mètres dans Skyrim, ça varie selon les jeux et les versions — le moteur déclenche ce qu'on appelle un **chargement de chunk**. La zone autour du fort entre en mémoire. Les bandits *commencent à exister*. Le feu prend. Les positions s'instancient. Une routine est lancée pour décider qui fait quoi. Et au moment où tu arrives à portée de vue, les bandits sont là, en train de cuisiner leur cerf, comme s'ils l'avaient toujours fait.

Mais ils ne l'avaient *pas* toujours fait. Ils ont commencé à le faire au moment où tu es entré dans leur sphère d'observation.

Tu joues à un univers qui ne tourne que là où tu poses ton regard. C'est l'épisode où on parle d'une idée vertigineuse : peut-être que c'est un peu vrai pour l'univers réel aussi.

---

## Ce que Skyrim fait sans vergogne

Skyrim, comme tous les jeux à monde ouvert un peu sérieux, n'a tout simplement pas le luxe de calculer tout ce qui se passe dans tout le monde en même temps. La carte est trop grande, la mémoire est trop limitée, le processeur a d'autres choses à faire. Donc le moteur fait un choix radical : il ne simule que ce qui est *suffisamment proche du joueur*.

Plus précisément, Skyrim découpe le monde extérieur en blocs (chunks), des cases de 4096 unités de côté — environ soixante mètres dans l'échelle du jeu. À chaque instant, seules les cells dans un certain rayon autour du joueur sont **chargées** : leurs objets existent en mémoire, leurs créatures pensent, leurs PNJ vivent. Les autres cells, qui constituent l'écrasante majorité de Bordeciel à tout moment, sont **déchargées** : elles existent comme des descriptions abstraites sur le disque dur, mais aucun calcul ne tourne pour elles.

Quand tu marches, des cells se chargent devant toi et se déchargent derrière. Le monde se *constitue* autour de toi, par paquets, en permanence. C'est invisible quand le matériel suit. Quand le matériel suit moins bien, tu vois des arbres apparaître, des PNJ se téléporter sur place, des objets surgir. Tu vois, en transparence, le mécanisme.

Cette technique a un nom dans le métier : on l'appelle **streaming**. C'est utilisé par à peu près tous les jeux à monde ouvert depuis vingt ans, de GTA à Breath of the Wild en passant par No Man's Sky. C'est une nécessité purement technique, dictée par la finitude de la RAM. Et c'est, par accident, une métaphore stupéfiante d'une question philosophique que les physiciens se posent depuis cent ans.

---

## La question vertigineuse de Wheeler

John Wheeler était un physicien américain mort en 2008, l'un des plus inventifs de son siècle. Il a baptisé les "trous noirs", il a travaillé avec Einstein et Bohr, il a formé Feynman. Et vers la fin de sa vie, il a passé beaucoup de temps sur une question qui le rendait fou : *qu'est-ce qui existe quand personne ne regarde ?*

Sa réponse, formulée dans les années 1980, est connue sous le nom de **principe anthropique participatif** ou plus simplement *participatory universe*. Elle dit, en très gros, ceci : l'univers n'a pas de propriétés définies indépendamment des actes d'observation qui les révèlent. Ce n'est pas que l'observation *découvre* une propriété qui existait déjà. C'est que l'observation *participe à constituer* cette propriété.

Wheeler résumait ça par une formule restée célèbre : *"no phenomenon is a real phenomenon until it is an observed phenomenon"*. Aucun phénomène n'est un phénomène réel tant qu'il n'est pas un phénomène observé.

C'est une thèse radicale, et il faut être très prudent sur ce qu'elle dit exactement. Wheeler ne disait pas que la Lune n'existe pas quand personne ne la regarde. Il disait quelque chose de plus subtil : pour les phénomènes quantiques élémentaires (la trajectoire d'un photon, par exemple), les propriétés que nous appelons "réelles" ne sont pas inscrites dans le monde indépendamment de la mesure. Elles émergent dans la rencontre entre le système quantique et l'acte d'observation.

Pour soutenir cette idée, Wheeler avait imaginé une expérience théorique appelée *delayed choice* (choix retardé), qui a depuis été réalisée en laboratoire. Sans entrer dans le détail technique : on peut concevoir des dispositifs où la décision de mesurer ou non un photon d'une certaine manière, *prise après que le photon ait déjà voyagé*, comme si le récit que nous sommes autorisés à faire de son trajet ne se décidait qu'au moment de la mesure.

Si tu trouves ça incompréhensible, c'est normal. C'est le but. Wheeler voulait que ce soit incompréhensible, parce que le monde quantique l'est. Et c'est exactement de ça qu'on parle quand on parle d'*observateur participatif* : pas que tu inventes le monde en y pensant, mais que la frontière entre l'observateur et l'observé est plus floue, plus négociable, plus *participative* qu'on ne le croit.

---

## Pourquoi Skyrim t'a déjà donné l'intuition juste

Maintenant, je dois être honnête avec toi. Skyrim ne *réalise* pas la thèse de Wheeler. Le moteur Skyrim ne charge des chunks que pour des raisons techniques bêtement matérielles : ta carte graphique n'a pas la puissance pour faire tourner Bordeciel entier en parallèle. Wheeler, lui, parlait de mécanique quantique fondamentale, où la limite n'est pas matérielle mais structurelle.

Mais Skyrim, en t'imposant cette limitation matérielle, t'entraîne à *sentir* trois choses qui sont exactement celles que Wheeler voulait te faire sentir.

**La première :** l'existence d'une chose peut dépendre du regard porté sur elle. Tu as joué pendant des heures, voire des années, dans un monde dont les trois quarts n'existaient à aucun moment précis. Les bandits du fort distant n'avaient ni position définie, ni action en cours, ni état mental. Ils étaient en attente, en superposition d'états possibles, jusqu'à ce que ton arrivée les actualise. Tu n'as pas trouvé ça absurde. Tu as joué normalement. Tu as accepté, sans même y penser, l'idée que l'existence d'une chose puisse être conditionnée à l'observation.

**La deuxième :** la frontière entre observateur et observé n'est pas une ligne nette. Quand tu approches du fort, à quel moment exactement les bandits commencent-ils à exister ? Il y a un seuil de chargement, oui, mais ce seuil est arbitraire. Si tu le déplaçais de 50 mètres, le moment précis où les bandits "naissent" changerait. Donc cette naissance n'a pas de réalité intrinsèque indépendante de l'algorithme qui la déclenche. Les bandits ne s'allument pas parce que c'est l'heure de s'allumer. Ils s'allument parce que tu es entré dans leur cell. L'événement d'apparition est *relatif* à toi, à ta position, à ton regard. Tu as appris, manette en main, qu'il existe des événements dont l'existence même est relative à l'observateur.

**La troisième :** ce qui n'est pas observé ne consomme aucune ressource. C'est un point presque économique, mais profond. Dans Skyrim, l'inobservé n'a pas de coût pour le système : il dort. C'est pour ça que le streaming marche. Et Wheeler suggérait quelque chose d'étrangement parallèle pour l'univers : peut-être que les propriétés non-observées n'ont pas non plus de "réalité actualisée", elles n'occupent pas de place dans l'inventaire des faits du monde. Elles sont en potentiel, en attente, sans avoir à être *déjà* quelque chose de précis.

Ces trois intuitions, tu les as. Pas parce que tu as lu Wheeler, mais parce que tu as joué à Skyrim. La question vertigineuse devient : *est-ce que l'univers réel est, à un niveau profond, un peu comme un Skyrim cosmique ?*

---

## Où Skyrim cesse d'être l'image de la physique

Comme on l'a fait pour Tetris et pour Minecraft, il faut poser la limite honnête de l'analogie. Sinon on bricole une mystique facile et un physicien sérieux a raison de lever les yeux au ciel.

Skyrim a un observateur unique : toi, le joueur. Tu es le centre absolu, le référentiel privilégié, celui pour qui le monde se charge. Si tu fais une partie en multijoueur (avec un mod approprié), les choses deviennent vite incohérentes : les deux joueurs ne peuvent pas être centres simultanés. Le moteur doit bricoler. Le modèle Skyrim suppose un observateur unique au monde.

La physique réelle, elle, ne tolère aucun observateur privilégié. C'est même un principe fondateur depuis Einstein : les lois sont les mêmes pour tous les observateurs, peu importe leur position, leur vitesse, leur état. Donc une théorie sérieuse du "tout n'existe qu'à travers l'observation" doit s'arranger pour que *tous* les observateurs voient un monde cohérent, sans qu'aucun ne soit le centre. C'est une contrainte technique très lourde, et c'est l'une des raisons pour lesquelles l'interprétation participative de Wheeler reste minoritaire parmi les physiciens.

Plusieurs courants modernes ont essayé de la sauver en la modifiant. La **mécanique quantique relationnelle** de Carlo Rovelli dit que les propriétés d'un système n'existent que relativement à d'autres systèmes (pas seulement à des humains conscients), et que chaque "observateur" voit sa propre version cohérente du réel. Le **QBisme** (Quantum Bayesianism) dit que la fonction d'onde n'est pas une description du monde mais un état de croyance d'un agent particulier. Ces approches gardent l'esprit Wheeler — l'observation comme participante — sans le piège du "vous êtes spéciaux".

Donc l'analogie Skyrim est juste pour te donner *l'intuition de fond* qu'une existence peut être conditionnée à une observation. Elle est trompeuse si tu en conclus que toi, joueur humain, es au centre cosmique de l'univers. Tu n'es pas le centre. Tu es un observateur parmi d'autres, et chaque observateur fait, peut-être, charger ses propres chunks.

C'est plus subtil et plus démocratique que Skyrim. Mais le geste profond — *l'existence comme acte conditionné, pas comme propriété intrinsèque* — est exactement le bon.

---

## Bonus : le quicksave comme post-sélection de trajectoires

Il y a une autre chose qu'on fait dans Skyrim qui mérite un mot, parce qu'elle pointe vers un autre coin de la physique fondamentale qu'on reverra plus tard.

Quand tu sauvegardes avant un combat difficile, et que tu meurs, tu charges. Tu refais. Tu meurs encore. Tu charges. Cette fois tu réussis. Tu sauvegardes par-dessus. Et tu continues comme si la version où tu réussis était la "vraie".

Réfléchis trente secondes à ce que tu as fait. Tu as exploré plusieurs *branches* de réalité (mort, mort, succès), tu n'en as gardé qu'une, et tu présentes ensuite cette branche comme étant *la* trajectoire de ton personnage. La trajectoire vécue n'est pas la trajectoire totale : c'est la trajectoire *post-sélectionnée*. Tu as effacé les branches défavorables.

C'est cousin lointain de ce qu'on appelle l'**interprétation des mondes multiples** d'Everett en mécanique quantique, où toutes les branches d'une superposition se réalisent dans des univers parallèles, mais chaque observateur ne vit consciemment qu'une seule. La différence majeure : dans Everett, les autres branches *continuent d'exister*, tu n'es juste pas dans la position de les vivre. Dans Skyrim, les autres branches *sont effacées* du disque dur.

On reverra cette idée plus en détail dans un épisode futur, parce qu'elle ouvre une autre porte vertigineuse. Pour l'instant, garde simplement ça en tête : à chaque fois que tu charges une sauvegarde, tu fais, en miniature, ce qu'un univers à mondes multiples ferait à ton insu en permanence — sauf que les "univers manqués" d'Everett, eux, ne s'effacent pas. Ils continuent quelque part, peuplés de versions de toi qui ont fait d'autres choix.

---

## Ce que tu sais maintenant sans avoir vu d'équations

Si tu as suivi jusqu'ici, tu as trois intuitions que la plupart des gens n'ont pas formulées clairement :

**Un**. L'existence d'une chose peut être conditionnée à un acte d'observation. Tu l'as vécue dans Skyrim sans y penser : les bandits du fort lointain n'existaient pas, et c'était normal. Wheeler proposait que quelque chose d'analogue, à un niveau profond, soit peut-être vrai des phénomènes quantiques élémentaires dans la nature.

**Deux**. La frontière entre observateur et observé n'est pas une ligne nette inscrite dans le monde. Elle dépend de protocoles, de seuils, d'algorithmes. Dans Skyrim, c'est le rayon de chargement des *portions (chunks) de la carte*. En physique fondamentale, c'est le choix d'appareils de mesure, les contextes expérimentaux, et — pour Rovelli ou le QBisme — le système de référence qu'on adopte. La nature ne pose pas une frontière, elle la négocie.

**Trois**. L'analogie Skyrim a une limite essentielle : tu es l'observateur unique du jeu. La physique sérieuse ne tolère aucun observateur privilégié. Si l'intuition participative est vraie, elle doit valoir pour *tous* les observateurs simultanément, sans centre cosmique. C'est ce qui rend la question vraiment difficile, et c'est aussi ce qui empêche de basculer dans la mystique New Age. Wheeler ne disait pas que tu crées l'univers en y pensant. Il disait que l'univers n'a peut-être pas, à son niveau le plus fin, de propriétés définies indépendamment de toute observation par quoi que ce soit.

Dans les épisodes qui viennent, on continuera d'explorer ces frontières. Pour l'instant, retiens juste ça : la prochaine fois que tu approches d'un fort de bandits dans Bordeciel et que tu sens, à un instant précis, le monde se charger autour de toi, tu ne joues pas seulement à un RPG bien fichu. Tu reproduis, à l'échelle d'une partie de jeu vidéo, une question que les physiciens posent depuis cent ans sans avoir trouvé de réponse : *qu'est-ce qui existe vraiment quand personne ne regarde ?*

Tu ne sais pas. Wheeler ne savait pas. Personne ne sait. Mais tu as déjà passé des heures à jouer dans un monde où la réponse était *"pas grand-chose, en attendant que quelqu'un passe par là"*.

---

*Cette série explore une douzaine de jeux et leurs concepts physiques associés. Pour ne rien rater, abonne-toi.*

🌀
