---
title: "Blog : Le jour où Tetris t'a appris que l'ordre fait tout"
layout: default
---

> Version originale en anglais sur [giftheory.substack.com](https://giftheory.substack.com/p/episode-2-the-day-tetris-taught-you)

# Épisode 2 : Le jour où Tetris t'a appris que l'ordre fait tout

*Épisode 2 : Tetris et la non-commutativité productive.*

---

Tu as un puits.

Un trou de quatre cases de profondeur sur la colonne la plus à droite, prêt à recevoir un I-piece (la longue barre de Tetris) pour faire un Tetris à quatre lignes. Tu attends ce I-piece depuis trois pièces. Tu vois enfin la barre arriver en haut de l'écran.

Tu la tournes en vertical. Tu la fais glisser à droite. Tu la lâches dans le puits.

Quatre lignes disparaissent. Le carillon Tetris résonne. Ton score grimpe.

Maintenant rejoue la même séquence dans l'autre ordre. Tu fais d'abord glisser le I-piece à droite, *puis* tu le tournes en vertical. La pièce arrive contre le mur de droite, à plat, dans une orientation horizontale. Quand tu essaies de la tourner après, le moteur du jeu refuse, ou la pièce remonte d'une case par le *wall kick*, ou pire elle se loge dans une cavité que tu n'avais pas vue.

Même pièce. Mêmes actions. Ordre différent. Résultat différent.

Tu viens de toucher quelque chose que les physiciens ont mis une dizaine d'années à formaliser proprement, entre 1925 et le milieu des années 1930. Le nom technique est **non-commutativité**, et c'est le cœur opérationnel de la mécanique quantique. Mais tu n'en as pas besoin pour ressentir ce qui se passe. Tu sais déjà, dans tes pouces, que **dans Tetris, l'ordre des opérations n'est pas accessoire. Il *est* le jeu.**

C'est l'épisode où on déballe pourquoi.

---

## Une opération, deux opérations, et tout change

Reviens au I-piece. Tu as deux opérations à ta disposition :

- **R** : rotation de 90 degrés
- **D** : déplacement d'une case vers la droite (D comme droite, pas comme *down*)

Si tu fais R puis D, tu obtiens une certaine position finale.

Si tu fais D puis R, tu obtiens *une autre* position finale.

Pas une position légèrement décalée. Une position fondamentalement différente, parce que la pièce ne rencontre pas les mêmes obstacles dans les deux séquences. Faire pivoter une barre horizontale qui est encore au milieu de la grille, c'est facile, elle a de la place. Faire pivoter une barre qui s'est déjà collée contre le mur droit, c'est un autre problème, elle doit kicker, ou elle reste bloquée à l'horizontale.

En notation mathématique, on écrirait ça : **R·D ≠ D·R**

C'est la propriété fondamentale qui définit les **opérations non-commutatives**. Et il faut poser un mot clair là-dessus, un seul, parce que c'est *le* mot de tout l'épisode. Deux gestes **commutent** quand tu peux les faire dans l'ordre que tu veux et finir exactement au même endroit. Additionner 2 puis 3, ou 3 puis 2 ; multiplier 3 par 5 ou 5 par 3 : ça commute, l'ordre est sans conséquence. Tourner puis déplacer ton I-piece, ou déplacer puis tourner : ça **ne commute pas**, l'ordre fait partie du résultat. C'est tout ce que veut dire ce mot savant : commuter, c'est *pouvoir échanger l'ordre sans que rien ne change*. Garde-le, on s'en sert jusqu'au bout. Et tout joueur de Tetris a senti, des milliers de fois, que l'univers du jeu obéit à des règles non-commutatives.

C'est une intuition que les écoliers n'ont pas. On leur apprend que 2 + 3 = 3 + 2 et on leur fait penser que c'est *toujours* comme ça. La non-commutativité, quand on la rencontre plus tard, est alors présentée comme une bizarrerie réservée à la mécanique quantique. Elle est en réalité partout, dès qu'on touche à des actions sur un espace structuré.

Mets ta chaussette puis ta chaussure : ça marche.
Mets ta chaussure puis ta chaussette : impossible, ou alors ce n'est plus la même chose.

Ouvre la bouteille puis verse : ça marche.
Verse puis ouvre : tu as du jus sur la table.

Tourne à droite au feu rouge puis va tout droit : tu arrives quelque part.
Va tout droit puis tourne à droite : tu arrives ailleurs.

Ces exemples sont stupides, et c'est précisément ce qui les rend précieux. La non-commutativité est la règle ordinaire de tout système où les actions interagissent avec l'état présent du monde. La commutativité est l'exception, pas la règle.

Tetris t'a entraîné à ça depuis ton enfance. Et si Tetris ne te parle pas, prends Candy Crush. Quand tu échanges deux dragées pour créer un match, le board entier se reconfigure : la cascade tombe, des combinaisons qui étaient possibles avant ton coup ne le sont plus après, d'autres apparaissent qui n'existaient pas. L'ordre de tes coups crée littéralement le jeu qui suit. Faire coup A puis coup B ne donne pas la même partie que B puis A. Même un puzzle game mobile obéit déjà à cette logique : chaque coup modifie l'espace des coups futurs.

---

## Le T-spin, ou la non-commutativité récompensée

Mais Tetris fait quelque chose de plus subtil que juste te montrer que l'ordre compte. Tetris **récompense** la non-commutativité. Et c'est là que ça devient profondément intéressant pour la physique.

Le T-spin est un mouvement où tu prends une pièce en T, tu la fais entrer dans une cavité où elle ne devrait pas pouvoir aller par mouvement classique, en exploitant une rotation *à la dernière seconde*. La pièce kick dans le slot, occupe une position qui aurait été impossible à atteindre en plaçant simplement la pièce et en la faisant glisser.

Le jeu te donne plus de points pour un T-spin que pour un placement standard. Le T-spin double et le T-spin triple sont parmi les mouvements les plus valorisés du jeu compétitif. Pourquoi ?

Parce qu'ils exploitent le fait que **certains états du monde ne sont accessibles que par certaines séquences d'opérations**. Tu ne peux pas atteindre la même configuration par un chemin "naturel" (glisser puis tourner). Tu ne peux y arriver qu'en saisissant le bon ordre, au bon moment, avec une compréhension viscérale du fait que les opérations ne commutent pas.

Les concepteurs de Tetris ont inscrit dans les règles du jeu que **savoir naviguer dans la non-commutativité est une compétence supérieure**. Pas une bizarrerie à éviter, une ressource à exploiter.

Et c'est exactement ce que la mécanique quantique a découvert sur la nature.

---

## Le principe d'incertitude, en pouces de joueur Tetris

Et c'est là que le cerveau fait souvent une erreur. On imagine que la mécanique quantique parle d'objets mystérieux, de particules fantomatiques, d'un monde microscopique qui n'aurait rien à voir avec le nôtre. Alors qu'au départ, elle parle surtout d'autre chose de beaucoup plus simple : d'actions qui ne se laissent pas permuter.

Le principe d'incertitude de Heisenberg dit que tu ne peux pas connaître simultanément la position et l'impulsion (sa vitesse, son élan) d'une particule avec une précision arbitraire. Plus tu connais l'une, moins tu connais l'autre. Cette formulation se présente toujours comme un mystère contre-intuitif. On te raconte qu'à l'échelle quantique, "les choses ne se comportent pas comme à notre échelle", comme si c'était une exception bizarre.

Voilà la vraie formulation du principe d'incertitude, celle que les physiciens utilisent dans leurs équations. Position et impulsion sont deux **opérateurs** au sens technique : ce sont des actions qu'on applique à un système. Et ces deux opérateurs **ne commutent pas**. Mesurer la position puis mesurer l'impulsion ne donne pas le même résultat que mesurer l'impulsion puis mesurer la position.

Une précision importante, parce que c'est l'endroit où la vulgarisation se trompe le plus souvent. Le principe d'incertitude n'est pas seulement l'histoire d'une première mesure qui "dérangerait" la seconde en la perturbant physiquement. C'est plus profond que ça : les deux grandeurs ne peuvent pas être simultanément nettes dans un même état, même avant toute mesure. Prépare la même particule mille fois, dans le même état, mesure la position sur cinq cents exemplaires et l'impulsion sur les cinq cents autres : la dispersion des résultats obéit déjà à la contrainte. La non-commutativité n'est pas un accident de l'appareil de mesure. Elle est inscrite dans l'algèbre même des grandeurs.

C'est tout. Le principe d'incertitude découle directement de cette non-commutativité.

Si tu as compris pourquoi tourner-puis-déplacer dans Tetris donne un résultat différent de déplacer-puis-tourner, tu as compris la structure logique du principe d'incertitude. La seule différence avec Tetris : dans le cas quantique, les deux gestes sont **savoir où est la particule** et **savoir à quelle vitesse elle va**. Et leur non-commutativité a une conséquence très concrète : tu ne peux pas avoir les deux nettes en même temps. Pense à une balançoire à bascule, celle des cours de récré : d'un côté ta précision sur la position, de l'autre ta précision sur la vitesse. Plus tu écrases un côté vers le sol (plus tu es précis là-dessus), plus l'autre bout s'envole vers le ciel : l'autre devient flou. La barre est rigide : impossible de plaquer les deux bouts au sol en même temps. Ce n'est ni un manque d'effort, ni un mauvais instrument : c'est la bascule elle-même qui l'interdit. C'est ça, le fameux principe d'incertitude : pas un mystère, une balançoire.

> **Même le joueur le plus rapide du monde ne place pas une pièce plus vite que ce que le moteur du jeu autorise.** Passé un seuil, le facteur limitant n'est plus ton talent, c'est le système lui-même. L'incertitude quantique, c'est exactement ça : une limite inscrite dans les règles, pas dans ton habileté.

L'idée que Heisenberg introduit dans son article de 1925, et que Born et Jordan mettent en forme mathématique quelques mois plus tard, c'est que les grandeurs observables (position, impulsion) ne se comportent pas comme des nombres ordinaires : ce sont des objets qu'on ne peut pas permuter librement, des objets qui ne commutent pas entre eux.

Tu peux remplacer "grandeurs observables" par "actions sur le I-piece" et l'idée reste exactement la même. Heisenberg, sans le savoir, t'a décrit Tetris.

---

## Pourquoi on ne le voit pas ailleurs

Une objection légitime à ce stade : pourquoi est-ce qu'on n'a pas l'impression de vivre dans un monde non-commutatif tout le temps ? Pourquoi 2 + 3 = 3 + 2, et pas seulement dans Tetris ou en mécanique quantique ?

Parce que la commutativité existe, elle est juste l'exception spéciale qui apparaît quand les opérations sont **trop simples pour interagir avec leur contexte**. Ajouter 2 et ajouter 3 ne dépend pas de l'ordre parce que ce sont des opérations qui ne changent pas la nature de l'objet sur lequel elles agissent. Tu peux les inverser sans rien casser.

Mais dès que les opérations *modifient l'état* de l'objet sur lequel elles agissent, l'ordre devient critique. Une rotation modifie la pièce, ce qui change quelles positions de translation sont accessibles ensuite. Une mesure quantique modifie l'état de la particule, ce qui change ce que la mesure suivante peut révéler.

Tetris est un système où les opérations modifient l'état du monde. Mécanique quantique aussi. **Tous les systèmes où les opérations interagissent vraiment avec quelque chose sont non-commutatifs**. La commutativité est le cas dégénéré, pas la règle.

C'est un retournement assez vertigineux quand on le réalise. À l'école on apprend que 2 + 3 = 3 + 2 est *normal* et que les opérateurs qui ne commutent pas en mécanique quantique sont *bizarres*. C'est l'inverse. Le monde réel est presque toujours non-commutatif. La commutativité est ce qu'on observe quand le monde n'agit pas vraiment.

Un dernier point, et c'est le plus honnête : l'analogie entre Tetris et la quantique a une limite, et cette limite est elle-même instructive. Dans Tetris, R et D ne commutent pas *à cause des murs*. Prends une barre seule au milieu du vide, loin de tout obstacle : tu peux la tourner puis la déplacer, ou la déplacer puis la tourner, tu obtiens exactement la même position finale. La non-commutativité de Tetris est *contingente* : elle naît de la rencontre de la pièce avec les bords et le tas.

En mécanique quantique, position et impulsion ne commutent *jamais*. Pas même dans le vide, pas même sans le moindre obstacle. Ce n'est pas une collision qu'on pourrait retirer, c'est inscrit dans la structure mathématique elle-même. La non-commutativité quantique est *structurelle*, pas circonstancielle.

Mais la porte d'entrée (l'idée que deux actions peuvent ne pas se laisser permuter, et que cet ordre porte du sens) est exactement la bonne. Et c'est Tetris qui te l'a ouverte, des années avant que quiconque te parle de Heisenberg. La nuance contingent/structurel, on la précisera le jour où on aura besoin des équations ; l'intuition, elle, tu l'as déjà.

---

## Ce que tu sais maintenant sans avoir vu d'équations

Si tu as suivi jusqu'ici, tu as compris quatre choses qui sont absolument fondamentales pour la suite :

**Un**. Les opérations dans le monde réel ne sont presque jamais commutatives. L'ordre fait partie de l'action, pas un détail accessoire.

**Deux**. La non-commutativité n'est pas un bug ou une bizarrerie. C'est une **ressource**. Tetris la récompense par les T-spins. La nature en fait l'un de ses traits structurels les plus profonds.

**Trois**. Le principe d'incertitude de Heisenberg n'est pas un mystère contre-intuitif. C'est la traduction quantitative d'un fait qualitatif que tu connais déjà : certaines paires d'actions ne commutent pas, et la quantité par laquelle elles ne commutent pas mesure à quel point elles interfèrent.

**Quatre**. La pédagogie classique te présente ces idées comme étrangères et la mécanique quantique comme exotique. Or tu as joué à un jeu qui en exploite la structure pendant des heures sans le savoir. Tu as les intuitions, il manquait juste les noms.

Dans les épisodes qui viennent, on va creuser plus loin. On verra comment cette non-commutativité, combinée avec la mesure (qu'on a vue dans l'épisode 0 avec Pudge), produit l'intrication. Comment elle interagit avec l'aléatoire classique (qu'on a vu dans l'épisode 1 avec Fall Guys). Comment elle structure les particules elles-mêmes.

Pour l'instant, retiens juste ça : à chaque fois que tu fais un T-spin et que tu sens, sans pouvoir l'expliquer, que tu viens d'exploiter une faille dans l'espace géométrique du jeu, tu touches du doigt la même intuition que Heisenberg en 1925, quand il a compris pourquoi position et impulsion ne commutent pas.

Lui en a fait une matrice. Toi tu en as fait un T-spin triple. Le geste profond est le même.

---

*Cette série explore une douzaine de jeux et leurs concepts physiques associés. Pour ne rien rater, abonne-toi sur [giftheory.substack.com](https://giftheory.substack.com/).*

🌀
