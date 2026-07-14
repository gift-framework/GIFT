---
title: "Blog : On a publié notre brouillon de notice"
layout: default
---

> Version originale en anglais sur [arithmon.substack.com](https://arithmon.substack.com/p/weve-released-our-draft-blueprint)

# On a publié notre brouillon de notice

*Annonce de la version 3.4 du framework K₇, en langage simple. Une dizaine de minutes de lecture.*

---

Il y a un aimant sur ton frigo. Quelque part dans ta cuisine, un poireau attend dans le bac à légumes. Le soleil chauffe ta fenêtre. Si tu as un téléphone allumé à côté, son GPS sait à quelques mètres près où tu te trouves.

Tous ces phénomènes, qui n'ont l'air d'avoir aucun rapport, fonctionnent en réalité grâce à une vingtaine de nombres très précis. Si tu changeais l'un de ces nombres de quelques pourcents, tout ce que je viens de décrire cesserait d'exister. L'aimant ne tiendrait plus. Le soleil ne chaufferait plus pareil. Et la chimie même qui te permet de lire cette phrase n'existerait plus.

On les a tous mesurés, ces nombres. Avec une précision impressionnante. Mais on n'a jamais su d'où ils sortaient.

C'est de ça qu'il est question dans cette page.

---

## Une vingtaine de cases à remplir

Quand un physicien écrit le manuel d'utilisation de la réalité, il y a, dans son équation finale, quelque chose comme dix-neuf cases vides. Vraiment vides. Aucun raisonnement ne donne leur valeur. Pour les remplir, il faut sortir, mesurer, revenir, écrire le résultat dans la case correspondante. Et c'est ce qu'on fait depuis des décennies, avec un succès vertigineux.

Quelques exemples pour situer.

La première case, qu'on appelle alpha, vaut 1/137,036. C'est ce nombre qui détermine la force avec laquelle la lumière interagit avec la matière. C'est lui qui fait qu'un aimant tient sur ton frigo, que les couleurs existent, que ta peau chauffe au soleil. Si tu le décalais de 4 % vers le haut, les étoiles ne fabriqueraient plus de carbone, et donc plus de poireaux, et donc plus de toi. Si tu le décalais de 4 % vers le bas, les noyaux d'atomes s'effondreraient. Personne n'a jamais expliqué pourquoi cette valeur-là, et pas une autre.

Une autre case s'appelle Ngen et vaut exactement 3. C'est le nombre de "familles" de matière dans l'univers. Chaque particule élémentaire de matière existe en trois exemplaires : une version légère, une plus lourde, une encore plus lourde. Pas deux, pas cinq. Trois. Toujours. Personne ne sait pourquoi.

D'autres cases concernent les rapports de masses entre particules, les angles de mélange entre familles, la quantité de matière noire par rapport à la matière ordinaire, et ainsi de suite. Une vingtaine en tout. Toutes mesurées, toutes inexpliquées.

## L'hypothèse de la notice

Il y a un an, j'ai eu une question naïve, peut-être absurde : et si ces nombres n'étaient pas arbitraires ?

Et si, comme une forme musicale détermine les notes qu'un instrument peut jouer, une *forme* géométrique précise, à un niveau plus profond que celui auquel on regarde habituellement, déterminait les valeurs autorisées de ces dix-neuf cases ?

L'idée n'est pas que les nombres seraient choisis. L'idée est qu'ils seraient *contraints*. Comme dans un instrument, où la longueur d'une corde et la forme de la caisse ne sont pas des décisions libres : ce sont elles qui décident des notes.

Cette hypothèse a un nom dans la tradition mathématique : une *structure*. Une notice de montage. Une logique d'assemblage cachée derrière les mesures.

K₇ (anciennement Geometric Information Field Theory) est ma tentative de reconstituer cette notice.

## Ce qu'on a fait

On a choisi une forme géométrique très précise, qui vit en sept dimensions internes. Ne panique pas : il n'y a rien à voir. Ces dimensions ne sont pas des directions où marcher, juste des degrés de liberté abstraits, comme la "couleur" et la "taille" d'un poireau sont deux dimensions descriptives indépendantes.

Cette forme a une propriété rare : elle laisse très peu de place aux choix. À partir du moment où l'on décide qu'elle existe, presque toutes ses caractéristiques sont déterminées. Elle a même un nom de baptême en mathématiques (G₂), mais le nom n'est pas important pour cette page.

Ensuite, on a suivi méthodiquement les conséquences logiques de cette forme. Quel rapport de masses doit-on observer si la nature est faite comme ça ? Quels angles entre familles ? Quelle valeur d'alpha ?

## Quatre-vingt-quinze prédictions, zéro curseur

Le résultat tient en une phrase : à partir de cette forme, sans aucun paramètre ajustable, on peut produire **95 prédictions chiffrées** sur le monde. Pas dix-neuf. Quatre-vingt-quinze.

Toutes ne tombent pas parfaitement juste. Certaines à peine au pourcent près, d'autres avec une marge plus large, et plusieurs morceaux du raisonnement mathématique sont encore en chantier. Ce n'est pas une "solution finale" qu'on annonce, c'est une structure de travail dont une partie commence à coller au monde.

Trois exemples concrets pour donner une idée.

**Le rapport de Koide.** Un rapport mathématique entre les masses des trois leptons chargés (électron, muon, tau). On prédit qu'il vaut exactement 2/3. La mesure expérimentale donne 0,666661. Écart : 0,001 %.

**Le rapport tau / électron.** Le tau est la version la plus lourde de la même famille que l'électron. On prédit que sa masse est exactement 3477 fois celle de l'électron. La mesure donne 3477,15. Écart : 0,004 %.

**Le rapport matière-noire / matière-ordinaire.** Combien y a-t-il de matière noire dans l'univers, par rapport à la matière qu'on connaît ? On prédit 43/8, soit 5,375. L'observation cosmologique donne 5,375. Écart : invisible.

Sur toutes ces prédictions, 11 sont exactes à mieux que un cent-millième près. 53 sont exactes à mieux que 1 %. La précision moyenne est de l'ordre du pourcent.

Important : aucun de ces résultats n'a été *ajusté*. La forme a été choisie, la logique a été déroulée, et les nombres sont tombés. C'est ce qui rend la chose intéressante. Une mauvaise théorie peut toujours être recalibrée pour que les nombres tombent juste. Une théorie sans curseur, soit elle tombe juste, soit elle tombe faux.

## Vérifié par un comptable infatigable

Pendant qu'on faisait tout ça, on a aussi demandé à un programme informatique appelé Lean de relire le raisonnement. Lean fait un travail de comptable : il vérifie ligne par ligne que rien n'a été glissé sous le tapis. Il ne fait pas confiance, il ne saute aucune étape, il pose des questions stupides à chaque virgule. C'est insupportable et c'est exactement ce qu'on lui demande.

Lean a relu 140 affirmations clés du raisonnement, et n'y a trouvé aucune incohérence interne. C'est aussi proche d'un certificat d'absence d'erreur que la mathématique sait produire aujourd'hui.

## La pièce qu'on retire au Jenga

Comme une bonne théorie doit pouvoir se tromper, on a annoncé d'avance la pièce qu'on retire de la tour.

Il y a un nombre, dans la matière neutre qu'on appelle les neutrinos, qui mesure l'asymétrie entre la matière et l'antimatière dans leur comportement. On l'appelle delta-CP. On prédit qu'il vaut exactement 197°. C'est un nombre très précis, déjà partiellement mesuré par les expériences actuelles, et qui sera mesuré beaucoup plus précisément par une expérience appelée DUNE entre 2028 et 2040.

Voici la règle du jeu, annoncée publiquement : si DUNE mesure delta-CP entre 187° et 207°, K₇ a peut-être bien raconté l'histoire. Si DUNE mesure delta-CP en dehors de cet intervalle, on aura faux. Pas faux de manière vague, faux de manière documentée, datée, archivée à l'avance.

C'est exactement la posture que prend un joueur de Jenga qui annonce à toute la table quelle pièce il va retirer, avant de la retirer. Soit la tour reste debout, soit elle s'effondre. Et la victoire n'est pas d'avoir eu raison à tout prix : c'est d'avoir su, dans un cas comme dans l'autre.

## Ce qui reste ouvert

Une partie du raisonnement mathématique n'est pas encore complète : la construction géométrique précise de la forme à sept dimensions reste un problème ouvert que des géomètres travaillent en ce moment. Ce que nous avons, c'est une notice qui marche à l'extérieur (les nombres tombent juste, les vérifications informatiques passent), avec une promesse mathématique encore à finir d'attacher au reste. Les spécialistes le savent.

## Pourquoi je rends ça public

Tout ce que je viens de décrire est désormais accessible librement, gratuitement, sans inscription, sur une plateforme académique appelée Zenodo. Le code source des vérifications informatiques est sur GitHub. La version 3.4 du framework, sortie aujourd'hui, est l'état de référence à partir duquel on poursuit.

Je le publie maintenant pour deux raisons.

La première, c'est que je peux me tromper. Toute théorie peut se tromper. La seule manière de le savoir, c'est que d'autres gens, plus compétents que moi sur tel ou tel point, viennent regarder ce qu'il y a sous le capot. Plus tôt c'est lu, plus tôt on saura.

La deuxième, c'est que les théories scientifiques ne sont jamais le travail d'une seule personne. Ce sont des notices collectives. On les corrige ensemble. On en hérite, on en transmet, on en bricole. Cette page-ci est une étape parmi d'autres dans une conversation très ancienne.

---

Si tu veux aller plus loin :

- la **notice complète**, dans le détail mathématique, est sur Zenodo (lien dans la page d'accueil du blog),
- les **vérifications informatiques** sont sur [github.com/Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean),
- les épisodes précédents de ce blog (Pudge et la mécanique quantique, le Lego cosmique, l'algèbre qui a attendu, la géométrie de Roberto Carlos, la notice de Joyce) racontent les morceaux dans le désordre, par petites touches.

Et comme à la fin de l'épisode Lego : peut-être que nous faisons tous la même chose depuis des siècles. Trier des pièces, comparer des motifs, tester des assemblages, corriger nos erreurs, recommencer.
