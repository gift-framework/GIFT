---
title: "Blog : Le jour où Minecraft t'a appris à dessiner une sphère qui n'en est pas une"
layout: default
---

> Version originale en anglais sur [arithmon.substack.com](https://arithmon.substack.com/p/episode-4-the-day-minecraft-taught)

# Épisode 4 : Le jour où Minecraft t'a appris à dessiner une sphère qui n'en est pas une

*Minecraft, la discrétisation, et l'éternel problème du continu.*

---

Tu veux construire une sphère.

Tu sais à peu près à quoi ça doit ressembler : un objet rond, sans angles, sans arêtes, lisse en tout point. Tu as déjà vu mille sphères : une orange, une bille, la Terre depuis l'espace. C'est l'une des formes les plus simples que la géométrie connaisse.

Tu commences avec un rayon de quatre blocs. Le résultat ressemble à une patate cubique. Tu rigoles, tu détruis, tu recommences avec un rayon de huit. C'est mieux, mais c'est encore très anguleux : tu vois des escaliers à chaque "côté", des plateaux plats qui n'ont rien de sphérique. Tu pousses à seize. Là, ça commence à *suggérer* une sphère, vue de loin, en plissant un peu les yeux. À trente-deux blocs, tu y crois presque. À soixante-quatre, vue à distance, c'est vraiment une sphère.

Mais approche. Mets ton avatar à un bloc de la surface. Regarde. Tu vois des marches. Tu vois des coins. Tu vois la grille. Plus tu zoomes, plus la sphère redevient un assemblage de cubes empilés. Aucun raffinement ne fait disparaître cette nature cubique : il la rend juste *plus petite*.

Tu peux pousser à cent vingt-huit blocs, à mille vingt-quatre, à un million. La sphère sera *de plus en plus convaincante à distance*, mais elle ne sera *jamais* une vraie sphère. Une vraie sphère n'a pas de plus petit élément. Une sphère Minecraft, si.

Cette frustration que tu as ressentie en construisant ta première grosse boule, c'est exactement le problème central de toute la physique numérique moderne. Et c'est aussi, peut-être, le problème central de l'univers réel lui-même.

---

## Le monde lisse contre le monde en blocs

En mathématiques, depuis Euclide, on a une intuition très claire de la différence entre deux types d'objets.

Il y a les objets **discrets** : un sac de billes, une grille d'échecs, une liste de nombres entiers. Tu peux les compter. Entre deux éléments voisins, il n'y a rien. Tu sautes de l'un à l'autre. Pas de demi-bille, pas de case entre deux cases.

Et il y a les objets **continus** : une ligne droite, une sphère, l'eau d'un verre. Entre deux points, il y a toujours un troisième point. Tu peux zoomer infiniment, tu trouveras toujours quelque chose entre. Pas de plus petit segment, pas de plus petit volume.

Pendant longtemps, les physiciens ont supposé que la nature était fondamentalement continue. L'espace, le temps, les champs, tout pouvait être zoomé indéfiniment, et on trouverait toujours du lisse, toujours du fluide, jamais le moindre plus petit grain. C'est l'image qui domine Newton, Maxwell, Einstein. Le monde est une étendue continue dans laquelle des choses bougent continûment.

Et puis sont arrivés les ordinateurs.

---

## Le drame de la simulation

Un ordinateur ne sait pas faire du continu. Un ordinateur ne connaît que des nombres, et il en a un nombre *fini*. Sa mémoire est faite de bits, qui sont 0 ou 1, sans demi-bit possible. Un nombre dans un ordinateur a toujours un nombre limité de chiffres après la virgule. Tu peux en mettre beaucoup, mais jamais une infinité.

Donc dès qu'on a voulu *simuler* la physique dans un ordinateur, on s'est heurté à ce mur. Comment représenter une onde lisse avec un nombre fini de points ? Comment représenter un espace continu avec une grille ? Comment représenter une particule qui peut être *n'importe où* avec un système qui ne connaît qu'un nombre fini de positions possibles ?

La réponse, qu'on utilise depuis cinquante ans : on **discrétise**. On découpe l'espace en petites cellules, le temps en petits intervalles, les valeurs en petits paliers. On fait du Minecraft, en gros. Et on espère que si on découpe assez finement, le résultat ressemblera assez à ce qu'aurait donné le calcul continu.

Quand un physicien simule l'atmosphère terrestre pour prédire la météo, il découpe la Terre en cubes de quelques kilomètres de côté. Quand on simule le climat, on monte à des dizaines de kilomètres. Quand on simule une étoile en train de s'effondrer, on découpe en mailles adaptées à la densité locale. Quand on calcule la masse du proton à partir des quarks qui le composent, on découpe carrément l'espace-temps lui-même en une grille de quelques fermis : c'est la *lattice QCD*, et c'est notre meilleure preuve que la discrétisation, faite avec rigueur, donne des prédictions qui collent à dix décimales aux mesures expérimentales. Quand on simule la structure géométrique d'une variété abstraite en sept dimensions (j'y reviendrai) on découpe avec des réseaux de neurones qui apprennent à interpoler entre les mailles.

À chaque fois, le même problème : la discrétisation est une trahison du continu. On fait au mieux. On raffine. On vérifie que les résultats convergent quand on raffine encore. Mais on sait, intimement, qu'on construit une sphère Minecraft. La vraie sphère, on ne la touche pas. On l'approxime.

---

## Pourquoi Minecraft te donne pile la bonne intuition

Quand tu construis ta grosse sphère Minecraft, tu vis *exactement* ce qu'un physicien numérique vit chaque jour, à une différence près : toi, tu sais que tu fais du Minecraft, parce que les blocs sont gros et que tu peux les compter. Le physicien, lui, fait du Minecraft à très haute résolution, et il oublie parfois qu'il est en train de faire du Minecraft.

Trois intuitions précieuses te viennent automatiquement quand tu joues :

**La première :** plus la résolution monte, plus l'illusion est convaincante, mais l'illusion ne devient jamais la chose. Tu sais, viscéralement, qu'il y a toujours une *échelle* en dessous de laquelle ta sphère cesse d'être une sphère et redevient des blocs. C'est exactement ce que ressent un physicien quand il regarde sa simulation à très haute résolution : à grande échelle, ça ressemble. À petite échelle, ça redevient une grille.

**La deuxième :** certaines propriétés *survivent* à la discrétisation, d'autres deviennent délicates. Le rayon de ta sphère Minecraft est assez robuste, même grossièrement. Son volume aussi, à peu près. Mais sa *courbure ponctuelle*, elle, devient beaucoup plus délicate : si tu regardes bloc par bloc, tu ne vois plus une courbure lisse qui varie en douceur, tu vois des faces plates et des arêtes. On peut bricoler des notions de courbure discrète qui rendent quelque chose de proche à grande échelle, mais la grandeur "lisse en tout point" est *brisée* par la discrétisation locale. Le physicien numérique passe sa vie à savoir quelles grandeurs sont robustes à la discrétisation, et quelles autres deviennent des artefacts.

**La troisième :** la grille elle-même introduit des **artefacts** qui n'existent pas dans le continu. Ta sphère Minecraft a six "faces" privilégiées (haut, bas, nord, sud, est, ouest), parce que tes blocs sont alignés sur des axes. Une vraie sphère n'a aucune direction privilégiée, elle est parfaitement symétrique. Donc ta sphère Minecraft a *moins de symétrie* que la sphère qu'elle est censée représenter. Le simulateur numérique vit avec ça constamment : sa grille casse des symétries de la nature, et il faut le savoir pour ne pas confondre un effet vrai avec un artefact de grille.

Tu n'as pas appris ces trois choses à l'école. Tu les as apprises en passant trois après-midis à construire une boule pour la décorer.

---

## Deux questions à ne pas confondre

Avant d'aller plus loin, il faut séparer deux choses qui se ressemblent et que la suite de cet épisode va relier, mais qui sont logiquement distinctes.

La première question, c'est : *qu'est-ce qu'on perd quand on simule du continu avec du discret ?* C'est la question des météorologues, des climatologues, des physiciens numériques, et c'est celle que tu vis dans Minecraft. Elle existe même si la nature est, par ailleurs, parfaitement continue. Tu peux avoir un univers strictement lisse, et néanmoins faire des calculs en discret par nécessité instrumentale. C'est une *limite de nos outils*, pas une affirmation sur le monde.

La seconde question, c'est : *et si la nature elle-même était fondamentalement discrète à très petite échelle ?* C'est une question ouverte de physique fondamentale, sans rapport direct avec nos limitations de calcul. L'univers pourrait être lisse jusqu'au bout, et nous serions condamnés à le simuler en grossier. Ou l'univers pourrait être discret, et nos simulations discrètes seraient alors, par hasard, fidèles à sa nature profonde. Les deux questions ne se répondent pas l'une l'autre.

Minecraft, lui, est sans ambiguïté du côté de la première : c'est un outil d'approximation, avec une grille fixe. Et c'est là qu'il faut poser, comme on l'a fait pour Tetris, la limite honnête de l'analogie.

---

## Où Minecraft cesse d'être l'image de la physique

Minecraft a une grille fixe. Les blocs s'alignent sur trois axes privilégiés : haut-bas, nord-sud, est-ouest. Ces directions existent *avant* qu'on y mette quoi que ce soit. Si tu déplaces ta sphère Minecraft de quelques mètres, ou si tu la fais tourner de 30 degrés, elle cesse d'être bien définie sur la grille : les blocs ne s'alignent plus, il faut tout reconstruire. La grille porte une orientation absolue.

La vraie physique discrète, telle que les théoriciens essaient de la construire, refuse exactement ça. Une théorie sérieuse de l'espace-temps discret ne peut pas avoir d'orientation préférentielle, parce que la nature, à notre échelle, n'en a aucune : les lois sont les mêmes que tu te tournes vers le nord ou vers l'ouest, que tu te déplaces lentement ou rapidement. Donc les théoriciens construisent des modèles où il n'y a *pas* de grille de fond. Les éléments discrets, petits volumes, petits tétraèdres, petits nœuds reliés en réseau, n'existent pas *dans* un espace, ils *constituent* l'espace, par leurs relations mutuelles. Pas d'axes posés avant eux. Pas de Nord absolu.

La gravité quantique à boucles fait ça avec ce qu'on appelle des *réseaux de spins*. Les triangulations causales dynamiques font ça avec des assemblages de tétraèdres dont seule la topologie compte. Dans tous les cas, le mot-clé est *background-independent* : indépendant du fond. Le discret de Minecraft est lui *background-dependent* : il vit sur une grille préexistante.

C'est une différence profonde. Minecraft te donne l'intuition que le discret est possible et qu'il peut produire des illusions de continu convaincantes à grande échelle. Mais il te donne aussi, gratuitement et un peu trompeusement, une grille fixe que la vraie physique discrète n'a pas. À toi de retenir l'intuition de fond, *le lisse pourrait être une illusion macroscopique*, sans transposer l'orientation fixe des blocs Minecraft sur la nature.

Garde ça en tête pour ce qui suit.

---

## Et si l'univers était lui-même discret ?

Voilà la question qui change tout, et qui est encore ouverte aujourd'hui en physique fondamentale.

Toute la physique du XXe siècle a supposé que l'espace et le temps étaient continus. Tu peux zoomer indéfiniment dans la matière (et tu trouves des particules), mais l'*espace* dans lequel se déplacent ces particules, lui, est supposé lisse, sans plus petit élément.

Cette supposition pose des problèmes quand on essaie de combiner la mécanique quantique avec la relativité générale d'Einstein. Quand on essaie, à toutes petites échelles, de calculer ce qui se passe quand la gravité devient quantique, on tombe sur des infinis impossibles à gérer. La théorie *casse*. Et beaucoup de physiciens, depuis les années 1960, se sont demandé si ce ne serait pas le signe que l'hypothèse de fond *l'espace continu* était fausse.

L'échelle où ça devrait se passer s'appelle la **longueur de Planck**, et elle vaut à peu près 1,6 × 10⁻³⁵ mètre. Pour donner une idée : si tu prenais un atome et que tu le grossissais à la taille de l'univers observable, la longueur de Planck serait encore plus petite qu'un grain de sable à cette échelle. C'est ridiculement minuscule. On ne sait pas la mesurer directement aujourd'hui, et elle semble absurdement hors de portée expérimentale. Mais c'est, paraît-il, l'échelle où l'espace lui-même pourrait cesser d'être lisse.

Et ici, l'intuition Minecraft devient utile à condition de la corriger : tu en as déjà vu plus haut la limite, pas de grille fixe en physique fondamentale, pas d'axes privilégiés. Mais l'intuition *de fond* est juste : à très petite échelle, le lisse pourrait disparaître au profit d'une combinatoire de petits éléments, et la nappe continue que nous percevons à notre échelle ne serait que leur effet collectif moyenné, comme ta sphère paraît ronde de loin et révèle ses blocs de près.

Si c'est vrai, alors le rapport s'inverse complètement. Ce n'est plus toi, dans Minecraft, qui essaies de faire une vraie sphère avec des blocs imparfaits. C'est l'univers entier qui est fait de petits éléments discrets, et toi, sphère humaine, qui es l'illusion macroscopique d'un assemblage à très petite échelle. Toi, ta tasse de café, la Lune, tout. Des illusions de lisse tellement bien raffinées qu'elles trompent toute mesure que nous savons faire.

Personne ne sait encore si c'est le cas. Mais le fait même que la question soit prise au sérieux par les physiciens montre quelque chose : la frontière entre le discret et le continu, qu'on croyait évidente, est devenue floue. On ne sait plus de quel côté se range la nature.

---

## L'autre direction : simuler le continu avec du discret

Il y a un autre versant de ce problème, plus modeste mais tout aussi vertigineux, qui occupe une grande partie de la recherche actuelle. C'est celui dans lequel je passe une partie de mes journées, donc je vais le mentionner brièvement, sans en faire un épisode à part.

Quand tu veux étudier une forme géométrique compliquée qui vit dans un espace abstrait à sept dimensions (par exemple, parce que tu cherches à savoir si certaines structures mathématiques existent vraiment et avec quelles propriétés) tu ne peux évidemment pas la dessiner. Tu ne peux pas non plus la calculer entièrement à la main : les équations sont trop riches. Alors tu fais ce qu'on fait depuis trente ans en physique numérique : tu discrétises. Tu prends ta forme idéale, qui devrait être lisse et continue, et tu la représentes par un réseau de neurones qui apprend à donner une valeur en chaque point.

C'est exactement la sphère Minecraft du début de cet épisode. À ceci près qu'au lieu de blocs cubiques alignés sur une grille, tu as des neurones qui interpolent. Et au lieu d'une sphère à trois dimensions, tu as une variété à sept dimensions. Mais le geste profond est le même : tenter de capturer une forme lisse avec des éléments discrets, en raffinant assez pour que les invariants importants survivent à la discrétisation.

C'est un projet que j'appelle K₇, et qui a sa propre série d'articles sur ce blog si la question t'intéresse vraiment. Je n'en parle pas plus ici parce que ce n'est pas le sujet. Le sujet, c'est que cette tension entre le lisse et le grossier, entre l'idéal continu et l'approximation discrète, est *partout* dès qu'on essaie de faire de la science numérique. Et tu y as joué dans Minecraft avant même d'avoir vu une équation.

---

## Ce que tu sais maintenant sans avoir vu d'équations

Si tu as suivi jusqu'ici, tu as compris trois choses que beaucoup d'étudiants en physique mettent des années à formuler clairement :

**Un.** Toute simulation numérique de la nature est une sphère Minecraft. Plus on raffine, plus l'illusion est convaincante, mais l'illusion n'est jamais la chose. Le continu n'est pas dans l'ordinateur. Il est dans ce qu'on essaie de représenter avec.

**Deux.** Certaines propriétés survivent à la discrétisation, d'autres pas. Savoir lesquelles, c'est une compétence centrale de la physique numérique. Le rayon de ta sphère survit. Sa courbure non. Et il faut s'en souvenir.

**Trois.** La question "et si l'univers réel était lui-même, métaphoriquement, un Minecraft à très petite échelle ?" (non pas avec des cubes alignés, évidemment, mais avec une structure discrète sous-jacente) n'est pas une fantaisie de geek. C'est une hypothèse de travail sérieuse en gravité quantique. Si elle est vraie, alors le lisse qu'on touche du doigt, qu'on voit, qu'on construit comme dans Minecraft, est *toujours* une illusion macroscopique d'un substrat discret sous-jacent. Tu serais alors, comme ta sphère Minecraft, une illusion convaincante à grande échelle.

Dans les épisodes qui viennent, on continuera à explorer ces frontières. Pour l'instant, retiens juste ça : la prochaine fois que tu construis une boule de pierre lisse en Minecraft à mille blocs de rayon et que tu te dis que ça fait illusion, tu ne fais pas seulement un projet de jeu. Tu fais, à ta façon, un geste que les physiciens font tous les jours, et qui pose la question la plus profonde qu'on puisse poser sur la nature : *à partir de quelle échelle le réel cesse-t-il d'être grossier ?*

Personne ne sait. Toi, tu construis pendant qu'on cherche.

---

*Cette série explore une douzaine de jeux et leurs concepts physiques associés. Pour ne rien rater, abonne-toi sur [arithmon.substack.com](https://arithmon.substack.com/).*

🌀
